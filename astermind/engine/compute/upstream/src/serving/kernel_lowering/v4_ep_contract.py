"""V4 DP1/legal-PP expert caller validation, independent of the generic EP graph."""
from collections import defaultdict
from math import prod
from src.kernel.predictors.moe_ep_math import compact_counts_to_global,marlin_ep_work

def applicable(pool):
    from .v4_modelsp_contract import applicable as sp_applicable
    if sp_applicable(pool):return True
    from .v4_dp_contract import applicable as dp_applicable
    if dp_applicable(pool):return True
    c=pool.parallel_plan.config
    return (pool.hardware_id=='nvidia_h200' and c.enable_expert_parallel
            and c.tensor_parallel_size==c.expert_parallel_size
            and c.tensor_parallel_size in (2,4,8,16)
            and c.data_parallel_size==c.context_parallel_size==c.sequence_parallel_size==1
            and 1<=c.pipeline_parallel_size<=pool.parallel_plan.num_layers)

def expert_attributes(node):
    a=node.attributes
    if a.get("source_valid_tokens",node.shape["tokens"]) != node.shape["tokens"]:
        from .v4_modelsp_math_binding import expert_attributes as masked_attributes
        return masked_attributes(node)
    if (a['source_component']!='vllm_0_30_symbolic_v4_marlin'
            or a['model_execution_version']!='vllm_0_30_0' or node.dtype!='bf16'
            or a['model_source_profile']!='vllm_0_30_v4_sync_fp8_ds_mla'):
        raise ValueError('V4 EP source identity mismatch')
    if (a['source_expert_tensor_parallel_size']!=1 or a['source_marlin_padding_columns']!=0
            or a['source_logical_intermediate']!=node.shape['intermediate'] or ('source_valid_tokens' in a and (type(a['source_valid_tokens']) is not int or a['source_valid_tokens']!=node.shape['tokens']))
            or a['output_is_reduced'] is not False or a['expert_placement_strategy']!='linear'):
        raise NotImplementedError('V4 EP requires whole experts without padding or token masks')
    attrs={name:a[name] for name in ('tokens_per_expert','expert_parallel_size','expert_rank','owned_experts',
                                    'routed_scaling_factor','routed_weight_dtype','weight_group_size','swiglu_limit')}
    attrs.update(local_tokens_per_expert=compact_counts_to_global(node.shape,a),expert_weight_tile_rows=None,
                 mathematical_program='v4_masked_marlin_ep_v1')
    marlin_ep_work(node.shape,attrs,model='v4')
    return attrs

def validate(graph,pool,nodes,memory_context):
    from .v4_modelsp_contract import applicable as sp_applicable, validate as sp_validate
    if sp_applicable(pool):return sp_validate(graph,pool,nodes,memory_context)
    if pool.parallel_plan.config.data_parallel_size > 1:
        from .v4_dp_contract import validate as dp_validate
        return dp_validate(graph,pool,nodes,memory_context)
    for n in nodes:
        valid=n.attributes.get("source_valid_tokens",n.shape["tokens"])
        if type(valid) is not int or valid!=n.shape["tokens"]:
            raise ValueError("V4 non-SP caller requires all physical rows valid")
    if not applicable(pool):raise NotImplementedError('V4 EP initial source contract requires H200 DP1 legal PP CP1 SP1')
    if memory_context is None:raise ValueError('V4 EP requires current source memory ports for alias validation')
    memory=memory_context._by_node
    byid={n.node_id:n for n in graph.nodes};out=defaultdict(list);succ=defaultdict(list)
    for n in graph.nodes:
        for v in n.attributes.get('symbolic_tensor_outputs',()):out[v].append(n)
    for edge in graph.edges:
        if edge.kind=='data':succ[edge.source].append(byid[edge.destination])
    def inputs(n):return tuple(n.attributes.get('symbolic_tensor_inputs',()))
    def outputs(n):return tuple(n.attributes.get('symbolic_tensor_outputs',()))
    def fail_if(condition,message):
        if condition:raise ValueError('V4 EP caller: '+message)
    def port(n,name,shape,dtype,output=False):
        views=memory[n.node_id].outputs if output else memory[n.node_id].inputs
        matches=[v for v in views if v.view_id==name]
        fail_if(len(matches)!=1 or tuple(matches[0].shape)!=tuple(shape) or matches[0].dtype!=dtype,'typed tensor port mismatch')
    groups=defaultdict(list)
    for n in nodes:
        expert_attributes(n)
        groups[n.attributes['moe_invocation_id']].append(n)
    for members in groups.values():
        first=members[0];t=first.shape['tokens'];h=first.shape['hidden'];e=first.shape['experts']
        layer=first.attributes['routing_provenance']['layer_index']
        fail_if(type(layer) is not int or not 0<=layer<pool.parallel_plan.num_layers,'invalid layer identity')
        stage=pool.parallel_plan.layer_to_pipeline_stage[layer]
        tp=pool.parallel_plan.config.tensor_parallel_size
        expected=tuple(pool.physical_rank_for_logical_rank(stage*tp+lane) for lane in range(tp))
        participants=pool.physical_rank_group('tensor',pool.logical_rank_for_physical_rank(first.rank))
        fail_if(tuple(participants)!=expected,'expert group differs from actual pipeline layer placement')
        fail_if(len(members)!=len(participants) or {n.rank for n in members}!=set(participants),'incomplete expert group')
        fail_if(sorted(j for n in members for j in n.attributes['owned_experts'])!=list(range(e)),'expert coverage')
        adds={};reductions={}
        for n in members:
            a=n.attributes;ports=inputs(n)
            fail_if(n.shape!=first.shape or a['tokens_per_expert']!=first.attributes['tokens_per_expert']
                    or a['routing_provenance']!=first.attributes['routing_provenance'],'rank routing differs')
            fail_if(a['expert_rank']!=participants.index(n.rank) or len(ports)!=3,'rank or input ports')
            routes=[x for x in out[ports[1]] if x.variant=='dsv4_moe_route' and outputs(x)==ports[1:]]
            fail_if(len(routes)!=1,'missing exact weights/IDs producer')
            route=routes[0]
            fail_if(route.rank!=n.rank or route.pool_name!=n.pool_name or n not in succ[route.node_id]
                    or route.attributes['routed_scaling_factor']!=a['routed_scaling_factor'],'router rank, edge or scaling')
            fail_if(route.kind.value!='vector' or dict(route.shape)!={'elements':t*e,'rows':t,'experts':e,'topk':6},'routing shape')
            hash_route=a['routing_provenance']['layer_index']<3
            fail_if(route.attributes['expert_id_selection']!=('hash_table' if hash_route else 'score_topk')
                    or len(inputs(route))!=(2 if hash_route else 1),'hash/score routing branch')
            port(n,ports[0],(t,h),'bf16');port(n,ports[1],(t,6),'fp32');port(n,ports[2],(t,6),'int32')
            port(route,ports[1],(t,6),'fp32',True);port(route,ports[2],(t,6),'int32',True)
            gates=[x for x in out[inputs(route)[0]] if x.kind.value=='router']
            fail_if(len(gates)!=1,'missing logits gate')
            gate=gates[0]
            fail_if(gate.rank!=n.rank or gate.shape['m']!=t or gate.shape['n']!=e or gate.shape['k']!=h
                    or ports[0] not in inputs(gate) or route not in succ[gate.node_id],'gate does not consume same hidden')
            port(gate,outputs(gate)[0],(t,e),'fp32',True)
            candidates=[x for x in succ[n.node_id] if x.kind.value=='elementwise' and outputs(n)[0] in inputs(x)]
            fail_if(len(candidates)!=1,'missing unique shared/routed addition')
            add=candidates[0];other=[v for v in inputs(add) if v!=outputs(n)[0]]
            fail_if(len(inputs(add))!=2 or len(other)!=1 or add.rank!=n.rank,'shared addition ports')
            fail_if(add.variant!='add' or add.attributes.get('operation')!='add' or add.dtype!='bf16'
                    or dict(add.shape)!={'elements':t*h} or add.attributes.get('reads')!=2
                    or add.attributes.get('writes')!=1,'shared merge must be BF16 addition')
            shared=[x for x in out[other[0]] if x.rank==n.rank and x.kind.value=='linear' and x.variant=='gemm']
            width=-(-(n.shape['intermediate']//128)//len(participants))*128
            fail_if(len(shared)!=1 or dict(shared[0].shape)!={'b':1,'m':t,'n':h,'k':width},'shared expert down geometry')
            down=shared[0];act=[x for v in inputs(down) for x in out[v] if x.rank==n.rank and x.kind.value=='activation']
            fail_if(len(act)!=1 or act[0].variant!='silu' or act[0].attributes.get('swiglu_limit')!=10.0
                    or dict(act[0].shape)!={'elements':t*width,'tok':t,'intermediate':width},'shared activation')
            gateup=[x for v in inputs(act[0]) for x in out[v] if x.rank==n.rank and x.kind.value=='linear']
            fail_if(len(gateup)!=1 or dict(gateup[0].shape)!={'b':1,'m':t,'n':2*width,'k':h}
                    or inputs(gateup[0])!=(ports[0],),'shared branch must use same hidden and TP shard')
            for linear in (down,gateup[0]):
                la=linear.attributes
                fail_if(linear.dtype!='bf16' or la.get('weight_dtype')!='fp8' or la.get('weight_group_size')!=128
                        or la.get('weight_quantization')!='fp8_e4m3_block128x128_act1x128_dynamic'
                        or la.get('symbolic_scale_layout')!='row_major_group128_fp32'
                        or la.get('model_source_profile')!='vllm_0_30_v4_sync_fp8_ds_mla'
                        or la.get('model_execution_version')!='vllm_0_30_0' or la.get('source_projection_only') is not True,'shared FP8 source contract')
            rs=[x for x in succ[add.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce']
            fail_if(len(rs)!=1,'missing source TP reduction')
            red=rs[0];ra=red.attributes
            fail_if(tuple(ra['participants'])!=tuple(participants) or ra['reduction']!='sum'
                    or ra['bytes']!=2*t*h or ra['bytes_semantics']!='per_rank_input','reduction geometry')
            adds[n.rank]=add;reductions[red.node_id]=red
        fail_if(len(reductions)!=1,'ranks do not share one reduction')
        red=next(iter(reductions.values()))
        fail_if(inputs(red)!=tuple(outputs(adds[r])[0] for r in participants)
                or tuple(red.attributes['symbolic_collective_input_producers'])!=tuple(adds[r].node_id for r in participants)
                or len(outputs(red))!=len(participants),'reduction source ports')
        for r,v in zip(participants,outputs(red)):
            consumers=[x for x in succ[red.node_id] if x.rank==r
                       and x.attributes.get('source_component')=='vllm_0_30_symbolic_mhc']
            fail_if(not consumers,'reduced output not consumed by source mHC')
            source_views={x.view_id:x for x in memory[red.node_id].outputs}
            source=source_views[v]
            for consumer in consumers:
                target=memory[consumer.node_id].inputs[0]
                fail_if(source.allocation_id!=target.allocation_id or source.offset_bytes!=target.offset_bytes
                        or source.dtype!=target.dtype or prod(source.shape)!=prod(target.shape),'mHC input alias is not reduction output')
                for view in (source,target):
                    stride=2
                    for size,step in reversed(tuple(zip(view.shape,view.strides_bytes))):
                        fail_if(size>1 and step!=stride,'mHC reduction alias is not contiguous')
                        stride*=size
                owner=next(n for n in members if n.rank==r)
                hidden=inputs(owner)[0]
                pre=[x for x in out[hidden] if x.attributes.get('source_component')=='vllm_0_30_symbolic_mhc']
                fail_if(len(pre)!=1,'missing actual mHC pre producer')
                pre_outputs=outputs(pre[0])
                residual_ports=tuple(v for v in pre_outputs if v!=hidden)
                fail_if(inputs(consumer)[1:]!=residual_ports,'mHC residual/mixing inputs differ from source pre')
