"""V4 non-SP WideEP: DP dispatch/combine followed by the local TP tail."""
from collections import defaultdict
from contextvars import ContextVar

_validated_variables=ContextVar('v4_wideep_validated_variables',default=None)

def handles_collective(node,pool,graph):
    proof=_validated_variables.get()
    return (applicable(pool) and proof is not None and proof[0] is graph
            and node.node_id in proof[1] and node.attributes.get('collective_group')=='data')

def handles_hash_route(node):
    proof = _validated_variables.get()
    return proof is not None and node.node_id in proof[2] and proof[0].node(node.node_id) is node

def applicable(pool):
    c=pool.parallel_plan.config
    return (pool.hardware_id=='nvidia_h200' and pool.parallel_plan.num_layers in (43,61)
            and c.enable_expert_parallel and c.tensor_parallel_size in (1,2,4,8,16)
            and c.data_parallel_size>1
            # TP1 has no model sequence sharding even with one pipeline stage.
            and (1<c.pipeline_parallel_size<=pool.parallel_plan.num_layers
                 or c.tensor_parallel_size==c.pipeline_parallel_size==1)
            and c.expert_parallel_size==c.tensor_parallel_size*c.data_parallel_size
            and c.context_parallel_size==c.sequence_parallel_size==1)

def validate(graph,pool,nodes,memory_context):
    _validated_variables.set(None)
    from .v4_ep_contract import expert_attributes
    if not applicable(pool):raise NotImplementedError('V4 WideEP requires H200 DP, a non-SP TP1 or multi-stage pipeline, and uniform TP*DP experts')
    if memory_context is None:raise ValueError('V4 WideEP requires actual typed tensor ports')
    c=pool.parallel_plan.config;tp,dp,pp=c.tensor_parallel_size,c.data_parallel_size,c.pipeline_parallel_size
    memory=memory_context._by_node;out=defaultdict(list);succ=defaultdict(list);edges=set();byid={n.node_id:n for n in graph.nodes}
    for n in graph.nodes:
        for v in n.attributes.get('symbolic_tensor_outputs',()):out[v].append(n)
    for e in graph.edges:
        edges.add((e.source,e.destination))
        if e.kind=='data':succ[e.source].append(byid[e.destination])
    def ins(n):return tuple(n.attributes.get('symbolic_tensor_inputs',()))
    def data_ins(n):return tuple(v for v in ins(n) if v not in n.attributes.get('symbolic_parameter_inputs',()))
    def outs(n):return tuple(n.attributes.get('symbolic_tensor_outputs',()))
    def check(ok,msg):
        if not ok:raise ValueError('V4 WideEP caller: '+msg)
    def one(xs,msg):check(len(xs)==1,msg);return xs[0]
    def typed(n,v,shape,dtype,output=False):
        vs=memory[n.node_id].outputs if output else memory[n.node_id].inputs
        x=one([x for x in vs if x.view_id==v],'missing typed view')
        check(tuple(x.shape)==shape and x.dtype==dtype,'typed shape/dtype')
    def physical(replica,stage,lane):return pool.physical_rank_for_logical_rank((replica*pp+stage)*tp+lane)
    groups=defaultdict(list);variable_ids=set();hash_ids=set();masked_expert_ids=set()
    for n in nodes:
        attrs = expert_attributes(n)
        if attrs['mathematical_program'] == 'v4_modelsp_masked_marlin_ep_v1':
            masked_expert_ids.add(n.node_id)
        groups[n.attributes['moe_invocation_id']].append(n)
    for members in groups.values():
        first=members[0];total=first.shape['tokens'];h=first.shape['hidden'];e=first.shape['experts'];layer=first.attributes['routing_provenance']['layer_index']
        check(type(layer) is int and 0<=layer<pool.parallel_plan.num_layers,'layer identity')
        stage=pool.parallel_plan.layer_to_pipeline_stage[layer]
        epgroup=tuple(physical(d,stage,l) for d in range(dp) for l in range(tp))
        check(len(members)==tp*dp and {n.rank for n in members}==set(epgroup),'complete stage-local EP group')
        check(sorted(j for n in members for j in n.attributes['owned_experts'])==list(range(e)),'expert coverage')
        dummy_replicas = set()
        for replica in range(dp):
            group = tuple(physical(replica, stage, lane) for lane in range(tp))
            replica_nodes = [n for n in members if n.rank in group]
            dummy = [n for n in replica_nodes if n.attributes.get('source_dummy_run') is True]
            check(not dummy or len(dummy) == tp and all(n.attributes.get('source_actual_request_count') == 0 for n in dummy),
                  'all TP ranks of an idle replica must have no actual requests')
            if dummy:
                dummy_replicas.add(replica)
        check(len(dummy_replicas) < dp, 'at least one real replica')
        expert_outputs={n.rank:outs(n)[0] for n in members};adds={};reductions={};combines={};row_vectors=set();hidden_ports={}
        for n in members:
            a=n.attributes;r=n.rank;logical=pool.logical_rank_for_physical_rank(r);d=logical//(tp*pp);lane=logical%tp
            dg=tuple(physical(x,stage,lane) for x in range(dp));tg=tuple(physical(d,stage,l) for l in range(tp))
            check(a['expert_rank']==d*tp+lane and n.shape==first.shape and a['tokens_per_expert']==first.attributes['tokens_per_expert']
                  and a['routing_provenance']==first.attributes['routing_provenance'],'global routing/rank identity')
            p=ins(n);check(len(p)==3,'three expert inputs')
            local_ports=[];dispatch_rows=None
            for v,width,dtype,size in zip(p,(h,6,6),('bf16','fp32','int32'),(2,4,4)):
                typed(n,v,(total,width),dtype)
                ag=one([x for x in out[v] if x.kind.value=='collective' and x.attributes.get('operation')=='all_gatherv'],'actual dispatch producer')
                variable_ids.add(ag.node_id)
                aa=ag.attributes;rows=tuple(aa['symbolic_variable_rows'])
                check(len(rows)==dp and all(type(x) is int and x>0 for x in rows) and sum(rows)==total,'positive original DP rows')
                check(tuple(aa['participants'])==dg and aa['collective_group']=='data' and tuple(aa['symbolic_variable_row_shape'])==(width,)
                      and aa['bytes_semantics']=='aggregate_input' and aa['bytes']==total*width*size
                      and tuple(aa['partition_bytes'])==tuple(x*width*size for x in rows) and aa['reduction'] is None,'dispatch group/bytes')
                check(len(ins(ag))==len(outs(ag))==dp and outs(ag)[d]==v and n in succ[ag.node_id],'dispatch ordering/dependency')
                check(dispatch_rows is None or dispatch_rows==rows,'all dispatches share row partitions')
                dispatch_rows=rows;local_ports.append(ins(ag)[d])
            row_vectors.add(dispatch_rows);t=dispatch_rows[d];hidden_port,weight_port,id_port=local_ports
            hidden_ports[r]=hidden_port
            route=one([x for x in out[weight_port] if x.variant=='dsv4_moe_route' and x.rank==r and outs(x)==(weight_port,id_port)],'original replica router')
            check(route.kind.value=='vector' and dict(route.shape)=={'elements':t*e,'rows':t,'experts':e,'topk':6}
                  and route.attributes['routed_scaling_factor']==a['routed_scaling_factor'],'router semantics')
            hash_route=layer<3
            masked = hash_route and d in dummy_replicas
            check(route.attributes['expert_id_selection']==('hash_table' if hash_route else 'score_topk')
                  and len(ins(route)) == (3 if masked else 2 if hash_route else 1),'hash/score route')
            check(all(dispatch_rows[i] == 1 for i in dummy_replicas), 'idle replicas have one physical row')
            expected_valid = total - len(dummy_replicas) if hash_route else total
            check(a.get('source_valid_tokens', total) == expected_valid, 'expert routing uses the actual hash extent')
            if masked:
                check(route.attributes.get('symbolic_padding_valid_prefix') == 0
                      and route.attributes.get('source_padding_expert_ids') == -1
                      and route.attributes.get('source_padding_weights') == 0, 'idle hash rows are masked')
                typed(route, ins(route)[2], (t,), 'bool')
                hash_ids.add(route.node_id)
            typed(route,weight_port,(t,6),'fp32',True);typed(route,id_port,(t,6),'int32',True)
            gate=one([x for x in out[ins(route)[0]] if x.kind.value=='router' and x.rank==r],'router gate')
            check(dict(gate.shape)=={'b':1,'m':t,'n':e,'k':h} and data_ins(gate)==(hidden_port,),'same original hidden')
            typed(gate,outs(gate)[0],(t,e),'fp32',True)
            combine=one([x for x in succ[n.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='reduce_scatterv'],'DP combine')
            ca=combine.attributes
            variable_ids.add(combine.node_id)
            check(tuple(ca['participants'])==dg and ca['collective_group']=='data' and ca['reduction']=='sum'
                  and ca['bytes_semantics']=='per_rank_input' and ca['bytes']==2*total*h
                  and tuple(ca['partition_bytes'])==tuple(x*2*h for x in dispatch_rows)
                  and tuple(ca['symbolic_variable_rows'])==dispatch_rows and tuple(ca['symbolic_variable_row_shape'])==(h,)
                  and ins(combine)==tuple(expert_outputs[x] for x in dg),'combine groups/partition/input order')
            combines[combine.node_id]=combine;cv=outs(combine)[d]
            copy=one([x for x in succ[combine.node_id] if x.rank==r and x.kind.value=='gather' and ins(x)==(cv,)],'replica finalize copy')
            check(copy.dtype=='bf16' and dict(copy.shape)=={'elements':t*h,'rows':t,'hidden':h}
                  and copy.attributes['materialized_output'] is True and copy.attributes['gather_factor']==1,'materialized finalize')
            typed(copy,cv,(t,h),'bf16');typed(copy,outs(copy)[0],(t,h),'bf16',True)
            add=one([x for x in succ[copy.node_id] if x.rank==r and x.kind.value=='elementwise' and outs(copy)[0] in ins(x)],'local shared addition')
            check(add.variant=='add' and add.attributes['operation']=='add' and dict(add.shape)=={'elements':t*h}
                  and add.attributes['reads']==2 and add.attributes['writes']==1 and add.dtype=='bf16' and len(ins(add))==2,'shared addition')
            other=one([v for v in ins(add) if v!=outs(copy)[0]],'shared output')
            shared=one([x for x in out[other] if x.rank==r and x.kind.value=='linear'],'shared down')
            width=-(-(n.shape['intermediate']//128)//tp)*128
            check(dict(shared.shape)=={'b':1,'m':t,'n':h,'k':width},'block128 shared width')
            act=one([x for v in ins(shared) for x in out[v] if x.rank==r and x.kind.value=='activation'],'shared activation')
            check(act.variant=='silu' and act.attributes.get('swiglu_limit')==10.0 and dict(act.shape)=={'elements':t*width,'tok':t,'intermediate':width},'shared SwiGLU')
            gu=one([x for v in ins(act) for x in out[v] if x.rank==r and x.kind.value=='linear'],'shared gate/up')
            check(dict(gu.shape)=={'b':1,'m':t,'n':2*width,'k':h} and data_ins(gu)==(hidden_port,),'shared source')
            for linear in (shared,gu):
                la=linear.attributes
                check(linear.dtype=='bf16' and la.get('weight_dtype')=='fp8' and la.get('weight_group_size')==128
                      and la.get('weight_quantization')=='fp8_e4m3_block128x128_act1x128_dynamic'
                      and la.get('symbolic_scale_layout')=='row_major_group128_fp32'
                      and la.get('model_source_profile')=='vllm_0_30_v4_sync_fp8_ds_mla'
                      and la.get('model_execution_version')=='vllm_0_30_0' and la.get('source_projection_only') is True,'shared FP8 source contract')
            if tp==1:
                check(len(outs(add))==1,'TP1 local add has exactly one output')
                check(not [x for x in succ[add.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'TP1 source has no TP reduction')
                red=add
            else:
                red=one([x for x in succ[add.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'replica TP reduction')
                check(tuple(red.attributes['participants'])==tg and red.attributes['reduction']=='sum' and red.attributes['bytes']==2*t*h
                      and red.attributes['bytes_semantics']=='per_rank_input','replica TP geometry')
            adds[r]=add;reductions[red.node_id]=red
        check(len(row_vectors)==1 and len(reductions)==dp and len(combines)==tp,'complete DP/TP groups')
        from math import prod
        for red in reductions.values():
            participants=(red.rank,) if tp==1 else tuple(red.attributes['participants'])
            if tp>1:check(ins(red)==tuple(outs(adds[r])[0] for r in participants) and len(outs(red))==tp,'TP input order')
            for r,v in zip(participants,outs(red)):
                consumers=[x for x in succ[red.node_id] if x.rank==r and x.attributes.get('source_component')=='vllm_0_30_symbolic_mhc']
                check(bool(consumers),'mHC consumers')
                source=one([x for x in memory[red.node_id].outputs if x.view_id==v],'TP output view')
                for consumer in consumers:
                    target=memory[consumer.node_id].inputs[0]
                    check(source.allocation_id==target.allocation_id and source.offset_bytes==target.offset_bytes and source.dtype==target.dtype
                          and prod(source.shape)==prod(target.shape),'mHC reduction alias')
                    for view in (source,target):
                        stride=2
                        for size,step in reversed(tuple(zip(view.shape,view.strides_bytes))):
                            check(size<=1 or step==stride,'contiguous mHC alias');stride*=size
                    hidden=hidden_ports[r]
                    pre=one([x for x in out[hidden] if x.attributes.get('source_component')=='vllm_0_30_symbolic_mhc'],'original mHC pre')
                    check(ins(consumer)[1:]==tuple(v for v in outs(pre) if v!=hidden),'original mHC residual/mix')
    _validated_variables.set((graph, frozenset(variable_ids), frozenset(hash_ids), frozenset(masked_expert_ids)))


def masked_source_nodes(graph):
    proof = _validated_variables.get()
    if proof is None or proof[0] is not graph:
        return frozenset(), frozenset()
    return proof[2], proof[3]
