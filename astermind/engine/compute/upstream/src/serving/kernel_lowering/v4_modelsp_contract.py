"""V4 all-active WideEP model-SP: full EP experts and typed TP sequence restoration."""
from collections import defaultdict
from contextvars import ContextVar

_validated_variables=ContextVar('v4_modelsp_validated_variables',default=None)

def handles_collective(node,pool,graph):
    proof=_validated_variables.get()
    return (applicable(pool) and proof is not None and proof[0] is graph and proof[1] is pool
            and node.node_id in proof[2] and node.attributes.get('collective_group') in ('expert','tensor'))

def handles_fill(node):
    proof=_validated_variables.get()
    return proof is not None and node.node_id in proof[3] and proof[0].node(node.node_id) is node

def handles_hash_route(node):
    proof=_validated_variables.get()
    return proof is not None and node.node_id in proof[4] and proof[0].node(node.node_id) is node

def applicable(pool):
    c=pool.parallel_plan.config
    return (pool.hardware_id=='nvidia_h200' and pool.parallel_plan.num_layers in (43,61)
            and c.enable_expert_parallel and c.tensor_parallel_size in (2,4,8,16)
            and (c.data_parallel_size in (2,4,8) or (pool.parallel_plan.num_layers==61 and pool.parallel_plan.num_experts==384 and c.tensor_parallel_size==8 and c.data_parallel_size==48)) and c.pipeline_parallel_size==1
            and c.expert_parallel_size==c.tensor_parallel_size*c.data_parallel_size
            and c.context_parallel_size==c.sequence_parallel_size==1)

def validate(graph,pool,nodes,memory_context):
    _validated_variables.set(None)
    from .v4_modelsp_math_binding import expert_attributes
    if not applicable(pool):raise NotImplementedError('V4 model-SP requires H200 all-active DP, TP>1 PP1, uniform TP*DP experts')
    if memory_context is None:raise ValueError('V4 WideEP requires actual typed tensor ports')
    c=pool.parallel_plan.config;tp,dp,pp=c.tensor_parallel_size,c.data_parallel_size,c.pipeline_parallel_size
    memory=memory_context._by_node;out=defaultdict(list);succ=defaultdict(list);edges=set();byid={n.node_id:n for n in graph.nodes}
    for n in graph.nodes:
        if n.attributes.get('source_dummy_run') is True:raise NotImplementedError('V4 WideEP idle/dummy rows require a separate retained-state contract')
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
    groups=defaultdict(list);variable_ids=set();hash_ids=set()
    for n in nodes:expert_attributes(n);groups[n.attributes['moe_invocation_id']].append(n)
    for members in groups.values():
        first=members[0];total=first.shape['tokens'];h=first.shape['hidden'];e=first.shape['experts'];layer=first.attributes['routing_provenance']['layer_index']
        check(type(layer) is int and 0<=layer<pool.parallel_plan.num_layers,'layer identity')
        stage=pool.parallel_plan.layer_to_pipeline_stage[layer]
        epgroup=tuple(physical(d,stage,l) for d in range(dp) for l in range(tp))
        check(len(members)==tp*dp and {n.rank for n in members}==set(epgroup),'complete stage-local EP group')
        check(sorted(j for n in members for j in n.attributes['owned_experts'])==list(range(e)),'expert coverage')
        expert_outputs={n.rank:outs(n)[0] for n in members};adds={};reductions={};combines={};row_vectors=set();hidden_ports={}
        for n in members:
            a=n.attributes;r=n.rank;logical=pool.logical_rank_for_physical_rank(r);d=logical//(tp*pp);lane=logical%tp
            dg=epgroup;tg=tuple(physical(d,stage,l) for l in range(tp));owner=d*tp+lane
            check(a['expert_rank']==d*tp+lane and n.shape==first.shape and a['tokens_per_expert']==first.attributes['tokens_per_expert']
                  and a['routing_provenance']==first.attributes['routing_provenance'],'global routing/rank identity')
            p=ins(n);check(len(p)==3,'three expert inputs')
            local_ports=[];dispatch_rows=None
            for v,width,dtype,size in zip(p,(h,6,6),('bf16','fp32','int32'),(2,4,4)):
                typed(n,v,(total,width),dtype)
                ag=one([x for x in out[v] if x.kind.value=='collective' and x.attributes.get('operation')=='all_gatherv'],'actual dispatch producer')
                variable_ids.add(ag.node_id)
                aa=ag.attributes;rows=tuple(aa['symbolic_variable_rows'])
                check(len(rows)==dp*tp and all(type(x) is int and x>0 for x in rows) and sum(rows)==total,'positive original DP rows')
                check(tuple(aa['participants'])==dg and aa['collective_group']=='expert' and tuple(aa['symbolic_variable_row_shape'])==(width,)
                      and aa['bytes_semantics']=='aggregate_input' and aa['bytes']==total*width*size
                      and tuple(aa['partition_bytes'])==tuple(x*width*size for x in rows) and aa['reduction'] is None,'dispatch group/bytes')
                check(len(ins(ag))==len(outs(ag))==dp*tp and outs(ag)[owner]==v and n in succ[ag.node_id],'dispatch ordering/dependency')
                check(dispatch_rows is None or dispatch_rows==rows,'all dispatches share row partitions')
                dispatch_rows=rows;local_ports.append(ins(ag)[owner])
            row_vectors.add(dispatch_rows);t=dispatch_rows[owner];hidden_port,weight_port,id_port=local_ports
            hidden_ports[r]=hidden_port
            route=one([x for x in out[weight_port] if x.variant=='dsv4_moe_route' and x.rank==r and outs(x)==(weight_port,id_port)],'original replica router')
            check(route.kind.value=='vector' and route.dtype=='fp32' and dict(route.shape)=={'elements':t*e,'rows':t,'experts':e,'topk':6}
                  and route.attributes['routed_scaling_factor']==a['routed_scaling_factor'],'router semantics')
            hash_route=layer<3
            check(route.attributes['expert_id_selection']==('hash_table' if hash_route else 'score_topk') and len(ins(route))==(3 if hash_route else 1),'hash/score route')
            if hash_route:
                mask=ins(route)[2]
                typed(route,mask,(t,),'bool')
                mv=one([v for v in memory[route.node_id].inputs if v.view_id==mask],'source mask view')
                check(mask==route.node_id+':padding-mask' and mv.allocation_id==mask and mv.offset_bytes==0 and (t<=1 or mv.strides_bytes==(1,)),'source hash mask allocation')
                hash_ids.add(route.node_id)
            typed(route,weight_port,(t,6),'fp32',True);typed(route,id_port,(t,6),'int32',True)
            gate=one([x for x in out[ins(route)[0]] if x.kind.value=='router' and x.rank==r],'router gate')
            check(dict(gate.shape)=={'b':1,'m':t,'n':e,'k':h} and data_ins(gate)==(hidden_port,),'same original hidden')
            typed(gate,outs(gate)[0],(t,e),'fp32',True)
            combine=one([x for x in succ[n.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='reduce_scatterv'],'DP combine')
            ca=combine.attributes
            variable_ids.add(combine.node_id)
            check(tuple(ca['participants'])==dg and ca['collective_group']=='expert' and ca['reduction']=='sum'
                  and ca['bytes_semantics']=='per_rank_input' and ca['bytes']==2*total*h
                  and tuple(ca['partition_bytes'])==tuple(x*2*h for x in dispatch_rows)
                  and tuple(ca['symbolic_variable_rows'])==dispatch_rows and tuple(ca['symbolic_variable_row_shape'])==(h,)
                  and ins(combine)==tuple(expert_outputs[x] for x in dg),'combine groups/partition/input order')
            combines[combine.node_id]=combine;cv=outs(combine)[owner]
            copy=one([x for x in succ[combine.node_id] if x.rank==r and x.kind.value=='gather' and ins(x)==(cv,)],'replica finalize copy')
            check(copy.dtype=='bf16' and dict(copy.shape)=={'elements':t*h,'rows':t,'hidden':h}
                  and copy.attributes['materialized_output'] is True and copy.attributes['gather_factor']==1,'materialized finalize')
            typed(copy,cv,(t,h),'bf16');typed(copy,outs(copy)[0],(t,h),'bf16',True)
            add=one([x for x in succ[copy.node_id] if x.rank==r and x.kind.value=='elementwise' and outs(copy)[0] in ins(x)],'local shared addition')
            check(add.variant=='add' and add.attributes['operation']=='add' and dict(add.shape)=={'elements':t*h}
                  and add.attributes['reads']==2 and add.attributes['writes']==1 and add.dtype=='bf16' and len(ins(add))==2,'shared addition')
            other=one([v for v in ins(add) if v!=outs(copy)[0]],'shared output')
            shared=one([x for x in out[other] if x.rank==r and x.kind.value=='linear'],'shared down')
            width=n.shape['intermediate']
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
            check(len(outs(add))==1,'SP add has exactly one output')
            check(not [x for x in succ[add.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'model-SP has no TP reduction after shared-add')
            red=add
            adds[r]=add;reductions[red.node_id]=red
        check(len(row_vectors)==1 and len(reductions)==dp*tp and len(combines)==1,'complete DP/TP groups')
        from math import prod
        for red in reductions.values():
            participants=(red.rank,)
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
    tensor_ids,fill_ids=validate_sequence(graph,pool,memory_context,nodes)
    variable_ids.update(tensor_ids)
    _validated_variables.set((graph,pool,frozenset(variable_ids),frozenset(fill_ids),frozenset(hash_ids),frozenset(n.node_id for n in nodes)))

def validate_lowered(graph,plan):
    """Check the actual returned plan, including ordinary no-launch-input calls."""
    proof=_validated_variables.get()
    if proof is None or proof[0] is not graph:
        if any(str(c.kernel_spec.attributes.get('mathematical_program','')).startswith('v4_modelsp_') for c in plan.calls):
            raise ValueError('V4 model-SP plan lacks current graph proof')
        return
    from .v4_ep_contract import expert_attributes
    from .v4_modelsp_math_binding import valid_tokens
    from src.kernel.kernel_spec import KernelSpec, KernelKind
    wanted=proof[3]|proof[4]|proof[5];seen=set()
    for call in plan.calls:
        ids=set(call.source_node_ids)&wanted
        if not ids:continue
        if len(ids)!=1 or len(call.source_node_ids)!=1 or seen&ids:
            raise ValueError('V4 model-SP requires exactly one call per proved source node')
        identity=next(iter(ids));seen.add(identity);node=graph.node(identity);k=call.kernel_spec
        if identity in proof[4]:
            kind,variant=KernelKind.VECTOR,'dsv4_moe_route'
            expected={'expert_id_selection':'hash_table','routed_scaling_factor':node.attributes['routed_scaling_factor'],'routing_source':None,'mathematical_program':'v4_modelsp_hash_top6_masked_v1','valid_rows':node.attributes['symbolic_padding_valid_prefix']}
            shape={'element_count':node.shape['elements'],'rows':node.shape['rows'],'experts':node.shape['experts'],'topk':6,'dtype_bytes':4}
        elif identity in proof[3]:
            kind,variant=KernelKind.VECTOR,'elementwise'
            expected={'operation':'fill','reads':0,'writes':1,'mathematical_program':'v4_modelsp_zero_fill_v1'}
            shape={'element_count':node.shape['elements'],'dtype_bytes':2 if node.dtype=='bf16' else 4}
        else:
            kind,variant=KernelKind.FUSED_MOE,'routed_experts'
            expected=expert_attributes(node);shape={**node.shape,'dtype_bytes':2}
            if expected['mathematical_program']=='v4_modelsp_masked_marlin_ep_v1':shape['tokens']=valid_tokens(node)
        canonical=KernelSpec(kind=kind,hardware_id='nvidia_h200',shape=shape,dtype=node.dtype,variant=variant,attributes=expected,launch=None,execution_support='unsupported',unsupported_reason='launch_selector_unavailable')
        actual=(k.kind,k.variant,dict(k.attributes),dict(k.shape),k.dtype,k.hardware_id,k.launch,k.execution_support)
        required=(canonical.kind,canonical.variant,dict(canonical.attributes),dict(canonical.shape),canonical.dtype,canonical.hardware_id,canonical.launch,canonical.execution_support)
        if actual!=required:
            raise ValueError('V4 model-SP returned plan differs from validated source: '+identity+' actual='+repr(actual)+' expected='+repr(required))
    if seen!=wanted:raise ValueError('V4 model-SP plan omitted proved source calls')

def validate_sequence(graph,pool,memory_context,experts):
    """Prove the SP lane split, each attention gather/scatter, and final restore."""
    from math import prod
    c=pool.parallel_plan.config;tp=c.tensor_parallel_size;dp=c.data_parallel_size
    ns={n.node_id:n for n in graph.nodes};mem=memory_context._by_node;pred=defaultdict(list);succ=defaultdict(list)
    for edge in graph.edges:
        if edge.kind=='data':pred[edge.destination].append(ns[edge.source]);succ[edge.source].append(ns[edge.destination])
    def require(ok,msg):
        if not ok:raise ValueError('V4 model-SP sequence: '+msg)
    def one(xs,msg):require(len(xs)==1,msg);return xs[0]
    def iv(n):return tuple(n.attributes.get('symbolic_tensor_inputs',()))
    def ov(n):return tuple(n.attributes.get('symbolic_tensor_outputs',()))
    def port(n,name,shape,dtype,output=False):
        v=one([v for v in (mem[n.node_id].outputs if output else mem[n.node_id].inputs) if v.view_id==name],'typed view')
        require(v.shape==tuple(shape) and v.dtype==dtype,'typed shape/dtype');stride={'bf16':2,'int32':4,'int64':8,'fp32':4,'bool':1}[dtype]
        for dim,step in reversed(tuple(zip(v.shape,v.strides_bytes))):require(dim<=1 or step==stride,'contiguous typed view');stride*=dim
        return v
    def alias(a,b):require(a.allocation_id==b.allocation_id and a.offset_bytes==b.offset_bytes,'actual storage alias')
    logical=[];groups=[];h=experts[0].shape['hidden'];layers=pool.parallel_plan.num_layers;proved=set();fills=set()
    for d in range(dp):
        group=tuple(pool.physical_rank_for_logical_rank(d*tp+lane) for lane in range(tp));groups.append(group)
        embeds=[one([n for n in graph.nodes if n.rank==r and n.kind.value=='gather' and n.variant=='embedding'],'one original embedding') for r in group]
        count=embeds[0].shape['rows'];require(type(count) is int and count>0 and all(n.shape['rows']==count for n in embeds),'active logical replica rows');logical.append(count)
    expected_rows=tuple(-(-count//tp) for count in logical for _ in range(tp));total=sum(expected_rows)
    for n in graph.nodes:
        if n.kind.value=='collective' and n.attributes.get('collective_group')=='expert':
            require(tuple(n.attributes['symbolic_variable_rows'])==expected_rows,'EP physical partitions follow actual SP lanes')
    for n in experts:
        layer=n.attributes['routing_provenance']['layer_index'];v=n.attributes.get('source_valid_tokens',n.shape['tokens'])
        require(n.shape['tokens']==total and v==(sum(logical) if layer<3 else total),'valid hash versus physical score routing extent')
    # The full set is required, not merely whichever source nodes happened to appear.
    require({n.attributes['routing_provenance']['layer_index'] for n in experts}==set(range(layers)),'complete model layers')
    for d,group in enumerate(groups):
        count=logical[d];local=-(-count//tp);padded=local*tp
        for lane,r in enumerate(group):
            embed=one([n for n in graph.nodes if n.kind.value=='collective' and n.node_id.endswith('embedding:all-reduce') and tuple(n.attributes['participants'])==group],'embedding reduction')
            # This collective node has all TP output ports and may carry another rank.
            first=one([n for n in graph.nodes if n.rank==r and n.node_id.endswith('layer-0:pre')],'initial mHC')
            routes=[n for n in graph.nodes if n.rank==r and n.variant=='dsv4_moe_route' and ':layer-0:' in n.node_id]
            route=one(routes,'first hash route')
            for label,consumer,input_name,shape,dtype,size,source_name in (
                ('sp-embedding',first,iv(first)[0],(local,h),'bf16',2,ov(embed)[lane]),
                ('sp-input-ids',route,iv(route)[1],(local,),'int32',4,None)):
                target=port(consumer,input_name,shape,dtype);width=h if label=='sp-embedding' else 1
                if count==padded:
                    if source_name is not None:
                        source=port(embed,source_name,(count,h),dtype,True)
                        require(target.allocation_id==source.allocation_id and target.offset_bytes==source.offset_bytes+lane*local*width*size,'initial lane offset')
                    else:require(target.allocation_id.endswith(':input_ids') and target.offset_bytes==lane*local*size,'input IDs lane offset')
                else:
                    cp=one([n for n in graph.nodes if n.rank==r and n.node_id.endswith(label+':copy')],'initial padding copy')
                    zero=one([n for n in graph.nodes if n.rank==r and n.node_id.endswith(label+':pad-zero')],'initial padding zero')
                    require(cp.dtype==dtype and cp.kind.value=='gather' and cp.shape['rows']==count and cp.shape['hidden']==width and tuple(cp.attributes.get(k) for k in ('gather_factor','index_count','scattered','materialized_output'))==(1,0,False,True),'initial exact copy')
                    if source_name is not None:require(iv(cp)==(source_name,) and embed in pred[cp.node_id],'copy reduced embedding')
                    else:require(mem[cp.node_id].inputs[0].allocation_id.endswith(':input_ids'),'copy original IDs')
                    require(zero.dtype==dtype and zero.kind.value=='elementwise' and dict(zero.shape)=={'elements':(padded-count)*width} and tuple(zero.attributes.get(k) for k in ('operation','reads','writes','source_constant'))==('fill',0,1,0) and not iv(zero) and len(ov(zero))==1 and cp in pred[zero.node_id] and zero in pred[consumer.node_id],'initial zero and data chain')
                    cv=port(cp,ov(cp)[0],(count,h) if width==h else (count,),dtype,True);zv=port(zero,ov(zero)[0],(padded-count,h) if width==h else (padded-count,),dtype,True)
                    require(cv.allocation_id==zv.allocation_id==target.allocation_id and cv.offset_bytes==0 and zv.offset_bytes==count*width*size and target.offset_bytes==lane*local*width*size,'initial physical backing/lane')
                    fills.add(zero.node_id)
        for layer in range(layers):
            gather=one([n for n in graph.nodes if n.kind.value=='collective' and n.node_id.endswith(f'layer-{layer}:sp-attention-gather') and tuple(n.attributes['participants'])==group],'attention gather')
            scatter=one([n for n in graph.nodes if n.kind.value=='collective' and n.node_id.endswith(f'layer-{layer}:sp-attention-scatter') and tuple(n.attributes['participants'])==group],'attention scatter')
            ga,sa=gather.attributes,scatter.attributes
            require(ga['operation']=='all_gather' and ga['reduction'] is None and ga['bytes']==2*local*h and ga['bytes_semantics']=='per_rank_input' and ga['symbolic_collective_dim']==0 and ga['symbolic_collective_storage']=='fresh','attention gather geometry')
            require(sa['operation']=='reduce_scatterv' and sa['reduction']=='sum' and sa['collective_group']=='tensor' and tuple(sa['symbolic_variable_rows'])==(local,)*tp and tuple(sa['symbolic_variable_row_shape'])==(h,) and tuple(sa['partition_bytes'])==(2*local*h,)*tp and sa['bytes']==2*padded*h and sa['bytes_semantics']=='per_rank_input','attention scatter geometry')
            require(len(iv(gather))==len(ov(gather))==len(iv(scatter))==len(ov(scatter))==tp,'complete TP ports')
            require(set(n.node_id for n in pred[gather.node_id])==set(ga['symbolic_collective_input_producers']) and set(n.node_id for n in pred[scatter.node_id])==set(sa['symbolic_collective_input_producers']),'actual complete data dependencies')
            proved.add(scatter.node_id)
            for lane,r in enumerate(group):
                gv=port(gather,iv(gather)[lane],(local,h),'bf16');go=port(gather,ov(gather)[lane],(padded,h),'bf16',True)
                pre=one([x for x in pred[gather.node_id] if x.rank==r and x.attributes.get('source_component')=='vllm_0_30_symbolic_mhc'],'mHC pre to gather')
                require(iv(gather)[lane] in ov(pre),'TP gather ordered mHC lanes')
                linear=one([x for x in succ[gather.node_id] if x.rank==r and x.kind.value=='linear' and x.node_id.endswith(f'layer-{layer}:attn:fused-wqa-wkv')],'full attention input')
                lv=port(linear,iv(linear)[0],(count,h),'bf16');alias(go,lv)
                sv=port(scatter,iv(scatter)[lane],(padded,h),'bf16');so=port(scatter,ov(scatter)[lane],(local,h),'bf16',True)
                wo=one([x for x in graph.nodes if x.rank==r and x.node_id.endswith(f'layer-{layer}:attn:out:wo-b')],'attention full output')
                require(wo.shape['m']==count and wo.shape['n']==h,'attention output logical extent')
                if count==padded:
                    require(iv(scatter)[lane]==ov(wo)[0] and wo in pred[scatter.node_id],'direct full attention reduction')
                else:
                    cp=one([x for x in graph.nodes if x.rank==r and x.node_id.endswith(f'layer-{layer}:sp-attention-scatter:copy')],'attention padding copy')
                    zero=one([x for x in graph.nodes if x.rank==r and x.node_id.endswith(f'layer-{layer}:sp-attention-scatter:pad-zero')],'attention padding zero')
                    require(cp.kind.value=='gather' and cp.dtype=='bf16' and cp.shape['rows']==count and cp.shape['hidden']==h and iv(cp)==(ov(wo)[0],) and wo in pred[cp.node_id],'attention valid copy')
                    require(tuple(cp.attributes.get(k) for k in ('gather_factor','index_count','scattered','materialized_output'))==(1,0,False,True),'copy semantics')
                    require(zero.kind.value=='elementwise' and zero.dtype=='bf16' and dict(zero.shape)=={'elements':(padded-count)*h} and tuple(zero.attributes.get(k) for k in ('operation','reads','writes','source_constant'))==('fill',0,1,0) and cp in pred[zero.node_id] and zero in pred[scatter.node_id],'attention zero and data dependency')
                    fills.add(zero.node_id)
                    cv=port(cp,ov(cp)[0],(count,h),'bf16',True);zv=port(zero,ov(zero)[0],(padded-count,h),'bf16',True)
                    require(cv.allocation_id==zv.allocation_id==sv.allocation_id and cv.offset_bytes==sv.offset_bytes==0 and zv.offset_bytes==count*h*2,'attention padded backing')
                ffn=one([x for x in succ[scatter.node_id] if x.rank==r and x.node_id.endswith(f'layer-{layer}:ffn-post-pre')],'mHC after attention scatter')
                target=port(ffn,iv(ffn)[0],(local,h),'bf16');alias(so,target)
        final=one([n for n in graph.nodes if n.kind.value=='collective' and n.node_id.endswith('sp-final-hidden-gather') and tuple(n.attributes['participants'])==group],'final hidden restore')
        fa=final.attributes;require(fa['operation']=='all_gather' and fa['bytes']==local*4*h*2 and fa['bytes_semantics']=='per_rank_input' and fa['symbolic_collective_dim']==0 and len(iv(final))==len(ov(final))==tp,'final hidden gather geometry')
        require(set(n.node_id for n in pred[final.node_id])==set(fa['symbolic_collective_input_producers']),'final hidden complete dependencies')
        for lane,r in enumerate(group):
            port(final,iv(final)[lane],(local,4,h),'bf16');fv=port(final,ov(final)[lane],(padded,4,h),'bf16',True)
            producer=one([n for n in pred[final.node_id] if n.rank==r and n.variant=='mhc_post'],'final mHC post source')
            require(iv(final)[lane] in ov(producer),'final gather ordered rank outputs')
            head=one([x for x in succ[final.node_id] if x.rank==r and x.variant=='hc_head'],'logical mHC head');hv=port(head,iv(head)[0],(count,4,h),'bf16');alias(fv,hv)
    # Hash source valid-prefix masks are per lane; score layers route physical padding normally.
    for n in graph.nodes:
        if n.variant!='dsv4_moe_route':continue
        layer=int(n.node_id.split(':layer-')[1].split(':')[0]);rank=pool.logical_rank_for_physical_rank(n.rank);d,lane=divmod(rank,tp);local=expected_rows[rank]
        if layer<3:
            valid=max(0,min(local,logical[d]-lane*local));require(n.attributes.get('symbolic_padding_valid_prefix')==valid and n.attributes.get('source_padding_expert_ids')==-1 and n.attributes.get('source_padding_weights')==0,'hash local valid mask')
        else:require('symbolic_padding_valid_prefix' not in n.attributes,'score route cannot reuse hash mask')
    return proved,fills
