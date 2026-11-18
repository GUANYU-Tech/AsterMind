"""K3 all-active WideEP, PP>1: DP dispatch/combine followed by local TP tail."""
from collections import defaultdict
from contextvars import ContextVar

_validated_variables=ContextVar('k3_wideep_validated_variables',default=None)

def handles_collective(node,pool,graph):
    proof=_validated_variables.get()
    return (applicable(pool) and proof is not None and proof[0] is graph
            and node.node_id in proof[1] and node.attributes.get('collective_group')=='data')

def applicable(pool):
    c=pool.parallel_plan.config
    return (pool.hardware_id=='nvidia_h200' and pool.parallel_plan.num_layers==93
            and c.enable_expert_parallel and c.tensor_parallel_size in (1,2,4,8,16,32)
            and c.data_parallel_size>1 and 1<c.pipeline_parallel_size<=93
            and c.expert_parallel_size==c.tensor_parallel_size*c.data_parallel_size
            and 896%c.expert_parallel_size==0 and c.context_parallel_size==c.sequence_parallel_size==1)

def validate(graph,pool,nodes,memory_context):
    _validated_variables.set(None)
    from .k3_ep_contract import expert_attributes
    if not applicable(pool):raise NotImplementedError('K3 WideEP requires H200 all-active DP, PP>1 non-SP, uniform TP*DP experts')
    if memory_context is None:raise ValueError('K3 WideEP requires actual typed tensor ports')
    c=pool.parallel_plan.config;tp,dp,pp=c.tensor_parallel_size,c.data_parallel_size,c.pipeline_parallel_size
    memory=memory_context._by_node;out=defaultdict(list);succ=defaultdict(list);edges=set();byid={n.node_id:n for n in graph.nodes}
    for n in graph.nodes:
        if n.attributes.get('source_dummy_run') is True:raise NotImplementedError('K3 WideEP idle/dummy rows require a separate retained-state contract')
        for v in n.attributes.get('symbolic_tensor_outputs',()):out[v].append(n)
    for e in graph.edges:
        edges.add((e.source,e.destination))
        if e.kind=='data':succ[e.source].append(byid[e.destination])
    def ins(n):return tuple(n.attributes.get('symbolic_tensor_inputs',()))
    def data_ins(n):return tuple(v for v in ins(n) if v not in n.attributes.get('symbolic_parameter_inputs',()))
    def outs(n):return tuple(n.attributes.get('symbolic_tensor_outputs',()))
    def check(ok,msg):
        if not ok:raise ValueError('K3 WideEP caller: '+msg)
    def one(xs,msg):check(len(xs)==1,msg);return xs[0]
    def typed(n,v,shape,dtype,output=False):
        vs=memory[n.node_id].outputs if output else memory[n.node_id].inputs
        x=one([x for x in vs if x.view_id==v],'missing typed view')
        check(tuple(x.shape)==shape and x.dtype==dtype,'typed shape/dtype')
    def physical(replica,stage,lane):return pool.physical_rank_for_logical_rank((replica*pp+stage)*tp+lane)
    groups=defaultdict(list);variable_ids=set()
    for n in nodes:expert_attributes(n);groups[n.attributes['moe_invocation_id']].append(n)
    for members in groups.values():
        first=members[0];total=first.shape['tokens'];layer=first.attributes['routing_provenance']['layer_index']
        check(type(layer) is int and 0<=layer<93,'layer identity')
        stage=pool.parallel_plan.layer_to_pipeline_stage[layer]
        epgroup=tuple(physical(d,stage,l) for d in range(dp) for l in range(tp))
        check(len(members)==tp*dp and {n.rank for n in members}==set(epgroup),'complete stage-local EP group')
        check(sorted(j for n in members for j in n.attributes['owned_experts'])==list(range(896)),'expert coverage')
        expert_outputs={n.rank:outs(n)[0] for n in members};copies={};shared_outputs={};latents={};shareds={};combines={};row_vectors=set()
        for n in members:
            a=n.attributes;r=n.rank;logical=pool.logical_rank_for_physical_rank(r);d=logical//(tp*pp);lane=logical%tp
            dg=tuple(physical(x,stage,lane) for x in range(dp));tg=tuple(physical(d,stage,l) for l in range(tp))
            check(a['expert_rank']==d*tp+lane and n.shape==first.shape and a['tokens_per_expert']==first.attributes['tokens_per_expert']
                  and a['routing_provenance']==first.attributes['routing_provenance'],'global routing/rank identity')
            p=ins(n);check(len(p)==3,'three expert inputs')
            local_ports=[];dispatch_rows=None
            for v,width,dtype,size in zip(p,(3584,16,16),('bf16','fp32','int32'),(2,4,4)):
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
            route=one([x for x in out[weight_port] if x.variant=='kimi_k3_moe_route' and x.rank==r and outs(x)==(weight_port,id_port)],'original replica router')
            check(route.dtype=='fp32' and route.shape['rows']==t and route.attributes['routed_scaling_factor']==1.0
                  and route.attributes['score_function']=='sigmoid' and route.attributes['normalize_topk_probability'] is True,'router semantics')
            typed(route,weight_port,(t,16),'fp32',True);typed(route,id_port,(t,16),'int32',True)
            gate=one([x for x in out[ins(route)[0]] if x.kind.value=='router' and x.rank==r],'router gate')
            down=one([x for x in out[hidden_port] if x.kind.value=='linear' and x.rank==r],'latent down')
            check(dict(gate.shape)=={'b':1,'m':t,'n':896,'k':7168} and dict(down.shape)=={'b':1,'m':t,'n':3584,'k':7168}
                  and data_ins(gate)==data_ins(down) and len(data_ins(down))==1,'local hidden/route source')
            typed(down,hidden_port,(t,3584),'bf16',True);hidden=data_ins(down)[0]
            combine=one([x for x in succ[n.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='reduce_scatterv'],'DP combine')
            ca=combine.attributes
            variable_ids.add(combine.node_id)
            check(tuple(ca['participants'])==dg and ca['collective_group']=='data' and ca['reduction']=='sum'
                  and ca['bytes_semantics']=='per_rank_input' and ca['bytes']==2*total*3584
                  and tuple(ca['partition_bytes'])==tuple(x*2*3584 for x in dispatch_rows)
                  and tuple(ca['symbolic_variable_rows'])==dispatch_rows and tuple(ca['symbolic_variable_row_shape'])==(3584,)
                  and ins(combine)==tuple(expert_outputs[x] for x in dg),'combine groups/partition/input order')
            combines[combine.node_id]=combine;cv=outs(combine)[d]
            copy=one([x for x in succ[combine.node_id] if x.rank==r and x.kind.value=='gather' and ins(x)==(cv,)],'replica finalize copy')
            check(copy.dtype=='bf16' and dict(copy.shape)=={'elements':t*3584,'rows':t,'hidden':3584}
                  and copy.attributes['materialized_output'] is True and copy.attributes['gather_factor']==1,'materialized finalize')
            typed(copy,cv,(t,3584),'bf16');typed(copy,outs(copy)[0],(t,3584),'bf16',True);copies[r]=outs(copy)[0]
            if tp==1:
                check(len(outs(copy))==1 and not [x for x in succ[copy.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'TP1 finalize has no TP reduction')
                lr=copy
            else:
                lr=one([x for x in succ[copy.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'local TP latent reduction')
            lv=outs(lr)[lane];norm=one([x for x in succ[lr.node_id] if x.rank==r and x.kind.value=='norm' and ins(x)[0]==lv],'latent norm')
            check(norm.variant=='rmsnorm' and dict(norm.shape)=={'rows':t,'hidden':3584},'local normalization')
            up=one([x for x in succ[norm.node_id] if x.rank==r and x.kind.value=='linear' and ins(x)[0]==outs(norm)[0]],'latent up')
            check(dict(up.shape)=={'b':1,'m':t,'n':7168,'k':3584},'full latent up')
            add=one([x for x in succ[up.node_id] if x.rank==r and x.kind.value=='elementwise'],'shared merge')
            check(add.variant=='add' and add.attributes['operation']=='add' and dict(add.shape)=={'elements':t*7168}
                  and add.attributes['reads']==2 and add.attributes['writes']==1 and add.dtype=='bf16','shared addition')
            other=one([v for v in ins(add) if v!=outs(up)[0]],'shared output')
            if tp==1:
                shared=one([x for x in out[other] if x.rank==r and x.kind.value=='linear'],'TP1 direct shared down')
                check(len(outs(shared))==1 and not [x for x in succ[shared.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'TP1 shared has no TP reduction')
                check((shared.node_id,add.node_id) in edges and len(ins(add))==2 and len(outs(add))==1,'TP1 shared branch must join local up output')
                sr=shared
            else:
                sr=one([x for x in out[other] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'shared reduction')
                check((sr.node_id,norm.node_id) in edges,'shared reduction before norm')
                shared=one([x for x in out[ins(sr)[lane]] if x.rank==r and x.kind.value=='linear'],'shared down')
            width=6144//tp
            check(dict(shared.shape)=={'b':1,'m':t,'n':7168,'k':width},'shared TP width')
            act=one([x for v in ins(shared) for x in out[v] if x.rank==r and x.variant=='kimi_k3_situ'],'shared SiTU')
            check(act.shape['intermediate']==width and act.attributes['activation_beta']==4.0 and act.attributes['activation_linear_beta']==25.0,'shared activation')
            gu=one([x for v in ins(act) for x in out[v] if x.rank==r and x.kind.value=='linear'],'shared gate/up')
            check(dict(gu.shape)=={'b':1,'m':t,'n':2*width,'k':7168} and data_ins(gu)==(hidden,),'shared input')
            check(all(x.attributes.get('weight_dtype')=='bf16' and x.attributes.get('weight_group_size')==0
                      and x.attributes.get('has_bias') is False for x in (gu,shared)),'shared BF16 unquantized weights')
            if tp>1:
                for red,width_out in ((lr,3584),(sr,7168)):
                    check(tuple(red.attributes['participants'])==tg and red.attributes['reduction']=='sum'
                          and red.attributes['bytes']==2*t*width_out and red.attributes['bytes_semantics']=='per_rank_input','replica TP group/bytes')
            latents[lr.node_id]=lr;shareds[sr.node_id]=sr;shared_outputs[r]=outs(shared)[0]
        check(len(row_vectors)==1 and len(latents)==len(shareds)==dp and len(combines)==tp,'complete DP/TP communication groups')
        if tp>1:
            for red in latents.values():check(ins(red)==tuple(copies[x] for x in red.attributes['participants']),'latent TP consumes local finalized rows')
            for red in shareds.values():check(ins(red)==tuple(shared_outputs[x] for x in red.attributes['participants']),'shared TP consumes local shared outputs')
    _validated_variables.set((graph,frozenset(variable_ids)))
