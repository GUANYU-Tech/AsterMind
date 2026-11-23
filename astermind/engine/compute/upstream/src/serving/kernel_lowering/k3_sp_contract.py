"""K3 active model-SP WideEP PP1: full EP row dispatch/combine; local replicated shared tail."""
from collections import defaultdict
from contextvars import ContextVar

from src.kernel.hardware_spec import HWSpec

_validated_variables=ContextVar('k3_wideep_validated_variables',default=None)

def handles_collective(node,pool,graph):
    proof=_validated_variables.get()
    return (applicable(pool) and proof is not None and proof[0] is graph
            and proof[1] is pool and node.node_id in proof[2])

def applicable(pool):
    c=pool.parallel_plan.config
    return ((pool.hardware_id == "nvidia_h200" or HWSpec.source_contract_supported(
                "kimi_k3_portable_roofline", pool.hardware_id)) and pool.parallel_plan.num_layers==93
            and c.enable_expert_parallel and c.tensor_parallel_size in (2,4,8,16,32)
            and c.data_parallel_size>1 and c.pipeline_parallel_size==1
            and c.expert_parallel_size==c.tensor_parallel_size*c.data_parallel_size
            and 896%c.expert_parallel_size==0 and c.context_parallel_size==c.sequence_parallel_size==1)

def validate(graph,pool,nodes,memory_context):
    _validated_variables.set(None)
    from .k3_ep_contract import expert_attributes
    if not applicable(pool):raise NotImplementedError('K3 model-SP requires H200 TP>1 DP>1 PP1, uniform whole experts')
    if memory_context is None:raise ValueError('K3 WideEP requires actual typed tensor ports')
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
            dg=epgroup;tg=tuple(physical(d,stage,l) for l in range(tp));owner=d*tp+lane
            check(a['expert_rank']==d*tp+lane and n.shape==first.shape and a['tokens_per_expert']==first.attributes['tokens_per_expert']
                  and a['routing_provenance']==first.attributes['routing_provenance'],'global routing/rank identity')
            p=ins(n);check(len(p)==3,'three expert inputs')
            local_ports=[];dispatch_rows=None
            for v,width,dtype,size in zip(p,(3584,16,16),('bf16','fp32','int32'),(2,4,4)):
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
            route=one([x for x in out[weight_port] if x.variant=='kimi_k3_moe_route' and x.rank==r and outs(x)==(weight_port,id_port)],'original replica router')
            check(route.dtype=='fp32' and route.shape['rows']==t and route.attributes['routed_scaling_factor']==1.0
                  and route.attributes['score_function']=='sigmoid' and route.attributes['normalize_topk_probability'] is True,'router semantics')
            typed(route,weight_port,(t,16),'fp32',True);typed(route,id_port,(t,16),'int32',True)
            gate=one([x for x in out[ins(route)[0]] if x.kind.value=='router' and x.rank==r],'router gate')
            down=one([x for x in out[hidden_port] if x.kind.value=='linear' and x.rank==r],'latent down')
            check(dict(gate.shape)=={'b':1,'m':t,'n':896,'k':7168} and dict(down.shape)=={'b':1,'m':t,'n':3584,'k':7168}
                  and data_ins(gate)==data_ins(down) and len(data_ins(down))==1,'local hidden/route source')
            typed(down,hidden_port,(t,3584),'bf16',True);hidden=data_ins(down)[0]
            post=byid[n.node_id.rsplit(':moe:',1)[0]+':post-attn']
            check(post.rank==r and post.shape['rows']==t
                  and (post.node_id,down.node_id) in edges and (post.node_id,gate.node_id) in edges,'MoE reads its own local post-attention shard')
            post_view=one([v for v in memory[post.node_id].outputs if v.view_id==outs(post)[0]],'post residual output view')
            input_view=one([v for v in memory[down.node_id].inputs if v.view_id==hidden],'local hidden input view')
            check(post_view.allocation_id==input_view.allocation_id and post_view.offset_bytes==input_view.offset_bytes
                  and input_view.shape==post_view.shape==(t,7168) and input_view.dtype==post_view.dtype=='bf16','local post residual reshape alias')


            combine=one([x for x in succ[n.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='reduce_scatterv'],'DP combine')
            ca=combine.attributes
            variable_ids.add(combine.node_id)
            check(tuple(ca['participants'])==dg and ca['collective_group']=='expert' and ca['reduction']=='sum'
                  and ca['bytes_semantics']=='per_rank_input' and ca['bytes']==2*total*3584
                  and tuple(ca['partition_bytes'])==tuple(x*2*3584 for x in dispatch_rows)
                  and tuple(ca['symbolic_variable_rows'])==dispatch_rows and tuple(ca['symbolic_variable_row_shape'])==(3584,)
                  and ins(combine)==tuple(expert_outputs[x] for x in dg),'combine groups/partition/input order')
            combines[combine.node_id]=combine;cv=outs(combine)[owner]
            copy=one([x for x in succ[combine.node_id] if x.rank==r and x.kind.value=='gather' and ins(x)==(cv,)],'replica finalize copy')
            check(copy.dtype=='bf16' and dict(copy.shape)=={'elements':t*3584,'rows':t,'hidden':3584}
                  and copy.attributes['materialized_output'] is True and copy.attributes['gather_factor']==1,'materialized finalize')
            typed(copy,cv,(t,3584),'bf16');typed(copy,outs(copy)[0],(t,3584),'bf16',True);copies[r]=outs(copy)[0]
            check(len(outs(copy))==1 and not [x for x in succ[copy.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'SP finalize has no TP reduction')
            lr=copy
            lv=outs(lr)[0];norm=one([x for x in succ[lr.node_id] if x.rank==r and x.kind.value=='norm' and ins(x)[0]==lv],'latent norm')
            check(norm.variant=='rmsnorm' and dict(norm.shape)=={'rows':t,'hidden':3584},'local normalization')
            up=one([x for x in succ[norm.node_id] if x.rank==r and x.kind.value=='linear' and ins(x)[0]==outs(norm)[0]],'latent up')
            check(dict(up.shape)=={'b':1,'m':t,'n':7168,'k':3584},'full latent up')
            add=one([x for x in succ[up.node_id] if x.rank==r and x.kind.value=='elementwise'],'shared merge')
            check(add.variant=='add' and add.attributes['operation']=='add' and dict(add.shape)=={'elements':t*7168}
                  and add.attributes['reads']==2 and add.attributes['writes']==1 and add.dtype=='bf16','shared addition')
            other=one([v for v in ins(add) if v!=outs(up)[0]],'shared output')
            shared=one([x for x in out[other] if x.rank==r and x.kind.value=='linear'],'SP direct shared down')
            check(len(outs(shared))==1 and not [x for x in succ[shared.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'SP shared has no TP reduction')
            check((shared.node_id,add.node_id) in edges and len(ins(add))==2 and len(outs(add))==1,'SP shared branch joins local up output')
            sr=shared
            width=6144
            check(dict(shared.shape)=={'b':1,'m':t,'n':7168,'k':width},'shared TP width')
            act=one([x for v in ins(shared) for x in out[v] if x.rank==r and x.variant=='kimi_k3_situ'],'shared SiTU')
            check(act.shape['intermediate']==width and act.attributes['activation_beta']==4.0 and act.attributes['activation_linear_beta']==25.0,'shared activation')
            gu=one([x for v in ins(act) for x in out[v] if x.rank==r and x.kind.value=='linear'],'shared gate/up')
            check(dict(gu.shape)=={'b':1,'m':t,'n':2*width,'k':7168} and data_ins(gu)==(hidden,),'shared input')
            check(all(x.attributes.get('weight_dtype')=='bf16' and x.attributes.get('weight_group_size')==0
                      and x.attributes.get('has_bias') is False for x in (gu,shared)),'shared BF16 unquantized weights')
            latents[lr.node_id]=lr;shareds[sr.node_id]=sr;shared_outputs[r]=outs(shared)[0]
        check(len(row_vectors)==1 and len(latents)==len(shareds)==tp*dp and len(combines)==1,'complete EP communication groups')
    # Bind model-SP attention transport separately from the EP expert group.
    projections=defaultdict(list)
    for x in graph.nodes:
        if x.kind.value=='linear' and ':attention:' in x.node_id and x.shape.get('n')==7168:
            projections[(x.rank,x.node_id.split(':attention:',1)[0])].append(x)
    scatters=[x for x in graph.nodes if x.pool_name==pool.name and x.node_id.endswith(':sp-attention:reduce-scatter')]
    check(len(scatters)==93*dp,'all 93 attention scatter groups per replica')
    seen=set()
    for red in scatters:
        a=red.attributes;participants=tuple(a['participants']);logical_rank=pool.logical_rank_for_physical_rank(participants[0]);replica=logical_rank//tp
        tg=tuple(physical(replica,0,l) for l in range(tp));layer=int(red.node_id.rsplit(':layer-',1)[1].split(':',1)[0])
        check((replica,layer) not in seen and 0<=layer<93,'unique model-SP attention layer');seen.add((replica,layer))
        rows=tuple(a['symbolic_variable_rows']);check(len(rows)==tp and len(set(rows))==1 and type(rows[0]) is int and rows[0]>0,'equal TP row shards')
        t=rows[0];padded=t*tp
        check(participants==tg and a['operation']=='reduce_scatterv' and a['collective_group']=='tensor' and a['reduction']=='sum'
              and a['bytes_semantics']=='per_rank_input' and a['bytes']==padded*7168*2 and tuple(a['partition_bytes'])==(t*7168*2,)*tp
              and tuple(a['symbolic_variable_row_shape'])==(7168,) and len(ins(red))==len(outs(red))==tp,'attention scatter bytes/groups')
        variable_ids.add(red.node_id)
        ag=byid[red.node_id.replace(':sp-attention:reduce-scatter',':sp-attention-gather')]
        aa=ag.attributes
        check(tuple(aa['participants'])==tg and aa['operation']=='all_gather' and aa['bytes']==t*7168*2 and aa['bytes_semantics']=='per_rank_input'
              and aa['reduction'] is None and tuple(aa['symbolic_collective_input_shape'])==(t,7168) and aa['symbolic_collective_dim']==0
              and len(ins(ag))==len(outs(ag))==tp,'attention gather bytes/groups')
        for lane,r in enumerate(tg):
            pre=one([x for x in out[ins(ag)[lane]] if x.rank==r and x.node_id.endswith(':layer-'+str(layer)+':pre-attn')],'local pre-attention residual')
            check(pre.shape['rows']==t and (pre.node_id,ag.node_id) in edges,'local row shard gather dependency')
            typed(ag,ins(ag)[lane],(t,7168),'bf16');typed(ag,outs(ag)[lane],(padded,7168),'bf16',True)
            typed(red,ins(red)[lane],(padded,7168),'bf16');typed(red,outs(red)[lane],(t,7168),'bf16',True)
            post=one([x for x in succ[red.node_id] if x.rank==r and x.node_id.endswith(':layer-'+str(layer)+':post-attn')],'local post-attention residual')
            check(post.shape['rows']==t and outs(red)[lane] in ins(post),'scatter before local residual')
            # Projection/copy provenance is checked against typed storage, including zero-padding.
            candidates=projections[(r,pre.node_id.rsplit(':pre-attn',1)[0])]
            projection=one(candidates,'unreduced local attention output projection');actual=projection.shape['m']
            check((actual+tp-1)//tp==t and projection.dtype=='bf16','actual attention rows/padding')
            rv=one([v for v in memory[red.node_id].inputs if v.view_id==ins(red)[lane]],'scatter input view')
            pv=one([v for v in memory[projection.node_id].outputs if v.view_id==outs(projection)[0]],'attention output view')
            if actual==padded:
                check(rv.allocation_id==pv.allocation_id and rv.offset_bytes==pv.offset_bytes and (projection.node_id,red.node_id) in edges,'direct attention scatter provenance')
            else:
                cp=one([x for x in succ[projection.node_id] if x.rank==r and x.node_id.endswith(':sp-attention:copy')],'attention padding copy')
                cv=one([v for v in memory[cp.node_id].inputs if v.view_id==ins(cp)[0]],'padding source')
                ov=one([v for v in memory[cp.node_id].outputs if v.view_id==outs(cp)[0]],'padding destination')
                z=byid[cp.node_id.replace(':copy',':pad-zero')]
                zv=one(list(memory[z.node_id].outputs),'zero output')
                check(z.kind.value=='elementwise' and z.variant=='elementwise' and z.dtype=='bf16'
                      and z.attributes.get('operation')=='fill' and z.attributes.get('reads')==0 and z.attributes.get('writes')==1
                      and type(z.attributes.get('source_constant')) is int and z.attributes['source_constant']==0
                      and dict(z.shape)=={'elements':(padded-actual)*7168} and not ins(z) and len(outs(z))==1
                      and z in succ[cp.node_id] and red in succ[z.node_id],'zero padding value and actual copy/zero/scatter data order')

                check(cv.allocation_id==pv.allocation_id and cv.offset_bytes==pv.offset_bytes and cv.shape==(actual,7168)
                      and ov.allocation_id==rv.allocation_id==zv.allocation_id and ov.offset_bytes==rv.offset_bytes
                      and zv.offset_bytes==rv.offset_bytes+actual*7168*2 and zv.shape==(padded-actual,7168)
                      and (z.node_id,red.node_id) in edges,'valid and zero padded regions')
    _validated_variables.set((graph,pool,frozenset(variable_ids)))
