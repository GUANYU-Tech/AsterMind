"""Strict V3 0.27.1 H200 TP8 DP1 PP1 EP8 caller contract."""
from collections import defaultdict
from src.kernel.predictors.v3_ep_math import PROGRAM,logical_work
PROFILE='vllm_0_27_1_deepseek_v3_sync_fp8'

def applicable(pool):
    c=pool.parallel_plan.config
    return (pool.hardware_id=='nvidia_h200' and c.enable_expert_parallel
        and c.tensor_parallel_size==c.expert_parallel_size and c.tensor_parallel_size in (2,4,8,16)
        and c.data_parallel_size==c.sequence_parallel_size==c.context_parallel_size==1
        and c.pipeline_parallel_size in (1,2,3,4,6,8)
        and len(pool.ranks)==c.worker_count==c.tensor_parallel_size*c.pipeline_parallel_size)

def source_node(node):
    a=node.attributes
    from src.serving.workload.model_config import DEEPSEEK_V3_CONFIG_SHA256,DEEPSEEK_V3_L32_CONFIG_SHA256
    identities={DEEPSEEK_V3_CONFIG_SHA256:61,DEEPSEEK_V3_L32_CONFIG_SHA256:32}
    return (a.get('source_projection_only') is True and a.get('model_execution_version')=='vllm_0_27_1'
        and a.get('model_source_profile')==PROFILE and a.get('source_model_profile')==PROFILE
        and a.get('source_model_config_sha256') in identities
        and a.get('source_target_layers')==identities[a['source_model_config_sha256']]
        and a.get('source_tensor_parallel_size') in (2,4,8,16) and a.get('source_model_execution_version')=='vllm_0_27_1')

def expert_attributes(node):
    a=node.attributes
    if (not source_node(node) or a.get('source_component')!='symbolic_dsv3_routed_experts'
        or node.kind.value!='fused_moe' or node.variant!='full' or node.dtype!='bf16'
        or a.get('launch_policy')!='analytical_launch' or a.get('source_expert_tensor_parallel_size')!=1
        or a.get('expert_placement_strategy')!='linear' or a.get('output_is_reduced') is not False
        or a.get('checkpoint_weight_dtype')!='fp8' or a.get('routing_source_stage')!='moe_topk'
        or 'source_dispatched_input_dtype' in a or 'source_input_quantization' in a):
        raise ValueError('V3 EP8 requires exact replicated BF16-input source')
    counts=tuple(a['tokens_per_expert']);owned=tuple(a['owned_experts']);compact=tuple(a['local_tokens_per_expert'])
    if type(a['expert_parallel_size']) is not int or a['expert_parallel_size'] not in (2,4,8,16):raise ValueError('unsupported V3 EP width')
    per_expert=256//a['expert_parallel_size']
    if len(owned)!=per_expert or len(compact)!=per_expert or any(type(v) is not int or v<0 for v in compact):
        raise ValueError('V3 source must retain exactly32 compact owned counts')
    if any(type(j) is not int or not 0<=j<256 for j in owned) or compact!=tuple(counts[j] for j in owned):
        raise ValueError('V3 compact counts differ from global histogram')
    attrs={name:a[name] for name in ('tokens_per_expert','expert_parallel_size','expert_rank','owned_experts',
        'routed_scaling_factor','routed_weight_dtype','weight_group_size')}
    attrs.update(local_tokens_per_expert=tuple(v if j in owned else 0 for j,v in enumerate(counts)),
        expert_weight_tile_rows=None,swiglu_limit=None,mathematical_program=PROGRAM)
    logical_work(node.shape,attrs)
    return attrs

def validate(graph,pool,nodes,memory_context):
    if not applicable(pool):raise NotImplementedError('V3 EP8 requires H200 TP8 DP1 PP1/2/3/4 SP1 CP1')
    if memory_context is None:raise ValueError('V3 EP8 requires source tensor memory ports')
    tp=pool.parallel_plan.config.tensor_parallel_size; shared_i=2048//tp
    memory=memory_context._by_node;byid={n.node_id:n for n in graph.nodes};out=defaultdict(list);succ=defaultdict(list)
    for n in graph.nodes:
        for v in n.attributes.get('symbolic_tensor_outputs',()):out[v].append(n)
    for edge in graph.edges:
        if edge.kind=='data':succ[edge.source].append(byid[edge.destination])
    def ins(n):return tuple(n.attributes.get('symbolic_tensor_inputs',()))
    def outs(n):return tuple(n.attributes.get('symbolic_tensor_outputs',()))
    def fail(c,msg):
        if c:raise ValueError('V3 EP8 caller: '+msg)
    def port(n,name,shape,dtype,output=False):
        views=memory[n.node_id].outputs if output else memory[n.node_id].inputs
        selected=[v for v in views if v.view_id==name]
        fail(len(selected)!=1 or tuple(selected[0].shape)!=tuple(shape) or selected[0].dtype!=dtype,'typed tensor port mismatch')
    groups=defaultdict(list)
    for n in nodes:
        expert_attributes(n);groups[n.attributes['moe_invocation_id']].append(n)
    fail(not nodes,'missing expert nodes')
    layers=nodes[0].attributes['source_target_layers']
    fail(len(groups)!=layers-3 or {n.attributes['routing_provenance']['layer_index'] for n in nodes}!=set(range(3,layers)),'incomplete MoE layers')
    for members in groups.values():
        first=members[0];t=first.shape['tokens'];h=7168;participants=tuple(pool.physical_rank_group('tensor',pool.logical_rank_for_physical_rank(first.rank)))
        fail(len(members)!=tp or {n.rank for n in members}!=set(participants),'incomplete EP rank group')
        adds={};reds={}
        for n in members:
            a=n.attributes;p=ins(n)
            fail(pool.parallel_plan.layer_to_pipeline_stage[a['routing_provenance']['layer_index']] != pool.logical_rank_for_physical_rank(n.rank)//tp,'expert layer belongs to another PP stage')
            fail(n.shape!=first.shape or a['tokens_per_expert']!=first.attributes['tokens_per_expert'] or a['routing_provenance']!=first.attributes['routing_provenance'],'rank routing differs')
            fail(a['source_tensor_parallel_size']!=tp or a['expert_parallel_size']!=tp,'source TP/EP disagrees with deployment')
            fail(a['expert_rank']!=participants.index(n.rank) or len(p)!=3 or len(set(p))!=3 or len(outs(n))!=1,'expert rank/ports')
            port(n,p[0],(t,h),'bf16');port(n,p[1],(t,8),'fp32');port(n,p[2],(t,8),'int32');port(n,outs(n)[0],(t,h),'bf16',True)
            routes=[x for x in out[p[1]] if outs(x)==p[1:]];fail(len(routes)!=1,'exact route producer absent');route=routes[0];ra=route.attributes
            expected=dict(source_component='vllm_0_27_1_symbolic_grouped_routing',source_program='vllm_0_27_1_bf16_sigmoid_bias_grouped_topk',
                mathematical_program='bf16_sigmoid_bias_grouped_topk_flops_bytes_v1',normalize_topk_probability=True,expert_group_count=8,topk_group_count=4,routed_scaling_factor=1.0)
            fail(not source_node(route) or any(ra.get(k)!=v for k,v in expected.items()) or route.kind.value!='vector' or route.variant!='moe_topk'
                or route.dtype!='bf16' or route.shape!=n.shape or route.rank!=n.rank or route.pool_name!=n.pool_name or n not in succ[route.node_id] or len(ins(route))!=2,'router contract')
            port(route,p[1],(t,8),'fp32',True);port(route,p[2],(t,8),'int32',True)
            gates=[x for x in out[ins(route)[0]] if x.kind.value=='router'];fail(len(gates)!=1,'missing same-hidden gate');gate=gates[0]
            fail(gate.rank!=n.rank or gate.pool_name!=n.pool_name or dict(gate.shape)!={'b':1,'m':t,'n':256,'k':h}
                or gate.dtype!='bf16' or ins(gate)[0]!=p[0] or route not in succ[gate.node_id] or ins(route)[1] in (*p,*outs(gate)),'gate/correction bias binding')
            port(gate,outs(gate)[0],(t,256),'bf16',True);port(route,ins(route)[1],(256,),'fp32')
            scales=[x for x in succ[n.node_id] if x.kind.value=='elementwise' and x.variant=='elementwise'];fail(len(scales)!=1,'missing unique external scale');scale=scales[0]
            fail(scale.rank!=n.rank or scale.pool_name!=n.pool_name or scale.dtype!='bf16' or dict(scale.shape)!={'elements':t*h}
                or ins(scale)!=outs(n) or outs(scale)!=outs(n) or tuple(scale.attributes.get(k) for k in ('operation','reads','writes','scalar_value'))!=('unary',1,1,2.5),'outer scale must be2.5 once')
            merges=[x for x in succ[scale.node_id] if x.kind.value=='elementwise'];fail(len(merges)!=1,'missing shared merge');add=merges[0]
            fail(add.variant!='add' or add.dtype!='bf16' or add.rank!=n.rank or add.pool_name!=n.pool_name or dict(add.shape)!={'elements':t*h}
                or tuple(add.attributes.get(k) for k in ('operation','reads','writes'))!=('add',2,1) or len(ins(add))!=2 or ins(add).count(outs(n)[0])!=1,'shared merge semantics')
            other=next(v for v in ins(add) if v!=outs(n)[0]);downs=[x for x in out[other] if x.kind.value=='linear' and x.rank==n.rank];fail(len(downs)!=1,'shared down absent');down=downs[0]
            fail(dict(down.shape)!={'b':1,'m':t,'n':h,'k':shared_i} or down.attributes.get('weight_dtype')!='fp8' or down.attributes.get('weight_group_size')!=128,'shared down shard/quantization')
            acts=[x for v in ins(down) for x in out[v] if x.kind.value=='activation' and x.rank==n.rank];fail(len(acts)!=1,'shared activation absent');act=acts[0]
            fail(act.variant!='silu' or act.dtype!='bf16' or act.attributes.get('swiglu_limit') is not None or dict(act.shape)!={'elements':t*shared_i,'tok':t,'intermediate':shared_i},'shared activation shape')
            ups=[x for v in ins(act) for x in out[v] if x.kind.value=='linear' and x.rank==n.rank];fail(len(ups)!=1,'shared up absent');up=ups[0]
            fail(dict(up.shape)!={'b':1,'m':t,'n':2*shared_i,'k':h} or ins(up)!=(p[0],) or up.attributes.get('weight_dtype')!='fp8' or up.attributes.get('weight_group_size')!=128,'shared up shard/input')
            reductions=[x for x in succ[add.node_id] if x.kind.value=='collective'];fail(len(reductions)!=1,'missing TP collective');red=reductions[0]
            fail(red.attributes.get('operation')!='all_reduce' or red.attributes.get('reduction')!='sum' or tuple(red.attributes['participants'])!=participants
                or red.attributes.get('bytes')!=2*t*h or red.attributes.get('bytes_semantics')!='per_rank_input' or red.attributes.get('moe_invocation_id')!=a['moe_invocation_id'],'TP reduction contract')
            adds[n.rank]=add;reds[red.node_id]=red
        fail(len(reds)!=1,'ranks must share one source reduction');red=next(iter(reds.values()))
        fail(ins(red)!=tuple(outs(adds[r])[0] for r in participants) or tuple(red.attributes['symbolic_collective_input_producers'])!=tuple(adds[r].node_id for r in participants) or len(outs(red))!=tp,'reduction ports')
        for v in outs(red):port(red,v,(t,h),'bf16',True)

