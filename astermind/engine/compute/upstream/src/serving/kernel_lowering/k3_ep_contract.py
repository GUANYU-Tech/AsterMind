"""K3 uniform stage-local tensor/expert parallel source, DP1, Triton KDA only."""
from collections import defaultdict
from src.kernel.predictors.moe_ep_math import marlin_ep_work

from src.kernel.hardware_spec import HWSpec


def applicable(pool):
    from .k3_sp_contract import applicable as sp_applicable
    if sp_applicable(pool):return True
    from .k3_dp_contract import applicable as dp_applicable
    if dp_applicable(pool):return True
    c=pool.parallel_plan.config
    return ((pool.hardware_id == "nvidia_h200" or HWSpec.source_contract_supported(
            "kimi_k3_portable_roofline", pool.hardware_id)) and pool.parallel_plan.num_layers==93 and c.enable_expert_parallel
            and c.tensor_parallel_size==c.expert_parallel_size and c.tensor_parallel_size in (2,4,8,16,32)
            and c.data_parallel_size==c.context_parallel_size==c.sequence_parallel_size==1
            and 1<=c.pipeline_parallel_size<=pool.parallel_plan.num_layers)

def allowed_node(node,pool):
    if not applicable(pool):return False
    a=node.attributes;s=node.shape;heads=96//pool.parallel_plan.config.tensor_parallel_size
    component=a.get('source_component','')
    if component=='vllm_0_30_kimi_k3_native_marlin_moe':
        return node.dtype=='fp32' and node.kind.value=='vector' and node.variant=='kimi_k3_moe_route' and s['experts']==896 and s['topk']==16
    if node.dtype!='bf16':return False
    if component=='vllm_0_30_kimi_k3_attention_residual':
        return (node.kind.value=='vector' and node.variant=='kimi_k3_attn_res'
                and s.get('hidden')==7168 and s.get('elements')==s['rows']*7168
                and a.get('source_projection_only') is True and a.get('model_execution_version')=='vllm_0_30_0')
    if component=='vllm_0_30_kimi_k3_ordinary_mlp':
        from .k3_sp_contract import applicable as sp_applicable
        if sp_applicable(pool):
            return (node.kind.value=='activation' and node.variant=='kimi_k3_situ'
                    and s['intermediate'] in (33792,6144) and a['activation_beta']==4.0 and a['activation_linear_beta']==25.0)
        return (node.kind.value=='activation' and node.variant=='kimi_k3_situ'
                and s['intermediate'] in (33792//pool.parallel_plan.config.tensor_parallel_size,6144//pool.parallel_plan.config.tensor_parallel_size)
                and a['activation_beta']==4.0 and a['activation_linear_beta']==25.0)
    if component in ('vllm_0_30_kimi_k3_triton_fresh_prefill','vllm_0_30_kimi_k3_triton_decode'):
        return (node.kind.value=='vector' and s.get('heads')==heads and s.get('head_dim')==128
                and node.variant == ('kimi_k3_kda_prefill' if component.endswith('fresh_prefill') else 'kimi_k3_kda_decode'))
    if component in ('vllm_0_30_kimi_k3_nope_fresh_prefill','vllm_0_30_kimi_k3_nope_decode'):
        return (node.kind.value=='vector' and s.get('heads')==heads and s.get('q_lora_rank')==1536
                and s.get('kv_lora_rank')==512 and node.variant == ('kimi_k3_nope_prefill' if component.endswith('fresh_prefill') else 'kimi_k3_nope_decode'))
    if (component == "vllm_0_30_symbolic_k3_prefill_attention"
            and HWSpec.source_contract_supported("kimi_k3_portable_roofline", pool.hardware_id)):
        return (node.kind.value == "flash_attn" and node.variant == "mla_prefill"
            and s.get("num_query_heads") == heads and s.get("qk_head_dim") == 192
            and s.get("v_head_dim") == 128 and s.get("kv_lora_rank") == 512)
    if component=='symbolic_mla_decode_mathematical':
        return (node.kind.value=='flash_attn' and node.variant=='mla_decode'
                and s.get('num_query_heads')==heads and s.get('qk_head_dim')==576
                and s.get('v_head_dim')==512 and s.get('kv_lora_rank')==512
                and bool(a.get('q_lens')) and all(q==1 for q in a['q_lens']))
    return False

def expert_attributes(node):
    a=node.attributes
    if (a['source_component']!='vllm_0_30_kimi_k3_marlin_experts' or node.dtype!='bf16'
            or a['expert_tensor_parallel_size']!=1 or a['logical_intermediate']!=3072
            or a['expert_output_is_reduced'] is not False):
        raise ValueError('K3 EP requires source whole experts and unreduced local output')
    attrs={k:a[k] for k in ('tokens_per_expert','local_tokens_per_expert','expert_parallel_size','expert_rank',
                           'owned_experts','expert_weight_tile_rows','activation_beta','activation_linear_beta','routed_scaling_factor')}
    attrs.update(routed_weight_dtype='mxfp4',weight_group_size=32,swiglu_limit=None,mathematical_program='k3_masked_marlin_ep_v1')
    marlin_ep_work(node.shape,attrs,model='k3')
    return attrs

def validate(graph,pool,nodes,memory_context):
    from .k3_sp_contract import applicable as sp_applicable,validate as sp_validate
    if sp_applicable(pool):return sp_validate(graph,pool,nodes,memory_context)
    if pool.parallel_plan.config.data_parallel_size>1:
        from .k3_dp_contract import validate as dp_validate
        return dp_validate(graph,pool,nodes,memory_context)
    if not applicable(pool):raise NotImplementedError('K3 EP initial prediction requires H200 DP1 legal PP CP1 SP1')
    if memory_context is None:raise ValueError('K3 EP requires current typed memory ports')
    memory=memory_context._by_node;out=defaultdict(list);succ=defaultdict(list);edges=set()
    for n in graph.nodes:
        for v in n.attributes.get('symbolic_tensor_outputs',()):out[v].append(n)
    byid={n.node_id:n for n in graph.nodes}
    for e in graph.edges:
        edges.add((e.source,e.destination))
        if e.kind=='data':succ[e.source].append(byid[e.destination])
    def ins(n):return tuple(n.attributes.get('symbolic_tensor_inputs',()))
    def data_ins(n):return tuple(v for v in ins(n) if v not in n.attributes.get('symbolic_parameter_inputs',()))
    def outs(n):return tuple(n.attributes.get('symbolic_tensor_outputs',()))
    def check(ok,msg):
        if not ok:raise ValueError('K3 EP caller: '+msg)
    def one(xs,msg):
        check(len(xs)==1,msg);return xs[0]
    def typed(n,v,shape,dtype,output=False):
        vs=memory[n.node_id].outputs if output else memory[n.node_id].inputs
        x=one([x for x in vs if x.view_id==v],'missing typed view')
        check(tuple(x.shape)==shape and x.dtype==dtype,'typed shape/dtype')
    groups=defaultdict(list)
    for n in nodes:expert_attributes(n);groups[n.attributes['moe_invocation_id']].append(n)
    for members in groups.values():
        first=members[0];t=first.shape['tokens']
        layer=first.attributes['routing_provenance']['layer_index']
        check(type(layer) is int and 0<=layer<pool.parallel_plan.num_layers,'invalid layer identity')
        stage=pool.parallel_plan.layer_to_pipeline_stage[layer];tp=pool.parallel_plan.config.tensor_parallel_size
        expected=tuple(pool.physical_rank_for_logical_rank(stage*tp+lane) for lane in range(tp))
        participants=pool.physical_rank_group('tensor',pool.logical_rank_for_physical_rank(first.rank))
        check(tuple(participants)==expected,'expert group differs from pipeline layer placement')
        check(len(members)==len(participants) and {n.rank for n in members}==set(participants),'all expert ranks required')
        check(sorted(j for n in members for j in n.attributes['owned_experts'])==list(range(896)),'expert coverage')
        latent_reductions={};shared_reductions={};expert_outputs={};shared_outputs={}
        for n in members:
            a=n.attributes;p=ins(n);r=n.rank
            check(n.shape==first.shape and a['tokens_per_expert']==first.attributes['tokens_per_expert']
                  and a['routing_provenance']==first.attributes['routing_provenance'],'global routing geometry')
            check(a['expert_rank']==participants.index(r) and len(p)==3,'rank and ports')
            typed(n,p[0],(t,3584),'bf16');typed(n,p[1],(t,16),'fp32');typed(n,p[2],(t,16),'int32')
            route=one([x for x in out[p[1]] if x.variant=='kimi_k3_moe_route' and outs(x)==p[1:]],'router outputs')
            check(route.rank==r and n in succ[route.node_id] and route.attributes['routed_scaling_factor']==1.0
                  and route.attributes['score_function']=='sigmoid' and route.attributes['normalize_topk_probability'] is True,'routing semantics')
            gate=one([x for x in out[ins(route)[0]] if x.kind.value=='router' and x.rank==r],'router gate')
            down=one([x for x in out[p[0]] if x.kind.value=='linear' and x.rank==r],'latent down')
            check(dict(gate.shape)=={'b':1,'m':t,'n':896,'k':7168}
                  and dict(down.shape)=={'b':1,'m':t,'n':3584,'k':7168} and data_ins(gate)==data_ins(down)
                  and len(data_ins(down))==1,'gate/latent source')
            hidden=data_ins(down)[0]
            lr=one([x for x in succ[n.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'latent reduction')
            index=participants.index(r);lv=outs(lr)[index]
            norm=one([x for x in succ[lr.node_id] if x.rank==r and x.kind.value=='norm' and ins(x)[0]==lv],'latent norm')
            check(norm.variant=='rmsnorm' and dict(norm.shape)=={'rows':t,'hidden':3584},'latent normalization geometry')
            up=one([x for x in succ[norm.node_id] if x.rank==r and x.kind.value=='linear' and ins(x)[0]==outs(norm)[0]],'latent up')
            check(dict(up.shape)=={'b':1,'m':t,'n':7168,'k':3584},'full latent up')
            add=one([x for x in succ[up.node_id] if x.rank==r and x.kind.value=='elementwise'],'shared merge')
            check(add.variant=='add' and add.attributes['operation']=='add' and dict(add.shape)=={'elements':t*7168}
                  and add.attributes['reads']==2 and add.attributes['writes']==1 and add.dtype=='bf16','shared addition')
            other=one([v for v in ins(add) if v!=outs(up)[0]],'shared output port')
            sr=one([x for x in out[other] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'shared reduction')
            check((sr.node_id,norm.node_id) in edges,'shared reduction barrier before norm')
            sv=ins(sr)[index];shared=one([x for x in out[sv] if x.rank==r and x.kind.value=='linear'],'shared down')
            width=6144//len(participants)
            check(dict(shared.shape)=={'b':1,'m':t,'n':7168,'k':width},'shared TP width')
            act=one([x for v in ins(shared) for x in out[v] if x.rank==r and x.variant=='kimi_k3_situ'],'shared SiTU')
            check(act.shape['intermediate']==width and act.attributes['activation_beta']==4.0 and act.attributes['activation_linear_beta']==25.0,'shared SiTU geometry')
            gu=one([x for v in ins(act) for x in out[v] if x.rank==r and x.kind.value=='linear'],'shared gate/up')
            check(dict(gu.shape)=={'b':1,'m':t,'n':2*width,'k':7168} and data_ins(gu)==(hidden,),'shared input and width')
            check(all(x.attributes.get('weight_dtype')=='bf16' and x.attributes.get('weight_group_size')==0
                      and x.attributes.get('has_bias') is False for x in (gu,shared)),'shared BF16 unquantized weights')
            for red,width_out in ((lr,3584),(sr,7168)):
                check(tuple(red.attributes['participants'])==tuple(participants) and red.attributes['reduction']=='sum'
                      and red.attributes['bytes']==2*t*width_out and red.attributes['bytes_semantics']=='per_rank_input','reduction participants and bytes')
            latent_reductions[lr.node_id]=lr;shared_reductions[sr.node_id]=sr;expert_outputs[r]=outs(n)[0];shared_outputs[r]=outs(shared)[0]
        check(len(latent_reductions)==len(shared_reductions)==1,'one complete group per reduction')
        check(ins(next(iter(latent_reductions.values())))==tuple(expert_outputs[r] for r in participants),'latent reduction inputs')
        check(ins(next(iter(shared_reductions.values())))==tuple(shared_outputs[r] for r in participants),'shared reduction inputs')
