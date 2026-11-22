"""H200 K3 DP1 EP-off TP source; separately named TP16/32 padded experts."""
from collections import defaultdict
from contextvars import ContextVar
from src.kernel.predictors.k3_tp_math import PROGRAM,PADDED_PROGRAM,validate as validate_math
_PROOF=ContextVar('k3_tp_source_proof',default=None)
from src.kernel.hardware_spec import HWSpec


def applicable(pool):
 c=pool.parallel_plan.config
 return ((pool.hardware_id == "nvidia_h200" or HWSpec.source_contract_supported(
            "kimi_k3_portable_roofline", pool.hardware_id)) and pool.parallel_plan.num_layers==93 and not c.enable_expert_parallel
  and c.tensor_parallel_size in (2,4,8,16,32) and c.data_parallel_size==c.context_parallel_size==c.sequence_parallel_size==1
  and 1<=c.pipeline_parallel_size<=93)
def allowed_node(node,pool):
 p=_PROOF.get()
 return bool(p and p[0] is pool and p[1].get(node.node_id) is node)
def handles_collective(node,pool,graph):
 p=_PROOF.get()
 return bool(p and p[0] is pool and p[2] is graph and p[3].get(node.node_id) is node)
def expert_attributes(n,pool):
 a=n.attributes;tp=pool.parallel_plan.config.tensor_parallel_size
 if (a.get('source_component')!='vllm_0_30_kimi_k3_marlin_experts' or n.dtype!='bf16'
  or n.kind.value!='fused_moe' or n.variant!='full' or a.get('model_execution_version')!='vllm_0_30_0'
  or a.get('source_projection_only') is not True or a.get('symbolic_workspace_complete') is not False
  or a.get('expert_tensor_rank')!=a.get('source_tensor_parallel_rank')
  or a.get('tensor_parallel_size')!=tp or a.get('expert_tensor_parallel_size')!=tp
  or a.get('expert_output_is_reduced') is not False):raise ValueError('K3 TP expert source identity')
 attrs={k:a[k] for k in ('tokens_per_expert','local_tokens_per_expert','expert_parallel_size','expert_rank','owned_experts','expert_weight_tile_rows','activation_beta','activation_linear_beta','routed_scaling_factor')}
 attrs.update(mathematical_program=PADDED_PROGRAM if tp in (16,32) else PROGRAM,routed_weight_dtype='mxfp4',weight_group_size=32,swiglu_limit=None,
   tensor_parallel_size=tp,tensor_rank=a['source_tensor_parallel_rank'],logical_intermediate=a['logical_intermediate'])
 validate_math(n.shape,attrs);return attrs

def validate(graph,pool,nodes,memory_context):
    _PROOF.set(None)
    if not applicable(pool):raise ValueError("K3 TP source scope")
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
        return x
    groups=defaultdict(list);verified_collectives={}
    check(bool(nodes),'nonempty K3 expert graph')
    for n in nodes:expert_attributes(n,pool);groups[n.attributes['moe_invocation_id']].append(n)
    for members in groups.values():
        first=members[0];t=first.shape['tokens']
        layer=first.attributes['routing_provenance']['layer_index']
        check(type(layer) is int and 0<=layer<pool.parallel_plan.num_layers,'invalid layer identity')
        stage=pool.parallel_plan.layer_to_pipeline_stage[layer];tp=pool.parallel_plan.config.tensor_parallel_size
        expected=tuple(pool.physical_rank_for_logical_rank(stage*tp+lane) for lane in range(tp))
        participants=pool.physical_rank_group('tensor',pool.logical_rank_for_physical_rank(first.rank))
        check(tuple(participants)==expected,'expert group differs from pipeline layer placement')
        check(len(members)==len(participants) and {n.rank for n in members}==set(participants),'all expert ranks required')
        check(all(tuple(n.attributes['owned_experts'])==tuple(range(896)) for n in members),'full experts per TP lane')
        latent_reductions={};shared_reductions={};expert_outputs={};shared_outputs={}
        for n in members:
            a=n.attributes;p=ins(n);r=n.rank
            check(n.shape==first.shape and a['tokens_per_expert']==first.attributes['tokens_per_expert']
                  and a['routing_provenance']==first.attributes['routing_provenance'],'global routing geometry')
            check(a['expert_rank']==0 and a['source_tensor_parallel_rank']==participants.index(r) and len(p)==3,'rank and ports')
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
            tier=up.attributes.get('latent_tail_tier')
            if tier=='COLUMN_PARALLEL':
                check(up.kind.value=='linear' and up.variant=='gemm' and up.dtype=='bf16'
                    and up.attributes.get('model_execution_version')=='vllm_0_30_0' and up.attributes.get('source_projection_only') is True
                    and up.attributes.get('has_bias') is False
                    and dict(up.shape)=={'b':1,'m':t,'n':7168//tp,'k':3584} and up.attributes.get('source_operation')=='addmm_'
                    and up.attributes.get('alpha')==1.0 and up.attributes.get('beta')==1.0
                    and up.attributes.get('tensor_rank')==index and up.attributes.get('tensor_parallel_size')==tp,'column tail source')
                sr=one([x for x in succ[up.node_id] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'column shared reduction')
            else:
                check(tier is None and dict(up.shape)=={'b':1,'m':t,'n':7168,'k':3584},'full latent up')
                add=one([x for x in succ[up.node_id] if x.rank==r and x.kind.value=='elementwise'],'shared merge')
                check(add.variant=='add' and add.attributes['operation']=='add' and dict(add.shape)=={'elements':t*7168}
                      and add.attributes['reads']==2 and add.attributes['writes']==1 and add.dtype=='bf16' and len(outs(add))==1,'shared addition')
                other=one([v for v in ins(add) if v!=outs(up)[0]],'shared output port')
                sr=one([x for x in out[other] if x.kind.value=='collective' and x.attributes.get('operation')=='all_reduce'],'shared reduction')
            sv=ins(sr)[index]
            shared=byid.get(n.node_id.rsplit(':experts',1)[0]+':shared:down')
            check(shared is not None and shared.rank==r,'shared down')
            check(outs(shared)==(sv,), 'shared reduction uses original partial backing')
            if tier=='COLUMN_PARALLEL':
                data=data_ins(up);params=tuple(up.attributes.get('symbolic_parameter_inputs',()))
                check(len(data)==2 and data[0]==outs(norm)[0] and len(params)==1 and ins(up)==data+params,'column addmm input ordering')
                check(outs(up)==(data[1],sv) and up in succ[shared.node_id] and sr in succ[up.node_id],'actual shared→column-addmm→reduce chain')
                base=typed(shared,sv,(t,7168),'bf16',True)
                lane_in=typed(up,data[1],(t,7168//tp),'bf16')
                lane_out=typed(up,data[1],(t,7168//tp),'bf16',True)
                full_out=typed(up,sv,(t,7168),'bf16',True)
                offset=base.offset_bytes+index*(7168//tp)*2
                check(all(v.allocation_id==base.allocation_id for v in (lane_in,lane_out,full_out))
                      and lane_in.offset_bytes==lane_out.offset_bytes==offset and full_out.offset_bytes==base.offset_bytes
                      and tuple(lane_in.strides_bytes)==tuple(lane_out.strides_bytes)==tuple(full_out.strides_bytes)==(14336,2),
                      'column beta read and write must alias its own shared lane')
                weight=typed(up,params[0],(7168//tp,3584),'bf16')
                check(up.attributes.get('replicated_weight_shape')==(7168,3584)
                      and weight.allocation_id==up.node_id+':weight'
                      and weight.offset_bytes==index*(7168//tp)*3584*2
                      and tuple(weight.strides_bytes)==(7168,2),'column weight row slice')
            else:
                check(data_ins(up)==outs(norm) and add in succ[sr.node_id],'overlap full up and shared reduction merge')

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
            for red,width_out in ((lr,3584),(sr,7168)):
                check(len(ins(red))==len(outs(red))==len(participants),'reduction port count')
                for v in ins(red):typed(red,v,(t,width_out),'bf16')
                for v in outs(red):typed(red,v,(t,width_out),'bf16',True)
                verified_collectives[red.node_id]=red
            latent_reductions[lr.node_id]=lr;shared_reductions[sr.node_id]=sr;expert_outputs[r]=outs(n)[0];shared_outputs[r]=outs(shared)[0]
        check(len(latent_reductions)==len(shared_reductions)==1,'one complete group per reduction')
        check(ins(next(iter(latent_reductions.values())))==tuple(expert_outputs[r] for r in participants),'latent reduction inputs')
        check(ins(next(iter(shared_reductions.values())))==tuple(shared_outputs[r] for r in participants),'shared reduction inputs')

    # Source gates receive only current graph node identities after the complete MoE/PP proof.
    approved={}
    tp=pool.parallel_plan.config.tensor_parallel_size
    for n in graph.nodes:
        a=n.attributes;comp=a.get('source_component','');shape=n.shape
        if n.pool_name!=pool.name:continue
        if comp in ('vllm_0_30_kimi_k3_triton_fresh_prefill','vllm_0_30_kimi_k3_triton_decode'):
            check(n.kind.value=='vector' and shape.get('heads')==96//tp and shape.get('head_dim')==128,'local KDA heads')
        elif comp in ('vllm_0_30_kimi_k3_nope_fresh_prefill','vllm_0_30_kimi_k3_nope_decode'):
            check(n.kind.value=='vector' and shape.get('heads')==96//tp and shape.get('q_lora_rank')==1536 and shape.get('kv_lora_rank')==512,'local NoPE heads')
        elif comp=='vllm_0_30_symbolic_k3_prefill_attention':
            check(n.kind.value=='flash_attn' and shape.get('num_query_heads')==96//tp and shape.get('qk_head_dim')==192,'local NoPE fresh attention')
        elif comp=='symbolic_mla_decode_mathematical':
            check(n.kind.value=='flash_attn' and shape.get('num_query_heads')==96//tp and shape.get('qk_head_dim')==576,'local NoPE decode attention')
        elif comp=='vllm_0_30_kimi_k3_ordinary_mlp':
            check(n.kind.value=='activation' and shape.get('intermediate') in (33792//tp,6144//tp),'local shared/dense MLP')
        elif comp in ('vllm_0_30_kimi_k3_attention_residual','vllm_0_30_kimi_k3_native_marlin_moe',
            'vllm_0_30_kimi_k3_tp1_native_latent_output','vllm_0_30_kimi_k3_latent_tp_tail',
            'vllm_0_30_kimi_k3_fresh_prefill_logits','vllm_0_30_kimi_k3_plain_greedy','vllm_0_30_kimi_k3_marlin_experts'):
            pass
        else:continue
        approved[n.node_id]=n
    _PROOF.set((pool,approved,graph,verified_collectives))
