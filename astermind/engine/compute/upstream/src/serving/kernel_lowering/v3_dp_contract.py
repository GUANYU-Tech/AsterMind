"""V3 active DP2/4/8 four-input source contract."""
from collections import defaultdict
from contextvars import ContextVar
from src.kernel.hardware_spec import HWSpec
_PROOF=ContextVar('v3_active_dp_collective_proof',default=None)
def collective_supported(node,pool,graph):
 proof=_PROOF.get()
 return proof is not None and proof[0] is graph and proof[1] is pool and node.node_id in proof[2]
from src.kernel.predictors.v3_dp_math import PROGRAM,QUANT,logical_work
from src.kernel.predictors.v3_ep_math import DISPATCHED_EP_SIZES
PROFILE='vllm_0_27_1_deepseek_v3_sync_fp8'
def applicable(pool):
 c=pool.parallel_plan.config
 return ((pool.hardware_id == 'nvidia_h200' or
          HWSpec.source_contract_supported("deepseek_v3_expert_roofline", pool.hardware_id))
     and c.enable_expert_parallel
     and c.sequence_parallel_size == c.context_parallel_size == 1
     and c.tensor_parallel_size in (1, 2, 4, 8, 16)
     and c.data_parallel_size in (2, 4, 8)
     and c.expert_parallel_size == c.data_parallel_size * c.tensor_parallel_size
     and c.expert_parallel_size in DISPATCHED_EP_SIZES
     and (c.expert_parallel_size < 64 or c.pipeline_parallel_size == 1)
     and c.pipeline_parallel_size in (1, 2, 3, 4, 6, 8)
     and len(pool.ranks) == c.worker_count)
def marker(node):
 a=node.attributes
 return a.get('source_projection_only') is True and a.get('model_execution_version')=='vllm_0_27_1' and a.get('model_source_profile')==PROFILE
def source_node(node):
 from src.serving.workload.model_config import DEEPSEEK_V3_CONFIG_SHA256,DEEPSEEK_V3_L32_CONFIG_SHA256
 a=node.attributes;identities={DEEPSEEK_V3_CONFIG_SHA256:61,DEEPSEEK_V3_L32_CONFIG_SHA256:32}
 return marker(node) and a.get('source_model_profile')==PROFILE and a.get('source_model_config_sha256') in identities and a.get('source_target_layers')==identities[a['source_model_config_sha256']] and a.get('source_tensor_parallel_size') in (1,2,4,8,16) and a.get('source_model_execution_version')=='vllm_0_27_1'
def expert_attributes(node):
 a=node.attributes
 if not source_node(node) or node.kind.value!='fused_moe' or node.variant!='full' or node.dtype!='bf16' or a.get('source_component')!='symbolic_dsv3_routed_experts' or a.get('source_expert_tensor_parallel_size')!=1 or a.get('source_dispatched_input_dtype')!='fp8' or a.get('source_input_quantization')!='pre_dispatch_group128_fp32_scales' or a.get('output_is_reduced') is not False or a.get('checkpoint_weight_dtype')!='fp8' or a.get('expert_placement_strategy')!='linear' or a.get('launch_policy')!='analytical_launch':
  raise ValueError('V3 DP requires exact prequantized four-input FP8 source')
 ep=a['expert_parallel_size'];owned=tuple(a['owned_experts']);counts=tuple(a['tokens_per_expert']);local=tuple(a['local_tokens_per_expert'])
 if type(ep) is not int or ep not in DISPATCHED_EP_SIZES or len(owned)!=256//ep or len(local)!=256//ep or any(type(x) is not int for x in (*owned,*local)) or any(not 0<=j<256 for j in owned) or local!=tuple(counts[j] for j in owned):raise ValueError('V3 DP source ownership/counts')
 out={k:a[k] for k in ('tokens_per_expert','expert_parallel_size','expert_rank','owned_experts','routed_scaling_factor','routed_weight_dtype','weight_group_size')}
 out.update(local_tokens_per_expert=tuple(x if j in owned else 0 for j,x in enumerate(counts)),expert_weight_tile_rows=None,swiglu_limit=None,mathematical_program=PROGRAM)
 logical_work(node.shape,out);return out
def quant_attributes(node):
 a=node.attributes
 if not marker(node) or node.kind.value!='vector' or node.variant!='dsv3_expert_input_quant' or node.dtype!='bf16' or a.get('source_component')!='vllm_0_27_1_symbolic_expert_input_quant' or a.get('group_size')!=128 or a.get('output_dtype')!='fp8' or a.get('scale_dtype')!='fp32' or a.get('scale_layout')!='logical_row_major_unswizzled' or dict(node.shape)!={'rows':node.shape['rows'],'hidden':7168,'elements':node.shape['rows']*7168}:raise ValueError('V3 DP input quant source contract')
 return {'group_size':128,'source_program':QUANT}
def validate(graph,pool,nodes,memory_context):
 _PROOF.set(None);proven=set()
 if not applicable(pool):raise NotImplementedError('V3 active DP source requires legal H200 TP/PP with uniform EP=TP*DP up to 64, SP1 and CP1')
 if memory_context is None:raise ValueError('V3 DP requires source memory ports')
 c=pool.parallel_plan.config;dp=c.data_parallel_size;tp=c.tensor_parallel_size;pp=c.pipeline_parallel_size;ep=dp*tp;dimension='expert' if tp>1 else 'data';participants=tuple(pool.ranks);byid={n.node_id:n for n in graph.nodes};out=defaultdict(list);succ=defaultdict(list);pred=defaultdict(list);memory=memory_context._by_node
 model_sp = tp > 1 and pp == 1
 for n in graph.nodes:
  for v in n.attributes.get('symbolic_tensor_outputs',()):out[v].append(n)
 for e in graph.edges:
  if e.kind=='data':succ[e.source].append(byid[e.destination]);pred[e.destination].append(byid[e.source])
 def fail(c,msg):
  if c:raise ValueError('V3 DP caller: '+msg)
 def ins(n):return tuple(n.attributes.get('symbolic_tensor_inputs',()))
 def outs(n):return tuple(n.attributes.get('symbolic_tensor_outputs',()))
 def port(n,name,shape,dtype,output=False):
  views=memory[n.node_id].outputs if output else memory[n.node_id].inputs;v=[x for x in views if x.view_id==name]
  fail(len(v)!=1 or tuple(v[0].shape)!=tuple(shape) or v[0].dtype!=dtype,'typed port shape/dtype')
  from math import prod
  size={'bf16':2,'fp8':1,'fp32':4,'int32':4}[dtype]
  expected=tuple(size*prod(shape[j+1:]) for j in range(len(shape)))
  fail(any(dim>1 and actual!=want for dim,actual,want in zip(shape,v[0].strides_bytes,expected,strict=True)),'typed port must be contiguous row-major')
  return v[0]
 def unique(values,msg):
  fail(len(values)!=1,msg);return values[0]
 groups=defaultdict(list)
 for n in nodes:expert_attributes(n);groups[n.attributes['moe_invocation_id']].append(n)
 fail(not nodes,'no experts');layers=nodes[0].attributes['source_target_layers']
 fail(len(groups)!=layers-3 or {n.attributes['routing_provenance']['layer_index'] for n in nodes}!=set(range(3,layers)),'incomplete MoE layers')
 norms=[n for n in graph.nodes if n.node_id.endswith(':layer-0:input-norm')]
 first_ranks=tuple(pool.physical_rank_for_logical_rank(r*pp*tp+lane) for r in range(dp) for lane in range(tp))
 by_rank = {n.rank: n for n in norms}
 fail(len(norms)!=ep or set(by_rank)!=set(first_ranks) or any(n.shape['rows']<=0 for n in norms),'complete physical initial rows')
 logical_rows = []
 active_replicas = 0
 for replica in range(dp):
  members = [by_rank[r] for r in first_ranks[replica*tp:(replica+1)*tp]]
  rows = members[0].shape['rows']
  idle = [n.attributes.get('source_dummy_run') is True for n in members]
  fail(any(n.shape['rows'] != rows for n in members), 'initial full hidden rows must agree within TP')
  fail(any(idle) and (not all(idle) or rows != 1 or any(
   n.attributes.get('source_actual_request_count') != 0 for n in members)), 'idle TP group requires one dummy row and zero requests')
  active_replicas += int(not any(idle))
  logical_rows.append(rows)
 fail(active_replicas == 0, 'at least one replica must have a real request')
 initial_rows=tuple(-(-count//tp) for count in logical_rows for lane in range(tp))
 for invocation,members in groups.items():
  layer=members[0].attributes['routing_provenance']['layer_index'];stage=pool.parallel_plan.layer_to_pipeline_stage[layer]
  participants=tuple(pool.physical_rank_for_logical_rank((r*pp+stage)*tp+lane) for r in range(dp) for lane in range(tp))
  fail(len(members)!=ep or {n.rank for n in members}!=set(participants),'complete source-stage physical expert group')
  members=tuple(next(n for n in members if n.rank==r) for r in participants);first=members[0];t=first.shape['tokens'];h=7168
  fail(any(n.attributes['source_tensor_parallel_size']!=tp for n in members),'source TP must match physical deployment')
  dispatch=[]
  for idx,(width,dtype,size) in enumerate(((h,'fp8',1),(8,'fp32',4),(8,'int32',4),(56,'fp32',4))):
   d=unique([x for x in out[ins(first)[idx]] if x.kind.value=='collective'],'unique dispatch collective');a=d.attributes;rows=tuple(a['symbolic_variable_rows'])
   fail(rows!=initial_rows or any(type(x) is not int or x<=0 for x in rows) or sum(rows)!=t,'physical DP rows and global expert extent')
   fail(a.get('operation')!='all_gatherv' or a.get('reduction') is not None or a.get('collective_group')!=dimension or tuple(a['participants'])!=participants or tuple(a['symbolic_variable_row_shape'])!=(width,) or tuple(a['partition_bytes'])!=tuple(x*width*size for x in rows) or a.get('bytes')!=t*width*size or a.get('bytes_semantics')!='aggregate_input' or len(ins(d))!=ep or len(outs(d))!=ep,'dispatch bytes/group/operation')
   fail(tuple(x.node_id for x in pred[d.node_id])!=tuple(a['symbolic_collective_input_producers']) and set(x.node_id for x in pred[d.node_id])!=set(a['symbolic_collective_input_producers']),'dispatch data dependencies')
   for lane,n in enumerate(members):
    fail(len(ins(n))!=4 or len(set(ins(n)))!=4 or ins(n)[idx]!=outs(d)[lane] or n not in succ[d.node_id],'expert must consume own four dispatched ports')
    port(d,ins(d)[lane],(rows[lane],width),dtype);port(d,outs(d)[lane],(t,width),dtype,True);port(n,ins(n)[idx],(t,width),dtype)
   dispatch.append(d);proven.add(d.node_id)
  rows=initial_rows
  for lane,n in enumerate(members):
   a=n.attributes;local=rows[lane]
   fail(n.shape!=first.shape or a['tokens_per_expert']!=first.attributes['tokens_per_expert'] or a['routing_provenance']!=first.attributes['routing_provenance'] or a['expert_rank']!=lane or a['expert_parallel_size']!=ep,'rank ownership and common routing histogram')
   fail(len(outs(n))!=1,'expert output count');port(n,outs(n)[0],(t,h),'bf16',True)
   quant=byid[dispatch[0].attributes['symbolic_collective_input_producers'][lane]];quant_attributes(quant)
   fail(quant.node_id!=dispatch[3].attributes['symbolic_collective_input_producers'][lane] or quant.rank!=n.rank or outs(quant)!=(ins(dispatch[0])[lane],ins(dispatch[3])[lane]) or len(ins(quant))!=1,'same local input quant produces values and scales')
   port(quant,ins(quant)[0],(local,h),'bf16');port(quant,outs(quant)[0],(local,h),'fp8',True);port(quant,outs(quant)[1],(local,56),'fp32',True)
   if tp>1:
    prefix=n.node_id.rsplit(':routed',1)[0]+':sequence';logical=logical_rows[lane//tp];padded=local*tp;offset=(lane%tp)*local*h*2
    qv=port(quant,ins(quant)[0],(local,h),'bf16')
    norm=unique([x for x in graph.nodes if x.node_id==n.node_id.rsplit(':mlp:',1)[0]+':post-attn-norm'],'original pre-MoE normalization')
    norm_rows = local if model_sp else logical
    fail(norm.kind.value!='norm' or norm.rank!=n.rank or norm.dtype!='bf16'
        or dict(norm.shape)!={'rows':norm_rows,'hidden':h},'original normalized hidden rows')
    nv=port(norm,outs(norm)[0],(norm_rows,h),'bf16',True)
    if model_sp:
     group=participants[lane//tp*tp:(lane//tp+1)*tp]
     reduce=unique([x for x in pred[norm.node_id] if x.kind.value=='collective'
         and x.node_id.endswith(':sp-attention-output')], 'one TP attention reduce-scatter')
     ra=reduce.attributes
     fail(ra.get('operation')!='reduce_scatterv' or ra.get('reduction')!='sum'
         or ra.get('collective_group')!='tensor' or tuple(ra.get('participants',()))!=group
         or tuple(ra.get('partition_bytes',()))!=(local*h*2,)*tp
         or ra.get('bytes')!=padded*h*2 or ra.get('bytes_semantics')!='per_rank_input'
         or len(outs(reduce))!=tp or ins(norm)[0]!=outs(reduce)[lane%tp],
         'TP attention reduce-scatter must provide the local hidden slice')
     port(reduce,outs(reduce)[lane%tp],(local,h),'bf16',True)
     fail(norm not in pred[quant.node_id] or ins(quant)!=(outs(norm)[0],)
         or qv.allocation_id!=nv.allocation_id or qv.offset_bytes!=nv.offset_bytes,
         'model sequence shard must come directly from the local normalized rows')
     hidden_producer=norm
     proven.add(reduce.node_id)
    elif padded==logical:
     sh=unique([x for x in graph.nodes if x.node_id==prefix+':clone-copy'],'exact MoE shard clone')
     fail(sh.rank!=n.rank or sh.kind.value!='gather' or sh.dtype!='bf16' or dict(sh.shape)!={'elements':local*h,'rows':local,'hidden':h} or outs(sh)!=ins(quant) or sh not in pred[quant.node_id] or tuple(sh.attributes.get(k) for k in ('gather_factor','index_count','scattered','materialized_output'))!=(1,0,False,True),'MoE shard clone structure/dependency')
     iv=port(sh,ins(sh)[0],(local,h),'bf16');ov=port(sh,outs(sh)[0],(local,h),'bf16',True)
     fail(iv.offset_bytes!=offset or ov.offset_bytes!=0 or iv.allocation_id==ov.allocation_id,'actual TP lane slice and independent clone')
     fail(norm not in pred[sh.node_id] or iv.allocation_id!=nv.allocation_id or iv.offset_bytes!=nv.offset_bytes+offset,'clone must slice its own normalized hidden allocation')
     hidden_producer=sh
    else:
     cp=unique([x for x in graph.nodes if x.node_id==prefix+':pad-copy'],'exact MoE pad copy');z=unique([x for x in graph.nodes if x.node_id==prefix+':pad-zero'],'exact MoE pad zero')
     fail(cp.rank!=n.rank or z.rank!=n.rank or cp.kind.value!='gather' or z.kind.value!='elementwise' or tuple(z.attributes.get(k) for k in ('operation','reads','writes','source_constant'))!=('fill',0,1,0) or dict(cp.shape)!={'elements':logical*h,'rows':logical,'hidden':h} or dict(z.shape)!={'elements':(padded-logical)*h},'MoE explicit padding computation')
     fail(tuple(cp.attributes.get(k) for k in ('gather_factor','index_count','scattered','materialized_output'))!=(1,0,False,True) or cp not in pred[z.node_id] or z not in pred[quant.node_id],'actual padding materialization and quant dependencies')
     fail(len(ins(cp))!=1 or norm not in pred[cp.node_id],'padding copy must read original normalized hidden')
     pv=port(cp,ins(cp)[0],(logical,h),'bf16');fail(pv.allocation_id!=nv.allocation_id or pv.offset_bytes!=nv.offset_bytes,'padding copy source allocation')
     cv=port(cp,outs(cp)[0],(logical,h),'bf16',True);zv=port(z,outs(z)[0],(padded-logical,h),'bf16',True)
     fail(qv.offset_bytes!=offset or cv.offset_bytes!=0 or zv.offset_bytes!=logical*h*2 or len({qv.allocation_id,cv.allocation_id,zv.allocation_id})!=1,'padding backing allocation and valid/zero regions')
     hidden_producer=z
   route=byid[dispatch[1].attributes['symbolic_collective_input_producers'][lane]];ra=route.attributes
   fail(route.node_id!=dispatch[2].attributes['symbolic_collective_input_producers'][lane] or route.rank!=n.rank or not source_node(route) or outs(route)!=(ins(dispatch[1])[lane],ins(dispatch[2])[lane]) or route.shape!={**n.shape,'tokens':local} or len(ins(route))!=2,'route identity and local rows')
   expected={'source_component':'vllm_0_27_1_symbolic_grouped_routing','source_program':'vllm_0_27_1_bf16_sigmoid_bias_grouped_topk','mathematical_program':'bf16_sigmoid_bias_grouped_topk_flops_bytes_v1','normalize_topk_probability':True,'expert_group_count':8,'topk_group_count':4,'routed_scaling_factor':1.0}
   fail(any(ra.get(k)!=v for k,v in expected.items()),'grouped source routing semantics')
   gate=unique([x for x in out[ins(route)[0]] if x.kind.value=='router'],'local router gate')
   fail(gate.rank!=n.rank or dict(gate.shape)!={'b':1,'m':local,'n':256,'k':h} or gate.dtype!='bf16' or ins(gate)[0]!=ins(quant)[0] or route not in succ[gate.node_id],'routing and quantization share original BF16 hidden')
   if tp>1:
    fail(hidden_producer not in pred[gate.node_id],'routing gate requires actual hidden shard producers')
   port(route,ins(route)[1],(256,),'fp32');port(route,outs(route)[0],(local,8),'fp32',True);port(route,outs(route)[1],(local,8),'int32',True)
  combine=unique([x for x in succ[first.node_id] if x.kind.value=='collective'],'one expert combine');a=combine.attributes
  fail(a.get('operation')!='reduce_scatterv' or a.get('reduction')!='sum' or a.get('collective_group')!=dimension or tuple(a['participants'])!=participants or tuple(a['symbolic_variable_rows'])!=rows or tuple(a['symbolic_variable_row_shape'])!=(h,) or tuple(a['partition_bytes'])!=tuple(x*h*2 for x in rows) or a.get('bytes')!=t*h*2 or a.get('bytes_semantics')!='per_rank_input' or a.get('moe_invocation_id')!=invocation,'combine sum/bytes/rows')
  fail(ins(combine)!=tuple(outs(n)[0] for n in members) or tuple(a['symbolic_collective_input_producers'])!=tuple(n.node_id for n in members) or len(outs(combine))!=ep or set(x.node_id for x in pred[combine.node_id])!=set(n.node_id for n in members),'all physical expert contributions required')
  proven.add(combine.node_id)
  for lane,n in enumerate(members):
   local=rows[lane];port(combine,ins(combine)[lane],(t,h),'bf16');port(combine,outs(combine)[lane],(local,h),'bf16',True)
   copy=unique([x for x in succ[combine.node_id] if x.rank==n.rank and x.kind.value=='gather'],'rank-local finalize copy');ca=copy.attributes
   fail(copy.dtype!='bf16' or dict(copy.shape)!={'elements':local*h,'rows':local,'hidden':h} or ins(copy)!=(outs(combine)[lane],) or len(outs(copy))!=1 or ca.get('materialized_output') is not True or ca.get('gather_factor')!=1 or ca.get('index_count')!=0 or ca.get('scattered') is not False,'combine restores own local rows')
   cv=port(copy,ins(copy)[0],(local,h),'bf16');ov=port(copy,outs(copy)[0],(local,h),'bf16',True);fail(cv.allocation_id==ov.allocation_id,'finalize copy must materialize independently')
   scale=unique([x for x in succ[copy.node_id] if x.kind.value=='elementwise'],'unique external scaling')
   fail(scale.rank!=n.rank or ins(scale)!=outs(copy) or outs(scale)!=outs(copy) or scale.dtype!='bf16' or dict(scale.shape)!={'elements':local*h} or tuple(scale.attributes.get(k) for k in ('operation','reads','writes','scalar_value'))!=('unary',1,1,2.5),'exactly one post-combine factor2.5')
   add=unique([x for x in succ[scale.node_id] if x.kind.value=='elementwise'],'local shared merge');fail(add.rank!=n.rank or add.variant!='add' or add.dtype!='bf16' or dict(add.shape)!={'elements':local*h} or len(outs(add))!=1 or len(ins(add))!=2 or ins(add).count(outs(scale)[0])!=1 or tuple(add.attributes.get(k) for k in ('operation','reads','writes'))!=('add',2,1),'shared add local rows')
   other=next(v for v in ins(add) if v!=outs(scale)[0]);down=unique([x for x in out[other] if x.kind.value=='linear' and x.rank==n.rank],'shared down')
   fail(dict(down.shape)!={'b':1,'m':local,'n':h,'k':2048} or down.attributes.get('weight_dtype')!='fp8' or down.attributes.get('weight_group_size')!=128,'TP1 unsharded shared expert down')
   act=unique([x for v in ins(down) for x in out[v] if x.kind.value=='activation' and x.rank==n.rank],'shared silu')
   fail(act.variant!='silu' or dict(act.shape)!={'elements':local*2048,'tok':local,'intermediate':2048} or act.attributes.get('swiglu_limit') is not None,'shared silu geometry')
   up=unique([x for v in ins(act) for x in out[v] if x.kind.value=='linear' and x.rank==n.rank],'shared up');quant=byid[dispatch[0].attributes['symbolic_collective_input_producers'][lane]]
   fail(dict(up.shape)!={'b':1,'m':local,'n':4096,'k':h} or ins(up)!=ins(quant) or up.attributes.get('weight_dtype')!='fp8' or up.attributes.get('weight_group_size')!=128,'shared expert uses original local hidden')
   if model_sp:
    prefix=n.node_id.rsplit(':mlp:',1)[0]
    norm=byid[prefix+':post-attn-norm']
    fail(norm not in pred[up.node_id], 'shared expert requires the normalized model sequence shard')
    consumers=[y for y in succ[add.node_id] if y.rank==n.rank
        and y.node_id.endswith((':input-norm',':sp-final-pair',':final-norm'))]
    consumer=unique(consumers,'model sequence shard must feed the next layer or final merge')
    fail(outs(add)[0] not in ins(consumer),'next model stage must consume the local expert contribution')
   elif tp>1:
    suffix=':clone-copy' if local*tp==logical_rows[lane//tp] else ':pad-zero'
    final_shard=byid[n.node_id.rsplit(':routed',1)[0]+':sequence'+suffix]
    fail(final_shard not in pred[up.node_id],'shared expert requires actual finalized hidden shard')
  if tp>1 and not model_sp:
   for replica in range(dp):
    group=participants[replica*tp:(replica+1)*tp];local=initial_rows[replica*tp]
    adds=[unique([x for x in graph.nodes if x.rank==rank and x.node_id==members[replica*tp+lane].node_id.rsplit(':routed',1)[0]+':shared-add'],'shared add before restore') for lane,rank in enumerate(group)]
    restore=unique([x for x in succ[adds[0].node_id] if x.kind.value=='collective'],'one TP sequence restore');a=restore.attributes
    fail(a.get('operation')!='all_gather' or a.get('reduction') is not None or a.get('collective_group') is not None or tuple(a['participants'])!=group or a.get('bytes')!=local*h*2 or a.get('bytes_semantics')!='per_rank_input' or a.get('symbolic_collective_dim')!=0 or a.get('symbolic_collective_storage')!='fresh' or ins(restore)!=tuple(outs(x)[0] for x in adds) or tuple(a['symbolic_collective_input_producers'])!=tuple(x.node_id for x in adds) or set(x.node_id for x in pred[restore.node_id])!=set(x.node_id for x in adds) or len(outs(restore))!=tp,'MoE sequence restore rows, ordered lanes and dependencies')
    for lane,x in enumerate(adds):
     port(restore,ins(restore)[lane],(local,h),'bf16');rv=port(restore,outs(restore)[lane],(local*tp,h),'bf16',True)
     # Intra-stage successor norms must consume the cropped full logical rows;
     # PP boundaries retain the original complete typed sliced-PP validator.
     consumers=[y for y in succ[restore.node_id] if y.rank==x.rank and y.kind.value=='norm']
     if (layer+1==layers or pool.parallel_plan.layer_to_pipeline_stage[layer+1]==stage):
      fail(len(consumers)!=1,'one restored hidden consumer norm')
      views=[v for v in memory[consumers[0].node_id].inputs if v.allocation_id==rv.allocation_id]
      fail(len(views)!=1 or views[0].shape!=(logical_rows[replica],h) or views[0].offset_bytes!=0,'restored hidden must crop padding before next norm')
 _PROOF.set((graph,pool,frozenset(proven)))


