"""Pinned vLLM0.30 Pro EP-off TP16: logical192, physically padded256."""
PROGRAM='v4_pro_tp16_padded_marlin_w4a16_v1'
def validate(node,pool):
 a=node.attributes;c=pool.parallel_plan.config
 if not (pool.hardware_id=='nvidia_h200' and c.tensor_parallel_size==16 and not c.enable_expert_parallel and c.expert_parallel_size==1 and c.data_parallel_size==c.context_parallel_size==c.sequence_parallel_size==1 and c.pipeline_parallel_size in (1,2,3,4) and pool.parallel_plan.num_layers==61):
  raise NotImplementedError('Padded Pro source requires H200 TP16 DP1 EPoff PP1/2/3/4 CP/SP1')
 if node.dtype!='bf16' or a.get('source_component')!='vllm_0_30_symbolic_v4_marlin' or a.get('model_execution_version')!='vllm_0_30_0' or a.get('model_source_profile')!='vllm_0_30_v4_sync_fp8_ds_mla':
  raise ValueError('Padded Pro requires its pinned BF16 source identity')
 if tuple(node.shape[k] for k in ('experts','topk','hidden','intermediate'))!=(384,6,7168,256) or any(type(a.get(k)) is not int or a[k]!=v for k,v in (('source_expert_tensor_parallel_size',16),('source_logical_intermediate',192),('source_marlin_padding_columns',64),('expert_parallel_size',1),('expert_rank',0))):
  raise ValueError('Padded Pro logical192/physical256/TP16 geometry mismatch')
 if a.get('routed_weight_dtype')!='mxfp4' or a.get('weight_group_size')!=32 or a.get('swiglu_limit')!=10.0 or a.get('routed_scaling_factor')!=2.5 or a.get('output_is_reduced') is not False or a.get('expert_placement_strategy')!='linear':
  raise ValueError('Padded Pro quantization/clamp/router-scale contract mismatch')
 if tuple(a['local_tokens_per_expert'])!=tuple(a['tokens_per_expert']) or tuple(a['owned_experts'])!=tuple(range(384)):
  raise ValueError('Padded Pro EPoff must retain full identical expert ownership/histogram')
 if 'source_valid_tokens' in a and (type(a['source_valid_tokens']) is not int or a['source_valid_tokens']!=node.shape['tokens']):
  raise NotImplementedError('Padded Pro token masks are not bound')
 return True
