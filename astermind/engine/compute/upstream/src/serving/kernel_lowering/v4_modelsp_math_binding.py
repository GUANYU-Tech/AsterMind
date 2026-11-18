"""Source-only FP8 V4 routing binding for model sequence parallelism."""
from src.kernel.predictors.moe_ep_math import compact_counts_to_global
from src.kernel.predictors.v4_modelsp_math import PROGRAM,logical_work

def valid_tokens(node):
 a=node.attributes;t=node.shape['tokens'];valid=a.get('source_valid_tokens',t)
 if a.get('source_component')!='vllm_0_30_symbolic_v4_marlin' or a.get('model_execution_version')!='vllm_0_30_0' or a.get('model_source_profile')!='vllm_0_30_v4_sync_fp8_ds_mla' or node.dtype!='bf16':
  raise ValueError('V4 model-SP requires pinned Marlin source identity')
 if type(valid) is not int or not 0<valid<=t:raise ValueError('V4 model-SP valid rows outside physical extent')
 if valid!=t and (a.get('source_padding_routing')!='hash_topk_masked_ids_minus_one' or a['routing_provenance']['layer_index'] not in (0,1,2)):
  raise ValueError('V4 reduced routing extent belongs only to hash padding minus-one IDs')
 return valid

def expert_attributes(node):
 a=node.attributes;valid=valid_tokens(node)
 if a.get('source_expert_tensor_parallel_size')!=1 or a.get('source_marlin_padding_columns')!=0 or a.get('source_logical_intermediate')!=node.shape['intermediate'] or a.get('output_is_reduced') is not False or a.get('expert_placement_strategy')!='linear':
  raise ValueError('V4 model-SP retains full unpadded whole experts')
 out={k:a[k] for k in ('tokens_per_expert','expert_parallel_size','expert_rank','owned_experts','routed_scaling_factor','routed_weight_dtype','weight_group_size','swiglu_limit')}
 out.update(local_tokens_per_expert=compact_counts_to_global(node.shape,a),expert_weight_tile_rows=None,physical_tokens=node.shape['tokens'],mathematical_program=PROGRAM)
 logical_work({**node.shape,'tokens':valid},out)
 return out
