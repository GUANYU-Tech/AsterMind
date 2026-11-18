"""vLLM0.30 masked CUDA hash top6: source work, not native issue or calibration."""
PROGRAM='v4_modelsp_hash_top6_masked_v1'
def logical_work(shape,attrs):
 r,e,k=(shape[x] for x in ('rows','experts','topk'));v=attrs['valid_rows']
 if any(type(x) is not int for x in (r,e,k,v)) or r<=0 or e not in (256,384) or k!=6 or not 0<=v<=r:
  raise ValueError('V4 masked hash requires physical positive rows and valid prefix including zero')
 if attrs.get('mathematical_program')!=PROGRAM or attrs.get('routing_source') is not None or attrs.get('expert_id_selection')!='hash_table' or attrs['routed_scaling_factor']!=(1.5 if e==256 else 2.5):
  raise ValueError('V4 masked hash requires the declared pure-text source and model scale')
 # CUDA source: only non-padding rows fetch six logits/table entries and
 # evaluate stable softplus. Every live physical row executes warp sum and
 # stores six weights/IDs; padded values are 0/-1. Input IDs/mask are read
 # before that conditional. Keep original source FLOP/resource convention.
 groups={'softplus_additions':2*v*k,'output_scale_multiply':r*k,'output_divide':r*k,'full_warp_sum_lane_additions':r*32*5}
 demand={x:0 for x in ('compute_tensor','compute_vector','compute_fma','compute_sfu','compute_alu','compute_reduction')}
 demand.update(compute_fma=sum(groups.values()),compute_sfu=3*v*k)
 reads={'input_ids':4*r,'padding_mask':r,'table_entries':4*v*k,'selected_logits':4*v*k}
 writes={'weights':4*r*k,'ids':4*r*k}
 return dict(physical_rows=r,valid_rows=v,arithmetic_groups=groups,demand=demand,read_bytes=reads,write_bytes=writes,
             logical_bytes=sum(reads.values())+sum(writes.values()),
             limitations=('source_expression_work_not_native_instruction_count','typed_mask_predicate_and_address_service_unpriced',
                          'selected_role_payload_not_observed_dram_or_cache_traffic','mask_construction_and_launch_service_unbound','no_hardware_calibration'))
