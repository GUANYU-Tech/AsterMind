"""Useful-work contracts for source-specific masked Marlin EP calls.

No claim of a native launch, physical cache traffic or hardware calibration.
V4 masks activation; K3 SiTU does not. Keep these two source paths distinct.
"""

def marlin_ep_work(shape, attrs, *, model):
    t,e,q,h,i=(shape[n] for n in ('tokens','experts','topk','hidden','intermediate'))
    if any(type(v) is not int or v <= 0 for v in (t,e,q,h,i)):
        raise ValueError('EP dimensions must be positive integers')
    geometry=(e,q,h,i)
    if attrs['routed_weight_dtype']!='mxfp4' or attrs['weight_group_size']!=32 or attrs['expert_weight_tile_rows'] is not None:
        raise ValueError('Marlin EP requires MXFP4 group32 and unpadded expert rows')
    if model=='v4':
        if geometry not in ((256,6,4096,2048),(384,6,7168,3072)) or attrs['swiglu_limit']!=10.0:
            raise ValueError('V4 EP requires exact unpadded Flash/Pro geometry and clamp10')
        if attrs['routed_scaling_factor']!=(1.5 if e==256 else 2.5):
            raise ValueError('V4 scaling must be applied by its declared router exactly once')
    elif model=='k3':
        if geometry!=(896,16,3584,3072) or (attrs['activation_beta'],attrs['activation_linear_beta'])!=(4.0,25.0):
            raise ValueError('K3 EP requires exact latent geometry and SiTU beta4/linear_beta25')
        if attrs['routed_scaling_factor']!=1.0:
            raise ValueError('K3 EP expert weights require unit internal scaling')
    else:
        raise ValueError('Unknown EP source model')
    ep,rank=attrs['expert_parallel_size'],attrs['expert_rank']
    if type(ep) is not int or ep<2 or e%ep or type(rank) is not int or not 0<=rank<ep:
        raise ValueError('Uniform EP requires integral expert ownership')
    owned=tuple(attrs['owned_experts'])
    expected=tuple(range(rank*(e//ep),(rank+1)*(e//ep)))
    global_counts=tuple(attrs['tokens_per_expert'])
    local=tuple(attrs['local_tokens_per_expert'])
    if owned!=expected or any(type(x) is not int for x in owned):
        raise ValueError('Owned experts disagree with linear EP rank')
    if len(global_counts)!=e or len(local)!=e or any(type(x) is not int or x<0 for x in (*global_counts,*local)):
        raise ValueError('EP histograms require nonnegative global-index integer counts')
    if sum(global_counts)!=t*q or any(x>t for x in global_counts):
        raise ValueError('Global routing must conserve tokens*topk with distinct experts per token')
    if local!=tuple(global_counts[j] if j in owned else 0 for j in range(e)):
        raise ValueError('Local routing disagrees with global counts and ownership')
    r=t*q;l=sum(local);a=sum(x>0 for x in local)
    z=l*i if model=='v4' else r*i
    def stage(name,read,write,**demand):
        full={role:0 for role in ('compute_tensor','compute_vector','compute_reduction','compute_fma','compute_sfu','compute_alu')}
        full.update(demand)
        return dict(stage=name,read_bytes=read,write_bytes=write,logical_bytes=read+write,demand=full)
    # Metadata is a logical scan, not an exact issued or DRAM access count.
    metadata=8*r
    w13=a*h*i;w2=a*h*i//2
    scale13=a*2*i*(h//32);scale2=a*h*(i//32)
    stages=[stage('route_metadata',metadata,0),
        stage('w13',2*l*h+w13+scale13,4*l*i,compute_tensor=4*l*h*i,
              compute_fma=4*a*h*i if model=='k3' else 0)]
    if model=='v4':
        stages.append(stage('masked_clamped_silu',4*z+metadata,2*z,
                            compute_fma=3*z,compute_sfu=z))
    else:
        stages.append(stage('full_buffer_situ',4*z,2*z,compute_fma=8*z,compute_sfu=3*z))
    stages.extend((stage('weighted_w2',2*l*i+w2+scale2+4*l,2*l*h,
                         compute_tensor=2*l*h*i,compute_fma=l*h+(2*a*h*i if model=='k3' else 0)),
                   stage('masked_sum',2*l*h+metadata,2*t*h,compute_fma=l*h)))
    return dict(model=model,global_routes=r,local_routes=l,active_local_experts=a,
                activation_elements=z,output_bytes=2*t*h,stages=tuple(stages),
                activation_unpriced_primitives=({'fp32_minmax':3*z,'sign_negate':z,'bf16_to_fp32':2*z,'fp32_to_bf16':z} if model=='v4' else {}),
                limitations=('useful_math_serial_stage_estimate_not_native_launch',
                    'logical_unique_weights_not_physical_replay',
                    'metadata_unique_scan_not_instruction_or_cache_traffic',
                    'alignment_padding_scratch_and_launch_service_unbound',
                    'v4_minmax_negation_conversions_dequant_instruction_service_unpriced' if model=='v4' else 'k3_dequant_retains_existing_unique_weight_math_policy'))


def compact_counts_to_global(shape, attrs):
    """Normalize only after proving the compact source contract; never pad blindly."""
    e=shape['experts'];owned=tuple(attrs['owned_experts'])
    counts=tuple(attrs['tokens_per_expert']);local=tuple(attrs['local_tokens_per_expert'])
    if len(counts)!=e or len(local)!=len(owned) or len(set(owned))!=len(owned):
        raise ValueError('Malformed compact expert histogram')
    if any(type(j) is not int or not 0<=j<e for j in owned):
        raise ValueError('Invalid compact expert identity')
    if any(type(v) is not int or v<0 for v in (*counts,*local)) or local!=tuple(counts[j] for j in owned):
        raise ValueError('Compact histogram disagrees with declared source ownership')
    normalized=[0]*e
    for j,value in zip(owned,local):normalized[j]=value
    return tuple(normalized)
