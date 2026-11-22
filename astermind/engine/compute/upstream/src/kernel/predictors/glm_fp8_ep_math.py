"""GLM0.30/FlashInfer0.6.18.post1 block128 local-expert useful work.
Pinned FlashInfer 8bc3b578: BlockScaleFC1/2 -> moeGemm -> scale_1x128.
No provider, native launch, padded GEMM or physical cache-traffic claim.
"""
PROGRAM = "glm_flashinfer_block128_deferred_ep_v1"

def logical_work(shape, a):
    t,h,i,e,k=(shape[n] for n in ("tokens","hidden","intermediate","experts","topk"))
    if any(type(x) is not int for x in (t,h,i,e,k)) or t<=0 or (h,i,e,k)!=(6144,2048,256,8):
        raise ValueError("GLM deferred FP8 requires exact H6144 I2048 E256 top8 source geometry")
    ep,rank=a["expert_parallel_size"],a["expert_rank"]
    if (a.get("mathematical_program")!=PROGRAM or type(ep) is not int or ep<2 or e%ep
        or type(rank) is not int or not 0<=rank<ep or a["routed_weight_dtype"]!="fp8"
        or a["weight_group_size"]!=0 or a.get("swiglu_limit") is not None
        or a["expert_weight_tile_rows"] is not None or a["routed_scaling_factor"]!=2.5):
        raise ValueError("GLM deferred FP8 requires uniform EP, block weights and externally applied router scale2.5")
    owned=tuple(range(rank*(e//ep),(rank+1)*(e//ep)))
    counts,local=tuple(a["tokens_per_expert"]),tuple(a["local_tokens_per_expert"])
    if (tuple(a["owned_experts"])!=owned or any(type(x) is not int for x in a["owned_experts"])
        or len(counts)!=e or len(local)!=e or any(type(x) is not int or not 0<=x<=t for x in (*counts,*local))
        or sum(counts)!=t*k or local!=tuple(x if n in owned else 0 for n,x in enumerate(counts))):
        raise ValueError("GLM deferred FP8 requires conserved topk counts and exact linear local expert ownership")
    rows=sum(local);active=sum(x>0 for x in local);stages=[]
    def add(name,payload,fp8=0,scalar=0,sfu=0):
        stages.append(dict(stage=name,payload_bytes=payload,logical_bytes=sum(payload.values()),mathematical_fp8_flops=fp8,
            demand=dict(compute_tensor=0,compute_vector=0,compute_reduction=0,compute_fma=scalar,compute_sfu=sfu,compute_alu=0)))
    # expandInputRows copies only valid local-expert assignments before internal quantization.
    add("permute_input",dict(bf16_read=2*rows*h,bf16_write=2*rows*h))
    def quant(name,width):
        n=rows*width;g=n//128
        add(name,dict(bf16_read=2*n,fp8_write=n,fp32_scale_write=4*g),scalar=n+2*g)
    quant("quantize_fc1_input",h)
    add("w13",dict(fp8_input_read=rows*h,fp8_weight_read=active*2*i*h,input_scale_read=4*rows*(h//128),
        weight_scale_read=4*active*(2*i//128)*(h//128),bf16_output_write=4*rows*i),fp8=4*rows*h*i)
    add("silu_mul",dict(bf16_read=4*rows*i,bf16_write=2*rows*i),scalar=4*rows*i,sfu=rows*i)
    quant("quantize_fc2_input",i)
    add("w2",dict(fp8_input_read=rows*i,fp8_weight_read=active*h*i,input_scale_read=4*rows*(i//128),
        weight_scale_read=4*active*(h//128)*(i//128),bf16_output_write=2*rows*h),fp8=2*rows*h*i)
    # Finalize skips nonlocal experts, sums in FP32, writes every output row including zero-local rows.
    add("weighted_finalize",dict(bf16_local_read=2*rows*h,route_weight_read=4*rows,expert_id_read=4*t*k,
        bf16_output_write=2*t*h),scalar=2*rows*h)
    return dict(stages=tuple(stages),source_stage_order=tuple(x["stage"] for x in stages),local_routed_rows=rows,
        active_local_experts=active,actual_provider_bound=False,complete_source_service=False,
        limitations=("FlashInfer0.6.18.post1_valid_local_rows_only", "useful_role_bytes_not_cache_or_physical_transactions",
          "expert_map_sort_padding_addressing_and_launch_services_unpriced", "blockscale_MMA_scale_application_and_conversions_unpriced",
          "quantizer_abs_max_shuffle_comparison_services_unpriced", "quantizer_reciprocals_counted_as_scalar_not_calibrated_division_issue",
          "FP32_activation_then_BF16_materialization_before_second_internal_quantizer", "router_scale2.5_already_in_weights_not_reapplied"))

def predict(k,hw):
    portable = hw.supports_source_contract("glm_fp8_weight_only_roofline")
    if ((k.hardware_id != "nvidia_h200" and not portable) or k.dtype != "bf16" or k.kind.value != "FUSED_MOE"
            or k.variant != "routed_experts" or k.launch is not None or k.execution_support != "unsupported"):
        raise ValueError("GLM deferred FP8 requires supported unbound BF16-output expert math")
    from . import fused_moe as fm
    from .v3_ep_math import stage_seconds
    from dataclasses import replace
    w=logical_work(k.shape,k.attributes)
    times=stage_seconds(w["stages"], hw, dequantize_to_bf16=portable)
    return replace(fm.prediction_from_ns(sum(times.values())*1e9,"roofline"),t_per_group=times,
        diagnostics=fm.PredictionDiagnostics(compute_time_s={},memory=(),memory_access_classes=(),memory_arbitration=(),pipeline=None,
            critical_path=w["source_stage_order"],support_level="partial",limitations=w["limitations"]))
