"""DeepSeek-V3 replicated expert work, source vLLM 0.27.1 Triton branch.

Not a native launch model. Input quantization is internal; prequantized DP
ports belong to a different, deliberately unsupported contract.
"""
from math import isfinite
from src.kernel.hardware_spec import HWSpec
LEGACY_PROGRAM='v3_fp8_ep8_fused_quant_v1'
PROGRAM='v3_fp8_replicated_ep_fused_quant_v1'
DISPATCHED_EP_SIZES = (2, 4, 8, 16, 32, 64)

def logical_work(shape, attrs):
    t,h,i,e,k=(shape[n] for n in ('tokens','hidden','intermediate','experts','topk'))
    if any(type(x) is not int for x in (t,h,i,e,k)) or t<=0 or (h,i,e,k)!=(7168,2048,256,8):
        raise ValueError('V3 EP8 requires positive rows and H7168 I2048 E256 top8')
    allowed_ep=DISPATCHED_EP_SIZES if attrs.get('mathematical_program')=='v3_fp8_dispatched_ep_fused_quant_v1' else (2,4,8,16)
    if (attrs['expert_parallel_size'] not in allowed_ep or type(attrs['expert_parallel_size']) is not int
        or attrs['routed_weight_dtype']!='fp8' or attrs['weight_group_size']!=0
        or attrs['routed_scaling_factor']!=1.0 or attrs.get('swiglu_limit') is not None
        or attrs.get('expert_weight_tile_rows') is not None):
        raise ValueError('V3 EP8 requires unbound BF16/FP8 source, outer scaling and no clamp')
    ep=attrs['expert_parallel_size']; per_expert=256//ep
    if attrs.get('mathematical_program')==LEGACY_PROGRAM and ep!=8:raise ValueError('legacy EP8 program retains EP8 scope')
    rank=attrs['expert_rank'];counts=tuple(attrs['tokens_per_expert']);local=tuple(attrs['local_tokens_per_expert'])
    if type(rank) is not int or not 0<=rank<ep:raise ValueError('V3 expert rank outside EP8')
    owned=tuple(range(per_expert*rank,per_expert*(rank+1)))
    if (any(type(x) is not int for x in attrs['owned_experts']) or any(type(x) is not int for x in local) or tuple(attrs['owned_experts'])!=owned or len(counts)!=256 or len(local)!=256
        or any(type(x) is not int or not 0<=x<=t for x in counts)
        or sum(counts)!=t*8 or local!=tuple(x if n in owned else 0 for n,x in enumerate(counts))):
        raise ValueError('V3 EP8 requires conserved distinct routing and linear masked ownership')
    r=sum(local);active=sum(x>0 for x in local);rows=t*8;elements=rows*i;groups=elements//128
    stages=[]
    def add(name,payload,*,fp8=0,scalar=0,sfu=0):
        stages.append(dict(stage=name,payload_bytes=payload,logical_bytes=sum(payload.values()),
            mathematical_fp8_flops=fp8,demand=dict(compute_tensor=0,compute_vector=0,compute_reduction=0,
                compute_fma=scalar,compute_sfu=sfu,compute_alu=0)))
    add('quantize_input',dict(bf16_input_read=2*t*h,fp8_input_write=t*h,input_scale_write=4*t*(h//128)),scalar=t*h+t*(h//128))
    add('w13',dict(fp8_input_read=r*h,fp8_weight_read=active*2*i*h,input_scale_read=4*r*(h//128),
        weight_scale_read=4*active*(2*i//128)*(h//128),local_bf16_output_write=4*r*i,
        nonlocal_zero_write=4*(rows-r)*i),fp8=4*r*h*i)
    add('fused_silu_quant',dict(gate_up_bf16_read=4*elements,activated_fp8_write=elements,
        activation_scale_write=4*groups),scalar=5*elements+groups,sfu=elements)
    add('weighted_w2',dict(fp8_input_read=r*i,fp8_weight_read=active*h*i,
        input_scale_read=4*r*(i//128),weight_scale_read=4*active*(h//128)*(i//128),
        route_weight_read=4*r,local_bf16_output_write=2*r*h,nonlocal_zero_write=2*(rows-r)*h),fp8=2*r*h*i,scalar=r*h)
    add('sum',dict(full_bf16_input_read=2*rows*h,bf16_output_write=2*t*h),scalar=t*k*h)
    return dict(stages=tuple(stages),source_stage_order=tuple(s['stage'] for s in stages),
        local_routed_rows=r,global_routed_rows=rows,active_local_experts=active,
        actual_provider_bound=False,complete_source_service=False,
        limitations=('useful_role_bytes_not_issued_DRAM_or_padded_MMA','unbound_Triton_provider_and_launch',
            'routing_alignment_metadata_and_scale_application_services_unpriced',
            'fused_FP32_SiLU_result_quantized_without_BF16_intermediate',
            'source_comparisons_conversions_shared_reduction_and_division_issue_unpriced',
            'input_quantizer_ideal_unrounded_group128_scope_not_numerical_equivalence'))

def kernel_work(kernel):
    admitted = (kernel.hardware_id == "nvidia_h200" or (
        kernel.attributes.get("mathematical_program") == PROGRAM
        and HWSpec.source_contract_supported("deepseek_v3_expert_roofline", kernel.hardware_id)))
    if (not admitted or kernel.kind.value!='FUSED_MOE' or kernel.variant!='routed_experts'
        or kernel.dtype!='bf16' or kernel.launch is not None or kernel.execution_support!='unsupported'
        or kernel.attributes.get('mathematical_program') not in (PROGRAM,LEGACY_PROGRAM)):
        raise ValueError('V3 replicated experts require a supported unbound source KernelSpec')
    return logical_work(kernel.shape,kernel.attributes)

def stage_seconds(stages, hardware, *, dequantize_to_bf16=False):
    from .fused_moe import roofline_ns

    if not dequantize_to_bf16:
        fp8 = float(hardware.resource_by_role("compute_tensor_fp8").capacity)
        return {s["stage"]: max(s["mathematical_fp8_flops"] / fp8,
            roofline_ns(s["demand"], hardware, s["logical_bytes"]) * 1e-9) for s in stages}
    tensor = hardware.resource_by_role("compute_tensor").capacity
    times = {}
    for stage in stages:
        payload = stage["payload_bytes"]
        elements = payload.get("fp8_input_read", 0) + payload.get("fp8_weight_read", 0)
        demand = dict(stage["demand"])
        # Expanding FP8 inputs and weights inside the tile preserves the
        # dispatched bytes. Charge scale and conversion on this device.
        demand["compute_fma"] += elements
        demand["compute_alu"] += 2 * elements
        times[stage["stage"]] = max(stage["mathematical_fp8_flops"] / tensor,
            roofline_ns(demand, hardware, stage["logical_bytes"]) * 1e-9)
    return times

def predict(kernel,hardware):
    # Imports use the shared predictor namespace to preserve existing conversion conventions.
    from src.kernel.predictors import fused_moe as fm
    from dataclasses import replace
    work=kernel_work(kernel)
    times=stage_seconds(work["stages"], hardware, dequantize_to_bf16=
        hardware.supports_source_contract("deepseek_v3_expert_roofline"))
    result=fm.prediction_from_ns(sum(times.values())*1e9,'roofline')
    return replace(result,t_per_group=times,diagnostics=fm.PredictionDiagnostics(compute_time_s={},memory=(),
        memory_access_classes=(),memory_arbitration=(),pipeline=None,critical_path=work['source_stage_order'],
        support_level='partial',limitations=work['limitations']))



