"""V3 prequantized FP8 + FP32-scales distributed experts; no input quantization."""
from src.kernel.hardware_spec import HWSpec
from .v3_ep_math import DISPATCHED_EP_SIZES, logical_work as replicated_work, stage_seconds
PROGRAM='v3_fp8_dispatched_ep_fused_quant_v1'
QUANT='vllm_0_27_1_group_fp8_row_major'
def logical_work(shape,attrs):
    if attrs.get('mathematical_program')!=PROGRAM or attrs.get('expert_parallel_size') not in DISPATCHED_EP_SIZES:
        raise ValueError('V3 dispatched experts require named EP2/4/8/16/32/64 FP8 four-input program')
    work=replicated_work(shape,attrs)
    stages=tuple(s for s in work['stages'] if s['stage']!='quantize_input')
    assert len(stages)==4
    return {**work,'stages':stages,'source_stage_order':tuple(s['stage'] for s in stages),
        'input_contract':{'activation':'fp8','scales':'fp32','group_size':128,'input_quantization':'external_pre_dispatch'},
        'limitations':tuple(x for x in work['limitations'] if not x.startswith('input_quantizer_'))}
def kernel_work(k):
    if (k.hardware_id != "nvidia_h200" and not HWSpec.source_contract_supported(
            "deepseek_v3_expert_roofline", k.hardware_id)
        or k.kind.value!='FUSED_MOE' or k.variant!='routed_experts' or k.dtype!='bf16'
        or k.launch is not None or k.execution_support!='unsupported'):
        raise ValueError('V3 DP math requires a supported unbound FP8-input/BF16-output expert source')
    return logical_work(k.shape,k.attributes)
def predict(k,hw):
    from . import fused_moe as fm
    from dataclasses import replace
    w=kernel_work(k)
    times=stage_seconds(w["stages"], hw, dequantize_to_bf16=
        hw.supports_source_contract("deepseek_v3_expert_roofline"))
    return replace(fm.prediction_from_ns(sum(times.values())*1e9,'roofline'),t_per_group=times,
        diagnostics=fm.PredictionDiagnostics(compute_time_s={},memory=(),memory_access_classes=(),memory_arbitration=(),pipeline=None,critical_path=w['source_stage_order'],support_level='partial',limitations=w['limitations']))
def quant_work(k):
    if (k.hardware_id != "nvidia_h200" and not HWSpec.source_contract_supported(
            "deepseek_v3_expert_roofline", k.hardware_id)
        or k.kind.value!='VECTOR' or k.variant!='fp8_group_quant' or k.dtype!='bf16'
        or k.launch is not None or k.execution_support!='unsupported'
        or k.attributes.get('source_program')!=QUANT or k.attributes.get('group_size')!=128):
        raise ValueError('V3 input quant requires a supported unbound BF16 row-major group128 source')
    e,rows,width=(k.shape[x] for x in ('element_count','rows','columns'));g=e//128
    if width!=7168 or e!=rows*width or rows<=0:raise ValueError('V3 input quant shape mismatch')
    demand={x:0 for x in ('compute_tensor','compute_vector','compute_reduction','compute_fma','compute_sfu','compute_alu')};demand['compute_fma']=e+g
    return {'logical_bytes':3*e+4*g,'source_alternatives':tuple({'backend':'unbound_group128','scale_ue8m0':rounded,'demand':{**demand,'compute_sfu':2*g if rounded else 0}} for rounded in (False,True)), 'selected_scale_rounding':None,
        'input_read_bytes':2*e,'fp8_write_bytes':e,'fp32_scale_write_bytes':4*g,'scale_shape':(rows,width//128),
        'scale_layout':'logical_row_major_unswizzled','actual_provider_bound':False,'numerical_equivalence_asserted':False,
        'arithmetic_scope':'useful mathematical divisions; comparisons, conversion and lane replication unpriced'}

