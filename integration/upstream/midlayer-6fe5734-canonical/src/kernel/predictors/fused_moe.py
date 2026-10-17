"""Complete-spec fused-MoE grouped-projection prediction."""

import dataclasses
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
)
from src.kernel.lru_cache import (
    simulate_cann_grouped_moe_operands,
    simulate_moe_tile_program,
)
from src.kernel.launch_config.fused_moe_tiling import (
    CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES,
    cann_grouped_matmul_k_program,
)
from src.kernel.model_contract import (
    AnalyticalPhase,
    AnalyticalPhaseLedger,
    CannGroupedMatmulEnvelopeDemand,
    ComputeDemand,
    ComputeExactRequestDemand,
    GpuGroupedGemmPipelineEnvelopeDemand,
    MemoryAccessClassDemand,
    MemoryAccessClassSignature,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemorySourceDemand,
    METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS,
    MetaxGroupedOperandPipelineDemand,
    PhaseComputeDemand,
    PhaseMemoryDemand,
    PipelineStageDemand,
    PipelineTaskDemand,
    PipelineTaskSequenceDemand,
    SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
)
from src.kernel.predictors.common import (
    _with_graph_regime,
    PredictionInput,
    attach_structured_contract,
    bind_memory_access_class_template,
    build_forward_demand,
    gpu_memory_source,
    gpu_partial_limitations,
    launch_wave_geometry,
    npu_launch_floor_contract,
    prediction_from_ns,
    roofline_ns,
    validated_l2_miss_rate,
)


_GROUPED_OPERAND_SERVICE = "grouped_fused_moe_operand_read"
_GROUPED_OPERAND_PIPELINE_SERVICE = (
    "grouped_fused_moe_operand_pipeline")
_GROUPED_WEIGHT_OVERLAPPED_DRAM_MISS_SERVICE = (
    "grouped_fused_moe_weight_overlapped_dram_miss_pipeline")
_NVIDIA_BM64_GSM8_FIVE_THREE_W1_SERVICE = (
    "nvidia_bm64_gsm8_5_3_w1_operand_pipeline")
_NVIDIA_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE = (
    "nvidia_eight_expert_68_active_72_max_fused_moe_w1_operand_pipeline")
_NVIDIA_EIGHT_EXPERT_132_ACTIVE_136_MAX_W1_SERVICE = (
    "nvidia_eight_expert_132_active_136_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E512_DEEP_W2_SERVICE = (
    "nvidia_e512_deep_fused_moe_w2_operand_pipeline")
_NVIDIA_E128_DEEP_W2_SERVICE = (
    "nvidia_e128_deep_fused_moe_w2_operand_pipeline")
_NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_SERVICE = (
    "nvidia_e128_h4096_4156_active_4222_max_"
    "fused_moe_w1_operand_pipeline")
_NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_M_TILE_CENSUS = (
    33, 32, 32, 32, 32, 32, 32, 31, 33, 31, 32, 32, 33, 31, 32, 33,
    31, 32, 32, 31, 33, 32, 33, 31, 33, 33, 33, 33, 31, 32, 31, 33,
    33, 33, 32, 33, 32, 33, 32, 32, 31, 33, 33, 33, 33, 32, 32, 32,
    32, 33, 33, 32, 32, 33, 33, 31, 32, 33, 32, 33, 33, 34, 32, 33,
    34, 34, 33, 33, 32, 32, 33, 32, 32, 34, 33, 33, 33, 32, 32, 33,
    32, 33, 34, 32, 33, 34, 32, 33, 34, 33, 33, 32, 32, 33, 32, 31,
    32, 33, 32, 33, 32, 32, 33, 33, 31, 32, 33, 33, 34, 32, 33, 32,
    31, 33, 33, 33, 32, 33, 33, 33, 33, 32, 33, 32, 34, 32, 32, 32)
_NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT = {
    "w1": (
        "nvidia_e128_h2048_8250_active_8318_max_"
        "fused_moe_w1_operand_pipeline"),
    "w2": (
        "nvidia_e128_h2048_8250_active_8318_max_"
        "fused_moe_w2_operand_pipeline"),
}
_NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_M_TILE_CENSUS = (
    64, 66, 65, 64, 63, 65, 66, 64, 64, 65, 64, 65, 63, 66, 64, 65,
    64, 64, 64, 63, 66, 66, 65, 65, 66, 62, 64, 64, 65, 65, 64, 63,
    64, 65, 65, 64, 66, 64, 64, 64, 65, 65, 63, 65, 65, 66, 63, 64,
    65, 64, 64, 65, 64, 64, 64, 64, 65, 64, 65, 66, 65, 65, 65, 65,
    63, 65, 66, 64, 66, 64, 64, 64, 64, 65, 63, 64, 66, 64, 63, 65,
    63, 64, 64, 64, 65, 66, 65, 64, 65, 65, 66, 64, 65, 65, 64, 66,
    63, 66, 66, 65, 63, 65, 63, 63, 64, 64, 63, 65, 63, 65, 64, 65,
    64, 64, 64, 64, 64, 63, 64, 64, 64, 65, 64, 65, 65, 64, 65, 66,
)
_NVIDIA_E8_H6144_DEEP_W2_SERVICE = (
    "nvidia_e8_h6144_deep_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H6144_260_ACTIVE_264_MAX_W2_SERVICE = (
    "nvidia_e8_h6144_260_active_264_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H6144_DEEP_W1_SERVICE = (
    "nvidia_e8_h6144_deep_fused_moe_w1_operand_pipeline")
_NVIDIA_E8_H6144_DEEP_W1_M_TILE_CENSUS = (
    128, 129, 127, 128, 127, 130, 129, 130)
_NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W1_SERVICE = (
    "nvidia_e8_h6144_8_active_9_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W2_SERVICE = (
    "nvidia_e8_h6144_8_active_9_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H4096_515_ACTIVE_520_MAX_W2_SERVICE = (
    "nvidia_e8_h4096_515_active_520_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H4096_1027_ACTIVE_1032_MAX_W2_SERVICE = (
    "nvidia_e8_h4096_1027_active_1032_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY = {
    (515, 520): _NVIDIA_E8_H4096_515_ACTIVE_520_MAX_W2_SERVICE,
    (1027, 1032): _NVIDIA_E8_H4096_1027_ACTIVE_1032_MAX_W2_SERVICE,
}
_NVIDIA_E8_H4096_EXACT_W2_M_TILE_CENSUS_BY_TOPOLOGY = {
    (515, 520): (64, 65, 65, 64, 65, 65, 63, 64),
    (1027, 1032): (127, 131, 130, 127, 127, 128, 129, 128),
}
_NVIDIA_E512_H4096_DEEP_W1_SERVICE = (
    "nvidia_e512_h4096_deep_fused_moe_w1_operand_pipeline")
_NVIDIA_E512_H4096_DEEP_W1_M_TILE_CENSUS = (
    11, 10, 11, 10, 10, 10, 10, 10, 10, 11, 10, 10, 10, 11, 10, 11,
    10, 10, 10, 10, 11, 11, 11, 10, 11, 11, 10, 11, 10, 11, 10, 10,
    11, 11, 10, 10, 10, 10, 11, 11, 10, 11, 11, 11, 10, 11, 10, 10,
    11, 12, 10, 11, 10, 10, 11, 10, 10, 11, 11, 11, 11, 11, 11, 10,
    11, 11, 11, 11, 11, 10, 11, 10, 10, 11, 11, 11, 10, 10, 11, 11,
    11, 11, 11, 10, 10, 11, 10, 11, 11, 11, 10, 11, 10, 11, 10, 10,
    11, 10, 10, 11, 10, 11, 11, 11, 11, 11, 11, 10, 11, 10, 11, 10,
    10, 11, 11, 11, 11, 10, 11, 11, 10, 10, 11, 10, 11, 10, 10, 11,
    11, 10, 11, 10, 10, 10, 10, 11, 11, 10, 10, 11, 11, 10, 11, 10,
    11, 11, 11, 11, 10, 10, 11, 10, 11, 11, 11, 10, 10, 10, 10, 11,
    11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 11, 11, 11, 10, 11, 10, 11, 11, 10, 11, 11, 11, 10, 11,
    11, 11, 11, 11, 10, 10, 10, 11, 10, 11, 10, 10, 10, 11, 10, 11,
    10, 10, 11, 11, 10, 11, 11, 11, 11, 11, 10, 10, 10, 11, 11, 10,
    10, 10, 10, 11, 10, 10, 10, 10, 10, 11, 10, 10, 11, 10, 10, 11,
    10, 11, 11, 10, 10, 11, 10, 10, 11, 11, 11, 11, 10, 10, 11, 10,
    10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 10, 10, 11, 10, 10, 10,
    10, 11, 10, 10, 10, 10, 10, 10, 11, 10, 11, 11, 10, 9, 10, 11,
    11, 10, 11, 10, 11, 10, 10, 10, 10, 11, 11, 11, 11, 10, 10, 10,
    10, 11, 11, 10, 10, 10, 11, 10, 10, 10, 10, 11, 10, 11, 10, 11,
    11, 11, 11, 10, 10, 11, 11, 10, 11, 10, 11, 11, 11, 10, 10, 10,
    10, 11, 10, 11, 11, 10, 9, 10, 10, 11, 11, 10, 10, 10, 11, 10,
    11, 10, 11, 11, 11, 10, 11, 10, 10, 10, 11, 11, 11, 11, 10, 10,
    10, 11, 11, 11, 11, 11, 11, 10, 11, 10, 11, 11, 11, 11, 10, 11,
    11, 11, 10, 10, 10, 10, 10, 10, 11, 10, 11, 10, 11, 10, 11, 11,
    10, 11, 10, 10, 11, 11, 11, 10, 11, 11, 11, 11, 11, 10, 10, 11,
    11, 11, 10, 11, 10, 12, 10, 10, 11, 10, 10, 11, 10, 11, 10, 11,
    10, 10, 11, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 11, 11, 10,
    11, 10, 10, 10, 11, 10, 11, 10, 10, 11, 10, 11, 10, 10, 11, 11,
    10, 10, 11, 10, 11, 11, 10, 11, 11, 10, 11, 11, 11, 10, 11, 10,
    11, 11, 10, 10, 10, 10, 11, 11, 10, 11, 11, 10, 11, 10, 11, 10,
    10, 11, 10, 10, 10, 10, 11, 10, 11, 11, 10, 10, 11, 9, 10, 11)
_NVIDIA_E256_H7168_DEEP_W1_SERVICE = (
    "nvidia_e256_h7168_deep_fused_moe_w1_operand_pipeline")
_NVIDIA_E256_H7168_DEEP_W2_SERVICE = (
    "nvidia_e256_h7168_deep_fused_moe_w2_operand_pipeline")
_NVIDIA_E256_H7168_MID_WORKSPACE_W1_SERVICE = (
    "nvidia_e256_h7168_mid_workspace_fused_moe_w1_operand_pipeline")
_NVIDIA_E256_H7168_MID_WORKSPACE_W2_SERVICE = (
    "nvidia_e256_h7168_mid_workspace_fused_moe_w2_operand_pipeline")
_NVIDIA_E256_H7168_2166_EXACT_W1_SERVICE = (
    "nvidia_e256_h7168_2166_exact_w1_operand_pipeline")
_NVIDIA_E256_H7168_2166_W1_M_TILE_CENSUS = (
    8, 9, 8, 8, 9, 8, 8, 9, 8, 9, 8, 8, 9, 9, 9, 8,
    8, 9, 8, 8, 8, 9, 8, 8, 8, 9, 9, 9, 9, 9, 8, 8,
    10, 8, 8, 8, 9, 8, 8, 8, 9, 8, 9, 8, 8, 9, 8, 8,
    8, 8, 8, 8, 9, 9, 9, 9, 9, 8, 8, 8, 9, 9, 9, 8,
    8, 9, 9, 8, 8, 9, 8, 8, 8, 8, 8, 9, 8, 9, 8, 8,
    9, 9, 8, 8, 8, 8, 8, 9, 8, 9, 8, 8, 9, 8, 8, 9,
    10, 8, 8, 8, 9, 9, 9, 8, 9, 9, 9, 8, 9, 8, 8, 8,
    8, 8, 9, 9, 8, 8, 8, 8, 9, 8, 9, 9, 8, 9, 8, 8,
    8, 8, 8, 9, 9, 8, 9, 9, 8, 9, 8, 8, 8, 8, 8, 8,
    8, 9, 9, 9, 8, 10, 8, 8, 8, 9, 8, 9, 8, 8, 8, 8,
    9, 9, 9, 8, 9, 8, 9, 9, 9, 8, 9, 8, 9, 9, 8, 8,
    8, 8, 9, 9, 8, 8, 9, 8, 9, 8, 9, 9, 8, 9, 8, 9,
    8, 9, 8, 9, 8, 9, 9, 9, 9, 8, 8, 8, 9, 8, 8, 9,
    8, 9, 9, 8, 8, 9, 8, 8, 9, 9, 9, 9, 9, 9, 8, 9,
    8, 8, 9, 8, 8, 8, 9, 9, 9, 9, 8, 8, 8, 8, 9, 9,
    8, 9, 9, 8, 9, 9, 8, 9, 8, 9, 9, 8, 8, 8, 9, 9)
_NVIDIA_E256_H7168_DEEP_M_TILE_CENSUS = (
    17, 17, 16, 16, 16, 16, 17, 17, 17, 16, 17, 16, 16, 17, 17, 17,
    16, 16, 16, 17, 17, 16, 16, 16, 16, 16, 17, 16, 16, 16, 17, 17,
    16, 16, 17, 17, 17, 16, 16, 17, 16, 17, 16, 17, 16, 16, 16, 17,
    16, 17, 16, 16, 17, 17, 16, 16, 16, 17, 17, 16, 17, 17, 16, 17,
    16, 16, 16, 17, 16, 17, 16, 16, 17, 17, 17, 16, 17, 16, 16, 16,
    17, 17, 16, 16, 16, 17, 16, 16, 17, 17, 16, 16, 17, 17, 16, 17,
    17, 17, 16, 17, 16, 16, 17, 17, 16, 16, 16, 17, 17, 17, 17, 17,
    16, 17, 16, 16, 16, 17, 17, 17, 17, 16, 17, 15, 17, 16, 16, 16,
    17, 17, 17, 17, 17, 16, 17, 16, 16, 16, 17, 16, 16, 17, 17, 16,
    17, 16, 16, 17, 16, 17, 16, 16, 17, 17, 17, 16, 16, 16, 16, 16,
    16, 17, 16, 17, 17, 16, 17, 16, 17, 17, 17, 17, 17, 16, 16, 16,
    17, 17, 16, 16, 17, 16, 17, 17, 16, 16, 17, 16, 17, 17, 17, 17,
    17, 16, 16, 16, 16, 17, 16, 16, 17, 16, 16, 17, 16, 17, 17, 17,
    17, 17, 17, 17, 17, 16, 16, 16, 17, 17, 17, 17, 17, 18, 16, 17,
    16, 16, 18, 16, 17, 17, 16, 17, 17, 17, 17, 16, 17, 16, 17, 17,
    16, 17, 17, 16, 18, 16, 16, 17, 17, 16, 16, 17, 17, 16, 17, 17)
_NVIDIA_E256_H7168_2166_ACTIVE_2300_MAX_W2_SERVICE = (
    "nvidia_e256_h7168_2166_active_2300_max_fused_moe_w2_operand_pipeline")
_NVIDIA_DEEP_H2880_W1_W2_SERVICE = (
    "nvidia_deep_h2880_fused_moe_operand_pipeline")
_NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W1_SERVICE = (
    "nvidia_e32_h2880_32_active_36_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W2_SERVICE = (
    "nvidia_e32_h2880_32_active_36_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W1_SERVICE = (
    "nvidia_e32_h2880_143_active_160_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W2_SERVICE = (
    "nvidia_e32_h2880_143_active_160_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E512_1001_ACTIVE_1144_MAX_W1_SERVICE = (
    "nvidia_e512_1001_active_1144_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E512_1001_ACTIVE_1144_MAX_W2_SERVICE = (
    "nvidia_e512_1001_active_1144_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W1_SERVICE = (
    "nvidia_bm64_e128_574_active_638_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W2_SERVICE = (
    "nvidia_bm64_e128_574_active_638_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W1_SERVICE = (
    "nvidia_e128_h2880_1084_active_1150_exact_w1_operand_pipeline")
_NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W2_SERVICE = (
    "nvidia_e128_h2880_1084_active_1150_exact_w2_operand_pipeline")
_NVIDIA_E128_H2880_1084_ACTIVE_1150_M_TILE_CENSUS = (
    8, 8, 8, 8, 8, 9, 8, 8, 8, 9, 8, 9, 8, 8, 8, 9,
    9, 9, 8, 8, 9, 9, 8, 9, 8, 8, 9, 9, 8, 8, 9, 8,
    9, 9, 8, 8, 9, 8, 8, 8, 8, 8, 8, 9, 9, 8, 8, 9,
    9, 9, 8, 8, 9, 8, 8, 8, 8, 9, 8, 9, 9, 9, 8, 9,
    9, 8, 8, 8, 9, 9, 9, 8, 8, 9, 9, 9, 8, 9, 9, 9,
    9, 9, 8, 9, 8, 9, 8, 8, 9, 9, 8, 9, 8, 8, 9, 8,
    8, 9, 9, 8, 9, 8, 8, 8, 9, 9, 9, 8, 8, 8, 8, 8,
    9, 9, 9, 8, 9, 8, 9, 8, 8, 9, 9, 9, 9, 8, 9, 8)
_NVIDIA_BM16_H2880_31_ACTIVE_SERVICE = (
    "nvidia_bm16_h2880_31_active_fused_moe_operand_pipeline")
_NVIDIA_BM16_H2880_53_ACTIVE_SERVICE = (
    "nvidia_bm16_h2880_53_active_fused_moe_operand_pipeline")
_NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE = (
    "nvidia_bm16_e128_88_active_128_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE = (
    "nvidia_bm16_e128_88_active_128_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W1_SERVICE = (
    "nvidia_bm16_e128_177_active_248_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W2_SERVICE = (
    "nvidia_bm16_e128_177_active_248_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W1_SERVICE = (
    "nvidia_bm16_e8_7_active_8_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W2_SERVICE = (
    "nvidia_bm16_e8_7_active_8_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W1_SERVICE = (
    "nvidia_bm16_e8_h6144_7_active_8_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W2_SERVICE = (
    "nvidia_bm16_e8_h6144_7_active_8_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E256_16_ACTIVE_241_MAX_W2_SERVICE = (
    "nvidia_bm16_e256_16_active_241_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E512_20_ACTIVE_482_MAX_W2_SERVICE = (
    "nvidia_bm16_e512_20_active_482_max_fused_moe_w2_operand_pipeline")
_METAX_GROUPED_OPERAND_LONG_K_SERVICE = (
    "metax_grouped_fused_moe_long_k_operand_pipeline")
_METAX_GROUPED_OPERAND_WIDE_N_SERVICE = (
    "metax_grouped_fused_moe_wide_n_operand_pipeline")
_METAX_GLOBAL_DEEP_OPERAND_LONG_K_SERVICE = (
    "metax_global_deep_fused_moe_long_k_operand_pipeline")
_METAX_GLOBAL_LARGE_OPERAND_WIDE_N_SERVICE = (
    "metax_global_large_fused_moe_wide_n_operand_pipeline")
_METAX_LONG_LIFETIME_OPERAND_LONG_K_SERVICE = (
    "metax_long_lifetime_fused_moe_long_k_operand_pipeline")
_METAX_LONG_LIFETIME_OPERAND_WIDE_N_SERVICE = (
    "metax_long_lifetime_fused_moe_wide_n_operand_pipeline")
_METAX_REPEATED_OPERAND_DEEP_K_SERVICE = (
    "metax_repeated_fused_moe_deep_k_operand_pipeline")
_METAX_REPEATED_OPERAND_WIDE_N_SERVICE = (
    "metax_repeated_fused_moe_wide_n_operand_pipeline")
_METAX_REPEATED_OPERAND_BALANCED_N_SERVICE = (
    "metax_repeated_fused_moe_balanced_n_operand_pipeline")
_METAX_REPEATED_OPERAND_MID_LONG_K_SERVICE = (
    "metax_repeated_fused_moe_mid_long_k_operand_pipeline")
_METAX_REPEATED_OPERAND_MID_WIDE_N_SERVICE = (
    "metax_repeated_fused_moe_mid_wide_n_operand_pipeline")
_METAX_SINGLE_TILE_OPERAND_LONG_K_SERVICE = (
    "metax_single_tile_fused_moe_long_k_operand_pipeline")
_METAX_SINGLE_TILE_OPERAND_WIDE_N_SERVICE = (
    "metax_single_tile_fused_moe_wide_n_operand_pipeline")
_METAX_EIGHT_EXPERT_EIGHT_ACTIVE_TWELVE_MAX_W2_SERVICE = (
    "metax_eight_expert_eight_active_twelve_max_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W1_SERVICE = (
    "metax_eight_expert_single_tile_large_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W2_SERVICE = (
    "metax_eight_expert_single_tile_large_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W1_SERVICE = (
    "metax_eight_expert_small_single_tile_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W2_SERVICE = (
    "metax_eight_expert_small_single_tile_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_STEADY_WIDE_N_SERVICE = (
    "metax_eight_expert_steady_fused_moe_wide_n_operand_pipeline")
_METAX_EIGHT_EXPERT_STEADY_K192_N512_W1_SERVICE = (
    "metax_eight_expert_steady_k192_n512_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE = (
    "metax_eight_expert_68_active_72_max_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W2_SERVICE = (
    "metax_eight_expert_68_active_72_max_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K128_N448_W1_SERVICE = (
    "metax_eight_expert_68_active_72_max_k128_n448_"
    "fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K448_N64_W2_SERVICE = (
    "metax_eight_expert_68_active_72_max_k448_n64_"
    "fused_moe_w2_operand_pipeline")
_METAX_E32_32_ACTIVE_48_MAX_W1_SERVICE = (
    "metax_e32_32_active_48_max_fused_moe_w1_operand_pipeline")
_METAX_E32_32_ACTIVE_48_MAX_W2_SERVICE = (
    "metax_e32_32_active_48_max_fused_moe_w2_operand_pipeline")
_METAX_E32_47_ACTIVE_64_MAX_W1_SERVICE = (
    "metax_e32_47_active_64_max_fused_moe_w1_operand_pipeline")
_METAX_E32_47_ACTIVE_64_MAX_W2_SERVICE = (
    "metax_e32_47_active_64_max_fused_moe_w2_operand_pipeline")
_METAX_E32_78_ACTIVE_96_MAX_W1_SERVICE = (
    "metax_e32_78_active_96_max_fused_moe_w1_operand_pipeline")
_METAX_E32_78_ACTIVE_96_MAX_W2_SERVICE = (
    "metax_e32_78_active_96_max_fused_moe_w2_operand_pipeline")
_METAX_E32_143_ACTIVE_160_MAX_W1_SERVICE = (
    "metax_e32_143_active_160_max_fused_moe_w1_operand_pipeline")
_METAX_E32_143_ACTIVE_160_MAX_W2_SERVICE = (
    "metax_e32_143_active_160_max_fused_moe_w2_operand_pipeline")
_METAX_E256_LONG_LIFETIME_LONG_K_SERVICE = (
    "metax_e256_long_lifetime_fused_moe_long_k_operand_pipeline")
_METAX_E256_LONG_LIFETIME_WIDE_N_SERVICE = (
    "metax_e256_long_lifetime_fused_moe_wide_n_operand_pipeline")
_METAX_E256_LONG_LIFETIME_MID_W1_SERVICE = (
    "metax_e256_long_lifetime_mid_fused_moe_w1_operand_pipeline")
_METAX_E256_LONG_LIFETIME_MID_W2_SERVICE = (
    "metax_e256_long_lifetime_mid_fused_moe_w2_operand_pipeline")
_METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W1_SERVICE = (
    "metax_e512_dense_ten_to_eleven_tile_fused_moe_w1_operand_pipeline")
_METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W2_SERVICE = (
    "metax_e512_dense_ten_to_eleven_tile_fused_moe_w2_operand_pipeline")
_METAX_E512_DENSE_LARGE_W2_SERVICE = (
    "metax_e512_dense_activation_refreshed_large_w2_operand_pipeline")
_METAX_E512_1010_ACTIVE_1144_MAX_W2_SERVICE = (
    "metax_e512_1010_active_1144_max_fused_moe_w2_operand_pipeline")
_METAX_E128_LONG_LIFETIME_W1_SERVICE = (
    "metax_e128_long_lifetime_fused_moe_w1_operand_pipeline")
_METAX_E128_LONG_LIFETIME_W2_SERVICE = (
    "metax_e128_long_lifetime_fused_moe_w2_operand_pipeline")
_METAX_E128_DEEP_W2_SERVICE = (
    "metax_e128_deep_fused_moe_w2_operand_pipeline")
_METAX_E128_EIGHT_TILE_W1_SERVICE = (
    "metax_e128_eight_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_EIGHT_TILE_W2_SERVICE = (
    "metax_e128_eight_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_TOPK16_EIGHT_TILE_W1_SERVICE = (
    "metax_e128_topk16_eight_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_EIGHT_TILE_WIDE_N_W2_SERVICE = (
    "metax_e128_eight_tile_wide_n_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_TWELVE_TILE_W2_SERVICE = (
    "metax_eight_expert_twelve_tile_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_TWELVE_TILE_DEEP_W2_SERVICE = (
    "metax_eight_expert_twelve_tile_deep_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_TWELVE_TILE_K128_N448_W1_SERVICE = (
    "metax_eight_expert_twelve_tile_k128_n448_"
    "fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_TWELVE_TILE_K192_N512_W1_SERVICE = (
    "metax_eight_expert_twelve_tile_k192_n512_"
    "fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W1_SERVICE = (
    "metax_eight_expert_twenty_one_tile_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W2_SERVICE = (
    "metax_eight_expert_twenty_one_tile_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W1_SERVICE = (
    "metax_eight_expert_twenty_active_twenty_four_max_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W2_SERVICE = (
    "metax_eight_expert_twenty_active_twenty_four_max_fused_moe_w2_operand_pipeline")
_METAX_E128_SPARSE_TWO_TILE_W1_SERVICE = (
    "metax_e128_sparse_two_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_SPARSE_TWO_TILE_W2_SERVICE = (
    "metax_e128_sparse_two_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_188_ACTIVE_254_MAX_W1_SERVICE = (
    "metax_e128_188_active_254_max_fused_moe_w1_operand_pipeline")
_METAX_E128_188_ACTIVE_254_MAX_W2_SERVICE = (
    "metax_e128_188_active_254_max_fused_moe_w2_operand_pipeline")
_METAX_E128_SINGLE_TILE_W1_SERVICE = (
    "metax_e128_single_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_SINGLE_TILE_W2_SERVICE = (
    "metax_e128_single_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_SINGLE_TILE_MID_W1_SERVICE = (
    "metax_e128_single_tile_mid_fused_moe_w1_operand_pipeline")
_METAX_E128_SINGLE_TILE_MID_W2_SERVICE = (
    "metax_e128_single_tile_mid_fused_moe_w2_operand_pipeline")
_METAX_E128_SINGLE_TILE_LARGE_W1_SERVICE = (
    "metax_e128_single_tile_large_fused_moe_w1_operand_pipeline")
_METAX_E128_SINGLE_TILE_LARGE_W2_SERVICE = (
    "metax_e128_single_tile_large_fused_moe_w2_operand_pipeline")
_METAX_E128_UPPER_TWO_TILE_W1_SERVICE = (
    "metax_e128_upper_two_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_UPPER_TWO_TILE_W2_SERVICE = (
    "metax_e128_upper_two_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_TOPK8_128_ACTIVE_190_MAX_W1_SERVICE = (
    "metax_e128_topk8_128_active_190_max_fused_moe_w1_operand_pipeline")
_METAX_E128_TOPK16_128_ACTIVE_190_MAX_W1_SERVICE = (
    "metax_e128_topk16_128_active_190_max_fused_moe_w1_operand_pipeline")
_METAX_E128_128_ACTIVE_190_MAX_W2_SERVICE = (
    "metax_e128_128_active_190_max_fused_moe_w2_operand_pipeline")
_METAX_E128_REPEATED_SOURCE_TOPK8_W1_SERVICE = (
    "metax_e128_repeated_source_topk8_fused_moe_w1_operand_pipeline")
_METAX_E128_REPEATED_SOURCE_TOPK16_W1_SERVICE = (
    "metax_e128_repeated_source_topk16_fused_moe_w1_operand_pipeline")
_METAX_E128_REPEATED_K24_N32_W2_SERVICE = (
    "metax_e128_repeated_k24_n32_fused_moe_w2_operand_pipeline")
_METAX_E256_EIGHT_TILE_LONG_K_SERVICE = (
    "metax_e256_eight_tile_fused_moe_long_k_operand_pipeline")
_METAX_E256_EIGHT_TILE_WIDE_N_SERVICE = (
    "metax_e256_eight_tile_fused_moe_wide_n_operand_pipeline")
_METAX_E256_EIGHT_TILE_MID_W1_SERVICE = (
    "metax_e256_eight_tile_mid_fused_moe_w1_operand_pipeline")
_METAX_E256_EIGHT_TILE_MID_W2_SERVICE = (
    "metax_e256_eight_tile_mid_fused_moe_w2_operand_pipeline")
_METAX_E256_FOUR_TILE_LONG_K_SERVICE = (
    "metax_e256_four_tile_fused_moe_long_k_operand_pipeline")
_METAX_E256_FOUR_TILE_WIDE_N_SERVICE = (
    "metax_e256_four_tile_fused_moe_wide_n_operand_pipeline")
_METAX_E256_FOUR_TILE_MID_LONG_K_SERVICE = (
    "metax_e256_four_tile_mid_fused_moe_long_k_operand_pipeline")
_METAX_E256_FOUR_TILE_MID_WIDE_N_SERVICE = (
    "metax_e256_four_tile_mid_fused_moe_wide_n_operand_pipeline")
_METAX_E128_FOUR_TILE_W1_SERVICE = (
    "metax_e128_four_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_FOUR_TILE_W2_SERVICE = (
    "metax_e128_four_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_FOUR_TILE_LARGE_W1_SERVICE = (
    "metax_e128_four_tile_large_fused_moe_w1_operand_pipeline")
_METAX_E128_FOUR_TILE_LARGE_W2_SERVICE = (
    "metax_e128_four_tile_large_fused_moe_w2_operand_pipeline")
_METAX_E256_UPPER_TWO_TILE_LONG_K_SERVICE = (
    "metax_e256_upper_two_tile_fused_moe_long_k_operand_pipeline")
_METAX_E256_UPPER_TWO_TILE_WIDE_N_SERVICE = (
    "metax_e256_upper_two_tile_fused_moe_wide_n_operand_pipeline")
_METAX_E256_SINGLE_TILE_W1_SERVICE = (
    "metax_e256_single_tile_fused_moe_w1_operand_pipeline")
_METAX_E256_SINGLE_TILE_W2_SERVICE = (
    "metax_e256_single_tile_fused_moe_w2_operand_pipeline")
_METAX_E256_SINGLE_TILE_MID_DEEP_W1_SERVICE = (
    "metax_e256_single_tile_mid_deep_fused_moe_w1_operand_pipeline")
_METAX_E256_SINGLE_TILE_MID_DEEP_W2_SERVICE = (
    "metax_e256_single_tile_mid_deep_fused_moe_w2_operand_pipeline")
_METAX_E256_SINGLE_TILE_LARGE_W1_SERVICE = (
    "metax_e256_single_tile_large_fused_moe_w1_operand_pipeline")
_METAX_E256_SINGLE_TILE_LARGE_W2_SERVICE = (
    "metax_e256_single_tile_large_fused_moe_w2_operand_pipeline")
_METAX_E256_SINGLE_TILE_LARGE_DEEP_W1_SERVICE = (
    "metax_e256_single_tile_large_deep_fused_moe_w1_operand_pipeline")
_METAX_E256_SINGLE_TILE_LARGE_DEEP_W2_SERVICE = (
    "metax_e256_single_tile_large_deep_fused_moe_w2_operand_pipeline")
_METAX_E512_SINGLE_TILE_W1_SERVICE = (
    "metax_e512_single_tile_fused_moe_w1_operand_pipeline")
_METAX_E512_SINGLE_TILE_W2_SERVICE = (
    "metax_e512_single_tile_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W1_SERVICE = (
    "metax_bm16_e512_521_active_800_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W2_SERVICE = (
    "metax_bm16_e512_521_active_800_max_fused_moe_w2_operand_pipeline")
_METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W1_SERVICE = (
    "metax_bm64_e512_512_active_664_max_fused_moe_w1_operand_pipeline")
_METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W2_SERVICE = (
    "metax_bm64_e512_512_active_664_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_ACTIVE_GRID_WIDE_N_SERVICE = (
    "metax_bm16_active_grid_fused_moe_wide_n_operand_pipeline")
_METAX_BM16_E256_SIXTEEN_ACTIVE_W2_SERVICE = (
    "metax_bm16_e256_sixteen_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E256_SIXTEEN_ACTIVE_DEEP_W2_SERVICE = (
    "metax_bm16_e256_sixteen_active_deep_"
    "fused_moe_w2_operand_pipeline")
_METAX_BM16_SPARSE_PADDED_W2_SERVICE = (
    "metax_bm16_sparse_padded_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_SPARSE_PADDED_W2_SERVICE = (
    "metax_bm16_e512_sparse_padded_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_FORTY_ACTIVE_W1_SERVICE = (
    "metax_bm16_e512_forty_active_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_FORTY_ACTIVE_W2_SERVICE = (
    "metax_bm16_e512_forty_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_FORTY_ACTIVE_K8_N64_W2_SERVICE = (
    "metax_bm16_e512_forty_active_k8_n64_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_75_ACTIVE_80_MAX_W1_SERVICE = (
    "metax_bm16_e512_75_active_80_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_75_ACTIVE_485_MAX_W2_SERVICE = (
    "metax_bm16_e512_75_active_485_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_TWENTY_ACTIVE_W1_SERVICE = (
    "metax_bm16_e512_twenty_active_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_TWENTY_ACTIVE_W2_SERVICE = (
    "metax_bm16_e512_twenty_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_TWENTY_ACTIVE_K8_N64_W2_SERVICE = (
    "metax_bm16_e512_twenty_active_k8_n64_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_TEN_ACTIVE_W1_SERVICE = (
    "metax_bm16_e512_ten_active_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_TEN_ACTIVE_W2_SERVICE = (
    "metax_bm16_e512_ten_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_143_ACTIVE_490_MAX_W1_SERVICE = (
    "metax_bm16_e512_143_active_490_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_143_ACTIVE_490_MAX_W2_SERVICE = (
    "metax_bm16_e512_143_active_490_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_MID_SPARSE_W1_SERVICE = (
    "metax_bm16_e512_mid_sparse_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_MID_SPARSE_W2_SERVICE = (
    "metax_bm16_e512_mid_sparse_fused_moe_w2_operand_pipeline")
_METAX_BM16_E256_SPARSE_COLLIDING_W1_SERVICE = (
    "metax_bm16_e256_sparse_colliding_fused_moe_w1_operand_pipeline")
_METAX_BM16_E256_SPARSE_COLLIDING_W2_SERVICE = (
    "metax_bm16_e256_sparse_colliding_fused_moe_w2_operand_pipeline")
_METAX_BM16_E256_106_ACTIVE_248_MAX_W1_SERVICE = (
    "metax_bm16_e256_106_active_248_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E256_106_ACTIVE_248_MAX_W2_SERVICE = (
    "metax_bm16_e256_106_active_248_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_31_ACTIVE_32_MAX_W1_SERVICE = (
    "metax_bm16_31_active_32_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_31_ACTIVE_W2_SERVICE = (
    "metax_bm16_e128_31_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_53_ACTIVE_64_MAX_W1_SERVICE = (
    "metax_bm16_e128_53_active_64_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_53_ACTIVE_124_MAX_W2_SERVICE = (
    "metax_bm16_e128_53_active_124_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK8_16_ACTIVE_16_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk8_16_active_16_max_"
    "fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK8_16_ACTIVE_121_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk8_16_active_121_max_"
    "fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_16_ACTIVE_121_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_16_active_121_max_"
    "fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_16_ACTIVE_16_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_16_active_16_max_"
    "fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_152_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_152_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_152_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_152_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_64_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_64_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_124_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_124_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_128_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_128_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_128_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_128_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_136_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_136_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_136_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_136_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_184_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_184_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_184_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_184_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_248_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_248_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_248_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_248_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_FOUR_ACTIVE_MID_K_W1_SERVICE = (
    "metax_bm16_four_active_mid_k_fused_moe_w1_operand_pipeline")
_METAX_BM16_FOUR_ACTIVE_MID_K_W2_SERVICE = (
    "metax_bm16_four_active_mid_k_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W1_SERVICE = (
    "metax_bm16_e128_eight_active_mid_k_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W2_SERVICE = (
    "metax_bm16_e128_eight_active_mid_k_fused_moe_w2_operand_pipeline")
_METAX_BM16_EIGHT_ACTIVE_MID_K_W2_SERVICE = (
    "metax_bm16_eight_active_mid_k_fused_moe_w2_operand_pipeline")
_METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W1_SERVICE = (
    "metax_bm16_e32_seven_active_mid_k_fused_moe_w1_operand_pipeline")
_METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W2_SERVICE = (
    "metax_bm16_e32_seven_active_mid_k_fused_moe_w2_operand_pipeline")
_METAX_BM64_E32_32_ACTIVE_36_MAX_W1_SERVICE = (
    "metax_bm64_e32_32_active_36_max_fused_moe_w1_operand_pipeline")
_METAX_BM64_E32_32_ACTIVE_36_MAX_W2_SERVICE = (
    "metax_bm64_e32_32_active_36_max_fused_moe_w2_operand_pipeline")
_METAX_BM64_E32_32_ACTIVE_40_MAX_W1_SERVICE = (
    "metax_bm64_e32_32_active_40_max_fused_moe_w1_operand_pipeline")
_METAX_BM64_E32_32_ACTIVE_40_MAX_W2_SERVICE = (
    "metax_bm64_e32_32_active_40_max_fused_moe_w2_operand_pipeline")
_GROUPED_OPERAND_STREAM_COUNTS = (1, 2, 4, 8)
_CANN_OPERAND_A_L2_HIT_SERVICE = (
    "cann_grouped_moe_operand_a_l2_hit")
_CANN_OPERAND_A_DRAM_MISS_SERVICE = (
    "cann_grouped_moe_operand_a_dram_miss")
_CANN_OPERAND_B_L2_HIT_SERVICE = (
    "cann_grouped_moe_operand_b_l2_hit")
_CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE = (
    "cann_grouped_moe_operand_b_k2880_stride_l2_hit")
_CANN_OPERAND_B_DRAM_MISS_SERVICE = (
    "cann_grouped_moe_operand_b_dram_miss")
_CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE = (
    "cann_grouped_moe_operand_b_k2880_strided_dram_miss")
_CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE = (
    "cann_grouped_moe_operand_b_k6144_m1_dram_miss")
_CANN_GROUPED_K_OUTER_BASE_TILES = 4


def _projection(kernel: KernelSpec) -> Tuple[str, int, int, int]:
    tokens, topk, hidden, intermediate = (
        int(kernel.shape[name])
        for name in ("tokens", "topk", "hidden", "intermediate"))
    variant = kernel.variant.lower()
    if variant == "w1":
        return variant, tokens, hidden, 2 * intermediate
    elif variant == "w2":
        return variant, tokens * topk, intermediate, hidden
    else:
        raise ValueError("FUSED_MOE variant must be 'w1' or 'w2'")


def _tile(kernel: KernelSpec) -> Tuple[int, int, int]:
    tiling = kernel.launch.tiling
    if tiling is None or not {"m", "n", "k"}.issubset(tiling.tile_shape):
        raise ValueError("FUSED_MOE requires m/n/k tiling in LaunchConfig")
    return tuple(int(tiling.tile_shape[name]) for name in ("m", "n", "k"))


def _routing(kernel: KernelSpec):
    experts = int(kernel.shape["experts"])
    global_counts = tuple(
        int(value) for value in kernel.attributes["tokens_per_expert"])
    local_counts = tuple(
        int(value)
        for value in kernel.attributes["local_tokens_per_expert"])
    if len(global_counts) != experts:
        raise ValueError(
            "tokens_per_expert has "
            f"{len(global_counts)} entries, expected {experts}")
    if len(local_counts) != experts:
        raise ValueError(
            "local_tokens_per_expert has "
            f"{len(local_counts)} entries, expected {experts}")
    return global_counts, local_counts


def _source_w2_routed_weight_program(kernel: KernelSpec) -> Optional[bool]:
    """Return whether C500 selected W2's routed-weight source program.

    The frozen chain writes the activation directly to ``intermediate_cache2``
    and passes that buffer to W2.  This marker therefore denotes only W2's
    later router-weight load and multiply, never a standalone activation copy.
    It remains optional because it is a C500-specific source fact.
    """

    parameter = SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER
    if parameter not in kernel.launch.parameters:
        return None
    raw_invocations = kernel.launch.parameters[parameter]
    if (isinstance(raw_invocations, bool)
            or not isinstance(raw_invocations, (int, float))
            or not float(raw_invocations).is_integer()):
        raise TypeError(
            "FUSED_MOE source W2 routed-weight program must be an integer")
    program_selected = int(raw_invocations)
    if program_selected not in (0, 1):
        raise ValueError(
            "FUSED_MOE source W2 routed-weight program must be zero or one")
    variant = kernel.variant.lower()
    if program_selected == 0 and variant != "w1":
        raise ValueError(
            "FUSED_MOE absent W2 routed-weight program is valid only for W1")
    if program_selected == 1 and variant != "w2":
        raise ValueError(
            "FUSED_MOE W2 routed-weight program is valid only for W2")
    return bool(program_selected)


def _source_routed_weight_scale_operations(
        kernel: KernelSpec, *, routed_rows: float,
        projection_n: int) -> float:
    """Return the source-defined C500 W2 router-scale ALU work.

    The C500 Triton W2 program selected by the routed-weight marker
    loads one router weight for each valid routed row and multiplies the
    accumulated logical output row before its masked store. W1 has no such
    source operation. Do not infer this edge for backends whose launch lacks
    the C500 source marker.
    """

    program_selected = _source_w2_routed_weight_program(kernel)
    if program_selected is None or not program_selected:
        return 0.0
    if routed_rows < 0.0:
        raise ValueError("FUSED_MOE routed rows cannot be negative")
    if projection_n <= 0:
        raise ValueError("FUSED_MOE routed-weight scale needs positive N")
    return routed_rows * float(projection_n)


def _source_routed_weight_read_traffic(
        kernel: KernelSpec, *, routed_rows: float,
        n_tiles: int) -> Tuple[float, float]:
    """Return C500 W2 FP32 router-weight ingress and unique bytes.

    The frozen MetaX Triton W2 program loads ``topk_weights_ptr`` after its
    K loop for every valid routed row in every output-N program.  The router
    tensor is created as FP32 by ``fused_topk``.  Its per-program requests
    must remain a distinct post-matmul stream: treating it as activation
    ingress would move a source-visible dependency ahead of the matrix
    multiply.  The source marker is C500-specific, so backends without it do
    not infer this read.
    """

    program_selected = _source_w2_routed_weight_program(kernel)
    if program_selected is None or not program_selected:
        return 0.0, 0.0
    if routed_rows < 0.0:
        raise ValueError("FUSED_MOE routed rows cannot be negative")
    if n_tiles <= 0:
        raise ValueError("FUSED_MOE routed-weight read needs output tiles")
    router_weight_bytes = 4.0
    compulsory = routed_rows * router_weight_bytes
    return compulsory * float(n_tiles), compulsory


def _validate_expert_parallel_launch(kernel: KernelSpec) -> None:
    expert_parallel_size = int(kernel.attributes["expert_parallel_size"])
    if expert_parallel_size == 1:
        return
    parameters = kernel.launch.parameters
    if "expert_parallel_size" not in parameters:
        raise NotImplementedError(
            "FUSED_MOE expert-parallel launch is not source-backed")
    launch_value = parameters["expert_parallel_size"]
    if (isinstance(launch_value, bool)
            or not isinstance(launch_value, (int, float))
            or not float(launch_value).is_integer()):
        raise ValueError(
            "FUSED_MOE launch expert_parallel_size must be an integer")
    if int(launch_value) != expert_parallel_size:
        raise ValueError(
            "FUSED_MOE launch expert_parallel_size disagrees with the "
            "KernelSpec")


@dataclass(frozen=True)
class _CannGroupedMatmulWork:
    tensor_operations: int
    worker_count: int
    expert_group_count: int
    routed_rows: int
    populated_m_tiles: int
    n_tiles: int


@dataclass(frozen=True)
class _CannGroupedMatmulTraffic:
    """Source-derived AIC interfaces and unique GM footprint, in bytes."""

    activation_l2_read_bytes: float
    weight_l2_read_bytes: float
    output_l2_write_bytes: float
    dram_read_bytes: float
    dram_write_bytes: float

    @property
    def l2_read_bytes(self) -> float:
        return self.activation_l2_read_bytes + self.weight_l2_read_bytes

    @property
    def l2_total_bytes(self) -> float:
        return self.l2_read_bytes + self.output_l2_write_bytes

    @property
    def dram_total_bytes(self) -> float:
        return self.dram_read_bytes + self.dram_write_bytes


@dataclass(frozen=True)
class _CannGroupedOperandResidency:
    """Source-ordered full-tile service classes and all operand bytes."""

    a_hits_by_worker: Tuple[int, ...]
    a_misses_by_worker: Tuple[int, ...]
    b_hits_by_worker: Tuple[int, ...]
    b_misses_by_worker: Tuple[int, ...]
    pair_hit_hit_by_worker: Tuple[int, ...]
    pair_hit_miss_by_worker: Tuple[int, ...]
    pair_miss_hit_by_worker: Tuple[int, ...]
    pair_miss_miss_by_worker: Tuple[int, ...]
    tile_classes_by_worker: Tuple[Tuple[Tuple[int, int, int], ...], ...]
    l2_read_bytes: int
    dram_read_bytes: int


@dataclass(frozen=True)
class _GpuGroupedGemmTraffic:
    """Operand-specific grouped-GEMM hierarchy traffic, in bytes."""

    activation_l2_read_bytes: float
    weight_l2_read_bytes: float
    routed_weight_l2_read_bytes: float
    output_l2_write_bytes: float
    activation_compulsory_read_bytes: float
    weight_compulsory_read_bytes: float
    routed_weight_compulsory_read_bytes: float
    activation_dram_read_bytes: float
    weight_dram_read_bytes: float
    routed_weight_dram_read_bytes: float
    output_dram_write_bytes: float

    @property
    def l2_read_bytes(self) -> float:
        return (
            self.activation_l2_read_bytes
            + self.weight_l2_read_bytes
            + self.routed_weight_l2_read_bytes)

    @property
    def l2_total_bytes(self) -> float:
        return self.l2_read_bytes + self.output_l2_write_bytes

    @property
    def dram_read_bytes(self) -> float:
        return (
            self.activation_dram_read_bytes
            + self.weight_dram_read_bytes
            + self.routed_weight_dram_read_bytes)

    @property
    def dram_total_bytes(self) -> float:
        return self.dram_read_bytes + self.output_dram_write_bytes

    @property
    def compulsory_bytes(self) -> float:
        return (
            self.activation_compulsory_read_bytes
            + self.weight_compulsory_read_bytes
            + self.routed_weight_compulsory_read_bytes
            + self.output_l2_write_bytes)


def _cann_mn_block_index(
        relative_block: int, block_dim_m: int,
        block_dim_n: int) -> Tuple[int, int]:
    """Mirror CANN 8.5.1 ``MNBlockIdxCompute`` for Ascend 910B."""

    if relative_block < 0 or relative_block >= block_dim_m * block_dim_n:
        raise ValueError("GroupedMatmul relative block is out of range")
    if block_dim_m <= 0 or block_dim_n <= 0:
        raise ValueError("GroupedMatmul block dimensions must be positive")
    threshold_block_num = 8
    threshold_dim_m = 5
    if block_dim_m <= threshold_dim_m:
        return relative_block // block_dim_n, relative_block % block_dim_n

    threshold_m_dim_n = threshold_block_num * block_dim_n
    full_threshold_extent = (
        block_dim_m * block_dim_n // threshold_m_dim_n
        * threshold_m_dim_n)
    if relative_block >= full_threshold_extent:
        current_threshold_m = block_dim_m % threshold_block_num
    else:
        current_threshold_m = threshold_block_num
    if current_threshold_m <= 0:
        raise AssertionError("GroupedMatmul diagonal M threshold is empty")
    current_threshold_mn = current_threshold_m * threshold_block_num
    local_n_extent = (
        current_threshold_m * block_dim_n // current_threshold_mn
        * current_threshold_mn)
    if relative_block % threshold_m_dim_n >= local_n_extent:
        current_threshold_n = block_dim_n % threshold_block_num
    else:
        current_threshold_n = threshold_block_num
    if current_threshold_n <= 0:
        raise AssertionError("GroupedMatmul diagonal N threshold is empty")

    local_relative = (
        relative_block % threshold_m_dim_n % current_threshold_mn)
    m_index = (
        local_relative % current_threshold_m
        + relative_block // threshold_m_dim_n * threshold_block_num)
    n_index = (
        (local_relative
         + local_relative
         // math.lcm(current_threshold_m, current_threshold_n))
        % current_threshold_n
        + relative_block % threshold_m_dim_n
        // current_threshold_mn * threshold_block_num)
    if m_index >= block_dim_m or n_index >= block_dim_n:
        raise AssertionError("GroupedMatmul diagonal tile index is invalid")
    return m_index, n_index


@lru_cache(maxsize=256)
def _cann_mn_block_order(
        block_dim_m: int, block_dim_n: int) -> Tuple[Tuple[int, int], ...]:
    """Return one immutable CANN diagonal order for a launch tile grid."""

    if block_dim_m <= 0 or block_dim_n <= 0:
        raise ValueError("GroupedMatmul block dimensions must be positive")
    return tuple(
        _cann_mn_block_index(relative_block, block_dim_m, block_dim_n)
        for relative_block in range(block_dim_m * block_dim_n))


def _cann_grouped_matmul_work(kernel: KernelSpec) -> _CannGroupedMatmulWork:
    """Price version-bound BF16 launch tiles on their assigned AICs."""

    if kernel.launch.execution_variant != kernel.variant:
        raise ValueError(
            "CANN GroupedMatmul launch variant disagrees with the kernel")
    if kernel.launch.grid != (kernel.launch.grid_size, 1, 1):
        raise ValueError("CANN GroupedMatmul requires a one-dimensional grid")
    if kernel.launch.block != (1, 1, 1):
        raise ValueError("CANN GroupedMatmul requires one mixed task per block")
    worker_count = kernel.launch.grid_size
    _, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    if tile_m % 16 or tile_n % 16 or tile_k % 16:
        raise ValueError("CANN GroupedMatmul BF16 tiles must be 16-aligned")
    global_counts, local_counts = _routing(kernel)
    if any(rows < 0 for rows in global_counts):
        raise ValueError("CANN GroupedMatmul global expert rows cannot be negative")
    if any(rows < 0 for rows in local_counts):
        raise ValueError("CANN GroupedMatmul expert rows cannot be negative")
    expected_global_rows = (
        int(kernel.shape["tokens"]) * int(kernel.shape["topk"]))
    if sum(global_counts) != expected_global_rows:
        raise ValueError(
            "CANN GroupedMatmul global routing must conserve tokens*topk")
    if any(local > global_ for local, global_ in zip(
            local_counts, global_counts)):
        raise ValueError(
            "CANN GroupedMatmul local expert rows cannot exceed global rows")
    routed_rows = sum(local_counts)
    expected_tile = (
        min(128, ((routed_rows + 15) // 16) * 16), 256, 64)
    if (tile_m, tile_n, tile_k) != expected_tile:
        raise ValueError(
            "CANN 8.5.1 BF16 GroupedMatmul requires exact base tile "
            f"{expected_tile}, got {(tile_m, tile_n, tile_k)}")
    if projection_k % 16:
        raise ValueError("CANN GroupedMatmul K must be Cube-fragment aligned")

    populated_m_tiles = 0
    n_tiles = math.ceil(projection_n / tile_n)
    if int(kernel.attributes["expert_parallel_size"]) == 1:
        if (global_counts != local_counts
                or sum(local_counts) != expected_global_rows):
            raise ValueError(
                "single-rank CANN GroupedMatmul routing must conserve "
                "tokens*topk in both global and local counts")
    padded_n_total = (
        (n_tiles - 1) * tile_n
        + math.ceil((projection_n - (n_tiles - 1) * tile_n) / 16) * 16)
    tensor_operations = 0
    for rows in local_counts:
        if rows == 0:
            continue
        block_dim_m = math.ceil(rows / tile_m)
        block_dim_n = math.ceil(projection_n / tile_n)
        if block_dim_n != n_tiles:
            raise AssertionError(
                "CANN GroupedMatmul N-tile count changed across experts")
        populated_m_tiles += block_dim_m
        tail_rows = rows - (block_dim_m - 1) * tile_m
        padded_m_total = (
            (block_dim_m - 1) * tile_m
            + math.ceil(tail_rows / 16) * 16)
        tensor_operations += 2 * padded_m_total * padded_n_total * projection_k
    if tensor_operations <= 0:
        raise ValueError("CANN GroupedMatmul routing produced no Tensor work")
    return _CannGroupedMatmulWork(
        tensor_operations=tensor_operations,
        worker_count=worker_count,
        expert_group_count=len(local_counts),
        routed_rows=sum(local_counts),
        populated_m_tiles=populated_m_tiles,
        n_tiles=n_tiles,
    )


def _cann_grouped_b_l2_cache_disabled(kernel: KernelSpec) -> bool:
    """Return the static CANN B-cache mode for an all-single-M invocation.

    CANN 8.5.1's exact runtime ``GroupedMatmul`` source calls
    ``SetL2CacheHint(CACHE_MODE_DISABLE)`` on B when ``blockDimM == 1``.
    Restrict this to invocations whose every populated expert has exactly one
    source M tile: mixed launches need a separate per-expert residency trace.
    """

    expected_algorithm = f"cann_grouped_matmul_{kernel.variant}"
    if (kernel.launch.algorithm != expected_algorithm
            or kernel.dtype != "bf16"):
        return False
    tile_m, _, _ = _tile(kernel)
    _, local_counts = _routing(kernel)
    nonempty_rows = tuple(rows for rows in local_counts if rows > 0)
    if not nonempty_rows:
        raise ValueError("CANN GroupedMatmul has no populated experts")
    return all(rows <= tile_m for rows in nonempty_rows)


def _cann_grouped_operand_residency(
        kernel: KernelSpec, work: _CannGroupedMatmulWork,
        hardware) -> _CannGroupedOperandResidency:
    """Classify A/B tile requests using source order and physical L2 size."""

    if hardware.l2_size_bytes <= 0:
        raise ValueError(
            "CANN GroupedMatmul residency requires positive L2 capacity")
    _, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    _, local_counts = _routing(kernel)
    b_l2_cache_disabled = _cann_grouped_b_l2_cache_disabled(kernel)
    result = simulate_cann_grouped_moe_operands(
        projection_k,
        projection_n,
        tile_m,
        tile_n,
        tile_k,
        element_bytes,
        work.worker_count,
        hardware.l2_size_bytes,
        local_counts,
        b_l2_cache_disabled=b_l2_cache_disabled,
    )
    worker_fields = (
        "a_hits_by_worker",
        "a_misses_by_worker",
        "b_hits_by_worker",
        "b_misses_by_worker",
        "pair_hit_hit_by_worker",
        "pair_hit_miss_by_worker",
        "pair_miss_hit_by_worker",
        "pair_miss_miss_by_worker",
    )
    for field_name in worker_fields:
        values = result[field_name]
        if len(values) != work.worker_count:
            raise ValueError(
                "CANN GroupedMatmul residency worker count disagrees with "
                "the launch")
        if any(value < 0 for value in values):
            raise ValueError(
                "CANN GroupedMatmul residency counts cannot be negative")
    tile_classes_by_worker = result["tile_classes_by_worker"]
    if len(tile_classes_by_worker) != work.worker_count:
        raise ValueError(
            "CANN GroupedMatmul tile census worker count disagrees with "
            "the launch")
    if any(
            any(logical_m <= 0 or logical_n <= 0 or count <= 0
                for logical_m, logical_n, count in worker_classes)
            for worker_classes in tile_classes_by_worker):
        raise ValueError("CANN GroupedMatmul tile census is invalid")
    if result["l2_read_bytes"] <= 0 or result["dram_read_bytes"] <= 0:
        raise ValueError(
            "CANN GroupedMatmul residency must produce positive traffic")
    if result["dram_read_bytes"] > result["l2_read_bytes"]:
        raise ValueError(
            "CANN GroupedMatmul DRAM reads cannot exceed L2 ingress reads")
    return _CannGroupedOperandResidency(
        a_hits_by_worker=result["a_hits_by_worker"],
        a_misses_by_worker=result["a_misses_by_worker"],
        b_hits_by_worker=result["b_hits_by_worker"],
        b_misses_by_worker=result["b_misses_by_worker"],
        pair_hit_hit_by_worker=result["pair_hit_hit_by_worker"],
        pair_hit_miss_by_worker=result["pair_hit_miss_by_worker"],
        pair_miss_hit_by_worker=result["pair_miss_hit_by_worker"],
        pair_miss_miss_by_worker=result["pair_miss_miss_by_worker"],
        tile_classes_by_worker=tile_classes_by_worker,
        l2_read_bytes=result["l2_read_bytes"],
        dram_read_bytes=result["dram_read_bytes"],
    )


def _cann_grouped_operand_hit_fractions(
        kernel: KernelSpec,
        residency: _CannGroupedOperandResidency) -> Tuple[float, float]:
    """Recover exact full-stream A/B hit fractions when tiles are complete.

    The native source-order replay includes M/N tails in its aggregate L2 and
    DRAM byte totals, while its legacy per-worker counters intentionally cover
    only full descriptors.  When K and N are both exact tile multiples, every
    B request is represented by those counters.  Subtracting its exact bytes
    from the aggregate totals therefore recovers the complete A stream,
    including M tails, without adding an execution-order assumption.
    """

    _, _, projection_k, projection_n = _projection(kernel)
    _, tile_n, tile_k = _tile(kernel)
    a_total = sum(residency.a_hits_by_worker) + sum(
        residency.a_misses_by_worker)
    b_total = sum(residency.b_hits_by_worker) + sum(
        residency.b_misses_by_worker)
    a_hit_fraction = (
        sum(residency.a_hits_by_worker) / a_total if a_total else 0.0)
    b_hit_fraction = (
        sum(residency.b_hits_by_worker) / b_total if b_total else 0.0)
    if projection_k % tile_k or projection_n % tile_n:
        return a_hit_fraction, b_hit_fraction

    element_bytes = int(kernel.shape["dtype_bytes"])
    b_descriptor_bytes = tile_n * tile_k * element_bytes
    b_l2_read_bytes = b_total * b_descriptor_bytes
    b_dram_read_bytes = (
        sum(residency.b_misses_by_worker) * b_descriptor_bytes)
    a_l2_read_bytes = residency.l2_read_bytes - b_l2_read_bytes
    a_dram_read_bytes = residency.dram_read_bytes - b_dram_read_bytes
    if (a_l2_read_bytes <= 0 or a_dram_read_bytes < 0
            or a_dram_read_bytes > a_l2_read_bytes):
        raise AssertionError(
            "CANN GroupedMatmul source-order operand bytes are inconsistent")
    a_hit_fraction = 1.0 - a_dram_read_bytes / a_l2_read_bytes
    return a_hit_fraction, b_hit_fraction


def _gpu_operand_element_bytes(kernel: KernelSpec) -> int:
    """Return the A/B operand element width of the grouped GEMM.

    Pinned vLLM 0.11.0 ``fused_experts_impl`` quantizes the W1 input and the
    W2 input with ``moe_kernel_quantize_input`` before invoking the
    ``use_fp8_w8a8`` Triton kernel, so an FP8 routed-expert GEMM loads one-byte
    activations and one-byte weights; its output cache stays in the BF16
    compute dtype.  BF16 routed experts, and legacy specs that declare no
    routed quantization contract, load operands in the activation dtype.
    INT4 routed experts use the separate Marlin model.
    """

    activation_bytes = int(kernel.shape["dtype_bytes"])
    if "routed_weight_dtype" not in kernel.attributes:
        return activation_bytes
    routed_weight_dtype = kernel.attributes["routed_weight_dtype"]
    if routed_weight_dtype == "bf16":
        return activation_bytes
    if routed_weight_dtype == "fp8":
        return 1
    raise ValueError(
        "GPU grouped FUSED_MOE operand width requires bf16 or fp8 routed "
        f"weights, got {routed_weight_dtype!r}")


def _gpu_operand_replay_factors(kernel: KernelSpec) -> Tuple[int, int]:
    """Return launch-bound L2-ingress stream multiplicities.

    A value of one is the source-issued base stream.  MetaX BM64's value of
    two is a version-bound, same-invocation non-duration counter observation;
    it is not a cross-tile replay law.  In particular, BM16 contains counter
    outliers that prevent interpreting its value of one as a complete model
    of all physical L2 traffic.  The missing mechanism remains explicit in
    diagnostics instead of being inferred from production duration.
    """

    a_value = kernel.launch.parameters["operand_replay_a"]
    b_value = kernel.launch.parameters["operand_replay_b"]
    if (not float(a_value).is_integer()
            or not float(b_value).is_integer()):
        raise ValueError("FUSED_MOE operand replay counts must be integers")
    a_replay, b_replay = int(a_value), int(b_value)
    if a_replay <= 0 or b_replay <= 0:
        raise ValueError("FUSED_MOE operand replay counts must be positive")
    return a_replay, b_replay


def _gpu_grouped_operand_signature(
        kernel: KernelSpec, active_rows=None) -> MemoryAccessClassSignature:
    """Return the exact tile signature for the combined A+B read stream."""

    tile_m, tile_n, tile_k = _tile(kernel)
    if active_rows is None:
        active_rows = tile_m
    if (isinstance(active_rows, bool)
            or not isinstance(active_rows, int)
            or active_rows <= 0 or active_rows > tile_m):
        raise ValueError(
            "grouped FUSED_MOE active rows must be within the M tile")
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    # The signature keys a measured tile service class.  Every published
    # grouped-operand service was measured on BF16 tiles, so an FP8 operand
    # stream reuses its BF16 tile class as a declared proxy; the bytes it
    # moves are counted at the true operand width in the traffic model.
    element_bytes = int(kernel.shape["dtype_bytes"])
    return MemoryAccessClassSignature(
        access_class=_GROUPED_OPERAND_SERVICE,
        dtype=kernel.dtype,
        tile_m=tile_m,
        tile_k=tile_k,
        logical_row_bytes=(
            active_rows * tile_k + tile_k * tile_n) * element_bytes,
        operand_role="combined",
        tile_n=tile_n,
        group_size_m=int(raw_group_size),
    )


def _gpu_exact_grouped_operand_service(kernel: KernelSpec, hardware):
    """Return the exact combined service, without cross-tile fallback."""

    signature = _gpu_grouped_operand_signature(kernel)
    matches = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _GROUPED_OPERAND_SERVICE
        and service.access_kind == "read"
        and service.ingress_resource_role == "l2_cache"
        and service.signature == signature)
    if len(matches) > 1:
        raise ValueError(
            "grouped FUSED_MOE repeats an exact combined operand service")
    return matches[0] if matches else None


def _nvidia_source_depth_operand_service(kernel: KernelSpec, hardware):
    """Select full-source services for statically bounded launch depths."""

    service_names = (
        _NVIDIA_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE,
        _NVIDIA_EIGHT_EXPERT_132_ACTIVE_136_MAX_W1_SERVICE,
    )
    declared_w1_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in service_names}
    if declared_w1_names and declared_w1_names != set(service_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE source-depth service family must be "
            "complete")
    deep_w2_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_E512_DEEP_W2_SERVICE)
    if len(deep_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its deep W2 source service")
    e128_deep_w2_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_E128_DEEP_W2_SERVICE)
    if len(e128_deep_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E128 deep W2 source service")
    e128_h4096_deep_w1_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service
        == _NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_SERVICE)
    if len(e128_h4096_deep_w1_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E128 H4096 4156/4222 "
            "W1 source service")
    e128_h2048_names = set(
        _NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT.values())
    declared_e128_h2048_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e128_h2048_names}
    if (declared_e128_h2048_names
            and declared_e128_h2048_names != e128_h2048_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E128 H2048 exact source service "
            "family must be complete")
    e8_h6144_deep_w2_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_E8_H6144_DEEP_W2_SERVICE)
    if len(e8_h6144_deep_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E8 H6144 deep W2 "
            "source service")
    e8_h6144_260_active_264_w2_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_E8_H6144_260_ACTIVE_264_MAX_W2_SERVICE)
    if len(e8_h6144_260_active_264_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E8 H6144 260/264 W2 "
            "source service")
    e8_h6144_deep_w1_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_E8_H6144_DEEP_W1_SERVICE)
    if len(e8_h6144_deep_w1_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E8 H6144 deep W1 "
            "source service")
    e8_h4096_exact_w2_names = set(
        _NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY.values())
    declared_e8_h4096_exact_w2_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e8_h4096_exact_w2_names}
    if (declared_e8_h4096_exact_w2_names
            and declared_e8_h4096_exact_w2_names
            != e8_h4096_exact_w2_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E8 H4096 exact W2 source service "
            "family must be complete")
    e512_h4096_deep_w1_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_E512_H4096_DEEP_W1_SERVICE)
    if len(e512_h4096_deep_w1_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E512 H4096 deep W1 "
            "source service")
    e256_h7168_names = {
        _NVIDIA_E256_H7168_DEEP_W1_SERVICE,
        _NVIDIA_E256_H7168_DEEP_W2_SERVICE,
    }
    declared_e256_h7168_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e256_h7168_names}
    if (declared_e256_h7168_names
            and declared_e256_h7168_names != e256_h7168_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E256 H7168 source-depth service "
            "family must be complete")
    e256_h7168_mid_workspace_names = {
        _NVIDIA_E256_H7168_MID_WORKSPACE_W1_SERVICE,
        _NVIDIA_E256_H7168_MID_WORKSPACE_W2_SERVICE,
    }
    declared_e256_h7168_mid_workspace_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e256_h7168_mid_workspace_names}
    if (declared_e256_h7168_mid_workspace_names
            and declared_e256_h7168_mid_workspace_names
            != e256_h7168_mid_workspace_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E256 H7168 mid-workspace source "
            "service family must be complete")
    e256_h7168_2166_w2_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service
        == _NVIDIA_E256_H7168_2166_ACTIVE_2300_MAX_W2_SERVICE)
    if len(e256_h7168_2166_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E256 H7168 2166/2300 "
            "W2 source service")
    e256_h7168_2166_w1_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_E256_H7168_2166_EXACT_W1_SERVICE)
    if len(e256_h7168_2166_w1_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E256 H7168 2166/2300 "
            "W1 source service")
    e8_h6144_short_names = {
        _NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W1_SERVICE,
        _NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W2_SERVICE,
    }
    declared_e8_h6144_short_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e8_h6144_short_names}
    if (declared_e8_h6144_short_names
            and declared_e8_h6144_short_names != e8_h6144_short_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E8 H6144 8/9 source service family "
            "must be complete")
    deep_h2880_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_DEEP_H2880_W1_W2_SERVICE)
    if len(deep_h2880_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its deep H2880 source service")
    e32_h2880_short_names = {
        _NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W1_SERVICE,
        _NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W2_SERVICE,
    }
    declared_e32_h2880_short_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e32_h2880_short_names}
    if (declared_e32_h2880_short_names
            and declared_e32_h2880_short_names != e32_h2880_short_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E32 H2880 short source-depth service "
            "family must be complete")
    e32_h2880_names = {
        _NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W1_SERVICE,
        _NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W2_SERVICE,
    }
    declared_e32_h2880_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e32_h2880_names}
    if (declared_e32_h2880_names
            and declared_e32_h2880_names != e32_h2880_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E32 H2880 source-depth service "
            "family must be complete")
    e512_mid_depth_names = {
        _NVIDIA_E512_1001_ACTIVE_1144_MAX_W1_SERVICE,
        _NVIDIA_E512_1001_ACTIVE_1144_MAX_W2_SERVICE,
    }
    declared_e512_mid_depth_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e512_mid_depth_names}
    if (declared_e512_mid_depth_names
            and declared_e512_mid_depth_names != e512_mid_depth_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E512 source-depth service family "
            "must be complete")
    e128_bm64_names = {
        _NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W1_SERVICE,
        _NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W2_SERVICE,
    }
    declared_e128_bm64_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e128_bm64_names}
    if (declared_e128_bm64_names
            and declared_e128_bm64_names != e128_bm64_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE BM64 E128 source-depth service "
            "family must be complete")
    e128_h2880_1084_names = {
        _NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W1_SERVICE,
        _NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W2_SERVICE,
    }
    declared_e128_h2880_1084_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e128_h2880_1084_names}
    if (declared_e128_h2880_1084_names
            and declared_e128_h2880_1084_names != e128_h2880_1084_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E128 H2880 exact source service "
            "family must be complete")
    bm16_h2880_names = {
        _NVIDIA_BM16_H2880_31_ACTIVE_SERVICE,
        _NVIDIA_BM16_H2880_53_ACTIVE_SERVICE,
    }
    declared_bm16_h2880_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in bm16_h2880_names}
    if (declared_bm16_h2880_names
            and declared_bm16_h2880_names != bm16_h2880_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE BM16 H2880 source-depth service "
            "family must be complete")
    sparse_w2_names = {
        _NVIDIA_BM16_E256_16_ACTIVE_241_MAX_W2_SERVICE,
        _NVIDIA_BM16_E512_20_ACTIVE_482_MAX_W2_SERVICE,
    }
    declared_sparse_w2_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in sparse_w2_names}
    if (declared_sparse_w2_names
            and declared_sparse_w2_names != sparse_w2_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE sparse W2 source-depth service "
            "family must be complete")
    e128_shallow_names = {
        _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE,
        _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE,
    }
    declared_e128_shallow_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e128_shallow_names}
    if (declared_e128_shallow_names
            and declared_e128_shallow_names != e128_shallow_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE shallow E128 source-depth service "
            "family must be complete")
    e128_mid_depth_names = {
        _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W1_SERVICE,
        _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W2_SERVICE,
    }
    declared_e128_mid_depth_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e128_mid_depth_names}
    if (declared_e128_mid_depth_names
            and declared_e128_mid_depth_names != e128_mid_depth_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE mid-depth E128 source service "
            "family must be complete")
    e8_shallow_names = {
        _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W1_SERVICE,
        _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W2_SERVICE,
    }
    declared_e8_shallow_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e8_shallow_names}
    if (declared_e8_shallow_names
            and declared_e8_shallow_names != e8_shallow_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE shallow E8 source service family "
            "must be complete")
    e8_h6144_names = {
        _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W1_SERVICE,
        _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W2_SERVICE,
    }
    declared_e8_h6144_names = {
        service.service
        for service in hardware.memory_access_class_services
        if service.service in e8_h6144_names}
    if (declared_e8_h6144_names
            and declared_e8_h6144_names != e8_h6144_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE H6144 E8 source service family "
            "must be complete")
    if (not declared_w1_names and not deep_w2_services
            and not e128_deep_w2_services
            and not e128_h4096_deep_w1_services
            and not declared_e128_h2048_names
            and not e8_h6144_deep_w2_services
            and not e8_h6144_260_active_264_w2_services
            and not e8_h6144_deep_w1_services
            and not declared_e8_h4096_exact_w2_names
            and not e512_h4096_deep_w1_services
            and not declared_e256_h7168_names
            and not declared_e256_h7168_mid_workspace_names
            and not e256_h7168_2166_w2_services
            and not declared_e8_h6144_short_names
            and not deep_h2880_services
            and not declared_e32_h2880_short_names
            and not declared_e32_h2880_names
            and not declared_e512_mid_depth_names
            and not declared_e128_bm64_names
            and not declared_e128_h2880_1084_names
            and not declared_bm16_h2880_names
            and not declared_sparse_w2_names
            and not declared_e128_shallow_names
            and not declared_e128_mid_depth_names
            and not declared_e8_shallow_names
            and not declared_e8_h6144_names):
        return None
    tile_m, tile_n, tile_k = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    variant, source_rows, projection_k, projection_n = _projection(kernel)
    e32_n_tiles_by_variant = {"w1": 90, "w2": 45}
    e512_service_by_variant = {
        "w1": _NVIDIA_E512_1001_ACTIVE_1144_MAX_W1_SERVICE,
        "w2": _NVIDIA_E512_1001_ACTIVE_1144_MAX_W2_SERVICE,
    }
    experts = int(kernel.shape["experts"])
    _, local_counts = _routing(kernel)
    n_tiles = math.ceil(projection_n / tile_n)
    if kernel.launch.grid_size % n_tiles:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE grid must contain whole M-tile rows")
    m_tile_census = tuple(
        math.ceil(rows / tile_m) if rows > 0 else 0
        for rows in local_counts)
    topology = (
        sum(m_tile_census),
        kernel.launch.grid_size // n_tiles,
    )
    routed_rows = sum(local_counts)
    if kernel.dtype != "bf16":
        return None
    if ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_e8_h6144_names
            and experts == 8
            and topology == (7, 8)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (96, 1024))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (256, 192)))):
        service_name = {
            "w1": _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W2_SERVICE,
        }[variant]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_e8_shallow_names
            and experts == 8
            and topology == (7, 8)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (64, 896))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (224, 128)))):
        service_name = {
            "w1": _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W2_SERVICE,
        }[variant]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_e128_mid_depth_names
            and experts == 128
            and topology == (177, 248)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (32, 48))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (12, 64)))):
        service_name = {
            "w1": _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W2_SERVICE,
        }[variant]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_e128_shallow_names
            and experts == 128
            and topology == (88, 128)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (64, 96))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (24, 128)))):
        service_name = {
            "w1": _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE,
        }[variant]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_sparse_w2_names
            and variant == "w2"
            and (math.ceil(projection_k / tile_k), n_tiles) == (8, 64)
            and (experts, topology) in {
                (256, (16, 241)),
                (512, (20, 482)),
            }):
        service_name = {
            (256, (16, 241)):
                _NVIDIA_BM16_E256_16_ACTIVE_241_MAX_W2_SERVICE,
            (512, (20, 482)):
                _NVIDIA_BM16_E512_20_ACTIVE_482_MAX_W2_SERVICE,
        }[(experts, topology)]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_bm16_h2880_names
            and experts == 128
            and math.ceil(projection_k / tile_k) == 45
            and (
                (variant == "w1" and n_tiles == 180)
                or (variant == "w2" and n_tiles == 90))):
        if ((variant == "w1" and topology == (31, 32))
                or (variant == "w2" and topology == (31, 122))):
            service_name = _NVIDIA_BM16_H2880_31_ACTIVE_SERVICE
        elif topology == (53, 124):
            service_name = _NVIDIA_BM16_H2880_53_ACTIVE_SERVICE
        else:
            return None
    elif ((tile_m, tile_n, tile_k, raw_group_size)
            != (64, 64, 32, 8.0)):
        return None
    elif (declared_e128_bm64_names
            and experts == 128
            and topology == (574, 638)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (64, 24))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (24, 32)))):
        service_name = {
            "w1": _NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W2_SERVICE,
        }[variant]
    elif (declared_e32_h2880_short_names
            and experts == 32
            and math.ceil(projection_k / tile_k) == 90
            and variant in ("w1", "w2")
            and n_tiles == e32_n_tiles_by_variant[variant]
            and topology == (32, 36)):
        service_by_topology = {
            (32, 36): {
                "w1": _NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W1_SERVICE,
                "w2": _NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W2_SERVICE,
            },
        }
        service_name = service_by_topology[topology][variant]
    elif (declared_e32_h2880_names
            and experts == 32
            and math.ceil(projection_k / tile_k) == 90
            and variant in ("w1", "w2")
            and n_tiles == e32_n_tiles_by_variant[variant]
            and topology == (143, 160)):
        service_by_topology = {
            (143, 160): {
                "w1": _NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W1_SERVICE,
                "w2": _NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W2_SERVICE,
            },
        }
        service_name = service_by_topology[topology][variant]
    elif (declared_e512_mid_depth_names
            and experts == 512
            and topology == (1001, 1144)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (128, 32))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (32, 64)))):
        service_name = e512_service_by_variant[variant]
    elif (declared_e128_h2880_1084_names
            and experts == 128
            and topology == (1084, 1150)
            and m_tile_census
            == _NVIDIA_E128_H2880_1084_ACTIVE_1150_M_TILE_CENSUS
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (90, 90))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (90, 45)))):
        service_name = {
            "w1": _NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W1_SERVICE,
            "w2": _NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W2_SERVICE,
        }[variant]
    elif (deep_h2880_services
            and experts in (32, 128)
            and math.ceil(projection_k / tile_k) in (89, 90, 91)
            and (
                (variant == "w1" and n_tiles in (89, 90, 91))
                or (variant == "w2" and n_tiles in (44, 45, 46)))
            and topology[1] > (
                2 * hardware.num_compute_units * int(raw_group_size))):
        service_name = _NVIDIA_DEEP_H2880_W1_W2_SERVICE
    elif (deep_w2_services
            and variant == "w2"
            and experts == 512
            and (math.ceil(projection_k / tile_k), n_tiles) == (32, 64)
            and topology[1] > (
                2 * hardware.num_compute_units * int(raw_group_size))):
        service_name = _NVIDIA_E512_DEEP_W2_SERVICE
    elif (e128_deep_w2_services
            and variant == "w2"
            and experts == 128
            and (math.ceil(projection_k / tile_k), n_tiles) == (48, 64)
            and topology[1] > (
                2 * hardware.num_compute_units * int(raw_group_size))):
        service_name = _NVIDIA_E128_DEEP_W2_SERVICE
    elif (e128_h4096_deep_w1_services
            and variant == "w1"
            and experts == 128
            and source_rows == 32768
            and int(kernel.shape["topk"]) == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (128, 48)
            and topology == (4156, 4222)
            and m_tile_census
            == _NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_M_TILE_CENSUS):
        service_name = _NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_SERVICE
    elif (declared_e128_h2048_names
            and experts == 128
            and int(kernel.shape["topk"]) == 16
            and routed_rows == 524288
            and topology == (8250, 8318)
            and m_tile_census
            == _NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_M_TILE_CENSUS
            and (
                (variant == "w1"
                 and source_rows == 32768
                 and (math.ceil(projection_k / tile_k), n_tiles) == (64, 24))
                or (variant == "w2"
                    and source_rows == routed_rows
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (24, 32)))):
        service_name = (
            _NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT[
                variant])
    elif (declared_e8_h6144_short_names
            and experts == 8
            and topology == (8, 9)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (192, 512))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (512, 96)))):
        service_name = {
            "w1": _NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W1_SERVICE,
            "w2": _NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W2_SERVICE,
        }[variant]
    elif (e8_h6144_deep_w2_services
            and variant == "w2"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (512, 96)
            and topology[1] in (520, 1032)):
        service_name = _NVIDIA_E8_H6144_DEEP_W2_SERVICE
    elif (e8_h6144_260_active_264_w2_services
            and variant == "w2"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (512, 96)
            and topology == (260, 264)):
        service_name = _NVIDIA_E8_H6144_260_ACTIVE_264_MAX_W2_SERVICE
    elif (e8_h6144_deep_w1_services
            and variant == "w1"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (192, 512)
            and topology == (1028, 1032)
            and m_tile_census == _NVIDIA_E8_H6144_DEEP_W1_M_TILE_CENSUS):
        service_name = _NVIDIA_E8_H6144_DEEP_W1_SERVICE
    elif (declared_e8_h4096_exact_w2_names
            and variant == "w2"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (448, 64)
            and topology in _NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY
            and m_tile_census
            == _NVIDIA_E8_H4096_EXACT_W2_M_TILE_CENSUS_BY_TOPOLOGY[
                topology]):
        service_name = _NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY[
            topology]
    elif (e512_h4096_deep_w1_services
            and variant == "w1"
            and experts == 512
            and source_rows == 32768
            and routed_rows == 327680
            and topology == (5368, 5624)
            and m_tile_census == _NVIDIA_E512_H4096_DEEP_W1_M_TILE_CENSUS
            and (math.ceil(projection_k / tile_k), n_tiles) == (128, 32)):
        service_name = _NVIDIA_E512_H4096_DEEP_W1_SERVICE
    elif (declared_e256_h7168_mid_workspace_names
            and experts == 256
            and topology[1] == 1276
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (224, 64))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (64, 112)))):
        service_name = {
            "w1": _NVIDIA_E256_H7168_MID_WORKSPACE_W1_SERVICE,
            "w2": _NVIDIA_E256_H7168_MID_WORKSPACE_W2_SERVICE,
        }[variant]
    elif (e256_h7168_2166_w2_services
            and variant == "w2"
            and experts == 256
            and routed_rows == source_rows == 131072
            and topology == (2166, 2300)
            and (math.ceil(projection_k / tile_k), n_tiles) == (64, 112)):
        service_name = _NVIDIA_E256_H7168_2166_ACTIVE_2300_MAX_W2_SERVICE
    elif (e256_h7168_2166_w1_services
            and variant == "w1"
            and experts == 256
            and routed_rows == 131072
            and topology == (2166, 2300)
            and m_tile_census == _NVIDIA_E256_H7168_2166_W1_M_TILE_CENSUS
            and (math.ceil(projection_k / tile_k), n_tiles) == (224, 64)):
        service_name = _NVIDIA_E256_H7168_2166_EXACT_W1_SERVICE
    elif (declared_e256_h7168_names
            and experts == 256
            and topology == (4230, 4348)
            and m_tile_census == _NVIDIA_E256_H7168_DEEP_M_TILE_CENSUS
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (224, 64))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (64, 112)))):
        service_name = {
            "w1": _NVIDIA_E256_H7168_DEEP_W1_SERVICE,
            "w2": _NVIDIA_E256_H7168_DEEP_W2_SERVICE,
        }[variant]
    elif (declared_w1_names
            and variant == "w1"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (192, 512)
            and topology == (68, 72)):
        service_name = _NVIDIA_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE
    elif (declared_w1_names
            and variant == "w1"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (192, 512)
            and topology == (132, 136)):
        service_name = _NVIDIA_EIGHT_EXPERT_132_ACTIVE_136_MAX_W1_SERVICE
    else:
        return None
    signature = dataclasses.replace(
        _gpu_grouped_operand_signature(kernel), access_class=service_name)
    matches = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == service_name
        and service.access_kind == "read"
        and service.ingress_resource_role == "l2_cache"
        and service.signature == signature)
    if len(matches) != 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE source depth requires one exact "
            f"{service_name!r} signature")
    return matches[0]


def _metax_multi_tile_operand_pipeline_service(
        kernel: KernelSpec, hardware):
    """Select C500's source-shaped expert operand-pipeline envelope.

    The BM16 source clamps its launch storage for fewer than one M tile of
    activation rows. The BM64 source keeps one expert's weight pointer live
    across consecutive M tiles. Synthetic-routing mechanism probes separate
    these source-shaped lifetimes and their long-K versus wide-N pipelines;
    no production duration or case identity participates in the selection.
    """

    service_names = (
        _METAX_GROUPED_OPERAND_LONG_K_SERVICE,
        _METAX_GROUPED_OPERAND_WIDE_N_SERVICE,
        _METAX_GLOBAL_DEEP_OPERAND_LONG_K_SERVICE,
        _METAX_GLOBAL_LARGE_OPERAND_WIDE_N_SERVICE,
        _METAX_LONG_LIFETIME_OPERAND_LONG_K_SERVICE,
        _METAX_LONG_LIFETIME_OPERAND_WIDE_N_SERVICE,
        _METAX_REPEATED_OPERAND_DEEP_K_SERVICE,
        _METAX_REPEATED_OPERAND_WIDE_N_SERVICE,
        _METAX_REPEATED_OPERAND_BALANCED_N_SERVICE,
        _METAX_REPEATED_OPERAND_MID_LONG_K_SERVICE,
        _METAX_REPEATED_OPERAND_MID_WIDE_N_SERVICE,
        _METAX_SINGLE_TILE_OPERAND_LONG_K_SERVICE,
        _METAX_SINGLE_TILE_OPERAND_WIDE_N_SERVICE,
        _METAX_EIGHT_EXPERT_EIGHT_ACTIVE_TWELVE_MAX_W2_SERVICE,
        _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W1_SERVICE,
        _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W2_SERVICE,
        _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W1_SERVICE,
        _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W2_SERVICE,
        _METAX_EIGHT_EXPERT_STEADY_WIDE_N_SERVICE,
        _METAX_EIGHT_EXPERT_STEADY_K192_N512_W1_SERVICE,
        _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE,
        _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W2_SERVICE,
        _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K128_N448_W1_SERVICE,
        _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K448_N64_W2_SERVICE,
        _METAX_E32_32_ACTIVE_48_MAX_W1_SERVICE,
        _METAX_E32_32_ACTIVE_48_MAX_W2_SERVICE,
        _METAX_E32_47_ACTIVE_64_MAX_W1_SERVICE,
        _METAX_E32_47_ACTIVE_64_MAX_W2_SERVICE,
        _METAX_E32_78_ACTIVE_96_MAX_W1_SERVICE,
        _METAX_E32_78_ACTIVE_96_MAX_W2_SERVICE,
        _METAX_E32_143_ACTIVE_160_MAX_W1_SERVICE,
        _METAX_E32_143_ACTIVE_160_MAX_W2_SERVICE,
        _METAX_E256_LONG_LIFETIME_LONG_K_SERVICE,
        _METAX_E256_LONG_LIFETIME_WIDE_N_SERVICE,
        _METAX_E256_LONG_LIFETIME_MID_W1_SERVICE,
        _METAX_E256_LONG_LIFETIME_MID_W2_SERVICE,
        _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W1_SERVICE,
        _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W2_SERVICE,
        _METAX_E512_DENSE_LARGE_W2_SERVICE,
        _METAX_E512_1010_ACTIVE_1144_MAX_W2_SERVICE,
        _METAX_E128_LONG_LIFETIME_W1_SERVICE,
        _METAX_E128_LONG_LIFETIME_W2_SERVICE,
        _METAX_E128_DEEP_W2_SERVICE,
        _METAX_E128_EIGHT_TILE_W1_SERVICE,
        _METAX_E128_EIGHT_TILE_W2_SERVICE,
        _METAX_E128_TOPK16_EIGHT_TILE_W1_SERVICE,
        _METAX_E128_EIGHT_TILE_WIDE_N_W2_SERVICE,
        _METAX_EIGHT_EXPERT_TWELVE_TILE_W2_SERVICE,
        _METAX_EIGHT_EXPERT_TWELVE_TILE_DEEP_W2_SERVICE,
        _METAX_EIGHT_EXPERT_TWELVE_TILE_K128_N448_W1_SERVICE,
        _METAX_EIGHT_EXPERT_TWELVE_TILE_K192_N512_W1_SERVICE,
        _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W1_SERVICE,
        _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W2_SERVICE,
        _METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W1_SERVICE,
        _METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W2_SERVICE,
        _METAX_E128_SPARSE_TWO_TILE_W1_SERVICE,
        _METAX_E128_SPARSE_TWO_TILE_W2_SERVICE,
        _METAX_E128_188_ACTIVE_254_MAX_W1_SERVICE,
        _METAX_E128_188_ACTIVE_254_MAX_W2_SERVICE,
        _METAX_E128_SINGLE_TILE_W1_SERVICE,
        _METAX_E128_SINGLE_TILE_W2_SERVICE,
        _METAX_E128_SINGLE_TILE_MID_W1_SERVICE,
        _METAX_E128_SINGLE_TILE_MID_W2_SERVICE,
        _METAX_E128_SINGLE_TILE_LARGE_W1_SERVICE,
        _METAX_E128_SINGLE_TILE_LARGE_W2_SERVICE,
        _METAX_E128_UPPER_TWO_TILE_W1_SERVICE,
        _METAX_E128_UPPER_TWO_TILE_W2_SERVICE,
        _METAX_E128_TOPK8_128_ACTIVE_190_MAX_W1_SERVICE,
        _METAX_E128_TOPK16_128_ACTIVE_190_MAX_W1_SERVICE,
        _METAX_E128_128_ACTIVE_190_MAX_W2_SERVICE,
        _METAX_E128_REPEATED_SOURCE_TOPK8_W1_SERVICE,
        _METAX_E128_REPEATED_SOURCE_TOPK16_W1_SERVICE,
        _METAX_E128_REPEATED_K24_N32_W2_SERVICE,
        _METAX_E256_EIGHT_TILE_LONG_K_SERVICE,
        _METAX_E256_EIGHT_TILE_WIDE_N_SERVICE,
        _METAX_E256_EIGHT_TILE_MID_W1_SERVICE,
        _METAX_E256_EIGHT_TILE_MID_W2_SERVICE,
        _METAX_E256_FOUR_TILE_LONG_K_SERVICE,
        _METAX_E256_FOUR_TILE_WIDE_N_SERVICE,
        _METAX_E256_FOUR_TILE_MID_LONG_K_SERVICE,
        _METAX_E256_FOUR_TILE_MID_WIDE_N_SERVICE,
        _METAX_E128_FOUR_TILE_W1_SERVICE,
        _METAX_E128_FOUR_TILE_W2_SERVICE,
        _METAX_E128_FOUR_TILE_LARGE_W1_SERVICE,
        _METAX_E128_FOUR_TILE_LARGE_W2_SERVICE,
        _METAX_E256_UPPER_TWO_TILE_LONG_K_SERVICE,
        _METAX_E256_UPPER_TWO_TILE_WIDE_N_SERVICE,
        _METAX_E256_SINGLE_TILE_W1_SERVICE,
        _METAX_E256_SINGLE_TILE_W2_SERVICE,
        _METAX_E256_SINGLE_TILE_MID_DEEP_W1_SERVICE,
        _METAX_E256_SINGLE_TILE_MID_DEEP_W2_SERVICE,
        _METAX_E256_SINGLE_TILE_LARGE_W1_SERVICE,
        _METAX_E256_SINGLE_TILE_LARGE_W2_SERVICE,
        _METAX_E256_SINGLE_TILE_LARGE_DEEP_W1_SERVICE,
        _METAX_E256_SINGLE_TILE_LARGE_DEEP_W2_SERVICE,
        _METAX_E512_SINGLE_TILE_W1_SERVICE,
        _METAX_E512_SINGLE_TILE_W2_SERVICE,
        _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W1_SERVICE,
        _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W2_SERVICE,
        _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W1_SERVICE,
        _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W2_SERVICE,
        _METAX_BM16_ACTIVE_GRID_WIDE_N_SERVICE,
        _METAX_BM16_E256_SIXTEEN_ACTIVE_W2_SERVICE,
        _METAX_BM16_E256_SIXTEEN_ACTIVE_DEEP_W2_SERVICE,
        _METAX_BM16_SPARSE_PADDED_W2_SERVICE,
        _METAX_BM16_E512_SPARSE_PADDED_W2_SERVICE,
        _METAX_BM16_E512_FORTY_ACTIVE_W1_SERVICE,
        _METAX_BM16_E512_FORTY_ACTIVE_W2_SERVICE,
        _METAX_BM16_E512_FORTY_ACTIVE_K8_N64_W2_SERVICE,
        _METAX_BM16_E512_75_ACTIVE_80_MAX_W1_SERVICE,
        _METAX_BM16_E512_75_ACTIVE_485_MAX_W2_SERVICE,
        _METAX_BM16_E512_TWENTY_ACTIVE_W1_SERVICE,
        _METAX_BM16_E512_TWENTY_ACTIVE_W2_SERVICE,
        _METAX_BM16_E512_TWENTY_ACTIVE_K8_N64_W2_SERVICE,
        _METAX_BM16_E512_TEN_ACTIVE_W1_SERVICE,
        _METAX_BM16_E512_TEN_ACTIVE_W2_SERVICE,
        _METAX_BM16_E512_143_ACTIVE_490_MAX_W1_SERVICE,
        _METAX_BM16_E512_143_ACTIVE_490_MAX_W2_SERVICE,
        _METAX_BM16_E512_MID_SPARSE_W1_SERVICE,
        _METAX_BM16_E512_MID_SPARSE_W2_SERVICE,
        _METAX_BM16_E256_SPARSE_COLLIDING_W1_SERVICE,
        _METAX_BM16_E256_SPARSE_COLLIDING_W2_SERVICE,
        _METAX_BM16_E256_106_ACTIVE_248_MAX_W1_SERVICE,
        _METAX_BM16_E256_106_ACTIVE_248_MAX_W2_SERVICE,
        _METAX_BM16_31_ACTIVE_32_MAX_W1_SERVICE,
        _METAX_BM16_E128_31_ACTIVE_W2_SERVICE,
        _METAX_BM16_E128_53_ACTIVE_64_MAX_W1_SERVICE,
        _METAX_BM16_E128_53_ACTIVE_124_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK8_16_ACTIVE_16_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK8_16_ACTIVE_121_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_16_ACTIVE_121_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_16_ACTIVE_16_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_152_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_152_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_64_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_124_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_128_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_128_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_136_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_136_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_184_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_184_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_248_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_248_MAX_W2_SERVICE,
        _METAX_BM16_FOUR_ACTIVE_MID_K_W1_SERVICE,
        _METAX_BM16_FOUR_ACTIVE_MID_K_W2_SERVICE,
        _METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W1_SERVICE,
        _METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W2_SERVICE,
        _METAX_BM16_EIGHT_ACTIVE_MID_K_W2_SERVICE,
        _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W1_SERVICE,
        _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W2_SERVICE,
        _METAX_BM64_E32_32_ACTIVE_36_MAX_W1_SERVICE,
        _METAX_BM64_E32_32_ACTIVE_36_MAX_W2_SERVICE,
        _METAX_BM64_E32_32_ACTIVE_40_MAX_W1_SERVICE,
        _METAX_BM64_E32_32_ACTIVE_40_MAX_W2_SERVICE,
    )
    declared = tuple(
        service for service in hardware.memory_access_class_services
        if service.service in service_names)
    if not declared:
        return None
    tile_m, tile_n, tile_k = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    _, local_counts = _routing(kernel)
    populated_experts = sum(rows > 0 for rows in local_counts)
    routed_rows = sum(local_counts)
    populated_m_tiles = sum(
        math.ceil(rows / tile_m) for rows in local_counts if rows > 0)
    variant, source_rows, projection_k, projection_n = _projection(kernel)
    projection_n_tiles = math.ceil(projection_n / tile_n)
    if kernel.launch.grid_size % projection_n_tiles:
        raise ValueError(
            "MetaX grouped FUSED_MOE grid must contain whole M-tile rows")
    maximum_m_tiles = kernel.launch.grid_size // projection_n_tiles
    if (tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0):
        projection_tile_class = (
            math.ceil(projection_k / tile_k), projection_n_tiles)
        if (source_rows == 16
                and routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 241
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E256_SIXTEEN_ACTIVE_W2_SERVICE
        elif (variant == "w2"
                and int(kernel.shape["experts"]) == 256
                and int(kernel.shape["topk"]) == 8
                and source_rows == routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 241
                and projection_tile_class == (32, 224)):
            service_name = (
                _METAX_BM16_E256_SIXTEEN_ACTIVE_DEEP_W2_SERVICE)
        elif (source_rows == 80
                and populated_experts == populated_m_tiles == 75
                and maximum_m_tiles == 485
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_SPARSE_PADDED_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and int(kernel.shape["topk"]) == 10
                and variant == "w1"
                and source_rows == 4
                and routed_rows == 40
                and populated_experts == populated_m_tiles == 40
                and maximum_m_tiles == 40
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E512_FORTY_ACTIVE_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and int(kernel.shape["topk"]) == 10
                and variant == "w2"
                and source_rows == routed_rows == 40
                and populated_experts == populated_m_tiles == 40
                and maximum_m_tiles == 483
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_FORTY_ACTIVE_K8_N64_W2_SERVICE
        elif (source_rows == 40
                and routed_rows == 40
                and populated_experts == populated_m_tiles == 40
                and maximum_m_tiles == 483
                and projection_tile_class == (16, 128)):
            service_name = _METAX_BM16_E512_FORTY_ACTIVE_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and source_rows == 8
                and routed_rows == 80
                and populated_experts == populated_m_tiles == 75
                and maximum_m_tiles == 80
                and projection_tile_class == (64, 64)):
            service_name = _METAX_BM16_E512_75_ACTIVE_80_MAX_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and source_rows == 80
                and routed_rows == 80
                and populated_experts == populated_m_tiles == 75
                and maximum_m_tiles == 485
                and projection_tile_class == (16, 128)):
            service_name = _METAX_BM16_E512_75_ACTIVE_485_MAX_W2_SERVICE
        elif (source_rows == 2
                and routed_rows == 20
                and populated_experts == populated_m_tiles == 20
                and maximum_m_tiles == 20
                and projection_tile_class == (64, 64)):
            service_name = _METAX_BM16_E512_TWENTY_ACTIVE_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and int(kernel.shape["topk"]) == 10
                and variant == "w2"
                and source_rows == routed_rows == 20
                and populated_experts == populated_m_tiles == 20
                and maximum_m_tiles == 482
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_TWENTY_ACTIVE_K8_N64_W2_SERVICE
        elif (source_rows == 20
                and routed_rows == 20
                and populated_experts == populated_m_tiles == 20
                and maximum_m_tiles == 482
                and projection_tile_class == (16, 128)):
            service_name = _METAX_BM16_E512_TWENTY_ACTIVE_W2_SERVICE
        elif (source_rows == 1
                and routed_rows == 10
                and populated_experts == populated_m_tiles == 10
                and maximum_m_tiles == 10
                and projection_tile_class == (64, 64)):
            service_name = _METAX_BM16_E512_TEN_ACTIVE_W1_SERVICE
        elif (source_rows == 10
                and routed_rows == 10
                and populated_experts == populated_m_tiles == 10
                and maximum_m_tiles == 10
                and projection_tile_class == (16, 128)):
            service_name = _METAX_BM16_E512_TEN_ACTIVE_W2_SERVICE
        elif (source_rows == 16
                and routed_rows == 160
                and populated_experts == populated_m_tiles == 143
                and maximum_m_tiles == 490
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E512_143_ACTIVE_490_MAX_W1_SERVICE
        elif (source_rows == 160
                and routed_rows == 160
                and populated_experts == populated_m_tiles == 143
                and maximum_m_tiles == 490
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_143_ACTIVE_490_MAX_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and variant == "w1"
                and 16 < source_rows <= 64
                and routed_rows == 10 * source_rows
                and populated_experts == populated_m_tiles
                and 143 < populated_m_tiles <= 362
                and 490 < maximum_m_tiles <= 520
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E512_MID_SPARSE_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and variant == "w2"
                and 160 < source_rows <= 640
                and routed_rows == source_rows
                and populated_experts == populated_m_tiles
                and 143 < populated_m_tiles <= 362
                and 490 < maximum_m_tiles <= 520
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_MID_SPARSE_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and variant == "w1"
                and source_rows == 512
                and routed_rows == 5120
                and populated_experts == 512
                and populated_m_tiles == 521
                and maximum_m_tiles == 800
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and variant == "w2"
                and source_rows == routed_rows == 5120
                and populated_experts == 512
                and populated_m_tiles == 521
                and maximum_m_tiles == 800
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W2_SERVICE
        elif (source_rows == 8
                and routed_rows == 64
                and populated_experts == populated_m_tiles == 57
                and maximum_m_tiles == 64
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E256_SPARSE_COLLIDING_W1_SERVICE
        elif (source_rows == 64
                and routed_rows == 64
                and populated_experts == populated_m_tiles == 57
                and maximum_m_tiles == 244
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E256_SPARSE_COLLIDING_W2_SERVICE
        elif (source_rows == 16
                and routed_rows == 128
                and populated_experts == populated_m_tiles == 106
                and maximum_m_tiles == 248
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E256_106_ACTIVE_248_MAX_W1_SERVICE
        elif (source_rows == 128
                and routed_rows == 128
                and populated_experts == populated_m_tiles == 106
                and maximum_m_tiles == 248
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E256_106_ACTIVE_248_MAX_W2_SERVICE
        elif (source_rows < tile_m
                and routed_rows == 32
                and populated_experts == populated_m_tiles == 31
                and maximum_m_tiles == 32
                and projection_tile_class in ((32, 32), (32, 48))):
            service_name = _METAX_BM16_31_ACTIVE_32_MAX_W1_SERVICE
        elif (source_rows == 32
                and routed_rows == 32
                and populated_experts == populated_m_tiles == 31
                and maximum_m_tiles == 122
                and projection_tile_class == (12, 64)):
            service_name = _METAX_BM16_E128_31_ACTIVE_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and source_rows == 8
                and routed_rows == 64
                and populated_experts == populated_m_tiles == 53
                and maximum_m_tiles == 64
                and projection_tile_class == (32, 48)):
            service_name = _METAX_BM16_E128_53_ACTIVE_64_MAX_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and source_rows == 64
                and routed_rows == 64
                and populated_experts == populated_m_tiles == 53
                and maximum_m_tiles == 124
                and projection_tile_class == (12, 64)):
            service_name = _METAX_BM16_E128_53_ACTIVE_124_MAX_W2_SERVICE
        elif (variant == "w1"
                and int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 8
                and source_rows == 2
                and routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 16
                and projection_tile_class == (32, 48)):
            service_name = _METAX_BM16_E128_TOPK8_16_ACTIVE_16_MAX_W1_SERVICE
        elif (variant == "w2"
                and int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 8
                and source_rows == routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 121
                and projection_tile_class == (12, 64)):
            service_name = (
                _METAX_BM16_E128_TOPK8_16_ACTIVE_121_MAX_W2_SERVICE)
        elif (variant == "w2"
                and int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 16
                and source_rows == routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 121
                and projection_tile_class == (12, 64)):
            service_name = (
                _METAX_BM16_E128_TOPK16_16_ACTIVE_121_MAX_W2_SERVICE)
        elif (variant == "w1"
                and int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 16
                and source_rows == 1
                and routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 16
                and projection_tile_class == (32, 48)):
            service_name = _METAX_BM16_E128_TOPK16_16_ACTIVE_16_MAX_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 16
                and int(kernel.shape["tokens"]) in (4, 8, 16, 32, 64, 128)
                and projection_tile_class in ((32, 48), (12, 64))):
            source_depth_services = {
                (4, 64, (32, 48)):
                    _METAX_BM16_E128_TOPK16_64_MAX_W1_SERVICE,
                (4, 124, (12, 64)):
                    _METAX_BM16_E128_TOPK16_124_MAX_W2_SERVICE,
                (8, 128, (32, 48)):
                    _METAX_BM16_E128_TOPK16_128_MAX_W1_SERVICE,
                (8, 128, (12, 64)):
                    _METAX_BM16_E128_TOPK16_128_MAX_W2_SERVICE,
                (16, 136, (32, 48)):
                    _METAX_BM16_E128_TOPK16_136_MAX_W1_SERVICE,
                (16, 136, (12, 64)):
                    _METAX_BM16_E128_TOPK16_136_MAX_W2_SERVICE,
                (32, 152, (32, 48)):
                    _METAX_BM16_E128_TOPK16_152_MAX_W1_SERVICE,
                (32, 152, (12, 64)):
                    _METAX_BM16_E128_TOPK16_152_MAX_W2_SERVICE,
                (64, 184, (32, 48)):
                    _METAX_BM16_E128_TOPK16_184_MAX_W1_SERVICE,
                (64, 184, (12, 64)):
                    _METAX_BM16_E128_TOPK16_184_MAX_W2_SERVICE,
                (128, 248, (32, 48)):
                    _METAX_BM16_E128_TOPK16_248_MAX_W1_SERVICE,
                (128, 248, (12, 64)):
                    _METAX_BM16_E128_TOPK16_248_MAX_W2_SERVICE,
            }
            source_depth_key = (
                int(kernel.shape["tokens"]),
                maximum_m_tiles,
                projection_tile_class,
            )
            if source_depth_key not in source_depth_services:
                return None
            active_tile_bounds = {
                4: (45, 58),
                8: (80, 92),
                16: (104, 120),
                32: (120, 132),
                64: (120, 136),
                128: (168, 190),
            }
            lower_active_tiles, upper_active_tiles = active_tile_bounds[
                int(kernel.shape["tokens"])]
            if not (lower_active_tiles <= populated_m_tiles
                    <= upper_active_tiles):
                return None
            service_name = source_depth_services[source_depth_key]
        elif (source_rows == 1
                and routed_rows == 4
                and populated_experts == populated_m_tiles == 4
                and maximum_m_tiles == 4
                and projection_tile_class == (45, 180)):
            service_name = _METAX_BM16_FOUR_ACTIVE_MID_K_W1_SERVICE
        elif (source_rows == 4
                and routed_rows == 4
                and populated_experts == populated_m_tiles == 4
                and maximum_m_tiles == 4
                and projection_tile_class == (45, 90)):
            service_name = _METAX_BM16_FOUR_ACTIVE_MID_K_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and source_rows == 2
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 8
                and maximum_m_tiles == 8
                and projection_tile_class == (45, 180)):
            service_name = _METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and source_rows == 8
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 8
                and maximum_m_tiles == 8
                and projection_tile_class == (45, 90)):
            service_name = _METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W2_SERVICE
        elif (source_rows == 8
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 8
                and maximum_m_tiles == 8
                and projection_tile_class == (12, 64)):
            service_name = _METAX_BM16_EIGHT_ACTIVE_MID_K_W2_SERVICE
        elif (source_rows == 2
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 7
                and maximum_m_tiles == 8
                and projection_tile_class == (45, 180)):
            service_name = _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W1_SERVICE
        elif (source_rows == 8
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 7
                and maximum_m_tiles == 8
                and projection_tile_class == (45, 90)):
            service_name = _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W2_SERVICE
        elif (source_rows == 32
                and populated_experts == populated_m_tiles == 31
                and maximum_m_tiles == 242
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_SPARSE_PADDED_W2_SERVICE
        elif (variant == "w2"
                and source_rows < tile_m
                and populated_m_tiles == populated_experts == routed_rows
                and projection_k <= projection_n
                and projection_tile_class[0] == 8):
            service_name = _METAX_BM16_ACTIVE_GRID_WIDE_N_SERVICE
        else:
            service_name = None
        if service_name is None:
            return None
    elif (tile_m, tile_n, tile_k, raw_group_size) != (64, 64, 32, 8.0):
        return None
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles == 12
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (448, 64)):
        service_name = (
            _METAX_EIGHT_EXPERT_EIGHT_ACTIVE_TWELVE_MAX_W2_SERVICE)
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles <= 10
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (192, 512)):
        service_name = _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W1_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles <= 10
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (512, 96)):
        service_name = _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W2_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles <= 9
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (128, 448)):
        service_name = _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W1_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles <= 9
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (448, 64)):
        service_name = _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W2_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and routed_rows >= 2 * tile_m
            and (projection_k, projection_n) == (14336, 4096)):
        service_name = _METAX_SINGLE_TILE_OPERAND_LONG_K_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and routed_rows >= 2 * tile_m
            and (projection_k, projection_n) == (4096, 28672)):
        service_name = _METAX_SINGLE_TILE_OPERAND_WIDE_N_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 21
            and maximum_m_tiles == 24
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (128, 448)):
        service_name = _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W1_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 21
            and maximum_m_tiles == 24
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (448, 64)):
        service_name = _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W2_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 20
            and maximum_m_tiles == 24
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (192, 512)):
        service_name = (
            _METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W1_SERVICE)
    elif (populated_experts == 8
            and populated_m_tiles == 20
            and maximum_m_tiles == 24
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (512, 96)):
        service_name = (
            _METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W2_SERVICE)
    elif populated_experts == 8 and populated_m_tiles == 12:
        projection_tile_class = (
            math.ceil(projection_k / tile_k),
            math.ceil(projection_n / tile_n),
        )
        service_name = (
            _METAX_EIGHT_EXPERT_TWELVE_TILE_K128_N448_W1_SERVICE
            if projection_tile_class == (128, 448)
            else _METAX_EIGHT_EXPERT_TWELVE_TILE_K192_N512_W1_SERVICE
            if projection_tile_class == (192, 512)
            else _METAX_EIGHT_EXPERT_TWELVE_TILE_W2_SERVICE
            if projection_tile_class == (448, 64)
            else _METAX_EIGHT_EXPERT_TWELVE_TILE_DEEP_W2_SERVICE
            if projection_tile_class == (512, 96)
            else None)
        if service_name is None:
            return None
    elif (populated_experts == populated_m_tiles == 32
            and maximum_m_tiles == 48
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_E32_32_ACTIVE_48_MAX_W1_SERVICE
    elif (populated_experts == populated_m_tiles == 32
            and maximum_m_tiles == 48
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_E32_32_ACTIVE_48_MAX_W2_SERVICE
    elif (int(kernel.shape["experts"]) == 32
            and populated_experts == populated_m_tiles == 32
            and routed_rows == 256
            and maximum_m_tiles == 36
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_BM64_E32_32_ACTIVE_36_MAX_W1_SERVICE
    elif (int(kernel.shape["experts"]) == 32
            and populated_experts == populated_m_tiles == 32
            and routed_rows == 256
            and maximum_m_tiles == 36
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_BM64_E32_32_ACTIVE_36_MAX_W2_SERVICE
    elif (int(kernel.shape["experts"]) == 32
            and variant == "w1"
            and source_rows == 128
            and routed_rows == 512
            and populated_experts == populated_m_tiles == 32
            and maximum_m_tiles == 40
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_BM64_E32_32_ACTIVE_40_MAX_W1_SERVICE
    elif (int(kernel.shape["experts"]) == 32
            and variant == "w2"
            and source_rows == routed_rows == 512
            and populated_experts == populated_m_tiles == 32
            and maximum_m_tiles == 40
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_BM64_E32_32_ACTIVE_40_MAX_W2_SERVICE
    elif (int(kernel.shape["experts"]) == 512
            and source_rows == routed_rows == 40960
            and populated_experts == 512
            and populated_m_tiles == 1010
            and maximum_m_tiles == 1144
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (16, 32)):
        service_name = _METAX_E512_1010_ACTIVE_1144_MAX_W2_SERVICE
    elif (int(kernel.shape["experts"]) == 512
            and variant == "w1"
            and source_rows == 1024
            and routed_rows == 10240
            and populated_experts == populated_m_tiles == 512
            and maximum_m_tiles == 664
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (64, 16)):
        service_name = _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W1_SERVICE
    elif (int(kernel.shape["experts"]) == 512
            and variant == "w2"
            and source_rows == routed_rows == 10240
            and populated_experts == populated_m_tiles == 512
            and maximum_m_tiles == 664
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (16, 32)):
        service_name = _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W2_SERVICE
    elif populated_m_tiles <= populated_experts:
        projection_tile_class = (
            math.ceil(projection_k / tile_k),
            math.ceil(projection_n / tile_n),
        )
        if (populated_experts == populated_m_tiles == 128
                and routed_rows == 4096
                and maximum_m_tiles == 190
                and source_rows == 512
                and int(kernel.shape["topk"]) == 8
                and projection_tile_class == (64, 24)):
            service_name = _METAX_E128_TOPK8_128_ACTIVE_190_MAX_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and routed_rows == 4096
                and maximum_m_tiles == 190
                and source_rows == 256
                and int(kernel.shape["topk"]) == 16
                and projection_tile_class == (64, 24)):
            service_name = _METAX_E128_TOPK16_128_ACTIVE_190_MAX_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and routed_rows == source_rows == 4096
                and maximum_m_tiles == 190
                and projection_tile_class == (24, 32)):
            service_name = _METAX_E128_128_ACTIVE_190_MAX_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles <= 158
                and projection_tile_class == (64, 24)):
            service_name = _METAX_E128_SINGLE_TILE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles <= 158
                and projection_tile_class == (24, 32)):
            service_name = _METAX_E128_SINGLE_TILE_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles == 190
                and projection_tile_class == (90, 90)):
            service_name = _METAX_E128_SINGLE_TILE_MID_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles == 190
                and projection_tile_class == (90, 45)):
            service_name = _METAX_E128_SINGLE_TILE_MID_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles == 158
                and projection_tile_class == (128, 48)):
            service_name = _METAX_E128_SINGLE_TILE_LARGE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles == 158
                and projection_tile_class == (48, 64)):
            service_name = _METAX_E128_SINGLE_TILE_LARGE_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 316
                and projection_tile_class == (64, 16)):
            service_name = _METAX_E256_SINGLE_TILE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 316
                and projection_tile_class == (16, 32)):
            service_name = _METAX_E256_SINGLE_TILE_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 380
                and projection_tile_class == (64, 16)):
            service_name = _METAX_E256_SINGLE_TILE_MID_DEEP_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 380
                and projection_tile_class == (16, 32)):
            service_name = _METAX_E256_SINGLE_TILE_MID_DEEP_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 316
                and projection_tile_class == (224, 64)):
            service_name = _METAX_E256_SINGLE_TILE_LARGE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 316
                and projection_tile_class == (64, 112)):
            service_name = _METAX_E256_SINGLE_TILE_LARGE_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 380
                and projection_tile_class == (224, 64)):
            service_name = _METAX_E256_SINGLE_TILE_LARGE_DEEP_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 380
                and projection_tile_class == (64, 112)):
            service_name = _METAX_E256_SINGLE_TILE_LARGE_DEEP_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 512
                and maximum_m_tiles == 824
                and projection_tile_class == (128, 32)):
            service_name = _METAX_E512_SINGLE_TILE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 512
                and maximum_m_tiles == 824
                and projection_tile_class == (32, 64)):
            service_name = _METAX_E512_SINGLE_TILE_W2_SERVICE
        else:
            return None
    elif (populated_experts >= 4 * hardware.num_compute_units
            and populated_m_tiles <= 2 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 32)):
        service_name = _METAX_GLOBAL_DEEP_OPERAND_LONG_K_SERVICE
    elif (populated_experts >= 4 * hardware.num_compute_units
            and populated_m_tiles <= 2 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (32, 64)):
        service_name = _METAX_GLOBAL_LARGE_OPERAND_WIDE_N_SERVICE
    elif (populated_experts >= 4 * hardware.num_compute_units
            and populated_m_tiles <= 2 * populated_experts):
        service_name = (
            _METAX_GROUPED_OPERAND_LONG_K_SERVICE
            if projection_k > projection_n
            else _METAX_GROUPED_OPERAND_WIDE_N_SERVICE)
    elif (populated_experts == 512
            and routed_rows == 327680
            and 10 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and maximum_m_tiles == 5624
            and source_rows == 32768
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (64, 16)):
        service_name = _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W1_SERVICE
    elif (populated_experts == 512
            and routed_rows == 327680
            and 10 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and maximum_m_tiles == 5624
            and source_rows == 327680
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (16, 32)):
        service_name = _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W2_SERVICE
    elif (populated_experts == 512
            and routed_rows == 327680
            and 10 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and maximum_m_tiles == 5624
            and source_rows == 327680
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (32, 64)):
        service_name = _METAX_E512_DENSE_LARGE_W2_SERVICE
    elif (populated_experts >= 4 * hardware.num_compute_units
            and populated_m_tiles <= 11 * populated_experts):
        service_name = (
            _METAX_LONG_LIFETIME_OPERAND_LONG_K_SERVICE
            if projection_k > projection_n
            else _METAX_LONG_LIFETIME_OPERAND_WIDE_N_SERVICE)
    elif (populated_experts == 128
            and routed_rows == 16384
            and 315 <= populated_m_tiles <= 319
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 24)
            and source_rows == 2048
            and int(kernel.shape["topk"]) == 8):
        service_name = _METAX_E128_REPEATED_SOURCE_TOPK8_W1_SERVICE
    elif (populated_experts == 128
            and routed_rows == 16384
            and 315 <= populated_m_tiles <= 319
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 24)
            and source_rows == 1024
            and int(kernel.shape["topk"]) == 16):
        service_name = _METAX_E128_REPEATED_SOURCE_TOPK16_W1_SERVICE
    elif (populated_experts == 128
            and routed_rows == source_rows == 16384
            and 315 <= populated_m_tiles <= 319
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (24, 32)):
        service_name = _METAX_E128_REPEATED_K24_N32_W2_SERVICE
    elif (populated_experts == 256
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 16)):
        service_name = _METAX_E256_LONG_LIFETIME_MID_W1_SERVICE
    elif (populated_experts == 256
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (16, 32)):
        service_name = _METAX_E256_LONG_LIFETIME_MID_W2_SERVICE
    elif (populated_experts == 256
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (224, 64)):
        service_name = _METAX_E256_LONG_LIFETIME_LONG_K_SERVICE
    elif (populated_experts == 256
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 112)):
        service_name = _METAX_E256_LONG_LIFETIME_WIDE_N_SERVICE
    elif (populated_experts == 256
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 16)):
        service_name = _METAX_E256_EIGHT_TILE_MID_W1_SERVICE
    elif (populated_experts == 256
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (16, 32)):
        service_name = _METAX_E256_EIGHT_TILE_MID_W2_SERVICE
    elif (populated_experts == 256
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (224, 64)):
        service_name = _METAX_E256_EIGHT_TILE_LONG_K_SERVICE
    elif (populated_experts == 256
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 112)):
        service_name = _METAX_E256_EIGHT_TILE_WIDE_N_SERVICE
    elif (populated_experts == 256
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 16)):
        service_name = _METAX_E256_FOUR_TILE_MID_LONG_K_SERVICE
    elif (populated_experts == 256
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (16, 32)):
        service_name = _METAX_E256_FOUR_TILE_MID_WIDE_N_SERVICE
    elif (populated_experts == 256
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (224, 64)):
        service_name = _METAX_E256_FOUR_TILE_LONG_K_SERVICE
    elif (populated_experts == 256
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 112)):
        service_name = _METAX_E256_FOUR_TILE_WIDE_N_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 90)):
        service_name = _METAX_E128_LONG_LIFETIME_W1_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 45)):
        service_name = _METAX_E128_LONG_LIFETIME_W2_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_DEEP_W2_SERVICE
    elif (populated_experts == 128
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and int(kernel.shape["topk"]) == 16
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 24)):
        service_name = _METAX_E128_TOPK16_EIGHT_TILE_W1_SERVICE
    elif (populated_experts == 128
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 90)):
        service_name = _METAX_E128_EIGHT_TILE_W1_SERVICE
    elif (populated_experts == 128
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 45)):
        service_name = _METAX_E128_EIGHT_TILE_W2_SERVICE
    elif (populated_experts == 128
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_EIGHT_TILE_WIDE_N_W2_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles == 573
            and maximum_m_tiles == 638
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 48)):
        service_name = _METAX_E128_FOUR_TILE_LARGE_W1_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles == 573
            and maximum_m_tiles == 638
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_FOUR_TILE_LARGE_W2_SERVICE
    elif (populated_experts == 128
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 90)):
        service_name = _METAX_E128_FOUR_TILE_W1_SERVICE
    elif (populated_experts == 128
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 45)):
        service_name = _METAX_E128_FOUR_TILE_W2_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles == 188
            and maximum_m_tiles == 254
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 48)):
        service_name = _METAX_E128_188_ACTIVE_254_MAX_W1_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles == 188
            and maximum_m_tiles == 254
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_188_ACTIVE_254_MAX_W2_SERVICE
    elif (populated_experts == 128
            and populated_experts < populated_m_tiles
            and populated_m_tiles <= 2 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 48)):
        service_name = _METAX_E128_SPARSE_TWO_TILE_W1_SERVICE
    elif (populated_experts == 128
            and populated_experts < populated_m_tiles
            and populated_m_tiles <= 2 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_SPARSE_TWO_TILE_W2_SERVICE
    elif (populated_experts == 128
            and 2 * populated_experts < populated_m_tiles
            and 2 * populated_m_tiles <= 5 * populated_experts
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 48)):
        service_name = _METAX_E128_UPPER_TWO_TILE_W1_SERVICE
    elif (populated_experts == 128
            and 2 * populated_experts < populated_m_tiles
            and 2 * populated_m_tiles <= 5 * populated_experts
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_UPPER_TWO_TILE_W2_SERVICE
    elif (populated_experts == 256
            and 2 * populated_experts < populated_m_tiles
            and 2 * populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (224, 64)):
        service_name = _METAX_E256_UPPER_TWO_TILE_LONG_K_SERVICE
    elif (populated_experts == 256
            and 2 * populated_experts < populated_m_tiles
            and 2 * populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 112)):
        service_name = _METAX_E256_UPPER_TWO_TILE_WIDE_N_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and projection_k > projection_n
            and math.ceil(projection_k / tile_k) == 64
            and math.ceil(projection_n / tile_n) == 16):
        service_name = _METAX_REPEATED_OPERAND_MID_LONG_K_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and projection_k <= projection_n
            and math.ceil(projection_k / tile_k) == 16
            and math.ceil(projection_n / tile_n) == 32):
        service_name = _METAX_REPEATED_OPERAND_MID_WIDE_N_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and projection_k > projection_n
            and math.ceil(projection_k / tile_k) >= 220):
        service_name = _METAX_REPEATED_OPERAND_DEEP_K_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 45)):
        service_name = _METAX_REPEATED_OPERAND_BALANCED_N_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and projection_k <= projection_n
            and math.ceil(projection_k / tile_k) >= 62):
        service_name = _METAX_REPEATED_OPERAND_WIDE_N_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 68
            and maximum_m_tiles == 72
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (192, 512)):
        service_name = _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 68
            and maximum_m_tiles == 72
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (512, 96)):
        service_name = _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W2_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 68
            and maximum_m_tiles == 72
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (128, 448)):
        service_name = (
            _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K128_N448_W1_SERVICE)
    elif (populated_experts == 8
            and populated_m_tiles == 68
            and maximum_m_tiles == 72
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (448, 64)):
        service_name = (
            _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K448_N64_W2_SERVICE)
    elif (populated_experts == 32
            and populated_m_tiles == 47
            and maximum_m_tiles == 64
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_E32_47_ACTIVE_64_MAX_W1_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 47
            and maximum_m_tiles == 64
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_E32_47_ACTIVE_64_MAX_W2_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 78
            and maximum_m_tiles == 96
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_E32_78_ACTIVE_96_MAX_W1_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 78
            and maximum_m_tiles == 96
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_E32_78_ACTIVE_96_MAX_W2_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 143
            and maximum_m_tiles == 160
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_E32_143_ACTIVE_160_MAX_W1_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 143
            and maximum_m_tiles == 160
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_E32_143_ACTIVE_160_MAX_W2_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (192, 512)):
        service_name = _METAX_EIGHT_EXPERT_STEADY_K192_N512_W1_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles > 11 * populated_experts
            and projection_k <= projection_n
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) in ((128, 448), (192, 512))):
        service_name = _METAX_EIGHT_EXPERT_STEADY_WIDE_N_SERVICE
    else:
        return None
    signature = dataclasses.replace(
        _gpu_grouped_operand_signature(kernel), access_class=service_name)
    matches = tuple(
        service for service in declared
        if service.service == service_name
        and service.access_kind == "read"
        and service.ingress_resource_role == "l2_cache"
        and service.signature == signature)
    if len(matches) != 1:
        raise ValueError(
            "MetaX multi-tile grouped operand pipeline requires one exact "
            f"{service_name!r} signature")
    return matches[0]


def _gpu_split_operand_signature(
        kernel: KernelSpec, service: str,
        operand_role: str) -> MemoryAccessClassSignature:
    """Return one exact BM64 hit/miss service signature."""

    combined = _gpu_grouped_operand_signature(kernel)
    if operand_role == "combined":
        logical_row_bytes = combined.logical_row_bytes
    elif operand_role == "weight":
        _, tile_n, tile_k = _tile(kernel)
        logical_row_bytes = (
            tile_k * tile_n * int(kernel.shape["dtype_bytes"]))
    else:
        raise ValueError(
            "GPU split operand role must be combined or weight")
    return dataclasses.replace(
        combined,
        access_class=service,
        logical_row_bytes=logical_row_bytes,
        operand_role=operand_role,
    )


def _gpu_split_operand_services(kernel: KernelSpec, hardware):
    """Resolve the complete exact BM64 L2-hit/weight-miss pair."""

    tile_m, _, _ = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    group_size_m = int(raw_group_size)
    _, source_rows, projection_k, _ = _projection(kernel)
    source_footprint_bytes = (
        source_rows * projection_k * _gpu_operand_element_bytes(kernel))
    _, local_counts = _routing(kernel)
    expert_m_tiles = tuple(
        math.ceil(rows / tile_m) for rows in local_counts if rows > 0)
    overlaps_weight_miss = (
        source_footprint_bytes <= hardware.l2_size_bytes
        and bool(expert_m_tiles)
        and all(2 <= tiles <= group_size_m for tiles in expert_m_tiles))
    weight_service = (
        _GROUPED_WEIGHT_OVERLAPPED_DRAM_MISS_SERVICE
        if overlaps_weight_miss
        else _GROUPED_OPERAND_PIPELINE_SERVICE)
    specifications = (
        (_GROUPED_OPERAND_PIPELINE_SERVICE, "combined"),
        (weight_service, "weight"),
    )
    signatures = tuple(
        _gpu_split_operand_signature(kernel, service, operand_role)
        for service, operand_role in specifications)
    matches = tuple(
        tuple(
            candidate
            for candidate in hardware.memory_access_class_services
            if candidate.service == service
            and candidate.access_kind == "read"
            and candidate.ingress_resource_role == "l2_cache"
            and candidate.signature == signature)
        for (service, _), signature in zip(specifications, signatures))
    if not any(matches):
        return None
    if any(len(values) != 1 for values in matches):
        raise ValueError(
            "GPU grouped operand L2-hit/weight-miss services must be "
            "complete and exact")
    return {
        operand_role: values[0]
        for (_, operand_role), values in zip(specifications, matches)
    }


def _gpu_full_group_operand_services(kernel: KernelSpec, hardware):
    """Resolve exact complete-GSM8 services by weight-stream topology."""

    combined = _gpu_split_operand_signature(
        kernel, _GROUPED_OPERAND_PIPELINE_SERVICE, "combined")
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    group_size_m = int(raw_group_size)
    services = {}
    for weight_streams in (
            value for value in _GROUPED_OPERAND_STREAM_COUNTS
                if value <= group_size_m):
        signature = dataclasses.replace(
            combined,
            logical_row_bytes=(
                    combined.logical_row_bytes * group_size_m),
            weight_streams_per_group=weight_streams,
        )
        matches = tuple(
            candidate
            for candidate in hardware.memory_access_class_services
            if candidate.service == _GROUPED_OPERAND_PIPELINE_SERVICE
            and candidate.access_kind == "read"
            and candidate.ingress_resource_role == "l2_cache"
            and candidate.signature == signature)
        if len(matches) > 1:
            raise ValueError(
                "GPU grouped operand repeats an exact full-group service")
        if matches:
            services[weight_streams] = matches[0]
    variant, source_rows, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    exact_five_three_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_BM64_GSM8_FIVE_THREE_W1_SERVICE)
    if len(exact_five_three_services) > 1:
        raise ValueError(
            "GPU grouped operand repeats its exact 5+3 service")
    if exact_five_three_services:
        if not (
                variant == "w1"
                and int(kernel.shape["experts"]) == 256
                and int(kernel.shape["topk"]) == 8
                and source_rows >= 8192
                and (projection_k, projection_n) == (7168, 4096)
                and (tile_m, tile_n, tile_k, group_size_m) == (64, 64, 32, 8)):
            return services
        signature = dataclasses.replace(
            combined,
            access_class=_NVIDIA_BM64_GSM8_FIVE_THREE_W1_SERVICE,
            logical_row_bytes=combined.logical_row_bytes * group_size_m,
            weight_streams_per_group=2,
        )
        matches = tuple(
            service for service in exact_five_three_services
            if service.access_kind == "read"
            and service.ingress_resource_role == "l2_cache"
            and service.signature == signature)
        if len(matches) != 1:
            raise ValueError(
                "GPU grouped operand exact 5+3 service requires one "
                "matching signature")
        services[(5, 3)] = matches[0]
    if services and 1 not in services:
        raise ValueError(
            "GPU grouped operand mixed-stream services require the "
            "single-stream service")
    return services


def _gpu_full_group_operand_bytes_by_weight_streams(
        kernel: KernelSpec, *, projection_k: int,
        projection_n: int) -> dict:
    """Return complete-GSM8 operand bytes by distinct expert streams.

    Triton's source order flattens populated expert M tiles and forms fixed
    groups of ``GROUP_SIZE_M``.  The compact sequence of expert M-tile runs
    inside each full group carries its ordered weight-stream topology.
    Counting runs and tail-row padding is O(experts), without CTA expansion.
    """

    tile_m, tile_n, _ = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    group_size_m = int(raw_group_size)
    _, local_counts = _routing(kernel)
    populated_m_tiles = 0
    tile_runs_by_group = {}
    tail_padding_by_group = {}
    repeated_expert_boundary_groups = set()
    for count in local_counts:
        if count <= 0:
            continue
        expert_start = populated_m_tiles
        expert_m_tiles = math.ceil(count / tile_m)
        populated_m_tiles += expert_m_tiles
        first_group = expert_start // group_size_m
        last_group = (populated_m_tiles - 1) // group_size_m
        if first_group not in tile_runs_by_group:
            tile_runs_by_group[first_group] = []
        if first_group == last_group:
            tile_runs_by_group[first_group].append(expert_m_tiles)
        else:
            first_run = group_size_m - expert_start % group_size_m
            tile_runs_by_group[first_group].append(first_run)
            last_run = populated_m_tiles % group_size_m
            if last_run:
                if last_group not in tile_runs_by_group:
                    tile_runs_by_group[last_group] = []
                tile_runs_by_group[last_group].append(last_run)
        if expert_m_tiles > group_size_m + 1:
            repeated_expert_boundary_groups.update((
                expert_start // group_size_m,
                (populated_m_tiles - 1) // group_size_m,
            ))
        tail_rows = count % tile_m
        if tail_rows:
            group_index = (populated_m_tiles - 1) // group_size_m
            if group_index not in tail_padding_by_group:
                tail_padding_by_group[group_index] = 0
            tail_padding_by_group[group_index] += tile_m - tail_rows
    full_group_count = populated_m_tiles // group_size_m
    if full_group_count == 0:
        return {}
    group_count_by_topology = {1: 0}
    group_rows_by_topology = {1: 0}
    classified_topology_by_group = {}
    for group_index in range(full_group_count):
        tile_runs = tuple(
            tile_runs_by_group[group_index]
            if group_index in tile_runs_by_group else (group_size_m,))
        if sum(tile_runs) != group_size_m:
            raise AssertionError(
                "grouped FUSED_MOE full source group does not conserve M "
                "tiles")
        # A boundary can land exactly at the end of this GSM group.  The
        # compact tile-run sequence, unlike the old partial-boundary counter,
        # retains that source-order fact (e.g. the second run in 5+3).
        streams = len(tile_runs)
        if streams > group_size_m:
            raise ValueError(
                "grouped FUSED_MOE source group has too many weight streams")
        # The mixed-group source probes use a fixed fanout: every distinct B
        # stream owns the same number of M tiles in a complete GSM group.  A
        # stream count alone is not enough: e.g. 3+2+2+1 has four streams but
        # is not the measured 2+2+2+2 program.  Do not apply its service to
        # an uneven source topology.
        expected_run = group_size_m // streams
        is_uniform_exact_fanout = (
            streams in _GROUPED_OPERAND_STREAM_COUNTS
            and len(tile_runs) == streams
            and all(run == expected_run for run in tile_runs))
        topology = (
            streams if is_uniform_exact_fanout
            else (5, 3) if tile_runs == (5, 3) else None)
        if (topology is None
                or (streams > 1
                    and group_index in repeated_expert_boundary_groups)):
            continue
        classified_topology_by_group[group_index] = topology
        if topology not in group_count_by_topology:
            group_count_by_topology[topology] = 0
            group_rows_by_topology[topology] = 0
        group_count_by_topology[topology] += 1
        group_rows_by_topology[topology] += group_size_m * tile_m
    for group_index, padding_rows in tail_padding_by_group.items():
        if group_index not in classified_topology_by_group:
            continue
        topology = classified_topology_by_group[group_index]
        group_rows_by_topology[topology] -= padding_rows
    n_tiles = math.ceil(projection_n / tile_n)
    element_bytes = _gpu_operand_element_bytes(kernel)
    activation_replay, weight_replay = _gpu_operand_replay_factors(kernel)
    return {
        topology: float(
            n_tiles * projection_k * element_bytes * (
                group_rows * activation_replay
                + group_count_by_topology[topology]
                * group_size_m * tile_n * weight_replay))
        for topology, group_rows in group_rows_by_topology.items()
        if group_count_by_topology[topology] > 0
    }




def _gpu_grouped_operand_service_by_active_rows(kernel, hardware):
    """Return a complete exact masked-row service family when declared."""

    tile_m, tile_n, tile_k = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    matching = tuple(
        service
        for service in hardware.memory_access_class_services
        if service.service == _GROUPED_OPERAND_SERVICE
        and service.access_kind == "read"
        and service.ingress_resource_role == "l2_cache"
        and service.signature.dtype == kernel.dtype
        and service.signature.tile_m == tile_m
        and service.signature.tile_n == tile_n
        and service.signature.tile_k == tile_k
        and service.signature.operand_role == "combined"
        and service.signature.group_size_m == int(raw_group_size))
    if not matching:
        raise ValueError(
            "grouped FUSED_MOE has no exact hardware signature family")
    by_rows = {}
    for active_rows in range(1, tile_m + 1):
        signature = _gpu_grouped_operand_signature(
            kernel, active_rows=active_rows)
        matches = tuple(
            service for service in matching
            if service.signature == signature)
        if len(matches) > 1:
            raise ValueError(
                "grouped FUSED_MOE repeats an exact masked-row service")
        if matches:
            by_rows[active_rows] = matches[0]
    if tile_m not in by_rows:
        raise ValueError(
            "grouped FUSED_MOE service family lacks its full-row signature")
    if len(by_rows) not in (1, tile_m):
        raise ValueError(
            "grouped FUSED_MOE masked-row service family must be complete")
    replay_fractions = {
        service.replay_to_dram_fraction for service in by_rows.values()}
    if len(replay_fractions) != 1:
        raise ValueError(
            "grouped FUSED_MOE masked-row services disagree on replay")
    return by_rows


def _cann_grouped_operand_signature(
        kernel: KernelSpec, service: str,
        operand_role: str) -> MemoryAccessClassSignature:
    """Return one exact full-tile AIC operand signature."""

    tile_m, tile_n, tile_k = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    if operand_role == "activation":
        logical_row_bytes = tile_m * tile_k * element_bytes
    elif operand_role == "weight":
        logical_row_bytes = tile_k * tile_n * element_bytes
    else:
        raise ValueError(
            "CANN GroupedMatmul operand role must be activation or weight")
    return MemoryAccessClassSignature(
        access_class=service,
        dtype=kernel.dtype,
        tile_m=tile_m,
        tile_k=tile_k,
        logical_row_bytes=logical_row_bytes,
        operand_role=operand_role,
        tile_n=tile_n,
        # This identifies one source descriptor service unit; it is not the
        # unused GPU launch ``group_size_m`` parameter.
        group_size_m=1,
    )


def _cann_grouped_operand_services(kernel: KernelSpec, hardware):
    """Return each complete source-invariant A/B capability pair.

    The A descriptor contains ``baseM`` and therefore remains bound to the
    captured tile-M.  The B descriptor is always ``baseK x baseN`` in the
    versioned CANN source and capture, so its measured service is reusable
    when the launch selector changes only ``baseM``.  Keep an unavailable A
    pair on the generic MTE2 byte-capacity path instead of disabling the B
    service and the complete analytical tile pipeline.
    """

    role, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    parameters = kernel.launch.parameters
    b_l2_hit_service = _CANN_OPERAND_B_L2_HIT_SERVICE
    b_dram_miss_service = _CANN_OPERAND_B_DRAM_MISS_SERVICE
    b_program_names = ("cann_depth_b1", "cann_step_n", "cann_step_kb")
    if all(name in parameters for name in b_program_names):
        b_program = tuple(parameters[name] for name in b_program_names)
        if (all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and float(value).is_integer()
                and value > 0
                for value in b_program)
                and hardware.has_memory_access_class_service(
                    _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE)
                and kernel.dtype == "bf16"
                and projection_k == 2880
                and (tile_n, tile_k) == (256, 64)
                and kernel.launch.grid_size == 24
                and int(parameters["cann_depth_b1"]) == (
                    2 * int(parameters["cann_step_n"])
                    * int(parameters["cann_step_kb"]))):
            b_l2_hit_service = (
                _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE)
    # CANN binds the B GlobalTensor before it selects the B L2 hint, and
    # binds that same 64x256 descriptor into Matmul after the hint.  Hence
    # the independent strided B ingress service applies to every DRAM miss
    # of these exact W2 programs; the source-order residency trace alone
    # decides whether a request is such a miss.  Unlike the K6144 service
    # below, this selection must not conflate the descriptor geometry with
    # the all-single-M condition that disables B caching.
    # The B descriptor is source-identical for every captured K2880 W2
    # program.  Select its service from that descriptor itself, rather than
    # maintaining a second base-M -> program table beside launch selection.
    # In particular M64 and M80 share this descriptor while their A streams
    # differ only in the amount of scheduled work.
    k2880_w2_b_descriptor_program = {
        "cann_depth_b1": 8,
        "cann_step_kb": 4,
        "cann_step_n": 1,
        "cann_db_l0b": 2,
    }
    _, local_counts = _routing(kernel)
    nonempty_local_rows = tuple(rows for rows in local_counts if rows > 0)
    if all(name in parameters for name in k2880_w2_b_descriptor_program):
        source_b_descriptor_matches = all(
            not isinstance(parameters[name], bool)
            and isinstance(parameters[name], (int, float))
            and float(parameters[name]).is_integer()
            and int(parameters[name]) == expected
            for name, expected in k2880_w2_b_descriptor_program.items())
        if (source_b_descriptor_matches
                and hardware.has_memory_access_class_service(
                    _CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE)
                and kernel.dtype == "bf16"
                and role == "w2"
                and projection_k == projection_n == 2880
                and kernel.launch.grid_size == 24):
            b_dram_miss_service = (
                _CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE)
    expected_source_program = cann_grouped_matmul_k_program(
        tile_m, sum(local_counts))
    source_k6144_m1_program_matches = (
        expected_source_program is not None
        and all(
            name in parameters
            for name in CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES)
        and all(
            not isinstance(parameters[name], bool)
            and isinstance(parameters[name], (int, float))
            and float(parameters[name]).is_integer()
            and int(parameters[name]) == expected
            for name, expected in zip(
                CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES,
                expected_source_program)))
    if (source_k6144_m1_program_matches
            and hardware.has_memory_access_class_service(
                _CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE)
            and kernel.dtype == "bf16"
            and role == "w1"
            and projection_k == 6144
            and kernel.launch.grid_size == 24
            and nonempty_local_rows
            and all(rows <= tile_m for rows in nonempty_local_rows)):
        b_dram_miss_service = (
            _CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE)
    service_names = (
        _CANN_OPERAND_A_L2_HIT_SERVICE,
        _CANN_OPERAND_A_DRAM_MISS_SERVICE,
        b_l2_hit_service,
        b_dram_miss_service,
    )
    roles = ("activation", "activation", "weight", "weight")
    signatures = tuple(
        _cann_grouped_operand_signature(kernel, name, role)
        for name, role in zip(service_names, roles))
    matches = []
    for name, role, signature in zip(service_names, roles, signatures):
        values = []
        for candidate in hardware.memory_access_class_services:
            if (candidate.service != name
                    or candidate.access_kind != "read"
                    or candidate.ingress_resource_role != "l2_cache"):
                continue
            candidate_signature = candidate.signature
            if role == "weight":
                candidate_signature = dataclasses.replace(
                    candidate_signature, tile_m=signature.tile_m)
            if candidate_signature == signature:
                values.append(candidate)
        matches.append(tuple(values))
    matches = tuple(matches)
    if not any(matches):
        return None
    for operand_role, values in (
            ("activation", matches[:2]), ("weight", matches[2:])):
        if any(values) and any(len(item) != 1 for item in values):
            raise ValueError(
                "CANN GroupedMatmul operand residency service pair for "
                f"{operand_role} must be complete and unique")
    return {
        name: values[0]
        for name, values in zip(service_names, matches)
        if values
    }


def _apply_gpu_grouped_operand_downstream_replay(
        traffic: _GpuGroupedGemmTraffic,
        kernel: KernelSpec,
        hardware) -> _GpuGroupedGemmTraffic:
    """Map each operand's non-compulsory L2 replay to downstream reads."""

    service = _gpu_exact_grouped_operand_service(kernel, hardware)
    if service is None:
        return traffic
    if service.replay_to_dram_fraction is None:
        return traffic
    for role, l2_bytes, compulsory_bytes in (
            (
                "activation",
                traffic.activation_l2_read_bytes,
                traffic.activation_compulsory_read_bytes,
            ),
            (
                "weight",
                traffic.weight_l2_read_bytes,
                traffic.weight_compulsory_read_bytes,
            )):
        if compulsory_bytes < 0 or compulsory_bytes > l2_bytes:
            raise ValueError(
                f"grouped FUSED_MOE {role} compulsory bytes must be within "
                "its L2 request bytes")
    if (traffic.routed_weight_compulsory_read_bytes < 0
            or traffic.routed_weight_compulsory_read_bytes
            > traffic.routed_weight_l2_read_bytes):
        raise ValueError(
            "grouped FUSED_MOE routed-weight compulsory bytes must be within "
            "its L2 request bytes")
    if not math.isclose(
            traffic.routed_weight_dram_read_bytes,
            traffic.routed_weight_compulsory_read_bytes,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "grouped FUSED_MOE routed-weight DRAM bytes must retain only "
            "the unique source footprint")
    replay = service.replay_to_dram_fraction
    if replay is None:
        raise AssertionError("validated grouped operand replay is absent")
    return dataclasses.replace(
        traffic,
        activation_dram_read_bytes=(
            traffic.activation_compulsory_read_bytes
            + replay * (
                traffic.activation_l2_read_bytes
                - traffic.activation_compulsory_read_bytes)),
        weight_dram_read_bytes=(
            traffic.weight_compulsory_read_bytes
            + replay * (
                traffic.weight_l2_read_bytes
                - traffic.weight_compulsory_read_bytes)),
    )


def _gpu_grouped_weight_capacity_floor(
        kernel: KernelSpec, hardware, *, projection_k: int,
        projection_n: int, tile_m: int,
        local_counts: Tuple[int, ...], weight_l2_bytes: float,
        weight_compulsory_bytes: float) -> float:
    """Return the source-grouped lower bound on downstream weight reads.

    Triton's grouped-MoE program order keeps ``GROUP_SIZE_M`` consecutive M
    tiles in one source group.  An expert whose M tiles cross a group boundary
    can retain its weight matrix only while the distinct expert matrices in
    that group fit in L2.  When that footprint exceeds L2, the next group must
    stream the continuing expert again.  Count those group-boundary reloads
    from the routing histogram and matrix dimensions without expanding N/K
    CTAs or assuming a physical SM execution order.

    This floor is enabled only for the exact BM64 split hit/miss service.  Its
    independent probe establishes the two-slot source pipeline and strided
    weight-miss service; other GPU tile signatures retain the compulsory
    lower bound.
    """

    if (_gpu_split_operand_services(kernel, hardware) is None
            or int(kernel.attributes["expert_parallel_size"]) != 1):
        return weight_compulsory_bytes
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    group_size_m = int(raw_group_size)
    weight_matrix_bytes = (
        projection_k * projection_n * _gpu_operand_element_bytes(kernel))
    if weight_matrix_bytes <= 0 or hardware.l2_size_bytes <= 0:
        raise ValueError(
            "grouped FUSED_MOE weight residency requires positive capacity")

    spans = []
    populated_m_tiles = 0
    for count in local_counts:
        if count <= 0:
            continue
        expert_m_tiles = math.ceil(count / tile_m)
        first_group = populated_m_tiles // group_size_m
        last_group = (
            populated_m_tiles + expert_m_tiles - 1) // group_size_m
        spans.append((first_group, last_group))
        populated_m_tiles += expert_m_tiles
    if not spans:
        return weight_compulsory_bytes

    group_count = math.ceil(populated_m_tiles / group_size_m)
    group_expert_delta = [0] * (group_count + 1)
    for first_group, last_group in spans:
        group_expert_delta[first_group] += 1
        group_expert_delta[last_group + 1] -= 1
    oversize_group_prefix = [0]
    group_experts = 0
    for group_index in range(group_count):
        group_experts += group_expert_delta[group_index]
        if group_experts <= 0:
            raise ValueError(
                "grouped FUSED_MOE populated M group has no expert")
        oversize_group_prefix.append(
            oversize_group_prefix[-1]
            + int(group_experts * weight_matrix_bytes
                  > hardware.l2_size_bytes))

    weight_matrix_loads = len(spans)
    for first_group, last_group in spans:
        # The first group is already represented by the compulsory matrix.
        # Every later over-capacity group forces one source-ordered reload.
        weight_matrix_loads += (
            oversize_group_prefix[last_group + 1]
            - oversize_group_prefix[first_group + 1])
    capacity_floor = float(weight_matrix_loads * weight_matrix_bytes)
    if capacity_floor < weight_compulsory_bytes:
        raise AssertionError(
            "grouped FUSED_MOE capacity floor lost compulsory weights")
    return min(weight_l2_bytes, capacity_floor)


def _gpu_grouped_tile_program_traffic(
        kernel: KernelSpec, hardware, *, projection_k: int,
        projection_n: int, tile_m: int, tile_n: int, tile_k: int,
        n_tiles: int, m_tiles: int, local_counts: Tuple[int, ...],
        element_bytes: int, output_element_bytes: int, output_rows: int,
        routed_rows: int):
    """Replay the grouped-MoE weight stream through the shared miss core.

    The launch retains the group width its source swizzle uses, so the
    program-to-tile order is a launch fact and the routing histogram states
    which rows each M tile owns.  That is the whole access sequence; the
    replacement decision behind it is the same core the dense family uses.
    A launch that retains no group width keeps the previous envelope.

    The weight stream is always priced here.  The activation stream depends
    on the projection: W2 consumes the post-activation routed buffer, where
    every local route is its own row, so an M tile's rows are a distinct
    physical object and the replay is exact.  W1 consumes source tokens, and
    the routing histogram does not retain which routes came from one token,
    so two routes of a token would be counted as two objects; that projection
    keeps its source-unique bound instead.
    """

    group_size_m = int(kernel.launch.parameters["group_size_m"])
    if group_size_m <= 0 or not any(local_counts):
        return None
    traffic = simulate_moe_tile_program(
        m_tiles=m_tiles, n_tiles=n_tiles, group_size_m=group_size_m,
        k_extent=projection_k, block_k=tile_k,
        n_extent=projection_n, block_n=tile_n,
        element_bytes=element_bytes,
        c_row_bytes=tile_n * output_element_bytes, block_m=tile_m,
        wave_size=min(m_tiles * n_tiles, int(hardware.num_compute_units)),
        tokens_per_expert=local_counts,
        capacity_bytes=int(hardware.l2_replacement_capacity_bytes),
        residency_profile=hardware.l2_replacement_residency_profile)
    return traffic


def _gpu_traffic_components(
        kernel: KernelSpec, hardware) -> _GpuGroupedGemmTraffic:
    """Triton grouped-GEMM operand L2 and compulsory DRAM traffic.

    The launch grid covers the maximum routing workspace, while the kernel
    returns before loading operands for M tiles beyond
    ``num_tokens_post_padded``.  Within a populated tile, ``token_mask`` also
    suppresses activation loads and output stores for padding rows.  Traffic
    therefore follows semantic routed/source rows, not the max-EM launch
    extent or a fully populated BM tile.
    """

    if hardware.l2_size_bytes <= 0:
        raise ValueError("GPU FUSED_MOE requires positive L2 capacity")
    variant, source_rows, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    global_counts, local_counts = _routing(kernel)
    experts = int(kernel.shape["experts"])
    grid_size = kernel.launch.grid_size
    n_tiles = math.ceil(projection_n / tile_n)
    if grid_size % n_tiles:
        raise ValueError(
            "FUSED_MOE launch grid must be divisible by N-tile count")
    m_tiles = grid_size // n_tiles
    if m_tiles <= 0:
        raise ValueError("FUSED_MOE launch grid does not contain an M tile")
    configured_rows = int(kernel.launch.parameters["padded_rows"])
    configured_m_tiles = math.ceil(configured_rows / tile_m)
    if m_tiles != configured_m_tiles:
        raise ValueError(
            "FUSED_MOE launch grid disagrees with configured padded_rows")
    if tile_k <= 0:
        raise ValueError(f"FUSED_MOE tile_k must be positive, got {tile_k}")
    element_bytes = _gpu_operand_element_bytes(kernel)
    output_element_bytes = int(kernel.shape["dtype_bytes"])

    global_routed_rows = sum(global_counts)
    expected_routed_rows = (
        int(kernel.shape["tokens"]) * int(kernel.shape["topk"]))
    if global_routed_rows != expected_routed_rows:
        raise ValueError(
            "FUSED_MOE routing count sum disagrees with tokens * topk")
    global_populated_m_tiles = sum(
        math.ceil(count / tile_m)
        for count in global_counts if count > 0)
    if global_populated_m_tiles <= 0:
        raise ValueError(
            "FUSED_MOE routing must contain at least one populated M tile")
    if global_populated_m_tiles > m_tiles:
        raise ValueError(
            "FUSED_MOE populated M tiles exceed the configured launch grid")
    routed_rows = sum(local_counts)
    populated_m_tiles = sum(
        math.ceil(count / tile_m)
        for count in local_counts if count > 0)
    active_experts = sum(1 for count in local_counts if count > 0)

    # A is reloaded for each N tile, but token_mask admits only real routed
    # rows.  Every populated pid_m x pid_n program constructs and loads its
    # own B pointer stream.  The N coordinate is modulo N and has no N mask,
    # so a tail pid_n still issues the complete tile_n width (with wrapped
    # addresses).  C is masked in both M and N and therefore keeps only real
    # output columns.  With expert parallelism, pinned vLLM maps foreign
    # expert blocks to expert id -1 and explicitly writes zeros for their
    # valid routed rows before the cross-rank output reduction.  Each rank
    # therefore writes every global routed output row even though only its
    # locally owned rows load A/B and execute tensor operations.
    activation_l2 = (
        routed_rows * projection_k * n_tiles * element_bytes)
    expert_parallel_size = int(kernel.attributes["expert_parallel_size"])
    output_rows = (
        global_routed_rows if expert_parallel_size > 1 else routed_rows)
    output_l2 = output_rows * projection_n * output_element_bytes
    if not (tile_m <= 16 or tile_m >= 32):
        raise ValueError(f"unsupported fused-MoE M tile: {tile_m}")
    weight_l2 = (
        populated_m_tiles * projection_k * n_tiles * tile_n * element_bytes)

    # These are requests at the L2 ingress, so an L2 hit does not erase the A
    # request.  Independent NVIDIA NCU observations and MetaX layer-1 byte
    # counters both retain the complete valid-row A base stream and one C
    # stream; compiler-specific operand replay counts differ by launch profile.
    activation_replay, weight_replay = _gpu_operand_replay_factors(kernel)
    activation_l2 *= activation_replay
    weight_l2 *= weight_replay
    if variant == "w1":
        # The routing histogram does not retain source-token identities.  A
        # W1 input token contributes to at most ``topk`` local routes, so this
        # is the source-derived lower bound on unique rank-local input rows.
        local_source_rows = min(
            source_rows,
            math.ceil(routed_rows / int(kernel.shape["topk"])),
        )
    elif variant == "w2":
        # W2 consumes the post-activation routed buffer.  Every local route is
        # a distinct row even when several routes came from one source token.
        local_source_rows = routed_rows
    else:
        raise AssertionError(f"unexpected FUSED_MOE variant: {variant!r}")
    activation_compulsory = float(
        local_source_rows * projection_k * element_bytes)
    weight_compulsory = float(
        active_experts * projection_k * projection_n * element_bytes)
    routed_weight_l2, routed_weight_compulsory = (
        _source_routed_weight_read_traffic(
            kernel, routed_rows=float(routed_rows), n_tiles=n_tiles))
    output_compulsory = float(
        output_rows * projection_n * output_element_bytes)
    # Triton defines logical program groups but not physical AP/SM assignment.
    # Retain the uniquely addressed compulsory bytes generally.  For the
    # independently established BM64 two-slot service, additionally account
    # for source-group boundaries whose expert-weight footprint exceeds L2;
    # this remains a compact capacity floor, not a CTA-order cache replay.
    replay = _gpu_grouped_tile_program_traffic(
        kernel, hardware,
        projection_k=projection_k, projection_n=projection_n,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        n_tiles=n_tiles, m_tiles=m_tiles, local_counts=local_counts,
        element_bytes=element_bytes,
        output_element_bytes=output_element_bytes, output_rows=output_rows,
        routed_rows=routed_rows)
    activation_dram = activation_compulsory
    if replay is None:
        weight_dram = _gpu_grouped_weight_capacity_floor(
            kernel,
            hardware,
            projection_k=projection_k,
            projection_n=projection_n,
            tile_m=tile_m,
            local_counts=local_counts,
            weight_l2_bytes=float(weight_l2),
            weight_compulsory_bytes=weight_compulsory,
        )
    else:
        weight_dram = float(replay.b_dram_bytes)
        if variant == "w2":
            activation_dram = float(replay.a_dram_bytes)
    traffic = _GpuGroupedGemmTraffic(
        activation_l2_read_bytes=float(activation_l2),
        weight_l2_read_bytes=float(weight_l2),
        routed_weight_l2_read_bytes=routed_weight_l2,
        output_l2_write_bytes=float(output_l2),
        activation_compulsory_read_bytes=activation_compulsory,
        weight_compulsory_read_bytes=weight_compulsory,
        routed_weight_compulsory_read_bytes=routed_weight_compulsory,
        activation_dram_read_bytes=activation_dram,
        weight_dram_read_bytes=weight_dram,
        routed_weight_dram_read_bytes=routed_weight_compulsory,
        output_dram_write_bytes=output_compulsory,
    )
    traffic = _apply_gpu_grouped_operand_downstream_replay(
        traffic, kernel, hardware)
    validated_l2_miss_rate(
        traffic.dram_total_bytes, traffic.l2_total_bytes)
    return traffic


def _gpu_traffic(kernel: KernelSpec, hardware) -> Tuple[float, float, float]:
    """Return the legacy aggregate view of GPU grouped-GEMM traffic."""

    traffic = _gpu_traffic_components(kernel, hardware)
    return (
        traffic.l2_total_bytes,
        traffic.dram_total_bytes,
        traffic.compulsory_bytes,
    )


def _gpu_grouped_gemm_pipeline_envelope(
        kernel: KernelSpec, hardware,
        traffic: _GpuGroupedGemmTraffic
        ) -> Optional[GpuGroupedGemmPipelineEnvelopeDemand]:
    """Return the source-bound BM16 resident-wave K pipeline, when available."""

    mechanism = hardware.grouped_gemm_tile_pipeline
    if mechanism is None:
        return None
    tile_m, tile_n, tile_k = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    signature = (
        kernel.dtype,
        tile_m,
        tile_n,
        tile_k,
        int(raw_group_size),
        kernel.launch.block_size,
    )
    expected = (
        mechanism["dtype"],
        mechanism["tile_m"],
        mechanism["tile_n"],
        mechanism["tile_k"],
        mechanism["group_size_m"],
        mechanism["threads_per_cta"],
    )
    if signature != expected:
        return None
    variant, _, projection_k, projection_n = _projection(kernel)
    resident_ctas_by_variant = mechanism[
        "resident_ctas_per_sm_by_variant"]
    if set(resident_ctas_by_variant) != {"w1", "w2"}:
        raise ValueError(
            "GPU grouped-GEMM pipeline must declare W1/W2 residency")
    resident_ctas_per_sm = resident_ctas_by_variant[variant]
    if (isinstance(resident_ctas_per_sm, bool)
            or not isinstance(resident_ctas_per_sm, int)
            or resident_ctas_per_sm <= 0):
        raise ValueError(
            "GPU grouped-GEMM variant residency must be a positive integer")
    # Triton masks the final K iteration.  The aggregate traffic above uses
    # its logical, valid-K bytes, but this fixed-round pipeline envelope was
    # only independently established for complete K tiles.  Do not reject a
    # valid source launch or pretend that its partial round has the measured
    # complete-tile cadence.
    if projection_k % tile_k:
        return None
    _, local_counts = _routing(kernel)
    populated_m_tiles = sum(
        math.ceil(rows / tile_m) for rows in local_counts if rows > 0)
    active_ctas = populated_m_tiles * math.ceil(projection_n / tile_n)
    if active_ctas == 0:
        if traffic.dram_read_bytes != 0.0:
            raise ValueError(
                "GPU grouped-GEMM has DRAM reads without an active CTA")
        return None
    return GpuGroupedGemmPipelineEnvelopeDemand(
        active_ctas=active_ctas,
        k_tiles=math.ceil(projection_k / tile_k),
        resident_ctas_per_sm=resident_ctas_per_sm,
        dram_read_bytes=traffic.dram_read_bytes,
    )


def _metax_grouped_operand_pipeline_envelope(
        kernel: KernelSpec, hardware,
        traffic: _GpuGroupedGemmTraffic
        ) -> Optional[MetaxGroupedOperandPipelineDemand]:
    """Return the C500 per-CTA serial grouped-operand-pipeline demand.

    One resident CTA per AP drains its expert's operand stream round by
    round; a synthetic-routing mechanism probe over non-production K/N
    geometry separates the fixed per-CTA span from the per-round service for
    one exact compiled tile signature, and the DRAM/latency round service is
    priced from independent hardware facts.  Each deployed tile signature
    needs its own measurement: BM16/BN64/BK128 and BM32/BN64/BK128 have
    different compiled resources, so a launch whose signature has no
    measured mechanism keeps the generic memory model rather than borrowing
    another signature's coefficients.  No production duration or case
    identity participates in any mechanism.
    """

    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    tile_m, tile_n, tile_k = _tile(kernel)
    launch_parameters = kernel.launch.parameters
    # Tile shape and block threads do not identify the compiled program on
    # their own: BM64/BN128/BK64 compiles to a four-warp and an eight-warp
    # variant with the same tile, threads and dynamic shared memory.  A launch
    # that does not record its compiled-program facts cannot be matched to a
    # measured mechanism at all.
    if not {"num_warps", "num_stages"}.issubset(launch_parameters):
        return None
    signature = (
        kernel.dtype,
        tile_m,
        tile_n,
        tile_k,
        int(raw_group_size),
        kernel.launch.block_size,
        int(launch_parameters["num_warps"]),
        int(launch_parameters["num_stages"]),
    )
    mechanism = None
    mechanism_key = None
    for candidate in METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS:
        candidate_mechanism = getattr(hardware, candidate)
        if candidate_mechanism is None:
            continue
        expected = (
            candidate_mechanism["dtype"],
            candidate_mechanism["tile_m"],
            candidate_mechanism["tile_n"],
            candidate_mechanism["tile_k"],
            candidate_mechanism["group_size_m"],
            candidate_mechanism["threads_per_cta"],
            candidate_mechanism["num_warps"],
            candidate_mechanism["num_stages"],
        )
        if signature == expected:
            mechanism = candidate_mechanism
            mechanism_key = candidate
            break
    if mechanism_key is None:
        return None
    variant, _, projection_k, projection_n = _projection(kernel)
    # The per-round service was only independently established for complete
    # K tiles; do not pretend a masked partial round has the measured
    # complete-tile cadence.
    if projection_k % tile_k:
        return None
    _, local_counts = _routing(kernel)
    # Each mechanism declares how many M tiles one populated expert may span
    # inside its measured domain.  Beyond that bound several M tiles share one
    # expert weight stream, which the sweep did not establish; keep the
    # generic memory model instead of extrapolating the measured cadence.
    maximum_expert_m_tiles = mechanism["maximum_expert_m_tiles"]
    if maximum_expert_m_tiles is not None:
        for rows in local_counts:
            if rows > 0 and math.ceil(rows / tile_m) > maximum_expert_m_tiles:
                return None
    populated_m_tiles = sum(
        math.ceil(rows / tile_m) for rows in local_counts if rows > 0)
    active_ctas = populated_m_tiles * math.ceil(projection_n / tile_n)
    if active_ctas == 0:
        if traffic.dram_read_bytes != 0.0:
            raise ValueError(
                "MetaX grouped-GEMM has DRAM reads without an active CTA")
        return None
    return MetaxGroupedOperandPipelineDemand(
        active_ctas=active_ctas,
        k_tiles=projection_k // tile_k,
        dram_read_bytes=traffic.dram_read_bytes,
        mechanism_key=mechanism_key,
    )


def _cann_grouped_matmul_traffic(
        kernel: KernelSpec,
        work: _CannGroupedMatmulWork,
        residency: Optional[_CannGroupedOperandResidency] = None
        ) -> _CannGroupedMatmulTraffic:
    """Return the CANN 8.5.1 AIC ingress and output byte contract.

    The version-bound launch-tile census predicts one activation slab per
    output-N tile and one N-tail-clipped weight slab per populated expert M
    tile.  Canonical non-duration counters validate that relation against
    GM-to-L1; the logical C output is separately validated against the AIC
    write interface.  Neither interface proves per-invocation DRAM residency,
    descriptor timing, or a FixPipe lifecycle.  No target duration or
    per-shape traffic table is consumed here.
    """

    _, _, projection_k, projection_n = _projection(kernel)
    _, local_counts = _routing(kernel)
    active_experts = sum(1 for count in local_counts if count > 0)
    element_bytes = int(kernel.shape["dtype_bytes"])
    per_expert_windows = kernel.attributes["expert_weight_tile_rows"]
    selected_window = float(kernel.launch.parameters["weight_tile_rows"])
    if per_expert_windows is not None or selected_window != 0.0:
        raise ValueError(
            "NPU FUSED_MOE cannot consume traffic-derived residency windows "
            "without source-backed descriptor semantics")
    if work.routed_rows != sum(local_counts):
        raise ValueError(
            "CANN GroupedMatmul traffic and Tensor work disagree on routed rows")
    if work.populated_m_tiles < active_experts or work.n_tiles <= 0:
        raise ValueError("CANN GroupedMatmul produced an invalid tile census")

    activation_compulsory = (
        work.routed_rows * projection_k * element_bytes)
    output = work.routed_rows * projection_n * element_bytes
    l2_read = (
        activation_compulsory * work.n_tiles
        + work.populated_m_tiles * projection_k * projection_n
        * element_bytes)
    if residency is None:
        # CANN publishes the tensor ABI and the MTE2 ingress census but not a
        # cross-AIC cache replacement schedule.  A tile-by-tile LRU replay
        # would make that unobserved schedule part of Forward.  Use the
        # semantic unique operand footprint as the downstream analytical
        # lower bound, retaining the complete source-visible L2 ingress
        # stream above.  This is the same resource-envelope treatment used by
        # the GPU grouped projection when an execution-order contract is not
        # available.
        dram_read = float(
            activation_compulsory
            + active_experts * projection_k * projection_n * element_bytes)
    else:
        if residency.l2_read_bytes != l2_read:
            raise ValueError(
                "CANN GroupedMatmul LRU and analytical L2 traffic disagree: "
                f"lru={residency.l2_read_bytes}, analytical={l2_read}")
        dram_read = float(residency.dram_read_bytes)
    traffic = _CannGroupedMatmulTraffic(
        activation_l2_read_bytes=float(
            activation_compulsory * work.n_tiles),
        weight_l2_read_bytes=float(
            work.populated_m_tiles * projection_k * projection_n
            * element_bytes),
        output_l2_write_bytes=float(output),
        dram_read_bytes=dram_read,
        dram_write_bytes=float(output),
    )
    return traffic





def _cann_grouped_operand_service_paths_from_paths(
        memory_paths: Tuple[MemoryPathDemand, ...], kernel: KernelSpec,
        residency: _CannGroupedOperandResidency,
        hardware) -> Tuple[MemoryPathDemand, ...]:
    """Annotate one completed CANN path tuple with operand services."""

    if not isinstance(memory_paths, tuple):
        raise TypeError("CANN grouped operand memory paths must be a tuple")
    services = _cann_grouped_operand_services(kernel, hardware)
    if services is None:
        return memory_paths
    read_path, = tuple(
        path for path in memory_paths if path.name == "mte2_read")
    if read_path.access_class_demands:
        raise ValueError("CANN grouped operand path already has a typed service")
    b_l2_hit_service = (
        _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE
        if _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE in services
        else _CANN_OPERAND_B_L2_HIT_SERVICE)
    b_dram_miss_service = next(
        service_name for service_name in (
            _CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE,
            _CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE,
            _CANN_OPERAND_B_DRAM_MISS_SERVICE)
        if service_name in services)
    service_counts = {
        _CANN_OPERAND_A_L2_HIT_SERVICE:
            sum(residency.a_hits_by_worker),
        _CANN_OPERAND_A_DRAM_MISS_SERVICE:
            sum(residency.a_misses_by_worker),
        b_l2_hit_service:
            sum(residency.b_hits_by_worker),
        b_dram_miss_service:
            sum(residency.b_misses_by_worker),
    }
    demands = tuple(
        MemoryAccessClassDemand(
            service=service_name,
            ingress_resource_role=service.ingress_resource_role,
            logical_bytes=float(
                service_counts[service_name]
                * service.signature.logical_row_bytes),
            signature=service.signature,
        )
        for service_name, service in services.items()
        if service_counts[service_name] > 0
    )
    if not demands:
        return memory_paths
    l2_read_bytes = next(
        node.bytes for node in read_path.nodes
        if node.resource_role == "l2_cache")
    if math.fsum(demand.logical_bytes for demand in demands) > l2_read_bytes:
        raise AssertionError(
            "CANN grouped operand full descriptors exceed L2 ingress bytes")
    typed_read_path = dataclasses.replace(
        read_path,
        access_class_demands=demands,
    )
    paths = tuple(
        typed_read_path if path is read_path else path
        for path in memory_paths)
    return paths




def _cann_grouped_tile_barrier_sequence(
        *, class_index: int, logical_m: int, logical_n: int,
        projection_k: int, tile_m: int, tile_n: int, tile_k: int,
        element_bytes: int, tensor_operations: int, services,
        a_hit_fraction: float, b_hit_fraction: float,
        ) -> PipelineTaskSequenceDemand:
    """Build one compact, source-closed GroupedMatmul tile template.

    The template preserves CANN's K-chunk and double-buffer dependencies, but
    is evaluated once per ``(M, N)`` shape class rather than once per CTA.
    ``BarrierPipelineEnvelopeDemand`` applies its exact per-AIC histogram.
    """

    outer_k = _CANN_GROUPED_K_OUTER_BASE_TILES * tile_k
    k_chunks = tuple(
        min(outer_k, projection_k - offset)
        for offset in range(0, projection_k, outer_k))
    if not k_chunks:
        raise ValueError("CANN GroupedMatmul requires positive K work")
    padded_m = math.ceil(logical_m / 16) * 16
    padded_n = math.ceil(logical_n / 16) * 16
    for name, fraction in (
            ("a_hit_fraction", a_hit_fraction),
            ("b_hit_fraction", b_hit_fraction)):
        if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1]")

    b_l2_hit_service = (
        _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE
        if _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE in services
        else _CANN_OPERAND_B_L2_HIT_SERVICE)
    b_dram_miss_service = next(
        service_name for service_name in (
            _CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE,
            _CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE,
            _CANN_OPERAND_B_DRAM_MISS_SERVICE)
        if service_name in services)

    def operand_segments(
            prefix, total_bytes, typed_bytes, hit_fraction,
            hit_service_name, miss_service_name):
        hit_bytes = typed_bytes * hit_fraction
        miss_bytes = typed_bytes - hit_bytes
        tail_bytes = total_bytes - typed_bytes
        if tail_bytes < -1e-6:
            raise AssertionError("typed CANN operand bytes exceed its descriptor")
        segments = []
        for suffix, byte_count, service_name in (
                ("l2_hit", hit_bytes, hit_service_name),
                ("dram_miss", miss_bytes, miss_service_name),
                ("tail", max(tail_bytes, 0.0), None)):
            if byte_count <= 0.0:
                continue
            access_demands = ()
            if service_name is not None:
                service = services[service_name]
                access_demands = (MemoryAccessClassDemand(
                    service=service.service,
                    ingress_resource_role=service.ingress_resource_role,
                    logical_bytes=byte_count,
                    signature=service.signature,
                ),)
            segments.append((f"{prefix}_{suffix}", byte_count, access_demands))
        if not segments:
            raise AssertionError("CANN operand descriptor is empty")
        return segments

    def chunk_task(name, chunk_k, *, first, final, repeat_count=1):
        full_k_tiles = chunk_k // tile_k
        a_bytes = float(logical_m * chunk_k * element_bytes)
        b_bytes = float(logical_n * chunk_k * element_bytes)
        typed_a_available = (
            _CANN_OPERAND_A_L2_HIT_SERVICE in services
            and _CANN_OPERAND_A_DRAM_MISS_SERVICE in services)
        typed_b_available = (
            b_l2_hit_service in services
            and _CANN_OPERAND_B_DRAM_MISS_SERVICE in services)
        typed_a_bytes = float(
            full_k_tiles * tile_m * tile_k * element_bytes
            if typed_a_available and logical_m == tile_m else 0)
        typed_b_bytes = float(
            full_k_tiles * tile_n * tile_k * element_bytes
            if typed_b_available and logical_n == tile_n else 0)
        segments = (
            operand_segments(
                "load_a", a_bytes, typed_a_bytes, a_hit_fraction,
                _CANN_OPERAND_A_L2_HIT_SERVICE,
                _CANN_OPERAND_A_DRAM_MISS_SERVICE)
            + operand_segments(
                "load_b", b_bytes, typed_b_bytes, b_hit_fraction,
                b_l2_hit_service,
                b_dram_miss_service))
        stages = []
        predecessor = None
        for index, (stage_name, byte_count, access_demands) in enumerate(
                segments):
            if index == len(segments) - 1:
                stage_name = "load"
            stages.append(PipelineStageDemand(
                name=stage_name,
                service="memory:mte2_read",
                work_fraction=None,
                memory_node_demands=(PhaseMemoryDemand(
                    path_name="mte2_read",
                    resource_role="l2_cache",
                    bytes=byte_count,
                ),),
                memory_access_class_demands=access_demands,
                predecessors=(() if predecessor is None else (predecessor,)),
                prior_resources=(
                    ("compute:compute_tensor", "memory:aic_output_l2_write")
                    if first and predecessor is None else ()),
            ))
            predecessor = stage_name
        operations = float(2 * padded_m * padded_n * chunk_k)
        stages.append(PipelineStageDemand(
            name="compute",
            service="compute:compute_tensor",
            work_fraction=operations / tensor_operations,
            predecessors=("load",),
        ))
        if final:
            output_bytes = float(logical_m * logical_n * element_bytes)
            stages.append(PipelineStageDemand(
                name="store",
                service="memory:aic_output_l2_write",
                work_fraction=None,
                memory_node_demands=(
                    PhaseMemoryDemand(
                        path_name="aic_output_l2_write",
                        resource_role="l2_cache",
                        bytes=output_bytes,
                    ),
                    PhaseMemoryDemand(
                        path_name="aic_output_l2_write",
                        resource_role="dram",
                        bytes=output_bytes,
                    ),
                ),
                predecessors=("compute",),
            ))
        return PipelineTaskDemand(
            name=name,
            worker=0,
            stages=tuple(stages),
            repeat_count=repeat_count,
        )

    if len(k_chunks) == 1:
        tasks = (chunk_task("k_chunk_final", k_chunks[0], first=True, final=True),)
    else:
        tasks = [chunk_task("k_chunk_first", k_chunks[0], first=True, final=False)]
        if len(k_chunks) > 2:
            tasks.append(chunk_task(
                "k_chunk_body", outer_k, first=False, final=False,
                repeat_count=len(k_chunks) - 2))
        tasks.append(chunk_task("k_chunk_final", k_chunks[-1], first=False, final=True))
        tasks = tuple(tasks)
    return PipelineTaskSequenceDemand(
        name=f"tile_class_{class_index}_m{logical_m}_n{logical_n}",
        worker=0,
        tasks=tasks,
    )


def _cann_tile_class_census(residency, work):
    """Census the populated (M, N) tile classes across the AIC workers.

    Returns the sorted class keys, their index map, and each worker's count
    vector.  The two envelope callers name the returned keys differently
    (`class_keys` / `class_shapes`) but read the same triple; a residency plan
    with no populated class, or one whose census loses a worker, is a contract
    violation rather than an empty result.
    """

    class_keys = tuple(sorted({
        (logical_m, logical_n)
        for worker_classes in residency.tile_classes_by_worker
        for logical_m, logical_n, _ in worker_classes
    }))
    if not class_keys:
        raise ValueError("CANN GroupedMatmul has no populated tile classes")
    class_index = {key: index for index, key in enumerate(class_keys)}
    counts_by_worker = []
    for worker_classes in residency.tile_classes_by_worker:
        counts = [0] * len(class_keys)
        for logical_m, logical_n, count in worker_classes:
            counts[class_index[(logical_m, logical_n)]] = count
        counts_by_worker.append(tuple(counts))
    if len(counts_by_worker) != work.worker_count:
        raise AssertionError("CANN tile class census lost an AIC worker")
    return class_keys, class_index, counts_by_worker




def _cann_grouped_matmul_envelope_demand(
        kernel: KernelSpec, work: _CannGroupedMatmulWork,
        residency: _CannGroupedOperandResidency,
        hardware) -> Optional[CannGroupedMatmulEnvelopeDemand]:
    """Return the analytical source histogram used by Forward on 910B."""

    program_names = (
        "cann_depth_a1", "cann_depth_b1",
        "cann_step_ka", "cann_step_kb",
        "cann_step_m", "cann_step_n",
        "cann_db_l0a", "cann_db_l0b", "cann_db_l0c",
    )
    present = tuple(name in kernel.launch.parameters for name in program_names)
    if any(present) and not all(present):
        raise ValueError(
            "CANN GroupedMatmul launch has an incomplete K program")
    if not any(present):
        return None
    program = {}
    for name in program_names:
        value = kernel.launch.parameters[name]
        if not float(value).is_integer() or value <= 0:
            raise ValueError(
                f"CANN GroupedMatmul launch {name} must be a positive integer")
        program[name] = int(value)
    if program["cann_step_m"] != 1 or program["cann_step_n"] != 1:
        raise ValueError(
            "CANN GroupedMatmul analytical envelope requires unit M/N steps")
    if (program["cann_depth_a1"]
            % (program["cann_step_m"] * program["cann_step_ka"])
            or program["cann_depth_b1"]
            % (program["cann_step_n"] * program["cann_step_kb"])):
        raise ValueError(
            "CANN GroupedMatmul L1 depth must contain complete K groups")
    if program["cann_db_l0c"] != 1:
        raise ValueError(
            "CANN GroupedMatmul analytical envelope requires captured L0C")
    services = _cann_grouped_operand_services(kernel, hardware)
    if services is None:
        return None
    class_shapes, class_index, counts_by_worker = _cann_tile_class_census(
        residency, work)
    a_hit_fraction, b_hit_fraction = (
        _cann_grouped_operand_hit_fractions(kernel, residency))
    role, _, projection_k, projection_n = _projection(kernel)
    source_projection_n = (
        projection_n // 2 if role == "w1" else projection_n)
    _, local_counts = _routing(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)

    # ``GMMProcess`` issues each expert's MN tiles in its source diagonal
    # order and advances the AIC ring across expert boundaries.  Preserve the
    # order as compact runs, rather than materialising one Pipeline task per
    # CTA.  The controller recurrence consumes this exact static program;
    # counts_by_worker remains the independent conservation check above.
    source_runs_by_worker = [[] for _ in range(work.worker_count)]
    ring_offset = 0
    # The source program walks the full W1 output width.  The envelope's
    # source_projection_n retains the pre-existing per-projection service
    # field, while its histogram (and therefore this controller program)
    # covers both W1 halves exactly once.
    n_tiles = math.ceil(projection_n / tile_n)
    for rows in local_counts:
        if rows == 0:
            continue
        m_tiles = math.ceil(rows / tile_m)
        expert_tile_count = m_tiles * n_tiles
        for relative_block, (m_index, n_index) in enumerate(
                _cann_mn_block_order(m_tiles, n_tiles)):
            logical_m = min(tile_m, rows - m_index * tile_m)
            logical_n = min(tile_n, projection_n - n_index * tile_n)
            source_class_index = class_index[(logical_m, logical_n)]
            worker = (ring_offset + relative_block) % work.worker_count
            worker_runs = source_runs_by_worker[worker]
            if (worker_runs
                    and worker_runs[-1][0] == source_class_index):
                worker_runs[-1] = (
                    source_class_index, worker_runs[-1][1] + 1)
            else:
                worker_runs.append((source_class_index, 1))
        ring_offset = (ring_offset + expert_tile_count) % work.worker_count
    return CannGroupedMatmulEnvelopeDemand(
        worker_count=work.worker_count,
        zero_expert_count=sum(rows == 0 for rows in local_counts),
        class_shapes=class_shapes,
        counts_by_worker=tuple(counts_by_worker),
        source_class_runs_by_worker=tuple(
            tuple(runs) for runs in source_runs_by_worker),
        projection_k=projection_k,
        projection_n=source_projection_n,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_step_k_tiles=program["cann_step_ka"],
        b_step_k_tiles=program["cann_step_kb"],
        a1_buffer_slots=(
            program["cann_depth_a1"]
            // (program["cann_step_m"] * program["cann_step_ka"])),
        b1_buffer_slots=(
            program["cann_depth_b1"]
            // (program["cann_step_n"] * program["cann_step_kb"])),
        l0a_buffer_slots=program["cann_db_l0a"],
        l0b_buffer_slots=program["cann_db_l0b"],
        l0c_buffer_slots=program["cann_db_l0c"],
        dtype=kernel.dtype,
        element_bytes=int(kernel.shape["dtype_bytes"]),
        tensor_operations=float(work.tensor_operations),
        a_l2_hit_fraction=a_hit_fraction,
        b_l2_hit_fraction=b_hit_fraction,
    )




def _grouped_matmul_phase_ledger(
        inp: PredictionInput, *,
        memory_paths: Optional[Tuple[MemoryPathDemand, ...]] = None,
        compute_demands: Optional[Tuple[ComputeDemand, ...]] = None
        ) -> AnalyticalPhaseLedger:
    """Build the hardware-neutral grouped-projection semantic ledger.

    All three accelerators expose the same operand ingress, matrix multiply,
    and output-writeback semantics.  Their resource services and the amount
    of source ordering available differ, but they no longer choose different
    workload representations merely because of the hardware backend.
    """

    if inp.phase_ledger is not None:
        raise ValueError("FUSED_MOE phase ledger must be attached exactly once")
    if memory_paths is None:
        memory_paths = inp.memory_paths
    if compute_demands is None:
        compute_demands = inp.compute_demands
    return _grouped_matmul_phase_ledger_from_demands(
        memory_paths, compute_demands)


def _grouped_matmul_phase_ledger_from_demands(
        memory_paths: Tuple[MemoryPathDemand, ...],
        compute_demands: Tuple[ComputeDemand, ...]
        ) -> AnalyticalPhaseLedger:
    """Build the shared projection ledger from completed resource demands."""

    if not isinstance(memory_paths, tuple):
        raise TypeError("FUSED_MOE phase-ledger memory paths must be a tuple")
    if not isinstance(compute_demands, tuple):
        raise TypeError("FUSED_MOE phase-ledger compute demands must be a tuple")
    read_paths = tuple(
        path for path in memory_paths if path.access_kind == "read")
    write_paths = tuple(
        path for path in memory_paths if path.access_kind == "write")
    if not write_paths:
        raise ValueError(
            "FUSED_MOE grouped projection requires a write path")
    def memory_demands(paths):
        return tuple(
            PhaseMemoryDemand(
                path_name=path.name,
                resource_role=node.resource_role,
                bytes=node.bytes,
            )
            for path in paths
            for node in path.nodes
        )

    if not compute_demands:
        if read_paths:
            raise ValueError(
                "FUSED_MOE grouped projection without matrix work cannot "
                "retain operand reads")
        # An expert-parallel rank with no local route still publishes the
        # source-defined zero-filled foreign rows for the later cross-rank
        # reduction.  It is a write-only semantic phase, not a degenerate
        # matrix multiply and not an invented empty read stage.
        return AnalyticalPhaseLedger(
            phases=(AnalyticalPhase(
                name="foreign_route_zero_writeback",
                predecessors=(),
                compute_demands=(),
                memory_demands=memory_demands(write_paths),
                completion_barrier=True,
            ),),
        )
    routed_weight_paths = tuple(
        path for path in read_paths if path.name == "routed_weight_read")
    if len(routed_weight_paths) > 1:
        raise ValueError(
            "FUSED_MOE grouped projection has duplicate routed-weight paths")
    ingress_read_paths = tuple(
        path for path in read_paths if path.name != "routed_weight_read")
    if not ingress_read_paths:
        raise ValueError("FUSED_MOE grouped projection requires operand reads")

    operation_by_resource = {
        "compute_tensor": "grouped_matrix_multiply",
        "compute_fma": "finite_group_scan_control",
        "compute_alu": "routed_weight_scale",
    }
    if any(
            demand.resource_role not in operation_by_resource
            for demand in compute_demands):
        raise ValueError(
            "FUSED_MOE grouped projection has an unknown compute role")
    phase_compute_demands = tuple(
        PhaseComputeDemand(
            resource_role=demand.resource_role,
            operation=operation_by_resource[demand.resource_role],
            operations=demand.operations,
        )
        for demand in compute_demands
    )
    control_demands = tuple(
        demand for demand in phase_compute_demands
        if demand.resource_role == "compute_fma")
    routed_weight_scale_demands = tuple(
        demand for demand in phase_compute_demands
        if demand.resource_role == "compute_alu")
    matrix_demands = tuple(
        demand for demand in phase_compute_demands
        if demand.resource_role == "compute_tensor")
    if not matrix_demands:
        raise ValueError(
            "FUSED_MOE grouped projection requires Tensor matrix work")
    if routed_weight_scale_demands and len(routed_weight_paths) != 1:
        raise ValueError(
            "FUSED_MOE routed-weight scale requires one post-matmul "
            "routed-weight read")
    if not routed_weight_scale_demands and routed_weight_paths:
        raise ValueError(
            "FUSED_MOE routed-weight read requires routed-weight scale work")
    control_phase = (
        (AnalyticalPhase(
            name="group_list_scan",
            predecessors=(),
            compute_demands=control_demands,
            memory_demands=(),
        ),)
        if control_demands else ())
    ingress_predecessors = (("group_list_scan",) if control_demands else ())
    routed_weight_read_phase = (
        (AnalyticalPhase(
            name="routed_weight_read",
            predecessors=("matrix_multiply",),
            compute_demands=(),
            memory_demands=memory_demands(routed_weight_paths),
        ),)
        if routed_weight_paths else ())
    scale_phase = (
        (AnalyticalPhase(
            name="routed_weight_scale",
            predecessors=("routed_weight_read",),
            compute_demands=routed_weight_scale_demands,
            memory_demands=(),
        ),)
        if routed_weight_scale_demands else ())
    output_predecessor = (
        "routed_weight_scale"
        if routed_weight_scale_demands else "matrix_multiply")
    return AnalyticalPhaseLedger(
        phases=control_phase + (
            AnalyticalPhase(
                name="operand_ingress",
                predecessors=ingress_predecessors,
                compute_demands=(),
                memory_demands=memory_demands(ingress_read_paths),
            ),
            AnalyticalPhase(
                name="matrix_multiply",
                predecessors=("operand_ingress",),
                compute_demands=matrix_demands,
                memory_demands=(),
            ),
        ) + routed_weight_read_phase + scale_phase + (
            AnalyticalPhase(
                name="output_writeback",
                predecessors=(output_predecessor,),
                compute_demands=(),
                memory_demands=memory_demands(write_paths),
                completion_barrier=True,
            ),
        ),
    )




def _cann_grouped_matmul_analytical_input(
        kernel: KernelSpec, hardware, *,
        demand_per_resource: dict, eta: float, wave_efficiency: float,
        active_units: int, include_dispatch_overhead: bool,
        limitations: Tuple[str, ...], traffic: _CannGroupedMatmulTraffic,
        work: _CannGroupedMatmulWork,
        residency: _CannGroupedOperandResidency,
        attach_pipeline: bool = True) -> PredictionInput:
    """Construct CANN's final grouped-projection envelope in one pass.

    The aggregate AIC input, path split, typed operand annotation and
    analytical envelope are all immutable descriptions of one source-defined
    launch.  Materializing their final form directly avoids validating the
    three intermediate descriptions that never reach the formula.
    """

    if hardware.kernel_memory_model != "npu_tile":
        raise ValueError("CANN grouped analytical input requires npu_tile")
    if not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN grouped analytical input requires separated AIC/AIV")
    if not isinstance(limitations, tuple):
        raise TypeError("CANN grouped analytical limitations must be a tuple")
    final_demand = dict(demand_per_resource)
    for role in ("l1_cache", "smem", "xbar"):
        final_demand[role] = 0.0
    tensor_operations = float(final_demand["compute_tensor"])
    if tensor_operations <= 0:
        raise ValueError("CANN grouped analytical input requires Tensor work")
    source = MemorySourceDemand(
        active_units=active_units,
        producers_per_unit=1.0,
        independent_requests_per_producer=None,
        frontend_rate_per_unit_s=None,
    )
    paths = (
        MemoryPathDemand(
            name="mte2_read",
            access_kind="read",
            initial_transactions=None,
            nodes=(MemoryNodeDemand(
                resource_role="l2_cache",
                transactions=None,
                bytes=traffic.l2_read_bytes,
                transaction_bytes=None,
            ),),
            source=source,
        ),
        MemoryPathDemand(
            name="operand_dram_read",
            access_kind="read",
            initial_transactions=None,
            nodes=(MemoryNodeDemand(
                resource_role="dram",
                transactions=None,
                bytes=traffic.dram_read_bytes,
                transaction_bytes=None,
            ),),
            source=source,
        ),
        MemoryPathDemand(
            name="aic_output_l2_write",
            access_kind="write",
            initial_transactions=None,
            nodes=(
                MemoryNodeDemand(
                    resource_role="l2_cache",
                    transactions=None,
                    bytes=traffic.output_l2_write_bytes,
                    transaction_bytes=None,
                ),
                MemoryNodeDemand(
                    resource_role="dram",
                    transactions=None,
                    bytes=traffic.dram_write_bytes,
                    transaction_bytes=None,
                ),
            ),
            source=source,
        ),
    )
    paths = _cann_grouped_operand_service_paths_from_paths(
        paths, kernel, residency, hardware)
    compute_demands = [ComputeDemand(
        resource_role="compute_tensor",
        operations=tensor_operations,
        shape_efficiency=1.0,
        schedule_efficiency=1.0,
    )]
    scan_service_name = "cann_grouped_matmul_finite_group_scan_control"
    fma_resource = hardware.resource_by_role("compute_fma")
    if scan_service_name in fma_resource.exact_request_services:
        scan_service = fma_resource.exact_request_services[scan_service_name]
        scan_supported = work.expert_group_count in (
            scan_service["service_time_s_by_request_size"])
    else:
        scan_supported = False
    if scan_supported:
        scan_request = ComputeExactRequestDemand(
            service=scan_service_name,
            request_sizes_by_worker=tuple(
                (work.expert_group_count,)
                for _ in range(work.worker_count)),
        )
        scan_operations = scan_request.operations
        final_demand["compute_fma"] = scan_operations
        compute_demands.append(ComputeDemand(
            resource_role="compute_fma",
            operations=scan_operations,
            shape_efficiency=1.0,
            schedule_efficiency=1.0,
            exact_request_demands=(scan_request,),
        ))
    compute_demands = tuple(compute_demands)
    phase_ledger = (
        _grouped_matmul_phase_ledger_from_demands(paths, compute_demands)
        if attach_pipeline else None)
    pipeline_plan = (
        _cann_grouped_matmul_envelope_demand(kernel, work, residency, hardware)
        if attach_pipeline else None)
    analytical_limitations = (
        ("cann_grouped_matmul_uses_tile_class_analytical_envelope",)
        if pipeline_plan is not None else
        ("cann_grouped_matmul_tile_class_service_unavailable",))
    if not scan_supported:
        analytical_limitations += (
            "cann_grouped_matmul_finite_group_scan_service_unavailable",)
    resource_input = PredictionInput(
        demand_per_resource=final_demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=kernel.variant,
        mem_model="npu_tile",
        compute_demands=compute_demands,
        memory_paths=paths,
        pipeline_plan=pipeline_plan,
        launch_task_kind="aic",
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        include_dispatch_overhead=include_dispatch_overhead,
        support_level=("partial" if pipeline_plan is not None
                       else "aggregate_only"),
        limitations=limitations + analytical_limitations,
        phase_ledger=phase_ledger,
    )
    return resource_input




def _split_gpu_grouped_gemm_memory_paths(
        inp: PredictionInput,
        traffic: _GpuGroupedGemmTraffic) -> PredictionInput:
    """Keep operand, post-matmul router and output paths distinct."""

    if inp.mem_model != "gpu_mshr":
        raise ValueError("GPU grouped-GEMM paths require gpu_mshr memory")
    if len(inp.memory_paths) != 1:
        raise ValueError(
            "GPU grouped-GEMM path split requires one aggregate path")
    original = inp.memory_paths[0]
    if original.name != "global_memory" or original.access_kind != "read":
        raise ValueError(
            "GPU grouped-GEMM path split requires the global memory path")
    if original.access_class_demands:
        raise ValueError(
            "GPU grouped-GEMM aggregate path already has a typed service")
    original_by_role = {
        node.resource_role: node.bytes for node in original.nodes}
    expected_by_role = {
        "l2_cache": traffic.l2_total_bytes,
        "dram": traffic.dram_total_bytes,
    }
    if set(original_by_role) != set(expected_by_role) or any(
            not math.isclose(
                original_by_role[role], expected,
                rel_tol=1e-12, abs_tol=1e-6)
            for role, expected in expected_by_role.items()):
        raise ValueError(
            "GPU grouped-GEMM aggregate path does not conserve traffic")
    transaction_bytes_by_role = {
        node.resource_role: node.transaction_bytes
        for node in original.nodes}

    def make_path(name, access_kind, l2_bytes, dram_bytes):
        if l2_bytes <= 0:
            raise ValueError(
                f"GPU grouped-GEMM {name} requires positive L2 traffic")
        if dram_bytes < 0 or dram_bytes > l2_bytes:
            raise ValueError(
                f"GPU grouped-GEMM {name} has invalid DRAM traffic")
        l2_transaction_bytes = transaction_bytes_by_role["l2_cache"]
        dram_transaction_bytes = transaction_bytes_by_role["dram"]
        if l2_transaction_bytes is None or dram_transaction_bytes is None:
            raise ValueError(
                "GPU grouped-GEMM paths require resolved transactions")
        nodes = [MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=l2_bytes / l2_transaction_bytes,
            bytes=l2_bytes,
            transaction_bytes=l2_transaction_bytes,
        )]
        if dram_bytes > 0:
            nodes.append(MemoryNodeDemand(
                resource_role="dram",
                transactions=dram_bytes / dram_transaction_bytes,
                bytes=dram_bytes,
                transaction_bytes=dram_transaction_bytes,
            ))
        return MemoryPathDemand(
            name=name,
            access_kind=access_kind,
            initial_transactions=l2_bytes / l2_transaction_bytes,
            nodes=tuple(nodes),
            source=original.source,
            queues=original.queues,
        )

    paths = []
    for name, access_kind, l2_bytes, dram_bytes in (
        (
            "activation_read", "read",
            traffic.activation_l2_read_bytes,
            traffic.activation_dram_read_bytes,
        ),
        (
            "weight_read", "read",
            traffic.weight_l2_read_bytes,
            traffic.weight_dram_read_bytes,
        ),
        (
            "routed_weight_read", "read",
            traffic.routed_weight_l2_read_bytes,
            traffic.routed_weight_dram_read_bytes,
        ),
        (
            "output_write", "write",
            traffic.output_l2_write_bytes,
            traffic.output_dram_write_bytes,
        ),
    ):
        if l2_bytes > 0:
            paths.append(make_path(
                name, access_kind, l2_bytes, dram_bytes))
    if not paths:
        raise ValueError(
            "GPU grouped-GEMM requires at least one nonzero memory path")
    return dataclasses.replace(inp, memory_paths=tuple(paths))


def _architecture_input(inp: PredictionInput, kernel: KernelSpec, hardware):
    memory_model = hardware.kernel_memory_model
    if memory_model == "npu_tile":
        # The CANN branch already constructs the canonical npu_tile contract.
        # Avoid re-validating an observationally identical intermediate input.
        return inp
    elif memory_model == "gpu_mshr":
        return inp
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")


def _apply_metax_bm32_fused_moe_mma_occupancy(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Apply the independent C500 BM32 Tensor occupancy mechanism.

    The source-resolved Triton configuration and compiler resource record
    close this to BM32/BN64/BK128, four wave64 threads, and a 48-KiB dynamic
    shared allocation.  That allocation admits one CTA/AP on C500, so the
    register-only BF16 MMA point at four resident waves and two accumulator
    chains is directly applicable.  No Serving or kernel duration selects
    this path.
    """

    occupancy = hardware.metax_fused_moe_bm32_bf16_mma_occupancy
    if occupancy is None:
        return inp
    if kernel.dtype != occupancy["dtype"]:
        return inp
    if kernel.launch.algorithm not in {
            "triton_fused_moe_w1", "triton_fused_moe_w2"}:
        return inp
    if kernel.launch.block_size != occupancy["threads_per_cta"]:
        return inp
    if kernel.launch.grid_size < hardware.num_compute_units:
        return inp
    tiling = kernel.launch.tiling
    if tiling is None:
        raise ValueError("FUSED_MOE BM32 occupancy requires a tiled launch")
    if tuple(
            int(tiling.tile_shape[name]) for name in ("m", "n", "k")) != (
                occupancy["tile_m"], occupancy["tile_n"],
                occupancy["tile_k"]):
        return inp
    dynamic_shared_parameter = "dynamic_shared_memory_bytes"
    if dynamic_shared_parameter not in kernel.launch.parameters:
        return inp
    if kernel.launch.parameters[dynamic_shared_parameter] != occupancy[
            "dynamic_shared_memory_bytes"]:
        return inp
    tensor_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_tensor")
    if len(tensor_demands) != 1:
        raise ValueError(
            "FUSED_MOE BM32 occupancy requires one Tensor compute demand")
    tensor = tensor_demands[0]
    efficiency = (
        occupancy["flops_per_ap_cycle"]
        / occupancy["architecture_peak_flops_per_ap_cycle"])
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("FUSED_MOE BM32 MMA occupancy efficiency is invalid")
    adjusted = dataclasses.replace(
        tensor,
        schedule_efficiency=tensor.schedule_efficiency * efficiency,
    )
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            adjusted if demand is tensor else demand
            for demand in inp.compute_demands),
    )


def _undeclared_grouped_operand_tile_class(
        kernel: KernelSpec, hardware) -> bool:
    """Whether no declared grouped-operand service covers the tile class.

    Grouped-operand services are keyed by the launched Triton tile class
    (dtype, BM, BN, BK, GROUP_SIZE_M).  A launch whose tile class no declared
    service carries (e.g. a newer vLLM default configuration than the one a
    hardware's services were measured for) is outside every typed service's
    domain: it keeps the aggregate grouped-GEMM memory path together with
    its explicit ``moe_gather_service_class_not_separated`` limitation, the
    same contract as a masked-BK gather tail.  A launch whose tile class is
    declared but whose routing topology has no exact service remains a
    contract error.
    """

    tile_m, tile_n, tile_k = _tile(kernel)
    group_size_m = int(kernel.launch.parameters["group_size_m"])
    launch_class = (kernel.dtype, tile_m, tile_n, tile_k, group_size_m)
    declared_classes = {
        (service.signature.dtype, service.signature.tile_m,
         service.signature.tile_n, service.signature.tile_k,
         service.signature.group_size_m)
        for service in hardware.memory_access_class_services
        if service.signature.operand_role is not None
    }
    return launch_class not in declared_classes


def _attach_gpu_gather_service(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        *, traffic: _GpuGroupedGemmTraffic, routed_rows: float,
        projection_k: int, projection_n: int,
        element_bytes: int) -> PredictionInput:
    """Annotate exact grouped-operand services when hardware provides them."""

    # A800 and C500 select their own exact source service/signature below,
    # then share the immutable typed-memory contract binder.  Only the
    # source-conserved byte count remains dynamic for each prediction.
    MemoryAccessClassDemand = bind_memory_access_class_template

    metax_pipeline_service = _metax_multi_tile_operand_pipeline_service(
        kernel, hardware)
    nvidia_source_depth_service = _nvidia_source_depth_operand_service(
        kernel, hardware)
    if metax_pipeline_service is not None:
        combined_service = metax_pipeline_service
    elif nvidia_source_depth_service is not None:
        combined_service = nvidia_source_depth_service
    else:
        combined_service = _gpu_exact_grouped_operand_service(
            kernel, hardware)
    split_services = (
        None if nvidia_source_depth_service is not None
        else _gpu_split_operand_services(kernel, hardware))
    full_group_services = (
        {} if nvidia_source_depth_service is not None
        else _gpu_full_group_operand_services(kernel, hardware))
    declared_grouped_service = any(
        hardware.has_memory_access_class_service(service)
        for service in (
            _GROUPED_OPERAND_SERVICE,
            _GROUPED_OPERAND_PIPELINE_SERVICE,
            _GROUPED_WEIGHT_OVERLAPPED_DRAM_MISS_SERVICE,
            _NVIDIA_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE,
            _NVIDIA_EIGHT_EXPERT_132_ACTIVE_136_MAX_W1_SERVICE,
            _NVIDIA_E512_DEEP_W2_SERVICE,
            _NVIDIA_E128_DEEP_W2_SERVICE,
            *_NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT.values(),
            _NVIDIA_E8_H6144_DEEP_W2_SERVICE,
            _NVIDIA_E8_H6144_DEEP_W1_SERVICE,
            _NVIDIA_E8_H4096_515_ACTIVE_520_MAX_W2_SERVICE,
            _NVIDIA_E8_H4096_1027_ACTIVE_1032_MAX_W2_SERVICE,
            _NVIDIA_E512_H4096_DEEP_W1_SERVICE,
            _NVIDIA_DEEP_H2880_W1_W2_SERVICE,
            _NVIDIA_BM16_H2880_31_ACTIVE_SERVICE,
            _NVIDIA_BM16_H2880_53_ACTIVE_SERVICE,
            _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE,
            _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE,
            _NVIDIA_BM16_E256_16_ACTIVE_241_MAX_W2_SERVICE,
            _NVIDIA_BM16_E512_20_ACTIVE_482_MAX_W2_SERVICE,
            _METAX_GROUPED_OPERAND_LONG_K_SERVICE,
            _METAX_GROUPED_OPERAND_WIDE_N_SERVICE,
            _METAX_GLOBAL_DEEP_OPERAND_LONG_K_SERVICE,
            _METAX_GLOBAL_LARGE_OPERAND_WIDE_N_SERVICE,
            _METAX_LONG_LIFETIME_OPERAND_LONG_K_SERVICE,
            _METAX_LONG_LIFETIME_OPERAND_WIDE_N_SERVICE,
            _METAX_REPEATED_OPERAND_DEEP_K_SERVICE,
            _METAX_REPEATED_OPERAND_WIDE_N_SERVICE,
            _METAX_REPEATED_OPERAND_BALANCED_N_SERVICE,
            _METAX_REPEATED_OPERAND_MID_LONG_K_SERVICE,
            _METAX_REPEATED_OPERAND_MID_WIDE_N_SERVICE,
            _METAX_SINGLE_TILE_OPERAND_LONG_K_SERVICE,
            _METAX_SINGLE_TILE_OPERAND_WIDE_N_SERVICE,
            _METAX_EIGHT_EXPERT_EIGHT_ACTIVE_TWELVE_MAX_W2_SERVICE,
            _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W1_SERVICE,
            _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W2_SERVICE,
            _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W1_SERVICE,
            _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W2_SERVICE,
            _METAX_EIGHT_EXPERT_STEADY_WIDE_N_SERVICE,
            _METAX_EIGHT_EXPERT_STEADY_K192_N512_W1_SERVICE,
            _METAX_E256_LONG_LIFETIME_LONG_K_SERVICE,
            _METAX_E256_LONG_LIFETIME_WIDE_N_SERVICE,
            _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W1_SERVICE,
            _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W2_SERVICE,
            _METAX_E512_DENSE_LARGE_W2_SERVICE,
            _METAX_E128_LONG_LIFETIME_W1_SERVICE,
            _METAX_E128_LONG_LIFETIME_W2_SERVICE,
            _METAX_E128_DEEP_W2_SERVICE,
            _METAX_E128_EIGHT_TILE_W1_SERVICE,
            _METAX_E128_EIGHT_TILE_W2_SERVICE,
            _METAX_E128_TOPK16_EIGHT_TILE_W1_SERVICE,
            _METAX_E128_EIGHT_TILE_WIDE_N_W2_SERVICE,
            _METAX_EIGHT_EXPERT_TWELVE_TILE_W2_SERVICE,
            _METAX_EIGHT_EXPERT_TWELVE_TILE_K128_N448_W1_SERVICE,
            _METAX_EIGHT_EXPERT_TWELVE_TILE_K192_N512_W1_SERVICE,
            _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W1_SERVICE,
            _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W2_SERVICE,
            _METAX_E128_SPARSE_TWO_TILE_W1_SERVICE,
            _METAX_E128_SPARSE_TWO_TILE_W2_SERVICE,
            _METAX_E128_SINGLE_TILE_W1_SERVICE,
            _METAX_E128_SINGLE_TILE_W2_SERVICE,
            _METAX_E128_SINGLE_TILE_MID_W1_SERVICE,
            _METAX_E128_SINGLE_TILE_MID_W2_SERVICE,
            _METAX_E128_SINGLE_TILE_LARGE_W1_SERVICE,
            _METAX_E128_SINGLE_TILE_LARGE_W2_SERVICE,
            _METAX_E128_UPPER_TWO_TILE_W1_SERVICE,
            _METAX_E128_UPPER_TWO_TILE_W2_SERVICE,
            _METAX_E128_TOPK8_128_ACTIVE_190_MAX_W1_SERVICE,
            _METAX_E128_TOPK16_128_ACTIVE_190_MAX_W1_SERVICE,
            _METAX_E128_128_ACTIVE_190_MAX_W2_SERVICE,
            _METAX_E256_EIGHT_TILE_LONG_K_SERVICE,
            _METAX_E256_EIGHT_TILE_WIDE_N_SERVICE,
            _METAX_E256_EIGHT_TILE_MID_W1_SERVICE,
            _METAX_E256_EIGHT_TILE_MID_W2_SERVICE,
            _METAX_E256_FOUR_TILE_LONG_K_SERVICE,
            _METAX_E256_FOUR_TILE_WIDE_N_SERVICE,
            _METAX_E256_FOUR_TILE_MID_LONG_K_SERVICE,
            _METAX_E256_FOUR_TILE_MID_WIDE_N_SERVICE,
            _METAX_E128_FOUR_TILE_W1_SERVICE,
            _METAX_E128_FOUR_TILE_W2_SERVICE,
            _METAX_E128_REPEATED_SOURCE_TOPK8_W1_SERVICE,
            _METAX_E128_REPEATED_SOURCE_TOPK16_W1_SERVICE,
            _METAX_E128_REPEATED_K24_N32_W2_SERVICE,
            _METAX_E256_UPPER_TWO_TILE_LONG_K_SERVICE,
            _METAX_E256_UPPER_TWO_TILE_WIDE_N_SERVICE,
            _METAX_E256_SINGLE_TILE_W1_SERVICE,
            _METAX_E256_SINGLE_TILE_W2_SERVICE,
            _METAX_E256_SINGLE_TILE_MID_DEEP_W1_SERVICE,
            _METAX_E256_SINGLE_TILE_MID_DEEP_W2_SERVICE,
            _METAX_E256_SINGLE_TILE_LARGE_W1_SERVICE,
            _METAX_E256_SINGLE_TILE_LARGE_W2_SERVICE,
            _METAX_E256_SINGLE_TILE_LARGE_DEEP_W1_SERVICE,
            _METAX_E256_SINGLE_TILE_LARGE_DEEP_W2_SERVICE,
            _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W1_SERVICE,
            _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W2_SERVICE,
            _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W1_SERVICE,
            _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W2_SERVICE,
            _METAX_BM16_ACTIVE_GRID_WIDE_N_SERVICE,
            _METAX_BM16_E256_SIXTEEN_ACTIVE_W2_SERVICE,
            _METAX_BM16_SPARSE_PADDED_W2_SERVICE,
            _METAX_BM16_E512_SPARSE_PADDED_W2_SERVICE,
            _METAX_BM16_E512_FORTY_ACTIVE_W2_SERVICE,
            _METAX_BM16_E256_SPARSE_COLLIDING_W1_SERVICE,
            _METAX_BM16_E256_SPARSE_COLLIDING_W2_SERVICE,
            _METAX_BM16_E256_106_ACTIVE_248_MAX_W1_SERVICE,
            _METAX_BM16_E256_106_ACTIVE_248_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_152_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_152_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_64_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_124_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_128_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_128_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_136_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_136_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_184_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_184_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_248_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_248_MAX_W2_SERVICE,
            _METAX_BM16_FOUR_ACTIVE_MID_K_W1_SERVICE,
            _METAX_BM16_FOUR_ACTIVE_MID_K_W2_SERVICE,
            _METAX_BM16_EIGHT_ACTIVE_MID_K_W2_SERVICE,
            _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W1_SERVICE,
            _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W2_SERVICE,
        ))
    grouped_service = (
        combined_service is not None or split_services is not None)
    legacy_service = hardware.has_memory_access_class_service(
        "indirect_row_gather")
    metax_fallback_family = hardware.has_memory_access_class_service(
        _METAX_GROUPED_OPERAND_LONG_K_SERVICE)
    if (declared_grouped_service and legacy_service
            and not metax_fallback_family):
        raise ValueError(
            "FUSED_MOE hardware cannot declare grouped and legacy gather "
            "services together")
    if combined_service is not None and split_services is not None:
        raise ValueError(
            "FUSED_MOE launch cannot select combined and split grouped "
            "operand services together")
    if full_group_services and split_services is None:
        raise ValueError(
            "FUSED_MOE full-group service requires the split operand pair")
    if (declared_grouped_service and not grouped_service
            and not (metax_fallback_family and legacy_service)):
        if _undeclared_grouped_operand_tile_class(kernel, hardware):
            if "moe_gather_service_class_not_separated" not in (
                    inp.limitations):
                raise ValueError(
                    "aggregate grouped FUSED_MOE path lost its explicit "
                    "service-class limitation")
            return inp
        raise ValueError(
            "grouped FUSED_MOE has no exact hardware signature family")
    if not declared_grouped_service and not legacy_service:
        return inp
    if inp.mem_model != "gpu_mshr":
        raise ValueError(
            "FUSED_MOE indirect-row gather requires gpu_mshr memory")
    if grouped_service:
        tile_m, tile_n, _ = _tile(kernel)
        activation_path, = tuple(
            path for path in inp.memory_paths
            if path.name == "activation_read")
        weight_path, = tuple(
            path for path in inp.memory_paths
            if path.name == "weight_read")
        output_path, = tuple(
            path for path in inp.memory_paths
            if path.name == "output_write")
        routed_weight_paths = tuple(
            path for path in inp.memory_paths
            if path.name == "routed_weight_read")
        if len(routed_weight_paths) > 1:
            raise ValueError(
                "FUSED_MOE grouped operands have duplicate routed-weight "
                "paths")
        if routed_weight_paths and routed_weight_paths[0].access_class_demands:
            raise ValueError(
                "FUSED_MOE routed-weight path already has a typed service")
        if (activation_path.access_class_demands
                or weight_path.access_class_demands):
            raise ValueError(
                "FUSED_MOE grouped operand paths already have a typed service")
        if (activation_path.source != weight_path.source
                or activation_path.queues != weight_path.queues):
            raise ValueError(
                "FUSED_MOE grouped operand paths must share source and queues")
        activation_nodes = {
            node.resource_role: node for node in activation_path.nodes}
        weight_nodes = {
            node.resource_role: node for node in weight_path.nodes}
        if set(activation_nodes) != set(weight_nodes):
            raise ValueError(
                "FUSED_MOE grouped operand paths must traverse the same nodes")
        operand_nodes = []
        for role in ("l2_cache", "dram"):
            if role not in activation_nodes:
                continue
            activation_node = activation_nodes[role]
            weight_node = weight_nodes[role]
            if activation_node.transaction_bytes != weight_node.transaction_bytes:
                raise ValueError(
                    "FUSED_MOE grouped operand paths disagree on transaction "
                    f"size for {role!r}")
            transaction_bytes = activation_node.transaction_bytes
            bytes_total = activation_node.bytes + weight_node.bytes
            operand_nodes.append(MemoryNodeDemand(
                resource_role=role,
                transactions=(
                    bytes_total / transaction_bytes
                    if transaction_bytes is not None else None),
                bytes=bytes_total,
                transaction_bytes=transaction_bytes,
            ))
        l2_transaction_bytes = activation_nodes["l2_cache"].transaction_bytes
        if l2_transaction_bytes is None:
            raise ValueError(
                "FUSED_MOE grouped operand service requires L2 transactions")
        operand_bytes = (
            traffic.activation_l2_read_bytes
            + traffic.weight_l2_read_bytes)
        if split_services is not None:
            l2_hit_bytes = (
                operand_bytes
                - traffic.activation_dram_read_bytes
                - traffic.weight_dram_read_bytes)
            if l2_hit_bytes < 0.0:
                raise ValueError(
                    "grouped FUSED_MOE operand L2-hit bytes cannot be "
                    "negative")
            full_group_hit_bytes_by_weight_streams = {}
            variant, source_rows, _, _ = _projection(kernel)
            source_footprint_bytes = (
                source_rows * projection_k * element_bytes)
            if (full_group_services
                    and variant == "w1"
                    and source_footprint_bytes > hardware.l2_size_bytes):
                full_group_operand_bytes_by_weight_streams = (
                    _gpu_full_group_operand_bytes_by_weight_streams(
                        kernel, projection_k=projection_k,
                        projection_n=projection_n))
                dram_operand_bytes = (
                    traffic.activation_dram_read_bytes
                    + traffic.weight_dram_read_bytes)
                full_group_hit_bytes_by_weight_streams = {
                    topology: max(0.0, logical_bytes - dram_operand_bytes)
                    for topology, logical_bytes in (
                        full_group_operand_bytes_by_weight_streams.items())
                    if topology in full_group_services
                }
            classified_group_hit_bytes = math.fsum(
                full_group_hit_bytes_by_weight_streams.values())
            if classified_group_hit_bytes > l2_hit_bytes + 1e-6:
                raise ValueError(
                    "grouped FUSED_MOE classified group hits exceed total "
                    "L2-hit operands")
            split_demands = [(
                _GROUPED_OPERAND_PIPELINE_SERVICE,
                max(0.0, l2_hit_bytes - classified_group_hit_bytes),
                split_services["combined"].signature,
            )]
            split_demands.extend(
                (
                    full_group_services[topology].service,
                    logical_bytes,
                    full_group_services[topology].signature,
                )
                for topology, logical_bytes in sorted(
                    full_group_hit_bytes_by_weight_streams.items(),
                    key=lambda item: str(item[0]))
            )
            split_demands.append((
                split_services["weight"].service,
                traffic.weight_dram_read_bytes,
                split_services["weight"].signature,
            ))
            access_class_demands = tuple(
                MemoryAccessClassDemand(
                    service=service,
                    ingress_resource_role="l2_cache",
                    logical_bytes=logical_bytes,
                    signature=signature,
                )
                for service, logical_bytes, signature in split_demands
                if logical_bytes > 0.0)
        elif nvidia_source_depth_service is not None:
            access_class_demands = (MemoryAccessClassDemand(
                service=nvidia_source_depth_service.service,
                ingress_resource_role="l2_cache",
                logical_bytes=operand_bytes,
                signature=nvidia_source_depth_service.signature,
                replaces_ingress_path_service=True,
            ),)
        elif metax_pipeline_service is not None:
            access_class_demands = (MemoryAccessClassDemand(
                service=metax_pipeline_service.service,
                ingress_resource_role="l2_cache",
                logical_bytes=operand_bytes,
                signature=metax_pipeline_service.signature,
            ),)
        else:
            services_by_active_rows = (
                _gpu_grouped_operand_service_by_active_rows(
                    kernel, hardware))
            service = services_by_active_rows[tile_m]
            if len(services_by_active_rows) == 1:
                access_class_demands = (MemoryAccessClassDemand(
                    service=_GROUPED_OPERAND_SERVICE,
                    ingress_resource_role="l2_cache",
                    logical_bytes=operand_bytes,
                    signature=service.signature,
                ),)
            else:
                _, local_counts = _routing(kernel)
                n_tiles = math.ceil(projection_n / tile_n)
                programs_by_active_rows = {}
                for count in local_counts:
                    full_tiles, tail_rows = divmod(count, tile_m)
                    if full_tiles:
                        if tile_m in programs_by_active_rows:
                            programs_by_active_rows[tile_m] += (
                                full_tiles * n_tiles)
                        else:
                            programs_by_active_rows[tile_m] = (
                                full_tiles * n_tiles)
                    if tail_rows:
                        if tail_rows in programs_by_active_rows:
                            programs_by_active_rows[tail_rows] += n_tiles
                        else:
                            programs_by_active_rows[tail_rows] = n_tiles
                activation_replay, weight_replay = (
                    _gpu_operand_replay_factors(kernel))
                access_class_demands = tuple(
                    MemoryAccessClassDemand(
                        service=_GROUPED_OPERAND_SERVICE,
                        ingress_resource_role="l2_cache",
                        logical_bytes=float(
                            program_count
                            * projection_k
                            * element_bytes
                            * (
                                active_rows * activation_replay
                                + tile_n * weight_replay)),
                        signature=services_by_active_rows[
                            active_rows].signature,
                    )
                    for active_rows, program_count in sorted(
                        programs_by_active_rows.items()))
                segmented_operand_bytes = math.fsum(
                    demand.logical_bytes for demand in access_class_demands)
                if not math.isclose(
                        segmented_operand_bytes, operand_bytes,
                        rel_tol=1e-12, abs_tol=1e-6):
                    raise ValueError(
                        "grouped FUSED_MOE masked-row services do not "
                        "conserve operand traffic")
        operand_path = MemoryPathDemand(
            name="operand_read",
            access_kind="read",
            initial_transactions=operand_bytes / l2_transaction_bytes,
            nodes=tuple(operand_nodes),
            source=activation_path.source,
            queues=activation_path.queues,
            access_class_demands=access_class_demands,
        )
        paths = [operand_path, *routed_weight_paths, output_path]
        demand_by_path = None
    else:
        service_name = "indirect_row_gather"
        tile_m, tile_n, tile_k = _tile(kernel)
        activation_replay, _ = _gpu_operand_replay_factors(kernel)
        logical_bytes = float(
            activation_replay
            * routed_rows
            * projection_k
            * math.ceil(projection_n / tile_n)
            * element_bytes)
        signature = MemoryAccessClassSignature(
            access_class=service_name,
            dtype=kernel.dtype,
            tile_m=tile_m,
            tile_k=tile_k,
            # Measured services are keyed by their BF16 tile class.
            logical_row_bytes=tile_k * int(kernel.shape["dtype_bytes"]),
        )
        matching_services = tuple(
            service for service in hardware.memory_access_class_services
            if service.service == service_name
            and service.access_kind == "read"
            and service.ingress_resource_role == "l2_cache"
            and service.signature == signature)
        if len(matching_services) > 1:
            raise ValueError(
                "FUSED_MOE indirect gather repeats an exact hardware "
                "signature")
        if not matching_services:
            # Triton masks an incomplete K iteration.  The launch and its
            # logical bytes remain valid, but a full-BK gather measurement
            # cannot be reused for a BK tail.  Keep the aggregate memory path
            # and its explicit unsupported typed-service limitation instead
            # of fabricating a matching service signature.
            return inp
        demand_by_path = {
            "activation_read": MemoryAccessClassDemand(
                service=service_name,
                ingress_resource_role="l2_cache",
                logical_bytes=logical_bytes,
                signature=matching_services[0].signature,
            ),
        }
    if not grouped_service:
        matching_paths = tuple(
            path for path in inp.memory_paths if path.name == "activation_read")
        if len(matching_paths) != 1:
            raise ValueError(
                "FUSED_MOE gather service requires exactly one activation path")
        if matching_paths[0].access_class_demands:
            raise ValueError(
                "FUSED_MOE activation path already has a typed service")
        paths = []
        for path in inp.memory_paths:
            if path.name in demand_by_path:
                paths.append(dataclasses.replace(
                    path, access_class_demands=(demand_by_path[path.name],)))
            elif path.name in (
                    "activation_read", "weight_read", "routed_weight_read",
                    "output_write"):
                paths.append(path)
            else:
                raise ValueError(
                    "FUSED_MOE gather service encountered an unexpected "
                    f"memory path: {path.name!r}")
    absent_limitation = "moe_gather_service_class_not_separated"
    if absent_limitation not in inp.limitations:
        raise ValueError(
            "FUSED_MOE gather capability cannot clear an absent limitation")
    limitations = tuple(
        value for value in inp.limitations
        if value != absent_limitation
        and not (
            combined_service is not None
            and combined_service.replay_to_dram_fraction is not None
            and value == "moe_l2_residency_order_unavailable"))
    return dataclasses.replace(
        inp,
        memory_paths=tuple(paths),
        limitations=limitations,
    )


def _routed_experts_model(
        kernel: KernelSpec, hardware) -> Tuple[PredictionInput, float]:
    """Aggregate the pinned Marlin W4A16 two-GEMM MoE program.

    Packed weights, BF16 group scales, BF16 activations, SwiGLU materialization,
    and the final top-k reduction are counted directly from the typed shape and
    source-selected M block.  This path intentionally has no duration-derived
    correction or per-shape latency table.
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise NotImplementedError(
            "INT4 routed experts are currently modeled only on A800 GPU")
    if kernel.launch.algorithm != "vllm_marlin_moe_w4a16_aggregate":
        raise ValueError("routed_experts requires the pinned Marlin launch")
    if kernel.dtype != "bf16":
        raise ValueError("Kimi routed_experts requires BF16 activations")
    if kernel.attributes["routed_weight_dtype"] != "int4":
        raise ValueError("routed_experts requires INT4 packed weights")
    group_size = int(kernel.attributes["weight_group_size"])
    if group_size != 32:
        raise ValueError("Kimi routed_experts requires group size 32")

    tokens, topk, hidden, intermediate = (
        int(kernel.shape[name])
        for name in ("tokens", "topk", "hidden", "intermediate"))
    _, local_counts = _routing(kernel)
    routed_rows = sum(local_counts)
    expert_parallel_size = int(kernel.attributes["expert_parallel_size"])
    if expert_parallel_size == 1 and routed_rows != tokens * topk:
        raise ValueError("routed_experts local rows must conserve tokens*topk")
    if expert_parallel_size > 1 and routed_rows > tokens * topk:
        raise ValueError(
            "routed_experts local rows cannot exceed global tokens*topk")
    active_experts = sum(count > 0 for count in local_counts)
    if active_experts <= 0:
        raise ValueError("routed_experts requires an active expert")
    tile_m, _, _ = _tile(kernel)
    padded_rows = sum(
        math.ceil(count / tile_m) * tile_m
        for count in local_counts if count > 0)
    launch_padded_rows = kernel.launch.parameters["padded_rows"]
    if (not launch_padded_rows.is_integer()
            or int(launch_padded_rows) != padded_rows):
        raise ValueError(
            "routed_experts launch padded_rows disagrees with routing")

    activation_bytes = int(kernel.shape["dtype_bytes"])
    packed_weight_bytes_per_expert = (
        3.0 * hidden * intermediate * 4.0 / 8.0)
    scale_bytes_per_expert = float(
        (2 * intermediate * (hidden // group_size)
         + hidden * (intermediate // group_size))
        * activation_bytes)
    weight_bytes_per_expert = (
        packed_weight_bytes_per_expert + scale_bytes_per_expert)
    populated_blocks = padded_rows // tile_m
    weight_l2 = float(populated_blocks * weight_bytes_per_expert)
    weight_compulsory = float(active_experts * weight_bytes_per_expert)

    # Source sequence: W13 -> SwiGLU -> W2 -> top-k weighted reduction.
    # Reads and writes include the materialized BF16 intermediates between
    # those physical kernels.  Router indices and FP32 weights are explicit.
    routing_bytes = float(routed_rows * (4 + 4))
    activation_read = float(
        routed_rows * (
            2 * hidden + 4 * intermediate + 2 * intermediate
            + 2 * hidden)
        + routing_bytes)
    output_write = float(
        routed_rows * (4 * intermediate + 2 * intermediate + 2 * hidden)
        + tokens * hidden * activation_bytes)
    traffic = _GpuGroupedGemmTraffic(
        activation_l2_read_bytes=activation_read,
        weight_l2_read_bytes=weight_l2,
        routed_weight_l2_read_bytes=0.0,
        output_l2_write_bytes=output_write,
        activation_compulsory_read_bytes=activation_read,
        weight_compulsory_read_bytes=weight_compulsory,
        routed_weight_compulsory_read_bytes=0.0,
        activation_dram_read_bytes=activation_read,
        weight_dram_read_bytes=weight_l2,
        routed_weight_dram_read_bytes=0.0,
        output_dram_write_bytes=output_write,
    )
    l2_total = traffic.l2_total_bytes
    dram_bytes = traffic.dram_total_bytes
    validated_l2_miss_rate(dram_bytes, l2_total)

    tensor_flops = float(6 * padded_rows * hidden * intermediate)
    swiglu_elements = float(routed_rows * intermediate)
    reduction_elements = float(tokens * hidden * max(0, topk - 1))
    flops = {
        "compute_tensor": tensor_flops,
        "compute_fma": 0.0,
        "compute_alu": 3.0 * swiglu_elements + reduction_elements,
        "compute_sfu": swiglu_elements,
    }
    demand = build_forward_demand(l2_total, dram_bytes, False, flops)
    eta, wave_efficiency, active_units = launch_wave_geometry(
        kernel, hardware)
    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=kernel.variant,
        mem_model="gpu_mshr",
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind="generic",
        include_dispatch_overhead=True,
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        support_level="aggregate_only",
        limitations=gpu_partial_limitations(
            hardware,
            "marlin_w4a16_physical_kernels_aggregated",
            "marlin_w4a16_worker_assignment_unavailable",
            "marlin_w4a16_l2_residency_order_unavailable",
            "moe_gather_service_class_not_separated",
            "production_duration_calibration_not_used",
        ),
    )
    structured = attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name="global_memory",
        source_producers_per_unit=gpu_memory_source(hardware)[0],
        independent_requests_per_producer=gpu_memory_source(hardware)[1],
        frontend_rate_per_unit_s=None,
    )
    structured = _split_gpu_grouped_gemm_memory_paths(
        structured, traffic)
    return structured, traffic.compulsory_bytes


# wp11_moe_fp8: vLLM 0.24 FP8 block-quantized Triton grouped GEMM (H200).
_FP8_BLOCK_ALGORITHM_PREFIX = "vllm_triton_fused_moe_fp8_block128"
_FP8_BLOCK_SIZE = 128
_FP8_SCALE_BYTES = 4


# SGLang 0.5.12's own fused_moe_kernel (H200 launch selector): always
# aligned (moe_align_block_size), the _down configs read A and B via TMA.
_SGLANG_FP8_BLOCK_ALGORITHM_PREFIX = "sglang_triton_fused_moe_fp8_block128"


def _is_fp8_block_launch(kernel: KernelSpec) -> bool:
    return kernel.launch.algorithm.startswith(
        (_FP8_BLOCK_ALGORITHM_PREFIX, _SGLANG_FP8_BLOCK_ALGORITHM_PREFIX))


def _is_sglang_fp8_block_launch(kernel: KernelSpec) -> bool:
    return kernel.launch.algorithm.startswith(
        _SGLANG_FP8_BLOCK_ALGORITHM_PREFIX)


def _sglang_fp8_block_uses_tma(kernel: KernelSpec) -> bool:
    variant = kernel.variant.lower()
    if kernel.launch.algorithm == (
            f"{_SGLANG_FP8_BLOCK_ALGORITHM_PREFIX}_aligned_tma_{variant}"):
        return True
    if kernel.launch.algorithm == (
            f"{_SGLANG_FP8_BLOCK_ALGORITHM_PREFIX}_aligned_{variant}"):
        return False
    raise ValueError(
        "SGLang FP8 block FUSED_MOE algorithm must name its aligned "
        f"{variant} program: {kernel.launch.algorithm!r}")


def _fp8_block_assignment(kernel: KernelSpec) -> str:
    if _is_sglang_fp8_block_launch(kernel):
        _sglang_fp8_block_uses_tma(kernel)
        return "aligned"
    variant = kernel.variant.lower()
    for assignment in ("naive", "aligned"):
        if kernel.launch.algorithm == (
                f"{_FP8_BLOCK_ALGORITHM_PREFIX}_{assignment}_{variant}"):
            return assignment
    raise ValueError(
        "FP8 block FUSED_MOE algorithm must name the naive or aligned "
        f"{variant} source program: {kernel.launch.algorithm!r}")


def _fp8_block_counts(kernel: KernelSpec) -> Tuple[Tuple[int, ...], int]:
    """Per-expert rows the kernel sees and its top-k.

    With SGLang shared-experts fusion the shared expert is expert E of the
    routed kernel and every token is routed to it (topk.py appends the
    column), so it adds one expert with ``tokens`` rows and one to top-k.
    """

    _, local_counts = _routing(kernel)
    topk = int(kernel.shape["topk"])
    if "fused_shared_experts" in kernel.attributes:
        if not _is_sglang_fp8_block_launch(kernel):
            raise ValueError(
                "fused shared experts require the SGLang fused_moe program")
        fused = int(kernel.attributes["fused_shared_experts"])
        tokens = int(kernel.shape["tokens"])
        return tuple(local_counts) + (tokens,) * fused, topk + fused
    return tuple(local_counts), topk


def _fp8_block_tile_pipeline(kernel: KernelSpec, hardware):
    """Return the measured FP8 tile-program row selected by the launch."""

    tile_m, tile_n, tile_k = _tile(kernel)
    parameters = kernel.launch.parameters
    warps, stages = (
        int(parameters[name]) for name in ("num_warps", "num_stages"))
    program = f"bm{tile_m}_bn{tile_n}_bk{tile_k}_w{warps}_s{stages}"
    if _is_sglang_fp8_block_launch(kernel):
        prefix = "sgl_tma_" if _sglang_fp8_block_uses_tma(kernel) else "sgl_"
        program = prefix + program
    family = hardware.grouped_gemm_fp8_block_tile_pipelines
    if family is None or program not in family:
        raise NotImplementedError(
            "FP8 block FUSED_MOE tile program has no measured pipeline "
            f"mechanism on {hardware.hardware_id}: {program}")
    row = family[program]
    if kernel.launch.block_size != row["threads_per_cta"]:
        raise ValueError("FP8 block FUSED_MOE block disagrees with its program")
    if (int(parameters["dynamic_shared_memory_bytes"])
            != row["dynamic_shared_memory_bytes_per_cta"]):
        raise ValueError(
            "FP8 block FUSED_MOE shared memory disagrees with its program")
    return program, row


def _fp8_block_traffic(kernel: KernelSpec) -> Tuple[_GpuGroupedGemmTraffic, int]:
    """Source-derived FP8 block grouped-GEMM hierarchy traffic.

    ``fused_moe_kernel`` with ``use_fp8_w8a8`` and a [128, 128] block loads,
    per K block of every populated program, the valid FP8 A rows (one byte),
    the complete FP8 B panel (the N coordinate wraps, it is not masked), one
    FP32 per-token-group A scale per valid row and one FP32 B block scale
    (all ``BLOCK_SIZE_N`` lanes address the same ``offs_bn // 128`` block).
    The naive assignment gives every route its own M tile with one valid
    row; the aligned assignment packs each expert's routes into
    ``ceil(rows / BLOCK_SIZE_M)`` tiles.  W2 loads one FP32 router weight
    per valid row after its K loop (source routed-weight marker).  The BF16
    output is masked to real rows and columns.  DRAM carries the unique
    compulsory bytes; no cross-call residency is assumed.
    Returns the traffic and the populated M-tile count.
    """

    variant, source_rows, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    sglang = _is_sglang_fp8_block_launch(kernel)
    # SGLang's _down configs also use BLOCK_SIZE_K 64: the kernel then reads
    # the same 128x128 scale block on each of its two K rounds.
    allowed_tile_k = (64, _FP8_BLOCK_SIZE) if sglang else (_FP8_BLOCK_SIZE,)
    if tile_k not in allowed_tile_k or projection_k % _FP8_BLOCK_SIZE:
        raise ValueError(
            "FP8 block FUSED_MOE requires complete 128-wide K blocks")
    if projection_n % _FP8_BLOCK_SIZE:
        raise ValueError(
            "FP8 block FUSED_MOE requires complete 128-wide N blocks")
    if int(kernel.attributes["expert_parallel_size"]) != 1:
        raise NotImplementedError(
            "FP8 block FUSED_MOE is source-backed only without expert "
            "parallelism")
    global_counts, local_counts = _routing(kernel)
    if global_counts != local_counts:
        raise ValueError(
            "FP8 block FUSED_MOE without EP must own every routed row")
    local_counts, kernel_topk = _fp8_block_counts(kernel)
    routed_rows = sum(local_counts)
    if routed_rows != int(kernel.shape["tokens"]) * kernel_topk:
        raise ValueError(
            "FP8 block FUSED_MOE routing does not conserve tokens * topk")
    if variant == "w2":
        source_rows = routed_rows
    assignment = _fp8_block_assignment(kernel)
    if assignment == "naive":
        populated_m_tiles = routed_rows
    else:
        populated_m_tiles = sum(
            math.ceil(count / tile_m) for count in local_counts if count > 0)
    n_tiles = math.ceil(projection_n / tile_n)
    grid_m_tiles, remainder = divmod(kernel.launch.grid_size, n_tiles)
    if remainder or grid_m_tiles != math.ceil(
            int(kernel.launch.parameters["padded_rows"]) / tile_m):
        raise ValueError(
            "FP8 block FUSED_MOE grid disagrees with its expanded rows")
    if populated_m_tiles > grid_m_tiles:
        raise ValueError(
            "FP8 block FUSED_MOE populated tiles exceed the launch grid")
    k_blocks = projection_k // _FP8_BLOCK_SIZE
    k_rounds = projection_k // tile_k
    active_experts = sum(1 for count in local_counts if count > 0)
    activation_row_bytes = projection_k + k_rounds * _FP8_SCALE_BYTES
    # A TMA descriptor load (SGLang _down) moves the whole BLOCK_SIZE_M x
    # BLOCK_SIZE_K box of the expert-sorted A, padding rows included.
    tma_a = sglang and _sglang_fp8_block_uses_tma(kernel)
    loaded_a_rows = populated_m_tiles * tile_m if tma_a else routed_rows
    activation_l2 = float(loaded_a_rows * activation_row_bytes * n_tiles)
    weight_l2 = float(
        populated_m_tiles * n_tiles
        * (projection_k * tile_n + k_rounds * _FP8_SCALE_BYTES))
    if variant == "w1":
        unique_source_rows = min(
            source_rows, math.ceil(routed_rows / kernel_topk))
    elif variant == "w2":
        unique_source_rows = loaded_a_rows
    else:
        raise AssertionError(f"unexpected FUSED_MOE variant: {variant!r}")
    activation_compulsory = float(unique_source_rows * activation_row_bytes)
    weight_compulsory = float(
        active_experts
        * (projection_k * projection_n
           + (projection_n // _FP8_BLOCK_SIZE) * k_blocks * _FP8_SCALE_BYTES))
    routed_weight_l2, routed_weight_compulsory = (
        _source_routed_weight_read_traffic(
            kernel, routed_rows=float(routed_rows), n_tiles=n_tiles))
    output_bytes = float(
        routed_rows * projection_n * int(kernel.shape["dtype_bytes"]))
    traffic = _GpuGroupedGemmTraffic(
        activation_l2_read_bytes=activation_l2,
        weight_l2_read_bytes=weight_l2,
        routed_weight_l2_read_bytes=routed_weight_l2,
        output_l2_write_bytes=output_bytes,
        activation_compulsory_read_bytes=activation_compulsory,
        weight_compulsory_read_bytes=weight_compulsory,
        routed_weight_compulsory_read_bytes=routed_weight_compulsory,
        activation_dram_read_bytes=activation_compulsory,
        weight_dram_read_bytes=weight_compulsory,
        routed_weight_dram_read_bytes=routed_weight_compulsory,
        output_dram_write_bytes=output_bytes,
    )
    validated_l2_miss_rate(traffic.dram_total_bytes, traffic.l2_total_bytes)
    return traffic, populated_m_tiles


def _fp8_block_model(kernel: KernelSpec, hardware) -> Tuple[PredictionInput, float]:
    """Price the FP8 block Triton launch with the grouped-GEMM service model.

    The template is the resident-wave K pipeline of the BF16
    ``grouped_gemm_tile_pipeline``: one fixed span, then per resident wave one
    service round per BK128 block, each round the larger of the measured
    local K-round interval of this tile program and the DRAM round service,
    plus the early-return CTA tail.  Aggregate hierarchy traffic and Tensor
    work stay independent resource bounds.  Tensor operations are expressed
    in the primary Tensor variant's units using the HWSpec FP8/primary rate
    ratio; the per-block dequantization ``acc += dot * a_scale * b_scale``
    is FMA-pipe work (one multiply and one fused multiply-add per
    accumulator element per K block).
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise NotImplementedError(
            "FP8 block FUSED_MOE is modeled only for GPU memory hierarchies")
    if kernel.dtype != "bf16":
        raise ValueError("FP8 block FUSED_MOE requires BF16 compute dtype")
    if kernel.attributes["routed_weight_dtype"] != "fp8":
        raise ValueError("FP8 block FUSED_MOE requires FP8 routed weights")
    variant, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, _ = _tile(kernel)
    program, pipeline_row = _fp8_block_tile_pipeline(kernel, hardware)
    traffic, populated_m_tiles = _fp8_block_traffic(kernel)
    local_counts, _ = _fp8_block_counts(kernel)
    routed_rows = float(sum(local_counts))
    n_tiles = math.ceil(projection_n / tile_n)
    # K rounds of the tile program (BLOCK_SIZE_K 64 or 128); the scaled dot
    # accumulates once per round.
    k_blocks = projection_k // _tile(kernel)[2]
    fp8_rate = hardware.resource_by_role("compute_tensor_fp8").capacity
    primary_rate = hardware.resource_by_role("compute_tensor").capacity
    if fp8_rate <= 0.0 or primary_rate <= 0.0:
        raise ValueError("FP8 block FUSED_MOE requires positive Tensor rates")
    fp8_tensor_flops = float(
        2 * populated_m_tiles * tile_m * projection_k * n_tiles * tile_n)
    dequantization_flops = float(
        3 * populated_m_tiles * tile_m * n_tiles * tile_n * k_blocks)
    flops = {
        "compute_tensor": fp8_tensor_flops * primary_rate / fp8_rate,
        "compute_fma": dequantization_flops,
        "compute_alu": _source_routed_weight_scale_operations(
            kernel, routed_rows=routed_rows, projection_n=projection_n),
        "compute_sfu": 0.0,
    }
    demand = build_forward_demand(
        traffic.l2_total_bytes, traffic.dram_total_bytes, False, flops)
    eta, wave_efficiency, active_units = launch_wave_geometry(kernel, hardware)
    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=variant,
        mem_model="gpu_mshr",
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind="generic",
        include_dispatch_overhead=True,
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        support_level="aggregate_only",
        limitations=gpu_partial_limitations(
            hardware,
            "moe_gather_service_class_not_separated",
            "moe_physical_worker_assignment_unavailable",
            "moe_l2_residency_order_unavailable",
            "moe_load_compute_store_buffer_lifecycle_unavailable",
        ),
    )
    producers, requests_per_producer = gpu_memory_source(hardware)
    structured = attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name="global_memory",
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=None,
    )
    structured = _split_gpu_grouped_gemm_memory_paths(structured, traffic)
    pipeline_plan = GpuGroupedGemmPipelineEnvelopeDemand(
        active_ctas=populated_m_tiles * n_tiles,
        k_tiles=k_blocks,
        resident_ctas_per_sm=pipeline_row[
            "resident_ctas_per_sm_by_variant"][variant],
        dram_read_bytes=traffic.dram_read_bytes,
        fp8_block_program=program,
    )
    structured = dataclasses.replace(
        structured,
        pipeline_plan=pipeline_plan,
        phase_ledger=_grouped_matmul_phase_ledger(structured),
    )
    return structured, traffic.compulsory_bytes


def _model(
        kernel: KernelSpec, hardware, *, attach_pipeline: bool = True,
        ) -> Tuple[PredictionInput, float]:
    _validate_expert_parallel_launch(kernel)
    if kernel.variant.lower() == "routed_experts":
        return _routed_experts_model(kernel, hardware)
    if _is_fp8_block_launch(kernel):
        return _fp8_block_model(kernel, hardware)
    variant, _, projection_k, projection_n = _projection(kernel)
    _, local_counts = _routing(kernel)
    _, tile_n, _ = _tile(kernel)
    memory_model = hardware.kernel_memory_model
    if memory_model == "npu_tile":
        cann_tensor_work = _cann_grouped_matmul_work(kernel)
        # GroupedMatmul's source order and the physical shared-L2 capacity
        # determine the requests that escape to DRAM.  Preserve that
        # source-derived residency accounting in the compressed analytical
        # envelope; replacing it with a unique-footprint lower bound erases
        # the repeated operand traffic of populated expert tiles.
        cann_residency = _cann_grouped_operand_residency(
            kernel, cann_tensor_work, hardware)
        cann_traffic = _cann_grouped_matmul_traffic(
            kernel, cann_tensor_work, cann_residency)
        gpu_traffic = None
        l2_total = cann_traffic.l2_total_bytes
        dram_bytes = cann_traffic.dram_total_bytes
        compulsory = dram_bytes
    elif memory_model == "gpu_mshr":
        gpu_traffic = _gpu_traffic_components(kernel, hardware)
        l2_total = gpu_traffic.l2_total_bytes
        dram_bytes = gpu_traffic.dram_total_bytes
        compulsory = gpu_traffic.compulsory_bytes
        cann_tensor_work = None
        cann_traffic = None
        cann_residency = None
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")

    routed_rows = float(sum(local_counts))
    if cann_tensor_work is None:
        tile_m, tile_n, _ = _tile(kernel)
        populated_m_tiles = sum(
            math.ceil(count / tile_m)
            for count in local_counts if count > 0)
        # Triton masks operand loads and output stores at the logical M/N
        # boundaries, but tl.dot still issues the complete tile.  Price the
        # physical MMA fragments rather than only the unpadded routed rows.
        tensor_flops = float(
            2
            * populated_m_tiles
            * tile_m
            * projection_k
            * math.ceil(projection_n / tile_n)
            * tile_n)
    else:
        tensor_flops = float(cann_tensor_work.tensor_operations)
    routed_weight_scale_operations = _source_routed_weight_scale_operations(
        kernel, routed_rows=routed_rows, projection_n=projection_n)
    flops = {
        "compute_tensor": tensor_flops,
        "compute_fma": 0.0,
        "compute_alu": routed_weight_scale_operations,
        "compute_sfu": 0.0,
    }
    validated_l2_miss_rate(dram_bytes, l2_total)
    demand = build_forward_demand(l2_total, dram_bytes, False, flops)
    eta, wave_efficiency, active_units = (
        launch_wave_geometry(
            kernel, hardware)
    )
    dram_efficiency = hardware.dram_protocol_efficiency_l1cached
    if memory_model == "npu_tile":
        pass
    elif memory_model == "gpu_mshr":
        n_tiles = math.ceil(projection_n / tile_n)
        if kernel.launch.grid_size % n_tiles:
            raise ValueError(
                "FUSED_MOE launch grid must be divisible by N-tile count")
    else:
        raise AssertionError(
            f"unhandled FUSED_MOE memory model: {memory_model!r}")

    if memory_model == "npu_tile":
        launch_task_kind = "aic"
        include_launch_floor, launch_limitations = (
            npu_launch_floor_contract(hardware, launch_task_kind))
        support_level = "aggregate_only"
        limitations = (
            "aic_grouped_matmul_dram_residency_uses_source_order_lru",
            "aic_output_writeback_dram_residency_unavailable",
            "aic_store_fixpipe_tile_lifecycle_unavailable",
            "aic_read_write_transaction_split_unavailable",
            "mte_transaction_geometry_unavailable",
            "memory_first_request_latency_unavailable",
            "mte2_descriptor_and_credit_evidence_unavailable",
            "workload_producer_independence_unavailable",
        ) + launch_limitations
    elif memory_model == "gpu_mshr":
        launch_task_kind = "generic"
        include_launch_floor = True
        support_level = "aggregate_only"
        limitations = gpu_partial_limitations(
            hardware,
            "moe_gather_service_class_not_separated",
            "moe_physical_worker_assignment_unavailable",
            "moe_l2_residency_order_unavailable",
            "moe_load_compute_store_buffer_lifecycle_unavailable",
        )
    else:
        raise AssertionError(
            f"unhandled FUSED_MOE memory model: {memory_model!r}")

    if memory_model == "npu_tile":
        if cann_tensor_work is None:
            raise AssertionError("CANN GroupedMatmul work is missing")
        if cann_traffic is None:
            raise AssertionError("CANN GroupedMatmul physical traffic is missing")
        if cann_residency is None:
            raise AssertionError("CANN GroupedMatmul residency is missing")
        return (
            _cann_grouped_matmul_analytical_input(
                kernel,
                hardware,
                demand_per_resource=demand,
                eta=eta,
                wave_efficiency=wave_efficiency,
                active_units=active_units,
                include_dispatch_overhead=include_launch_floor,
                limitations=limitations,
                traffic=cann_traffic,
                work=cann_tensor_work,
                residency=cann_residency,
                attach_pipeline=attach_pipeline,
            ),
            compulsory,
        )
    if gpu_traffic is None:
        raise AssertionError("GPU grouped-GEMM traffic is missing")
    nvidia_source_service = _nvidia_source_depth_operand_service(
        kernel, hardware)
    source_completion_services = {
        _NVIDIA_BM16_H2880_31_ACTIVE_SERVICE,
        _NVIDIA_BM16_H2880_53_ACTIVE_SERVICE,
        _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE,
        _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE,
        _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W1_SERVICE,
        _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W2_SERVICE,
        _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W1_SERVICE,
        _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W2_SERVICE,
        _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W1_SERVICE,
        _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W2_SERVICE,
    }
    pipeline_plan = (
        None
        if (nvidia_source_service is not None
            and nvidia_source_service.service in source_completion_services)
        else (
            _gpu_grouped_gemm_pipeline_envelope(
                kernel, hardware, gpu_traffic)
            or _metax_grouped_operand_pipeline_envelope(
                kernel, hardware, gpu_traffic)))
    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=variant,
        mem_model=memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind=launch_task_kind,
        include_dispatch_overhead=include_launch_floor,
        dram_capacity_multiplier=dram_efficiency,
        support_level=support_level,
        limitations=limitations,
    )
    normalized = _architecture_input(inp, kernel, hardware)
    if memory_model != "gpu_mshr":
        raise AssertionError(
            f"unhandled FUSED_MOE memory model: {memory_model!r}")
    producers, requests_per_producer = gpu_memory_source(hardware)
    transaction_bytes = None
    path_name = "global_memory"
    frontend_rate = None
    structured = attach_structured_contract(
        normalized,
        hardware,
        transaction_bytes=transaction_bytes,
        path_name=path_name,
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=frontend_rate,
    )
    structured = _apply_metax_bm32_fused_moe_mma_occupancy(
        structured, kernel, hardware)
    if l2_total > 0:
        if gpu_traffic is None:
            raise AssertionError("GPU grouped-GEMM traffic is missing")
        structured = _split_gpu_grouped_gemm_memory_paths(
            structured, gpu_traffic)
        if routed_rows > 0:
            structured = _attach_gpu_gather_service(
                structured,
                kernel,
                hardware,
                traffic=gpu_traffic,
                routed_rows=routed_rows,
                projection_k=projection_k,
                projection_n=projection_n,
                element_bytes=_gpu_operand_element_bytes(kernel),
            )
    structured = dataclasses.replace(
        structured,
        pipeline_plan=pipeline_plan,
        phase_ledger=_grouped_matmul_phase_ledger(structured),
    )
    return structured, compulsory


class FusedMoEPredictor(KernelPredictor):
    kind = KernelKind.FUSED_MOE

    def __init__(self):
        self._cached_resource_kernel = None
        self._cached_resource_hardware = None
        self._cached_resource_value = None

    def resource_input(self, kernel: KernelSpec, hardware):
        # One evaluator case first materializes this analytical input for its
        # Formula-Ceiling observation, then immediately invokes Forward and
        # Roofline for the exact same immutable KernelSpec.  The source-derived
        # L2 replay and grouped-matmul envelope are independent of the method,
        # so retain just that one current-object construction.  Return fresh
        # outer mappings so callers cannot mutate the retained input; object
        # identity keeps the cache strictly within the currently evaluated
        # spec rather than becoming a shape/result lookup table.
        if (self._cached_resource_kernel is kernel
                and self._cached_resource_hardware is hardware):
            if self._cached_resource_value is None:
                raise RuntimeError("FusedMoE resource-input cache is incomplete")
            cached_input, cached_compulsory = self._cached_resource_value
            return dataclasses.replace(
                cached_input,
                demand_per_resource=dict(cached_input.demand_per_resource),
                resource_time_overrides_ns=dict(
                    cached_input.resource_time_overrides_ns),
            ), cached_compulsory
        resource_input, compulsory = _model(kernel, hardware)
        self._cached_resource_kernel = kernel
        self._cached_resource_hardware = hardware
        self._cached_resource_value = (resource_input, compulsory)
        return resource_input, compulsory

    def midlayer(self, kernel: KernelSpec, hardware):
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(
            _with_graph_regime(inp, kernel))

    def roofline(self, kernel: KernelSpec, hardware):
        # Roofline consumes only aggregate demand and compulsory traffic, both
        # of which are identical in the complete Forward input.  Reusing it
        # avoids rebuilding the source residency census and then dropping its
        # pipeline representation without changing the roofline calculation.
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory), "roofline")

    def pipeweave(self, kernel: KernelSpec, hardware, *, model_hardware=None):
        if kernel.attributes["expert_parallel_size"] != 1:
            raise NotImplementedError(
                "PipeWeave FUSED_MOE does not model rank-local expert "
                "parallel execution")
        from src.kernel.predictors.pipeweave_adapter import predict_fused_moe
        return predict_fused_moe(
            kernel, hardware, model_hardware=model_hardware)




ROUTING_ORDER_SENSITIVE_RECIPE_VARIANTS = frozenset((
    # This kernel walks the routing histogram as a cumulative prefix stream and
    # resolves a sorted position back to the expert index that produced it, so
    # it reads the axis in a finer way than the grouped projections do and the
    # relabelling below is not established for it.
    "moe_compute_expert_tokens",
))


def _expert_axis_class(counts, owned_experts):
    """Decompose one routing histogram into its physical class.

    The grouped projection turns each populated expert of this rank's shard
    into M tiles, in launch order; every other term the predictor derives from
    the axis is a sum over the whole axis or over its populated entries.  The
    class is therefore the shard's populated counts in launch order, how many
    of its experts stayed empty, and the multiset of the counts outside it.
    """

    shard_first = owned_experts[0]
    shard_last = owned_experts[-1]
    if shard_last - shard_first != len(owned_experts) - 1:
        raise ValueError(
            "FUSED_MOE expert shard must be a contiguous expert range")
    shard_populated = []
    shard_empty = 0
    other_populated = []
    other_empty = 0
    for expert, count in enumerate(counts):
        if shard_first <= expert <= shard_last:
            if count:
                shard_populated.append(count)
            else:
                shard_empty += 1
        elif count:
            other_populated.append(count)
        else:
            other_empty += 1
    other_populated.sort(reverse=True)
    return (
        shard_first,
        shard_last,
        tuple(shard_populated),
        shard_empty,
        tuple(other_populated),
        other_empty,
    )


def canonical_expert_axis(tokens_per_expert, owned_experts):
    """Return the canonical routing of this histogram's physical class.

    The canonical member keeps each populated shard expert's rows in launch
    order and moves the shard's empty experts behind them, and orders the
    counts outside the shard by size.  The relabelling never crosses the shard
    boundary, so the contiguous expert placement, the local-equals-global mask
    inside the shard, and ``tokens * topk`` conservation all survive.
    """

    counts = tuple(tokens_per_expert)
    owned = tuple(owned_experts)
    (
        shard_first, shard_last, shard_populated, shard_empty,
        other_populated, other_empty,
    ) = _expert_axis_class(counts, owned)
    shard_values = iter(shard_populated + (0,) * shard_empty)
    other_values = iter(other_populated + (0,) * other_empty)
    canonical = tuple(
        next(shard_values) if shard_first <= expert <= shard_last
        else next(other_values)
        for expert in range(len(counts))
    )
    canonical_local = tuple(
        count if shard_first <= expert <= shard_last else 0
        for expert, count in enumerate(canonical)
    )
    return canonical, canonical_local


def prediction_equivalence_attributes(kernel: KernelSpec):
    """Name the physical routing class of one grouped-MoE launch.

    Every term this predictor derives from the routing histogram is one of
    three things: a sum over the whole expert axis, a sum over the populated
    experts, or the ordered sequence of populated local experts that the
    grouped launch turns into M tiles.  An expert index itself never enters a
    byte count, a tile count, or a service: it reaches the replay only as the
    label that separates one expert's weight objects from another's.

    The class therefore replaces the three expert-indexed contract fields.
    The shard is the contiguous placement of ``expert_rank`` inside
    ``expert_parallel_size`` and ``experts``, all of which the surrounding
    contract still carries, so the class plus that contract names exactly one
    physical routing up to the relabelling the hardware cannot observe.

    Returns ``None`` when the launch carries a per-expert payload that this
    relabelling would not carry with it.
    """

    if kernel.kind is not KernelKind.FUSED_MOE:
        return None
    attributes = kernel.attributes
    if attributes["expert_weight_tile_rows"] is not None:
        # A per-expert residency window is indexed by expert, so the class
        # would have to carry it through the same relabelling.  No launch
        # reaching this path declares one; keep the exact contract instead.
        return None
    equivalence = {
        name: value for name, value in attributes.items()
        if name not in (
            "tokens_per_expert", "local_tokens_per_expert", "owned_experts")
    }
    equivalence["expert_axis_class"] = _expert_axis_class(
        tuple(attributes["tokens_per_expert"]),
        tuple(attributes["owned_experts"]),
    )
    return equivalence
