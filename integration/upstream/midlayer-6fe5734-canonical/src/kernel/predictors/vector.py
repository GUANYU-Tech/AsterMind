"""Complete-spec vector, activation, routing, and reduction prediction."""

import dataclasses
import math
from bisect import bisect_right
from contextvars import ContextVar, copy_context
from functools import lru_cache
from typing import Dict, Optional, Tuple

from src.kernel.bank_stream_model import (
    bank_stream_ns, normalization_stream, pointwise_stream)
from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelPrediction, KernelSpec
from src.kernel.launch_config import LaunchConfig
from src.kernel.model_contract import (
    AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
    AIV_WORKER_START_FILL,
    AnalyticalDependencyPath,
    AnalyticalPhase,
    AnalyticalPhaseLedger,
    CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_PROLOGUE,
    ComputeDemand,
    ComputeExactRequestDemand,
    ComputeServiceDemand,
    MemoryAccessClassDemand,
    MemoryAccessClassSignature,
    MemoryExactRequestDemand,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemorySourceDemand,
    PhaseComputeDemand,
    PhaseMemoryDemand,
    PipelineBuffer,
    PipelineStageDemand,
    PipelineTaskDemand,
    PipelineTaskProgramDemand,
    PipelineTaskSequenceDemand,
    PipelineWorkPlan,
    MEMORY_DEPENDENCY_TASK_INCREMENT_SERVICE,
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
    SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER,
    SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
    SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER,
    SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
    SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
    SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER,
    SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER,
    SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER,
    SOURCE_QK_ROPE_LOOP_PARAMETER,
    SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER,
    SOURCE_SCALAR_KV_LOOP_PARAMETER,
    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
    SOURCE_VECTORIZED_KV_COPY_PARAMETER,
    SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER,
    PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
    PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS,
    TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
    TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR,
    WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
)
from src.kernel.predictors.common import (
    _with_graph_regime,
    PredictionInput,
    append_npu_write_path,
    attach_pipeline_work_plan,
    attach_structured_contract,
    attach_uniform_pipeline,
    bind_pipeline_stage_template,
    bind_typed_compute_pipeline_stage_template,
    build_forward_demand,
    gpu_memory_source,
    gpu_partial_limitations,
    launch_wave_geometry,
    npu_launch_floor_contract,
    prediction_from_ns,
    roofline_ns,
    split_global_memory_paths,
)


# One scheduler-visible completion synchronization, measured independently by
# the CANN cross-domain microbenchmark (630 ns serial round trip).  This
# remains an Oracle-side single-mechanism observation.  Forward does not use
# it as a fitted production-kernel floor.
_CANN_NORM_COMPLETION_SYNC_NS = 630.0
_A800_ADD_RMS_MIXED_L2_READ_SERVICE = (
    "bf16_vec8_add_rmsnorm_mixed_l2_read")
_A800_ADD_RMS_MIXED_L2_WRITE_SERVICE = (
    "bf16_vec8_add_rmsnorm_mixed_l2_write")
_SOURCE_BOUND_RESIDENT_CTAS_PER_UNIT_PARAMETER = (
    "source_bound_resident_ctas_per_unit")
_A800_ADD_RMS_MIXED_DRAM_READ_SERVICE = (
    "bf16_vec8_add_rmsnorm_mixed_dram_read")
_A800_ADD_RMS_MIXED_DRAM_WRITE_SERVICE = (
    "bf16_vec8_add_rmsnorm_mixed_dram_write")
_A800_ADD_RMS_MIXED_L2_MAX_WAVES = 16
_C500_ELEMENTWISE_2R1W_DRAM_READ_SERVICE = (
    "bf16_vec4_elementwise_2r1w_dram_read")
_C500_ELEMENTWISE_2R1W_DRAM_WRITE_SERVICE = (
    "bf16_vec4_elementwise_2r1w_dram_write")
_C500_ELEMENTWISE_2R1W_PARTIAL_DRAM_READ_SERVICE = (
    "bf16_vec4_elementwise_2r1w_partial_resident_dram_read")
_C500_ELEMENTWISE_2R1W_PARTIAL_DRAM_WRITE_SERVICE = (
    "bf16_vec4_elementwise_2r1w_partial_resident_dram_write")
_C500_ELEMENTWISE_2R1W_MIN_PARTIAL_RESIDENT_CTAS = 416
_C500_ELEMENTWISE_2R1W_MIN_FULL_RESIDENT_CTAS = 832
_C500_RMSNORM_PHASE_CHAIN_EXCESS_SERVICES = {
    1: "rmsnorm_phase_chain_excess_first_1024",
    2: "rmsnorm_phase_chain_excess_first_1024_active2",
    4: "rmsnorm_phase_chain_excess_first_1024_active4",
    8: "rmsnorm_phase_chain_excess_first_1024_active8",
}
_RMSNORM_PHASE_ENTRY_SERVICES = {
    1: "rmsnorm_phase_entry_first_1024",
    2: "rmsnorm_phase_entry_first_1024_active2",
    4: "rmsnorm_phase_entry_first_1024_active4",
    8: "rmsnorm_phase_entry_first_1024_active8",
    16: "rmsnorm_phase_entry_first_1024_active16",
}
# A800's independently measured grid-64 entry must not alias C500's
# active-one service with the same historical base name.
_RMSNORM_PHASE_ENTRY_ACTIVE64_SERVICE = (
    "rmsnorm_phase_entry_first_1024_active64")
_RMSNORM_PHASE_ENTRY_MIN_DEPTH = 8
_C500_ADD_RMSNORM_PHASE_CHAIN_EXCESS_SERVICES = {
    2: "add_rmsnorm_phase_chain_excess_first_1024_active2",
    4: "add_rmsnorm_phase_chain_excess_first_1024_active4",
    8: "add_rmsnorm_phase_chain_excess_first_1024_active8",
    16: "add_rmsnorm_phase_chain_excess_first_1024_active16",
    32: "add_rmsnorm_phase_chain_excess_first_1024_active32",
}
_C500_ADD_RMSNORM_PHASE_CHAIN_MIN_DEPTH = 8
_C500_ADD_RMSNORM_PHASE_CHAIN_MAX_DEPTH = 32
_C500_COMPLETE_NORM_PHASE_CHAIN_SERVICES = {
    "rmsnorm": "rmsnorm_phase_chain_complete_first_1024",
    "add_rmsnorm": "add_rmsnorm_phase_chain_complete_first_1024",
}
_C500_COMPLETE_NORM_PHASE_CHAIN_MIN_DEPTH = 8
_C500_COMPLETE_NORM_PHASE_CHAIN_MAX_DEPTH = 32
_C500_ADD_RMSNORM_BLOCK256_SCALAR_INPUT_SERVICES = {
    "first": "add_rmsnorm_scalar_input_add_square_first_256",
    "resident": "add_rmsnorm_scalar_input_add_square_resident_256",
}
_C500_ADD_RMSNORM_BLOCK256_SCALAR_ACTIVE_UNITS = (
    1, 2, 4, 8, 16, 32, 64, 104)
_C500_ADD_RMSNORM_BLOCK256_SCALAR_MIN_DEPTH = 4
_C500_ADD_RMSNORM_BLOCK256_SCALAR_MAX_DEPTH = 64
_C500_ADD_RMSNORM_SCALAR_FIRST_SERVICES = {
    "add_square": "add_rmsnorm_input_add_square_first_1024",
    "scale_multiply": "add_rmsnorm_output_scale_first_1024",
}
_C500_ADD_RMSNORM_SCALAR_FIRST_MIN_DEPTH = 8
_C500_ADD_RMSNORM_SCALAR_FIRST_MAX_DEPTH = 32
_C500_ADD_RMSNORM_GENERIC_FIXED_INSTRUCTIONS_PER_WAVE = 46
_C500_ADD_RMSNORM_GRID64_FIXED_INPUT_SERVICES = {
    depth: f"add_rmsnorm_grid64_fixed_input_loop_depth{depth}_1024"
    for depth in (4, 5, 8, 16, 32)
}

_TYPED_SWIGLU_INVOCATION_PARAMETERS = (
    "vector_exp_serial_invocations",
    "vector_div_serial_invocations",
)

_TYPED_SWIGLU_PIPELINE_PARAMETERS = (
    "swiglu_base_row_len",
    "swiglu_base_col_len",
    "swiglu_total_tiles",
    "swiglu_queue_depth",
)

_CANN_RESHAPE_CACHE_EQUAL_KEY = 200_000_000
_CANN_RESHAPE_CACHE_FALLBACK_KEY = 202_000_000
_CANN_NORM_EXACT_SERVICE = "norm_sqrt_duplicate_div_serialized"
_CANN_RMSNORM_LOCAL_SCALAR_EXACT_SERVICE = (
    "rmsnorm_local_scalar_dependency")
_CANN_RMSNORM_TYPED_SERVICES = (
    "cast_bf16_to_fp32_serialized",
    "cast_fp32_to_bf16_serialized",
    "muls_serialized",
    "mul_serialized",
    "rmsnorm_half_interval_reduce",
)
_CANN_ADD_RMSNORM_REDUCE_SUM_FP32_SERVICE = (
    "add_rmsnorm_reduce_sum_fp32")
_CANN_ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_FULL64_SERVICE = (
    "cann_add_rmsnorm_split_d_block_reduce_full64")
_CANN_ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_TAIL8_SERVICE = (
    "cann_add_rmsnorm_split_d_block_reduce_tail8")
_CANN_SCALAR_GET_VALUE_EXACT_SERVICE = (
    "scalar_global_get_value_serialized")
_CANN_MTE2_COMPLETION_EXACT_SERVICE = "aiv_mte2_request_completion"
_CANN_MTE3_COMPLETION_EXACT_SERVICE = "aiv_mte3_request_completion"

# Source-DAG materialization remains available through ``resource_input`` for
# source-contract validation.  Forward projects the same phase ledger and
# physical services into its final analytical dependency envelope, avoiding
# a Python replay of every AIV worker and tile.
_FORWARD_CANN_ANALYTICAL_ENVELOPE = ContextVar(
    "forward_cann_analytical_envelope", default=False)
_A800_FP16_RMSNORM_SINGLE_CTA_FLOOR = ContextVar(
    "a800_fp16_rmsnorm_single_cta_floor", default=False)
_CANN_KV_EQUAL_CHAIN_EXACT_SERVICES = {
    1: "aiv_reshape_cache_equal_chain_completion",
    48: "aiv_reshape_cache_equal_chain_completion_48_workers",
}
_CANN_KV_EQUAL_WARM_CHAIN_EXACT_SERVICES = {
    2: "aiv_reshape_cache_equal_warm_chain_completion_2_workers",
    4: "aiv_reshape_cache_equal_warm_chain_completion_4_workers",
    8: "aiv_reshape_cache_equal_warm_chain_completion_8_workers",
    16: "aiv_reshape_cache_equal_warm_chain_completion_16_workers",
}
_CANN_KV_UNEQUAL_REPEATED_PAIR_EXACT_SERVICE = (
    "aiv_reshape_cache_unequal_repeated_pair_completion_48_workers_2048b")
_CANN_KV_EQUAL_SINGLE_TBUF_LIFECYCLE_EXACT_SERVICE = (
    "aiv_single_tbuf_lifecycle")
_CANN_KV_EQUAL_SINGLE_TBUF_LIFECYCLE_WORKERS = {1, 2, 4, 8, 16, 48}
_CANN_KV_UNEQUAL_REPEATED_PAIR_EXACT_DOMAIN = {(48, 2048, 22)}
_CANN_ROPE_MTE2_TO_VECTOR_EXACT_SERVICES = {
    1: "aiv_rope_mte2_to_vector_completion",
    32: "aiv_rope_mte2_to_vector_completion_32_workers",
    43: "aiv_rope_mte2_to_vector_completion_43_workers",
    47: "aiv_rope_mte2_to_vector_completion_47_workers",
}
_CANN_ROPE_KEY24_STEADY_MTE2_EXACT_SERVICE = (
    "aiv_rope_key24_mte2_steady_completion_47_workers_4096b")
_CANN_ROPE_KEY24_STEADY_MTE2_EXACT_DOMAIN = {
    (47, 20),
    (47, 21),
}
_CANN_ROPE_VECTOR_TO_MTE3_EXACT_SERVICES = {
    1: "aiv_rope_vector_to_mte3_completion",
    43: "aiv_rope_vector_to_mte3_completion_43_workers",
    47: "aiv_rope_vector_to_mte3_completion_47_workers",
}
_CANN_ROPE_KEY22_FIRST_Q_SLICE_MTE2_TO_VECTOR_EXACT_SERVICES = {
    1: "aiv_rope_key22_first_q_slice_mte2_to_vector_completion",
    4: "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_4_workers",
    8: "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_8_workers",
    16: "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_16_workers",
}
_CANN_ROPE_KEY22_FIRST_Q_SLICE_VECTOR_TO_MTE3_EXACT_SERVICES = {
    1: "aiv_rope_key22_first_q_slice_vector_to_mte3_completion",
    4: "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_4_workers",
    8: "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_8_workers",
    16: "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_16_workers",
}
_CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE = (
    "aiv_datacopypad_mte2_to_vector_completion")
_CANN_DATACOPYPAD_MTE2_TO_VECTOR_TWO_WORKER_EXACT_SERVICE = (
    "aiv_datacopypad_mte2_to_vector_completion_two_workers")
_CANN_ADD_RMS_SINGLE_N_INPUT_PAIR_ONE_WORKER_EXACT_SERVICE = (
    "aiv_add_rmsnorm_single_n_input_pair_completion_one_worker")
_CANN_RMS_KEY0_GAMMA_INPUT_SEQUENCE_EXACT_SERVICE = (
    "aiv_rmsnorm_key0_gamma_input_sequence_completion")
_CANN_DATACOPYPAD_MTE2_TO_VECTOR_ROW_GROUP_EXACT_SERVICES = {
    32: "aiv_datacopypad_mte2_to_vector_completion_32_workers",
    43: "aiv_datacopypad_mte2_to_vector_completion_43_workers",
    47: "aiv_datacopypad_mte2_to_vector_completion_47_workers",
}
_CANN_DATACOPYPAD_VECTOR_TO_MTE3_EXACT_SERVICE = (
    "aiv_datacopypad_vector_to_mte3_completion")
_CANN_DATACOPYPAD_VECTOR_TO_MTE3_TWO_WORKER_EXACT_SERVICE = (
    "aiv_datacopypad_vector_to_mte3_completion_two_workers")
_CANN_RMS_KEY0_FINITE_SOURCE_ENVELOPE_EXACT_SERVICE = (
    "aiv_rmsnorm_key0_finite_source_envelope")
_CANN_RMS_KEY0_FINITE_SOURCE_ENVELOPE_DOMAINS = frozenset({
    (2, 2, 8192),
    (4, 4, 8192),
    (32, 64, 4096),
})
_CANN_SWIGLU_TQUE_MTE2_TO_VECTOR_EXACT_SERVICES = {
    24: "aiv_swiglu_tque_mte2_to_vector_completion_24_workers",
    40: "aiv_swiglu_tque_mte2_to_vector_completion_40_workers",
    48: "aiv_swiglu_tque_mte2_to_vector_completion_48_workers",
}
_CANN_SWIGLU_TQUE_VECTOR_TO_MTE3_EXACT_SERVICES = {
    24: "aiv_swiglu_tque_vector_to_mte3_completion_24_workers",
    40: "aiv_swiglu_tque_vector_to_mte3_completion_40_workers",
    48: "aiv_swiglu_tque_vector_to_mte3_completion_48_workers",
}
_CANN_SWIGLU_TQUE_INPUT_PAIR_EXACT_SERVICES = {
    (24, 2): "aiv_swiglu_tque_input_pair_completion_24_workers_q2",
    (40, 2): "aiv_swiglu_tque_input_pair_completion_40_workers_q2",
    (48, 1): "aiv_swiglu_tque_input_pair_completion_48_workers_q1",
    (48, 2): "aiv_swiglu_tque_input_pair_completion_48_workers_q2",
}
_CANN_SWIGLU_AIV_INGRESS_GROUP_SIZE = 2
_CANN_AIV_BUFFER_LIFECYCLE_EXACT_SERVICE = (
    "aiv_three_queue_two_tbuf_lifecycle")
_CANN_AIV_QUEUE_CONTROL_EXACT_SERVICE = "aiv_three_queue_control_chain"
_CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE = (
    "aiv_four_queue_three_tbuf_lifecycle")
_CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE = (
    "aiv_queue_control_cycle")
_CANN_GATHER_V3_TWO_TQUE_LIFECYCLE_EXACT_SERVICE = (
    "aiv_gather_v3_two_tque_lifecycle")
# CANN 8.5.1 ``BroadcastOneDimSch`` lowers the seven-node BF16 Add/Mul
# elementwise DAG to twelve PipeLock/PipeUnlock lifetimes per source tile:
# two MTE2, nine Vector, and one MTE3.  Exact MTE completion requests already
# contain their matching hard-event lifetime, so only uncovered lifetimes are
# priced separately as scalar control.  The retained static accounting is in
# Hardware-specific evidence files are resolved through HWSpec.
# elementwise_add_mul_dag_accounting_cann_8_5_1.json.  One lifetime is priced
# by the workload-independent AIV queue-control-cycle mechanism.
_CANN_ELEMENTWISE_QUEUE_CONTROL_CYCLES_PER_TILE = 12
# AddWithCastCompute<bfloat16_t> and MulXfp16Op<bfloat16_t> share the same
# seven-node DAG.  DAGSch's LEVEL_2 compile-time allocation is five buffers
# at the widest FP32 node width, hence 20 UB bytes per logical tile element.
_CANN_ELEMENTWISE_TBUF_BYTES_PER_UB_ELEMENT = 5 * 4
_CANN_ELEMENTWISE_TBUF_LIFECYCLE_EXACT_SERVICE = (
    "aiv_elementwise_tbuf_lifecycle_110080b")
_CANN_ELEMENTWISE_TBUF_LIFECYCLE_ALLOCATION_BYTES = 110_080
_CANN_ELEMENTWISE_MUL_COMPUTE_SEGMENT_SERVICES = {
    46: (
        "aiv_elementwise_mul_prefix_46_workers",
        "aiv_elementwise_mul_suffix_46_workers",
    ),
    47: (
        "aiv_elementwise_mul_prefix_47_workers",
        "aiv_elementwise_mul_suffix_47_workers",
    ),
}
_CANN_ELEMENTWISE_2R1W_GM_READ_SERVICE = (
    "cann_bf16_binary_two_read_one_write_gm_read")
_CANN_ELEMENTWISE_2R1W_GM_WRITE_SERVICE = (
    "cann_bf16_binary_two_read_one_write_gm_write")
_CANN_ELEMENTWISE_MUL_Q4Q1_2R1W_GM_READ_SERVICE = (
    "cann_bf16_mul_q4q1_two_read_one_write_gm_read")
_CANN_ELEMENTWISE_MUL_Q4Q1_2R1W_GM_WRITE_SERVICE = (
    "cann_bf16_mul_q4q1_two_read_one_write_gm_write")
_CANN_ELEMENTWISE_MUL_Q4Q1_FINITE_COMPLETION_SERVICES = {
    (11_008, 2): (
        "aiv_mul_q4q1_finite_worker_completion_48_workers_11008b_x2"),
    (13_824, 2): (
        "aiv_mul_q4q1_finite_worker_completion_48_workers_13824b_x2"),
    (14_592, 3): "aiv_mul_q4q1_finite_worker_completion_48_workers",
    (18_432, 3): (
        "aiv_mul_q4q1_finite_worker_completion_48_workers_18432b_x3"),
}
_CANN_ELEMENTWISE_MUL_Q4Q1_FIVE_TILE_FINITE_COMPLETION_SERVICE = (
    "aiv_mul_q4q1_finite_worker_completion_48_workers_17664b_x5")
# Routing-mask compare/cast element ABI (bytes read, bytes written per
# element).  aclnnNeScalar_NotEqual consumes one int32 and writes one bool;
# the dynamic Cast promotes one bool to one BF16.  The source-level ABI, not
# the BF16 spec dtype, fixes the byte envelope of these two ops.
_CANN_ELEMENTWISE_COMPARE_CAST_SOURCE_ABI = {
    "notequal_int32_bool": (4, 1),
    "cast_bool_bf16": (1, 2),
}
_CANN_ELEMENTWISE_COMPARE_CAST_SERVICES = {
    "notequal_int32_bool": "notequal_int32_bool_serialized",
    "cast_bool_bf16": "cast_bool_bf16_serialized",
}
_CANN_ADD_RMSNORM_LOCAL_SCALAR_HANDOFF_EXACT_SERVICE = (
    "add_rmsnorm_local_scalar_handoff")
_CANN_ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_HANDOFF_EXACT_SERVICE = (
    "add_rmsnorm_split_d_local_scalar_handoff")
_CANN_VECTOR_TO_MTE2_FENCE_EXACT_SERVICE = "vector_to_mte2_fence"
_CANN_SCALAR_U64_DIVREM_EXACT_SERVICE = (
    "aiv_scalar_u64_divrem_dependency")
_CANN_SWIGLU_TILE_ADDRESS_EXACT_SERVICE = (
    "aiv_swiglu_tile_address_control")
# CANN 8.5.1 ``swi_glu.cpp`` materializes the typed payload once at kernel
# entry, then ``SwigluSingle::GetTilingAndOffset`` materializes it again from
# the original GM address.  Both calls precede the tile loop.
_CANN_SWIGLU_TILING_MATERIALIZATION_PHASES = 2
_CANN_NORM_SPLIT_ROW_FACTOR = 64
_CANN_RMSNORM_RSTD_VALUES_PER_TRANSACTION = 8
_CANN_RMSNORM_RSTD_TRANSACTION_BYTES = 32
_CANN_ADD_RMS_SINGLE_N_DAG_PARAMETER = (
    "source_bound_add_rmsnorm_single_n_dag")
_CANN_ADD_RMS_ROW_GROUP_DAG_PARAMETER = (
    "source_bound_add_rmsnorm_row_group_dag")
_CANN_ADD_RMS_SPLIT_D_DAG_PARAMETER = (
    "source_bound_add_rmsnorm_split_d_dag")
_CANN_RMS_SINGLE_ROW_DAG_PARAMETER = (
    "source_bound_rmsnorm_single_row_dag")
_CANN_RMS_ROW_GROUP_DAG_PARAMETER = (
    "source_bound_rmsnorm_row_group_dag")
_CANN_RMS_SPLIT_D_DAG_PARAMETER = (
    "source_bound_rmsnorm_split_d_dag")
_CANN_RMS_MERGE_N_DAG_PARAMETER = (
    "source_bound_rmsnorm_merge_n_dag")
_CANN_MERGE_N_BF16_ELEMENTS_PER_BLOCK = 16
# CANN 8.5.1 ``KernelRmsNormMergeN`` row-group Vector primitives, measured as
# whole source requests by the independent AIV opcode campaign.
_CANN_MERGE_N_REDUCE_SUM_MULTI_N_EXACT_SERVICE = (
    "merge_n_reduce_sum_multi_n")
_CANN_MERGE_N_BROADCAST_GAMMA_EXACT_SERVICE = (
    "merge_n_broadcast_gamma")
_CANN_MERGE_N_BROADCAST_RSTD_EXACT_SERVICE = (
    "merge_n_broadcast_rstd")
_CANN_MERGE_N_PERFORMANCE_RSTD_BRCB_EXACT_SERVICE = (
    "merge_n_performance_rstd_brcb")
_CANN_MERGE_N_PERFORMANCE_MUL_ROW_RSTD_EXACT_SERVICE = (
    "merge_n_performance_mul_row_rstd")
_CANN_MERGE_N_PERFORMANCE_MUL_ROW_GAMMA_EXACT_SERVICE = (
    "merge_n_performance_mul_row_gamma")
_CANN_ADD_RMS_SPLIT_D_TILE_ELEMENTS = 8_192
_CANN_ATB_BF16_ROPE_DAG_PARAMETER = (
    "source_bound_atb_bf16_rope_dag")
_CANN_LOCAL_SCALAR_FILL_SV_EXACT_SERVICE = "local_scalar_fill_s_v"
_CANN_LOCAL_UB_COPY_CONTIGUOUS_EXACT_SERVICE = (
    "local_ub_copy_contiguous")
_CANN_LOCAL_UB_COPY_STRIDE_64B_EXACT_SERVICE = (
    "local_ub_copy_stride_64b")
_CANN_LOCAL_UB_COPY_STRIDE_128B_EXACT_SERVICE = (
    "local_ub_copy_stride_128b")
_SOURCE_BOUND_BLOCK_REDUCTION_PARAMETER = "source_bound_block_reduction"
_BLOCK_REDUCTION_RESIDENT_CTAS = {128: 16, 256: 8, 1024: 2}
_BLOCK_REDUCTION_ACTIVE_UNIT_POINTS = (1, 2, 4, 8, 16, 32, 64, 104)
# MACA 3.7.2 mcProfiler non-duration instruction-cycle counters for the
# version-pinned legacy vLLM 0.8.5 RMSNorm programs.  Values are raw Vector
# execution cycles summed over all waves in one CTA, before parallel-worker
# normalization.  The block128/H128 binary has 1,804 cycles for its two
# waves.  The block1024 CFG has 5,268 fixed cycles plus 2,496 cycles per
# scalar source-loop round across its sixteen waves.  A CTA can distribute
# this work over no more than its launched waves or the four physical PEUs in
# one AP.  Both counts already include reduction instructions, so the
# normalized demand is a competing whole-program floor rather than another
# serial phase duration.
# Evidence: ``rmsnorm_block128_program_counters_20260911_r1`` reports
# 188,064 workgroups, 376,128 waves, the 1,804-cycle Vector counter sum per
# workgroup, and RM dispatch exclusively to P0.  The block1024 affine counter
# law comes from the same MACA 3.7.2 code object family.  Neither law consumes
# profiler ``during`` or any canonical/evaluation duration.
_C500_LEGACY_RMSNORM_RAW_VECTOR_CYCLES = {
    128: (1804.0, 0.0, 1, 1),
    1024: (5268.0, 2496.0, 1, None),
}
_C500_LEGACY_RMSNORM_COMPLETION_SERVICE = (
    "rmsnorm_block128_wave_lifetime_completion")
_SILU_SCALAR_ROUND_SERVICE = "silu_scalar_round"
_A800_SILU_NONRECURRENT_SERVICES = {
    regime: {
        component: (
            f"silu_nonrecurrent_{regime}_{component}_block1024")
        for component in ("entry", "round")
    }
    for regime in ("first", "resident", "queued")
}
_MOE_ROUTING_GLOBAL_ATOMIC_SERVICES = {
    8: "moe_routing_global_atomic_e8",
    32: "moe_routing_global_atomic_e32",
    128: "moe_routing_global_atomic_e128",
}
_SILU_MATH_SERVICE_THREADS = 512
_EXP_DEPENDENCY_SERVICE = "exp_dependency"
_RECIPROCAL_ONE_PLUS_DEPENDENCY_SERVICE = (
    "reciprocal_one_plus_dependency")
_INTEGER_DIV_U32_DEPENDENCY_SERVICE = "integer_div_u32_dependency"
_INTEGER_REM_U32_DEPENDENCY_SERVICE = "integer_rem_u32_dependency"
_INTEGER_DIV_U64_DEPENDENCY_SERVICE = "integer_div_u64_dependency"
_INTEGER_REM_U64_DEPENDENCY_SERVICE = "integer_rem_u64_dependency"
_BF16_ROUNDTRIP_PAIR_DEPENDENCY_SERVICE = (
    "bf16_roundtrip_pair_dependency")
_ADDRESS_MULTIPLY_ADD_U64_DEPENDENCY_SERVICE = (
    "address_multiply_add_u64_dependency")
# [MEASURED_MECHANISM] Integer AIV cycles from an independent CANN 8.5.1
# UB-resident target-step campaign; no production duration is consumed.
# Evidence: the CANN hardware package's microbench references/
# moe_compute_expert_tokens_search_chain_cann_8_5_1.json.
_CANN_MOE_COMPUTE_EXPERT_TOKENS_SEARCH_COMPARISON_AIV_CYCLES = 32
_CANN_MOE_COMPUTE_EXPERT_TOKENS_TARGET_SETUP_AIV_CYCLES = 37
# [MEASURED_MECHANISM] CANN 8.5.1 key-1003 source phases following each
# search segment.  These are workload-independent source-program probes, not
# production-kernel durations.  The retained CANN reference artifacts are
# version locked by their repository manifests.  ProcessAfter is currently
# closed only for the E512, one-input-worker source domain.
_CANN_MOE_COMPUTE_EXPERT_TOKENS_SCATTER_MTE3_NS = 9724.11376953
_CANN_MOE_COMPUTE_EXPERT_TOKENS_FIRST_SYNC_NS = 1407.87597656
_CANN_MOE_COMPUTE_EXPERT_TOKENS_SECOND_SYNC_NS = 1355.6640625
_CANN_MOE_COMPUTE_EXPERT_TOKENS_PROCESS_AFTER_NS = 4360.0
_CANN_MOE_COMPUTE_EXPERT_TOKENS_KEY1001_INIT_SYNC_SERVICE = (
    "moe_compute_expert_tokens_key1001_init_sync_phase")
_CANN_MOE_COMPUTE_EXPERT_TOKENS_KEY1001_INIT_SYNC_EXPERT_COUNTS = (
    8, 32, 128)
_CANN_MOE_FINALIZE_ROUTING_BF16_NO_BIAS_TILING_KEY = 20_036
_BF16_VECTOR_ROUND_FUNCTIONS = ("add_square", "scale_multiply")
_BF16_VECTOR_ROUND_REGIMES = ("first", "resident", "queued")
_RMSNORM_SCALAR_ROUND_FUNCTIONS = ("input_square", "output_scale")
# Per-element arithmetic of the two RMSNorm source programs, counted from
# ``vllm/csrc/libtorch_stable/layernorm_kernels.cu`` together with
# ``csrc/type_convert.cuh`` on the production ``HasWeight`` path.  Both
# variants run the same two source passes over the row, so both are priced by
# one shared per-element tally; the fused variant differs only by the residual
# add its first pass performs before squaring.
#
#   pass 1  ``rms_norm_kernel``          ``variance += x * x``     1 mul, 1 add
#           ``fused_add_rms_norm_kernel``  ``temp += residual_v``          1 add
#                                        ``sum_squares()`` accumulate
#                                        ``z.x*z.x + z.y*z.y``     1 mul, 1 add
#   pass 2  both                         ``x * s_variance * w``        2 mul
#
# The fused variant's ``residual_v[id] = temp`` writeback is a store, not
# arithmetic; it is carried by the row's write byte account and must not be
# repriced here.  The block reduction, epsilon add and reciprocal square root
# are per-row terms and stay outside this per-element tally.
_NORM_SQUARE_ACCUMULATE_OPERATIONS_PER_ELEMENT = 2.0
_NORM_OUTPUT_MULTIPLY_OPERATIONS_PER_ELEMENT = 2.0
_NORM_RESIDUAL_ADD_OPERATIONS_PER_ELEMENT = {
    "rmsnorm": 0.0,
    "add_rmsnorm": 1.0,
}


def _norm_arithmetic_operations_per_element(variant: str) -> float:
    """Return the source-counted per-element arithmetic of one NORM variant."""

    if variant not in _NORM_RESIDUAL_ADD_OPERATIONS_PER_ELEMENT:
        raise ValueError(f"unexpected normalized RMSNorm variant: {variant!r}")
    return math.fsum((
        _NORM_RESIDUAL_ADD_OPERATIONS_PER_ELEMENT[variant],
        _NORM_SQUARE_ACCUMULATE_OPERATIONS_PER_ELEMENT,
        _NORM_OUTPUT_MULTIPLY_OPERATIONS_PER_ELEMENT,
    ))


# The source loop is valid from two iterations onward.  The independent C500
# campaign includes depths two and three and retains the same compiled loop
# body and occupancy as the deeper steady-round measurements.
_RMSNORM_SCALAR_ROUND_MIN_DEPTH = 2
_RMSNORM_SCALAR_ROUND_MAX_DEPTH = 32
_CANN_MOE_AUX_VARIANTS = {
    "moe_swiglu", "moe_init_routing", "moe_init_routing_v2",
    "moe_init_routing_v3", "moe_finalize_routing",
    "moe_compute_expert_tokens", "moe_transpose", "moe_cast",
    "moe_range", "moe_gating_topk", "moe_abs", "moe_token_unpermute",
}


_GPU_MOE_AUX_VARIANTS = {
    "moe_act_and_mul", "moe_silu", "moe_mul", "moe_sum", "moe_align",
    "moe_align_opt_128", "moe_align_small_batch",
    "moe_count_and_sort", "moe_reduce", "moe_fill", "moe_enable_if",
    "moe_routing_weight_norm", "moe_routing_weight_scale",
    "moe_routed_id_extent", "moe_expert_map_lookup", "moe_router_logit_cast",
    "moe_routed_id_widen",
}

_SHARED_MOE_AUX_VARIANTS = {"moe_topk", "moe_softmax", "moe_gating"}

# Supplied C500 architecture reference: one dependent shared-memory access
# completes in 64 XCORE cycles.  This is an architectural latency, not a
# production-kernel timing fit.
_METAX_C500_SHARED_DEPENDENCY_CYCLES = 64.0


@dataclasses.dataclass(frozen=True)
class _VectorPhaseSpec:
    """Pre-normalization work for one non-timing Vector phase."""

    name: str
    predecessors: Tuple[str, ...]
    completion_barrier: bool
    l2_read_bytes: float
    dram_read_bytes: float
    write_bytes: float
    compute_demands: Tuple[PhaseComputeDemand, ...]
    dram_write_bytes: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Vector phase name must be a non-empty string")
        if not isinstance(self.predecessors, tuple):
            raise TypeError("Vector phase predecessors must be a tuple")
        if any(not isinstance(value, str) or not value.strip()
               for value in self.predecessors):
            raise ValueError(
                "Vector phase predecessors must be non-empty strings")
        if not isinstance(self.completion_barrier, bool):
            raise TypeError("Vector phase completion_barrier must be boolean")
        byte_values = {
            "l2_read_bytes": self.l2_read_bytes,
            "dram_read_bytes": self.dram_read_bytes,
            "write_bytes": self.write_bytes,
        }
        invalid = {
            name: value for name, value in byte_values.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0)
        }
        if invalid:
            raise ValueError(
                f"Vector phase bytes must be finite and non-negative: "
                f"{invalid}")
        if self.dram_read_bytes > self.l2_read_bytes:
            raise ValueError(
                "Vector phase DRAM reads cannot exceed L2-path reads")
        if self.dram_write_bytes is not None:
            if (isinstance(self.dram_write_bytes, bool)
                    or not isinstance(self.dram_write_bytes, (int, float))
                    or not math.isfinite(self.dram_write_bytes)
                    or self.dram_write_bytes < 0
                    or self.dram_write_bytes > self.write_bytes):
                raise ValueError(
                    "Vector phase DRAM writes must be finite and within its "
                    "L2 write extent")
        if not isinstance(self.compute_demands, tuple):
            raise TypeError("Vector phase compute_demands must be a tuple")
        if any(not isinstance(value, PhaseComputeDemand)
               for value in self.compute_demands):
            raise TypeError(
                "Vector phase compute_demands must contain "
                "PhaseComputeDemand")


@dataclasses.dataclass(frozen=True)
class _VectorPhaseBlueprint:
    phases: Tuple[_VectorPhaseSpec, ...]
    lowered_sfu_phase: str

    def __post_init__(self) -> None:
        if not isinstance(self.phases, tuple) or not self.phases:
            raise TypeError(
                "Vector phase blueprint requires a non-empty phase tuple")
        if any(not isinstance(value, _VectorPhaseSpec)
               for value in self.phases):
            raise TypeError(
                "Vector phase blueprint must contain _VectorPhaseSpec")
        names = tuple(value.name for value in self.phases)
        if len(names) != len(set(names)):
            raise ValueError("Vector phase blueprint names must be unique")
        if self.lowered_sfu_phase not in names:
            raise ValueError(
                "Vector lowered-SFU phase must name a blueprint phase")


def _phase_compute_demands(
        *entries: Tuple[str, str, float]) -> Tuple[PhaseComputeDemand, ...]:
    demands = []
    for resource_role, operation, operations in entries:
        if (isinstance(operations, bool)
                or not isinstance(operations, (int, float))
                or not math.isfinite(operations)
                or operations < 0):
            raise ValueError(
                "Vector phase operations must be finite and non-negative: "
                f"{resource_role}/{operation}={operations!r}")
        if operations > 0:
            demands.append(PhaseComputeDemand(
                resource_role=resource_role,
                operation=operation,
                operations=float(operations),
            ))
    return tuple(demands)


def _linear_phase_blueprint(
        *, l2_read_bytes: float, dram_read_bytes: float, write_bytes: float,
        compute_demands: Tuple[PhaseComputeDemand, ...],
        dram_write_bytes: Optional[float] = None,
        ) -> _VectorPhaseBlueprint:
    return _VectorPhaseBlueprint(
        phases=(
            _VectorPhaseSpec(
                name="load",
                predecessors=(),
                completion_barrier=False,
                l2_read_bytes=l2_read_bytes,
                dram_read_bytes=dram_read_bytes,
                write_bytes=0.0,
                compute_demands=(),
            ),
            _VectorPhaseSpec(
                name="compute",
                predecessors=("load",),
                completion_barrier=False,
                l2_read_bytes=0.0,
                dram_read_bytes=0.0,
                write_bytes=0.0,
                compute_demands=compute_demands,
            ),
            _VectorPhaseSpec(
                name="store",
                predecessors=("compute",),
                completion_barrier=False,
                l2_read_bytes=0.0,
                dram_read_bytes=0.0,
                write_bytes=write_bytes,
                compute_demands=(),
                dram_write_bytes=dram_write_bytes,
            ),
        ),
        lowered_sfu_phase="compute",
    )


def _has_compute_service(hardware, role: str, service: str) -> bool:
    if role not in hardware._role_map:
        return False
    resource = hardware.resource_by_role(role)
    return (
        resource.type == "compute"
        and resource.has_typed_service(service)
    )


def _moe_routing_global_atomic_service(
        kernel: KernelSpec, hardware) -> Optional[str]:
    """Return the published global-atomic service, if its source domain fits.

    vLLM's non-small-batch MoE helper launches one 256-thread CTA per 256
    routed ids and issues one global ``atomicAdd`` for every accepted id.  The
    independent service measurements cover 8-, 32- and 128-expert counters
    after the launch saturates the device; each card publishes its own
    measured rates.  The discrete counter-array extent, not a production
    routing histogram or duration, selects a rate.
    """

    if (kernel.variant != "moe_count_and_sort"
            or hardware.kernel_memory_model != "gpu_mshr"):
        return None
    tokens = _shape(kernel, "tokens")
    topk = _shape(kernel, "topk")
    experts = _shape(kernel, "experts")
    routed_items = tokens * topk
    if experts not in _MOE_ROUTING_GLOBAL_ATOMIC_SERVICES:
        return None
    expected_grid = min(math.ceil(routed_items / 256), 65535)
    if (expected_grid < hardware.num_compute_units
            or kernel.launch.grid != (expected_grid, 1, 1)
            or kernel.launch.block != (256, 1, 1)):
        return None
    service = _MOE_ROUTING_GLOBAL_ATOMIC_SERVICES[experts]
    if not _has_compute_service(hardware, "compute_alu", service):
        return None
    return service


def _cann_swiglu_service_demands(
        hardware, services: Tuple[Tuple[str, float, int], ...]
        ) -> Tuple[ComputeServiceDemand, ...]:
    """Use each independently published SwiGLU service without all-or-none fallback."""

    operations_by_service = {}
    invocations_by_service = {}
    service_order = []
    for source_service, operations, invocations in services:
        service = (
            source_service
            if _has_compute_service(
                hardware, "compute_vector", source_service)
            else "generic")
        if service not in operations_by_service:
            service_order.append(service)
            operations_by_service[service] = 0.0
            invocations_by_service[service] = 0
        operations_by_service[service] += operations
        if service != "generic":
            invocations_by_service[service] += invocations
    return tuple(
        ComputeServiceDemand(
            service,
            operations_by_service[service],
            serial_invocations=invocations_by_service[service],
        )
        for service in service_order)


def _cann_swiglu_logical_shape(kernel: KernelSpec) -> Tuple[int, int, int]:
    """Return the source program's logical rows, columns, and elements."""

    variant = kernel.variant.lower()
    if variant in ("activation", "silu"):
        rows = _shape(kernel, "tok")
        columns = _shape(kernel, "intermediate")
        elements = _shape(kernel, "element_count")
    elif variant == "moe_swiglu":
        rows = int(kernel.attributes["routed_rows"])
        columns = _shape(kernel, "intermediate")
        elements = rows * columns
    else:
        raise ValueError(
            "CANN SwiGLU source model requires activation/silu or "
            "moe_swiglu")
    if elements != rows * columns:
        raise ValueError(
            "typed CANN SwiGLU element count must equal rows * columns")
    return rows, columns, elements




def _bf16_vector_round_widths(
        active_threads: int, maximum_threads: int,
        additional_widths: Tuple[int, ...] = (),
        ) -> Tuple[Tuple[int, float], ...]:
    """Resolve one active CTA width against measured physical endpoints."""

    if active_threads <= 0 or active_threads > maximum_threads:
        raise ValueError("BF16 Vector round active width is outside its CTA")
    supported = tuple(
        threads for threads in sorted({256, 512, 1024, *additional_widths})
        if threads <= maximum_threads)
    if not supported:
        raise ValueError("BF16 Vector round has no measured width")
    if active_threads <= supported[0]:
        # A typed round is normalized to the source-equivalent work of every
        # active vec8 lane. The smallest measured endpoint is one 256-thread
        # CTA; a tail with fewer active threads contributes its exact wave
        # fraction while the physical CTA residency is modeled separately.
        return ((supported[0], active_threads / supported[0]),)
    for lower, upper in zip(supported, supported[1:]):
        if active_threads == upper:
            return ((upper, 1.0),)
        if active_threads < upper:
            upper_weight = (active_threads - lower) / (upper - lower)
            return (
                (lower, 1.0 - upper_weight),
                (upper, upper_weight),
            )
    raise ValueError("BF16 Vector round exceeds the measured width domain")


def _add_rmsnorm_additional_round_widths(
        hardware, block_threads: int) -> Tuple[int, ...]:
    """Return source-measured active widths for the selected CTA class."""

    block256 = hardware.add_rmsnorm_block256_source_round_pipeline
    if block256 is not None and block_threads == block256["threads_per_cta"]:
        if block256["model"] != (
                "sm80_bf16_add_rmsnorm_block256_resident_v1"):
            raise ValueError("unknown block-256 Add-RMSNorm source-round model")
        return block256["full_tail_pair_active_threads"]
    mechanism = hardware.add_rmsnorm_source_round_pipeline
    if mechanism is None or block_threads != mechanism["threads_per_cta"]:
        return ()
    if mechanism["model"] != (
            "sm80_bf16_add_rmsnorm_block1024_one_wave_v8"):
        raise ValueError("unknown block-1024 Add-RMSNorm source-round model")
    return (
        mechanism["additional_active_threads_per_source_round"],
        mechanism["second_additional_active_threads_per_source_round"],
        mechanism["third_additional_active_threads_per_source_round"],
        mechanism["fourth_additional_active_threads_per_source_round"],
    )


def _add_rmsnorm_active_unit_service_weights(
        hardware, function: str, threads: int,
        active_units: int, block_threads: int) -> Tuple[Tuple[str, float], ...]:
    """Return one CTA-local source-round service for a block1024 program.

    The clock64 grid-one cell isolates the intrinsic source round. Larger
    grids observe that same local program together with launch-wide cache and
    memory contention, which is already conserved by the aggregate memory
    envelope. Treating those validation captures as alternate local service
    rates would encode grid twice and create artificial shape transitions.
    """

    mechanism = hardware.add_rmsnorm_source_round_pipeline
    if (mechanism is None
            or block_threads != mechanism["threads_per_cta"]
            or mechanism["model"] != (
                "sm80_bf16_add_rmsnorm_block1024_one_wave_v8")):
        return ()
    source_widths = {
        mechanism["active_threads_per_source_round"],
        mechanism["additional_active_threads_per_source_round"],
        mechanism["second_additional_active_threads_per_source_round"],
        mechanism["third_additional_active_threads_per_source_round"],
        mechanism["fourth_additional_active_threads_per_source_round"],
    }
    if threads not in source_widths:
        return ()
    if (active_units <= 0
            or active_units > mechanism["maximum_one_wave_grid_blocks"]):
        raise ValueError(
            "Add-RMSNorm active units are outside the one-wave source "
            "program domain")
    return ((f"bf16_{function}_first_{threads}_block1024", 1.0),)


def _has_typed_add_rms_round_services(
        kernel: KernelSpec, hardware) -> bool:
    """Return whether this exact launch has every required vec8 service."""

    if (kernel.variant.lower() != "add_rmsnorm"
            or kernel.dtype != "bf16"
            or _SOURCE_BOUND_BLOCK_REDUCTION_PARAMETER
            not in kernel.launch.parameters
            or "compute_vector" not in hardware._role_map):
        return False
    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    block_threads = kernel.launch.block_size
    mechanisms = tuple(
        mechanism for mechanism in (
            hardware.add_rmsnorm_source_round_pipeline,
            hardware.add_rmsnorm_block256_source_round_pipeline)
        if mechanism is not None)

    def _matches(mechanism) -> bool:
        if (kernel.dtype != mechanism["dtype"]
                or block_threads != mechanism["threads_per_cta"]):
            return False
        if mechanism["model"] == (
                "sm80_bf16_add_rmsnorm_block1024_one_wave_v8"):
            active_threads = hidden // mechanism["vector_width_elements"]
            exact_hidden_widths = (
                (
                    mechanism["hidden_size"],
                    mechanism["active_threads_per_source_round"],
                ),
                (
                    mechanism["additional_hidden_size"],
                    mechanism[
                        "additional_active_threads_per_source_round"],
                ),
                (
                    mechanism["second_additional_hidden_size"],
                    mechanism[
                        "second_additional_active_threads_per_source_round"],
                ),
                (
                    mechanism["third_additional_hidden_size"],
                    mechanism[
                        "third_additional_active_threads_per_source_round"],
                ),
                (
                    mechanism["fourth_additional_hidden_size"],
                    mechanism[
                        "fourth_additional_active_threads_per_source_round"],
                ),
            )
            return (
                hidden % 256 == 0
                and (hidden, active_threads) in exact_hidden_widths
                and rows <= mechanism["maximum_one_wave_grid_blocks"])
        if mechanism["model"] == (
                "sm80_bf16_add_rmsnorm_block256_resident_v1"):
            vector_packs = hidden // mechanism["vector_width_elements"]
            full_rounds, tail_threads = divmod(
                vector_packs, mechanism["threads_per_cta"])
            return (
                hidden % 256 == 0
                and (tail_threads == 0
                     or (full_rounds > 0 and tail_threads in (64, 128)))
                and mechanism["minimum_grid_blocks"] <= rows
                <= mechanism["maximum_grid_blocks"])
        raise ValueError("unknown Add-RMSNorm source-round pipeline model")

    if mechanisms and not any(_matches(mechanism) for mechanism in mechanisms):
        return False
    if (kernel.launch.grid != (rows, 1, 1)
            or kernel.launch.block != (block_threads, 1, 1)
            or hidden % 256 != 0
            or block_threads not in _BLOCK_REDUCTION_RESIDENT_CTAS):
        return False
    vector_packs = hidden // 8
    full_rounds, tail_threads = divmod(vector_packs, block_threads)
    widths = {block_threads} if full_rounds else set()
    if tail_threads:
        widths.update(
            threads for threads, weight in _bf16_vector_round_widths(
                tail_threads, block_threads,
                _add_rmsnorm_additional_round_widths(
                    hardware, block_threads))
            if weight > 0)
    if not widths:
        return False
    active_units = min(rows, hardware.num_compute_units)
    ctas_per_worker = math.ceil(rows / active_units)
    resident_ctas = _thread_limited_resident_ctas(kernel, hardware)
    regimes = ["first"]
    if ctas_per_worker > 1:
        regimes.append("resident")
    if ctas_per_worker > resident_ctas:
        regimes.append("queued")
    resource = hardware.resource_by_role("compute_vector")

    def _service_name(function: str, regime: str, threads: int) -> str:
        base = f"bf16_{function}_{regime}_{threads}"
        return f"{base}_block{block_threads}" if mechanisms else base

    required_services = []
    for function in _BF16_VECTOR_ROUND_FUNCTIONS:
        for regime in regimes:
            for threads in widths:
                if regime == "first":
                    curve = _add_rmsnorm_active_unit_service_weights(
                        hardware, function, threads, active_units,
                        block_threads)
                elif regime in {"resident", "queued"}:
                    curve = ()
                else:
                    raise ValueError(f"unknown scalar-round regime: {regime}")
                if curve:
                    required_services.extend(
                        service for service, weight in curve if weight > 0)
                else:
                    required_services.append(
                        _service_name(function, regime, threads))
    return all(resource.has_typed_service(service)
               for service in required_services)


def _has_c500_add_rmsnorm_scalar_first_services(
        kernel: KernelSpec, hardware) -> bool:
    """Return whether the exact single-CTA scalar-loop services apply."""

    if (kernel.variant.lower() != "add_rmsnorm"
            or kernel.dtype != "bf16"
            or _SOURCE_BOUND_BLOCK_REDUCTION_PARAMETER
            not in kernel.launch.parameters
            or "compute_vector" not in hardware._role_map):
        return False
    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    block_threads = kernel.launch.block_size
    if (rows != 1
            or kernel.launch.grid != (1, 1, 1)
            or kernel.launch.block != (1024, 1, 1)
            or block_threads != 1024
            or hidden % block_threads != 0):
        return False
    source_round_depth = hidden // block_threads
    if not (_C500_ADD_RMSNORM_SCALAR_FIRST_MIN_DEPTH
            <= source_round_depth
            <= _C500_ADD_RMSNORM_SCALAR_FIRST_MAX_DEPTH):
        return False
    resource = hardware.resource_by_role("compute_vector")
    return all(
        resource.has_typed_service(service)
        for service in _C500_ADD_RMSNORM_SCALAR_FIRST_SERVICES.values())


def _c500_add_rmsnorm_grid64_fixed_input_service(
        kernel: KernelSpec, hardware) -> Optional[str]:
    """Return the exact complete fixed-address input-loop service.

    The standalone clock64 campaign covers one generic BF16 block1024 CTA at
    grid 64.  Its fixed-address interval excludes launch, CTA admission,
    reduction, output, and address progression, so it can only compete with
    the existing complete input branch in the exact measured depth domain.
    """

    if (kernel.variant.lower() != "add_rmsnorm"
            or kernel.dtype != "bf16"
            or kernel.launch.grid != (64, 1, 1)
            or kernel.launch.block != (1024, 1, 1)):
        return None
    if "compute_vector" not in hardware._role_map:
        return None
    resource = hardware.resource_by_role("compute_vector")
    if not any(
            resource.has_typed_service(candidate)
            for candidate in
            _C500_ADD_RMSNORM_GRID64_FIXED_INPUT_SERVICES.values()):
        return None
    hidden = _shape(kernel, "hidden")
    if hidden % 1024 != 0:
        return None
    depth = hidden // 1024
    if depth not in _C500_ADD_RMSNORM_GRID64_FIXED_INPUT_SERVICES:
        return None
    service = _C500_ADD_RMSNORM_GRID64_FIXED_INPUT_SERVICES[depth]
    if not resource.has_typed_service(service):
        raise ValueError(
            "C500 grid64 fixed AddRMSNorm input service is unavailable: "
            f"{service!r}")
    return service


def _has_typed_rmsnorm_scalar_round_services(
        kernel: KernelSpec, hardware) -> bool:
    """Return whether the selected legacy scalar source rounds are covered."""

    if (kernel.variant.lower() != "rmsnorm"
            or kernel.dtype != "bf16"
            or _SOURCE_BOUND_BLOCK_REDUCTION_PARAMETER
            not in kernel.launch.parameters
            or SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER
            not in kernel.launch.parameters
            or "compute_vector" not in hardware._role_map):
        return False
    resource = hardware.resource_by_role("compute_vector")
    legacy_program_services = resource.has_typed_service(
        "rmsnorm_input_square_resident_128")
    if (legacy_program_services
            and SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER
            not in kernel.launch.parameters):
        return False
    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    block_threads = kernel.launch.block_size
    source_round_depth = hidden // block_threads
    if (kernel.launch.grid != (rows, 1, 1)
            or kernel.launch.block != (block_threads, 1, 1)
            or block_threads not in (128, 1024)
            or hidden % hardware.parallel_width != 0
            or source_round_depth < (
                1 if block_threads == 128
                else _RMSNORM_SCALAR_ROUND_MIN_DEPTH)
            or source_round_depth > _RMSNORM_SCALAR_ROUND_MAX_DEPTH):
        return False
    active_units = min(rows, hardware.num_compute_units)
    ctas_per_worker = math.ceil(rows / active_units)
    resident_ctas = _thread_limited_resident_ctas(kernel, hardware)
    exact_active64_program = resource.has_typed_service(
        _RMSNORM_PHASE_ENTRY_ACTIVE64_SERVICE)
    if exact_active64_program:
        # The SM80 grid-64 probe reports a joint load/compute source-loop
        # envelope.  It does not expose a separable CTA-local memory
        # completion that can be composed with reduction and store, and its
        # exact active-64 domain cannot define another active-grid regime.
        # Keep it as validation evidence; the production path uses the common
        # source program and finite-source memory model at every grid.
        return False
    regimes = ["first"]
    if ctas_per_worker > 1:
        regimes.append("resident")
    if ctas_per_worker > resident_ctas:
        regimes.append("queued")
    required_services = [
        f"rmsnorm_{function}_{regime}_{block_threads}"
        for function in _RMSNORM_SCALAR_ROUND_FUNCTIONS
        for regime in regimes
    ]
    if (block_threads == 1024
            and active_units in _C500_RMSNORM_PHASE_CHAIN_EXCESS_SERVICES):
        required_services.append(
            _C500_RMSNORM_PHASE_CHAIN_EXCESS_SERVICES[active_units])
    if (block_threads == 1024
            and rows == active_units
            and source_round_depth >= _RMSNORM_PHASE_ENTRY_MIN_DEPTH):
        if active_units in _RMSNORM_PHASE_ENTRY_SERVICES:
            required_services.append(
                _RMSNORM_PHASE_ENTRY_SERVICES[active_units])
        elif (active_units == 64
                and resource.has_typed_service(
                    _RMSNORM_PHASE_ENTRY_ACTIVE64_SERVICE)):
            required_services.append(_RMSNORM_PHASE_ENTRY_ACTIVE64_SERVICE)
    return all(
        resource.has_typed_service(
            service) for service in required_services)


def _c500_legacy_rmsnorm_program_vector_cycles_per_cta(
        kernel: KernelSpec, hardware) -> Optional[float]:
    """Return AP-normalized Vector cycles for one legacy RMSNorm CTA.

    The program identity is supplied by launch selection.  The formulas are
    instruction-work laws for two compiled block programs, not timing fits:
    the block1024 counter series is exactly linear in source-loop rounds and
    the block128 binary has one statically valid H128 round.  Unsupported
    compiled block programs remain unpriced instead of borrowing a neighbor.
    """

    parameter = SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER
    if (kernel.variant.lower() != "rmsnorm"
            or kernel.dtype != "bf16"
            or parameter not in kernel.launch.parameters):
        return None
    if kernel.launch.parameters[parameter] != 1.0:
        raise ValueError(
            "legacy C500 RMSNorm program identity must equal one")
    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    block_threads = kernel.launch.block_size
    if (kernel.launch.grid != (rows, 1, 1)
            or kernel.launch.block != (block_threads, 1, 1)):
        raise ValueError(
            "legacy C500 RMSNorm issue work requires one whole-row CTA")
    if hidden % block_threads:
        return None
    if block_threads not in _C500_LEGACY_RMSNORM_RAW_VECTOR_CYCLES:
        return None
    fixed_cycles, cycles_per_round, minimum_rounds, maximum_rounds = (
        _C500_LEGACY_RMSNORM_RAW_VECTOR_CYCLES[block_threads])
    rounds = hidden // block_threads
    if (rounds < minimum_rounds
            or (maximum_rounds is not None and rounds > maximum_rounds)):
        return None
    fma_resource = hardware.resource_by_role("compute_fma")
    if fma_resource.total_units % hardware.num_compute_units:
        raise ValueError(
            "legacy C500 RMSNorm PEU count must divide the AP count")
    peus_per_ap = fma_resource.total_units // hardware.num_compute_units
    waves_per_cta = math.ceil(block_threads / hardware.parallel_width)
    participating_peus = min(waves_per_cta, peus_per_ap)
    if participating_peus <= 0:
        raise ValueError(
            "legacy C500 RMSNorm requires a positive Vector worker count")
    return (fixed_cycles + cycles_per_round * rounds) / participating_peus


def _silu_scalar_rounds(kernel: KernelSpec) -> float:
    """Return semantic SiLU rounds across all source-launched CTAs."""

    variant = kernel.variant.lower()
    if variant in ("activation", "silu"):
        tokens = _shape(kernel, "tok")
        intermediate = _shape(kernel, "intermediate")
        elements = _shape(kernel, "element_count")
    elif variant == "moe_act_and_mul":
        tokens = _shape(kernel, "tokens") * _shape(kernel, "topk")
        intermediate = _shape(kernel, "intermediate")
        elements = tokens * intermediate
    else:
        raise ValueError(
            "SiLU semantic rounds require activation/silu or "
            "moe_act_and_mul")
    if elements != tokens * intermediate:
        raise ValueError(
            "SiLU element_count must equal tok * intermediate")
    if kernel.launch.grid_size != tokens:
        raise ValueError("SiLU semantic rounds require one CTA per token")
    if kernel.launch.block_size != 1024:
        raise ValueError(
            "silu_scalar_round service is characterized only for 1024-thread "
            "CTAs")
    return float(tokens * math.ceil(intermediate / kernel.launch.block_size))


def _silu_source_cta_round_elements(
        kernel: KernelSpec) -> Tuple[Tuple[int, ...], ...]:
    """Return source-loop element rounds for every launched SiLU CTA."""

    tokens = _shape(kernel, "tok")
    intermediate = _shape(kernel, "intermediate")
    elements = _shape(kernel, "element_count")
    if elements != tokens * intermediate:
        raise ValueError(
            "SiLU element_count must equal tok * intermediate")
    parameters = kernel.launch.parameters
    if SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER not in parameters:
        raise ValueError(
            "SiLU exp dependency requires a source vector-width fact")
    source_elements_per_thread = int(
        parameters[SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER])
    if (source_elements_per_thread <= 0
            or parameters[SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER]
            != source_elements_per_thread):
        raise ValueError("SiLU source vector width must be positive")
    if intermediate % source_elements_per_thread:
        raise ValueError(
            "SiLU exp dependency launch disagrees with source vector width")
    source_vectors = intermediate // source_elements_per_thread
    one_dimensional_block_size = min(source_vectors, 1024)
    if (kernel.launch.grid == (tokens, 1, 1)
            and kernel.launch.block
            == (one_dimensional_block_size, 1, 1)):
        elements_per_round = (
            one_dimensional_block_size * source_elements_per_thread)
        token_rounds = tuple(
            min(elements_per_round, intermediate - offset)
            for offset in range(0, intermediate, elements_per_round))
        cta_rounds = tuple(token_rounds for _ in range(tokens))
    else:
        elements_per_cta = (
            kernel.launch.block_size * source_elements_per_thread)
        cta_columns = math.ceil(intermediate / elements_per_cta)
        if (kernel.launch.block[1:] != (1, 1)
                or kernel.launch.grid != (cta_columns, tokens, 1)):
            raise ValueError(
                "SiLU exp dependency launch disagrees with source vector "
                "width")
        row_tiles = tuple(
            min(elements_per_cta, intermediate - offset)
            for offset in range(0, intermediate, elements_per_cta))
        cta_rounds = tuple(
            (tile_elements,)
            for _token in range(tokens)
            for tile_elements in row_tiles)
    if (not cta_rounds
            or any(not rounds for rounds in cta_rounds)
            or sum(sum(rounds) for rounds in cta_rounds) != elements
            or len(cta_rounds) != kernel.launch.grid_size):
        raise ValueError("SiLU source CTA rounds do not conserve elements")
    return cta_rounds


def _silu_exp_dependency_rounds(kernel: KernelSpec) -> float:
    """Return source-level math layers across launched SiLU CTAs."""

    source_elements_per_thread = int(kernel.launch.parameters[
        SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER])
    # The typed service unit is one dependency layer across the independently
    # characterized 512-thread block. Source-loop rounds partition the active
    # vectors, and independently launched packed tiles conserve the same work.
    return math.fsum(
        source_elements_per_thread
        * math.ceil(tile_elements / source_elements_per_thread)
        / _SILU_MATH_SERVICE_THREADS
        for cta_rounds in _silu_source_cta_round_elements(kernel)
        for tile_elements in cta_rounds)


def _source_wave_dependency_layers(
        active_threads: int, hardware) -> int:
    """Return active wave64 layers executing one source dependency."""

    if active_threads <= 0:
        raise ValueError("source dependency requires active threads")
    if hardware.parallel_width <= 0:
        raise ValueError("hardware parallel width must be positive")
    # The typed microbenchmark's calibrated operation is one dependency chain
    # executed by every thread in one physical wave64.  Aggregate resource
    # capacity already supplies parallelism across APs; dividing the demand by
    # the four PEUs in an AP here would apply that parallelism a second time.
    return math.ceil(active_threads / hardware.parallel_width)


def _source_bf16_conversion_operations(kernel: KernelSpec) -> float:
    """Return source-published BF16 conversion work for a typed kernel."""

    if _has_graph_padding(kernel):
        return 0.0
    parameter = SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER
    if parameter not in kernel.launch.parameters:
        return 0.0
    if kernel.dtype != "bf16":
        raise ValueError("source BF16 conversion work requires BF16")
    return float(kernel.launch.parameters[parameter])


def _attach_linear_vector_dependency_path(
        inp: PredictionInput, hardware, *, name: str,
        compute_roles: Tuple[str, ...],
        additional_dependent_reads: int = 0,
        dependent_task_stages: int = 0,
        include_memory_services: bool = True,
        stream_data_services: bool = False,
        additional_concurrent_service_groups: Tuple[
            Tuple[str, ...], ...] = (),
        support_level: Optional[str] = None,
        additional_limitations: Tuple[str, ...] = ()) -> PredictionInput:
    """Attach one architecture-neutral Submit -> Load -> Compute -> Store path."""

    if not isinstance(include_memory_services, bool):
        raise TypeError("include_memory_services must be boolean")
    if not isinstance(stream_data_services, bool):
        raise TypeError("stream_data_services must be boolean")
    if not isinstance(additional_concurrent_service_groups, tuple):
        raise TypeError(
            "additional concurrent service groups must be a tuple")
    if support_level is not None and support_level not in (
            "complete", "partial", "aggregate_only", "compatibility"):
        raise ValueError("unsupported analytical dependency support level")
    if (not isinstance(additional_limitations, tuple)
            or any(not isinstance(value, str) or not value
                   for value in additional_limitations)):
        raise TypeError(
            "analytical dependency additional limitations must be strings")
    if inp.pipeline_plan is not None:
        return inp
    if inp.mem_model == "gpu_mshr":
        read_path_name = "global_read"
        write_path_name = "global_write"
    elif inp.mem_model == "npu_tile":
        read_path_name = "mte2_read"
        write_path_name = "mte3_write"
    else:
        raise ValueError(
            "analytical Vector dependency paths require gpu_mshr or "
            "npu_tile memory")
    path_names = {path.name for path in inp.memory_paths}
    expected_path_names = {read_path_name, write_path_name}
    unexpected_path_names = path_names - expected_path_names
    if unexpected_path_names:
        raise ValueError(
            "analytical Vector dependency path has unexpected memory paths: "
            f"{sorted(unexpected_path_names)}")
    services = []
    if hardware.launch_latency_model is not None:
        services.append("launch:submit")
    if include_memory_services and read_path_name in path_names:
        services.append(f"memory:{read_path_name}")
    if additional_dependent_reads:
            if (isinstance(additional_dependent_reads, bool)
                    or not isinstance(additional_dependent_reads, int)
                    or additional_dependent_reads <= 0):
                raise ValueError(
                    "additional dependent reads must be a positive integer")
            if inp.mem_model == "gpu_mshr":
                services.append(f"memory_latency:{read_path_name}")
            elif inp.mem_model == "npu_tile":
                # Current NPU HWSpecs deliberately leave first-request MTE2
                # latency unknown, so the dependency remains explicit in the
                # phase ledger but cannot yet be priced here.
                pass
            else:
                raise AssertionError(
                    f"unhandled Vector memory model: {inp.mem_model!r}")
    if dependent_task_stages:
        if (isinstance(dependent_task_stages, bool)
                or not isinstance(dependent_task_stages, int)
                or dependent_task_stages <= 0):
            raise ValueError(
                "dependent task stages must be a positive integer")
        services.append(MEMORY_DEPENDENCY_TASK_INCREMENT_SERVICE)
    if (include_memory_services
            and inp.demand_per_resource["l1_cache"] > 0):
        services.append("local_memory:l1_cache")
    for role in compute_roles:
        if inp.demand_per_resource[role] > 0:
            services.append(f"compute:{role}")
    if include_memory_services and write_path_name in path_names:
        services.append(f"memory:{write_path_name}")
    if not services:
        raise ValueError(
            "analytical Vector dependency path has no positive service")
    if stream_data_services:
        streaming_group = tuple(
            service for service in services
            if service.startswith(("memory:", "local_memory:", "compute:")))
        if len(streaming_group) < 2:
            raise ValueError(
                "streaming Vector path requires at least two data services")
        concurrent_groups = (streaming_group,)
    elif (include_memory_services
          and read_path_name in path_names
          and inp.demand_per_resource["l1_cache"] > 0):
        concurrent_groups = ((
            f"memory:{read_path_name}", "local_memory:l1_cache"),)
    else:
        concurrent_groups = ()
    concurrent_groups += additional_concurrent_service_groups
    serial_service_counts = []
    if additional_dependent_reads and inp.mem_model == "gpu_mshr":
        serial_service_counts.append((
            f"memory_latency:{read_path_name}",
            additional_dependent_reads,
        ))
    if dependent_task_stages:
        serial_service_counts.append((
            MEMORY_DEPENDENCY_TASK_INCREMENT_SERVICE,
            dependent_task_stages,
        ))
    if inp.dependency_paths:
        raise ValueError(
            "analytical dependency path must be attached exactly once")
    return dataclasses.replace(
        inp,
        dependency_paths=(AnalyticalDependencyPath(
            name=name,
            services=tuple(services),
            concurrent_service_groups=concurrent_groups,
            serial_service_counts=tuple(serial_service_counts),
        ),),
        support_level=(inp.support_level
                       if support_level is None else support_level),
        limitations=inp.limitations + additional_limitations,
    )


def _thread_limited_resident_ctas(
        kernel: KernelSpec, hardware) -> int:
    """Return the architectural block/thread/warp residency bound."""

    warps_per_block = math.ceil(
        kernel.launch.block_size / hardware.parallel_width)
    resident_ctas = min(
        hardware.max_blocks_per_unit,
        hardware.max_threads_per_unit // kernel.launch.block_size,
        hardware.max_warps_per_unit // warps_per_block,
    )
    if resident_ctas <= 0:
        raise ValueError("launch exceeds the architectural CTA capacity")
    return resident_ctas


def _source_bound_resident_ctas(
        kernel: KernelSpec, hardware) -> int:
    """Return an exact source-bound residency or the architectural bound."""

    architectural_bound = _thread_limited_resident_ctas(kernel, hardware)
    parameter = _SOURCE_BOUND_RESIDENT_CTAS_PER_UNIT_PARAMETER
    if parameter not in kernel.launch.parameters:
        return architectural_bound
    resident_ctas = int(kernel.launch.parameters[parameter])
    if resident_ctas > architectural_bound:
        raise ValueError(
            "source-bound CTA residency exceeds the architectural bound")
    return resident_ctas


def _source_bound_cta_batches(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> int:
    """Return ideal concurrent CTA batches from static launch resources."""

    resident_ctas = _source_bound_resident_ctas(kernel, hardware)
    concurrent_ctas = inp.active_units * resident_ctas
    if concurrent_ctas <= 0:
        raise ValueError("source-bound CTA concurrency must be positive")
    return math.ceil(kernel.launch.grid_size / concurrent_ctas)


def _memory_dependency_timer_domain_matches_launch(hardware) -> bool:
    """Return whether task and launch durations share one timer domain."""

    task_model = hardware.memory_dependency_task_latency_model
    launch_model = hardware.launch_latency_model
    if task_model is None or launch_model is None:
        return False
    return task_model["timer_domain"] == launch_model["timer_domain"]


def _source_bound_memory_dependency_stages(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> int:
    """Map one source-ordered task stage to its measured mechanism scope."""

    model = hardware.memory_dependency_task_latency_model
    if model is None:
        # The source keeps its dependent task stage, but this hardware
        # publishes no task mechanism to price it (the H200 probe fails its
        # publication gates).  _regular_model records the unpriced stage as a
        # limitation instead of borrowing another card's curve.
        return 0
    if not _memory_dependency_timer_domain_matches_launch(hardware):
        raise ValueError(
            "source-bound memory dependency requires task and launch "
            "measurements from the same timer domain")
    if model["model"] == "isolated_warm_default_cache_memory_dependency_task_v1":
        return _source_bound_cta_batches(inp, kernel, hardware)
    elif model["model"] in (
            "paired_batched_warm_default_cache_memory_dependency_task_v1",
            "paired_batched_multiround_warm_default_cache_"
            "memory_dependency_task_v1",
            "isolated_mcevent_multiround_warm_default_cache_"
            "memory_dependency_task_v1",
            "paired_mcevent_active_unit_"
            "memory_dependency_increment_v1"):
        # The C500 mechanisms measure one complete task body in the same
        # device-event domain as the empty launch. Independent CTAs have no
        # source-order edge between them, so resource and scheduler models
        # carry grid throughput while the critical path contains the isolated
        # non-empty increment once.
        return 1
    else:
        raise ValueError(
            "unsupported source-bound memory-dependency task model: "
            f"{model['model']!r}")


def _generic_memory_dependency_stage_batches(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> int:
    """Return one whole-kernel stage for the generic paired task mechanism.

    The microbenchmark characterizes the incremental latency of one complete
    load -> dependent operation -> store stage for a launch.  A production
    kernel may contain many resident CTA batches, but they are executions of
    that same stage, not additional source-ordered stages.  Multiplying this
    count by launch-wave depth would therefore charge the complete task body
    repeatedly; launch concurrency is already handled by the resource models.
    """

    model = hardware.memory_dependency_task_latency_model
    if (model is None
            or not _memory_dependency_timer_domain_matches_launch(hardware)
            or model["model"] not in (
                "paired_batched_warm_default_cache_memory_dependency_task_v1",
                "paired_batched_multiround_warm_default_cache_"
                "memory_dependency_task_v1",
                "isolated_mcevent_multiround_warm_default_cache_"
                "memory_dependency_task_v1",
                "paired_mcevent_active_unit_"
                "memory_dependency_increment_v1")):
        return 0
    if not _has_semantic_memory_dependency_task(kernel):
        return 0
    if not _multiround_memory_dependency_source_supported(kernel, hardware):
        return 0
    if hardware.task_duration_floor_contract_supports_launch(
            WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
            kernel.launch.grid_size, kernel.launch.block_size,
            kernel.launch.grid, kernel.launch.block, inp.sm_freq_hz):
        return 1
    # The benchmark contract is whole-launch scoped.  A supported prefix of
    # a larger launch is not itself a measured whole launch, so it cannot be
    # reused as a task-duration lower bound outside the published grid range.
    return 0


def _attach_silu_semantic_dependency_path(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the whole-kernel Load -> SiLU -> gate -> Store resource DAG."""

    if not _has_compute_service(
            hardware, "compute_vector", _SILU_SCALAR_ROUND_SERVICE):
        return inp
    if {path.name for path in inp.memory_paths} != {
            "global_read", "global_write"}:
        raise ValueError(
            "SiLU semantic dependency path requires split GPU memory paths")
    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "SiLU semantic dependency path requires one compute_vector demand")
    vector_demand, = vector_demands
    expected_rounds = _silu_scalar_rounds(kernel)
    if vector_demand.operations != expected_rounds:
        raise ValueError(
            "SiLU scalar-round demand disagrees with semantic shape")
    typed_vector_demand = dataclasses.replace(
        vector_demand,
        service_demands=(ComputeServiceDemand(
            _SILU_SCALAR_ROUND_SERVICE, expected_rounds),),
    )
    compute_demands = tuple(
        typed_vector_demand if demand is vector_demand else demand
        for demand in inp.compute_demands)
    typed = dataclasses.replace(
        inp,
        compute_demands=compute_demands,
    )
    has_issue_lower_bound = (
        "warp_instructions_lower_bound" in kernel.launch.parameters)
    compute_roles = (
        ("compute_vector", "compute_fma", "warp_issue")
        if has_issue_lower_bound
        else ("compute_vector", "compute_fma"))
    concurrent_groups = (
        (("compute:compute_vector", "compute:compute_fma",
          "compute:warp_issue"),)
        if has_issue_lower_bound
        else ())
    dependent_task_stages = (
        _source_bound_memory_dependency_stages(inp, kernel, hardware)
        if SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
        in kernel.launch.parameters
        else 0)
    return _attach_linear_vector_dependency_path(
        typed, hardware, name="silu",
        compute_roles=compute_roles,
        # Warp issue is a competing lower bound over the same source body as
        # the typed SiLU and gate arithmetic, not an additional arithmetic
        # phase.  One CTA per active SM makes that maximum source-ordered;
        # later resident CTAs require a separate overlap schedule.
        additional_concurrent_service_groups=concurrent_groups,
        dependent_task_stages=dependent_task_stages)


def _attach_a800_silu_nonrecurrent_cta_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the two-resident-slot source SiLU pipeline to every CTA wave."""

    marker = SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER
    if kernel.launch.parameters[marker] != 1.0:
        raise ValueError("source SiLU non-recurrent pipeline marker must equal one")
    tokens = _shape(kernel, "tok")
    intermediate = _shape(kernel, "intermediate")
    elements = _shape(kernel, "element_count")
    if (kernel.variant.lower() not in ("activation", "silu")
            or kernel.dtype != "bf16"
            or hardware.kernel_memory_model != "gpu_mshr"
            # The CTA program and service curves are A800-derived, while the
            # H200 proxy keeps its physical 132-SM wave census.
            or hardware.num_compute_units not in (108, 132)
            or tokens < hardware.num_compute_units
            or elements != tokens * intermediate
            or kernel.launch.grid != (tokens, 1, 1)
            or kernel.launch.block != (1024, 1, 1)
            or inp.active_units != hardware.num_compute_units
            or inp.pipeline_plan is not None):
        raise ValueError(
            "source SiLU non-recurrent pipeline left its supported NVIDIA "
            "A800/H200 proxy domain")
    if {path.name for path in inp.memory_paths} != {
            "global_read", "global_write"}:
        raise ValueError(
            "source SiLU non-recurrent pipeline requires split GPU memory paths")
    if inp.demand_per_resource["compute_alu"] != 0.0:
        raise ValueError(
            "source SiLU non-recurrent service already includes conversion work")
    if "warp_instructions_lower_bound" not in kernel.launch.parameters:
        raise ValueError(
            "source SiLU non-recurrent pipeline requires its conserved "
            "launch-wide warp-issue lower bound")

    resident_ctas = _thread_limited_resident_ctas(kernel, hardware)
    if resident_ctas != 2:
        raise ValueError(
            "source SiLU non-recurrent pipeline requires two resident CTAs")
    rounds = math.ceil(intermediate / kernel.launch.block_size)
    ctas_by_worker = tuple(
        len(range(worker, tokens, inp.active_units))
        for worker in range(inp.active_units))
    if min(ctas_by_worker) < 1:
        raise ValueError(
            "source SiLU non-recurrent pipeline requires an active CTA on "
            "every worker")

    resource = hardware.resource_by_role("compute_vector")
    required_services = {
        service
        for regime_services in _A800_SILU_NONRECURRENT_SERVICES.values()
        for service in regime_services.values()
    }
    unavailable = {
        service for service in required_services
        if not resource.has_typed_service(service)}
    if unavailable:
        raise ValueError(
            "source SiLU non-recurrent pipeline services are unavailable: "
            f"{sorted(unavailable)}")

    service_demands = []
    for regime_index, regime in enumerate(("first", "resident", "queued")):
        regime_ctas_by_worker = tuple(
            (1 if cta_count > regime_index else 0)
            if regime != "queued"
            else max(cta_count - resident_ctas, 0)
            for cta_count in ctas_by_worker)
        if not any(regime_ctas_by_worker):
            continue
        invocations_by_worker = (0,) * inp.active_units
        for component, operations_per_cta in (
                ("entry", 1.0), ("round", float(rounds))):
            operations_by_worker = tuple(
                operations_per_cta * regime_ctas
                for regime_ctas in regime_ctas_by_worker)
            total_operations = math.fsum(operations_by_worker)
            if total_operations == 0.0:
                continue
            service_demands.append(ComputeServiceDemand(
                _A800_SILU_NONRECURRENT_SERVICES[regime][component],
                total_operations,
                operations_by_worker=operations_by_worker,
                serial_invocations_by_worker=invocations_by_worker,
            ))
    typed = _replace_vector_service_demands(
        inp, tuple(service_demands))

    round_fraction = 1.0 / (tokens * rounds)

    def _round_sequence(
            worker: int, regime: str, *,
            include_entry: bool) -> PipelineTaskSequenceDemand:
        """Return one source-loop round of one CTA on this worker.

        The vLLM ``act_and_mul`` body is
        ``for (idx = threadIdx.x; idx < d; idx += blockDim.x)`` and consumes
        both operands of iteration ``idx`` inside that same iteration.  The
        source performs no multi-round prefetch and keeps no operand buffer,
        so every round exposes its own global access latency ahead of the
        register-resident round it feeds.  Each round's operand bytes are
        touched exactly once by this kernel, so that exposed access is a
        compulsory miss on the round's critical path rather than a reuse hit.
        """

        services = _A800_SILU_NONRECURRENT_SERVICES[regime]
        round_services = ((ComputeServiceDemand(services["entry"], 1.0),)
                          if include_entry else ())
        round_services += (
            ComputeServiceDemand(services["round"], 1.0),)
        name = f"{regime}_entry_round" if include_entry else f"{regime}_round"
        task = PipelineTaskDemand(
            name=name,
            worker=worker,
            stages=(
                PipelineStageDemand(
                    # Every thread of the CTA issues this round's two operand
                    # loads before any of them is consumed, so the round's
                    # transactions leave as one burst and drain at the path's
                    # service capacity.
                    "load", "memory:global_read", round_fraction,
                    include_first_request_latency=True,
                    burst_drains_at_path_capacity=True),
                PipelineStageDemand(
                    "source_silu", "compute:compute_vector", None,
                    compute_service_demands=round_services,
                    predecessors=("load",)),
                PipelineStageDemand(
                    "source_silu_issue", "compute:warp_issue", round_fraction,
                    predecessors=("load",)),
                PipelineStageDemand(
                    "gate_multiply", "compute:compute_fma", round_fraction,
                    predecessors=("source_silu", "source_silu_issue")),
                PipelineStageDemand(
                    # The round's BF16 store is issued into the write path and
                    # drains asynchronously: the source loop advances to the
                    # next iteration without waiting for it, and the store
                    # instruction's own issue slot is already inside the
                    # round's warp-issue stage.  The write path therefore
                    # stays an independent concurrent envelope instead of
                    # gating the round's dependency chain.
                    "store", "memory:global_write", round_fraction,
                    predecessors=()),
            ),
        )
        return PipelineTaskSequenceDemand(
            name=name, worker=worker, tasks=(task,))

    # Rounds retire one at a time on a worker.  Both resident CTA slots enter
    # their round operands together and stall together, and the independently
    # measured resident round service already carries the issue share of a
    # two-CTA worker, so residency overlaps CTA admission rather than the
    # source round chain itself.
    regime_order = ("first", "resident", "queued")
    programs = []
    for cta_count in tuple(dict.fromkeys(ctas_by_worker)):
        physical_workers = tuple(
            worker for worker, count in enumerate(ctas_by_worker)
            if count == cta_count)
        representative = physical_workers[0]
        regimes = ("first",) + (("resident",) if cta_count >= 2 else ())
        if cta_count > resident_ctas:
            regimes += ("queued",)
        patterns = tuple(
            _round_sequence(representative, regime, include_entry=entry)
            for regime in regimes
            for entry in (True, False))
        pattern_index = {
            (regime, entry): index
            for index, (regime, entry) in enumerate(
                (regime, entry)
                for regime in regimes
                for entry in (True, False))}
        cta_regimes = []
        for cta_index in range(cta_count):
            if cta_index < min(resident_ctas, len(regime_order) - 1):
                cta_regimes.append(regime_order[cta_index])
            else:
                cta_regimes.append("queued")
        order = tuple(
            pattern_index[(regime, round_index == 0)]
            for regime in cta_regimes
            for round_index in range(rounds))
        programs.append(PipelineTaskProgramDemand(
            name=f"silu_{cta_count}_cta_worker_class",
            worker=representative,
            patterns=patterns,
            order=order,
            worker_multiplicity=len(physical_workers),
            physical_workers=physical_workers,
        ))
    plan = PipelineWorkPlan(
        worker_count=inp.active_units,
        tasks=tuple(programs),
    )
    structured = attach_pipeline_work_plan(typed, hardware, plan)
    return dataclasses.replace(
        structured,
        resident_cta_cohort_refills=math.ceil(
            max(ctas_by_worker) / resident_ctas),
    )


def _replace_compute_service_demands(
        inp: PredictionInput, resource_role: str,
        service_demands: Tuple[ComputeServiceDemand, ...]) -> PredictionInput:
    """Bind conserved analytical work to one typed compute resource."""

    matching_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == resource_role)
    if len(matching_demands) != 1:
        raise ValueError(
            f"typed services require one {resource_role} demand")
    aggregate_demand, = matching_demands
    expected_operations = math.fsum(
        demand.operations for demand in service_demands)
    exact_request_operations = math.fsum(
        demand.operations for demand in aggregate_demand.exact_request_demands)
    generic_operations = (
        aggregate_demand.operations - exact_request_operations)
    if generic_operations != expected_operations:
        raise ValueError(
            f"typed {resource_role} service work disagrees with aggregate "
            "demand after exact requests")
    typed_demand = dataclasses.replace(
        aggregate_demand, service_demands=service_demands)
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            typed_demand if demand is aggregate_demand else demand
            for demand in inp.compute_demands),
    )


def _replace_vector_service_demands(
        inp: PredictionInput,
        service_demands: Tuple[ComputeServiceDemand, ...]) -> PredictionInput:
    """Bind conserved analytical work to generic typed Vector services."""

    return _replace_compute_service_demands(
        inp, "compute_vector", service_demands)




def _attach_silu_source_cta_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        use_exp_dependency: bool,
        use_reciprocal_dependency: bool) -> PredictionInput:
    """Attach the source-ordered C500 packed-BF16 SiLU CTA program.

    Each source-loop round loads both 128-bit operands, evaluates eight
    independent element chains per active thread, and stores one 128-bit
    result. A backend may express those rounds either inside one CTA per token
    or as independently launched packed CTAs. Rounds within a CTA preserve
    Load -> conversion -> SFU -> multiply -> Store order. This remains an
    operator-stage model rather than an instruction simulation.
    """

    if SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER not in (
            kernel.launch.parameters):
        return inp
    if not use_exp_dependency and not use_reciprocal_dependency:
        raise ValueError("SiLU CTA pipeline requires a typed math service")
    if (inp.mem_model != "gpu_mshr"
            or inp.pipeline_plan is not None
            or inp.phase_ledger is None):
        raise ValueError(
            "source-bound SiLU CTA pipeline requires one structured GPU "
            "input")
    PipelineStageDemand = bind_pipeline_stage_template
    tokens = _shape(kernel, "tok")
    intermediate = _shape(kernel, "intermediate")
    elements = _shape(kernel, "element_count")
    elements_per_thread = int(kernel.launch.parameters[
        SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER])
    if elements != tokens * intermediate:
        raise ValueError("SiLU CTA pipeline element count is inconsistent")
    cta_round_elements = _silu_source_cta_round_elements(kernel)
    cta_count = len(cta_round_elements)
    expected_rounds = _silu_exp_dependency_rounds(kernel)

    worker_count = min(cta_count, inp.active_units)
    ctas_by_worker = tuple(
        tuple(range(worker, cta_count, worker_count))
        for worker in range(worker_count))
    resident_ctas = _thread_limited_resident_ctas(kernel, hardware)
    rounds_by_worker = tuple(
        math.fsum(
            elements_per_thread
            * math.ceil(tile_elements / elements_per_thread)
            / _SILU_MATH_SERVICE_THREADS
            for cta in worker_ctas
            for tile_elements in cta_round_elements[cta])
        for worker_ctas in ctas_by_worker)
    if not math.isclose(
            math.fsum(rounds_by_worker), expected_rounds,
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError("SiLU CTA rounds do not conserve typed math work")
    invocations_by_worker = tuple(
        sum(len(cta_round_elements[cta]) for cta in worker_ctas)
        for worker_ctas in ctas_by_worker)
    critical_invocations = max(invocations_by_worker)
    source_math_services = []
    if use_exp_dependency:
        source_math_services.append(ComputeServiceDemand(
            _EXP_DEPENDENCY_SERVICE, expected_rounds,
            serial_invocations=critical_invocations,
        ))
    if use_reciprocal_dependency:
        source_math_services.append(ComputeServiceDemand(
            _RECIPROCAL_ONE_PLUS_DEPENDENCY_SERVICE, expected_rounds,
            serial_invocations=critical_invocations,
        ))

    inp = _replace_compute_service_demands(
        inp, "compute_sfu", tuple(source_math_services))
    sfu_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_sfu")
    if len(sfu_demands) != 1:
        raise ValueError("SiLU CTA pipeline requires one typed SFU demand")
    sfu_demand, = sfu_demands
    typed_math_service_names = tuple(
        service for service in (
            _EXP_DEPENDENCY_SERVICE,
            _RECIPROCAL_ONE_PLUS_DEPENDENCY_SERVICE)
        if any(demand.service == service
               for demand in sfu_demand.service_demands))
    round_stages_by_contract = {}

    def _round_stages(
            tile_elements: int, *, suffix: str,
            first_stage: bool, last_stage: bool) -> Tuple[
                PipelineStageDemand, ...]:
        contract = (tile_elements, suffix, first_stage, last_stage)
        if contract in round_stages_by_contract:
            return round_stages_by_contract[contract]
        # Read and write are distinct dependency stages even though both
        # consume the shared DRAM resource.  Binding them to one pipeline
        # resource would make the next CTA's read wait for the preceding
        # CTA's write and serialize the complete CTA body.  Aggregate DRAM
        # demand remains the shared capacity lower bound, while these two
        # stage identities express only source order and CTA overlap.
        fraction = tile_elements / elements
        load_name = "load" if first_stage else f"load{suffix}"
        store_name = "store" if last_stage else f"store{suffix}"
        stages = [PipelineStageDemand(
            load_name, "memory:global_read", fraction,
            include_first_request_latency=True)]
        if inp.demand_per_resource["compute_alu"] > 0:
            stages.append(PipelineStageDemand(
                f"conversion{suffix}", "compute:compute_alu", fraction))
        active_threads = math.ceil(tile_elements / elements_per_thread)
        typed_operations = tuple(
            float(elements_per_thread * active_threads
                  / _SILU_MATH_SERVICE_THREADS)
            for _ in typed_math_service_names)
        stages.append(bind_typed_compute_pipeline_stage_template(
            f"source_sfu_math{suffix}", "compute:compute_sfu",
            typed_math_service_names, typed_operations))
        if inp.demand_per_resource["compute_fma"] > 0:
            stages.append(PipelineStageDemand(
                f"arithmetic{suffix}", "compute:compute_fma", fraction))
        stages.append(PipelineStageDemand(
            store_name, "memory:global_write", fraction))
        resolved_stages = tuple(stages)
        round_stages_by_contract[contract] = resolved_stages
        return resolved_stages

    def _serial_cta_task(
            worker: int, round_elements: Tuple[int, ...],
            name: str) -> PipelineTaskDemand:
        stages = tuple(
            stage
            for round_index, tile_elements in enumerate(round_elements)
            for stage in _round_stages(
                tile_elements, suffix=f"_{round_index}",
                first_stage=round_index == 0,
                last_stage=round_index == len(round_elements) - 1)
        )
        return PipelineTaskDemand(
            name=name, worker=worker, stages=stages)

    def _resident_cta_batch(
            worker: int,
            cta_specs: Tuple[Tuple[int, ...], ...], name: str
            ) -> PipelineTaskSequenceDemand:
        if len(cta_specs) == 1:
            round_elements = cta_specs[0]
            tasks = (_serial_cta_task(
                worker, round_elements, f"{name}_cta0"),)
        else:
            tasks = tuple(
                PipelineTaskDemand(
                    name=f"{name}_cta{cta_slot}_round{round_index}",
                    worker=worker,
                    stages=_round_stages(
                        round_elements[round_index], suffix="",
                        first_stage=True, last_stage=True),
                )
                for round_index in range(max(
                    len(rounds) for rounds in cta_specs))
                for cta_slot, round_elements in enumerate(cta_specs)
                if round_index < len(round_elements)
            )
        return PipelineTaskSequenceDemand(
            name=name, worker=worker, tasks=tasks)

    programs = []
    cta_slots = resident_ctas
    for worker, worker_ctas in enumerate(ctas_by_worker):
        batch_order = tuple(
            tuple(
                cta_round_elements[worker_ctas[local_index]]
                for local_index in range(
                    batch_start,
                    min(batch_start + cta_slots, len(worker_ctas))))
            for batch_start in range(0, len(worker_ctas), cta_slots)
        )
        batch_kinds = tuple(dict.fromkeys(batch_order))
        patterns = tuple(
            _resident_cta_batch(
                worker, cta_specs, f"cta_batch_pattern{pattern_index}")
            for pattern_index, cta_specs in enumerate(batch_kinds))
        pattern_index = {
            batch_kind: index
            for index, batch_kind in enumerate(batch_kinds)}
        programs.append(PipelineTaskProgramDemand(
            name=f"worker{worker}_silu_ctas",
            worker=worker,
            patterns=patterns,
            order=tuple(pattern_index[value] for value in batch_order),
        ))
    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tuple(programs),
        buffers=(PipelineBuffer(
            name="silu_cta_slots",
            producer_stage="load",
            consumer_stage="store",
            slots=cta_slots,
            scope="worker",
        ),),
    )
    structured = attach_pipeline_work_plan(inp, hardware, plan)
    cohort_refills = max(
        math.ceil(max(len(worker_ctas) for worker_ctas in ctas_by_worker)
                  / resident_ctas) - 1,
        0,
    )
    # A first cohort that is full on every active AP crosses the same complete
    # descriptor/wave admission boundary as each refill.  Partial initial
    # cohorts remain outside this full-cohort mechanism's exact domain.
    if min(len(worker_ctas) for worker_ctas in ctas_by_worker) >= resident_ctas:
        cohort_refills += 1
    if cohort_refills == 0:
        return structured
    return dataclasses.replace(
        structured,
        resident_cta_cohort_refills=cohort_refills,
    )


def _target_phase_blueprint(
        hardware, blueprint: _VectorPhaseBlueprint
        ) -> Optional[_VectorPhaseBlueprint]:
    memory_model = hardware.kernel_memory_model
    if memory_model in ("gpu_mshr", "npu_tile"):
        return blueprint
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        return None
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")


def _append_phase_compute(
        blueprint: _VectorPhaseBlueprint, phase_name: str,
        demand: PhaseComputeDemand) -> _VectorPhaseBlueprint:
    if not isinstance(demand, PhaseComputeDemand):
        raise TypeError("Vector phase append requires PhaseComputeDemand")
    matches = sum(phase.name == phase_name for phase in blueprint.phases)
    if matches != 1:
        raise ValueError(
            f"Vector phase append requires exactly one {phase_name!r} phase")
    phases = tuple(
        dataclasses.replace(
            phase,
            compute_demands=phase.compute_demands + (demand,))
        if phase.name == phase_name else phase
        for phase in blueprint.phases)
    return dataclasses.replace(blueprint, phases=phases)


def _fold_phase_compute_to_aiv(
        blueprint: _VectorPhaseBlueprint) -> _VectorPhaseBlueprint:
    folded_roles = {
        "compute_vector", "compute_fma", "compute_sfu", "compute_alu"}
    phases = tuple(
        dataclasses.replace(
            phase,
            compute_demands=tuple(
                dataclasses.replace(demand, resource_role="compute_vector")
                if demand.resource_role in folded_roles else demand
                for demand in phase.compute_demands),
        )
        for phase in blueprint.phases)
    return dataclasses.replace(blueprint, phases=phases)


def _remove_phase_compute_role(
        blueprint: _VectorPhaseBlueprint, resource_role: str,
        context: str) -> _VectorPhaseBlueprint:
    removed = tuple(
        demand
        for phase in blueprint.phases
        for demand in phase.compute_demands
        if demand.resource_role == resource_role)
    if len(removed) != 1 or removed[0].operations <= 0:
        raise ValueError(
            f"{context} must replace one positive {resource_role} phase")
    return dataclasses.replace(
        blueprint,
        phases=tuple(
            dataclasses.replace(
                phase,
                compute_demands=tuple(
                    demand for demand in phase.compute_demands
                    if demand.resource_role != resource_role),
            )
            for phase in blueprint.phases),
    )


def _phase_memory_demands(
        memory_paths: Tuple[MemoryPathDemand, ...], mem_model: str,
        phase: _VectorPhaseSpec
        ) -> Tuple[PhaseMemoryDemand, ...]:
    paths = {path.name: path for path in memory_paths}
    if mem_model == "gpu_mshr":
        read_path_name = "global_read"
        write_path_name = "global_write"
        read_roles = ("l2_cache", "dram")
        write_roles = ("l2_cache", "dram")
    elif mem_model == "npu_tile":
        read_path_name = "mte2_read"
        write_path_name = "mte3_write"
        read_roles = ("mte2", "l2_cache", "dram")
        write_roles = ("mte3", "l2_cache", "dram")
    else:
        raise ValueError(
            "Vector phase ledger requires gpu_mshr or npu_tile memory")

    demands = []
    if phase.l2_read_bytes > 0:
        if read_path_name not in paths:
            raise ValueError(
                f"Vector phase requires missing read path {read_path_name!r}")
        read_nodes = {
            node.resource_role for node in paths[read_path_name].nodes}
        required_read_roles = tuple(
            role for role in read_roles
            if role != "dram" or phase.dram_read_bytes > 0)
        missing_read_roles = set(required_read_roles) - read_nodes
        if missing_read_roles:
            raise ValueError(
                "Vector read path lacks required physical nodes: "
                f"{sorted(missing_read_roles)}")
        read_bytes_by_role = dict.fromkeys(
            required_read_roles, phase.l2_read_bytes)
        if "dram" in read_bytes_by_role:
            read_bytes_by_role["dram"] = phase.dram_read_bytes
        for role in required_read_roles:
            byte_count = read_bytes_by_role[role]
            demands.append(PhaseMemoryDemand(
                path_name=read_path_name,
                resource_role=role,
                bytes=byte_count,
            ))
    elif phase.dram_read_bytes != 0:
        raise ValueError("Vector phase has DRAM reads without ingress reads")

    if phase.write_bytes > 0:
        if write_path_name not in paths:
            raise ValueError(
                f"Vector phase requires missing write path "
                f"{write_path_name!r}")
        write_path = paths[write_path_name]
        write_nodes = {
            node.resource_role for node in write_path.nodes}
        expected_write_roles = tuple(
            role for role in write_roles if role in write_nodes)
        if not expected_write_roles or set(expected_write_roles) != write_nodes:
            raise ValueError(
                "Vector write path node schema does not match its hardware "
                f"mechanism: {sorted(write_nodes)}")
        dram_write_bytes = (
            phase.write_bytes
            if phase.dram_write_bytes is None else phase.dram_write_bytes)
        write_bytes_by_role = dict.fromkeys(
            expected_write_roles, phase.write_bytes)
        if "dram" in write_bytes_by_role:
            write_bytes_by_role["dram"] = dram_write_bytes
        for role in expected_write_roles:
            byte_count = write_bytes_by_role[role]
            if byte_count > 0:
                demands.append(PhaseMemoryDemand(
                    path_name=write_path_name,
                    resource_role=role,
                    bytes=byte_count,
                ))
    return tuple(demands)


def _attach_phase_ledger(
        inp: PredictionInput, blueprint: _VectorPhaseBlueprint,
        unassigned_compute: Tuple[PhaseComputeDemand, ...]) -> PredictionInput:
    ledger = _build_phase_ledger(
        inp.demand_per_resource,
        inp.memory_paths,
        inp.mem_model,
        blueprint,
        unassigned_compute,
    )
    return dataclasses.replace(inp, phase_ledger=ledger)


def _build_phase_ledger(
        demand_per_resource: Dict[str, float],
        memory_paths: Tuple[MemoryPathDemand, ...], mem_model: str,
        blueprint: _VectorPhaseBlueprint,
        unassigned_compute: Tuple[PhaseComputeDemand, ...]
        ) -> AnalyticalPhaseLedger:
    """Build a source-semantic phase ledger before final input assembly."""

    represented_roles = {
        demand.resource_role
        for phase in blueprint.phases
        for demand in phase.compute_demands
    } | {demand.resource_role for demand in unassigned_compute}
    scalar_aiv_operations = demand_per_resource["scalar_aiv"]
    if scalar_aiv_operations > 0 and "scalar_aiv" not in represented_roles:
        unassigned_compute = unassigned_compute + (PhaseComputeDemand(
            resource_role="scalar_aiv",
            operation="source_queue_buffer_control",
            operations=scalar_aiv_operations,
        ),)
    phases = tuple(
        AnalyticalPhase(
            name=phase.name,
            predecessors=phase.predecessors,
            compute_demands=phase.compute_demands,
            memory_demands=_phase_memory_demands(
                memory_paths, mem_model, phase),
            completion_barrier=phase.completion_barrier,
        )
        for phase in blueprint.phases)
    limitations = []
    if any(demand.resource_role == "warp_issue"
           for demand in unassigned_compute):
        limitations.append(
            "whole_kernel_warp_issue_phase_assignment_unavailable")
    if any(demand.operation == "backend_lowered_vector"
           for demand in unassigned_compute):
        limitations.append(
            "backend_lowered_vector_phase_assignment_unavailable")
    if any(demand.resource_role == "scalar_aiv"
           for demand in unassigned_compute):
        limitations.append(
            "coarse_scalar_aiv_phase_assignment_unavailable")
    if any(demand.operation == "cann_source_dag_phase_assignment_unavailable"
           for demand in unassigned_compute):
        limitations.append(
            "cann_source_dag_phase_assignment_unavailable")
    return AnalyticalPhaseLedger(
        phases=phases,
        unassigned_compute_demands=unassigned_compute,
        limitations=tuple(limitations),
    )


def cann_vector_observed_aiv_floor_ns(
        variant: str, observed_aiv_ns: float, block_dim: int, hardware):
    """Normalize an observed CANN AIV duration for Oracle consumption.

    Forward contains no operator-duration or AIV-duration regression.  Oracle
    may still consume a direct AIV observation.  The only additive term kept
    here is the independently measured NORM completion synchronization.
    """
    if not hardware.aic_aiv_separated:
        return None
    if (isinstance(observed_aiv_ns, bool)
            or not isinstance(observed_aiv_ns, (int, float))
            or not math.isfinite(observed_aiv_ns)
            or observed_aiv_ns <= 0):
        raise ValueError(
            "observed CANN AIV time must be finite and positive")
    if isinstance(block_dim, bool) or not isinstance(block_dim, int):
        raise TypeError("CANN AIV block_dim must be an integer")
    if block_dim <= 0:
        raise ValueError("CANN AIV block_dim must be positive")

    canonical_variant = variant.lower()
    aliases = {
        "rotary": "rope",
        "kv_cache": "kvcache",
        "paged_kv_cache": "kvcache",
    }
    if canonical_variant in aliases:
        canonical_variant = aliases[canonical_variant]
    if canonical_variant not in {
            "rmsnorm", "add_rmsnorm", "rope", "kvcache"}:
        return None

    observed_floor_ns = float(observed_aiv_ns)
    if canonical_variant in ("rmsnorm", "add_rmsnorm"):
        observed_floor_ns += _CANN_NORM_COMPLETION_SYNC_NS
    return observed_floor_ns


def _pim_vector_stages(kernel: KernelSpec, hardware):
    """Return the bank-stream stages of one VECTOR operator, or None.

    A bank-attached compute unit runs these operators directly on the array
    that holds their tensors.  Element-wise work is one read-modify-write pass
    over the data; a normalization is a reduction followed by a rescale of the
    data the reduction just consumed.  Variants outside that dataflow fall
    back to the generic aggregate representation.
    """

    variant = kernel.variant.lower()
    if variant in ("rmsnorm", "add_rmsnorm"):
        elements = _shape(kernel, "rows") * _shape(kernel, "hidden")
        stages = (normalization_stream(elements),)
        if variant == "add_rmsnorm":
            # PIM has no unified buffer, so the residual add is a real
            # element-wise pass rather than an in-place epilogue.
            stages = (pointwise_stream(elements),) + stages
        return stages
    if variant in ("activation", "silu"):
        elements = _shape(kernel, "tok") * _shape(kernel, "intermediate")
    elif variant in ("rotary", "rope"):
        rotary_dim = (
            _shape(kernel, "rotary_dim") if variant == "rope"
            else _shape(kernel, "head_dim"))
        elements = (
            _shape(kernel, "seq_len")
            * (_shape(kernel, "num_query_heads")
               + _shape(kernel, "num_kv_heads"))
            * rotary_dim)
    elif variant in ("kv_cache", "kvcache", "paged_kv_cache"):
        # One logical element count describes either K or V; both streams run
        # through the same read-modify-write protocol.
        elements = 2 * int(kernel.shape["element_count"])
    elif variant == "elementwise":
        elements = int(kernel.shape["element_count"])
    else:
        return None
    if elements <= 0:
        raise ValueError(f"PIM {variant} requires a positive element count")
    return (pointwise_stream(elements),)


def _architecture_input(inp: PredictionInput, kernel: KernelSpec, hardware):
    memory_model = hardware.kernel_memory_model
    if memory_model == "pim_bank":
        demand = dict(inp.demand_per_resource)
        for role in ("xbar", "smem", "l1_cache"):
            demand[role] = 0.0
        stages = _pim_vector_stages(kernel, hardware)
        if stages is None:
            # Outside the bank-stream dataflow: keep the generic aggregate
            # representation rather than reporting an unpriced zero.
            return dataclasses.replace(
                inp, demand_per_resource=demand, mem_model="pim_bank",
                active_units=hardware.num_compute_units), None
        # No cache stands between a bank-attached unit and its array, and the
        # unit has one SIMD datapath rather than the separate issue ports a
        # GPU-style instruction mix is charged against.  The bank stream
        # prices both.
        for role in ("l2_cache", "dram", "compute_fma", "compute_sfu",
                     "compute_alu"):
            demand[role] = 0.0
        return dataclasses.replace(
            inp, demand_per_resource=demand, mem_model="pim_bank",
            active_units=hardware.num_compute_units,
            # A bank-attached launch has no CTA waves to quantize against:
            # the partitioner spreads the operand over every unit, and the
            # bank stream carries the resulting per-unit column count.
            eta=1.0, wave_efficiency=1.0,
            execution_time_floor_ns=bank_stream_ns(stages, hardware)), None
    elif memory_model == "systolic_scratchpad":
        demand = dict(inp.demand_per_resource)
        for role in (
                "xbar", "smem", "l1_cache", "compute_fma",
                "compute_sfu", "compute_alu"):
            demand[role] = 0.0
        return dataclasses.replace(
            inp, demand_per_resource=demand,
            mem_model="systolic_scratchpad"), None
    elif memory_model == "npu_tile":
        return inp, None
    elif memory_model == "gpu_mshr":
        return inp, None
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")


def _attach_cann_kv_scalar_get_value_service(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        requests_per_token: int) -> PredictionInput:
    """Bind source-counted slot-map GetValue requests to AIV workers."""

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN KV scalar GetValue service requires separated npu_tile "
            "AIVs")
    if requests_per_token not in (1, 2):
        raise ValueError(
            "CANN KV scalar GetValue requests per token must be one or two")
    tokens = _shape(kernel, "seq_len")
    worker_count = min(tokens, inp.active_units)
    if worker_count != kernel.launch.grid_size:
        raise ValueError(
            "CANN KV scalar GetValue worker count must equal blockDim")
    base_tokens, remainder = divmod(tokens, worker_count)
    request_sizes_by_worker = tuple(
        (1,) * ((base_tokens + (1 if worker < remainder else 0))
                * requests_per_token)
        for worker in range(worker_count)
    )
    exact = ComputeExactRequestDemand(
        _CANN_SCALAR_GET_VALUE_EXACT_SERVICE,
        request_sizes_by_worker,
    )
    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN KV scalar GetValue requires one compute_vector demand")
    vector_demand = vector_demands[0]
    if (vector_demand.operations != exact.operations
            or vector_demand.service_demands
            or vector_demand.exact_request_demands):
        raise ValueError(
            "CANN KV scalar GetValue aggregate demand disagrees with source")
    typed_vector_demand = dataclasses.replace(
        vector_demand, exact_request_demands=(exact,))
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            typed_vector_demand if demand is vector_demand else demand
            for demand in inp.compute_demands),
    )


def _attach_cann_atb_bf16_rope_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False,
        ) -> PredictionInput:
    """Attach the source-defined ATB BF16 RoPE worker program.

    Public ATB key 22 processes tokens through one UB allocation, while key
    24 uses two token buffers.  Within a token both kernels broadcast and cast
    cos/sin once, then execute all Q slices followed by all K slices.  Every
    aligned Q/K slice performs two strided UB half-rotation copies, three
    BF16->FP32 casts, three multiplies, one add, one FP32->BF16 cast, and a
    store.  The plan retains those source-defined mechanism stages without
    modeling instructions or assigning a duration to the source's hard
    events.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "ATB BF16 RoPE pipeline requires separated npu_tile AIVs")
    if kernel.variant.lower() not in ("rotary", "rope"):
        raise ValueError("ATB BF16 RoPE pipeline requires a RoPE kernel")
    if kernel.dtype != "bf16":
        raise ValueError("ATB BF16 RoPE pipeline requires BF16")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "ATB BF16 RoPE pipeline must be attached exactly once")
    if not isinstance(analytical_compact, bool):
        raise TypeError("ATB BF16 RoPE analytical_compact must be boolean")

    parameters = kernel.launch.parameters
    required = (
        "cann_tiling_key",
        "lowered_vector_operations",
        _CANN_ATB_BF16_ROPE_DAG_PARAMETER,
        "atb_rope_slice_elements",
        "atb_rope_rotary_coefficient",
        "atb_rope_cos_format",
    )
    missing = tuple(name for name in required if name not in parameters)
    if missing:
        raise ValueError(
            f"ATB BF16 RoPE source parameters are incomplete: {missing}")
    if parameters[_CANN_ATB_BF16_ROPE_DAG_PARAMETER] != 1.0:
        raise ValueError("ATB BF16 RoPE DAG marker must equal one")

    tiling_key = int(parameters["cann_tiling_key"])
    if tiling_key not in (22, 24):
        raise ValueError("ATB BF16 RoPE pipeline requires key 22 or 24")
    tokens = _shape(kernel, "seq_len")
    head_dim = _shape(kernel, "head_dim")
    query_elements = _shape(kernel, "num_query_heads") * head_dim
    key_elements = _shape(kernel, "num_kv_heads") * head_dim
    elements = _shape(kernel, "element_count")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    if dtype_bytes != 2:
        raise ValueError("ATB BF16 RoPE requires two-byte elements")
    slice_elements = int(parameters["atb_rope_slice_elements"])
    expected_key = 24 if tokens >= 64 else 22
    nominal_slice = 2_048 if expected_key == 24 else 4_096
    expected_slice = nominal_slice // head_dim * head_dim
    if (tiling_key != expected_key
            or slice_elements != expected_slice
            or elements != tokens * (query_elements + key_elements)):
        raise ValueError(
            "ATB BF16 RoPE pipeline disagrees with source tiling or shape")

    worker_count = kernel.launch.grid_size
    if worker_count != inp.active_units:
        raise ValueError(
            "ATB BF16 RoPE worker count must equal active AIVs")
    base_tokens, remainder = divmod(tokens, worker_count)
    read_bytes = float(
        tokens
        * ((query_elements + key_elements + head_dim) * dtype_bytes + 4))
    write_bytes = float(
        tokens * (query_elements + key_elements) * dtype_bytes)
    if not math.isclose(
            inp.demand_per_resource["l2_cache"], read_bytes + write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "ATB BF16 RoPE L2 demand disagrees with its read/write ABI")
    if not math.isclose(
            inp.demand_per_resource["mte3"], write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("ATB BF16 RoPE write demand disagrees with its ABI")

    def _slices(extent):
        return tuple(
            min(slice_elements, extent - offset)
            for offset in range(0, extent, slice_elements))

    query_slices = _slices(query_elements)
    key_slices = _slices(key_elements)
    cos_cast_elements = min(query_elements, slice_elements)
    rounded_query_slices = tuple(
        math.ceil(size / 64) * 64 for size in query_slices)
    rounded_key_slices = tuple(
        math.ceil(size / 64) * 64 for size in key_slices)
    rounded_slices = rounded_query_slices + rounded_key_slices
    source_vector_compute = float(
        tokens
        * (2 * cos_cast_elements
           + sum(8 * rounded for rounded in rounded_slices)))
    base_tokens, remainder = divmod(tokens, worker_count)
    request_sizes_by_worker = tuple(
        (head_dim,) * (
            2 if tiling_key == 24
            and base_tokens + (1 if worker < remainder else 0) > 1
            else 1)
        for worker in range(worker_count))
    scalar_fill = ComputeExactRequestDemand(
        _CANN_LOCAL_SCALAR_FILL_SV_EXACT_SERVICE,
        request_sizes_by_worker,
    )
    query_heads_per_slice = cos_cast_elements // head_dim
    mask_copy_payload_bytes = head_dim * 4
    mask_copy_requests_by_worker = tuple(
        (mask_copy_payload_bytes,)
        * ((query_heads_per_slice - 1) * len(mask_requests))
        for mask_requests in request_sizes_by_worker)
    cos_sin_copy_payload_bytes = head_dim * dtype_bytes
    cos_sin_copy_requests_per_token = (
        (cos_sin_copy_payload_bytes,)
        * (2 * (query_heads_per_slice - 1)))
    cos_sin_copy_requests_by_worker = tuple(
        cos_sin_copy_requests_per_token
        * (base_tokens + (1 if worker < remainder else 0))
        for worker in range(worker_count))
    contiguous_copy_requests_by_worker = tuple(
        mask_requests + cos_sin_requests
        for mask_requests, cos_sin_requests in zip(
            mask_copy_requests_by_worker,
            cos_sin_copy_requests_by_worker))
    half_rotation_requests_per_token = tuple(
        payload_bytes
        for size in query_slices + key_slices
        for payload_bytes in (size * dtype_bytes // 2,) * 2)
    half_rotation_requests_by_worker = tuple(
        half_rotation_requests_per_token + tuple(
            payload_bytes
            for size in query_slices + key_slices
            for payload_bytes in (
                size * dtype_bytes // 2,
                size * dtype_bytes // 2,
            ) * (base_tokens + (1 if worker < remainder else 0) - 1))
        for worker in range(worker_count))
    half_rotation_block_bytes = head_dim * dtype_bytes // 2
    if half_rotation_block_bytes == 64:
        half_rotation_service = (
            _CANN_LOCAL_UB_COPY_STRIDE_64B_EXACT_SERVICE)
    elif half_rotation_block_bytes == 128:
        half_rotation_service = (
            _CANN_LOCAL_UB_COPY_STRIDE_128B_EXACT_SERVICE)
    else:
        half_rotation_service = None
    copy_schedules = (
        (
            _CANN_LOCAL_UB_COPY_CONTIGUOUS_EXACT_SERVICE,
            contiguous_copy_requests_by_worker,
        ),
        (half_rotation_service, half_rotation_requests_by_worker),
    )
    vector_resource = hardware.resource_by_role("compute_vector")
    exact_services = vector_resource.exact_request_services
    exact_copy_demands = []
    exact_copy_service_names = set()
    generic_copy_operations = 0.0
    local_copy_operations = 0.0
    for service, schedule in copy_schedules:
        request_sizes = {
            request_size
            for worker_requests in schedule
            for request_size in worker_requests}
        schedule_operations = float(math.fsum(
            request_size
            for worker_requests in schedule
            for request_size in worker_requests))
        local_copy_operations += schedule_operations
        service_is_covered = (
            service is not None
            and exact_services is not None
            and service in exact_services
            and request_sizes.issubset(
                exact_services[service][
                    "service_time_s_by_request_size"].keys()))
        if service_is_covered:
            exact_copy_demands.append(ComputeExactRequestDemand(
                service, schedule))
            exact_copy_service_names.add(service)
        else:
            generic_copy_operations += schedule_operations
    expected_compute = (
        source_vector_compute + scalar_fill.operations
        + local_copy_operations)
    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "ATB BF16 RoPE requires one aggregate Vector demand")
    if not math.isclose(
            vector_demands[0].operations, expected_compute,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "ATB BF16 RoPE Vector work disagrees with the source program")
    typed_rope_services = all(
        _has_compute_service(hardware, "compute_vector", service)
        for service in (
            "cast_bf16_to_fp32_serialized",
            "cast_fp32_to_bf16_serialized",
            "mul_serialized",
        ))
    maximum_tokens_per_worker = math.ceil(tokens / worker_count)
    if typed_rope_services:
        rounded_elements_per_token = math.fsum(rounded_slices)
        source_slice_count = len(rounded_slices)
        service_demands = (
                ComputeServiceDemand(
                    "cast_bf16_to_fp32_serialized",
                    float(tokens * (
                        2 * cos_cast_elements
                        + 3 * rounded_elements_per_token)),
                    serial_invocations=(
                        maximum_tokens_per_worker
                        * (2 + 3 * source_slice_count))),
                ComputeServiceDemand(
                    "cast_fp32_to_bf16_serialized",
                    float(tokens * rounded_elements_per_token),
                    serial_invocations=(
                        maximum_tokens_per_worker * source_slice_count)),
                ComputeServiceDemand(
                    "mul_serialized",
                    float(tokens * 3 * rounded_elements_per_token),
                    serial_invocations=(
                        maximum_tokens_per_worker
                        * 3 * source_slice_count)),
                ComputeServiceDemand(
                    "generic",
                    float(tokens * math.fsum(
                        rounded for rounded in rounded_slices)
                        + generic_copy_operations),
                ),
            )
    else:
        service_demands = (
            ComputeServiceDemand(
                "generic",
                source_vector_compute + generic_copy_operations),)
    vector_demand = vector_demands[0]
    typed_vector_demand = dataclasses.replace(
        vector_demand,
        service_demands=service_demands,
        exact_request_demands=(scalar_fill, *exact_copy_demands),
    )
    inp = dataclasses.replace(
        inp,
        compute_demands=tuple(
            typed_vector_demand if demand is vector_demand else demand
            for demand in inp.compute_demands),
    )

    def _copy_stage_demands(service, requests):
        if not requests:
            return (), ()
        if service in exact_copy_service_names:
            return (), (ComputeExactRequestDemand(service, (requests,)),)
        return (
            (ComputeServiceDemand("generic", float(math.fsum(requests))),),
            (),
        )

    def _token_stages(
            mask_requests, mask_copy_requests,
            include_key22_first_mte_completion,
            steady_mte2_request_count=None, *,
            source_entry_worker_count=None):
        if source_entry_worker_count is None:
            source_entry_worker_count = worker_count
        stages = []
        if mask_requests:
            mask_copy_services, mask_copy_exact = _copy_stage_demands(
                _CANN_LOCAL_UB_COPY_CONTIGUOUS_EXACT_SERVICE,
                mask_copy_requests)
            stages.append(PipelineStageDemand(
                "mask_prologue", "compute:compute_vector", None,
                compute_service_demands=mask_copy_services,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_LOCAL_SCALAR_FILL_SV_EXACT_SERVICE,
                        (mask_requests,)),
                    *mask_copy_exact,
                )))
        cos_copy_services, cos_copy_exact = _copy_stage_demands(
            _CANN_LOCAL_UB_COPY_CONTIGUOUS_EXACT_SERVICE,
            cos_sin_copy_requests_per_token)
        stages.append(PipelineStageDemand(
            "cos_sin_load", "memory:mte2_read",
            ((head_dim * dtype_bytes + 4) / read_bytes)))
        if cos_sin_copy_requests_per_token:
            stages.append(PipelineStageDemand(
                "cos_sin_broadcast", "compute:compute_vector", None,
                compute_service_demands=cos_copy_services,
                compute_exact_request_demands=cos_copy_exact))
        stages.append(PipelineStageDemand(
            "cos_sin_cast", "compute:compute_vector", None,
            compute_service_demands=(ComputeServiceDemand(
                ("cast_bf16_to_fp32_serialized"
                 if typed_rope_services else "generic"),
                float(2 * cos_cast_elements),
                serial_invocations=(2 if typed_rope_services else 0)),),
            prior_resources=(
                ("memory:mte3_write",) if tiling_key == 24 else ())))
        # Key 24's ping-pong source may issue the next Q/K load after the
        # preceding store starts.  Its Vector work still waits for that store
        # to complete through MTE3_V, so retain both the issue-order and
        # completion edges instead of serializing the independent MTE pipes.
        previous_compute = None
        previous_store = None
        for prefix, slices in (("query", query_slices), ("key", key_slices)):
            for index, current_slice in enumerate(slices):
                payload_bytes = current_slice * dtype_bytes
                # Key 22's evidence is explicitly limited to the first Q
                # slice per worker.  Key 24 uses the same first full-Q-slice
                # hard edge at each token boundary in its ping-pong program;
                # later slices retain their steady byte-capacity stage.
                use_exact_completion = (
                    prefix == "query" and index == 0 and (
                        tiling_key == 24
                        or (tiling_key == 22
                            and include_key22_first_mte_completion)))
                steady_mte2_completion = (
                    _cann_rope_key24_steady_mte2_completion_request(
                        hardware, payload_bytes, worker_count,
                        steady_mte2_request_count)
                    if prefix == "query" and index == 0
                    and steady_mte2_request_count is not None else ())
                mte2_completion = (
                    steady_mte2_completion
                    if steady_mte2_completion else (
                        _cann_rope_completion_request(
                            hardware, tiling_key, "mte2", payload_bytes,
                            source_entry_worker_count)
                        if use_exact_completion else ()))
                mte3_completion = (
                    _cann_rope_completion_request(
                        hardware, tiling_key, "mte3", payload_bytes,
                        source_entry_worker_count)
                    if use_exact_completion else ())
                load_name = f"{prefix}_load_{index}"
                rotation_name = f"{prefix}_half_rotation_copy_{index}"
                compute_name = f"{prefix}_compute_{index}"
                store_name = f"{prefix}_store_{index}"
                overlap_prior_store = (
                    tiling_key == 24 and previous_store is not None)
                stages.append(PipelineStageDemand(
                    load_name, "memory:mte2_read",
                    payload_bytes / read_bytes,
                    memory_exact_request_demands=mte2_completion,
                    predecessors=(
                        (previous_compute,)
                        if overlap_prior_store else None),
                    start_predecessors=(
                        (previous_store,) if overlap_prior_store else ())))
                rotation_requests = (
                    current_slice * dtype_bytes // 2,
                    current_slice * dtype_bytes // 2,
                )
                rotation_copy_services, rotation_copy_exact = (
                    _copy_stage_demands(
                        half_rotation_service, rotation_requests))
                stages.append(PipelineStageDemand(
                    rotation_name,
                    "compute:compute_vector", None,
                    compute_service_demands=rotation_copy_services,
                    compute_exact_request_demands=rotation_copy_exact,
                    predecessors=((load_name,) if tiling_key == 24 else None)))
                compute_predecessors = None
                compute_prior_resources = ()
                if tiling_key == 24:
                    compute_predecessors = (
                        (rotation_name,) if previous_store is None
                        else (rotation_name, previous_store))
                    if previous_store is None:
                        compute_prior_resources = ("memory:mte3_write",)
                stages.extend((
                    PipelineStageDemand(
                        compute_name,
                        "compute:compute_vector", None,
                        compute_service_demands=(
                            (
                                ComputeServiceDemand(
                                    "cast_bf16_to_fp32_serialized",
                                    float(3 * math.ceil(
                                        current_slice / 64) * 64),
                                    serial_invocations=3),
                                ComputeServiceDemand(
                                    "cast_fp32_to_bf16_serialized",
                                    float(math.ceil(
                                        current_slice / 64) * 64),
                                    serial_invocations=1),
                                ComputeServiceDemand(
                                    "mul_serialized",
                                    float(3 * math.ceil(
                                        current_slice / 64) * 64),
                                    serial_invocations=3),
                                ComputeServiceDemand(
                                    "generic",
                                    float(math.ceil(
                                        current_slice / 64) * 64)),
                            )
                            if typed_rope_services
                            else (ComputeServiceDemand(
                                "generic",
                                float(8 * math.ceil(
                                    current_slice / 64) * 64)),)
                        ),
                        predecessors=compute_predecessors,
                        prior_resources=compute_prior_resources),
                    PipelineStageDemand(
                        store_name, "memory:mte3_write",
                        payload_bytes / write_bytes,
                        memory_exact_request_demands=mte3_completion,
                        predecessors=(
                            (compute_name,) if tiling_key == 24 else None)),
                ))
                previous_compute = compute_name
                previous_store = store_name
        return tuple(stages)

    if analytical_compact:
        # The source gives exactly two possible worker programs: the first
        # ``remainder`` workers own one extra token and every other AIV owns
        # ``base_tokens``.  Represent each program once and retain its exact
        # physical-worker population.  This is the source's analytical worker
        # envelope, not a per-AIV or per-token materialization.
        worker_classes = []
        if remainder:
            worker_classes.append((
                tuple(range(remainder)), base_tokens + 1,
            ))
        if remainder < worker_count:
            worker_classes.append((
                tuple(range(remainder, worker_count)), base_tokens,
            ))
        source_entry_classes = []
        for physical_workers, token_count in worker_classes:
            if physical_workers[0] == 0:
                source_entry_classes.append(((0,), token_count, 1))
                physical_workers = physical_workers[1:]
            if physical_workers:
                source_entry_classes.append((
                    physical_workers, token_count, worker_count))
        tasks = []
        for compact_worker, (
                physical_workers, token_count, source_entry_worker_count,
                ) in enumerate(source_entry_classes):
            representative = physical_workers[0]
            worker_tasks = [PipelineTaskDemand(
                name=f"class{compact_worker}_first_token",
                worker=compact_worker,
                stages=_token_stages(
                    request_sizes_by_worker[representative],
                    mask_copy_requests_by_worker[representative], True,
                    source_entry_worker_count=source_entry_worker_count),
            )]
            if token_count > 1:
                worker_tasks.append(PipelineTaskDemand(
                    name=f"class{compact_worker}_remaining_tokens",
                    worker=compact_worker,
                    stages=_token_stages(
                        (), (), False, token_count - 1),
                    repeat_count=token_count - 1,
                ))
            sequence = PipelineTaskSequenceDemand(
                name=f"class{compact_worker}_program",
                worker=compact_worker,
                tasks=tuple(worker_tasks),
            )
            tasks.append(PipelineTaskProgramDemand(
                name=f"class{compact_worker}_program_envelope",
                worker=compact_worker,
                patterns=(sequence,),
                order=(0,),
                worker_multiplicity=len(physical_workers),
                physical_workers=physical_workers,
            ))
    else:
        tasks = []
        for worker in range(worker_count):
            token_count = base_tokens + (1 if worker < remainder else 0)
            mask_requests = request_sizes_by_worker[worker]
            worker_tasks = [PipelineTaskDemand(
                name=f"worker{worker}_first_token",
                worker=worker,
                stages=_token_stages(
                    mask_requests, mask_copy_requests_by_worker[worker], True,
                    source_entry_worker_count=(1 if worker == 0
                                               else worker_count)),
            )]
            if token_count > 1:
                worker_tasks.append(PipelineTaskDemand(
                    name=f"worker{worker}_remaining_tokens",
                    worker=worker,
                    stages=_token_stages(
                        (), (), False, token_count - 1),
                    repeat_count=token_count - 1,
                ))
            tasks.append(PipelineTaskSequenceDemand(
                name=f"worker{worker}_program",
                worker=worker,
                tasks=tuple(worker_tasks),
            ))
    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tuple(tasks),
        buffers=(PipelineBuffer(
            name="token_buffer",
            producer_stage="cos_sin_load",
            consumer_stage=f"key_store_{len(key_slices) - 1}",
            slots=2 if tiling_key == 24 else 1,
            scope="worker",
        ),),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    limitations = (
        ("atb_rope_local_ub_copy_service_evidence_pending",)
        if generic_copy_operations > 0 else ())
    slice_payloads = {
        current_slice * dtype_bytes
        for current_slice in query_slices + key_slices}
    hard_edge_evidence_partial = tiling_key == 22 or any(
        not _cann_rope_completion_request(
            hardware, tiling_key, resource_role, payload_bytes, worker_count)
        for resource_role in ("mte2", "mte3")
        for payload_bytes in slice_payloads)
    steady_mte2_evidence_partial = tiling_key == 24 and any(
        not _cann_rope_key24_steady_mte2_completion_request(
            hardware, slice_elements * dtype_bytes, worker_count,
            token_count - 1)
        for token_count in {base_tokens, base_tokens + int(remainder > 0)}
        if token_count > 1)
    return dataclasses.replace(
        attached,
        # The ATB source assigns a complete token program to every launched
        # AIV.  Its worker DAG is concurrent after dispatch: descriptor
        # progress is not a serial prefix for every token or worker.  Price
        # task admission once with the device-counter worker-entry fill.  The
        # token program above is the complete source body, so the runtime
        # contributes only the independently measured empty AIV task shell,
        # exactly like the other source-closed CANN AIV programs; a complete
        # nonempty-task curve used as a competing floor would instead hide the
        # explicitly represented operator body inside a generic task duration.
        # The partially observed multi-worker MTE spans above are deliberately
        # not replayed as worker-local edges.
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
        limitations=attached.limitations + (
            ("atb_rope_hard_event_latency_evidence_partial",)
            if hard_edge_evidence_partial else ()) + (
            ("atb_rope_steady_mte2_completion_evidence_partial",)
            if steady_mte2_evidence_partial else ()) + limitations,
    )


_CANN_PUBLIC_ROPE_GATHER_V3_DIRECT_KEY = 10_330
_CANN_PUBLIC_ROPE_GATHER_V3_TILING_PAYLOAD_BYTES = 192


def _cann_public_rope_index_select_geometry(
        kernel: KernelSpec, hardware) -> Tuple[int, int, int, int, int]:
    """Validate one selected GatherV3 launch and return its typed geometry.

    This is deliberately a launch-contract validator, not a second tiler.
    The Ascend launch package supplies the selected CANN 8.5.1 program and
    its ``realCoreNum``.  Keeping those facts at KernelSpec makes Dataset and
    Serving consume the same program instead of inferring a key from a case
    name or a duration.
    """

    if (hardware.kernel_memory_model != "npu_tile"
            or not hardware.aic_aiv_separated):
        raise ValueError(
            "public RoPE IndexSelect requires Ascend separated AIV hardware")
    if kernel.variant not in ("rope", "rotary") or kernel.dtype != "bf16":
        raise ValueError(
            "public RoPE IndexSelect requires a BF16 rope KernelSpec")
    parameters = kernel.launch.parameters
    required = (
        SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
        SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
        "cann_rope_gather_v3_real_core_num",
        "cann_rope_gather_v3_selected_row_bytes",
        "cann_rope_gather_v3_tiling_payload_bytes",
    )
    missing = tuple(name for name in required if name not in parameters)
    if missing:
        raise ValueError(
            "public CANN RoPE GatherV3 launch contract is incomplete: "
            f"{missing}")
    gather_count = parameters[SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER]
    if gather_count != 2.0:
        raise ValueError(
            "public CANN RoPE requires exactly two IndexSelect launches")
    program_key = int(parameters[SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER])
    if program_key != _CANN_PUBLIC_ROPE_GATHER_V3_DIRECT_KEY:
        raise ValueError(
            "public CANN RoPE GatherV3 has an unsupported source program: "
            f"{program_key}")
    tokens = _shape(kernel, "seq_len")
    row_elements = _shape(kernel, "head_dim")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    if dtype_bytes != 2:
        raise ValueError("public CANN RoPE IndexSelect requires BF16 rows")
    vector_cores = int(
        hardware.resource_by_role("compute_vector").total_units)
    worker_count = int(parameters["cann_rope_gather_v3_real_core_num"])
    if worker_count != min(tokens, vector_cores):
        raise ValueError(
            "public CANN RoPE GatherV3 realCoreNum disagrees with its "
            "source row ownership")
    selected_row_bytes = int(
        parameters["cann_rope_gather_v3_selected_row_bytes"])
    if selected_row_bytes != tokens * row_elements * dtype_bytes:
        raise ValueError(
            "public CANN RoPE GatherV3 selected-row bytes disagree with "
            "the [T, D] BF16 output ABI")
    if parameters["cann_rope_gather_v3_tiling_payload_bytes"] != (
            float(_CANN_PUBLIC_ROPE_GATHER_V3_TILING_PAYLOAD_BYTES)):
        raise ValueError(
            "public CANN RoPE GatherV3 requires its fixed 192-byte tiling "
            "payload")
    if worker_count <= 0:
        raise ValueError("public CANN RoPE IndexSelect requires active rows")
    return tokens, row_elements * dtype_bytes, worker_count, program_key, int(
        gather_count)


def _cann_public_rope_index_select_input(
        kernel: KernelSpec, hardware) -> PredictionInput:
    """Build one source-selected GatherV3 prefix of public CANN RoPE.

    Tmpl0/10330 directly copies the selected rows.  The complete RoPE cache
    may be much larger than the current token batch, but the public CANN 8.5.1
    ABI still dispatches this direct-row program; its untouched cache rows do
    not enter traffic demand.

    The 10330 TQue lifetime has an independent mechanism service.
    """

    tokens, row_bytes, worker_count, program_key, _ = (
        _cann_public_rope_index_select_geometry(kernel, hardware))
    dtype_bytes = _shape(kernel, "dtype_bytes")
    row_elements = row_bytes // dtype_bytes
    output_elements = tokens * row_elements
    index_bytes = tokens * 4
    if program_key != _CANN_PUBLIC_ROPE_GATHER_V3_DIRECT_KEY:
        raise AssertionError("validated public RoPE GatherV3 key drifted")
    read_bytes = float(output_elements * dtype_bytes + index_bytes)
    dram_read_bytes = read_bytes
    vector_operations = float(tokens)
    write_bytes = float(output_elements * dtype_bytes)
    gather_kernel = KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=kernel.hardware_id,
        shape={"element_count": output_elements, "seq_len": tokens},
        dtype=kernel.dtype,
        variant="gather",
        launch=LaunchConfig(
            algorithm="cann_gather_v3_public_rope",
            grid=(worker_count, 1, 1),
            block=(1, 1, 1),
            # The public RoPE launch owns the GatherV3 source-program key.
            # This nested KernelSpec only gives the common Formula a typed
            # workload envelope; carrying the key here would make it look
            # like an independently dispatched generic Gather operation.
            parameters={},
            execution_variant="gather",
        ),
        attributes={
            "gather_factor": 1,
            "index_count": tokens,
            "index_bytes": 4,
            "scattered": False,
        },
    )
    demand = _vector_demand(
        hardware,
        l2_read_bytes=read_bytes,
        dram_read_bytes=dram_read_bytes,
        write_bytes=write_bytes,
        flops={
            "compute_tensor": 0.0,
            "compute_fma": 0.0,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
            "compute_vector": vector_operations,
        },
    )
    inp = _input(
        gather_kernel,
        hardware,
        demand,
        write_bytes=write_bytes,
        phase_blueprint=None,
        dram_write_bytes=write_bytes,
    )
    inp = _attach_cann_kv_scalar_get_value_service(
        inp, gather_kernel, hardware, 1)

    base_rows, remainder = divmod(tokens, worker_count)
    worker_classes = ()
    if remainder:
        worker_classes += ((tuple(range(remainder)), base_rows + 1),)
    if remainder < worker_count:
        worker_classes += ((tuple(range(remainder, worker_count)), base_rows),)
    # ``ProcessSingleTile10330`` prefetches the worker's whole index extent
    # once, then walks its rows in ``ubLineLimit_`` groups.  Every row issues
    # its own ``CopyInDataLine`` DataCopy of ``aSize_`` elements after the
    # scalar ``GetValue`` supplies that row's address, and one grouped
    # ``DoCopyUB2GM`` writes the group out.  The captured 10330 tiling sets
    # ``ubLineLimit`` to the maximum per-worker row count (1 at one row per
    # worker, 2 at two), so a worker runs exactly one group.  The row loads
    # are therefore ``rows`` separate source MTE2 requests, not one bulk
    # transfer; only their completion count is taken from the independent
    # request-completion mechanism, and the same bytes stay on the capacity
    # stage under ``max``.
    programs = []
    row_prefix_bytes, row_completion = (
        _cann_memory_completion_prefix_request(
            hardware, "mte2", _CANN_MTE2_COMPLETION_EXACT_SERVICE,
            row_bytes))
    for compact_worker, (physical_workers, rows) in enumerate(worker_classes):
        index_completion = _cann_memory_completion_prefix_request(
            hardware, "mte2", _CANN_MTE2_COMPLETION_EXACT_SERVICE,
            rows * 4)[1]
        row_load_completion = tuple(
            dataclasses.replace(
                demand,
                request_sizes_by_worker=((row_prefix_bytes,) * rows,))
            for demand in row_completion)
        store_completion = _cann_memory_completion_prefix_request(
            hardware, "mte3", _CANN_MTE3_COMPLETION_EXACT_SERVICE,
            rows * row_bytes)[1]
        stages = (
            PipelineStageDemand(
                "buffer_lifecycle", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_GATHER_V3_TWO_TQUE_LIFECYCLE_EXACT_SERVICE,
                        ((1,),)),)),
            PipelineStageDemand(
                "index_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),)),),
                predecessors=("buffer_lifecycle",)),
            PipelineStageDemand(
                "index_load", "memory:mte2_read", rows * 4 / read_bytes,
                memory_exact_request_demands=index_completion,
                predecessors=("index_queue_control",)),
            PipelineStageDemand(
                "index_decode", "compute:compute_vector", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_SCALAR_GET_VALUE_EXACT_SERVICE,
                        ((1,) * rows,)),),
                predecessors=("index_load",)),
            PipelineStageDemand(
                "data_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),)),),
                predecessors=("index_decode",)),
            PipelineStageDemand(
                "source_rows_load", "memory:mte2_read",
                rows * row_bytes / read_bytes,
                memory_exact_request_demands=row_load_completion,
                predecessors=("data_queue_control",)),
            PipelineStageDemand(
                "output_rows_store", "memory:mte3_write",
                rows * row_bytes / write_bytes,
                memory_exact_request_demands=store_completion,
                predecessors=("source_rows_load",)),
        )
        sequence = PipelineTaskSequenceDemand(
            name=f"class{compact_worker}_gather_sequence",
            worker=compact_worker,
            tasks=(PipelineTaskDemand(
                name=f"class{compact_worker}_gather_tile",
                worker=compact_worker,
                stages=stages),),
        )
        programs.append(PipelineTaskProgramDemand(
            name=f"class{compact_worker}_gather_envelope",
            worker=compact_worker,
            patterns=(sequence,),
            order=(0,),
            worker_multiplicity=len(physical_workers),
            physical_workers=physical_workers,
        ))
    inp = _attach_scalar_aiv_exact_demands(
        inp,
        (
            ComputeExactRequestDemand(
                _CANN_GATHER_V3_TWO_TQUE_LIFECYCLE_EXACT_SERVICE,
                ((1,),) * worker_count),
            ComputeExactRequestDemand(
                _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                ((1, 1),) * worker_count),
        ),
    )
    attached = attach_pipeline_work_plan(
        inp,
        hardware,
        PipelineWorkPlan(worker_count=worker_count, tasks=tuple(programs)),
        promote_npu_partial=True,
    )
    # The 10330 worker body above is now the complete source program: index
    # prefetch, per-row source requests, and the grouped store.  Following the
    # other source-closed CANN AIV programs, the runtime therefore contributes
    # only the independently measured empty AIV task shell.  Keeping a
    # complete nonempty-task curve as the competing floor instead would hide
    # the explicitly represented operator body inside a generic task duration.
    return dataclasses.replace(
        attached,
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
        task_prologue_contract=CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE,
        task_prologue_repetitions=1,
        limitations=attached.limitations,
    )


def _cann_public_rope_index_select_duration_s(
        kernel: KernelSpec, hardware) -> Tuple[float, float, int]:
    """Price public GatherV3 through the same Formula path as Vector work."""

    _, _, _, _, gather_count = _cann_public_rope_index_select_geometry(
        kernel, hardware)
    prediction = PerformanceFormula(hardware).predict_kernel(
        _cann_public_rope_index_select_input(kernel, hardware))
    return prediction.t_total_s, prediction.t_launch_s, gather_count


def _append_cann_public_rope_index_selects(
        prediction: KernelPrediction, kernel: KernelSpec,
        hardware) -> KernelPrediction:
    """Serially compose public RoPE's two source-selected GatherV3 tasks."""

    # The main RoPE spec retains its logical shape at the Serving boundary,
    # while launch selection is performed on the captured model-input shape.
    # Bind a padded CANN source program to that physical shape before pricing
    # its source-authored IndexSelect prefix.  Omitting the prefix merely
    # because the graph is padded makes a larger captured launch cheaper than
    # the corresponding unpadded launch.
    execution_kernel = _cann_full_graph_source_program_execution_kernel(
        kernel, hardware)
    parameters = execution_kernel.launch.parameters
    if SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER not in parameters:
        return prediction
    gather_s, gather_launch_s, gather_count = (
        _cann_public_rope_index_select_duration_s(execution_kernel, hardware))
    gather_sequence_s = gather_count * gather_s
    group_name = "public_op_index_select_sequence"
    groups = dict(prediction.t_per_group)
    groups[group_name] = gather_sequence_s
    resources = dict(prediction.t_per_resource)
    resources[group_name] = gather_sequence_s
    diagnostics = prediction.diagnostics
    if diagnostics is not None:
        diagnostics = dataclasses.replace(
            diagnostics,
            limitations=tuple(dict.fromkeys(diagnostics.limitations + (
                "public_rope_index_select_diagnostics_are_serial_subtasks",
            ))),
        )
    return KernelPrediction(
        t_total_s=prediction.t_total_s + gather_sequence_s,
        t_per_group=groups,
        t_launch_s=prediction.t_launch_s + gather_count * gather_launch_s,
        bottleneck_group=(
            group_name
            if gather_sequence_s > prediction.t_total_s
            else prediction.bottleneck_group),
        t_per_resource=resources,
        diagnostics=diagnostics,
    )


def _cann_memory_completion_request(
        hardware, resource_role: str, service: str,
        payload_bytes: int) -> Tuple[MemoryExactRequestDemand, ...]:
    """Return an exact completion floor when its size is measured."""

    resource = hardware.resource_by_role(resource_role)
    exact_services = resource.exact_request_services
    if (exact_services is None or service not in exact_services
            or payload_bytes not in exact_services[service][
                "service_time_s_by_request_size"]):
        return ()
    return (MemoryExactRequestDemand(
        resource_role, service, ((payload_bytes,),)),)


def _cann_memory_completion_prefix_request(
        hardware, resource_role: str, service: str,
        payload_bytes: int) -> Tuple[
            int, Tuple[MemoryExactRequestDemand, ...]]:
    """Return the largest measured aligned prefix of one memory request.

    Exact completion evidence covers the latency-bearing prefix of the same
    source request.  A larger request therefore retains that causal prefix and
    sends only its remaining bytes through the capacity stage.  Selection is
    based solely on byte coverage; service-time values never choose a cell.
    """

    resource = hardware.resource_by_role(resource_role)
    exact_services = resource.exact_request_services
    if exact_services is None or service not in exact_services:
        return 0, ()
    covered_sizes = tuple(
        request_size
        for request_size in exact_services[service][
            "service_time_s_by_request_size"]
        if request_size <= payload_bytes)
    if not covered_sizes:
        return 0, ()
    prefix_bytes = max(covered_sizes)
    return prefix_bytes, (MemoryExactRequestDemand(
        resource_role, service, ((prefix_bytes,),)),)


def _cann_rope_key24_steady_mte2_completion_request(
        hardware, payload_bytes: int, worker_count: int,
        request_count: Optional[int]) -> Tuple[MemoryExactRequestDemand, ...]:
    """Return one exact per-request key-24 steady MTE2 service."""

    if (payload_bytes != 4096 or request_count is None
            or (worker_count, request_count)
            not in _CANN_ROPE_KEY24_STEADY_MTE2_EXACT_DOMAIN):
        return ()
    return _cann_memory_completion_request(
        hardware, "mte2", _CANN_ROPE_KEY24_STEADY_MTE2_EXACT_SERVICE,
        request_count)


def _cann_rope_completion_request(
        hardware, tiling_key: int, resource_role: str, payload_bytes: int,
        worker_count: int) -> Tuple[MemoryExactRequestDemand, ...]:
    """Return an exact source-entry completion for CANN RoPE.

    Population-specific probes measure the maximum edge completion across
    the simultaneously admitted AIV set.  The first-token worker programs
    execute concurrently, so assigning that exact span to their common first
    Q-slice boundary preserves the measured launch-wide maximum without
    multiplying it by the worker population.  Unsupported populations remain
    unavailable; no interpolation or nearest-population fallback is allowed.
    """

    if tiling_key == 22 and resource_role == "mte2":
        services = _CANN_ROPE_KEY22_FIRST_Q_SLICE_MTE2_TO_VECTOR_EXACT_SERVICES
    elif tiling_key == 22 and resource_role == "mte3":
        services = _CANN_ROPE_KEY22_FIRST_Q_SLICE_VECTOR_TO_MTE3_EXACT_SERVICES
    elif tiling_key == 24 and resource_role == "mte2":
        services = _CANN_ROPE_MTE2_TO_VECTOR_EXACT_SERVICES
    elif tiling_key == 24 and resource_role == "mte3":
        services = _CANN_ROPE_VECTOR_TO_MTE3_EXACT_SERVICES
    elif tiling_key not in (22, 24):
        raise ValueError("RoPE completion requires tiling key 22 or 24")
    else:
        raise ValueError(
            "RoPE completion resource role must be mte2 or mte3")
    if worker_count not in services:
        return ()
    return _cann_memory_completion_request(
        hardware, resource_role, services[worker_count], payload_bytes)


def _cann_unloaded_memory_completion_request(
        hardware, resource_role: str, service: str, payload_bytes: int,
        worker_count: int) -> Tuple[MemoryExactRequestDemand, ...]:
    """Return an exact completion in a measured worker-admission domain."""

    if worker_count == 1:
        selected_service = service
    elif worker_count == 2:
        if (resource_role == "mte2"
                and service
                == _CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE):
            selected_service = (
                _CANN_DATACOPYPAD_MTE2_TO_VECTOR_TWO_WORKER_EXACT_SERVICE)
        elif (resource_role == "mte3"
              and service
              == _CANN_DATACOPYPAD_VECTOR_TO_MTE3_EXACT_SERVICE):
            selected_service = (
                _CANN_DATACOPYPAD_VECTOR_TO_MTE3_TWO_WORKER_EXACT_SERVICE)
        else:
            return ()
    elif (resource_role == "mte2"
          and service == _CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE
          and worker_count
          in _CANN_DATACOPYPAD_MTE2_TO_VECTOR_ROW_GROUP_EXACT_SERVICES):
        selected_service = (
            _CANN_DATACOPYPAD_MTE2_TO_VECTOR_ROW_GROUP_EXACT_SERVICES[
                worker_count])
    else:
        return ()
    return _cann_memory_completion_request(
        hardware, resource_role, selected_service, payload_bytes)








def _cann_swiglu_queue_completion_request(
        hardware, resource_role: str, payload_bytes: int,
        worker_count: int) -> Tuple[MemoryExactRequestDemand, ...]:
    """Return only an exact production-valid SwiGLU queue-edge cell."""

    if resource_role == "mte2":
        services = _CANN_SWIGLU_TQUE_MTE2_TO_VECTOR_EXACT_SERVICES
    elif resource_role == "mte3":
        services = _CANN_SWIGLU_TQUE_VECTOR_TO_MTE3_EXACT_SERVICES
    else:
        raise ValueError(
            "SwiGLU queue completion resource role must be mte2 or mte3")
    if worker_count not in services:
        return ()
    return _cann_memory_completion_request(
        hardware, resource_role, services[worker_count], payload_bytes)


def _cann_swiglu_input_pair_service(
        hardware, worker_count: int, queue_depth: int) -> Optional[str]:
    """Return an exact first-source-pair service if published."""

    domain = (worker_count, queue_depth)
    if domain not in _CANN_SWIGLU_TQUE_INPUT_PAIR_EXACT_SERVICES:
        return None
    service = _CANN_SWIGLU_TQUE_INPUT_PAIR_EXACT_SERVICES[domain]
    exact_services = hardware.resource_by_role("mte2").exact_request_services
    if exact_services is None or service not in exact_services:
        return None
    return service


def _cann_equal_kv_chain_completion_request(
        hardware, payload_bytes: int,
        worker_count: int, *, cold_source_entry: bool,
        first_token: bool,
        one_token_per_worker: bool) -> Tuple[MemoryExactRequestDemand, ...]:
    """Return one source-entry-aware key-200 chain completion cell.

    Worker zero executes the launch's cold source entry before the remaining
    descriptors can observe the concurrent first-token chain.  The 48-AIV
    cell is paired from that first interval; it is not a cache-steady service
    that may be repeated for every later token.  Only the dedicated warm
    cells may describe a cache-steady completion, and they are published
    solely for one-token-per-worker launches.
    """

    if not isinstance(first_token, bool):
        raise TypeError("CANN KV first-token flag must be boolean")
    if cold_source_entry:
        service = _CANN_KV_EQUAL_CHAIN_EXACT_SERVICES[1]
    elif (first_token
          and worker_count in _CANN_KV_EQUAL_CHAIN_EXACT_SERVICES):
        service = _CANN_KV_EQUAL_CHAIN_EXACT_SERVICES[worker_count]
    elif (one_token_per_worker
          and worker_count in _CANN_KV_EQUAL_WARM_CHAIN_EXACT_SERVICES):
        service = _CANN_KV_EQUAL_WARM_CHAIN_EXACT_SERVICES[worker_count]
    else:
        return ()
    return _cann_memory_completion_request(
        hardware, "mte3", service, payload_bytes)


def _cann_unequal_kv_repeated_pair_completion_request(
        hardware, payload_bytes: int, worker_count: int,
        repeat_count: int) -> Tuple[MemoryExactRequestDemand, ...]:
    """Return one exact maximum-worker repeated key-202 K/V-pair cell."""

    if ((worker_count, payload_bytes, repeat_count)
            not in _CANN_KV_UNEQUAL_REPEATED_PAIR_EXACT_DOMAIN):
        return ()
    return _cann_memory_completion_request(
        hardware, "mte3", _CANN_KV_UNEQUAL_REPEATED_PAIR_EXACT_SERVICE,
        repeat_count)


def _cann_reshape_cache_worker_repeats(tokens, inp, kernel):
    """Return the ReshapeAndCache worker count and its maximum repeat count.

    The launch runs one worker per token up to the active unit count, so the
    worker count is the smaller of the two and must agree with the launch's
    own blockDim; each worker then takes an even share of the tokens with the
    remainder forming one extra repeat on the leading workers.
    """

    worker_count = min(tokens, inp.active_units)
    if worker_count != kernel.launch.grid_size:
        raise ValueError(
            "CANN ReshapeAndCache worker count must equal blockDim")
    base_repeats, remainder = divmod(tokens, worker_count)
    maximum_repeats = base_repeats + int(remainder > 0)
    return worker_count, maximum_repeats, base_repeats, remainder


def _attach_cann_equal_kv_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False,
        ) -> PredictionInput:
    """Attach the source-proven CANN key-200 K/V serialization.

    CANN 8.5.1 ``ProcessKVEqual`` owns one ``tokenBuf_`` and executes, for
    every token, K CopyIn, K CopyOut, V CopyIn, and V CopyOut with a hard event
    wait between adjacent operations.  The buffer is not double-buffered, so
    its lifetime also prevents the next token's K load from starting before
    the current token's V store completes.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN ReshapeAndCache pipeline requires separated npu_tile AIVs")
    if kernel.variant.lower() not in (
            "kv_cache", "kvcache", "paged_kv_cache"):
        raise ValueError(
            "CANN ReshapeAndCache pipeline requires a KV-cache kernel")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN ReshapeAndCache pipeline must be attached exactly once")
    if not isinstance(analytical_compact, bool):
        raise TypeError("CANN equal KV analytical_compact must be boolean")

    tokens = _shape(kernel, "seq_len")
    num_kv_heads = _shape(kernel, "num_kv_heads")
    head_dim = _shape(kernel, "head_dim")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    tensor_bytes_per_token = num_kv_heads * head_dim * dtype_bytes
    slot_bytes_per_token = 4
    read_bytes = float(
        tokens * (2 * tensor_bytes_per_token + slot_bytes_per_token))
    write_bytes = float(tokens * 2 * tensor_bytes_per_token)
    if not math.isclose(
            inp.demand_per_resource["l2_cache"], read_bytes + write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "CANN ReshapeAndCache pipeline L2 demand disagrees with its ABI")
    if not math.isclose(
            inp.demand_per_resource["mte3"], write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "CANN ReshapeAndCache pipeline write demand disagrees with its ABI")

    (worker_count, maximum_repeats, base_repeats,
     remainder) = _cann_reshape_cache_worker_repeats(tokens, inp, kernel)
    # The published multi-token cells time a *complete* maximum-worker
    # interval at a few exact repeat counts.  They cannot be reused as the
    # per-token steady service, and substituting them for the source DAG
    # makes a non-uniform worker tail switch to a different model at the
    # next repeat count.  Keep every repeated token on the same one-token
    # ProcessKVEqual DAG; only a source-entry/first-token cell may augment
    # that DAG at its published scope.
    lifecycle_exact = (
        (ComputeExactRequestDemand(
            _CANN_KV_EQUAL_SINGLE_TBUF_LIFECYCLE_EXACT_SERVICE,
            ((tensor_bytes_per_token,),)),)
        if (tokens == worker_count
            and worker_count
            in _CANN_KV_EQUAL_SINGLE_TBUF_LIFECYCLE_WORKERS
            and tensor_bytes_per_token in (1024, 2048))
        else ())
    # The older plain-DataCopy completion cells remain valid only for an
    # unloaded single AIV and only when no exact full-chain cell exists.
    cold_chain_exact = _cann_equal_kv_chain_completion_request(
        hardware, tensor_bytes_per_token, worker_count,
        cold_source_entry=True,
        first_token=True,
        one_token_per_worker=maximum_repeats == 1)
    warm_chain_exact = _cann_equal_kv_chain_completion_request(
        hardware, tensor_bytes_per_token, worker_count,
        cold_source_entry=False,
        first_token=True,
        one_token_per_worker=maximum_repeats == 1)
    has_any_chain_exact = bool(cold_chain_exact or warm_chain_exact)
    all_workers_have_chain_exact = bool(
        cold_chain_exact
        and (worker_count == 1 or warm_chain_exact))
    mte2_exact = (
        _cann_memory_completion_request(
            hardware, "mte2", _CANN_MTE2_COMPLETION_EXACT_SERVICE,
            tensor_bytes_per_token)
        if worker_count == 1 and not has_any_chain_exact else ())
    mte3_exact = (
        _cann_memory_completion_request(
            hardware, "mte3", _CANN_MTE3_COMPLETION_EXACT_SERVICE,
            tensor_bytes_per_token)
        if worker_count == 1 and not has_any_chain_exact else ())

    def stages(token_count, *, cold_source_entry, first_token):
        chain_exact = _cann_equal_kv_chain_completion_request(
            hardware, tensor_bytes_per_token, worker_count,
            cold_source_entry=cold_source_entry,
            first_token=first_token,
            one_token_per_worker=maximum_repeats == 1)
        lifecycle_stages = (
            (PipelineStageDemand(
                "buffer_lifecycle", "compute:scalar_aiv", None,
                compute_exact_request_demands=lifecycle_exact),)
            if lifecycle_exact else ())
        slot_predecessors = (
            ("buffer_lifecycle",) if lifecycle_exact else None)
        component_stages = (
            PipelineStageDemand(
                "slot_get", "compute:compute_vector", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_SCALAR_GET_VALUE_EXACT_SERVICE,
                        ((1,),)),),
                predecessors=slot_predecessors),
            PipelineStageDemand(
                "key_load", "memory:mte2_read",
                token_count * (tensor_bytes_per_token + slot_bytes_per_token)
                / read_bytes,
                memory_exact_request_demands=mte2_exact,
                predecessors=("slot_get",)),
            PipelineStageDemand(
                "key_store", "memory:mte3_write",
                token_count * tensor_bytes_per_token / write_bytes,
                memory_exact_request_demands=mte3_exact,
                predecessors=("key_load",)),
            PipelineStageDemand(
                "value_load", "memory:mte2_read",
                token_count * tensor_bytes_per_token / read_bytes,
                memory_exact_request_demands=mte2_exact,
                predecessors=("key_store",)),
        )
        return lifecycle_stages + component_stages + (
            PipelineStageDemand(
                "value_store", "memory:mte3_write",
                token_count * tensor_bytes_per_token / write_bytes,
                memory_exact_request_demands=(
                    chain_exact if chain_exact else mte3_exact),
                predecessors=("value_load",)),
        )

    def worker_tasks(name, worker, token_count, *, cold_source_entry):
        tasks = [PipelineTaskDemand(
            name=f"{name}_first_token",
            worker=worker,
            stages=stages(
                1, cold_source_entry=cold_source_entry,
                first_token=True),
        )]
        if token_count > 1:
            tasks.append(PipelineTaskDemand(
                name=f"{name}_steady_tokens",
                worker=worker,
                stages=stages(
                    1, cold_source_entry=False, first_token=False),
                repeat_count=token_count - 1,
            ))
        return tuple(tasks)

    if analytical_compact:
        worker_classes = []
        if remainder:
            worker_classes.append((
                tuple(range(remainder)), base_repeats + 1,
            ))
        if remainder < worker_count:
            worker_classes.append((
                tuple(range(remainder, worker_count)), base_repeats,
            ))
        source_entry_classes = []
        for physical_workers, token_count in worker_classes:
            if physical_workers[0] == 0:
                source_entry_classes.append(((0,), token_count, True))
                physical_workers = physical_workers[1:]
            if physical_workers:
                source_entry_classes.append(
                    (physical_workers, token_count, False))
        tasks = tuple(
            PipelineTaskProgramDemand(
                name=f"worker_class{compact_worker}_tokens",
                worker=compact_worker,
                patterns=(PipelineTaskSequenceDemand(
                    name=f"worker_class{compact_worker}_sequence",
                    worker=compact_worker,
                    tasks=worker_tasks(
                        f"worker_class{compact_worker}", compact_worker,
                        token_count,
                        cold_source_entry=cold_source_entry),
                ),),
                order=(0,),
                worker_multiplicity=len(physical_workers),
                physical_workers=physical_workers,
            )
            for compact_worker, (
                physical_workers, token_count, cold_source_entry,
            ) in enumerate(source_entry_classes)
        )
    else:
        task_entries = []
        for worker in range(worker_count):
            token_count = base_repeats + (1 if worker < remainder else 0)
            sequence = worker_tasks(
                f"worker{worker}", worker, token_count,
                cold_source_entry=worker == 0)
            if len(sequence) == 1:
                task_entries.append(sequence[0])
            else:
                task_entries.append(PipelineTaskSequenceDemand(
                    name=f"worker{worker}_tokens",
                    worker=worker,
                    tasks=sequence,
                ))
        tasks = tuple(task_entries)
    if lifecycle_exact:
        inp = _attach_scalar_aiv_exact_demands(inp, (
            ComputeExactRequestDemand(
                _CANN_KV_EQUAL_SINGLE_TBUF_LIFECYCLE_EXACT_SERVICE,
                ((tensor_bytes_per_token,),) * worker_count),
        ))
    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tasks,
        buffers=(PipelineBuffer(
            name="token_buffer",
            producer_stage="key_load",
            consumer_stage="value_store",
            slots=1,
            scope="worker",
        ),),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    if not lifecycle_exact:
        attached = dataclasses.replace(
            attached,
            limitations=attached.limitations + (
                "cann_kv_single_tbuf_lifecycle_evidence_pending",))
    if not all_workers_have_chain_exact and worker_count != 1:
        attached = dataclasses.replace(
            attached,
            limitations=attached.limitations + (
                "cann_kv_concurrent_hard_event_service_evidence_pending",))
    elif not all_workers_have_chain_exact and (
            not mte2_exact or not mte3_exact):
        attached = dataclasses.replace(
            attached,
            limitations=attached.limitations + (
                "cann_kv_exact_mte_completion_service_evidence_pending",))
    return attached


def _attach_cann_unequal_kv_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False,
        ) -> PredictionInput:
    """Attach the source-proven CANN key-202 two-queue K/V pipeline.

    CANN 8.5.1 ``ProcessKVUnequal`` assigns a contiguous token interval to
    each AIV.  K and V use independent two-slot ``TQueBind`` queues.  For each
    queue, a chunk's MTE3 store waits for its MTE2 load and queue-slot reuse
    waits for that store; the source adds no K-to-V or token-to-token
    completion edge.  Both paths read the slot mapping independently.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN ReshapeAndCache pipeline requires separated npu_tile AIVs")
    if kernel.variant.lower() not in (
            "kv_cache", "kvcache", "paged_kv_cache"):
        raise ValueError(
            "CANN ReshapeAndCache pipeline requires a KV-cache kernel")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN ReshapeAndCache pipeline must be attached exactly once")
    if not isinstance(analytical_compact, bool):
        raise TypeError("CANN unequal KV analytical_compact must be boolean")

    tokens = _shape(kernel, "seq_len")
    num_kv_heads = _shape(kernel, "num_kv_heads")
    head_dim = _shape(kernel, "head_dim")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    tensor_bytes_per_token = num_kv_heads * head_dim * dtype_bytes
    slot_bytes_per_path = 4
    read_bytes = float(
        tokens * (2 * tensor_bytes_per_token + 2 * slot_bytes_per_path))
    write_bytes = float(tokens * 2 * tensor_bytes_per_token)
    if not math.isclose(
            inp.demand_per_resource["l2_cache"], read_bytes + write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "CANN fallback ReshapeAndCache L2 demand disagrees with its ABI")
    if not math.isclose(
            inp.demand_per_resource["mte3"], write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "CANN fallback ReshapeAndCache write demand disagrees with its ABI")

    (worker_count, maximum_repeats, base_repeats,
     remainder) = _cann_reshape_cache_worker_repeats(tokens, inp, kernel)
    repeated_pair_exact = _cann_unequal_kv_repeated_pair_completion_request(
        hardware, tensor_bytes_per_token, worker_count, maximum_repeats)
    # The published queue-control primitive covers a complete
    # AllocTensor/EnQue/DeQue/FreeTensor lifetime and includes a 48-worker
    # cell.  Bind exactly two such source-counted lifetimes (K and V) per
    # token only to the canonical key-202 48-AIV domain.  Small unaligned
    # fallback launches remain explicitly partial instead of inheriting this
    # worker geometry.
    queue_control_exact = worker_count == 48
    queue_control_demands = (
        (ComputeExactRequestDemand(
            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
            ((1,),)),)
        if queue_control_exact else ())
    mte2_completion = _cann_memory_completion_request(
        hardware, "mte2", _CANN_MTE2_COMPLETION_EXACT_SERVICE,
        tensor_bytes_per_token)
    mte3_completion = _cann_memory_completion_request(
        hardware, "mte3", _CANN_MTE3_COMPLETION_EXACT_SERVICE,
        tensor_bytes_per_token)
    mte2_completion = tuple(
        dataclasses.replace(demand, pipelined_completion=True)
        for demand in mte2_completion)
    mte3_completion = tuple(
        dataclasses.replace(demand, pipelined_completion=True)
        for demand in mte3_completion)
    def stages(token_count):
        if repeated_pair_exact:
            return (
                PipelineStageDemand(
                    "slot_get_sequence", "compute:compute_vector", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_SCALAR_GET_VALUE_EXACT_SERVICE,
                            ((1,) * (2 * token_count),)),)),
                PipelineStageDemand(
                    "unequal_kv_queue_pair_sequence", "memory:mte3_write",
                    token_count * 2 * tensor_bytes_per_token / write_bytes,
                    memory_exact_request_demands=(
                        repeated_pair_exact
                        if token_count == maximum_repeats else ()),
                    predecessors=("slot_get_sequence",)),
            )
        if not queue_control_exact:
            return (
                PipelineStageDemand(
                    "key_slot_get", "compute:compute_vector", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_SCALAR_GET_VALUE_EXACT_SERVICE,
                            ((1,),)),)),
                PipelineStageDemand(
                    "key_load", "memory:mte2_read",
                    (tensor_bytes_per_token + slot_bytes_per_path)
                    / read_bytes,
                    memory_exact_request_demands=mte2_completion,
                    predecessors=("key_slot_get",)),
                PipelineStageDemand(
                    "key_store", "memory:mte3_write",
                    tensor_bytes_per_token / write_bytes,
                    memory_exact_request_demands=mte3_completion,
                    predecessors=("key_load",)),
                PipelineStageDemand(
                    "value_slot_get", "compute:compute_vector", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_SCALAR_GET_VALUE_EXACT_SERVICE,
                            ((1,),)),),
                    predecessors=()),
                PipelineStageDemand(
                    "value_load", "memory:mte2_read",
                    (tensor_bytes_per_token + slot_bytes_per_path)
                    / read_bytes,
                    memory_exact_request_demands=mte2_completion,
                    predecessors=("value_slot_get",)),
                PipelineStageDemand(
                    "value_store", "memory:mte3_write",
                    tensor_bytes_per_token / write_bytes,
                    memory_exact_request_demands=mte3_completion,
                    predecessors=("value_load",)),
            )
        return (
            PipelineStageDemand(
                "key_slot_get", "compute:compute_vector", None,
                compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_SCALAR_GET_VALUE_EXACT_SERVICE, ((1,),)),)),
            PipelineStageDemand(
                "key_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=queue_control_demands,
                predecessors=("key_slot_get",)),
            PipelineStageDemand(
                "key_load", "memory:mte2_read",
                (tensor_bytes_per_token + slot_bytes_per_path)
                / read_bytes,
                memory_exact_request_demands=mte2_completion,
                predecessors=("key_queue_control",)),
            PipelineStageDemand(
                "key_store", "memory:mte3_write",
                tensor_bytes_per_token / write_bytes,
                memory_exact_request_demands=mte3_completion,
                predecessors=("key_load",)),
            PipelineStageDemand(
                "value_slot_get", "compute:compute_vector", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_SCALAR_GET_VALUE_EXACT_SERVICE, ((1,),)),),
                predecessors=()),
            PipelineStageDemand(
                "value_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=queue_control_demands,
                predecessors=("value_slot_get",)),
            PipelineStageDemand(
                "value_load", "memory:mte2_read",
                (tensor_bytes_per_token + slot_bytes_per_path)
                / read_bytes,
                memory_exact_request_demands=mte2_completion,
                predecessors=("value_queue_control",)),
            PipelineStageDemand(
                "value_store", "memory:mte3_write",
                tensor_bytes_per_token / write_bytes,
                memory_exact_request_demands=mte3_completion,
                predecessors=("value_load",)),
        )

    if analytical_compact:
        worker_classes = ()
        if remainder:
            worker_classes += ((
                tuple(range(remainder)), base_repeats + 1,
            ),)
        if remainder < worker_count:
            worker_classes += ((
                tuple(range(remainder, worker_count)), base_repeats,
            ),)
        tasks = tuple(
            PipelineTaskProgramDemand(
                name=f"worker_class{compact_worker}_tokens",
                worker=compact_worker,
                patterns=(PipelineTaskSequenceDemand(
                    name=f"worker_class{compact_worker}_sequence",
                    worker=compact_worker,
                    tasks=(PipelineTaskDemand(
                        name=f"worker_class{compact_worker}_tokens",
                        worker=compact_worker,
                        stages=stages(token_count),
                        repeat_count=(
                            1 if repeated_pair_exact else token_count),
                    ),),
                ),),
                order=(0,),
                worker_multiplicity=len(physical_workers),
                physical_workers=physical_workers,
            )
            for compact_worker, (physical_workers, token_count) in enumerate(
                worker_classes)
        )
    else:
        tasks = tuple(
            PipelineTaskDemand(
                name=f"worker{worker}_tokens",
                worker=worker,
                stages=stages(
                    base_repeats + (1 if worker < remainder else 0)),
                repeat_count=(
                    1 if repeated_pair_exact else
                    base_repeats + (1 if worker < remainder else 0)),
            )
            for worker in range(worker_count)
        )
    if queue_control_exact and not repeated_pair_exact:
        queue_requests = tuple(
            (1,) * (2 * (
                base_repeats + (1 if worker < remainder else 0)))
            for worker in range(worker_count))
        inp = _attach_scalar_aiv_exact_demands(inp, (
            ComputeExactRequestDemand(
                _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                queue_requests),))
    elif not queue_control_exact:
        inp = dataclasses.replace(
            inp,
            limitations=inp.limitations + (
                "cann_kv_queue_control_service_evidence_pending",))

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tasks,
        buffers=() if repeated_pair_exact else (
            PipelineBuffer(
                name="key_queue",
                producer_stage="key_load",
                consumer_stage="key_store",
                slots=2,
                scope="worker",
            ),
            PipelineBuffer(
                name="value_queue",
                producer_stage="value_load",
                consumer_stage="value_store",
                slots=2,
                scope="worker",
            ),
        ),
    )
    return attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)


def _attach_cann_swiglu_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware
        ) -> PredictionInput:
    """Attach the source-proven CANN 8.5.1 BF16 SwiGLU tile program.

    Key 27 assigns row-major tiles by ``tile = worker + k * blockDim``.
    Every tile uses independent A-input, B-input, and output queues whose
    one/two-slot depth comes from the reflected serialized tiling payload.
    The typed Vector stages conserve the existing aggregate generic/Exp/Div
    demand while exposing the source order and drain to the common worker
    scheduler.  The two source-level typed tiling materializations are task
    prologues, not four per-worker GlobalTensor::GetValue requests.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN SwiGLU pipeline requires separated npu_tile AIVs")
    if kernel.variant.lower() not in (
            "activation", "silu", "moe_swiglu"):
        raise ValueError(
            "CANN SwiGLU pipeline requires activation/silu or moe_swiglu")
    if inp.pipeline_plan is not None:
        raise ValueError("CANN SwiGLU pipeline must be attached exactly once")

    parameters = kernel.launch.parameters
    missing = tuple(
        name for name in _TYPED_SWIGLU_PIPELINE_PARAMETERS
        if name not in parameters)
    if missing:
        raise ValueError(
            f"typed CANN SwiGLU pipeline parameters are incomplete: "
            f"{missing}")
    rows, columns, elements = _cann_swiglu_logical_shape(kernel)
    base_rows = int(parameters["swiglu_base_row_len"])
    base_columns = int(parameters["swiglu_base_col_len"])
    total_tiles = int(parameters["swiglu_total_tiles"])
    queue_depth = int(parameters["swiglu_queue_depth"])
    if base_rows <= 0 or base_columns <= 0:
        raise ValueError("typed CANN SwiGLU base tile must be positive")
    expected_tiles = math.ceil(rows / base_rows) * math.ceil(
        columns / base_columns)
    if total_tiles != expected_tiles:
        raise ValueError(
            "typed CANN SwiGLU total_tiles disagrees with its base tile")
    if queue_depth not in (1, 2):
        raise ValueError(
            "typed CANN SwiGLU queue depth must be source-defined as 1 or 2")
    worker_count = kernel.launch.grid_size
    if worker_count <= 0 or worker_count > total_tiles:
        raise ValueError(
            "typed CANN SwiGLU worker count must cover at least one tile per "
            "launched AIV")
    tile_count_by_worker = tuple(
        (total_tiles - worker + worker_count - 1) // worker_count
        for worker in range(worker_count))
    terminal_store_completion = _cann_unloaded_memory_completion_request(
        hardware, "mte3", _CANN_MTE3_COMPLETION_EXACT_SERVICE,
        32, worker_count)
    read_bytes = float(2 * elements * _shape(kernel, "dtype_bytes"))
    write_bytes = float(elements * _shape(kernel, "dtype_bytes"))
    if not math.isclose(
            inp.demand_per_resource["l2_cache"], read_bytes + write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "typed CANN SwiGLU pipeline L2 demand disagrees with its ABI")
    if not math.isclose(
            inp.demand_per_resource["mte3"], write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "typed CANN SwiGLU pipeline write demand disagrees with its ABI")

    column_tiles = math.ceil(columns / base_columns)
    # The independently measured pair ends at the first two DeQue operations.
    # Before a worker has executed a tile, that source-entry boundary has no
    # preceding Vector or MTE3 work, so it remains the same edge for a finite
    # program and the first tile of a repeated program.  Later tiles have the
    # preceding CopyIn -> Compute -> CopyOut loop state and retain only the
    # generic source DAG until a matching steady mechanism exists.
    input_pair_service = _cann_swiglu_input_pair_service(
        hardware, worker_count, queue_depth)

    def _tile_shape(tile_index):
        row_tile, column_tile = divmod(tile_index, column_tiles)
        row_start = row_tile * base_rows
        column_start = column_tile * base_columns
        tile_rows = min(base_rows, rows - row_start)
        tile_columns = min(base_columns, columns - column_start)
        if tile_rows <= 0 or tile_columns <= 0:
            raise ValueError(
                f"invalid typed CANN SwiGLU tile index: {tile_index}")
        return tile_rows, tile_columns

    # Ascend910B has 24 AIC groups with two AIVs per group.  Key-27 assigns
    # its first one-tile workers across groups before it assigns a second AIV
    # lane, so CopyIn uses a shared group ingress when no versioned source
    # entry service describes that population. The group scheduler serializes
    # only second-lane requests; traffic and source order stay unchanged.
    aiv_grouped_ingress = (
        total_tiles == worker_count
        and queue_depth == 2
        and input_pair_service is None
        and worker_count <= hardware.resource_by_role("mte2").total_units)

    def _tile_task(
            worker, tile_rows, tile_columns, include_lifecycle,
            include_terminal_completion):
        tile_elements = tile_rows * tile_columns
        tile_payload_bytes = tile_elements * _shape(kernel, "dtype_bytes")
        read_fraction = tile_elements / (2.0 * elements)
        write_fraction = tile_elements / float(elements)
        # The exact pair probe is an entry-only source edge. It replaces the
        # first tile's two request completion only; it is never replayed for
        # a later ``tile = worker + k * blockDim`` iteration.
        input_pair_completion = (
            _cann_memory_completion_request(
                hardware, "mte2", input_pair_service, tile_payload_bytes)
            if include_lifecycle and input_pair_service is not None else ())
        store_completion = ()
        if not store_completion and include_terminal_completion:
            store_completion = terminal_store_completion
        generic = float(tile_elements)
        typed_service = "compute:compute_vector"
        cast_a_demands = _cann_swiglu_service_demands(
            hardware,
            (("cast_bf16_to_fp32_serialized", generic, 1),))
        activation_demands = _cann_swiglu_service_demands(
            hardware,
            (
                ("muls_serialized", generic, 1),
                ("exp", generic, 1),
                ("adds_serialized", generic, 1),
                ("div", generic, 1),
            ))
        cast_b_demands = cast_a_demands
        multiply_demands = _cann_swiglu_service_demands(
            hardware, (("mul_serialized", generic, 1),))
        cast_out_demands = _cann_swiglu_service_demands(
            hardware,
            (("cast_fp32_to_bf16_serialized", generic, 1),))
        index_control_requests = 3 if include_lifecycle else 1
        # ``SWIGLU_SINGLE_PROCESS`` invokes CopyIn, Compute, and CopyOut in
        # one loop body before it advances ``tileIdx``.  The next iteration
        # cannot enter its scalar tiling path until the preceding CopyOut has
        # released the worker's MTE3 path.  The three TQue instances permit
        # the source-defined in-tile handoffs; they do not authorize a later
        # loop iteration to issue before this enclosing CopyOut.  Carry the
        # loop edge through the worker-local MTE3 resource rather than
        # inventing a SwiGLU-specific duration or a shape regime.
        index_control_stage = PipelineStageDemand(
            "tiling_index_control", "compute:scalar_aiv", None,
            compute_exact_request_demands=(
                ComputeExactRequestDemand(
                    _CANN_SCALAR_U64_DIVREM_EXACT_SERVICE,
                    ((1,) * index_control_requests,)),
                # CaclOneTileParam first forms row/column quotient and
                # remainder, then CaclOneTileOffsetParam executes the five
                # uint64 multiply/three add GM-offset DAG and chooses its
                # base/tail copy parameters.  The latter is one dynamic
                # source control group per tile, including the first and
                # terminal tile; it is not part of the two one-time tiling
                # materializations or the preceding div/rem group count.
                ComputeExactRequestDemand(
                    _CANN_SWIGLU_TILE_ADDRESS_EXACT_SERVICE, ((1,),)),
            ),
            predecessors=(),
            prior_resources=("memory:mte3_write",))
        control_stages = (index_control_stage,) + ((PipelineStageDemand(
            "buffer_lifecycle", "compute:scalar_aiv", None,
            compute_exact_request_demands=(ComputeExactRequestDemand(
                _CANN_AIV_BUFFER_LIFECYCLE_EXACT_SERVICE,
                ((queue_depth,),)),),
            predecessors=("tiling_index_control",)),)
            if include_lifecycle else ())
        queue_predecessors = (
            ("buffer_lifecycle",)
            if include_lifecycle else ("tiling_index_control",))
        queue_control_stage = PipelineStageDemand(
            "queue_control", "compute:scalar_aiv", None,
            compute_exact_request_demands=(ComputeExactRequestDemand(
                _CANN_AIV_QUEUE_CONTROL_EXACT_SERVICE,
                ((queue_depth,),)),),
            predecessors=queue_predecessors)
        load_stages = (queue_control_stage, PipelineStageDemand(
            "load_input_pair", "memory:mte2_read", 2.0 * read_fraction,
            memory_exact_request_demands=input_pair_completion,
            aiv_ingress_group_size=(
                _CANN_SWIGLU_AIV_INGRESS_GROUP_SIZE
                if aiv_grouped_ingress else None),
            predecessors=("queue_control",)))
        stages = control_stages + load_stages + (
            PipelineStageDemand(
                "cast_a", typed_service, None,
                compute_service_demands=cast_a_demands,
                predecessors=("load_input_pair",)),
            PipelineStageDemand(
                "activation", typed_service, None,
                compute_service_demands=activation_demands),
            PipelineStageDemand(
                "cast_b", typed_service, None,
                compute_service_demands=cast_b_demands,
                predecessors=("activation",)),
            PipelineStageDemand(
                "multiply", typed_service, None,
                compute_service_demands=multiply_demands),
            PipelineStageDemand(
                "cast_out", typed_service, None,
                compute_service_demands=cast_out_demands),
            PipelineStageDemand(
                "store", "memory:mte3_write", write_fraction,
                memory_exact_request_demands=store_completion),
        )
        return PipelineTaskDemand(
            name=f"tile_r{tile_rows}_c{tile_columns}",
            worker=worker,
            stages=tuple(stages),
        )

    task_templates = {}

    def _task_for_shape(
            worker, tile_shape, include_lifecycle,
            include_terminal_completion, suffix):
        tile_rows, tile_columns = tile_shape
        pattern_key = (
            tile_rows, tile_columns, include_lifecycle,
            include_terminal_completion)
        if pattern_key not in task_templates:
            task_templates[pattern_key] = _tile_task(
                0, tile_rows, tile_columns,
                include_lifecycle, include_terminal_completion)
        return dataclasses.replace(
            task_templates[pattern_key],
            name=(f"tile_r{tile_rows}_c{tile_columns}_{suffix}"),
            worker=worker,
        )

    def _append_body_segment_layout(
            layout, start_tile, count, segment_index):
        if count == 0:
            return
        column_period = column_tiles // math.gcd(worker_count, column_tiles)
        full_cycles, tail_count = divmod(count, column_period)
        if full_cycles:
            cycle_shapes = tuple(
                _tile_shape(start_tile + position * worker_count)
                for position in range(column_period))
            layout.append((
                f"body{segment_index}_cycle", full_cycles,
                cycle_shapes, False, False))
        if tail_count:
            tail_start = start_tile + full_cycles * column_period * worker_count
            tail_shapes = tuple(
                _tile_shape(tail_start + position * worker_count)
                for position in range(tail_count))
            layout.append((
                f"body{segment_index}_tail", 1,
                tail_shapes, False, False))

    def _worker_layout(worker):
        tile_count = tile_count_by_worker[worker]
        if tile_count <= 0:
            raise AssertionError("CANN SwiGLU launched an idle AIV")
        layout = []
        first_terminal = tile_count == 1
        layout.append((
            "first", 1, (_tile_shape(worker),), True, first_terminal))
        body_count = tile_count - 2
        if body_count > 0:
            body_start = worker + worker_count
            final_row_start = (
                (math.ceil(rows / base_rows) - 1) * column_tiles)
            first_segment_count = min(
                body_count,
                max(0, (final_row_start - body_start + worker_count - 1)
                    // worker_count),
            )
            _append_body_segment_layout(
                layout, body_start, first_segment_count, 0)
            _append_body_segment_layout(
                layout,
                body_start + first_segment_count * worker_count,
                body_count - first_segment_count, 1)
        if tile_count > 1:
            terminal_index = worker + (tile_count - 1) * worker_count
            layout.append((
                "terminal", 1, (_tile_shape(terminal_index),), False, True))
        return tuple(layout)

    program_groups = {}
    for physical_worker in range(worker_count):
        layout = _worker_layout(physical_worker)
        if layout in program_groups:
            program_groups[layout].append(physical_worker)
        else:
            program_groups[layout] = [physical_worker]
    if aiv_grouped_ingress:
        # One tile per worker is a finite source program.  Keep its at-most-48
        # AIV tasks explicit so the common scheduler can represent the two
        # lanes sharing each AIC group's ingress.
        pipeline_tasks = tuple(
            _task_for_shape(
                worker, _tile_shape(worker), True, True,
                f"worker{worker}")
            for worker in range(worker_count))
    else:
        compressed_programs = []
        for worker, (layout, physical_workers) in enumerate(
                program_groups.items()):
            patterns = tuple(PipelineTaskSequenceDemand(
                name=name,
                worker=worker,
                tasks=tuple(_task_for_shape(
                    worker, tile_shape, include_lifecycle,
                    include_terminal_completion,
                    f"{name}_{task_index}")
                    for task_index, tile_shape in enumerate(tile_shapes)),
                repeat_count=repeat_count,
            ) for (
                name, repeat_count, tile_shapes,
                include_lifecycle, include_terminal_completion) in layout)
            compressed_programs.append(PipelineTaskProgramDemand(
                name=f"worker_class{worker}_tiles",
                worker=worker,
                patterns=patterns,
                order=tuple(range(len(patterns))),
                worker_multiplicity=len(physical_workers),
                physical_workers=tuple(physical_workers),
            ))
        pipeline_tasks = tuple(compressed_programs)

    lifecycle_requests = tuple(
        (queue_depth,) for _worker in range(worker_count))
    queue_request_runs = tuple(
        ((queue_depth, tile_count),)
        for tile_count in tile_count_by_worker)
    divrem_request_runs = tuple(
        ((1, 2 + tile_count),)
        for tile_count in tile_count_by_worker)
    tile_address_request_runs = tuple(
        ((1, tile_count),)
        for tile_count in tile_count_by_worker)
    scalar_exact_demands = (
        ComputeExactRequestDemand(
            _CANN_AIV_BUFFER_LIFECYCLE_EXACT_SERVICE,
            lifecycle_requests),
        ComputeExactRequestDemand(
            _CANN_AIV_QUEUE_CONTROL_EXACT_SERVICE,
            (), request_size_runs_by_worker=queue_request_runs),
        ComputeExactRequestDemand(
            _CANN_SCALAR_U64_DIVREM_EXACT_SERVICE,
            (), request_size_runs_by_worker=divrem_request_runs),
        ComputeExactRequestDemand(
            _CANN_SWIGLU_TILE_ADDRESS_EXACT_SERVICE,
            (), request_size_runs_by_worker=tile_address_request_runs),
    )
    inp = _attach_scalar_aiv_exact_demands(inp, scalar_exact_demands)

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=pipeline_tasks,
        buffers=(
            PipelineBuffer(
                name="input_a_queue",
                producer_stage="load_input_pair",
                consumer_stage="cast_a",
                slots=queue_depth,
                scope="worker",
            ),
            PipelineBuffer(
                name="input_b_queue",
                producer_stage="load_input_pair",
                consumer_stage="cast_b",
                slots=queue_depth,
                scope="worker",
            ),
            PipelineBuffer(
                name="output_c_queue",
                # ``Compute`` reserves its VECOUT tensor before its first
                # Cast(A), then enqueues it only after CastOut.  The finite
                # queue credit is therefore owned from the first Compute
                # stage through CopyOut/FreeTensor, not merely from EnQue.
                # Starting this lifetime at ``cast_out`` would let a later
                # tile execute its Vector body despite both source VECOUT
                # slots being allocated.  ``cast_a`` is the first modeled
                # stage after that source allocation, so it is the exact
                # credit-acquisition boundary in the typed DAG.
                producer_stage="cast_a",
                consumer_stage="store",
                slots=queue_depth,
                scope="worker",
            ),
        ),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    # The source assigns every launched AIV a nonempty program, including
    # uneven terminal worker classes.  Its body is now represented explicitly
    # by the typed source worker DAG, so the runtime contributes only the
    # independently measured empty AIV task shell.  Using a complete nonempty
    # task as a competing floor would instead hide an unmodelled operator body
    # inside a generic task duration.  SYS_CNT supplies one launch-wide
    # worker-entry fill; it composes with the longest source worker body and
    # the empty shell.  This is valid for the source's finite and repeated
    # programs alike: the fill is an admission envelope, never a per-tile or
    # per-worker-order service.  The typed AIC-group ingress above remains
    # inside the worker DAG.
    return dataclasses.replace(
        attached,
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
        limitations=attached.limitations + (
            "cann_swiglu_multiworker_queue_completion_not_local",
        ),
    )


def _attach_scalar_aiv_exact_demands(
        inp: PredictionInput,
        exact_demands: Tuple[ComputeExactRequestDemand, ...]
        ) -> PredictionInput:
    """Attach source-counted control work to the physical AIV scalar pipe."""

    if not exact_demands:
        raise ValueError("scalar AIV exact demands must not be empty")
    scalar_control_operations = math.fsum(
        demand.operations for demand in exact_demands)
    demand_per_resource = dict(inp.demand_per_resource)
    if demand_per_resource["scalar_aiv"] != 0.0:
        raise ValueError("scalar AIV demand must start at zero")
    demand_per_resource["scalar_aiv"] = scalar_control_operations
    return dataclasses.replace(
        inp,
        demand_per_resource=demand_per_resource,
        compute_demands=inp.compute_demands + (ComputeDemand(
            resource_role="scalar_aiv",
            operations=scalar_control_operations,
            shape_efficiency=1.0,
            schedule_efficiency=1.0,
            exact_request_demands=exact_demands,
        ),),
    )


def _attach_cann_elementwise_serialized_completion(
        inp: PredictionInput, kernel: KernelSpec,
        hardware) -> PredictionInput:
    """Bind the routing-mask compare/cast work to its serialized service.

    The independent microbench campaign measured the per-launch completion of
    back-to-back aclnnNeScalar/aten-cast chains as one affine law: a fixed
    setup term plus a steady chip-level serialized completion-stream rate.
    The steady rate does NOT parallelize across the 48 AIV workers, so the
    complete element count is assigned to the critical worker as a
    single-owner stream; the remaining workers hold zero work and zero
    invocations, which keeps the grid-invariance of the law (the demand
    duration is the worker maximum, not a grid-scaled aggregate).

    Provenance: microbench/ascend-910b references directory,
    elementwise_compare_cast_serialized_service_cann_8_5_1.json
    (calibration run1+run2, validation run3, no production or evaluation
    duration consumed).
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN elementwise compare/cast services require separated "
            "npu_tile AIVs")
    if (kernel.variant.lower() != "elementwise"
            or kernel.dtype != "bf16"):
        raise ValueError(
            "CANN elementwise compare/cast requires a BF16 elementwise "
            "kernel")
    operation = kernel.attributes["operation"]
    if operation not in _CANN_ELEMENTWISE_COMPARE_CAST_SERVICES:
        raise ValueError(
            "CANN elementwise serialized completion requires a compare or "
            f"cast operation, got {operation!r}")
    service = _CANN_ELEMENTWISE_COMPARE_CAST_SERVICES[operation]
    if not _has_compute_service(hardware, "compute_vector", service):
        raise ValueError(
            f"CANN elementwise compare/cast service is unavailable: "
            f"{service!r}")
    elements = _shape(kernel, "element_count")
    vector_resource = hardware.resource_by_role("compute_vector")
    execution_domain_units = hardware.execution_units_for_task(
        inp.launch_task_kind)
    worker_count = (
        vector_resource.total_units
        * min(inp.active_units, execution_domain_units)
        // execution_domain_units)
    if worker_count != inp.active_units:
        raise ValueError(
            "CANN elementwise compare/cast worker count disagrees with "
            "active units")
    return _replace_vector_service_demands(
        inp, (ComputeServiceDemand(
            service,
            float(elements),
            serial_invocations=1,
            operations_by_worker=(float(elements),) + (0.0,) * (
                worker_count - 1),
            serial_invocations_by_worker=(1,) + (0,) * (worker_count - 1),
        ),))


def _attach_cann_elementwise_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False,
        ) -> PredictionInput:
    """Attach the CANN 8.5.1 BF16 one-dimensional Add/Mul tile DAG.

    ``BroadcastOneDimSch`` assigns one contiguous ``blockFactor`` interval to
    each AIV and walks it in ``ubFactor`` tiles.  The AddWithCastCompute and
    MulXfp16Op DAGs both issue two GM-to-UB copies, two BF16-to-FP32 casts,
    one arithmetic node, one FP32-to-BF16 cast, and one UB-to-GM copy per
    tile.  The launch package supplies the source-derived ownership and
    finite-buffer depths; this function adds only that ordering and conserves
    the existing aggregate traffic and Vector work.  In particular, source
    ``Run(..., tile_index & 1)`` gives each BF16 input stream two physical
    ping/pong slots.  It is not the Mul q4/q1 four-tile admission condition
    used by the separately-published steady GM service.

    Source identity (installed CANN 8.5.1, Ascend910B): add_apt.cpp
    f28d5649f956222d67a2381d1b241e739c73171544fa39d12cb4514e674f80c9,
    mul_apt.cpp 682455b1a1f77fdb83ddebfa259d1aab1efe17cecd84b56692864e723ffbbd02,
    add_dag.h 478d891f6cf237cf383853555b7681e549c56cce272d17f21c56b5ad45b79fdf,
    mul_dag.h 20335a70953c6a9c62088762f69e7fa74c4acf2e751cf62874dd0294424cbf,
    and broadcast_sch_one_dim.h
    4eebc1471b440575e3017a4be3d2e4f8b1c70c6764c7aa5f25a9573c687f3e67.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN elementwise pipeline requires separated npu_tile AIVs")
    if (kernel.variant.lower() != "elementwise"
            or kernel.dtype != "bf16"):
        raise ValueError(
            "CANN elementwise pipeline requires a BF16 elementwise kernel")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN elementwise pipeline must be attached exactly once")
    if not isinstance(analytical_compact, bool):
        raise TypeError("CANN elementwise analytical_compact must be boolean")

    parameters = kernel.launch.parameters
    required = (
        "cann_tiling_key",
        "cann_tiling_payload_bytes",
        "elementwise_block_factor",
        "elementwise_ub_factor",
        "elementwise_input_queue_depth",
        "elementwise_input_ping_pong_slots",
        "elementwise_output_queue_depth",
        SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER,
    )
    missing = tuple(name for name in required if name not in parameters)
    if missing:
        raise ValueError(
            f"CANN elementwise source parameters are incomplete: {missing}")
    if parameters[SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER] != 1.0:
        raise ValueError(
            "CANN elementwise source DAG marker must equal one")
    if parameters["cann_tiling_payload_bytes"] != 56.0:
        raise ValueError(
            "CANN elementwise source program requires its fixed 56-byte "
            "BroadcastOneDim tiling payload")

    operation = kernel.attributes["operation"]
    tiling_key = int(parameters["cann_tiling_key"])
    expected_keys = {
        "add": (210_000_000, 210_010_000),
        "mul": (210_000_000, 210_020_000),
    }
    if operation not in expected_keys:
        raise ValueError(
            "CANN elementwise pipeline supports only Add and Mul")
    if tiling_key not in expected_keys[operation]:
        raise ValueError(
            "CANN elementwise tiling key disagrees with its operation")

    elements = _shape(kernel, "element_count")
    block_factor = int(parameters["elementwise_block_factor"])
    ub_factor = int(parameters["elementwise_ub_factor"])
    input_depth = int(parameters["elementwise_input_queue_depth"])
    input_ping_pong_slots = int(
        parameters["elementwise_input_ping_pong_slots"])
    output_depth = int(parameters["elementwise_output_queue_depth"])
    worker_count = kernel.launch.grid_size
    if (block_factor <= 0 or ub_factor <= 0 or ub_factor > block_factor
            or input_depth <= 0 or input_ping_pong_slots <= 0
            or output_depth <= 0
            or worker_count != inp.active_units
            or worker_count != math.ceil(elements / block_factor)):
        raise ValueError(
            "CANN elementwise launch disagrees with its source tiling")
    if (tiling_key == 210_000_000
            and (ub_factor != block_factor
                 or input_depth != 1 or input_ping_pong_slots != 2
                 or output_depth != 1)):
        raise ValueError(
            "CANN elementwise key 210000000 requires one full-block UB tile")
    if (tiling_key == 210_010_000
            and (input_depth != 2 or input_ping_pong_slots != 2
                 or output_depth != 1)):
        raise ValueError(
            "CANN Add key 210010000 requires two input ping/pong slots and "
            "one CopyOut slot")
    if (tiling_key == 210_020_000
            and (input_depth != 4 or input_ping_pong_slots != 2
                 or output_depth != 1)):
        raise ValueError(
            "CANN Mul key 210020000 requires its source admission and "
            "ping/pong schedule")

    dtype_bytes = 2
    read_bytes = float(2 * elements * dtype_bytes)
    write_bytes = float(elements * dtype_bytes)
    if not math.isclose(
            inp.demand_per_resource["l2_cache"], read_bytes + write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "CANN elementwise L2 demand disagrees with its two-input ABI")
    if not math.isclose(
            inp.demand_per_resource["mte3"], write_bytes,
            rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "CANN elementwise MTE3 demand disagrees with its output ABI")
    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN elementwise pipeline requires one Vector demand")
    vector_demand, = vector_demands
    if (vector_demand.operations != float(4 * elements)
            or vector_demand.service_demands
            or vector_demand.exact_request_demands):
        raise ValueError(
            "CANN BF16 elementwise compute demand disagrees with its four "
            "source operations per element")

    tile_counts_by_worker = []
    worker_elements_by_worker = []
    for worker in range(worker_count):
        worker_elements = min(
            block_factor, elements - worker * block_factor)
        if worker_elements <= 0:
            raise ValueError(
                "CANN elementwise launched a worker without source work")
        worker_elements_by_worker.append(worker_elements)
        tile_counts_by_worker.append(math.ceil(worker_elements / ub_factor))

    finite_program_key = (
        ub_factor * dtype_bytes, min(tile_counts_by_worker))
    finite_program_service = None
    if (finite_program_key
            in _CANN_ELEMENTWISE_MUL_Q4Q1_FINITE_COMPLETION_SERVICES):
        finite_program_service = (
            _CANN_ELEMENTWISE_MUL_Q4Q1_FINITE_COMPLETION_SERVICES[
                finite_program_key])
    five_tile_worker_classes_are_published = (
        ub_factor * dtype_bytes == 17_664
        and block_factor == 43_776
        and tuple(worker_elements_by_worker)
        == (43_776,) * 47 + (39_680,))
    if five_tile_worker_classes_are_published:
        if finite_program_service is not None:
            raise ValueError(
                "CANN q4/q1 finite program has ambiguous exact services")
        finite_program_service = (
            _CANN_ELEMENTWISE_MUL_Q4Q1_FIVE_TILE_FINITE_COMPLETION_SERVICE)
    finite_program_is_published = (
        operation == "mul"
        and tiling_key == 210_020_000
        and worker_count == 48
        and input_depth == 4
        and output_depth == 1
        and min(tile_counts_by_worker) == max(tile_counts_by_worker)
        and finite_program_service is not None)
    # The Add/Mul source serializes two BF16 casts, its arithmetic operation,
    # and the output cast within every tile.  Use the independently measured
    # per-AIV cast and Mul services for multi-tile programs; Add keeps the
    # architecture-defined generic Vector rate because no independent
    # vector-vector Add service exists.  A complete finite Mul program owns
    # the same work, and single-tile Mul programs keep their exact segment
    # path below.
    serialized_compute = (
        tiling_key == expected_keys[operation][1]
        and worker_count == 48
        and input_depth == {"add": 2, "mul": 4}[operation]
        and output_depth == 1
        and min(tile_counts_by_worker) > 1
        and not finite_program_is_published)
    if serialized_compute:
        required_services = (
            "cast_bf16_to_fp32_serialized",
            *(("mul_serialized",) if operation == "mul" else ()),
            "cast_fp32_to_bf16_serialized",
        )
        missing_services = tuple(
            service for service in required_services
            if not _has_compute_service(
                hardware, "compute_vector", service))
        if missing_services:
            raise ValueError(
                "CANN elementwise serialized Vector services are unavailable: "
                f"{missing_services}")
        tile_counts = tuple(tile_counts_by_worker)
        worker_elements = tuple(float(value) for value in (
            worker_elements_by_worker))
        operation_service = (
            "mul_serialized" if operation == "mul" else "generic")
        operation_serial_invocations = (
            max(tile_counts) if operation == "mul" else 0)
        operation_serial_invocations_by_worker = (
            tile_counts if operation == "mul"
            else (0,) * worker_count)
        inp = _replace_vector_service_demands(inp, (
            ComputeServiceDemand(
                "cast_bf16_to_fp32_serialized", float(2 * elements),
                serial_invocations=2 * max(tile_counts),
                operations_by_worker=tuple(
                    2.0 * value for value in worker_elements),
                serial_invocations_by_worker=tuple(
                    2 * count for count in tile_counts)),
            ComputeServiceDemand(
                operation_service, float(elements),
                serial_invocations=operation_serial_invocations,
                operations_by_worker=worker_elements,
                serial_invocations_by_worker=(
                    operation_serial_invocations_by_worker)),
            ComputeServiceDemand(
                "cast_fp32_to_bf16_serialized", float(elements),
                serial_invocations=max(tile_counts),
                operations_by_worker=worker_elements,
                serial_invocations_by_worker=tile_counts),
        ))

    finite_program_completion = ()
    finite_program_name = None
    if finite_program_is_published:
        finite_program_completion = _cann_memory_completion_request(
            hardware, "mte3", finite_program_service, 1)
        finite_program_name = "mul_q4q1_finite_worker"

    compute_segment_services = None
    if (operation == "mul"
            and tiling_key == 210_020_000
            and min(tile_counts_by_worker) == 1
            and max(tile_counts_by_worker) == 1
            and not finite_program_completion
            and worker_count
            in _CANN_ELEMENTWISE_MUL_COMPUTE_SEGMENT_SERVICES):
        candidate_services = (
            _CANN_ELEMENTWISE_MUL_COMPUTE_SEGMENT_SERVICES[worker_count])
        exact_services = hardware.resource_by_role(
            "compute_vector").exact_request_services
        request_sizes = {
            candidate_services[0]: set(worker_elements_by_worker),
            candidate_services[1]: {
                3 * value for value in worker_elements_by_worker},
        }
        if (exact_services is not None
                and all(
                    service in exact_services
                    and sizes <= set(exact_services[service][
                        "service_time_s_by_request_size"])
                    for service, sizes in request_sizes.items())):
            compute_segment_services = candidate_services
            prefix_service, suffix_service = compute_segment_services
            exact_vector_demands = (
                ComputeExactRequestDemand(
                    prefix_service,
                    tuple((value,) for value in worker_elements_by_worker)),
                ComputeExactRequestDemand(
                    suffix_service,
                    tuple((3 * value,)
                          for value in worker_elements_by_worker)),
            )
            if math.fsum(
                    demand.operations for demand in exact_vector_demands
                    ) != vector_demand.operations:
                raise ValueError(
                    "CANN elementwise Mul segment requests must conserve "
                    "the four source Vector operations per element")
            typed_vector_demand = dataclasses.replace(
                vector_demand, exact_request_demands=exact_vector_demands)
            inp = dataclasses.replace(
                inp,
                compute_demands=tuple(
                    typed_vector_demand if demand is vector_demand else demand
                    for demand in inp.compute_demands),
            )

    tbuf_allocation_bytes = (
        ub_factor * _CANN_ELEMENTWISE_TBUF_BYTES_PER_UB_ELEMENT)
    scalar_exact_services = hardware.resource_by_role(
        "scalar_aiv").exact_request_services
    tbuf_service_is_covered = (
        tbuf_allocation_bytes
        == _CANN_ELEMENTWISE_TBUF_LIFECYCLE_ALLOCATION_BYTES
        and scalar_exact_services is not None
        and _CANN_ELEMENTWISE_TBUF_LIFECYCLE_EXACT_SERVICE
        in scalar_exact_services
        and 1 in scalar_exact_services[
            _CANN_ELEMENTWISE_TBUF_LIFECYCLE_EXACT_SERVICE][
                "service_time_s_by_request_size"])
    tbuf_lifecycle_exact = (
        (ComputeExactRequestDemand(
            _CANN_ELEMENTWISE_TBUF_LIFECYCLE_EXACT_SERVICE,
            ((1,),)),)
        if tbuf_service_is_covered else ())

    add_service_names = {
        "read": _CANN_ELEMENTWISE_2R1W_GM_READ_SERVICE,
        "write": _CANN_ELEMENTWISE_2R1W_GM_WRITE_SERVICE,
    }
    mul_service_names = {
        "read": _CANN_ELEMENTWISE_MUL_Q4Q1_2R1W_GM_READ_SERVICE,
        "write": _CANN_ELEMENTWISE_MUL_Q4Q1_2R1W_GM_WRITE_SERVICE,
    }
    if operation == "add" and tiling_key == 210_010_000:
        expected_service_names = add_service_names
    elif (operation == "mul" and tiling_key == 210_020_000
          and min(tile_counts_by_worker) >= input_depth):
        # The independent full-chip service is a steady-state 4-input/1-output
        # queue bound.  A source program must fill all four input slots before
        # that overlapped regime exists; shorter programs retain the generic
        # finite-pipeline path.
        expected_service_names = mul_service_names
    else:
        expected_service_names = None
    expected_signatures = {
        "read": MemoryAccessClassSignature(
            access_class=(expected_service_names or add_service_names)[
                "read"],
            dtype="bf16",
            tile_m=1,
            tile_k=ub_factor,
            logical_row_bytes=ub_factor * dtype_bytes,
        ),
        "write": MemoryAccessClassSignature(
            access_class=(expected_service_names or add_service_names)[
                "write"],
            dtype="bf16",
            tile_m=1,
            tile_k=ub_factor,
            logical_row_bytes=ub_factor * dtype_bytes,
        ),
    }
    gm_services = {}
    if expected_service_names is not None:
        for access_kind, signature in expected_signatures.items():
            matches = tuple(
                service for service in hardware.memory_access_class_services
                if (service.service == expected_service_names[access_kind]
                    and service.signature == signature))
            if len(matches) > 1:
                raise ValueError(
                    "CANN elementwise GM service repeats an exact signature")
            if matches:
                service, = matches
                expected_ingress_role = {
                    "read": "mte2",
                    "write": "mte3",
                }[access_kind]
                if (service.access_kind != access_kind
                        or service.ingress_resource_role
                        != expected_ingress_role):
                    raise ValueError(
                        "CANN elementwise GM service direction or ingress "
                        "resource changed")
                gm_services[access_kind] = service
    if gm_services and set(gm_services) != {"read", "write"}:
        raise ValueError(
            "CANN elementwise GM service requires an exact read/write pair")
    if gm_services:
        paths_by_name = {path.name: path for path in inp.memory_paths}
        if set(paths_by_name) != {"mte2_read", "mte3_write"}:
            raise ValueError(
                "CANN elementwise GM service requires split MTE2/MTE3 paths")
        typed_paths = []
        for path in inp.memory_paths:
            if path.access_class_demands:
                raise ValueError(
                    "CANN elementwise GM path already has a typed service")
            access_kind = path.access_kind
            logical_bytes = {
                "read": read_bytes,
                "write": write_bytes,
            }[access_kind]
            service = gm_services[access_kind]
            typed_paths.append(dataclasses.replace(
                path,
                access_class_demands=(MemoryAccessClassDemand(
                    service=service.service,
                    ingress_resource_role=service.ingress_resource_role,
                    logical_bytes=logical_bytes,
                    signature=service.signature,
                ),),
            ))
        inp = dataclasses.replace(inp, memory_paths=tuple(typed_paths))

    completion_prefix_cache = {}

    def completion_prefix(resource_role, service, payload_bytes):
        key = (resource_role, service, payload_bytes)
        if key not in completion_prefix_cache:
            completion_prefix_cache[key] = (
                _cann_memory_completion_prefix_request(
                    hardware, resource_role, service, payload_bytes))
        return completion_prefix_cache[key]

    def _tile_task(worker, tile_elements, *, include_tbuf_lifecycle=False):
        payload_bytes = tile_elements * dtype_bytes
        mte2_prefix_bytes, mte2_exact = (
            completion_prefix(
                "mte2", _CANN_MTE2_COMPLETION_EXACT_SERVICE,
                payload_bytes))
        mte3_prefix_bytes, mte3_exact = (
            completion_prefix(
                "mte3", _CANN_MTE3_COMPLETION_EXACT_SERVICE,
                payload_bytes))
        read_fraction = tile_elements / (2.0 * elements)
        write_fraction = tile_elements / float(elements)
        queue_control_cycles = (
            _CANN_ELEMENTWISE_QUEUE_CONTROL_CYCLES_PER_TILE
            - (2 if mte2_exact else 0)
            - (1 if mte3_exact else 0)
        )
        if queue_control_cycles <= 0:
            raise ValueError(
                "CANN elementwise queue-control coverage is invalid")
        read_nodes = ()
        write_nodes = ()
        read_access_class = ()
        write_access_class = ()
        if gm_services:
            read_path = paths_by_name["mte2_read"]
            write_path = paths_by_name["mte3_write"]
            read_nodes = tuple(PhaseMemoryDemand(
                path_name=read_path.name,
                resource_role=node.resource_role,
                bytes=node.bytes * payload_bytes / read_bytes,
            ) for node in read_path.nodes)
            write_nodes = tuple(PhaseMemoryDemand(
                path_name=write_path.name,
                resource_role=node.resource_role,
                bytes=node.bytes * payload_bytes / write_bytes,
            ) for node in write_path.nodes)
            read_service = gm_services["read"]
            write_service = gm_services["write"]
            read_access_class = (MemoryAccessClassDemand(
                service=read_service.service,
                ingress_resource_role=read_service.ingress_resource_role,
                logical_bytes=float(payload_bytes),
                signature=read_service.signature,
            ),)
            write_access_class = (MemoryAccessClassDemand(
                service=write_service.service,
                ingress_resource_role=write_service.ingress_resource_role,
                logical_bytes=float(payload_bytes),
                signature=write_service.signature,
            ),)
            read_fraction = None
            write_fraction = None

        def _scaled_nodes(nodes, fragment_bytes):
            return tuple(dataclasses.replace(
                node, bytes=node.bytes * fragment_bytes / payload_bytes)
                for node in nodes)

        def _scaled_access_classes(demands, fragment_bytes):
            return tuple(dataclasses.replace(
                demand, logical_bytes=(
                    demand.logical_bytes
                    * fragment_bytes / payload_bytes))
                for demand in demands)

        def _memory_request_stages(
                name, service, total_fraction, aggregate_bytes,
                nodes, access_classes, prefix_bytes, exact_completion, *,
                predecessors):
            fragments = []
            if exact_completion:
                fragments.append((
                    f"{name}_completion_prefix", prefix_bytes,
                    exact_completion, predecessors))
                remaining_bytes = payload_bytes - prefix_bytes
                if remaining_bytes:
                    fragments.append((
                        name, remaining_bytes, (),
                        (f"{name}_completion_prefix",)))
            else:
                fragments.append((name, payload_bytes, (), predecessors))
            if len(fragments) == 1:
                fragment_name, fragment_bytes, exact, fragment_predecessors = (
                    fragments[0])
                fragments = [(
                    name, fragment_bytes, exact, fragment_predecessors)]
            stages = []
            for (fragment_name, fragment_bytes, exact,
                 fragment_predecessors) in fragments:
                fraction = (
                    None if total_fraction is None
                    else fragment_bytes / aggregate_bytes)
                stages.append(PipelineStageDemand(
                    fragment_name, service, fraction,
                    memory_node_demands=_scaled_nodes(
                        nodes, fragment_bytes),
                    memory_exact_request_demands=exact,
                    memory_access_class_demands=_scaled_access_classes(
                        access_classes, fragment_bytes),
                    predecessors=fragment_predecessors,
                ))
            return tuple(stages)
        cast_a_demands = (
            (ComputeServiceDemand(
                "cast_bf16_to_fp32_serialized", float(tile_elements),
                serial_invocations=1),)
            if serialized_compute else ())
        cast_b_demands = cast_a_demands
        operation_demands = (
            (ComputeServiceDemand(
                ("mul_serialized" if operation == "mul" else "generic"),
                float(tile_elements),
                serial_invocations=(1 if operation == "mul" else 0)),)
            if serialized_compute else ())
        cast_out_demands = (
            (ComputeServiceDemand(
                "cast_fp32_to_bf16_serialized", float(tile_elements),
                serial_invocations=1),)
            if serialized_compute else ())
        compute_fraction = (
            None if serialized_compute
            else tile_elements / (4.0 * elements))
        if compute_segment_services is not None:
            prefix_service, suffix_service = compute_segment_services
            compute_stages = (
                PipelineStageDemand(
                    "cast_a", "compute:compute_vector", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            prefix_service, ((tile_elements,),)),),
                    predecessors=("load_a",),
                    start_predecessors=("load_b",)),
                PipelineStageDemand(
                    "mul_suffix", "compute:compute_vector", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            suffix_service, ((3 * tile_elements,),)),),
                    predecessors=("cast_a", "load_b")),
            )
        else:
            compute_stages = (
                PipelineStageDemand(
                    "cast_a", "compute:compute_vector", compute_fraction,
                    compute_service_demands=cast_a_demands,
                    predecessors=("load_a",),
                    start_predecessors=("load_b",)),
                PipelineStageDemand(
                    "cast_b", "compute:compute_vector", compute_fraction,
                    compute_service_demands=cast_b_demands,
                    predecessors=("load_b",)),
                PipelineStageDemand(
                    operation, "compute:compute_vector", compute_fraction,
                    compute_service_demands=operation_demands,
                    predecessors=("cast_a", "cast_b")),
                PipelineStageDemand(
                    "cast_out", "compute:compute_vector", compute_fraction,
                    compute_service_demands=cast_out_demands),
            )
        lifecycle_stages = (
            (PipelineStageDemand(
                "tbuf_lifecycle", "compute:scalar_aiv", None,
                compute_exact_request_demands=tbuf_lifecycle_exact),)
            if include_tbuf_lifecycle else ())
        load_a_stages = _memory_request_stages(
            "load_a", "memory:mte2_read", read_fraction, read_bytes,
            read_nodes, read_access_class, mte2_prefix_bytes, mte2_exact,
            predecessors=("queue_control",))
        load_b_stages = _memory_request_stages(
            "load_b", "memory:mte2_read", read_fraction, read_bytes,
            read_nodes, read_access_class, mte2_prefix_bytes, mte2_exact,
            predecessors=())
        store_stages = _memory_request_stages(
            "store", "memory:mte3_write", write_fraction, write_bytes,
            write_nodes, write_access_class, mte3_prefix_bytes, mte3_exact,
            predecessors=None)
        return PipelineTaskDemand(
            name=f"tile_{tile_elements}",
            worker=worker,
            stages=lifecycle_stages + (
                PipelineStageDemand(
                    "queue_control", "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,) * queue_control_cycles,),),)),
            ) + load_a_stages + load_b_stages + compute_stages + store_stages,
        )

    # Keep the aggregate scalar demand identical to the per-tile stages.  A
    # completion service is an exact source edge, not an extra memory floor;
    # charging its corresponding mutex lifetime here would count that edge
    # twice on the same serial worker path.
    queue_control_cycles_by_worker = []
    for worker in range(worker_count):
        worker_elements = min(
            block_factor, elements - worker * block_factor)
        queue_control_cycles_by_worker.append(sum(
            _CANN_ELEMENTWISE_QUEUE_CONTROL_CYCLES_PER_TILE
            - (2 if completion_prefix(
                "mte2", _CANN_MTE2_COMPLETION_EXACT_SERVICE,
                min(ub_factor, worker_elements - offset) * dtype_bytes)[1]
               else 0)
            - (1 if completion_prefix(
                "mte3", _CANN_MTE3_COMPLETION_EXACT_SERVICE,
                min(ub_factor, worker_elements - offset) * dtype_bytes)[1]
               else 0)
            for offset in range(0, worker_elements, ub_factor)
        ))
    scalar_exact_demands = [ComputeExactRequestDemand(
        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
        (),
        request_size_runs_by_worker=tuple(
            ((1, queue_control_cycles),)
            for queue_control_cycles in queue_control_cycles_by_worker))]
    if tbuf_lifecycle_exact:
        scalar_exact_demands.append(ComputeExactRequestDemand(
            _CANN_ELEMENTWISE_TBUF_LIFECYCLE_EXACT_SERVICE,
            ((1,),) * worker_count))
    if not finite_program_completion:
        inp = _attach_scalar_aiv_exact_demands(
            inp, tuple(scalar_exact_demands))

    if finite_program_completion:
        # The independent finite-program probes preserve their complete 2R1W
        # queue program and replace only production Add/Mul arithmetic. Price
        # that memory/control program as one competing maximum-worker
        # completion, while retaining the conserved aggregate MTE2 and Vector
        # work as independent lower bounds. This avoids serially charging the
        # measured program on top of its own loads, queue control, and stores.
        worker_fraction = 1.0 / worker_count
        if finite_program_name is None:
            raise RuntimeError("finite elementwise program has no identity")
        programs = (PipelineTaskProgramDemand(
            name=f"{finite_program_name}_program",
            worker=0,
            patterns=(
                PipelineTaskSequenceDemand(
                    name=f"{finite_program_name}_sequence",
                    worker=0,
                    tasks=(PipelineTaskDemand(
                        name=f"{finite_program_name}_completion",
                        worker=0,
                        stages=(
                            PipelineStageDemand(
                                "mte2_read", "memory:mte2_read",
                                worker_fraction, predecessors=()),
                            PipelineStageDemand(
                                "vector_compute", "compute:compute_vector",
                                worker_fraction, predecessors=()),
                            PipelineStageDemand(
                                "finite_memory_program", "memory:mte3_write",
                                worker_fraction,
                                memory_exact_request_demands=(
                                    finite_program_completion),
                                predecessors=()),
                        ),
                    ),),
                ),
            ),
            order=(0,),
            worker_multiplicity=worker_count,
            physical_workers=tuple(range(worker_count)),
        ),)
        buffers = ()
    elif analytical_compact:
        # CANN assigns contiguous equal-size blocks, with at most one tail.
        # The Forward representation carries one source program per distinct
        # block extent and maps it to its physical AIV population; it never
        # expands an identical program once per worker or per UB tile.
        worker_classes = {}
        for worker in range(worker_count):
            worker_elements = min(
                block_factor, elements - worker * block_factor)
            if worker_elements <= 0:
                raise ValueError(
                    "CANN elementwise launched a worker without source work")
            if worker_elements not in worker_classes:
                worker_classes[worker_elements] = []
            worker_classes[worker_elements].append(worker)
        programs = []
        for compact_worker, (worker_elements, physical_workers) in enumerate(
                worker_classes.items()):
            class_indexes = {}
            patterns = []
            order = []
            for offset in range(0, worker_elements, ub_factor):
                tile_elements = min(ub_factor, worker_elements - offset)
                if offset == 0 and tbuf_lifecycle_exact:
                    patterns.append(PipelineTaskSequenceDemand(
                        name=f"initial_tile_{tile_elements}",
                        worker=compact_worker,
                        tasks=(_tile_task(
                            compact_worker, tile_elements,
                            include_tbuf_lifecycle=True),),
                    ))
                    order.append(len(patterns) - 1)
                    continue
                if tile_elements not in class_indexes:
                    class_indexes[tile_elements] = len(patterns)
                    patterns.append(PipelineTaskSequenceDemand(
                        name=f"tile_{tile_elements}",
                        worker=compact_worker,
                        tasks=(_tile_task(compact_worker, tile_elements),),
                    ))
                order.append(class_indexes[tile_elements])
            programs.append(PipelineTaskProgramDemand(
                name=f"worker_class{compact_worker}_tiles",
                worker=compact_worker,
                patterns=tuple(patterns),
                order=tuple(order),
                worker_multiplicity=len(physical_workers),
                physical_workers=tuple(physical_workers),
            ))
    else:
        programs = []
        for worker in range(worker_count):
            worker_elements = min(
                block_factor, elements - worker * block_factor)
            if worker_elements <= 0:
                raise ValueError(
                    "CANN elementwise launched a worker without source work")
            class_indexes = {}
            patterns = []
            order = []
            for offset in range(0, worker_elements, ub_factor):
                tile_elements = min(ub_factor, worker_elements - offset)
                if offset == 0 and tbuf_lifecycle_exact:
                    patterns.append(PipelineTaskSequenceDemand(
                        name=f"initial_tile_{tile_elements}",
                        worker=worker,
                        tasks=(_tile_task(
                            worker, tile_elements,
                            include_tbuf_lifecycle=True),),
                    ))
                    order.append(len(patterns) - 1)
                    continue
                if tile_elements not in class_indexes:
                    class_indexes[tile_elements] = len(patterns)
                    patterns.append(PipelineTaskSequenceDemand(
                        name=f"tile_{tile_elements}",
                        worker=worker,
                        tasks=(_tile_task(worker, tile_elements),),
                    ))
                order.append(class_indexes[tile_elements])
            programs.append(PipelineTaskProgramDemand(
                name=f"worker{worker}_tiles",
                worker=worker,
                patterns=tuple(patterns),
                order=tuple(order),
            ))

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tuple(programs),
        buffers=buffers if finite_program_completion else (
            PipelineBuffer(
                name="input_a_buffer",
                producer_stage="load_a",
                consumer_stage="cast_a",
                slots=input_ping_pong_slots,
                scope="worker",
            ),
            PipelineBuffer(
                name="input_b_buffer",
                producer_stage="load_b",
                consumer_stage=(
                    "mul_suffix"
                    if compute_segment_services is not None else "cast_b"),
                slots=input_ping_pong_slots,
                scope="worker",
            ),
            PipelineBuffer(
                name="output_buffer",
                producer_stage=(
                    "mul_suffix"
                    if compute_segment_services is not None else "cast_out"),
                consumer_stage="store",
                slots=output_depth,
                scope="worker",
            ),
        ),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    result = dataclasses.replace(
        attached,
        task_duration_floor_contract=AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
    )
    return result


def _validate_cann_elementwise_tiling_prologue_contract(
        kernel: KernelSpec, hardware) -> None:
    """Fail closed until the 56-B BroadcastOneDim prologue is measured.

    CANN 8.5.1 Add/Mul source enters ``BroadcastSch<201>`` before the source
    DAG. That entry performs ``GET_TILING_DATA_WITH_STRUCT`` on the 56-B
    ``BroadcastOneDimTilingData`` layout. The only published generic prologue
    curve is the distinct 28-B RmsNorm materialization and cannot price the
    second 32-B MTE2 burst or its source-visible completion. Returning a
    prediction with that curve would silently give Dataset and Serving the
    same wrong physical program, so both routes refuse it until an
    independent 56-B mechanism service is published.
    """

    if (hardware.kernel_memory_model != "npu_tile"
            or not hardware.aic_aiv_separated):
        raise ValueError(
            "CANN elementwise BroadcastOneDim source program requires "
            "Ascend separated AIV hardware")
    if kernel.variant.lower() != "elementwise" or kernel.dtype != "bf16":
        raise ValueError(
            "CANN elementwise BroadcastOneDim source program requires a "
            "BF16 elementwise KernelSpec")
    parameters = kernel.launch.parameters
    if "cann_tiling_payload_bytes" not in parameters:
        raise ValueError(
            "CANN elementwise source program is missing its typed tiling "
            "payload size")
    if parameters["cann_tiling_payload_bytes"] != 56.0:
        raise ValueError(
            "CANN elementwise source program has an invalid "
            "BroadcastOneDim tiling payload size")


def _rows_by_worker(rows, worker_count, block_factor):
    """Split one tile-pipelined row space evenly across the launch's workers.

    Every worker but the last covers ``block_factor`` rows; the last takes the
    remainder.  This is the CANN tile-pipeline row partition shared by the
    RMSNorm and AddRMSNorm exact-service envelopes.
    """

    return tuple(
        block_factor
        if worker < worker_count - 1
        else rows - (worker_count - 1) * block_factor
        for worker in range(worker_count)
    )


def _attach_cann_norm_exact_service(
        inp: PredictionInput, kernel: KernelSpec, hardware
        ) -> PredictionInput:
    """Replace one logical reciprocal-sqrt per row with its AIV schedule.

    The fixed CANN 8.5.1 selected-key sources assign contiguous row ranges to
    AIVs through ``blockFactor``.  Keys 1/31 issue one serialized
    Sqrt->Duplicate->Div request per at-most-64-row group, while keys 30/33
    issue one size-one request per row.  The complete request service is
    measured independently in UB and is priced on the critical worker; no
    production-kernel duration or inferred overlap enters this contract.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN NORM exact service requires separated npu_tile AIVs")
    variant = kernel.variant.lower()
    if variant not in ("rmsnorm", "add_rmsnorm"):
        raise ValueError("CANN NORM exact service requires a NORM kernel")
    parameters = kernel.launch.parameters
    required = ("cann_tiling_key", "norm_block_factor")
    missing = tuple(name for name in required if name not in parameters)
    if missing:
        raise ValueError(
            f"CANN NORM exact service launch parameters are incomplete: "
            f"{missing}")
    tiling_key = int(parameters["cann_tiling_key"])
    if tiling_key in (0, 2):
        if variant != "rmsnorm":
            raise ValueError(
                "CANN NORM keys 0 and 2 are valid only for RMSNorm")
    elif variant == "rmsnorm":
        if tiling_key != 1:
            raise ValueError("CANN RMSNorm exact service requires key 1")
    elif tiling_key not in (30, 31, 33):
        raise ValueError(
            "CANN AddRmsNorm exact service requires key 30, 31, or 33")
    else:
        pass

    rows = _shape(kernel, "rows")
    worker_count = kernel.launch.grid_size
    if worker_count != inp.active_units:
        raise ValueError(
            "CANN NORM exact service worker count must equal active AIVs")
    block_factor = int(parameters["norm_block_factor"])
    if block_factor <= 0:
        raise ValueError("CANN NORM block factor must be positive")
    expected_block_factor = math.ceil(rows / worker_count)
    if block_factor != expected_block_factor:
        raise ValueError(
            "CANN NORM block factor disagrees with rows and blockDim")

    rows_by_worker = _rows_by_worker(rows, worker_count, block_factor)
    if (any(worker_rows <= 0 or worker_rows > block_factor
            for worker_rows in rows_by_worker)
            or sum(rows_by_worker) != rows):
        raise ValueError("CANN NORM worker row ownership is inconsistent")

    if tiling_key in (0, 2):
        # Keys 0 and 2 own their reciprocal-sqrt request inside their
        # source-closed worker programs, which type the whole Vector demand.
        return inp

    if tiling_key in (1, 31):
        def _requests(worker_rows):
            full_groups, tail = divmod(
                worker_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
            return ((
                (_CANN_NORM_SPLIT_ROW_FACTOR,) * full_groups
                + ((tail,) if tail else ())))
    elif tiling_key == 30:
        def _requests(worker_rows):
            return (1,) * worker_rows
    elif tiling_key == 33:
        if block_factor != 1 or any(
                worker_rows != 1 for worker_rows in rows_by_worker):
            raise ValueError(
                "CANN AddRmsNorm key 33 requires one row per AIV")

        def _requests(worker_rows):
            if worker_rows != 1:
                raise ValueError(
                    "CANN AddRmsNorm key 33 worker must own one row")
            return (1,)
    else:
        raise AssertionError(f"unexpected CANN NORM key: {tiling_key}")

    exact_demand = ComputeExactRequestDemand(
        service=_CANN_NORM_EXACT_SERVICE,
        request_sizes_by_worker=tuple(
            _requests(worker_rows) for worker_rows in rows_by_worker),
    )
    if exact_demand.operations != float(rows):
        raise ValueError(
            "CANN NORM exact requests must conserve one request element per "
            "row")
    vector_demands = tuple(
        item for item in inp.compute_demands
        if item.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN NORM exact service requires one compute_vector demand")
    vector_demand = vector_demands[0]
    if vector_demand.service_demands or vector_demand.exact_request_demands:
        raise ValueError(
            "CANN NORM exact service cannot replace an already typed Vector "
            "demand")
    generic_operations = vector_demand.operations - exact_demand.operations
    if generic_operations <= 0:
        raise ValueError(
            "CANN NORM generic Vector operations must remain positive")
    generic_operations_per_row = generic_operations / rows
    generic_operations_by_worker = tuple(
        generic_operations_per_row * worker_rows
        for worker_rows in rows_by_worker)
    typed_vector_demand = dataclasses.replace(
        vector_demand,
        service_demands=(
            ComputeServiceDemand(
                "generic", generic_operations,
                operations_by_worker=generic_operations_by_worker,
                serial_invocations_by_worker=(0,) * worker_count,
            ),),
        exact_request_demands=(exact_demand,),
    )
    typed_compute_demands = list(inp.compute_demands)
    vector_demand_index = typed_compute_demands.index(vector_demand)
    typed_compute_demands[vector_demand_index] = typed_vector_demand
    return dataclasses.replace(
        inp, compute_demands=tuple(typed_compute_demands))


def _cann_norm_reduction_dependency_fraction(
        inp: PredictionInput, rows: int) -> Optional[float]:
    """Return one row's share of the published ReduceSum dependency.

    The aggregate NPU contract prices the independently measured excess of
    ``ReduceSum + PIPE_V barrier`` over an ideal Vector add stream on the
    ``compute_reduction`` resource.  A source-closed CANN NORM worker DAG must
    place that excess between its row reduction and scalar normalization;
    otherwise the formula lets it overlap the complete Load/Vector/Store path
    and retains it only as an unrelated resource lower bound.
    """

    if rows <= 0:
        raise ValueError("CANN NORM reduction dependency requires rows")
    reduction_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_reduction")
    if not reduction_demands:
        return None
    if len(reduction_demands) != 1:
        raise ValueError(
            "CANN NORM requires at most one reduction dependency demand")
    reduction = reduction_demands[0]
    if (reduction.operations <= 0
            or reduction.service_demands
            or reduction.exact_request_demands):
        raise ValueError(
            "CANN NORM ReduceSum dependency must be one positive aggregate "
            "service")
    return 1.0 / rows


def _cann_rmsnorm_row_stage_demands(hardware, hidden: int):
    """Return the source-ordered typed Vector work for one key-0 row."""

    if hidden <= 0:
        raise ValueError("CANN RMSNorm typed row requires positive hidden")
    missing = tuple(
        service for service in _CANN_RMSNORM_TYPED_SERVICES
        if not _has_compute_service(
            hardware, "compute_vector", service))
    if missing:
        raise ValueError(
            "CANN RMSNorm key-0 typed services are unavailable: "
            f"{missing}")
    elements = float(hidden)
    return {
        "input_square_reduce": (
            ComputeServiceDemand(
                "cast_bf16_to_fp32_serialized", elements,
                serial_invocations=1),
            ComputeServiceDemand(
                "mul_serialized", elements, serial_invocations=1),
            ComputeServiceDemand(
                "rmsnorm_half_interval_reduce", elements,
                serial_invocations=1),
        ),
        "output_scale": (ComputeServiceDemand(
            "muls_serialized", elements, serial_invocations=1),),
        "gamma_apply": (
            ComputeServiceDemand(
                "cast_bf16_to_fp32_serialized", elements,
                serial_invocations=1),
            ComputeServiceDemand(
                "mul_serialized", elements, serial_invocations=1),
        ),
        "final_cast": (ComputeServiceDemand(
            "cast_fp32_to_bf16_serialized", elements,
            serial_invocations=1),),
    }


def _cann_add_rmsnorm_row_stage_demands(
        hardware, hidden: int, *, single_n: bool):
    """Return source-ordered typed Vector work for one AddRmsNorm row."""

    if hidden <= 0:
        raise ValueError("CANN AddRmsNorm typed row requires positive hidden")
    if not isinstance(single_n, bool):
        raise TypeError("single_n must be bool")
    required_services = _CANN_RMSNORM_TYPED_SERVICES + (
        (() if single_n else (
            _CANN_ADD_RMSNORM_REDUCE_SUM_FP32_SERVICE,)))
    missing = tuple(
        service for service in required_services
        if not _has_compute_service(
            hardware, "compute_vector", service))
    if missing:
        raise ValueError(
            "CANN AddRmsNorm typed services are unavailable: "
            f"{missing}")
    elements = float(hidden)
    input_compute = (
        ComputeServiceDemand(
            "cast_bf16_to_fp32_serialized", 2 * elements,
            serial_invocations=2),
        ComputeServiceDemand("generic", elements),
        ComputeServiceDemand(
            "cast_fp32_to_bf16_serialized", elements,
            serial_invocations=1),
    )
    reduce_compute = (
        *((ComputeServiceDemand(
            "cast_bf16_to_fp32_serialized", elements,
            serial_invocations=1),) if single_n else ()),
        ComputeServiceDemand(
            "mul_serialized", elements, serial_invocations=1),
        ComputeServiceDemand(
            "muls_serialized", elements, serial_invocations=1),
        ComputeServiceDemand(
            "generic" if single_n
            else _CANN_ADD_RMSNORM_REDUCE_SUM_FP32_SERVICE,
            (2 if single_n else 1) * elements,
            serial_invocations=(0 if single_n else 1)),
    )
    output_scale = (ComputeServiceDemand(
        "muls_serialized", elements, serial_invocations=1),)
    output_tail = {
        "cast_bf16_to_fp32_serialized": 2,
        "mul_serialized": 1,
        "cast_fp32_to_bf16_serialized": 2,
    }
    if single_n:
        return {
            "input_x1_cast": (ComputeServiceDemand(
                "cast_bf16_to_fp32_serialized", elements,
                serial_invocations=1),),
            "input_compute": (
                ComputeServiceDemand(
                    "cast_bf16_to_fp32_serialized", elements,
                    serial_invocations=1),
                ComputeServiceDemand("generic", elements),
                ComputeServiceDemand(
                    "cast_fp32_to_bf16_serialized", elements,
                    serial_invocations=1),
            ),
            "reduce_compute": reduce_compute,
            "epsilon_add": (ComputeServiceDemand("generic", 1.0),),
            "output_scale": output_scale,
            "residual_roundtrip": (
                ComputeServiceDemand(
                    "cast_fp32_to_bf16_serialized", elements,
                    serial_invocations=1),
                ComputeServiceDemand(
                    "cast_bf16_to_fp32_serialized", elements,
                    serial_invocations=1),
            ),
            "gamma_apply": (
                ComputeServiceDemand(
                    "cast_bf16_to_fp32_serialized", elements,
                    serial_invocations=1),
                ComputeServiceDemand(
                    "mul_serialized", elements, serial_invocations=1),
            ),
            "final_cast": (ComputeServiceDemand(
                "cast_fp32_to_bf16_serialized", elements,
                serial_invocations=1),),
        }
    return {
        "input_compute": input_compute,
        "reduce_compute": reduce_compute,
        "epsilon_add": (ComputeServiceDemand("generic", 1.0),),
        "output_scale": output_scale,
        "output_finalize": tuple(
            ComputeServiceDemand(
                service, count * elements, serial_invocations=count)
            for service, count in output_tail.items()),
    }


def _cann_add_rmsnorm_key31_row_stage_demands(hardware, hidden: int):
    """Return the typed Vector stages of the BF16 Split-D row program."""

    stages = _cann_add_rmsnorm_row_stage_demands(
        hardware, hidden, single_n=False)
    # Split-D reloads the BF16 residual into ``xFp32Buf`` before ComputeY's
    # V->S rstd handoff.  Keep that exact first cast separate from the later
    # BF16 gamma cast and residual roundtrip so the source-visible scalar
    # event is placed at its real instruction boundary.
    residual_reload = ()
    output_finalize = []
    augmented = 0
    for demand in stages["output_finalize"]:
        if demand.service == "cast_bf16_to_fp32_serialized":
            if (demand.operations != float(2 * hidden)
                    or demand.serial_invocations != 2):
                raise ValueError(
                    "CANN key-31 output cast source count changed")
            residual_reload = (ComputeServiceDemand(
                demand.service, float(hidden), serial_invocations=1),)
            augmented += 1
        output_finalize.append(demand)
    if augmented != 1 or not residual_reload:
        raise ValueError("CANN key-31 residual reload source is unavailable")
    stages["residual_reload"] = residual_reload
    stages["output_finalize"] = tuple(output_finalize)
    return stages


def _type_cann_add_rmsnorm_key31_split_d_rows(
        inp: PredictionInput, hardware, rows_by_worker: Tuple[int, ...],
        tile_widths: Tuple[int, ...]):
    """Type one key-31 Split-D source invocation per row and column tile.

    ``AddRmsNormSplitD`` calls its Vector helpers once for every 8,192-wide
    column tile.  Its arithmetic remains the same full-row work, but setup
    and queue lifetimes are tile-local source calls.  Keeping that distinction
    here lets both Dataset and Serving enter the same normalized launch
    program without inventing a shape-duration rule.
    """

    if (not rows_by_worker or any(rows <= 0 for rows in rows_by_worker)
            or not tile_widths or any(width <= 0 for width in tile_widths)):
        raise ValueError(
            "CANN AddRMSNorm key-31 requires positive row and tile geometry")
    tile_stage_demands = tuple(
        _cann_add_rmsnorm_key31_row_stage_demands(hardware, width)
        for width in tile_widths)
    service_order = []
    operations_per_row = {}
    invocations_per_row = {}

    def _add_demands(demands):
        for demand in demands:
            if demand.service not in operations_per_row:
                service_order.append(demand.service)
                operations_per_row[demand.service] = 0.0
                invocations_per_row[demand.service] = 0
            operations_per_row[demand.service] += demand.operations
            invocations_per_row[demand.service] += demand.serial_invocations

    # ``ComputeEpsilon`` executes after the complete row-group reduction, not
    # once per column tile.  The first tile supplies its fixed one-element
    # source operation; all tile-local Vector bodies remain explicit.
    for stages in tile_stage_demands:
        for stage_name in (
                "input_compute", "reduce_compute", "residual_reload",
                "output_scale", "output_finalize"):
            _add_demands(stages[stage_name])
    _add_demands(tile_stage_demands[0]["epsilon_add"])
    # ``AddRmsNormSplitD::SubProcess`` initializes the depth-one rstd output
    # tensor with ``Duplicate(0.0, calc_row_num)`` before its Former loop.
    # It is one ordinary Vector operation per logical row, not a duration
    # floor and not another row-tile body.  Preserve it in the aggregate
    # contract so the explicit group-init stage below owns every operation.
    _add_demands((ComputeServiceDemand("generic", 1.0),))

    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN AddRMSNorm key-31 requires one Vector aggregate demand")
    vector_demand, = vector_demands
    total_rows = sum(rows_by_worker)
    service_demands = tuple(
        ComputeServiceDemand(
            service,
            operations_per_row[service] * total_rows,
            serial_invocations=(
                invocations_per_row[service] * max(rows_by_worker)),
            operations_by_worker=tuple(
                operations_per_row[service] * rows
                for rows in rows_by_worker),
            serial_invocations_by_worker=tuple(
                invocations_per_row[service] * rows
                for rows in rows_by_worker),
        )
        for service in service_order
    )
    typed_operations = math.fsum(
        demand.operations for demand in service_demands)
    exact_operations = math.fsum(
        demand.operations for demand in vector_demand.exact_request_demands)
    if vector_demand.operations != typed_operations + exact_operations:
        raise ValueError(
            "CANN AddRMSNorm key-31 tiled source work disagrees with the "
            "aggregate Vector contract")
    typed_vector_demand = dataclasses.replace(
        vector_demand, service_demands=service_demands)
    return (
        dataclasses.replace(
            inp,
            compute_demands=tuple(
                typed_vector_demand if demand is vector_demand else demand
                for demand in inp.compute_demands)),
        tile_stage_demands,
    )


def _type_cann_add_rmsnorm_key31_block_reduce(
        inp: PredictionInput, hardware, *,
        groups_by_worker: Tuple[Tuple[int, ...], ...],
        tile_count: int) -> PredictionInput:
    """Type the source's per-group ``BlockReduceSumFP32`` branches.

    ``AddRmsNormSplitD::ComputeFormer`` calls ``BlockReduceSumFP32`` once
    for every owned row group and 8,192-wide column tile, with exactly
    ``count = group_rows * 8`` FP32 partials.  Its fixed CANN 8.5.1 helper
    first executes a 64-wide-repeat branch when ``count // 64`` is nonzero,
    then one 8-wide-aligned tail branch when ``count % 64`` is nonzero.  The
    independent mechanism services price those two source branches; they do
    not represent a whole-hidden generic ReduceSum or an evaluation-derived
    correction.
    """

    if (not groups_by_worker
            or any(not groups for groups in groups_by_worker)
            or any(group_rows <= 0
                   or group_rows > _CANN_NORM_SPLIT_ROW_FACTOR
                   for groups in groups_by_worker for group_rows in groups)):
        raise ValueError(
            "CANN AddRMSNorm key-31 BlockReduce requires 1..64-row groups")
    if isinstance(tile_count, bool) or not isinstance(tile_count, int):
        raise TypeError("CANN AddRMSNorm key-31 BlockReduce tile count must be int")
    if tile_count <= 0:
        raise ValueError(
            "CANN AddRMSNorm key-31 BlockReduce requires positive tile count")
    if "compute_reduction" not in hardware._role_map:
        raise ValueError(
            "CANN AddRMSNorm key-31 BlockReduce requires compute_reduction")
    reduction_resource = hardware.resource_by_role("compute_reduction")
    if reduction_resource.service_rates_per_unit_s is None:
        raise ValueError(
            "CANN AddRMSNorm key-31 BlockReduce services are unavailable")
    for service in (
            _CANN_ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_FULL64_SERVICE,
            _CANN_ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_TAIL8_SERVICE):
        if service not in reduction_resource.service_rates_per_unit_s:
            raise ValueError(
                "CANN AddRMSNorm key-31 BlockReduce source service is "
                f"unavailable: {service!r}")

    full_repeats_by_worker = tuple(
        tile_count * sum(group_rows // 8 for group_rows in groups)
        for groups in groups_by_worker)
    full_invocations_by_worker = tuple(
        tile_count * sum(group_rows >= 8 for group_rows in groups)
        for groups in groups_by_worker)
    tail_invocations_by_worker = tuple(
        tile_count * sum(group_rows % 8 != 0 for group_rows in groups)
        for groups in groups_by_worker)
    if any(full_repeats < full_invocations
           for full_repeats, full_invocations in zip(
               full_repeats_by_worker, full_invocations_by_worker)):
        raise AssertionError(
            "CANN AddRMSNorm key-31 BlockReduce repeat accounting drifted")

    services = []
    full_repeats = sum(full_repeats_by_worker)
    if full_repeats:
        services.append(ComputeServiceDemand(
            _CANN_ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_FULL64_SERVICE,
            float(full_repeats),
            serial_invocations=max(full_invocations_by_worker),
            operations_by_worker=tuple(
                float(value) for value in full_repeats_by_worker),
            serial_invocations_by_worker=full_invocations_by_worker,
        ))
    tail_invocations = sum(tail_invocations_by_worker)
    if tail_invocations:
        # A tail is one masked BlockReduceSum plus its source-required
        # barrier.  Its fixed mechanism interval is represented as one
        # service operation rather than pretending tail width is throughput
        # work; zero setup prevents charging a second synthetic invocation.
        services.append(ComputeServiceDemand(
            _CANN_ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_TAIL8_SERVICE,
            float(tail_invocations),
            operations_by_worker=tuple(
                float(value) for value in tail_invocations_by_worker),
            serial_invocations_by_worker=(0,) * len(groups_by_worker),
        ))
    if not services:
        raise AssertionError(
            "positive CANN AddRMSNorm key-31 groups have no BlockReduce work")

    reduction_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_reduction")
    if len(reduction_demands) != 1:
        raise ValueError(
            "CANN AddRMSNorm key-31 BlockReduce requires one aggregate "
            "reduction demand")
    reduction_demand, = reduction_demands
    if (reduction_demand.operations <= 0.0
            or reduction_demand.service_demands
            or reduction_demand.exact_request_demands):
        raise ValueError(
            "CANN AddRMSNorm key-31 generic reduction contract changed")
    total_operations = math.fsum(
        demand.operations for demand in services)
    typed_reduction = ComputeDemand(
        resource_role="compute_reduction",
        operations=total_operations,
        shape_efficiency=1.0,
        schedule_efficiency=1.0,
        service_demands=tuple(services),
    )
    demand_per_resource = dict(inp.demand_per_resource)
    if demand_per_resource["compute_reduction"] != reduction_demand.operations:
        raise ValueError(
            "CANN AddRMSNorm key-31 aggregate reduction demand disagrees "
            "with resource accounting")
    demand_per_resource["compute_reduction"] = total_operations
    return dataclasses.replace(
        inp,
        demand_per_resource=demand_per_resource,
        compute_demands=tuple(
            typed_reduction if demand is reduction_demand else demand
            for demand in inp.compute_demands),
    )


def _type_cann_norm_rows(
        inp: PredictionInput, rows_by_worker: Tuple[int, ...],
        row_stage_demands, *, include_rmsnorm_local_scalar: bool,
        pipeline_across_rows: bool = False,
        ) -> PredictionInput:
    """Bind repeated CANN NORM source stages to conserved worker services."""

    if not rows_by_worker or any(rows <= 0 for rows in rows_by_worker):
        raise ValueError("CANN NORM typed rows require positive ownership")
    service_order = []
    operations_per_row = {}
    invocations_per_row = {}
    for stage_demands in row_stage_demands.values():
        for demand in stage_demands:
            service = demand.service
            if service not in operations_per_row:
                service_order.append(service)
                operations_per_row[service] = 0.0
                invocations_per_row[service] = 0
            operations_per_row[service] += demand.operations
            invocations_per_row[service] += demand.serial_invocations
    total_rows = sum(rows_by_worker)
    service_demands = tuple(
        ComputeServiceDemand(
            service,
            operations_per_row[service] * total_rows,
            serial_invocations=(
                invocations_per_row[service]
                if pipeline_across_rows else
                invocations_per_row[service] * max(rows_by_worker)),
            operations_by_worker=tuple(
                operations_per_row[service] * rows
                for rows in rows_by_worker),
            serial_invocations_by_worker=tuple(
                invocations_per_row[service]
                if pipeline_across_rows else
                invocations_per_row[service] * rows
                for rows in rows_by_worker),
        )
        for service in service_order
    )
    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError("CANN NORM typed rows require one Vector demand")
    vector_demand, = vector_demands
    typed_operations = math.fsum(
        demand.operations for demand in service_demands)
    if not isinstance(include_rmsnorm_local_scalar, bool):
        raise TypeError("include_rmsnorm_local_scalar must be bool")
    if include_rmsnorm_local_scalar:
        if vector_demand.exact_request_demands:
            raise ValueError(
                "CANN RMSNorm local scalar service cannot replace an "
                "already exact Vector demand")
        scalar_requests = (ComputeExactRequestDemand(
            _CANN_RMSNORM_LOCAL_SCALAR_EXACT_SERVICE,
            tuple((2,) * rows for rows in rows_by_worker),
            pipelined_completion=pipeline_across_rows,
        ),)
    else:
        scalar_requests = ()
    exact_demands = vector_demand.exact_request_demands + scalar_requests
    exact_operations = math.fsum(
        demand.operations for demand in exact_demands)
    if vector_demand.operations != typed_operations + exact_operations:
        raise ValueError(
            "typed CANN NORM services and exact requests must conserve the "
            "aggregate Vector demand")
    typed_vector_demand = dataclasses.replace(
        vector_demand,
        service_demands=service_demands,
        exact_request_demands=exact_demands,
    )
    typed = dataclasses.replace(
        inp,
        compute_demands=tuple(
            typed_vector_demand if demand is vector_demand else demand
            for demand in inp.compute_demands),
    )
    if not include_rmsnorm_local_scalar:
        return typed
    reduction_demands = tuple(
        demand for demand in typed.compute_demands
        if demand.resource_role == "compute_reduction")
    if len(reduction_demands) != 1:
        raise ValueError(
            "CANN RMSNorm exact reduction must replace one generic "
            "reduction demand")
    flat = dict(typed.demand_per_resource)
    if flat["compute_reduction"] != reduction_demands[0].operations:
        raise ValueError(
            "CANN RMSNorm aggregate and structured reduction demands "
            "disagree")
    flat["compute_reduction"] = 0.0
    return dataclasses.replace(
        typed,
        demand_per_resource=flat,
        compute_demands=tuple(
            demand for demand in typed.compute_demands
            if demand.resource_role != "compute_reduction"),
    )


def _cann_rmsnorm_rstd_transfer_bytes(rows_by_worker: Tuple[int, ...]) -> int:
    """Return CANN key-0/1 physical rstd DataCopyCustom payload bytes.

    The public rstd tensor has one FP32 value per row, whereas the fixed CANN
    source issues each AIV-owned extent through a minimum eight-FP32 (32-byte)
    DataCopyCustom transaction.  The row ownership comes from launch, not
    from an observed duration or workload identity.
    """

    if (not rows_by_worker or any(
            isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
            for rows in rows_by_worker)):
        raise ValueError("CANN RMSNorm rstd transfer requires positive rows")
    return sum(
        math.ceil(rows / _CANN_RMSNORM_RSTD_VALUES_PER_TRANSACTION)
        * _CANN_RMSNORM_RSTD_TRANSACTION_BYTES
        for rows in rows_by_worker)


def _cann_rmsnorm_finite_source_floor(
        inp: PredictionInput, hardware, *, rows: int, hidden: int,
        worker_count: int) -> Optional[PipelineWorkPlan]:
    """Return a competing key-0 finite-source mechanism floor.

    The exact service spans the admitted worker set but replaces production
    normalization arithmetic with ``Duplicate``.  It is therefore a lower
    bound on the same source program, not a replacement for the production
    row DAG.  Keeping it as a competing plan preserves every production row,
    queue, reduction and terminal edge while still retaining the independently
    measured launch-wide source-entry envelope.
    """

    domain = (worker_count, rows, hidden)
    if domain not in _CANN_RMS_KEY0_FINITE_SOURCE_ENVELOPE_DOMAINS:
        return None
    completion = _cann_memory_completion_request(
        hardware, "mte3",
        _CANN_RMS_KEY0_FINITE_SOURCE_ENVELOPE_EXACT_SERVICE,
        worker_count)
    if not completion:
        raise ValueError(
            "CANN RMSNorm finite source envelope is unavailable for an "
            f"exact published domain: {domain}")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN RMSNorm finite source envelope must be attached once")
    if set(path.name for path in inp.memory_paths) != {
            "mte2_read", "mte3_write"}:
        raise ValueError(
            "CANN RMSNorm finite source envelope requires split MTE paths")
    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN RMSNorm finite source envelope requires one conserved "
            "Vector demand")
    vector_demand, = vector_demands
    if any(
            len(service.operations_by_worker) != worker_count
            or len(service.serial_invocations_by_worker) != worker_count
            or len(set(service.operations_by_worker)) != 1
            or len(set(service.serial_invocations_by_worker)) != 1
            for service in vector_demand.service_demands):
        raise ValueError(
            "CANN RMSNorm finite source envelope requires uniform typed "
            "Vector ownership")
    if any(
            len(exact.request_sizes_by_worker) != worker_count
            or len(set(exact.request_sizes_by_worker)) != 1
            for exact in vector_demand.exact_request_demands):
        raise ValueError(
            "CANN RMSNorm finite source envelope requires uniform exact "
            "Vector requests")
    worker_vector_services = tuple(
        ComputeServiceDemand(
            service.service,
            service.operations_by_worker[0],
            serial_invocations=service.serial_invocations_by_worker[0],
        )
        for service in vector_demand.service_demands)
    worker_vector_exact = tuple(
        ComputeExactRequestDemand(
            exact.service, (exact.request_sizes_by_worker[0],))
        for exact in vector_demand.exact_request_demands)

    worker_fraction = 1.0 / worker_count
    task = PipelineTaskDemand(
        name="finite_source_envelope_completion",
        worker=0,
        stages=(
            PipelineStageDemand(
                "mte2_capacity", "memory:mte2_read", worker_fraction,
                predecessors=()),
            PipelineStageDemand(
                "finite_source_envelope", "memory:mte3_write",
                worker_fraction,
                memory_exact_request_demands=completion,
                predecessors=()),
            PipelineStageDemand(
                "production_vector", "compute:compute_vector",
                None,
                compute_service_demands=worker_vector_services,
                compute_exact_request_demands=worker_vector_exact,
                predecessors=("finite_source_envelope",)),
        ),
    )
    return PipelineWorkPlan(
        worker_count=worker_count,
        tasks=(PipelineTaskProgramDemand(
            name="finite_source_worker_class",
            worker=0,
            patterns=(PipelineTaskSequenceDemand(
                name="finite_source_sequence",
                worker=0,
                tasks=(task,),
            ),),
            order=(0,),
            worker_multiplicity=worker_count,
            physical_workers=tuple(range(worker_count)),
        ),),
        buffers=(),
    )


def _attach_cann_rms_single_row_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False
        ) -> PredictionInput:
    """Attach the CANN 8.5.1 key-0 one-row-per-AIV source DAG.

    ``KernelRmsNorm<bfloat16_t, bfloat16_t>`` first completes one gamma
    CopyIn, then processes each owned row as CopyIn -> Compute -> CopyOutY.
    After the row loop it issues CopyOutRstd.  In the supported domain every
    launched AIV owns exactly one row, so the fixed BUFFER_NUM=1 queues and
    TBuf lifetimes close the complete worker order without assuming a
    repeated-row schedule.  The scalar C++ reciprocal-sqrt and its two V->S
    plus one S->V handoffs are one measured exact local-scalar dependency
    request per row.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN RMSNorm single-row pipeline requires separated npu_tile "
            "AIVs")
    parameters = kernel.launch.parameters
    if _CANN_RMS_SINGLE_ROW_DAG_PARAMETER not in parameters:
        raise ValueError(
            "CANN RMSNorm key 0 requires its source-bound single-row DAG "
            "identity")
    if parameters[_CANN_RMS_SINGLE_ROW_DAG_PARAMETER] != 1.0:
        raise ValueError(
            "source-bound RMSNorm single-row DAG marker must equal one")
    if (kernel.variant.lower() != "rmsnorm"
            or kernel.dtype != "bf16"
            or kernel.launch.algorithm != "aclnn_rmsnorm"
            or int(parameters["cann_tiling_key"]) != 0):
        raise ValueError(
            "source-bound RMSNorm single-row DAG requires the BF16 CANN "
            "key-0 dispatch")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN RMSNorm single-row pipeline must be attached exactly once")

    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    elements = rows * hidden
    required_parameters = (
        "norm_block_factor", "norm_weight_l2_copies", "rstd_output_bytes",
        "cann_rmsnorm_rstd_transfer_bytes",
        "lowered_vector_input_operations",
        "lowered_vector_output_operations",
    )
    missing = tuple(
        name for name in required_parameters if name not in parameters)
    if missing:
        raise ValueError(
            "CANN RMSNorm single-row launch parameters are incomplete: "
            f"{missing}")
    if (dtype_bytes != 2
            or hidden > 12_288
            or kernel.launch.grid != (rows, 1, 1)
            or kernel.launch.block != (1, 1, 1)
            or inp.active_units != rows
            or int(parameters["norm_block_factor"]) != 1
            or int(parameters["norm_weight_l2_copies"]) != rows
            or float(parameters["rstd_output_bytes"]) != float(rows * 4)
            or float(parameters["cann_rmsnorm_rstd_transfer_bytes"])
            != float(_cann_rmsnorm_rstd_transfer_bytes((1,) * rows))
            or "input_reread_count" in parameters
            or "lowered_vector_reduce_operations" in parameters):
        raise ValueError(
            "CANN RMSNorm single-row launch disagrees with its one-row "
            "BF16 source contract")
    rstd_transfer_bytes = float(
        parameters["cann_rmsnorm_rstd_transfer_bytes"])
    expected_lowered = {
        "lowered_vector_input_operations": float(elements),
        "lowered_vector_output_operations": float(2 * elements),
    }
    mismatched_lowered = {
        name: (parameters[name], expected)
        for name, expected in expected_lowered.items()
        if float(parameters[name]) != expected
    }
    if mismatched_lowered:
        raise ValueError(
            "CANN RMSNorm single-row lowered work disagrees with source: "
            f"{mismatched_lowered}")

    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN RMSNorm single-row requires one compute_vector demand")
    vector_demand = vector_demands[0]
    generic_per_row = float(7 * hidden + 2)
    if (vector_demand.operations != generic_per_row * rows
            or vector_demand.service_demands
            or vector_demand.exact_request_demands):
        raise ValueError(
            "CANN RMSNorm single-row aggregate compute demand disagrees "
            "with source")
    typed_vector_demand = dataclasses.replace(
        vector_demand,
        service_demands=(ComputeServiceDemand(
            "generic", vector_demand.operations,
            operations_by_worker=(generic_per_row,) * rows,
            serial_invocations_by_worker=(0,) * rows,
        ),),
    )
    typed_compute_demands = tuple(
        typed_vector_demand if demand is vector_demand else demand
        for demand in inp.compute_demands)
    inp = dataclasses.replace(inp, compute_demands=typed_compute_demands)
    row_stage_demands = _cann_rmsnorm_row_stage_demands(hardware, hidden)
    inp = _type_cann_norm_rows(
        inp, (1,) * rows, row_stage_demands,
        include_rmsnorm_local_scalar=True)
    path_by_name = {path.name: path for path in inp.memory_paths}
    if set(path_by_name) != {"mte2_read", "mte3_write"}:
        raise ValueError(
            "CANN RMSNorm single-row requires exactly the MTE2-read and "
            "MTE3-write paths")
    actual_path_bytes = {
        path_name: {
            node.resource_role: node.bytes for node in path.nodes
        }
        for path_name, path in path_by_name.items()
    }
    expected_path_bytes = {
        "mte2_read": {
            "mte2": float(4 * rows * hidden),
            "l2_cache": float(4 * rows * hidden),
            "dram": float(2 * rows * hidden + 2 * hidden),
        },
        "mte3_write": {
            "mte3": float(2 * rows * hidden + rstd_transfer_bytes),
            "l2_cache": float(2 * rows * hidden + rstd_transfer_bytes),
            "dram": float(2 * rows * hidden + rstd_transfer_bytes),
        },
    }
    if actual_path_bytes != expected_path_bytes:
        raise ValueError(
            "CANN RMSNorm single-row traffic disagrees with its BF16 source "
            f"contract: actual={actual_path_bytes}, "
            f"expected={expected_path_bytes}")

    finite_source_floor = _cann_rmsnorm_finite_source_floor(
        inp, hardware, rows=rows, hidden=hidden, worker_count=rows)

    # The source-form MTE2 completion requests already contain the gamma/x
    # AllocTensor -> DataCopyPad -> EnQue -> DeQue lifetimes, and the MTE3
    # completion contains the y EnQue -> DeQue -> DataCopyPad -> FreeTensor
    # lifetime.  Retain only the fourth, terminal rstd queue lifetime beside
    # the independently measured TPipe/TQue/TBuf initialization lifecycle.
    scalar_control_demands = (
        ComputeExactRequestDemand(
            _CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE,
            ((1,),) * rows),
        ComputeExactRequestDemand(
            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
            ((1,),) * rows),
    )
    inp = _attach_scalar_aiv_exact_demands(
        inp, scalar_control_demands)

    mte2_service = "memory:mte2_read"
    mte3_service = "memory:mte3_write"
    vector_service = "compute:compute_vector"
    x_bytes = float(2 * hidden)
    gamma_l2_bytes = float(2 * hidden)
    gamma_dram_bytes = float(2 * hidden)
    y_bytes = float(2 * hidden)
    rstd_bytes = rstd_transfer_bytes / rows
    # The exact request is worker-local.  Cross-worker admission is composed
    # once by AIV_WORKER_START_FILL below, while aggregate path arbitration
    # retains shared MTE/L2/DRAM capacity.  Do not substitute a complete
    # multi-worker edge here: it already contains admission and would count
    # that span again in the source pipeline.
    gamma_input_sequence = (
        _cann_memory_completion_request(
            hardware, "mte2",
            _CANN_RMS_KEY0_GAMMA_INPUT_SEQUENCE_EXACT_SERVICE,
            int(x_bytes))
        if rows == 1 else ())
    mte2_to_vector = (
        () if gamma_input_sequence else
        _cann_memory_completion_request(
            hardware, "mte2",
            _CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE,
            int(x_bytes)))
    # Multi-worker edge probes report the maximum completion after all AIVs
    # have entered.  The worker-local completion plus AIV_WORKER_START_FILL
    # reconstructs that quantity for every active population (including the
    # measured 16-worker domain), so binding the launch-wide observation to
    # each worker here would count admission twice.
    first_mte2_to_vector = mte2_to_vector
    mte3_to_vector = _cann_memory_completion_request(
        hardware, "mte3",
        _CANN_DATACOPYPAD_VECTOR_TO_MTE3_EXACT_SERVICE,
        int(y_bytes))
    # CopyOutRstd issues one minimum-granularity 32-byte terminal MTE3
    # request for the logical fp32 result.  The four-byte payload remains the
    # independent capacity demand on the stage below.
    # The exact service is one terminal request per AIV.  Concurrent descriptor
    # admission, byte capacity, and worker-entry fill remain independent
    # envelopes, so the local drain applies to every one-row worker rather
    # than only to a one-worker launch.
    rstd_completion = _cann_memory_completion_request(
        hardware, "mte3", _CANN_MTE3_COMPLETION_EXACT_SERVICE, 32)
    reduction_dependency_fraction = (
        _cann_norm_reduction_dependency_fraction(inp, rows))
    reduction_terminal = (
        "reduction_dependency"
        if reduction_dependency_fraction is not None
        else "input_square_reduce")

    def _memory(path_name, **bytes_by_role):
        return tuple(
            PhaseMemoryDemand(path_name, role, byte_count)
            for role, byte_count in bytes_by_role.items())

    if gamma_input_sequence:
        gamma_producer = "gamma_input_load"
        input_producer = "gamma_input_load"
        source_load_stages = (PipelineStageDemand(
            "gamma_input_load", mte2_service, None,
            memory_node_demands=_memory(
                "mte2_read", mte2=gamma_l2_bytes + x_bytes,
                l2_cache=gamma_l2_bytes + x_bytes),
            launch_shared_memory_node_demands=_memory(
                "mte2_read", dram=gamma_dram_bytes + x_bytes),
            memory_exact_request_demands=gamma_input_sequence,
            memory_exact_request_replaces_resource_capacity=True,
            predecessors=("source_control",)),)
    else:
        gamma_producer = "gamma_load"
        input_producer = "input_load"
        source_load_stages = (
            PipelineStageDemand(
                "gamma_load", mte2_service, None,
                memory_node_demands=_memory(
                    "mte2_read", mte2=gamma_l2_bytes,
                    l2_cache=gamma_l2_bytes),
                launch_shared_memory_node_demands=_memory(
                    "mte2_read", dram=gamma_dram_bytes),
                memory_exact_request_demands=first_mte2_to_vector,
                predecessors=("source_control",)),
            PipelineStageDemand(
                "input_load", mte2_service, None,
                memory_node_demands=_memory(
                    "mte2_read", mte2=x_bytes,
                    l2_cache=x_bytes, dram=x_bytes),
                memory_exact_request_demands=mte2_to_vector,
                predecessors=("gamma_load",)),
        )

    tasks = tuple(
        PipelineTaskDemand(
            name=f"worker{worker}_single_row",
            worker=worker,
            stages=(
                PipelineStageDemand(
                    "source_control", "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE,
                            ((1,),)),
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),)),
                    ),
                    predecessors=()),
            ) + source_load_stages + (
                PipelineStageDemand(
                    "input_square_reduce", vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "input_square_reduce"],
                    predecessors=(input_producer,)),
            ) + ((PipelineStageDemand(
                "reduction_dependency", "compute:compute_reduction",
                reduction_dependency_fraction,
                predecessors=("input_square_reduce",)),)
                 if reduction_dependency_fraction is not None else ()) + (
                PipelineStageDemand(
                    "local_scalar_dependency", vector_service, None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_RMSNORM_LOCAL_SCALAR_EXACT_SERVICE,
                            ((2,),)),),
                    predecessors=(reduction_terminal,)),
                PipelineStageDemand(
                    "output_scale", vector_service, None,
                    compute_service_demands=row_stage_demands["output_scale"],
                    predecessors=("local_scalar_dependency",)),
                PipelineStageDemand(
                    "gamma_apply", vector_service, None,
                    compute_service_demands=row_stage_demands["gamma_apply"],
                    predecessors=(gamma_producer, "output_scale")),
                PipelineStageDemand(
                    "final_cast", vector_service, None,
                    compute_service_demands=row_stage_demands["final_cast"],
                    predecessors=("gamma_apply",)),
                PipelineStageDemand(
                    "output_store", mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=y_bytes, l2_cache=y_bytes,
                        dram=y_bytes),
                    memory_exact_request_demands=mte3_to_vector,
                    predecessors=("final_cast",)),
                PipelineStageDemand(
                    "rstd_store", mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=rstd_bytes,
                        l2_cache=rstd_bytes, dram=rstd_bytes),
                    memory_exact_request_demands=rstd_completion,
                    predecessors=("local_scalar_dependency",),
                    start_predecessors=("output_store",)),
            ),
        )
        for worker in (range(1) if analytical_compact else range(rows))
    )
    if analytical_compact:
        tasks = (PipelineTaskProgramDemand(
            name="single_row_worker_class",
            worker=0,
            patterns=(PipelineTaskSequenceDemand(
                name="single_row_sequence",
                worker=0,
                tasks=tasks,
            ),),
            order=(0,),
            worker_multiplicity=rows,
            physical_workers=tuple(range(rows)),
        ),)
    plan = PipelineWorkPlan(
        worker_count=rows,
        tasks=tasks,
        buffers=(
            PipelineBuffer(
                name="rms_gamma_ub", producer_stage=gamma_producer,
                consumer_stage="gamma_apply", slots=1, scope="worker"),
            PipelineBuffer(
                name="rms_x_fp32_ub", producer_stage=input_producer,
                consumer_stage="output_scale", slots=1, scope="worker"),
            PipelineBuffer(
                name="rms_y_queue", producer_stage="final_cast",
                consumer_stage="output_store", slots=1, scope="worker"),
            PipelineBuffer(
                name="rms_rstd_queue",
                producer_stage="local_scalar_dependency",
                consumer_stage="rstd_store", slots=1, scope="worker"),
        ),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    limitations = (
        ("cann_rmsnorm_vector_edge_completion_evidence_pending",)
        if (not gamma_input_sequence and not mte2_to_vector)
        or not mte3_to_vector else ())
    if not rstd_completion:
        limitations += (
            "cann_rmsnorm_terminal_rstd_completion_evidence_pending",)
    return dataclasses.replace(
        attached,
        competing_pipeline_floor=finite_source_floor,
        # ``rms_norm.cpp`` contains only GET_TILING_DATA followed by this
        # program's ``Init`` and ``Process`` calls.  The typed prologue and
        # worker DAG above account for all three source-visible regions, so
        # the generic nonempty-AIV task can no longer stand in for an
        # unmodelled body.  Retain the independently measured empty task as
        # the common outer shell, exactly as for the source-closed SwiGLU
        # program.  This is a program-composition rule, not a row, hidden,
        # Dataset, or Serving dispatch.
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
        limitations=attached.limitations + limitations,
    )


def _attach_cann_rms_row_group_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False
        ) -> PredictionInput:
    """Attach the CANN 8.5.1 BF16 key-0 repeated-row source program.

    The normal ``KernelRmsNorm`` source loads gamma once per AIV, assigns
    ``blockFactor`` contiguous rows, and holds one rstd tensor across each
    group of at most 64 rows.  Input MTE2 for the next row may start after
    the previous y MTE3 is issued, while reuse of the depth-one y queue waits
    for that store to complete.  The next group waits for the depth-one rstd
    queue.  All work, bytes, and dependencies below are static source facts.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN RMSNorm row-group pipeline requires separated npu_tile "
            "AIVs")
    parameters = kernel.launch.parameters
    if _CANN_RMS_ROW_GROUP_DAG_PARAMETER not in parameters:
        raise ValueError(
            "CANN RMSNorm key 0 requires its source-bound row-group DAG "
            "identity")
    if parameters[_CANN_RMS_ROW_GROUP_DAG_PARAMETER] != 1.0:
        raise ValueError(
            "source-bound RMSNorm row-group DAG marker must equal one")
    if (kernel.variant.lower() != "rmsnorm"
            or kernel.dtype != "bf16"
            or kernel.launch.algorithm != "aclnn_rmsnorm"
            or int(parameters["cann_tiling_key"]) != 0):
        raise ValueError(
            "source-bound RMSNorm row-group DAG requires the BF16 CANN "
            "key-0 dispatch")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN RMSNorm row-group pipeline must be attached exactly once")

    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    elements = rows * hidden
    required_parameters = (
        "norm_block_factor", "norm_weight_l2_copies", "rstd_output_bytes",
        "cann_rmsnorm_rstd_transfer_bytes",
        "norm_epsilon_add_flops", "lowered_vector_input_operations",
        "lowered_vector_output_operations",
    )
    missing = tuple(
        name for name in required_parameters if name not in parameters)
    if missing:
        raise ValueError(
            "CANN RMSNorm row-group launch parameters are incomplete: "
            f"{missing}")
    vector_cores = int(
        hardware.resource_by_role("compute_vector").total_units)
    block_factor = math.ceil(rows / vector_cores)
    worker_count = math.ceil(rows / block_factor)
    rows_by_worker = (
        (block_factor,) * (worker_count - 1)
        + (rows - (worker_count - 1) * block_factor,)
    )
    if (dtype_bytes != 2
            or rows <= vector_cores
            or hidden > 12_288
            or worker_count > vector_cores
            or kernel.launch.grid != (worker_count, 1, 1)
            or kernel.launch.block != (1, 1, 1)
            or inp.active_units != worker_count
            or int(parameters["norm_block_factor"]) != block_factor
            or int(parameters["norm_weight_l2_copies"]) != worker_count
            or float(parameters["rstd_output_bytes"]) != float(rows * 4)
            or float(parameters["cann_rmsnorm_rstd_transfer_bytes"])
            != float(_cann_rmsnorm_rstd_transfer_bytes(rows_by_worker))
            or float(parameters["norm_epsilon_add_flops"]) != float(rows)
            or "input_reread_count" in parameters
            or "lowered_vector_reduce_operations" in parameters
            or any(worker_rows <= 0 or worker_rows > block_factor
                   for worker_rows in rows_by_worker)
            or sum(rows_by_worker) != rows):
        raise ValueError(
            "CANN RMSNorm key-0 launch disagrees with its repeated-row BF16 "
            "source contract")
    rstd_transfer_bytes = float(
        parameters["cann_rmsnorm_rstd_transfer_bytes"])
    expected_lowered = {
        "lowered_vector_input_operations": float(elements),
        "lowered_vector_output_operations": float(2 * elements),
    }
    mismatched_lowered = {
        name: (parameters[name], expected)
        for name, expected in expected_lowered.items()
        if float(parameters[name]) != expected
    }
    if mismatched_lowered:
        raise ValueError(
            "CANN RMSNorm row-group lowered work disagrees with source: "
            f"{mismatched_lowered}")

    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN RMSNorm row-group requires one compute_vector demand")
    vector_demand = vector_demands[0]
    generic_per_row = float(7 * hidden + 2)
    expected_generic_by_worker = tuple(
        generic_per_row * worker_rows for worker_rows in rows_by_worker)
    if (vector_demand.operations != generic_per_row * rows
            or vector_demand.service_demands
            or vector_demand.exact_request_demands):
        raise ValueError(
            "CANN RMSNorm row-group aggregate compute demand disagrees "
            "with source")
    typed_vector_demand = dataclasses.replace(
        vector_demand,
        service_demands=(ComputeServiceDemand(
            "generic", vector_demand.operations,
            operations_by_worker=expected_generic_by_worker,
            serial_invocations_by_worker=(0,) * worker_count,
        ),),
    )
    typed_compute_demands = tuple(
        typed_vector_demand if demand is vector_demand else demand
        for demand in inp.compute_demands)
    inp = dataclasses.replace(inp, compute_demands=typed_compute_demands)
    row_stage_demands = _cann_rmsnorm_row_stage_demands(hardware, hidden)
    inp = _type_cann_norm_rows(
        inp, rows_by_worker, row_stage_demands,
        include_rmsnorm_local_scalar=True,
        pipeline_across_rows=True)
    path_by_name = {path.name: path for path in inp.memory_paths}
    if set(path_by_name) != {"mte2_read", "mte3_write"}:
        raise ValueError(
            "CANN RMSNorm row-group requires exactly the MTE2-read and "
            "MTE3-write paths")
    actual_path_bytes = {
        path_name: {
            node.resource_role: node.bytes for node in path.nodes
        }
        for path_name, path in path_by_name.items()
    }
    expected_path_bytes = {
        "mte2_read": {
            "mte2": float(2 * rows * hidden + 2 * worker_count * hidden),
            "l2_cache": float(2 * rows * hidden + 2 * worker_count * hidden),
            "dram": float(2 * rows * hidden + 2 * hidden),
        },
        "mte3_write": {
            "mte3": float(
                2 * rows * hidden + rstd_transfer_bytes),
            "l2_cache": float(
                2 * rows * hidden + rstd_transfer_bytes),
            "dram": float(
                2 * rows * hidden + rstd_transfer_bytes),
        },
    }
    if actual_path_bytes != expected_path_bytes:
        raise ValueError(
            "CANN RMSNorm row-group traffic disagrees with its BF16 source "
            f"contract: actual={actual_path_bytes}, "
            f"expected={expected_path_bytes}")

    # The finite-source probe measures an admitted *launch* span.  It is a
    # useful mechanism validation of the source program, but cannot be added
    # as a shape-keyed production floor: doing so makes the 32-AIV probe a
    # special case and permits the identical 33-AIV local program to become
    # faster.  The production DAG below instead composes local source edges,
    # the one launch-wide worker-entry fill, and aggregate resource capacity.
    finite_source_floor = None

    queue_cycles_by_worker = tuple(
        (1,) * (
            1 + 2 * worker_rows
            + math.ceil(worker_rows / _CANN_NORM_SPLIT_ROW_FACTOR))
        for worker_rows in rows_by_worker)
    scalar_control_demands = (
        ComputeExactRequestDemand(
            _CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE,
            ((1,),) * worker_count),
        ComputeExactRequestDemand(
            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
            queue_cycles_by_worker,
            pipelined_completion=True),
    )
    inp = _attach_scalar_aiv_exact_demands(
        inp, scalar_control_demands)

    mte2_service = "memory:mte2_read"
    mte3_service = "memory:mte3_write"
    vector_service = "compute:compute_vector"
    x_bytes = float(2 * hidden)
    gamma_l2_bytes = float(2 * hidden)
    gamma_dram_bytes = float(2 * hidden)
    y_bytes = float(2 * hidden)
    # Multi-AIV edge probes report the maximum completion of an admitted
    # population.  The source stage remains one worker-local request; the
    # launch-wide AIV entry fill and aggregate MTE/L2/DRAM capacity are priced
    # separately.  Reusing a population maximum as every worker's request
    # both double-counts admission and creates discontinuities at unmeasured
    # worker counts.
    first_mte2_to_vector = _cann_memory_completion_request(
        hardware, "mte2", _CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE,
        int(x_bytes))
    steady_mte2_to_vector = _cann_memory_completion_request(
        hardware, "mte2", _CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE,
        int(x_bytes))
    steady_mte2_to_vector = tuple(
        dataclasses.replace(demand, pipelined_completion=True)
        for demand in steady_mte2_to_vector)
    # Every depth-one y-queue reuse waits for a worker-local Vector->MTE3
    # completion.  Concurrent-worker cells describe admission of the first
    # request and must not be repeated as a per-row service.  The unloaded
    # source-form completion is the recurrent edge; aggregate MTE3 bytes keep
    # the independent cross-worker capacity constraint.
    mte3_to_vector = _cann_memory_completion_request(
        hardware, "mte3", _CANN_DATACOPYPAD_VECTOR_TO_MTE3_EXACT_SERVICE,
        int(y_bytes))
    mte3_to_vector = tuple(
        dataclasses.replace(demand, pipelined_completion=True)
        for demand in mte3_to_vector)
    first_mte3_to_vector = _cann_memory_completion_request(
        hardware, "mte3", _CANN_DATACOPYPAD_VECTOR_TO_MTE3_EXACT_SERVICE,
        int(y_bytes))
    first_mte3_to_vector = tuple(
        dataclasses.replace(demand, pipelined_completion=True)
        for demand in first_mte3_to_vector)
    reduction_dependency_fraction = (
        _cann_norm_reduction_dependency_fraction(inp, rows))

    def _rstd_completion(group_rows):
        # CopyOutRstd rounds the fp32 row count to one 32-byte data block
        # before issuing DataCopyCustom.  Worker admission is modeled once by
        # AIV_WORKER_START_FILL, so compose the independent worker-local MTE3
        # completion here instead of a multi-worker envelope.
        request_bytes = 32 * math.ceil(group_rows / 8)
        return _cann_memory_completion_request(
            hardware, "mte3", _CANN_MTE3_COMPLETION_EXACT_SERVICE,
            request_bytes)

    def _memory(path_name, **bytes_by_role):
        return tuple(
            PhaseMemoryDemand(path_name, role, byte_count)
            for role, byte_count in bytes_by_role.items())

    def _row_name(base, row_index, group_rows):
        if base in ("input_queue_control", "input_load") and row_index == 0:
            return base
        if base == "output_store" and row_index == group_rows - 1:
            return "output_store"
        return f"row{row_index}_{base}"

    group_task_templates = {}

    def _group_task_template(group_rows, *, include_gamma):
        """Build one immutable source program per exact row-group shape."""

        template_key = (group_rows, include_gamma)
        if template_key in group_task_templates:
            return group_task_templates[template_key]
        if group_rows <= 0 or group_rows > _CANN_NORM_SPLIT_ROW_FACTOR:
            raise ValueError(
                "CANN RMSNorm key-0 group must contain 1..64 rows")
        group_rstd_transfer_bytes = float(
            _cann_rmsnorm_rstd_transfer_bytes((group_rows,)))
        stages = []
        if include_gamma:
            stages.extend((
                PipelineStageDemand(
                    "source_control", "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE,
                            ((1,),)),
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),)),
                    ),
                    predecessors=()),
                PipelineStageDemand(
                    "gamma_load", mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read", mte2=gamma_l2_bytes,
                        l2_cache=gamma_l2_bytes),
                    launch_shared_memory_node_demands=_memory(
                        "mte2_read", dram=gamma_dram_bytes),
                    memory_exact_request_demands=first_mte2_to_vector,
                    predecessors=("source_control",)),
            ))
        prior_output = None
        last_scalar_dependency = None

        def _streamed_demands(stage_name, row_index):
            fill = include_gamma and row_index == 0
            return tuple(
                demand if fill else dataclasses.replace(
                    demand, serial_invocations=0)
                for demand in row_stage_demands[stage_name])

        for row_index in range(group_rows):
            input_name = _row_name("input_load", row_index, group_rows)
            input_queue_name = _row_name(
                "input_queue_control", row_index, group_rows)
            reduce_name = f"row{row_index}_input_square_reduce"
            reduction_dependency_name = (
                f"row{row_index}_reduction_dependency")
            scalar_dependency_name = (
                f"row{row_index}_local_scalar_dependency")
            output_scale_name = f"row{row_index}_output_scale"
            gamma_apply_name = f"row{row_index}_gamma_apply"
            final_cast_name = f"row{row_index}_final_cast"
            output_queue_name = f"row{row_index}_output_queue_control"
            output_store_name = _row_name(
                "output_store", row_index, group_rows)
            if row_index == 0:
                input_queue_predecessors = (
                    ("gamma_load",) if include_gamma else ())
                input_queue_start_predecessors = ()
            else:
                # KernelRmsNorm::SubProcess invokes CopyIn, Compute, and
                # CopyOutY for one row before it begins the next row.  A
                # CopyOutY call releases the source loop once its MTE3
                # request is issued; it does *not* require that transfer to
                # have drained.  Keep the source issue edge separate from
                # the depth-one Y-queue reuse edge below, which does require
                # completion before another row may enqueue its Y result.
                input_queue_predecessors = ()
                input_queue_start_predecessors = (prior_output,)
            stages.extend((
                PipelineStageDemand(
                    input_queue_name, "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),), pipelined_completion=True),),
                    predecessors=input_queue_predecessors,
                    start_predecessors=input_queue_start_predecessors),
                PipelineStageDemand(
                    input_name, mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read", mte2=x_bytes,
                        l2_cache=x_bytes, dram=x_bytes),
                    memory_exact_request_demands=steady_mte2_to_vector,
                    predecessors=(input_queue_name,)),
                PipelineStageDemand(
                    reduce_name, vector_service, None,
                    compute_service_demands=_streamed_demands(
                        "input_square_reduce", row_index),
                    predecessors=(input_name,)),
            ))
            if reduction_dependency_fraction is not None:
                stages.append(PipelineStageDemand(
                    reduction_dependency_name,
                    "compute:compute_reduction",
                    reduction_dependency_fraction,
                    predecessors=(reduce_name,)))
            stages.append(
                PipelineStageDemand(
                    scalar_dependency_name, vector_service, None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_RMSNORM_LOCAL_SCALAR_EXACT_SERVICE,
                            ((2,),), pipelined_completion=True),),
                    predecessors=((reduction_dependency_name,)
                                  if reduction_dependency_fraction is not None
                                  else (reduce_name,))))
            stages.extend((
                PipelineStageDemand(
                    output_scale_name, vector_service, None,
                    compute_service_demands=_streamed_demands(
                        "output_scale", row_index),
                    predecessors=(scalar_dependency_name,)),
                PipelineStageDemand(
                    gamma_apply_name, vector_service, None,
                    compute_service_demands=_streamed_demands(
                        "gamma_apply", row_index),
                    predecessors=(
                        (output_scale_name, prior_output)
                        if prior_output is not None
                        else (output_scale_name,))),
                PipelineStageDemand(
                    final_cast_name, vector_service, None,
                    compute_service_demands=_streamed_demands(
                        "final_cast", row_index),
                    predecessors=(gamma_apply_name,)),
                PipelineStageDemand(
                    output_queue_name, "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),), pipelined_completion=True),),
                    predecessors=(),
                    start_predecessors=(final_cast_name,)),
                PipelineStageDemand(
                    output_store_name, mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=y_bytes, l2_cache=y_bytes,
                        dram=y_bytes),
                    completion_resource_roles=("mte3",),
                    memory_exact_request_demands=(
                        first_mte3_to_vector
                        if include_gamma and row_index == 0
                        else mte3_to_vector),
                    predecessors=(final_cast_name, output_queue_name,)),
            ))
            prior_output = output_store_name
            last_scalar_dependency = scalar_dependency_name
        stages.extend((
            PipelineStageDemand(
                "rstd_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),)),),
                predecessors=(),
                start_predecessors=(last_scalar_dependency, prior_output,)),
            PipelineStageDemand(
                "rstd_store", mte3_service, None,
                memory_node_demands=_memory(
                    "mte3_write", mte3=group_rstd_transfer_bytes,
                    l2_cache=group_rstd_transfer_bytes,
                    dram=group_rstd_transfer_bytes),
                completion_resource_roles=("mte3",),
                memory_exact_request_demands=(
                    _rstd_completion(group_rows)),
                predecessors=(last_scalar_dependency, "rstd_queue_control",)),
        ))
        template = PipelineTaskDemand(
            name="cann_rmsnorm_row_group_template",
            worker=0,
            stages=tuple(stages),
        )
        group_task_templates[template_key] = template
        return template

    def _group_task(worker, group_rows, *, include_gamma, suffix):
        """Bind a verified row-group template to its source worker label."""

        template = _group_task_template(
            group_rows, include_gamma=include_gamma)
        task = object.__new__(PipelineTaskDemand)
        object.__setattr__(task, "name", f"worker{worker}_{suffix}")
        object.__setattr__(task, "worker", worker)
        object.__setattr__(task, "stages", template.stages)
        object.__setattr__(task, "repeat_count", template.repeat_count)
        return task

    def _worker_program_entries(worker, worker_rows):
        first_group_rows = min(
            worker_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
        first_group = _group_task(
            worker, first_group_rows, include_gamma=True,
            suffix="first_group")
        remaining_rows = worker_rows - first_group_rows
        if remaining_rows == 0:
            return (first_group,)
        full_groups, tail_rows = divmod(
            remaining_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
        if full_groups > 0:
            patterns = [PipelineTaskSequenceDemand(
                name=f"worker{worker}_full_group_sequence",
                worker=worker,
                tasks=(_group_task(
                    worker, _CANN_NORM_SPLIT_ROW_FACTOR,
                    include_gamma=False, suffix="full_group"),),
            )]
            tail_order = ()
            if tail_rows > 0:
                patterns.append(PipelineTaskSequenceDemand(
                    name=f"worker{worker}_tail_group_sequence",
                    worker=worker,
                    tasks=(_group_task(
                        worker, tail_rows, include_gamma=False,
                        suffix="tail_group"),),
                ))
                tail_order = (1,)
            return (first_group, PipelineTaskProgramDemand(
                name=f"worker{worker}_remaining_groups",
                worker=worker,
                patterns=tuple(patterns),
                order=(0,),
                repeat_count=full_groups,
                tail_order=tail_order,
            ))
        elif tail_rows > 0:
            return (first_group, _group_task(
                worker, tail_rows, include_gamma=False,
                suffix="tail_group"))
        else:
            raise AssertionError(
                "positive key-0 remaining rows produced no source group")

    if not analytical_compact:
        entries = tuple(
            entry
            for worker, worker_rows in enumerate(rows_by_worker)
            for entry in _worker_program_entries(worker, worker_rows))
    else:
        # Workers with the same row extent execute the same source program.
        # Keep its row-wise queue and rstd recurrence intact, but express the
        # physical AIV population once rather than materializing the program
        # once per worker.
        worker_classes = {}
        for physical_worker, worker_rows in enumerate(rows_by_worker):
            if worker_rows in worker_classes:
                worker_classes[worker_rows].append(physical_worker)
            else:
                worker_classes[worker_rows] = [physical_worker]
        entries = []
        for compact_worker, (worker_rows, physical_workers) in enumerate(
                worker_classes.items()):
            first_group_rows = min(
                worker_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
            tasks = [_group_task(
                compact_worker, first_group_rows, include_gamma=True,
                suffix="first_group")]
            remaining_rows = worker_rows - first_group_rows
            if remaining_rows > 0:
                full_groups, tail_rows = divmod(
                    remaining_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
                if full_groups > 0:
                    tasks.append(dataclasses.replace(
                        _group_task(
                            compact_worker, _CANN_NORM_SPLIT_ROW_FACTOR,
                            include_gamma=False, suffix="full_group"),
                        repeat_count=full_groups))
                if tail_rows > 0:
                    tasks.append(_group_task(
                        compact_worker, tail_rows, include_gamma=False,
                        suffix="tail_group"))
                if full_groups == 0 and tail_rows == 0:
                    raise AssertionError(
                        "positive key-0 remaining rows produced no source "
                        "group")
            entries.append(PipelineTaskProgramDemand(
                name=f"worker_class{compact_worker}_row_groups",
                worker=compact_worker,
                patterns=(PipelineTaskSequenceDemand(
                    name=f"worker_class{compact_worker}_row_sequence",
                    worker=compact_worker,
                    tasks=tuple(tasks)),),
                order=(0,),
                worker_multiplicity=len(physical_workers),
                physical_workers=tuple(physical_workers),
            ))
        entries = tuple(entries)

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=entries,
        buffers=(
            PipelineBuffer(
                name="rms_key0_y_queue",
                producer_stage="input_load",
                consumer_stage="output_store",
                slots=1,
                scope="worker",
            ),
            PipelineBuffer(
                name="rms_key0_rstd_queue",
                # ``SubProcess`` acquires the depth-one rstd tensor before
                # it invokes the first row's ``CopyIn``.  The first input
                # queue control is the immediately following modeled source
                # event, so it is the zero-gap acquisition boundary.  It
                # must not wait until the later MTE2 request begins: that
                # would allow the input queue lifecycle to pass a full rstd
                # queue, contrary to the source order.
                producer_stage="input_queue_control",
                consumer_stage="rstd_store",
                slots=1,
                scope="worker",
            ),
        ),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    limitations = (
        ("cann_rmsnorm_vector_edge_completion_evidence_pending",)
        if not first_mte2_to_vector or not mte3_to_vector else ())
    if not steady_mte2_to_vector:
        limitations += (
            "cann_rmsnorm_steady_mte2_completion_evidence_pending",)
    row_group_sizes = {
        min(_CANN_NORM_SPLIT_ROW_FACTOR, remaining_rows)
        for worker_rows in rows_by_worker
        for remaining_rows in range(
            worker_rows, 0, -_CANN_NORM_SPLIT_ROW_FACTOR)
    }
    if any(not _rstd_completion(group_rows)
           for group_rows in row_group_sizes):
        limitations += (
            "cann_rmsnorm_rstd_mte3_to_vector_completion_evidence_pending",)
    return dataclasses.replace(
        attached,
        competing_pipeline_floor=finite_source_floor,
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
        limitations=attached.limitations + limitations,
    )


def _cann_rmsnorm_merge_n_row_groups(rows_by_worker, row_factor):
    """Return the per-AIV MERGE_N row-group extents.

    ``rms_norm_tiling.cpp:554-557`` derives ``rowLoop`` and ``rowTail`` from
    the AIV's own row extent, and ``KernelRmsNormMergeN::Process`` (``:91-94``)
    runs ``rowLoop - 1`` full groups followed by that tail group.
    """

    if not rows_by_worker or any(rows <= 0 for rows in rows_by_worker):
        raise ValueError("CANN RMSNorm MERGE_N groups require positive rows")
    if row_factor <= 0:
        raise ValueError("CANN RMSNorm MERGE_N row factor must be positive")
    groups = []
    for worker_rows in rows_by_worker:
        row_loop = math.ceil(worker_rows / row_factor)
        tail = worker_rows - (row_loop - 1) * row_factor
        groups.append((row_factor,) * (row_loop - 1) + (tail,))
    return tuple(groups)


def _cann_rmsnorm_merge_n_group_services(hidden, group_rows, performance):
    """Return the source-ordered typed Vector work of one MERGE_N group.

    Every entry is one literal ``rms_norm_merge_n.h`` call: the stage name is
    the source step and the operations are its element extent.  The two
    branches are the ``isNorm``-free ``Compute`` body (``:231-288``) under the
    host tiler's ``is_performance`` flag.
    """

    elements = float(group_rows * hidden)
    rows = float(group_rows)
    services = {
        # :243 Cast(xFp32, xLocal) and :244 FreeTensor(xLocal)
        "input_cast": (ComputeServiceDemand(
            "cast_bf16_to_fp32_serialized", elements, serial_invocations=1),),
        # :246 Mul(sqx, xFp32, xFp32)
        "input_square": (ComputeServiceDemand(
            "mul_serialized", elements, serial_invocations=1),),
        # :252 ReduceSumMultiN over the whole group
        "reduce_request": (ComputeServiceDemand(
            _CANN_MERGE_N_REDUCE_SUM_MULTI_N_EXACT_SERVICE, elements,
            serial_invocations=1),),
        # :254 Muls(avgFactor) and :256 Adds(epsilon); the Sqrt/Duplicate/Div
        # triple at :258-261 is the separately measured request service.
        "normalize": (ComputeServiceDemand(
            "muls_serialized", 2.0 * rows, serial_invocations=2),),
        # :180 Cast(yLocal, sqx, CAST_RINT)
        "final_cast": (ComputeServiceDemand(
            "cast_fp32_to_bf16_serialized", elements, serial_invocations=1),),
    }
    if performance:
        # :268 doPerformanceRstd -> Brcb, :269 repeatByRow(rstd),
        # :165 repeatByRow(gamma) inside ComputeMulGammaCast.
        services["rstd_expand"] = (ComputeServiceDemand(
            _CANN_MERGE_N_PERFORMANCE_RSTD_BRCB_EXACT_SERVICE, rows,
            serial_invocations=1),)
        services["output_scale"] = (ComputeServiceDemand(
            _CANN_MERGE_N_PERFORMANCE_MUL_ROW_RSTD_EXACT_SERVICE, elements,
            serial_invocations=1),)
        services["gamma_apply"] = (ComputeServiceDemand(
            _CANN_MERGE_N_PERFORMANCE_MUL_ROW_GAMMA_EXACT_SERVICE, elements,
            serial_invocations=1),)
    else:
        # :274 BroadCast(sqx, rstdLocal), :282 Mul(sqx, xFp32, sqx),
        # :175 Mul(sqx, sqx, gammaLocal).
        services["rstd_expand"] = (ComputeServiceDemand(
            _CANN_MERGE_N_BROADCAST_RSTD_EXACT_SERVICE, elements,
            serial_invocations=1),)
        services["output_scale"] = (ComputeServiceDemand(
            "mul_serialized", elements, serial_invocations=1),)
        services["gamma_apply"] = (ComputeServiceDemand(
            "mul_serialized", elements, serial_invocations=1),)
    return services


def _attach_cann_rms_merge_n_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False
        ) -> PredictionInput:
    """Attach the CANN 8.5.1 BF16 key-2 MERGE_N row-group source program.

    ``KernelRmsNormMergeN`` loads gamma once per AIV, optionally expands it
    over the whole ``rowFactor`` tile, and then processes its rows in groups
    of ``rowFactor``: one ``DataCopyCustom`` per group instead of one per row,
    one batched ``ReduceSumMultiN`` for the whole group's rstd, and one
    ``DataCopyCustom`` per group on the way out.  The depth-one X, Y and rstd
    queues make the recurrence a group boundary rather than a row boundary.
    All work, bytes and edges below are static source facts.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN RMSNorm MERGE_N pipeline requires separated npu_tile AIVs")
    parameters = kernel.launch.parameters
    if parameters[_CANN_RMS_MERGE_N_DAG_PARAMETER] != 1.0:
        raise ValueError(
            "source-bound RMSNorm MERGE_N DAG marker must equal one")
    if (kernel.variant.lower() != "rmsnorm"
            or kernel.dtype != "bf16"
            or kernel.launch.algorithm != "aclnn_rmsnorm"
            or int(parameters["cann_tiling_key"]) != 2):
        raise ValueError(
            "source-bound RMSNorm MERGE_N DAG requires the BF16 CANN key-2 "
            "dispatch")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN RMSNorm MERGE_N pipeline must be attached exactly once")
    if not isinstance(analytical_compact, bool):
        raise TypeError(
            "CANN RMSNorm MERGE_N analytical_compact must be boolean")

    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    elements = rows * hidden
    row_factor = int(parameters["norm_merge_n_row_factor"])
    performance = int(parameters["norm_merge_n_performance_flag"])
    num_col_align = (
        math.ceil(hidden / _CANN_MERGE_N_BF16_ELEMENTS_PER_BLOCK)
        * _CANN_MERGE_N_BF16_ELEMENTS_PER_BLOCK)
    if num_col_align != hidden:
        raise NotImplementedError(
            "CANN RMSNorm MERGE_N is modelled only for block-aligned column "
            "counts; the padded DataCopyPad branch is unsupported")
    worker_count = kernel.launch.grid_size
    block_factor = int(parameters["norm_block_factor"])
    rows_by_worker = _rows_by_worker(rows, worker_count, block_factor)
    groups_by_worker = _cann_rmsnorm_merge_n_row_groups(
        rows_by_worker, row_factor)
    rstd_transfer_bytes = float(sum(
        _cann_rmsnorm_rstd_transfer_bytes((group_rows,))
        for worker_groups in groups_by_worker
        for group_rows in worker_groups))
    if (dtype_bytes != 2
            or performance not in (0, 1)
            or kernel.launch.grid != (worker_count, 1, 1)
            or kernel.launch.block != (1, 1, 1)
            or inp.active_units != worker_count
            or block_factor != math.ceil(rows / worker_count)
            or int(parameters["norm_weight_l2_copies"]) != worker_count
            or float(parameters["rstd_output_bytes"]) != float(rows * 4)
            or float(parameters["cann_rmsnorm_rstd_transfer_bytes"])
            != rstd_transfer_bytes
            or "input_reread_count" in parameters
            or any(worker_rows <= 0 or worker_rows > block_factor
                   for worker_rows in rows_by_worker)
            or sum(rows_by_worker) != rows
            or any(sum(worker_groups) != worker_rows
                   for worker_groups, worker_rows
                   in zip(groups_by_worker, rows_by_worker))
            or any(group_rows <= 0 or group_rows > row_factor
                   for worker_groups in groups_by_worker
                   for group_rows in worker_groups)):
        raise ValueError(
            "CANN RMSNorm key-2 launch disagrees with its MERGE_N BF16 "
            "source contract")

    gamma_operations = worker_count * hidden
    gamma_broadcast_operations = (
        0 if performance else worker_count * row_factor * num_col_align)
    expected_lowered = {
        "lowered_vector_input_operations": float(
            elements + gamma_operations + gamma_broadcast_operations),
        "lowered_vector_reduce_operations": float(
            (1 if not performance else 2) * rows),
        "lowered_vector_output_operations": float(
            (2 if not performance else 1) * elements),
    }
    mismatched_lowered = {
        name: (parameters[name], expected)
        for name, expected in expected_lowered.items()
        if float(parameters[name]) != expected
    }
    if mismatched_lowered:
        raise ValueError(
            "CANN RMSNorm MERGE_N lowered work disagrees with source: "
            f"{mismatched_lowered}")

    missing_services = tuple(
        service for service in (
            _CANN_MERGE_N_REDUCE_SUM_MULTI_N_EXACT_SERVICE,
            _CANN_MERGE_N_BROADCAST_GAMMA_EXACT_SERVICE,
            _CANN_MERGE_N_BROADCAST_RSTD_EXACT_SERVICE,
            _CANN_MERGE_N_PERFORMANCE_RSTD_BRCB_EXACT_SERVICE,
            _CANN_MERGE_N_PERFORMANCE_MUL_ROW_RSTD_EXACT_SERVICE,
            _CANN_MERGE_N_PERFORMANCE_MUL_ROW_GAMMA_EXACT_SERVICE,
        )
        if not _has_compute_service(hardware, "compute_vector", service))
    if missing_services:
        raise ValueError(
            "CANN RMSNorm MERGE_N typed services are unavailable: "
            f"{missing_services}")

    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN RMSNorm MERGE_N requires one compute_vector demand")
    vector_demand, = vector_demands
    if vector_demand.service_demands or vector_demand.exact_request_demands:
        raise ValueError(
            "CANN RMSNorm MERGE_N cannot replace an already typed Vector "
            "demand")

    group_services = {}

    def _group_services(group_rows):
        if group_rows not in group_services:
            group_services[group_rows] = (
                _cann_rmsnorm_merge_n_group_services(
                    hidden, group_rows, performance))
        return group_services[group_rows]

    # One launch prologue per AIV: CopyInGamma casts the gamma vector
    # (``:135-143``) and, outside the is_performance branch, BroadCastGamma
    # replicates it across the whole rowFactor tile (``:88-90``).
    def _prologue_services():
        prologue = {
            "gamma_cast": (ComputeServiceDemand(
                "cast_bf16_to_fp32_serialized", float(hidden),
                serial_invocations=1),),
        }
        if not performance:
            prologue["gamma_broadcast"] = (ComputeServiceDemand(
                _CANN_MERGE_N_BROADCAST_GAMMA_EXACT_SERVICE,
                float(row_factor * num_col_align), serial_invocations=1),)
        return prologue

    services_in_order = []
    for stage_demands in _prologue_services().values():
        for demand in stage_demands:
            if demand.service not in services_in_order:
                services_in_order.append(demand.service)
    for stage_demands in _group_services(row_factor).values():
        for demand in stage_demands:
            if demand.service not in services_in_order:
                services_in_order.append(demand.service)
    operations_by_service_worker = {
        service: [0.0] * worker_count for service in services_in_order}
    invocations_by_service_worker = {
        service: [0] * worker_count for service in services_in_order}

    def _accumulate(worker, stage_demands_by_name):
        for stage_demands in stage_demands_by_name.values():
            for demand in stage_demands:
                if demand.service not in operations_by_service_worker:
                    raise AssertionError(
                        "CANN RMSNorm MERGE_N service inventory is "
                        f"incomplete: {demand.service!r}")
                operations_by_service_worker[demand.service][worker] += (
                    demand.operations)
                invocations_by_service_worker[demand.service][worker] += (
                    demand.serial_invocations)

    for worker, worker_groups in enumerate(groups_by_worker):
        _accumulate(worker, _prologue_services())
        for group_rows in worker_groups:
            _accumulate(worker, _group_services(group_rows))

    service_demands = tuple(
        ComputeServiceDemand(
            service,
            math.fsum(operations_by_service_worker[service]),
            operations_by_worker=tuple(operations_by_service_worker[service]),
            serial_invocations=max(invocations_by_service_worker[service]),
            serial_invocations_by_worker=tuple(
                invocations_by_service_worker[service]),
        )
        for service in services_in_order)
    normalize_requests = ComputeExactRequestDemand(
        _CANN_NORM_EXACT_SERVICE,
        tuple(tuple(worker_groups) for worker_groups in groups_by_worker),
    )
    typed_operations = math.fsum(
        demand.operations for demand in service_demands)
    if (typed_operations + normalize_requests.operations
            != vector_demand.operations):
        raise ValueError(
            "typed CANN RMSNorm MERGE_N services must conserve the aggregate "
            f"Vector demand: typed={typed_operations}, "
            f"requests={normalize_requests.operations}, "
            f"aggregate={vector_demand.operations}")
    typed_vector_demand = dataclasses.replace(
        vector_demand,
        service_demands=service_demands,
        exact_request_demands=(normalize_requests,),
    )
    inp = dataclasses.replace(
        inp,
        compute_demands=tuple(
            typed_vector_demand if demand is vector_demand else demand
            for demand in inp.compute_demands),
    )
    # ``ReduceSumMultiN`` is the group's whole reduction and is priced by its
    # own measured request service above, so the separate aggregate
    # ReduceSum-over-ideal-add excess must not also be charged, exactly as
    # the key-0 programs drop it.
    reduction_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_reduction")
    if len(reduction_demands) != 1:
        raise ValueError(
            "CANN RMSNorm MERGE_N must replace one generic reduction demand")
    flat = dict(inp.demand_per_resource)
    if flat["compute_reduction"] != reduction_demands[0].operations:
        raise ValueError(
            "CANN RMSNorm MERGE_N aggregate and structured reduction "
            "demands disagree")
    flat["compute_reduction"] = 0.0
    inp = dataclasses.replace(
        inp,
        demand_per_resource=flat,
        compute_demands=tuple(
            demand for demand in inp.compute_demands
            if demand.resource_role != "compute_reduction"),
    )

    # ``Process`` builds the four depth-one queues once per AIV and then
    # issues one AllocTensor/EnQue/DeQue/FreeTensor cycle for the X, rstd and
    # Y queues of every row group.
    queue_cycles_by_worker = tuple(
        (1,) * (1 + 3 * len(worker_groups))
        for worker_groups in groups_by_worker)
    inp = _attach_scalar_aiv_exact_demands(inp, (
        ComputeExactRequestDemand(
            _CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE,
            ((1,),) * worker_count),
        ComputeExactRequestDemand(
            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
            queue_cycles_by_worker,
            pipelined_completion=True),
    ))

    path_by_name = {path.name: path for path in inp.memory_paths}
    if set(path_by_name) != {"mte2_read", "mte3_write"}:
        raise ValueError(
            "CANN RMSNorm MERGE_N requires exactly the MTE2-read and "
            "MTE3-write paths")
    actual_path_bytes = {
        path_name: {node.resource_role: node.bytes for node in path.nodes}
        for path_name, path in path_by_name.items()
    }
    expected_path_bytes = {
        "mte2_read": {
            "mte2": float(2 * elements + 2 * worker_count * hidden),
            "l2_cache": float(2 * elements + 2 * worker_count * hidden),
            "dram": float(2 * elements + 2 * hidden),
        },
        "mte3_write": {
            "mte3": float(2 * elements + rstd_transfer_bytes),
            "l2_cache": float(2 * elements + rstd_transfer_bytes),
            "dram": float(2 * elements + rstd_transfer_bytes),
        },
    }
    if actual_path_bytes != expected_path_bytes:
        raise ValueError(
            "CANN RMSNorm MERGE_N traffic disagrees with its BF16 source "
            f"contract: actual={actual_path_bytes}, "
            f"expected={expected_path_bytes}")

    mte2_service = "memory:mte2_read"
    mte3_service = "memory:mte3_write"
    vector_service = "compute:compute_vector"
    gamma_bytes = float(hidden * dtype_bytes)

    def _memory(path_name, **bytes_by_role):
        return tuple(
            PhaseMemoryDemand(path_name, role, byte_count)
            for role, byte_count in bytes_by_role.items())

    group_task_templates = {}

    def _group_task_template(group_rows, *, include_gamma):
        template_key = (group_rows, include_gamma)
        if template_key in group_task_templates:
            return group_task_templates[template_key]
        services = _group_services(group_rows)
        group_bytes = float(group_rows * hidden * dtype_bytes)
        group_rstd_bytes = float(
            _cann_rmsnorm_rstd_transfer_bytes((group_rows,)))
        stages = []
        if include_gamma:
            prologue = _prologue_services()
            stages.extend((
                PipelineStageDemand(
                    "source_control", "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE,
                            ((1,),)),
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),)),
                    ),
                    predecessors=()),
                PipelineStageDemand(
                    "gamma_load", mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read", mte2=gamma_bytes, l2_cache=gamma_bytes),
                    launch_shared_memory_node_demands=_memory(
                        "mte2_read", dram=gamma_bytes),
                    predecessors=("source_control",)),
                PipelineStageDemand(
                    "gamma_cast", vector_service, None,
                    compute_service_demands=prologue["gamma_cast"],
                    predecessors=("gamma_load",)),
            ))
            entry_stage = "gamma_cast"
            if not performance:
                stages.append(PipelineStageDemand(
                    "gamma_broadcast", vector_service, None,
                    compute_service_demands=prologue["gamma_broadcast"],
                    predecessors=("gamma_cast",)))
                entry_stage = "gamma_broadcast"
            input_queue_predecessors = (entry_stage,)
        else:
            input_queue_predecessors = ()
        stages.extend((
            PipelineStageDemand(
                "input_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),), pipelined_completion=True),),
                predecessors=input_queue_predecessors),
            PipelineStageDemand(
                "input_load", mte2_service, None,
                memory_node_demands=_memory(
                    "mte2_read", mte2=group_bytes, l2_cache=group_bytes,
                    dram=group_bytes),
                predecessors=("input_queue_control",)),
            PipelineStageDemand(
                "input_cast", vector_service, None,
                compute_service_demands=services["input_cast"],
                predecessors=("input_load",)),
            PipelineStageDemand(
                "input_square", vector_service, None,
                compute_service_demands=services["input_square"],
                predecessors=("input_cast",)),
            PipelineStageDemand(
                "reduce_request", vector_service, None,
                compute_service_demands=services["reduce_request"],
                predecessors=("input_square",)),
            PipelineStageDemand(
                "normalize", vector_service, None,
                compute_service_demands=services["normalize"],
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_NORM_EXACT_SERVICE, ((group_rows,),)),),
                predecessors=("reduce_request",)),
            PipelineStageDemand(
                "rstd_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),), pipelined_completion=True),),
                predecessors=(),
                start_predecessors=("normalize",)),
            PipelineStageDemand(
                "rstd_store", mte3_service, None,
                memory_node_demands=_memory(
                    "mte3_write", mte3=group_rstd_bytes,
                    l2_cache=group_rstd_bytes, dram=group_rstd_bytes),
                completion_resource_roles=("mte3",),
                predecessors=("normalize", "rstd_queue_control")),
            PipelineStageDemand(
                "rstd_expand", vector_service, None,
                compute_service_demands=services["rstd_expand"],
                predecessors=("normalize",)),
            PipelineStageDemand(
                "output_scale", vector_service, None,
                compute_service_demands=services["output_scale"],
                predecessors=("rstd_expand",)),
            PipelineStageDemand(
                "gamma_apply", vector_service, None,
                compute_service_demands=services["gamma_apply"],
                predecessors=("output_scale",)),
            PipelineStageDemand(
                "final_cast", vector_service, None,
                compute_service_demands=services["final_cast"],
                predecessors=("gamma_apply",)),
            PipelineStageDemand(
                "output_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),), pipelined_completion=True),),
                predecessors=(),
                start_predecessors=("final_cast",)),
            PipelineStageDemand(
                "output_store", mte3_service, None,
                memory_node_demands=_memory(
                    "mte3_write", mte3=group_bytes, l2_cache=group_bytes,
                    dram=group_bytes),
                completion_resource_roles=("mte3",),
                predecessors=("final_cast", "output_queue_control")),
        ))
        template = PipelineTaskDemand(
            name="cann_rmsnorm_merge_n_template",
            worker=0,
            stages=tuple(stages),
        )
        group_task_templates[template_key] = template
        return template

    def _group_task(worker, group_rows, *, include_gamma, suffix):
        template = _group_task_template(
            group_rows, include_gamma=include_gamma)
        task = object.__new__(PipelineTaskDemand)
        object.__setattr__(task, "name", f"worker{worker}_{suffix}")
        object.__setattr__(task, "worker", worker)
        object.__setattr__(task, "stages", template.stages)
        object.__setattr__(task, "repeat_count", template.repeat_count)
        return task

    def _worker_tasks(worker, worker_groups):
        tasks = [_group_task(
            worker, worker_groups[0], include_gamma=True,
            suffix="first_group")]
        remaining = worker_groups[1:]
        full_groups = sum(
            1 for group_rows in remaining if group_rows == row_factor)
        if full_groups:
            tasks.append(dataclasses.replace(
                _group_task(
                    worker, row_factor, include_gamma=False,
                    suffix="full_group"),
                repeat_count=full_groups))
        tail = tuple(
            group_rows for group_rows in remaining
            if group_rows != row_factor)
        if len(tail) > 1:
            raise AssertionError(
                "a MERGE_N worker owns at most one partial row group")
        if tail:
            tasks.append(_group_task(
                worker, tail[0], include_gamma=False, suffix="tail_group"))
        return tuple(tasks)

    if not analytical_compact:
        entries = tuple(
            PipelineTaskProgramDemand(
                name=f"worker{worker}_row_groups",
                worker=worker,
                patterns=(PipelineTaskSequenceDemand(
                    name=f"worker{worker}_row_sequence",
                    worker=worker,
                    tasks=_worker_tasks(worker, worker_groups)),),
                order=(0,),
            )
            for worker, worker_groups in enumerate(groups_by_worker))
    else:
        # Every AIV with the same row extent executes the same source
        # program.  Express each distinct program once and retain the exact
        # physical-worker population.
        worker_classes = {}
        for physical_worker, worker_groups in enumerate(groups_by_worker):
            if worker_groups in worker_classes:
                worker_classes[worker_groups].append(physical_worker)
            else:
                worker_classes[worker_groups] = [physical_worker]
        entries = tuple(
            PipelineTaskProgramDemand(
                name=f"worker_class{compact_worker}_merge_n_groups",
                worker=compact_worker,
                patterns=(PipelineTaskSequenceDemand(
                    name=f"worker_class{compact_worker}_merge_n_sequence",
                    worker=compact_worker,
                    tasks=_worker_tasks(compact_worker, worker_groups)),),
                order=(0,),
                worker_multiplicity=len(physical_workers),
                physical_workers=tuple(physical_workers),
            )
            for compact_worker, (worker_groups, physical_workers)
            in enumerate(worker_classes.items()))

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=entries,
        buffers=(
            # ``CopyIn`` allocates the depth-one X tensor (``:114``) and
            # ``Compute`` frees it immediately after the input Cast
            # (``:244``), so the next group's MTE2 waits only on that cast.
            PipelineBuffer(
                name="rms_merge_n_x_queue",
                producer_stage="input_load",
                consumer_stage="input_cast",
                slots=1,
                scope="worker",
            ),
            # ``ComputeMulGammaCast`` allocates the depth-one Y tensor
            # (``:161``) and ``CopyOutY`` frees it after the store drains
            # (``:304``).
            PipelineBuffer(
                name="rms_merge_n_y_queue",
                producer_stage="gamma_apply",
                consumer_stage="output_store",
                slots=1,
                scope="worker",
            ),
            # ``Compute`` allocates the depth-one rstd tensor (``:251``) and
            # ``CopyOutRstd`` frees it after its store drains (``:313``).
            PipelineBuffer(
                name="rms_merge_n_rstd_queue",
                producer_stage="reduce_request",
                consumer_stage="rstd_store",
                slots=1,
                scope="worker",
            ),
        ),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    return dataclasses.replace(
        attached,
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
    )


def _attach_cann_rms_split_d_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False
        ) -> PredictionInput:
    """Attach the CANN 8.5.1 BF16 key-1 Split-D worker program.

    Each AIV processes at most 64 rows per group.  The source completes the
    first input/reduction pass for the group, normalizes the row reductions,
    then reloads x and gamma for the output pass before publishing y and
    rstd.  This preserves that source order using only static work and traffic
    already present in the aggregate contract.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN RMSNorm Split-D pipeline requires separated npu_tile AIVs")
    parameters = kernel.launch.parameters
    if parameters[_CANN_RMS_SPLIT_D_DAG_PARAMETER] != 1.0:
        raise ValueError(
            "source-bound RMSNorm Split-D DAG marker must equal one")
    if (kernel.variant.lower() != "rmsnorm"
            or kernel.dtype != "bf16"
            or kernel.launch.algorithm != "aclnn_rmsnorm"
            or int(parameters["cann_tiling_key"]) != 1):
        raise ValueError(
            "source-bound RMSNorm Split-D DAG requires the BF16 CANN key-1 "
            "dispatch")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN RMSNorm Split-D pipeline must be attached exactly once")
    if not isinstance(analytical_compact, bool):
        raise TypeError(
            "CANN RMSNorm Split-D analytical_compact must be boolean")

    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    worker_count = kernel.launch.grid_size
    block_factor = int(parameters["norm_block_factor"])
    rows_by_worker = _rows_by_worker(rows, worker_count, block_factor)
    groups_by_worker = tuple(
        math.ceil(worker_rows / _CANN_NORM_SPLIT_ROW_FACTOR)
        for worker_rows in rows_by_worker)
    if "cann_rmsnorm_rstd_transfer_bytes" not in parameters:
        raise ValueError(
            "CANN RMSNorm Split-D requires rstd transfer geometry")
    if (dtype_bytes != 2
            or hidden <= 12_288
            or inp.active_units != worker_count
            or any(worker_rows <= 0 for worker_rows in rows_by_worker)
            or sum(rows_by_worker) != rows
            or int(parameters["input_reread_count"]) != 1
            or int(parameters["norm_weight_l2_copies"])
            != sum(groups_by_worker)
            or float(parameters["cann_rmsnorm_rstd_transfer_bytes"])
            != float(_cann_rmsnorm_rstd_transfer_bytes(rows_by_worker))):
        raise ValueError(
            "CANN RMSNorm Split-D launch disagrees with its BF16 source "
            "contract")
    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN RMSNorm Split-D requires one compute_vector demand")
    vector_demand, = vector_demands
    if (len(vector_demand.service_demands) != 1
            or vector_demand.service_demands[0].service != "generic"
            or len(vector_demand.exact_request_demands) != 1
            or vector_demand.exact_request_demands[0].service
            != _CANN_NORM_EXACT_SERVICE):
        raise ValueError(
            "CANN RMSNorm Split-D compute demand is not source-typed")
    generic, = vector_demand.service_demands
    exact, = vector_demand.exact_request_demands
    expected_generic_by_worker = tuple(
        float((8 * hidden + 1) * worker_rows)
        for worker_rows in rows_by_worker)
    expected_exact_by_worker = tuple(
        ((_CANN_NORM_SPLIT_ROW_FACTOR,)
         * (worker_rows // _CANN_NORM_SPLIT_ROW_FACTOR)
         + ((worker_rows % _CANN_NORM_SPLIT_ROW_FACTOR,)
            if worker_rows % _CANN_NORM_SPLIT_ROW_FACTOR else ()))
        for worker_rows in rows_by_worker)
    if (generic.operations_by_worker != expected_generic_by_worker
            or exact.request_sizes_by_worker != expected_exact_by_worker):
        raise ValueError(
            "CANN RMSNorm Split-D worker compute ownership disagrees with "
            "source")

    path_by_name = {path.name: path for path in inp.memory_paths}
    if set(path_by_name) != {"mte2_read", "mte3_write"}:
        raise ValueError(
            "CANN RMSNorm Split-D requires exactly MTE2-read and MTE3-write "
            "paths")
    x_bytes = float(hidden * dtype_bytes)
    gamma_bytes = float(hidden * dtype_bytes)
    mte2_service = "memory:mte2_read"
    mte3_service = "memory:mte3_write"
    vector_service = "compute:compute_vector"

    def _memory(path_name, **bytes_by_role):
        return tuple(
            PhaseMemoryDemand(path_name, role, byte_count)
            for role, byte_count in bytes_by_role.items())

    def _group_task(worker, group_rows, *, include_gamma_dram, suffix):
        if group_rows <= 0 or group_rows > _CANN_NORM_SPLIT_ROW_FACTOR:
            raise ValueError(
                "CANN RMSNorm Split-D group must contain 1..64 rows")
        group_rstd_transfer_bytes = float(
            _cann_rmsnorm_rstd_transfer_bytes((group_rows,)))
        reduce_operations = float(3 * hidden * group_rows)
        normalize_operations = float(group_rows)
        output_operations = float(5 * hidden * group_rows)
        stages = (
                PipelineStageDemand(
                    "load_reduce", mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read", mte2=x_bytes * group_rows,
                        l2_cache=x_bytes * group_rows,
                        dram=x_bytes * group_rows)),
                PipelineStageDemand(
                    "reduce_compute", vector_service, None,
                    compute_service_demands=(ComputeServiceDemand(
                        "generic", reduce_operations),),
                    predecessors=("load_reduce",)),
                PipelineStageDemand(
                    "normalize_input", vector_service, None,
                    compute_service_demands=(ComputeServiceDemand(
                        "generic", normalize_operations),),
                    predecessors=("reduce_compute",)),
                PipelineStageDemand(
                    "normalize_request", vector_service, None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_NORM_EXACT_SERVICE, ((group_rows,),)),),
                    predecessors=("normalize_input",)),
        )
        if include_gamma_dram:
            stages += (PipelineStageDemand(
                "gamma_dram_load", mte2_service, None,
                launch_shared_memory_node_demands=_memory(
                    "mte2_read", dram=gamma_bytes),
                predecessors=("normalize_request",)),)
            load_output_predecessors = ("gamma_dram_load",)
        else:
            load_output_predecessors = ("normalize_request",)
        stages += (
                PipelineStageDemand(
                    "load_output", mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read",
                        mte2=x_bytes * group_rows + gamma_bytes,
                        l2_cache=x_bytes * group_rows + gamma_bytes),
                    predecessors=load_output_predecessors),
                PipelineStageDemand(
                    "output_compute", vector_service, None,
                    compute_service_demands=(ComputeServiceDemand(
                        "generic", output_operations),),
                    predecessors=("load_output",)),
                PipelineStageDemand(
                    "output_store", mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=x_bytes * group_rows,
                        l2_cache=x_bytes * group_rows,
                        dram=x_bytes * group_rows),
                    predecessors=("output_compute",)),
                PipelineStageDemand(
                    "rstd_store", mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=group_rstd_transfer_bytes,
                        l2_cache=group_rstd_transfer_bytes,
                        dram=group_rstd_transfer_bytes),
                    predecessors=("output_store",)),
        )
        return PipelineTaskDemand(
            name=f"worker{worker}_{suffix}",
            worker=worker,
            stages=stages,
        )

    def _worker_program_entries(worker, worker_rows):
        first_group_rows = min(worker_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
        first_group = _group_task(
            worker, first_group_rows, include_gamma_dram=True,
            suffix="first_group")
        remaining_rows = worker_rows - first_group_rows
        if remaining_rows == 0:
            return (first_group,)
        full_groups, tail_rows = divmod(
            remaining_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
        if full_groups:
            patterns = [PipelineTaskSequenceDemand(
                name=f"worker{worker}_full_group_sequence",
                worker=worker,
                tasks=(_group_task(
                    worker, _CANN_NORM_SPLIT_ROW_FACTOR,
                    include_gamma_dram=False, suffix="full_group"),),
            )]
            tail_order = ()
            if tail_rows:
                patterns.append(PipelineTaskSequenceDemand(
                    name=f"worker{worker}_tail_group_sequence",
                    worker=worker,
                    tasks=(_group_task(
                        worker, tail_rows, include_gamma_dram=False,
                        suffix="tail_group"),),
                ))
                tail_order = (1,)
            return (first_group, PipelineTaskProgramDemand(
                name=f"worker{worker}_remaining_groups",
                worker=worker,
                patterns=tuple(patterns),
                order=(0,),
                repeat_count=full_groups,
                tail_order=tail_order,
            ))
        elif tail_rows:
            return (first_group, _group_task(
                worker, tail_rows, include_gamma_dram=False,
                suffix="tail_group"))
        else:
            raise AssertionError(
                "positive Split-D remaining rows produced no source group")

    if not analytical_compact:
        entries = tuple(
            entry
            for worker, worker_rows in enumerate(rows_by_worker)
            for entry in _worker_program_entries(worker, worker_rows))
    else:
        # CANN assigns the same row program to every AIV with the same row
        # extent.  Represent each distinct program once and retain the exact
        # physical-worker population instead of expanding one DAG per AIV.
        worker_classes = {}
        for physical_worker, worker_rows in enumerate(rows_by_worker):
            if worker_rows in worker_classes:
                worker_classes[worker_rows].append(physical_worker)
            else:
                worker_classes[worker_rows] = [physical_worker]
        compact_entries = []
        for compact_worker, (worker_rows, physical_workers) in enumerate(
                worker_classes.items()):
            first_group_rows = min(
                worker_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
            compact_tasks = [_group_task(
                compact_worker, first_group_rows,
                include_gamma_dram=True, suffix="first_group")]
            remaining_rows = worker_rows - first_group_rows
            full_groups, tail_rows = divmod(
                remaining_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
            if full_groups:
                compact_tasks.append(dataclasses.replace(
                    _group_task(
                        compact_worker, _CANN_NORM_SPLIT_ROW_FACTOR,
                        include_gamma_dram=False, suffix="full_group"),
                    repeat_count=full_groups,
                ))
            if tail_rows:
                compact_tasks.append(_group_task(
                    compact_worker, tail_rows,
                    include_gamma_dram=False, suffix="tail_group"))
            compact_entries.append(PipelineTaskProgramDemand(
                name=f"worker_class{compact_worker}_split_d_groups",
                worker=compact_worker,
                patterns=(PipelineTaskSequenceDemand(
                    name=f"worker_class{compact_worker}_split_d_sequence",
                    worker=compact_worker,
                    tasks=tuple(compact_tasks)),),
                order=(0,),
                worker_multiplicity=len(physical_workers),
                physical_workers=tuple(physical_workers),
            ))
        entries = tuple(compact_entries)

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=entries,
        buffers=(),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    return dataclasses.replace(
        attached,
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
    )


def _attach_cann_add_rms_key31_completion_envelope(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False,
        ) -> PredictionInput:
    """Attach the fixed CANN 8.5.1 key-31 Split-D source program.

    ``AddRmsNormSplitD`` walks 8,192-wide column tiles outside its row loop.
    In its Former path, ``inQueueX.FreeTensor`` follows the BF16 cast/add
    body and precedes both residual CopyOut and ``ComputeSum``.  The enclosing
    source loop then issues ``ComputeSum`` before it invokes the next Former
    CopyIn, so its recurrence releases at the reduction *issue*, not its
    completion.  Conversely, the depth-one Former ``outQueueY`` is allocated
    at the beginning of the same Vector body and cannot be reused until its
    residual MTE3 CopyOut completes.  After all former tiles reduce to rstd,
    every latter tile loads gamma and walks its rows.  The latter input CopyIn
    may start once the preceding CopyOut is issued, while its VECOUT allocation
    waits for that CopyOut's MTE3 completion.  These are source-order edges,
    not completion-duration observations.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN AddRMSNorm key-31 requires separated npu_tile AIVs")
    if not isinstance(analytical_compact, bool):
        raise TypeError("CANN AddRMSNorm key-31 analytical_compact must be bool")
    parameters = kernel.launch.parameters
    if _CANN_ADD_RMS_SPLIT_D_DAG_PARAMETER not in parameters:
        raise ValueError(
            "CANN AddRMSNorm key 31 requires its source-bound Split-D DAG "
            "identity")
    if parameters[_CANN_ADD_RMS_SPLIT_D_DAG_PARAMETER] != 1.0:
        raise ValueError(
            "source-bound AddRMSNorm Split-D DAG marker must equal one")
    if (kernel.variant.lower() != "add_rmsnorm"
            or kernel.dtype != "bf16"
            or kernel.launch.algorithm != "aclnn_add_rmsnorm"
            or int(parameters["cann_tiling_key"]) != 31):
        raise ValueError(
            "source-bound AddRMSNorm Split-D DAG requires the BF16 CANN "
            "key-31 dispatch")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN AddRMSNorm key-31 pipeline must be attached exactly once")

    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    elements = rows * hidden
    worker_count = kernel.launch.grid_size
    block_factor = int(parameters["norm_block_factor"])
    rows_by_worker = tuple(
        block_factor if worker < worker_count - 1
        else rows - (worker_count - 1) * block_factor
        for worker in range(worker_count))
    tile_widths = tuple(
        min(_CANN_ADD_RMS_SPLIT_D_TILE_ELEMENTS, hidden - offset)
        for offset in range(0, hidden, _CANN_ADD_RMS_SPLIT_D_TILE_ELEMENTS))
    groups_by_worker = tuple(
        tuple(
            min(_CANN_NORM_SPLIT_ROW_FACTOR, worker_rows - offset)
            for offset in range(
                0, worker_rows, _CANN_NORM_SPLIT_ROW_FACTOR))
        for worker_rows in rows_by_worker)
    group_count = sum(len(groups) for groups in groups_by_worker)
    required_parameters = (
        "input_reread_count", "norm_weight_l2_copies",
        "lowered_vector_input_operations",
        "lowered_vector_reduce_operations",
        "lowered_vector_output_operations",
    )
    missing = tuple(
        name for name in required_parameters if name not in parameters)
    if missing:
        raise ValueError(
            "CANN AddRMSNorm key-31 launch parameters are incomplete: "
            f"{missing}")
    if (dtype_bytes != 2
            or inp.active_units != worker_count
            or any(worker_rows <= 0 for worker_rows in rows_by_worker)
            or sum(rows_by_worker) != rows
            or any(sum(groups) != worker_rows
                   for groups, worker_rows in zip(
                       groups_by_worker, rows_by_worker))
            or any(group_rows <= 0
                   or group_rows > _CANN_NORM_SPLIT_ROW_FACTOR
                   for groups in groups_by_worker for group_rows in groups)
            or int(parameters["input_reread_count"]) != 1
            or int(parameters["norm_weight_l2_copies"]) != group_count):
        raise ValueError(
            "CANN AddRMSNorm key-31 launch disagrees with its Split-D "
            "source contract")
    expected_lowered = {
        "lowered_vector_input_operations": float(3 * elements),
        "lowered_vector_reduce_operations": float(elements + rows),
        "lowered_vector_output_operations": float(5 * elements),
    }
    mismatched_lowered = {
        name: (parameters[name], expected)
        for name, expected in expected_lowered.items()
        if float(parameters[name]) != expected
    }
    if mismatched_lowered:
        raise ValueError(
            "CANN AddRMSNorm key-31 lowered work disagrees with source: "
            f"{mismatched_lowered}")

    inp, tile_stage_demands = _type_cann_add_rmsnorm_key31_split_d_rows(
        inp, hardware, rows_by_worker, tile_widths)
    inp = _type_cann_add_rmsnorm_key31_block_reduce(
        inp, hardware, groups_by_worker=groups_by_worker,
        tile_count=len(tile_widths))
    vector_demand, = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demand.exact_request_demands) != 1:
        raise ValueError(
            "CANN AddRMSNorm key-31 requires one exact rstd service")
    exact_rstd, = vector_demand.exact_request_demands
    expected_rstd_requests = tuple(groups_by_worker)
    if (exact_rstd.service != _CANN_NORM_EXACT_SERVICE
            or exact_rstd.request_sizes_by_worker != expected_rstd_requests):
        raise ValueError(
            "CANN AddRMSNorm key-31 rstd requests disagree with Split-D "
            "row-group ownership")

    # BF16 ComputeY crosses V->S and S->V once per source row and 8,192-wide
    # column tile.  That latter-path scalar-AIV dependency has its own
    # source-derived mechanism service; it is neither Vector work nor a
    # launch-wide completion observation.  Carry the same per-worker source
    # cardinality in the aggregate demand and in the pipeline stages below.
    scalar_handoff_requests = tuple(
        (1,) * (worker_rows * len(tile_widths))
        for worker_rows in rows_by_worker)
    # ``AddRmsNormSplitD`` owns four depth-one AscendC queues and five
    # distinct source lifetime roles.  The count is independent of a
    # measured kernel duration:
    #
    # * Former: one VECIN x queue and one VECOUT residual queue per row/tile;
    # * Latter: one VECIN gamma queue per group/tile, plus one VECIN x queue
    #   and one VECOUT y queue per row/tile;
    # * SubProcess: one VECOUT rstd queue per row group.
    #
    # Thus a worker owning ``R`` rows in ``G`` groups and ``T`` column tiles
    # has ``4 * R * T + G * T + G`` source queue lifetimes.  This is a
    # static CANN 8.5.1 program property, not the rejected DMA-descriptor
    # counter attribution.  The independent queue-cycle mechanism measures
    # an AllocTensor/EnQue/DeQue/FreeTensor control lifetime and is attached
    # below to the identical source stages that acquire each finite slot.
    queue_control_requests = tuple(
        (1,) * (
            4 * worker_rows * len(tile_widths)
            + len(groups) * len(tile_widths)
            + len(groups))
        for worker_rows, groups in zip(rows_by_worker, groups_by_worker))
    inp = _attach_scalar_aiv_exact_demands(inp, (
        ComputeExactRequestDemand(
            _CANN_ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_HANDOFF_EXACT_SERVICE,
            scalar_handoff_requests),
        ComputeExactRequestDemand(
            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
            queue_control_requests),
    ))

    path_by_name = {path.name: path for path in inp.memory_paths}
    actual_path_bytes = {
        path_name: {node.resource_role: node.bytes for node in path.nodes}
        for path_name, path in path_by_name.items()}
    row_bytes = hidden * dtype_bytes
    expected_path_bytes = {
        "mte2_read": {
            "mte2": float((3 * rows + group_count) * row_bytes),
            "l2_cache": float((3 * rows + group_count) * row_bytes),
            "dram": float((3 * rows + group_count) * row_bytes),
        },
        "mte3_write": {
            "mte3": float(2 * rows * row_bytes + 4 * rows),
            "l2_cache": float(2 * rows * row_bytes + 4 * rows),
            "dram": float(2 * rows * row_bytes + 4 * rows),
        },
    }
    if actual_path_bytes != expected_path_bytes:
        raise ValueError(
            "CANN AddRMSNorm key-31 traffic disagrees with Split-D source: "
            f"actual={actual_path_bytes}, expected={expected_path_bytes}")

    def _memory(path_name, **bytes_by_role):
        return tuple(
            PhaseMemoryDemand(path_name, role, byte_count)
            for role, byte_count in bytes_by_role.items())

    def _former_row_task(worker, tile_index, tile_bytes, tile_stages, *,
                         first_in_tile, first_in_group,
                         rstd_initialization_rows=0, repeat_count=1):
        """One exact Former row, with its source FIFO represented by a slot."""

        if (isinstance(rstd_initialization_rows, bool)
                or not isinstance(rstd_initialization_rows, int)
                or rstd_initialization_rows < 0):
            raise ValueError(
                "CANN AddRMSNorm key-31 rstd initialization rows must be a "
                "non-negative integer")
        if first_in_tile:
            prior_resources = (
                ("compute:compute_reduction",)
                if not first_in_group else ())
        else:
            prior_resources = ()
        stages = []
        if rstd_initialization_rows:
            # Source order is ``outQueueRstd.AllocTensor``, ``Duplicate`` and
            # ``PipeBarrier<PIPE_V>`` before the first Former CopyIn.  The
            # finite queue slot itself is represented below as a worker-local
            # buffer whose consumer is the terminal rstd CopyOut.
            stages.append(PipelineStageDemand(
                "rstd_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),)),),
                predecessors=()))
            stages.append(PipelineStageDemand(
                "rstd_init", "compute:compute_vector", None,
                compute_service_demands=(ComputeServiceDemand(
                    "generic", float(rstd_initialization_rows)),),
                predecessors=("rstd_queue_control",)))
        former_input_predecessors = (
            ("rstd_init",) if rstd_initialization_rows else ())
        stages.extend((
            PipelineStageDemand(
                "former_input_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),)),),
                predecessors=former_input_predecessors),
            PipelineStageDemand(
                "former_input", "memory:mte2_read", None,
                memory_node_demands=_memory(
                    "mte2_read", mte2=2.0 * tile_bytes,
                    l2_cache=2.0 * tile_bytes, dram=2.0 * tile_bytes),
                prior_resources=prior_resources,
                predecessors=("former_input_queue_control",)),
            PipelineStageDemand(
                "former_output_queue_control", "compute:scalar_aiv", None,
                compute_exact_request_demands=(
                    ComputeExactRequestDemand(
                        _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                        ((1,),)),),
                predecessors=("former_input",)),
            PipelineStageDemand(
                "former_compute", "compute:compute_vector", None,
                compute_service_demands=tile_stages["input_compute"],
                predecessors=("former_output_queue_control",)),
            PipelineStageDemand(
                "residual_store", "memory:mte3_write", None,
                memory_node_demands=_memory(
                    "mte3_write", mte3=tile_bytes,
                    l2_cache=tile_bytes, dram=tile_bytes),
                completion_resource_roles=("mte3",),
                predecessors=("former_compute",)),
            PipelineStageDemand(
                "former_reduce", "compute:compute_vector", None,
                compute_service_demands=tile_stages["reduce_compute"],
                predecessors=("former_compute",),
                start_predecessors=("residual_store",)),
        ))
        return PipelineTaskDemand(
            name=(
                f"key31_former_t{tile_index}_first"
                if first_in_tile else f"key31_former_t{tile_index}_steady"),
            worker=worker,
            repeat_count=repeat_count,
            stages=tuple(stages))

    def _group_sequence(worker, group_rows, *, first_group, suffix):
        """One source-ordered Split-D row group without row unrolling.

        The static source program has three repeated recurrences: Former
        rows, Latter rows, and full 64-row groups.  ``PipelineTaskDemand``
        repeat counts preserve the first two exactly; a program below
        preserves the third.  ``prior_resources`` only carries a source edge
        across task boundaries -- it does not introduce a duration model.
        """

        if group_rows <= 0 or group_rows > _CANN_NORM_SPLIT_ROW_FACTOR:
            raise ValueError(
                "CANN AddRMSNorm key-31 group must contain 1..64 rows")
        tasks = []
        for tile_index, (tile_width, tile_stages) in enumerate(
                zip(tile_widths, tile_stage_demands)):
            tile_bytes = float(tile_width * dtype_bytes)
            # The first tile starts after a preceding group's rstd MTE3
            # completion.  Later tiles instead start after the former tile's
            # block reduction.  The depth-one Former queue carries every
            # row-to-row reduction edge, including the steady recurrence.
            former_first = _former_row_task(
                worker, tile_index, tile_bytes, tile_stages,
                first_in_tile=True,
                first_in_group=(first_group and tile_index == 0),
                rstd_initialization_rows=(
                    group_rows if tile_index == 0 else 0))
            if not first_group and tile_index == 0:
                first_stage = former_first.stages[0]
                former_first = dataclasses.replace(
                    former_first,
                    stages=(dataclasses.replace(
                        first_stage,
                        prior_resources=("memory:mte3_write",)),)
                    + former_first.stages[1:])
            tasks.append(former_first)
            if group_rows > 1:
                tasks.append(_former_row_task(
                    worker, tile_index, tile_bytes, tile_stages,
                    first_in_tile=False, first_in_group=False,
                    repeat_count=group_rows - 1))
            block_reduce_services = []
            full_repeats = group_rows // 8
            if full_repeats:
                block_reduce_services.append(ComputeServiceDemand(
                    _CANN_ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_FULL64_SERVICE,
                    float(full_repeats), serial_invocations=1))
            if group_rows % 8:
                block_reduce_services.append(ComputeServiceDemand(
                    _CANN_ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_TAIL8_SERVICE,
                    1.0))
            if not block_reduce_services:
                raise AssertionError(
                    "CANN AddRMSNorm key-31 group has no BlockReduce branch")
            tasks.append(PipelineTaskDemand(
                name=f"key31_former_t{tile_index}_block_reduce",
                worker=worker,
                stages=(PipelineStageDemand(
                    "former_block_reduce", "compute:compute_reduction",
                    None,
                    compute_service_demands=tuple(block_reduce_services),
                    predecessors=(),
                    prior_resources=("compute:compute_vector",)),)))

        tasks.append(PipelineTaskDemand(
            name="key31_scalar_normalization",
            worker=worker,
            stages=(PipelineStageDemand(
                "scalar_normalization", "compute:compute_vector", None,
                compute_service_demands=tuple(
                    ComputeServiceDemand(
                        demand.service, demand.operations * group_rows,
                        serial_invocations=(
                            demand.serial_invocations * group_rows))
                    for demand in tile_stage_demands[0]["epsilon_add"]),
                compute_exact_request_demands=(ComputeExactRequestDemand(
                    exact_rstd.service, ((group_rows,),)),),
                predecessors=(),
                prior_resources=(
                    "compute:compute_reduction", "memory:mte3_write")),)))

        for tile_index, (tile_width, tile_stages) in enumerate(
                zip(tile_widths, tile_stage_demands)):
            tile_bytes = float(tile_width * dtype_bytes)
            gamma_name = (
                "key31_gamma_seed"
                if first_group and tile_index == 0 else "gamma_load")
            tasks.append(PipelineTaskDemand(
                name=f"key31_latter_t{tile_index}_first",
                worker=worker,
                stages=(
                    PipelineStageDemand(
                        "gamma_queue_control", "compute:scalar_aiv", None,
                        compute_exact_request_demands=(
                            ComputeExactRequestDemand(
                                _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                                ((1,),)),),
                        predecessors=(),
                        prior_resources=(
                            "compute:compute_vector", "memory:mte3_write")),
                    PipelineStageDemand(
                        gamma_name, "memory:mte2_read", None,
                        memory_node_demands=_memory(
                            "mte2_read", mte2=tile_bytes,
                            l2_cache=tile_bytes, dram=tile_bytes),
                        predecessors=("gamma_queue_control",)),
                    PipelineStageDemand(
                        "latter_input_queue_control",
                        "compute:scalar_aiv", None,
                        compute_exact_request_demands=(
                            ComputeExactRequestDemand(
                                _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                                ((1,),)),),
                        predecessors=(gamma_name,)),
                    PipelineStageDemand(
                        "latter_input", "memory:mte2_read", None,
                        memory_node_demands=_memory(
                            "mte2_read", mte2=tile_bytes,
                            l2_cache=tile_bytes, dram=tile_bytes),
                        predecessors=("latter_input_queue_control",)),
                    PipelineStageDemand(
                        "latter_residual_reload",
                        "compute:compute_vector", None,
                        compute_service_demands=tile_stages[
                            "residual_reload"],
                        predecessors=("latter_input",)),
                    PipelineStageDemand(
                        "latter_scalar_handoff", "compute:scalar_aiv", None,
                        compute_exact_request_demands=(
                            ComputeExactRequestDemand(
                                _CANN_ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_HANDOFF_EXACT_SERVICE,
                                ((1,),)),),
                        predecessors=("latter_residual_reload",)),
                    PipelineStageDemand(
                        "latter_output_queue_control",
                        "compute:scalar_aiv", None,
                        compute_exact_request_demands=(
                            ComputeExactRequestDemand(
                                _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                                ((1,),)),),
                        predecessors=("latter_scalar_handoff",)),
                    PipelineStageDemand(
                        "latter_output_scale",
                        "compute:compute_vector", None,
                        compute_service_demands=tile_stages["output_scale"],
                        predecessors=("latter_output_queue_control",)),
                    PipelineStageDemand(
                        "latter_output_finalize",
                        "compute:compute_vector", None,
                        compute_service_demands=tile_stages[
                            "output_finalize"],
                        predecessors=("latter_output_scale",)),
                    PipelineStageDemand(
                        "latter_store", "memory:mte3_write", None,
                        memory_node_demands=_memory(
                            "mte3_write", mte3=tile_bytes,
                            l2_cache=tile_bytes, dram=tile_bytes),
                        completion_resource_roles=("mte3",),
                        predecessors=("latter_output_finalize",)),
                )))
            if group_rows > 1:
                tasks.append(PipelineTaskDemand(
                    name=f"key31_latter_t{tile_index}_steady",
                    worker=worker,
                    repeat_count=group_rows - 1,
                    stages=(
                        PipelineStageDemand(
                            "latter_input_queue_control",
                            "compute:scalar_aiv", None,
                            compute_exact_request_demands=(
                                ComputeExactRequestDemand(
                                    _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                                    ((1,),)),),
                            predecessors=()),
                        PipelineStageDemand(
                            "latter_input", "memory:mte2_read", None,
                            memory_node_demands=_memory(
                                "mte2_read", mte2=tile_bytes,
                                l2_cache=tile_bytes, dram=tile_bytes),
                            predecessors=("latter_input_queue_control",)),
                        PipelineStageDemand(
                            "latter_residual_reload",
                            "compute:compute_vector", None,
                            compute_service_demands=tile_stages[
                                "residual_reload"],
                            predecessors=("latter_input",)),
                        PipelineStageDemand(
                            "latter_scalar_handoff",
                            "compute:scalar_aiv", None,
                            compute_exact_request_demands=(
                                ComputeExactRequestDemand(
                                    _CANN_ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_HANDOFF_EXACT_SERVICE,
                                    ((1,),)),),
                            predecessors=("latter_residual_reload",)),
                        PipelineStageDemand(
                            "latter_output_queue_control",
                            "compute:scalar_aiv", None,
                            compute_exact_request_demands=(
                                ComputeExactRequestDemand(
                                    _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                                    ((1,),)),),
                            predecessors=("latter_scalar_handoff",)),
                        PipelineStageDemand(
                            "latter_output_scale",
                            "compute:compute_vector", None,
                            compute_service_demands=tile_stages[
                                "output_scale"],
                            predecessors=("latter_output_queue_control",)),
                        PipelineStageDemand(
                            "latter_output_finalize",
                            "compute:compute_vector", None,
                            compute_service_demands=tile_stages[
                                "output_finalize"],
                            predecessors=("latter_output_scale",)),
                        PipelineStageDemand(
                            "latter_store", "memory:mte3_write", None,
                            memory_node_demands=_memory(
                                "mte3_write", mte3=tile_bytes,
                                l2_cache=tile_bytes, dram=tile_bytes),
                            completion_resource_roles=("mte3",),
                            predecessors=("latter_output_finalize",)),
                    )))
        tasks.append(PipelineTaskDemand(
            name="key31_rstd_store",
            worker=worker,
            stages=(PipelineStageDemand(
                "rstd_store", "memory:mte3_write", None,
                memory_node_demands=_memory(
                    "mte3_write", mte3=float(4 * group_rows),
                    l2_cache=float(4 * group_rows),
                    dram=float(4 * group_rows)),
                completion_resource_roles=("mte3",),
                predecessors=(),
                prior_resources=("memory:mte3_write",)),)))
        return PipelineTaskSequenceDemand(
            name=f"worker{worker}_key31_{suffix}",
            worker=worker,
            tasks=tuple(tasks))

    # This is a representation-only compression.  Both Dataset and Serving
    # use the same first-group program, row recurrences, and physical-worker
    # classes; Forward no longer selects a different source-DAG path.
    first_group_entries = []
    remaining_worker_classes = {}
    for physical_worker, groups in enumerate(groups_by_worker):
        rstd_requests = exact_rstd.request_sizes_by_worker[physical_worker]
        if rstd_requests != groups:
            raise ValueError(
                "CANN AddRMSNorm key-31 compact worker lost row-group "
                "rstd ownership")
        first_group_entries.append(_group_sequence(
            physical_worker, groups[0], first_group=True,
            suffix="first_group"))
        if len(groups) > 1:
            remaining_groups = groups[1:]
            if remaining_groups in remaining_worker_classes:
                remaining_worker_classes[remaining_groups].append(
                    physical_worker)
            else:
                remaining_worker_classes[remaining_groups] = [
                    physical_worker]

    program_entries = []
    for class_index, (remaining_groups, physical_workers) in enumerate(
            remaining_worker_classes.items()):
        if any(group_rows != _CANN_NORM_SPLIT_ROW_FACTOR
               for group_rows in remaining_groups[:-1]):
            raise ValueError(
                "CANN AddRMSNorm key-31 has a non-terminal short row group")
        representative_worker = physical_workers[0]
        full_groups = sum(
            group_rows == _CANN_NORM_SPLIT_ROW_FACTOR
            for group_rows in remaining_groups)
        tail_rows = (
            0 if remaining_groups[-1] == _CANN_NORM_SPLIT_ROW_FACTOR
            else remaining_groups[-1])
        if full_groups + (1 if tail_rows else 0) != len(remaining_groups):
            raise ValueError(
                "CANN AddRMSNorm key-31 row-group program is not canonical")
        patterns = []
        if full_groups:
            patterns.append(_group_sequence(
                representative_worker, _CANN_NORM_SPLIT_ROW_FACTOR,
                first_group=False, suffix="full_group"))
        tail_order = ()
        if tail_rows:
            patterns.append(_group_sequence(
                representative_worker, tail_rows, first_group=False,
                suffix="tail_group"))
            tail_order = (len(patterns) - 1,)
        if not patterns:
            raise AssertionError(
                "positive key-31 remaining rows produced no source group")
        order = (0,)
        repeat_count = full_groups if full_groups else 1
        if not full_groups and tail_rows:
            tail_order = ()
        program_entries.append(PipelineTaskProgramDemand(
            name=f"worker_class{class_index}_key31_remaining_groups",
            worker=representative_worker,
            patterns=tuple(patterns),
            order=order,
            repeat_count=repeat_count,
            tail_order=tail_order,
            worker_multiplicity=len(physical_workers),
            physical_workers=tuple(physical_workers)))

    entries = tuple(first_group_entries + program_entries)
    attached = attach_pipeline_work_plan(
        inp, hardware, PipelineWorkPlan(
            worker_count=worker_count,
            tasks=entries,
            buffers=(
                PipelineBuffer(
                    "key31_former_loop_issue", "former_input_queue_control",
                    "former_reduce", 1, "worker", "start"),
                PipelineBuffer(
                    "key31_former_output_queue",
                    "former_output_queue_control",
                    "residual_store", 1, "worker"),
                PipelineBuffer(
                    "key31_latter_input_issue_queue",
                    "latter_input_queue_control",
                    "latter_store", 1, "worker", "start"),
                PipelineBuffer(
                    "key31_latter_output_queue",
                    "latter_output_queue_control",
                    "latter_store", 1, "worker"),
                PipelineBuffer(
                    "key31_rstd_output_queue", "rstd_queue_control",
                    "rstd_store",
                    1, "worker"),
            )),
        promote_npu_partial=True)
    return dataclasses.replace(
        attached,
        # The empty AIV task curve spans first instruction through terminal
        # completion.  ``AddRmsNormSplitD`` starts its typed 80-B entry and
        # Former path at that first instruction, so that independent empty
        # task duration is a competing whole-task lower bound, never a serial
        # prefix before the source body.  The separately measured worker-fill
        # and tiling materialization phases start from the same admission
        # boundary and therefore compete as well.  This contract is bound to
        # the fixed key-31 source program, not to a row or hidden threshold.
        task_duration_floor_contract=(
            AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR),
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
        limitations=attached.limitations + (
            "cann_add_rmsnorm_key31_source_entry_absolute_latency_unavailable",
        ))


def _attach_cann_add_rms_row_group_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False
        ) -> PredictionInput:
    """Attach the CANN 8.5.1 BF16 key-30 row-group source program.

    ``KernelAddRmsNorm<bfloat16_t>`` assigns ``blockFactor`` contiguous rows
    to every non-tail AIV and processes them in groups of at most 64.  Gamma
    is loaded once per AIV.  Within a row, the residual MTE3 copy may overlap
    reduction and output scaling, but the final-y allocation waits for that
    copy.  The depth-one y queue gates the next row, and the depth-one rstd
    queue gates the next group.  All work and traffic below come directly
    from that source program; no production duration is consumed.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN AddRmsNorm key-30 pipeline requires separated npu_tile "
            "AIVs")
    parameters = kernel.launch.parameters
    if _CANN_ADD_RMS_ROW_GROUP_DAG_PARAMETER not in parameters:
        raise ValueError(
            "CANN AddRmsNorm key 30 requires its source-bound row-group DAG "
            "identity")
    if parameters[_CANN_ADD_RMS_ROW_GROUP_DAG_PARAMETER] != 1.0:
        raise ValueError(
            "source-bound AddRmsNorm row-group DAG marker must equal one")
    if (kernel.variant.lower() != "add_rmsnorm"
            or kernel.dtype != "bf16"
            or kernel.launch.algorithm != "aclnn_add_rmsnorm"
            or int(parameters["cann_tiling_key"]) != 30):
        raise ValueError(
            "source-bound AddRmsNorm row-group DAG requires the BF16 CANN "
            "key-30 dispatch")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN AddRmsNorm row-group pipeline must be attached exactly once")

    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    elements = rows * hidden
    required_parameters = (
        "norm_block_factor", "norm_weight_l2_copies", "rstd_output_bytes",
        "lowered_vector_input_operations",
        "lowered_vector_reduce_operations",
        "lowered_vector_output_operations",
    )
    missing = tuple(
        name for name in required_parameters if name not in parameters)
    if missing:
        raise ValueError(
            "CANN AddRmsNorm key-30 launch parameters are incomplete: "
            f"{missing}")
    vector_cores = int(
        hardware.resource_by_role("compute_vector").total_units)
    block_factor = math.ceil(rows / vector_cores)
    worker_count = math.ceil(rows / block_factor)
    rows_by_worker = (
        (block_factor,) * (worker_count - 1)
        + (rows - (worker_count - 1) * block_factor,)
    )
    if (dtype_bytes != 2
            or rows <= vector_cores
            or hidden < 2_001
            or hidden > 12_288
            or worker_count > vector_cores
            or kernel.launch.grid != (worker_count, 1, 1)
            or kernel.launch.block != (1, 1, 1)
            or inp.active_units != worker_count
            or int(parameters["norm_block_factor"]) != block_factor
            or int(parameters["norm_weight_l2_copies"]) != worker_count
            or float(parameters["rstd_output_bytes"]) != float(rows * 4)
            or "input_reread_count" in parameters
            or any(worker_rows <= 0 or worker_rows > block_factor
                   for worker_rows in rows_by_worker)
            or sum(rows_by_worker) != rows):
        raise ValueError(
            "CANN AddRmsNorm key-30 launch disagrees with its BF16 row-group "
            "source contract")
    expected_lowered = {
        "lowered_vector_input_operations": float(3 * elements),
        "lowered_vector_reduce_operations": float(elements),
        "lowered_vector_output_operations": float(4 * elements),
    }
    mismatched_lowered = {
        name: (parameters[name], expected)
        for name, expected in expected_lowered.items()
        if float(parameters[name]) != expected
    }
    if mismatched_lowered:
        raise ValueError(
            "CANN AddRmsNorm key-30 lowered work disagrees with source: "
            f"{mismatched_lowered}")

    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN AddRmsNorm key 30 requires one compute_vector demand")
    vector_demand = vector_demands[0]
    if (len(vector_demand.service_demands) != 1
            or vector_demand.service_demands[0].service != "generic"
            or len(vector_demand.exact_request_demands) != 1
            or vector_demand.exact_request_demands[0].service
            != _CANN_NORM_EXACT_SERVICE):
        raise ValueError(
            "CANN AddRmsNorm key 30 requires its typed generic and exact "
            "Vector contracts")
    # The existing aggregate operation convention counts the semantic
    # residual add, square/reduction and two output multiplies plus the eight
    # full-row BF16 lowering passes visible in this fixed source: 13H generic
    # element operations plus one scalar epsilon add per row.
    # Sqrt->Duplicate->Div remains one separately priced size-one exact request
    # per row.
    generic_per_row = float(13 * hidden + 1)
    generic = vector_demand.service_demands[0]
    exact = vector_demand.exact_request_demands[0]
    expected_generic_by_worker = tuple(
        generic_per_row * worker_rows for worker_rows in rows_by_worker)
    expected_exact_by_worker = tuple(
        (1,) * worker_rows for worker_rows in rows_by_worker)
    if (generic.operations != generic_per_row * rows
            or generic.operations_by_worker != expected_generic_by_worker
            or exact.request_sizes_by_worker != expected_exact_by_worker):
        raise ValueError(
            "CANN AddRmsNorm key-30 compute demand disagrees with source row "
            "ownership")
    row_stage_demands = _cann_add_rmsnorm_row_stage_demands(
        hardware, hidden, single_n=False)
    inp = _type_cann_norm_rows(
        inp, rows_by_worker, row_stage_demands,
        include_rmsnorm_local_scalar=False)
    reduction_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_reduction")
    if len(reduction_demands) != 1:
        raise ValueError(
            "CANN AddRMSNorm key-30 exact reduction must replace one generic "
            "reduction demand")
    flat = dict(inp.demand_per_resource)
    if flat["compute_reduction"] != reduction_demands[0].operations:
        raise ValueError(
            "CANN AddRMSNorm key-30 aggregate and structured reduction "
            "demands disagree")
    flat["compute_reduction"] = 0.0
    inp = dataclasses.replace(
        inp,
        demand_per_resource=flat,
        compute_demands=tuple(
            demand for demand in inp.compute_demands
            if demand.resource_role != "compute_reduction"),
    )
    queue_cycles_by_worker = tuple(
        (1,) * (
            1 + 3 * worker_rows
            + math.ceil(worker_rows / _CANN_NORM_SPLIT_ROW_FACTOR))
        for worker_rows in rows_by_worker)
    scalar_control_demands = (
        ComputeExactRequestDemand(
            _CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE,
            ((1,),) * worker_count),
        ComputeExactRequestDemand(
            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
            queue_cycles_by_worker),
        ComputeExactRequestDemand(
            _CANN_ADD_RMSNORM_LOCAL_SCALAR_HANDOFF_EXACT_SERVICE,
            tuple((1,) * worker_rows for worker_rows in rows_by_worker)),
        ComputeExactRequestDemand(
            _CANN_VECTOR_TO_MTE2_FENCE_EXACT_SERVICE,
            tuple((1,) * worker_rows for worker_rows in rows_by_worker)),
    )
    inp = _attach_scalar_aiv_exact_demands(
        inp, scalar_control_demands)

    path_by_name = {path.name: path for path in inp.memory_paths}
    if set(path_by_name) != {"mte2_read", "mte3_write"}:
        raise ValueError(
            "CANN AddRmsNorm key 30 requires exactly the MTE2-read and "
            "MTE3-write paths")
    actual_path_bytes = {
        path_name: {
            node.resource_role: node.bytes for node in path.nodes
        }
        for path_name, path in path_by_name.items()
    }
    expected_path_bytes = {
        "mte2_read": {
            "mte2": float(4 * rows * hidden + 2 * worker_count * hidden),
            "l2_cache": float(4 * rows * hidden + 2 * worker_count * hidden),
            "dram": float(4 * rows * hidden + 2 * hidden),
        },
        "mte3_write": {
            "mte3": float(4 * rows * hidden + 4 * rows),
            "l2_cache": float(4 * rows * hidden + 4 * rows),
            "dram": float(4 * rows * hidden + 4 * rows),
        },
    }
    if actual_path_bytes != expected_path_bytes:
        raise ValueError(
            "CANN AddRmsNorm key-30 traffic disagrees with its BF16 source "
            f"contract: actual={actual_path_bytes}, "
            f"expected={expected_path_bytes}")

    mte2_service = "memory:mte2_read"
    mte3_service = "memory:mte3_write"
    vector_service = "compute:compute_vector"
    input_bytes = float(4 * hidden)
    gamma_l2_bytes = float(2 * hidden)
    gamma_dram_bytes = float(2 * hidden)
    residual_bytes = float(2 * hidden)
    output_bytes = float(2 * hidden)
    # The published multi-AIV gamma/input/output probes are completion spans
    # for an admitted launch, not repeated edges in every worker's row DAG.
    # Price source-local DataCopyPad requests here; worker admission remains a
    # single AIV_WORKER_START_FILL envelope and MTE/L2/DRAM contention remains
    # in the aggregate paths.  This applies uniformly to every blockDim and
    # avoids a 32-worker observation becoming a shape-specific timing floor.
    first_mte2_to_vector = _cann_memory_completion_request(
        hardware, "mte2", _CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE,
        int(gamma_l2_bytes))
    first_mte3_to_vector = _cann_memory_completion_request(
        hardware, "mte3", _CANN_DATACOPYPAD_VECTOR_TO_MTE3_EXACT_SERVICE,
        int(residual_bytes))
    input_pair_completion = ()
    terminal_output_sequence = ()

    def _memory(path_name, **bytes_by_role):
        return tuple(
            PhaseMemoryDemand(path_name, role, byte_count)
            for role, byte_count in bytes_by_role.items())

    def _row_name(base, row_index, group_rows):
        if base == "input_load" and row_index == 0:
            return "input_load"
        if base == "output_store" and row_index == group_rows - 1:
            return "output_store"
        return f"row{row_index}_{base}"

    def _group_task(worker, group_rows, *, include_gamma, suffix):
        if group_rows <= 0 or group_rows > _CANN_NORM_SPLIT_ROW_FACTOR:
            raise ValueError(
                "CANN AddRmsNorm key-30 group must contain 1..64 rows")
        stages = []
        if include_gamma:
            stages.extend((
                PipelineStageDemand(
                    "source_control", "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_RMSNORM_BUFFER_LIFECYCLE_EXACT_SERVICE,
                            ((1,),)),
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),)),
                    ),
                    predecessors=()),
                PipelineStageDemand(
                    "gamma_load", mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read", mte2=gamma_l2_bytes,
                        l2_cache=gamma_l2_bytes),
                    launch_shared_memory_node_demands=_memory(
                        "mte2_read", dram=gamma_dram_bytes),
                    memory_exact_request_demands=first_mte2_to_vector,
                    predecessors=("source_control",)),
            ))
        prior_output = None
        for row_index in range(group_rows):
            input_name = _row_name("input_load", row_index, group_rows)
            input_queue_name = f"row{row_index}_input_queue_control"
            input_compute_name = f"row{row_index}_input_compute"
            residual_queue_name = (
                f"row{row_index}_residual_queue_control")
            residual_store_name = f"row{row_index}_residual_store"
            reduce_name = f"row{row_index}_reduce_compute"
            epsilon_name = f"row{row_index}_epsilon_add"
            reciprocal_sqrt_name = f"row{row_index}_reciprocal_sqrt"
            local_scalar_handoff_name = (
                f"row{row_index}_local_scalar_handoff")
            output_scale_name = f"row{row_index}_output_scale"
            output_finalize_name = f"row{row_index}_output_finalize"
            output_fence_name = f"row{row_index}_output_fence"
            output_queue_name = f"row{row_index}_output_queue_control"
            output_store_name = _row_name(
                "output_store", row_index, group_rows)
            exact_terminal_output = (
                bool(terminal_output_sequence)
                and row_index == group_rows - 1)
            if row_index == 0:
                input_queue_predecessors = (
                    ("gamma_load",) if include_gamma else ())
            else:
                input_queue_predecessors = (prior_output,)
            stages.extend((
                PipelineStageDemand(
                    input_queue_name, "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),)),),
                    predecessors=input_queue_predecessors),
                PipelineStageDemand(
                    input_name, mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read", mte2=input_bytes,
                        l2_cache=input_bytes,
                        dram=input_bytes),
                    memory_exact_request_demands=input_pair_completion,
                    predecessors=(input_queue_name,)),
                PipelineStageDemand(
                    input_compute_name, vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "input_compute"],
                    predecessors=(input_name,)),
                PipelineStageDemand(
                    residual_queue_name, "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),)),),
                    predecessors=(),
                    start_predecessors=(input_compute_name,)),
                PipelineStageDemand(
                    residual_store_name, mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=residual_bytes,
                        l2_cache=residual_bytes,
                        dram=residual_bytes),
                    completion_resource_roles=("mte3",),
                    memory_exact_request_demands=(
                        first_mte3_to_vector
                        if include_gamma and row_index == 0 else ()),
                    predecessors=(input_compute_name, residual_queue_name)),
                PipelineStageDemand(
                    reduce_name, vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "reduce_compute"],
                    predecessors=(input_compute_name,),
                    start_predecessors=(residual_store_name,)),
            ))
            stages.extend((
                PipelineStageDemand(
                    epsilon_name, vector_service, None,
                    compute_service_demands=row_stage_demands["epsilon_add"],
                    predecessors=(reduce_name,)),
                PipelineStageDemand(
                    reciprocal_sqrt_name, vector_service, None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_NORM_EXACT_SERVICE, ((1,),)),),
                    predecessors=(epsilon_name,)),
                PipelineStageDemand(
                    local_scalar_handoff_name, "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_ADD_RMSNORM_LOCAL_SCALAR_HANDOFF_EXACT_SERVICE,
                            ((1,),)),),
                    predecessors=(reciprocal_sqrt_name,)),
                PipelineStageDemand(
                    output_scale_name, vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "output_scale"],
                    predecessors=(local_scalar_handoff_name,)),
                PipelineStageDemand(
                    output_finalize_name, vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "output_finalize"],
                    predecessors=(output_scale_name, residual_store_name)),
                PipelineStageDemand(
                    output_fence_name, "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_VECTOR_TO_MTE2_FENCE_EXACT_SERVICE,
                            ((1,),)),),
                    predecessors=(output_finalize_name,)),
                PipelineStageDemand(
                    output_queue_name, "compute:scalar_aiv", None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                            ((1,),)),),
                    predecessors=(),
                    start_predecessors=(output_fence_name,)),
                PipelineStageDemand(
                    output_store_name, mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write",
                        mte3=(
                            output_bytes + 4.0 * group_rows
                            if exact_terminal_output else output_bytes),
                        l2_cache=(
                            output_bytes + 4.0 * group_rows
                            if exact_terminal_output else output_bytes),
                        dram=(
                            output_bytes + 4.0 * group_rows
                            if exact_terminal_output else output_bytes)),
                    completion_resource_roles=("mte3",),
                    memory_exact_request_demands=(
                        terminal_output_sequence if exact_terminal_output
                        else ()),
                    memory_exact_request_replaces_resource_capacity=(
                        exact_terminal_output),
                    predecessors=(output_fence_name, output_queue_name)),
            ))
            prior_output = output_store_name
        stages.append(PipelineStageDemand(
            "rstd_queue_control", "compute:scalar_aiv", None,
            compute_exact_request_demands=(
                ComputeExactRequestDemand(
                    _CANN_AIV_QUEUE_CONTROL_CYCLE_EXACT_SERVICE,
                    ((1,),)),),
            predecessors=(),
            start_predecessors=(prior_output,)))
        if not terminal_output_sequence:
            stages.append(PipelineStageDemand(
                "rstd_store", mte3_service, None,
                memory_node_demands=_memory(
                    "mte3_write", mte3=float(4 * group_rows),
                    l2_cache=float(4 * group_rows),
                    dram=float(4 * group_rows)),
                completion_resource_roles=("mte3",),
                predecessors=(prior_output, "rstd_queue_control")))
        return PipelineTaskDemand(
            name=f"worker{worker}_{suffix}",
            worker=worker,
            stages=tuple(stages),
        )

    def _worker_program_entries(worker, worker_rows):
        first_group_rows = min(
            worker_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
        first_group = _group_task(
            worker, first_group_rows, include_gamma=True,
            suffix="first_group")
        remaining_rows = worker_rows - first_group_rows
        if remaining_rows == 0:
            return (first_group,)
        full_groups, tail_rows = divmod(
            remaining_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
        if full_groups > 0:
            patterns = [PipelineTaskSequenceDemand(
                name=f"worker{worker}_full_group_sequence",
                worker=worker,
                tasks=(_group_task(
                    worker, _CANN_NORM_SPLIT_ROW_FACTOR,
                    include_gamma=False, suffix="full_group"),),
            )]
            tail_order = ()
            if tail_rows > 0:
                patterns.append(PipelineTaskSequenceDemand(
                    name=f"worker{worker}_tail_group_sequence",
                    worker=worker,
                    tasks=(_group_task(
                        worker, tail_rows, include_gamma=False,
                        suffix="tail_group"),),
                ))
                tail_order = (1,)
            return (first_group, PipelineTaskProgramDemand(
                name=f"worker{worker}_remaining_groups",
                worker=worker,
                patterns=tuple(patterns),
                order=(0,),
                repeat_count=full_groups,
                tail_order=tail_order,
            ))
        elif tail_rows > 0:
            return (first_group, _group_task(
                worker, tail_rows, include_gamma=False,
                suffix="tail_group"))
        else:
            raise AssertionError(
                "positive key-30 remaining rows produced no source group")

    if not analytical_compact:
        entries = tuple(
            entry
            for worker, worker_rows in enumerate(rows_by_worker)
            for entry in _worker_program_entries(worker, worker_rows))
    else:
        # The source program repeats by row group.  Classifying equal worker
        # row counts retains every row dependency while avoiding one Python
        # DAG construction per physically equivalent AIV.
        worker_classes = {}
        for physical_worker, worker_rows in enumerate(rows_by_worker):
            if worker_rows in worker_classes:
                worker_classes[worker_rows].append(physical_worker)
            else:
                worker_classes[worker_rows] = [physical_worker]
        entries = []
        for compact_worker, (worker_rows, physical_workers) in enumerate(
                worker_classes.items()):
            first_group_rows = min(
                worker_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
            tasks = [_group_task(
                compact_worker, first_group_rows, include_gamma=True,
                suffix="first_group")]
            remaining_rows = worker_rows - first_group_rows
            if remaining_rows > 0:
                full_groups, tail_rows = divmod(
                    remaining_rows, _CANN_NORM_SPLIT_ROW_FACTOR)
                if full_groups > 0:
                    tasks.append(dataclasses.replace(
                        _group_task(
                            compact_worker, _CANN_NORM_SPLIT_ROW_FACTOR,
                            include_gamma=False, suffix="full_group"),
                        repeat_count=full_groups))
                if tail_rows > 0:
                    tasks.append(_group_task(
                        compact_worker, tail_rows, include_gamma=False,
                        suffix="tail_group"))
                if full_groups == 0 and tail_rows == 0:
                    raise AssertionError(
                        "positive key-30 remaining rows produced no source "
                        "group")
            entries.append(PipelineTaskProgramDemand(
                name=f"worker_class{compact_worker}_row_groups",
                worker=compact_worker,
                patterns=(PipelineTaskSequenceDemand(
                    name=f"worker_class{compact_worker}_row_sequence",
                    worker=compact_worker,
                    tasks=tuple(tasks)),),
                order=(0,),
                worker_multiplicity=len(physical_workers),
                physical_workers=tuple(physical_workers),
            ))
        entries = tuple(entries)

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=entries,
        buffers=(
            PipelineBuffer(
                name="add_rms_key30_y_queue",
                producer_stage="input_load",
                consumer_stage="output_store",
                slots=1,
                scope="worker",
            ),
            PipelineBuffer(
                name="add_rms_key30_rstd_queue",
                producer_stage="input_load",
                consumer_stage=(
                    "output_store" if terminal_output_sequence
                    else "rstd_store"),
                slots=1,
                scope="worker",
            ),
        ),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    return dataclasses.replace(
        attached,
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
        limitations=attached.limitations + (
            ("cann_add_rmsnorm_first_mte2_completion_evidence_pending",)
            if not first_mte2_to_vector else ()),
    )


def _attach_cann_add_rms_single_n_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        analytical_compact: bool = False,
        ) -> PredictionInput:
    """Attach the source-closed CANN 8.5.1 key-33 single-row DAG.

    ``KernelAddRmsNormSingleN<bfloat16_t>`` owns exactly one row on each AIV
    and one TBuf.  It issues both input copies before waiting, then casts x1
    between the x1 and x2 completion waits.  Its remaining event waits
    establish the per-row order below: gamma MTE2 issue, residual MTE3 issue,
    serialized
    reduction/reciprocal-sqrt, rstd MTE3 issue, output scaling while that
    store is in flight, residual UB round-trip after its store completes,
    gamma multiply after all required waits, final cast, and the terminal y
    MTE3 drain.  Stage work is static source work and conserved traffic; no
    production duration is consumed.
    """

    if inp.mem_model != "npu_tile" or not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN AddRmsNorm SingleN pipeline requires separated npu_tile "
            "AIVs")
    parameters = kernel.launch.parameters
    if _CANN_ADD_RMS_SINGLE_N_DAG_PARAMETER not in parameters:
        raise ValueError(
            "CANN AddRmsNorm key 33 requires its source-bound SingleN DAG "
            "identity")
    if parameters[_CANN_ADD_RMS_SINGLE_N_DAG_PARAMETER] != 1.0:
        raise ValueError(
            "source-bound AddRmsNorm SingleN DAG marker must equal one")
    if (kernel.variant.lower() != "add_rmsnorm"
            or kernel.dtype != "bf16"
            or kernel.launch.algorithm != "aclnn_add_rmsnorm"
            or int(parameters["cann_tiling_key"]) != 33):
        raise ValueError(
            "source-bound AddRmsNorm SingleN DAG requires the BF16 CANN "
            "key-33 dispatch")
    if inp.pipeline_plan is not None:
        raise ValueError(
            "CANN AddRmsNorm SingleN pipeline must be attached exactly once")
    if not isinstance(analytical_compact, bool):
        raise TypeError(
            "CANN AddRmsNorm SingleN analytical_compact must be boolean")

    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    dtype_bytes = _shape(kernel, "dtype_bytes")
    elements = rows * hidden
    required_parameters = (
        "norm_block_factor", "norm_weight_l2_copies", "rstd_output_bytes",
        "lowered_vector_input_operations",
        "lowered_vector_reduce_operations",
        "lowered_vector_output_operations",
    )
    missing = tuple(
        name for name in required_parameters if name not in parameters)
    if missing:
        raise ValueError(
            "CANN AddRmsNorm SingleN launch parameters are incomplete: "
            f"{missing}")
    if (dtype_bytes != 2
            or hidden > 12_288
            or kernel.launch.grid != (rows, 1, 1)
            or kernel.launch.block != (1, 1, 1)
            or inp.active_units != rows
            or int(parameters["norm_block_factor"]) != 1
            or int(parameters["norm_weight_l2_copies"]) != rows
            or float(parameters["rstd_output_bytes"]) != float(rows * 4)
            or "input_reread_count" in parameters):
        raise ValueError(
            "CANN AddRmsNorm SingleN launch disagrees with its one-row BF16 "
            "source contract")
    expected_lowered = {
        "lowered_vector_input_operations": float(3 * elements),
        "lowered_vector_reduce_operations": float(3 * elements),
        "lowered_vector_output_operations": float(4 * elements),
    }
    mismatched_lowered = {
        name: (parameters[name], expected)
        for name, expected in expected_lowered.items()
        if float(parameters[name]) != expected
    }
    if mismatched_lowered:
        raise ValueError(
            "CANN AddRmsNorm SingleN lowered work disagrees with source: "
            f"{mismatched_lowered}")

    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN AddRmsNorm SingleN requires one compute_vector demand")
    vector_demand = vector_demands[0]
    if (len(vector_demand.service_demands) != 1
            or vector_demand.service_demands[0].service != "generic"
            or len(vector_demand.exact_request_demands) != 1
            or vector_demand.exact_request_demands[0].service
            != _CANN_NORM_EXACT_SERVICE):
        raise ValueError(
            "CANN AddRmsNorm SingleN requires its typed generic and exact "
            "Vector contracts")
    generic_per_row = float(15 * hidden + 1)
    generic = vector_demand.service_demands[0]
    exact = vector_demand.exact_request_demands[0]
    if (generic.operations != generic_per_row * rows
            or generic.operations_by_worker != (generic_per_row,) * rows
            or exact.request_sizes_by_worker != ((1,),) * rows):
        raise ValueError(
            "CANN AddRmsNorm SingleN compute demand disagrees with one-row "
            "AIV ownership")
    row_stage_demands = _cann_add_rmsnorm_row_stage_demands(
        hardware, hidden, single_n=True)
    inp = _type_cann_norm_rows(
        inp, (1,) * rows, row_stage_demands,
        include_rmsnorm_local_scalar=False)

    path_by_name = {path.name: path for path in inp.memory_paths}
    if set(path_by_name) != {"mte2_read", "mte3_write"}:
        raise ValueError(
            "CANN AddRmsNorm SingleN requires exactly the MTE2-read and "
            "MTE3-write paths")
    actual_path_bytes = {
        path_name: {
            node.resource_role: node.bytes for node in path.nodes
        }
        for path_name, path in path_by_name.items()
    }
    expected_path_bytes = {
        "mte2_read": {
            "mte2": float(6 * rows * hidden),
            "l2_cache": float(6 * rows * hidden),
            "dram": float(4 * rows * hidden + 2 * hidden),
        },
        "mte3_write": {
            "mte3": float(4 * rows * hidden + 4 * rows),
            "l2_cache": float(4 * rows * hidden + 4 * rows),
            "dram": float(4 * rows * hidden + 4 * rows),
        },
    }
    if actual_path_bytes != expected_path_bytes:
        raise ValueError(
            "CANN AddRmsNorm SingleN traffic disagrees with its BF16 source "
            f"contract: actual={actual_path_bytes}, "
            f"expected={expected_path_bytes}")

    mte2_service = "memory:mte2_read"
    mte3_service = "memory:mte3_write"
    vector_service = "compute:compute_vector"
    input_tensor_bytes = int(2 * hidden)
    gamma_l2_bytes = float(2 * hidden)
    # Every AIV issues one full gamma-vector read to L2, while the compulsory
    # DRAM footprint is one vector for the launch.  Keep those two ownership
    # scopes explicit instead of fabricating a fractional per-AIV DRAM miss.
    gamma_dram_bytes = float(2 * hidden)
    residual_bytes = float(2 * hidden)
    output_bytes = float(2 * hidden)
    rstd_bytes = 4.0
    reduction_dependency_fraction = (
        _cann_norm_reduction_dependency_fraction(inp, rows))
    reduction_terminal = (
        "reduction_dependency"
        if reduction_dependency_fraction is not None
        else "reduce_compute")
    input_pair_completion = (
        _cann_memory_completion_request(
            hardware, "mte2",
            _CANN_ADD_RMS_SINGLE_N_INPUT_PAIR_ONE_WORKER_EXACT_SERVICE,
            input_tensor_bytes)
        if rows == 1 and input_tensor_bytes in (
            4096, 5120, 8192, 10240, 16384) else ())
    if analytical_compact:
        # SingleN issues its two inputs as one source pair before the
        # following waits.  The available individual CopyIn completion
        # services do not represent that paired edge, so Forward retains the
        # conserved MTE/L2/DRAM demand and source DAG without composing two
        # incompatible exact requests.
        input_mte2_to_vector = ()
        gamma_mte2_to_vector = ()
        residual_mte3_to_vector = ()
    else:
        input_mte2_to_vector = _cann_unloaded_memory_completion_request(
            hardware, "mte2",
            _CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE,
            input_tensor_bytes, rows)
        gamma_mte2_to_vector = _cann_unloaded_memory_completion_request(
            hardware, "mte2",
            _CANN_DATACOPYPAD_MTE2_TO_VECTOR_EXACT_SERVICE,
            int(gamma_l2_bytes), rows)
        residual_mte3_to_vector = _cann_unloaded_memory_completion_request(
            hardware, "mte3",
            _CANN_DATACOPYPAD_VECTOR_TO_MTE3_EXACT_SERVICE,
            int(residual_bytes), rows)

    def _memory(path_name, **bytes_by_role):
        return tuple(
            PhaseMemoryDemand(path_name, role, byte_count)
            for role, byte_count in bytes_by_role.items())

    # BF16 source tally per row:
    # * input 4H: Cast(x1), Cast(x2), Add, Cast(residual for MTE3);
    # * reduce 5H: residual recast, square (mul+accumulate), Muls(mean), and
    #   ReduceSum, followed by the scalar epsilon add and one separately
    #   priced size-one Sqrt->Duplicate->Div exact request;
    # * output 6H: Muls(rstd), residual BF16/FP32 round-trip casts,
    #   Cast(gamma), Mul(gamma), and the final BF16 Cast.
    tasks = tuple(
        PipelineTaskDemand(
            name=f"worker{worker}_single_row",
            worker=worker,
            stages=(
                PipelineStageDemand(
                    "input_x1_load", mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read",
                        mte2=(2 * input_tensor_bytes
                              if input_pair_completion
                              else input_tensor_bytes),
                        l2_cache=(2 * input_tensor_bytes
                                  if input_pair_completion
                                  else input_tensor_bytes),
                        dram=(2 * input_tensor_bytes
                              if input_pair_completion
                              else input_tensor_bytes)),
                    memory_exact_request_demands=(
                        input_pair_completion or input_mte2_to_vector),
                    memory_exact_request_replaces_resource_capacity=bool(
                        input_pair_completion),
                    predecessors=()),
            ) + (() if input_pair_completion else (
                PipelineStageDemand(
                    "input_x2_load", mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read", mte2=input_tensor_bytes,
                        l2_cache=input_tensor_bytes,
                        dram=input_tensor_bytes),
                    memory_exact_request_demands=input_mte2_to_vector,
                    predecessors=(),
                    start_predecessors=("input_x1_load",)),
            )) + (
                PipelineStageDemand(
                    "input_x1_cast", vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "input_x1_cast"],
                    predecessors=("input_x1_load",)),
                PipelineStageDemand(
                    "input_compute", vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "input_compute"],
                    predecessors=(
                        ("input_x1_cast",) if input_pair_completion
                        else ("input_x1_cast", "input_x2_load"))),
                PipelineStageDemand(
                    "gamma_load", mte2_service, None,
                    memory_node_demands=_memory(
                        "mte2_read", mte2=gamma_l2_bytes,
                        l2_cache=gamma_l2_bytes),
                    launch_shared_memory_node_demands=_memory(
                        "mte2_read", dram=gamma_dram_bytes),
                    memory_exact_request_demands=gamma_mte2_to_vector),
                PipelineStageDemand(
                    "residual_store", mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=residual_bytes,
                        l2_cache=residual_bytes,
                        dram=residual_bytes),
                    memory_exact_request_demands=residual_mte3_to_vector,
                    predecessors=("input_compute",),
                    start_predecessors=("gamma_load",)),
                PipelineStageDemand(
                    "reduce_compute", vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "reduce_compute"],
                    predecessors=("input_compute",),
                    start_predecessors=("residual_store",)),
            ) + ((PipelineStageDemand(
                "reduction_dependency", "compute:compute_reduction",
                reduction_dependency_fraction,
                predecessors=("reduce_compute",)),)
                 if reduction_dependency_fraction is not None else ()) + (
                PipelineStageDemand(
                    "epsilon_add", vector_service, None,
                    compute_service_demands=row_stage_demands["epsilon_add"],
                    predecessors=(reduction_terminal,)),
                PipelineStageDemand(
                    "reciprocal_sqrt", vector_service, None,
                    compute_exact_request_demands=(
                        ComputeExactRequestDemand(
                            _CANN_NORM_EXACT_SERVICE, ((1,),)),),
                    predecessors=("epsilon_add",)),
                PipelineStageDemand(
                    "rstd_store", mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=rstd_bytes,
                        l2_cache=rstd_bytes,
                        dram=rstd_bytes),
                    predecessors=("reciprocal_sqrt",)),
                PipelineStageDemand(
                    "output_scale", vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "output_scale"],
                    predecessors=("reciprocal_sqrt",),
                    start_predecessors=("rstd_store",)),
                PipelineStageDemand(
                    "residual_roundtrip", vector_service, None,
                    compute_service_demands=row_stage_demands[
                        "residual_roundtrip"],
                    predecessors=("output_scale", "residual_store")),
                PipelineStageDemand(
                    "gamma_apply", vector_service, None,
                    compute_service_demands=row_stage_demands["gamma_apply"],
                    predecessors=(
                        "gamma_load", "rstd_store",
                        "residual_roundtrip")),
                PipelineStageDemand(
                    "final_cast", vector_service, None,
                    compute_service_demands=row_stage_demands["final_cast"],
                    predecessors=("gamma_apply",)),
                PipelineStageDemand(
                    "output_store", mte3_service, None,
                    memory_node_demands=_memory(
                        "mte3_write", mte3=output_bytes,
                        l2_cache=output_bytes,
                        dram=output_bytes),
                    predecessors=("final_cast",)),
            ),
        )
        for worker in (range(1) if analytical_compact else range(rows))
    )
    if analytical_compact:
        tasks = (PipelineTaskProgramDemand(
            name="single_n_worker_class",
            worker=0,
            patterns=(PipelineTaskSequenceDemand(
                name="single_n_sequence",
                worker=0,
                tasks=tasks,
            ),),
            order=(0,),
            worker_multiplicity=rows,
            physical_workers=tuple(range(rows)),
        ),)
    plan = PipelineWorkPlan(
        worker_count=rows,
        tasks=tasks,
        buffers=(PipelineBuffer(
            name="single_n_ub",
            producer_stage="input_x1_load",
            consumer_stage="output_store",
            slots=1,
            scope="worker",
        ),),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    limitations = (
        ("cann_add_rms_single_n_mte2_completion_evidence_pending",)
        if not input_mte2_to_vector or not gamma_mte2_to_vector else ())
    limitations += (
        ("cann_add_rms_single_n_residual_mte3_completion_evidence_pending",)
        if not residual_mte3_to_vector else ())
    return dataclasses.replace(
        attached,
        # The entry program is source-complete for both one-row and
        # multi-row SingleN launches.  Worker-entry fill is an independent
        # launch-wide admission envelope; it never changes whether the
        # source body is composed after the common empty-task shell.
        task_duration_floor_contract=AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
        aiv_worker_start_contract=AIV_WORKER_START_FILL,
        limitations=attached.limitations + limitations + (
            "cann_add_rms_single_n_rstd_and_terminal_mte3_completion_"
            "evidence_pending",
        ),
    )


def _block_reduction_first_service_weights(
        resource, block_threads: int,
        active_units: int) -> Tuple[Tuple[str, float], ...]:
    """Interpolate a first-CTA reduction over measured active AP counts."""

    def _service(point: int) -> str:
        suffix = "" if point == 1 else f"_active{point}"
        return f"block_reduce_first_{block_threads}{suffix}"

    services = tuple(_service(point)
                     for point in _BLOCK_REDUCTION_ACTIVE_UNIT_POINTS)
    available = tuple(resource.has_typed_service(service)
                      for service in services)
    if not any(available):
        raise ValueError(
            "compute_reduction lacks a first-CTA typed service")
    if not all(available):
        if available[0] and not any(available[1:]):
            return ((services[0], 1.0),)
        raise ValueError(
            "compute_reduction active-unit first-CTA curve is incomplete")
    points = _BLOCK_REDUCTION_ACTIVE_UNIT_POINTS
    if active_units < points[0] or active_units > points[-1]:
        raise ValueError(
            "block-reduction active units are outside the measured curve")
    if active_units in points:
        index = points.index(active_units)
        return ((services[index], 1.0),)
    upper = next(index for index, point in enumerate(points)
                 if active_units < point)
    lower = upper - 1
    upper_weight = (
        (active_units - points[lower])
        / (points[upper] - points[lower]))
    return (
        (services[lower], 1.0 - upper_weight),
        (services[upper], upper_weight),
    )


def _attach_source_bound_block_reduction(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach a source-equivalent GPU-local RMS reduction dependency."""

    parameter = _SOURCE_BOUND_BLOCK_REDUCTION_PARAMETER
    if parameter not in kernel.launch.parameters:
        return inp
    if kernel.launch.parameters[parameter] != 1.0:
        raise ValueError(
            "source-bound block reduction marker must equal one")
    variant = kernel.variant.lower()
    if (variant not in ("rmsnorm", "add_rmsnorm")
            or kernel.dtype != "bf16"
            or hardware.kernel_memory_model != "gpu_mshr"):
        raise ValueError(
            "source-bound block reduction requires GPU BF16 RMSNorm")
    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    block_threads = kernel.launch.block_size
    if (kernel.launch.grid != (rows, 1, 1)
            or kernel.launch.block != (block_threads, 1, 1)
            or hidden % 256 != 0
            and not (variant == "rmsnorm" and hidden == block_threads == 128)
            or block_threads not in _BLOCK_REDUCTION_RESIDENT_CTAS):
        # The measured resident-CTA reduction services cover only the
        # tail-free launch domain they were swept in.  Outside it the
        # mechanism does not mount and the kernel keeps its aggregate
        # representation, matching the repository rule that a mechanism
        # falls back to the general model beyond its measured domain.
        return inp
    resident_ctas = _BLOCK_REDUCTION_RESIDENT_CTAS[block_threads]
    if "compute_reduction" not in hardware._role_map:
        raise ValueError(
            "source-bound block reduction requires compute_reduction")
    resource = hardware.resource_by_role("compute_reduction")
    if (resource.total_units != hardware.num_compute_units
            or resource.service_rates_per_unit_s is None):
        raise ValueError(
            "source-bound block reduction requires one typed service "
            "resource per GPU SM")

    worker_count = inp.active_units
    base_ctas, extra_ctas = divmod(rows, worker_count)
    ctas_by_worker = tuple(
        base_ctas + (1 if worker < extra_ctas else 0)
        for worker in range(worker_count))
    if (sum(ctas_by_worker) != rows
            or max(ctas_by_worker) != math.ceil(
                rows / hardware.num_compute_units)):
        raise ValueError(
            "source-bound block reduction CTA-to-SM schedule is invalid")
    first_service_weights = _block_reduction_first_service_weights(
        resource, block_threads, inp.active_units)
    service_counts = tuple(
        (service, tuple(weight for _ in ctas_by_worker))
        for service, weight in first_service_weights) + (
        (f"block_reduce_resident_{block_threads}", tuple(
            float(max(min(count, resident_ctas) - 1, 0))
            for count in ctas_by_worker)),
        (f"block_reduce_queued_{block_threads}", tuple(
            float(max(count - resident_ctas, 0))
            for count in ctas_by_worker)),
    )
    service_demands = []
    for service, operations_by_worker in service_counts:
        operations = math.fsum(operations_by_worker)
        if operations == 0:
            continue
        if service not in resource.service_rates_per_unit_s:
            raise ValueError(
                f"compute_reduction lacks typed service {service!r}")
        service_demands.append(ComputeServiceDemand(
            service=service,
            operations=operations,
            operations_by_worker=operations_by_worker,
            serial_invocations_by_worker=(0,) * worker_count,
        ))
    if math.fsum(
            demand.operations for demand in service_demands) != float(rows):
        raise ValueError(
            "source-bound block reduction services do not conserve CTAs")

    reduction_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_reduction")
    if len(reduction_demands) != 1:
        raise ValueError(
            "source-bound block reduction requires one aggregate demand")
    reduction_demand = reduction_demands[0]
    if (reduction_demand.operations != float(rows)
            or reduction_demand.service_demands
            or reduction_demand.exact_request_demands):
        raise ValueError(
            "source-bound block reduction aggregate demand changed")
    typed_reduction = ComputeDemand(
        resource_role="compute_reduction",
        operations=float(rows),
        shape_efficiency=1.0,
        schedule_efficiency=1.0,
        service_demands=tuple(service_demands),
    )
    compute_demands = tuple(
        typed_reduction if demand is reduction_demand else demand
        for demand in inp.compute_demands)
    return dataclasses.replace(inp, compute_demands=compute_demands)


def _attach_a800_block256_add_rms_resident_cohort_floor(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        ctas_by_worker: Tuple[int, ...]) -> PredictionInput:
    """Retain the measured first resident cohort as a competing floor.

    The block-256 source-round campaign covers the first eight resident CTAs
    per SM, but it does not identify a stable queued-CTA assignment.  For a
    larger grid, keep the ordinary full-grid analytical pipeline and add only
    that measured resident cohort as an independent lower bound.  Later CTA
    admission overlaps already-running CTA work and is therefore never added
    as a serial scheduler refill.
    """

    mechanism = hardware.add_rmsnorm_block256_source_round_pipeline
    if (kernel.variant.lower() != "add_rmsnorm"
            or kernel.dtype != "bf16"
            or mechanism is None
            or mechanism["model"]
            != "sm80_bf16_add_rmsnorm_block256_resident_v1"
            or kernel.launch.block_size != mechanism["threads_per_cta"]
            or _shape(kernel, "rows") <= mechanism["maximum_grid_blocks"]):
        return inp
    resident_ctas = _thread_limited_resident_ctas(kernel, hardware)
    if (resident_ctas != mechanism["resident_ctas_per_sm"]
            or min(ctas_by_worker) < resident_ctas
            or inp.active_units != hardware.num_compute_units):
        return inp

    hidden = _shape(kernel, "hidden")
    vector_packs = hidden // mechanism["vector_width_elements"]
    full_rounds, tail_threads = divmod(
        vector_packs, mechanism["threads_per_cta"])
    if hidden % 256 != 0 or full_rounds <= 0:
        return inp
    round_widths = [((mechanism["threads_per_cta"], 1.0),)] * full_rounds
    if tail_threads:
        if tail_threads not in mechanism["full_tail_pair_active_threads"]:
            return inp
        round_widths[-1] = ((tail_threads, 2.0),)
    round_widths = tuple(round_widths)

    resource = hardware.resource_by_role("compute_vector")

    def _service(function: str, regime: str, threads: int) -> str:
        name = f"bf16_{function}_{regime}_{threads}_block256"
        if not resource.has_typed_service(name):
            raise ValueError(
                "A800 block-256 resident-cohort source service is "
                f"unavailable: {name!r}")
        return name

    operations_by_service = {}
    regime_counts = {
        "first": inp.active_units,
        "resident": inp.active_units * (resident_ctas - 1),
    }
    for function in _BF16_VECTOR_ROUND_FUNCTIONS:
        for regime, cta_count in regime_counts.items():
            for widths in round_widths:
                for threads, weight in widths:
                    name = _service(function, regime, threads)
                    if name not in operations_by_service:
                        operations_by_service[name] = 0.0
                    operations_by_service[name] += cta_count * weight
    total_operations = math.fsum(operations_by_service.values())
    if (inp.demand_per_resource["compute_vector"] != 0.0
            or any(demand.resource_role == "compute_vector"
                   for demand in inp.compute_demands)):
        raise ValueError(
            "A800 block-256 resident-cohort floor requires an unassigned "
            "compute_vector resource")
    participating_warps = min(
        math.ceil(hidden / 256),
        mechanism["threads_per_cta"] // hardware.parallel_width)
    cohort_ctas = inp.active_units * resident_ctas
    prologue_operations = float(22 * cohort_ctas * participating_warps)
    if (inp.demand_per_resource["warp_issue"] != 0.0
            or any(demand.resource_role == "warp_issue"
                   for demand in inp.compute_demands)):
        raise ValueError(
            "A800 block-256 resident-cohort floor requires an unassigned "
            "warp_issue resource")
    source_demand = ComputeDemand(
        resource_role="compute_vector",
        operations=total_operations,
        shape_efficiency=1.0,
        schedule_efficiency=1.0,
        service_demands=tuple(
            ComputeServiceDemand(name, operations)
            for name, operations in operations_by_service.items()),
    )
    prologue_demand = ComputeDemand(
        resource_role="warp_issue",
        operations=prologue_operations,
        shape_efficiency=1.0,
        schedule_efficiency=1.0,
    )
    demand_per_resource = dict(inp.demand_per_resource)
    demand_per_resource["compute_vector"] = total_operations
    demand_per_resource["warp_issue"] = prologue_operations
    input_operations = total_operations / 2.0
    phase_updates = {
        "input_compute": (
            "bf16_add_square_resident_cohort_floor",
            (PhaseComputeDemand(
                "warp_issue",
                "add_rmsnorm_resident_cohort_source_prologue",
                prologue_operations),),
        ),
        "output_compute": (
            "bf16_scale_multiply_resident_cohort_floor",
            (),
        ),
    }
    updated_phases = []
    for phase in inp.phase_ledger.phases:
        if phase.name not in phase_updates:
            updated_phases.append(phase)
            continue
        operation, extra_demands = phase_updates[phase.name]
        updated_phases.append(dataclasses.replace(
            phase,
            compute_demands=phase.compute_demands + (
                PhaseComputeDemand(
                    "compute_vector", operation, input_operations),
            ) + extra_demands,
        ))
    phase_ledger = dataclasses.replace(
        inp.phase_ledger,
        phases=tuple(updated_phases),
    )

    tasks = []
    for cta_index in range(resident_ctas):
        regime = "first" if cta_index == 0 else "resident"
        stages = [PipelineStageDemand(
            "source_prologue",
            "compute:warp_issue",
            1.0 / cohort_ctas,
        )]
        prior = "source_prologue"
        for function in _BF16_VECTOR_ROUND_FUNCTIONS:
            for round_index, widths in enumerate(round_widths):
                stage_name = f"{function}_{round_index}"
                stages.append(PipelineStageDemand(
                    stage_name,
                    "compute:compute_vector",
                    None,
                    compute_service_demands=tuple(
                        ComputeServiceDemand(
                            _service(function, regime, threads), weight)
                        for threads, weight in widths),
                    predecessors=(() if prior is None else (prior,)),
                ))
                prior = stage_name
        tasks.append(PipelineTaskDemand(
            name=f"resident_cta{cta_index}", worker=0,
            stages=tuple(stages)))
    sequence = PipelineTaskSequenceDemand(
        name="resident_cohort_sequence", worker=0, tasks=tuple(tasks))
    floor = PipelineWorkPlan(
        worker_count=inp.active_units,
        tasks=(PipelineTaskProgramDemand(
            name="resident_cohort_program",
            worker=0,
            patterns=(sequence,),
            order=(0,),
            worker_multiplicity=inp.active_units,
            physical_workers=tuple(range(inp.active_units)),
        ),),
        buffers=(PipelineBuffer(
            name="resident_cta_slot",
            producer_stage="source_prologue",
            consumer_stage=(
                f"scale_multiply_{len(round_widths) - 1}"),
            slots=1,
            scope="worker",
        ),),
    )
    return dataclasses.replace(
        inp,
        demand_per_resource=demand_per_resource,
        compute_demands=(
            inp.compute_demands + (source_demand, prologue_demand)),
        phase_ledger=phase_ledger,
        competing_pipeline_floor=floor,
        limitations=inp.limitations + (
            "a800_block256_add_rms_queued_ctas_use_static_source_pipeline",
            "a800_block256_add_rms_queued_cta_typed_scalar_service_unavailable",
        ),
    )


def _attach_source_bound_gpu_norm_cta_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the source loop and CTA-residency program for GPU RMSNorm.

    One worker represents one active compute unit. Its ordered tasks are the
    CTAs assigned by the launch, while a worker-local buffer limits concurrent
    CTA lifetimes to the statically derived residency. Within each CTA the
    scalar or width-eight source loops are expanded only into Load, ALU/FMA,
    Reduction, and Store rounds; no backend instruction stream is consumed.
    """

    parameter = SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER
    typed_add_rms_rounds = _has_typed_add_rms_round_services(
        kernel, hardware)
    typed_add_rms_scalar_rounds = (
        _has_c500_add_rmsnorm_scalar_first_services(kernel, hardware))
    typed_rmsnorm_scalar_rounds = (
        _has_typed_rmsnorm_scalar_round_services(kernel, hardware))
    if (parameter not in kernel.launch.parameters
            and not typed_add_rms_rounds
            and not typed_rmsnorm_scalar_rounds):
        return inp
    if (parameter in kernel.launch.parameters
            and kernel.launch.parameters[parameter] != 1.0):
        raise ValueError(
            "source-bound GPU NORM CTA pipeline marker must equal one")
    variant = kernel.variant.lower()
    if (variant not in ("rmsnorm", "add_rmsnorm")
            or kernel.dtype != "bf16"
            or inp.mem_model != "gpu_mshr"
            or inp.pipeline_plan is not None
            or inp.phase_ledger is None):
        raise ValueError(
            "source-bound GPU NORM CTA pipeline requires one structured "
            "BF16 GPU normalization input")
    typed_block_reduction = (
        _SOURCE_BOUND_BLOCK_REDUCTION_PARAMETER in kernel.launch.parameters)
    if variant == "rmsnorm" and not typed_block_reduction:
        raise ValueError(
            "source-bound RMSNorm CTA pipeline requires typed reduction")
    legacy_program_services = (
        "compute_vector" in hardware._role_map
        and hardware.resource_by_role("compute_vector").has_typed_service(
            "rmsnorm_input_square_resident_128"))
    if variant == "rmsnorm" and legacy_program_services:
        program_parameter = (
            SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER)
        if (program_parameter not in kernel.launch.parameters
                or kernel.launch.parameters[program_parameter] != 1.0):
            raise ValueError(
                "source-bound RMSNorm CTA pipeline requires the versioned "
                "vLLM 0.8.5 / MACA 3.7.2 program identity")
    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    legacy_program_vector_cycles_per_cta = (
        _c500_legacy_rmsnorm_program_vector_cycles_per_cta(
            kernel, hardware))
    if legacy_program_vector_cycles_per_cta is not None:
        if any(demand.resource_role == "warp_issue"
               for demand in inp.compute_demands):
            raise ValueError(
                "legacy C500 RMSNorm program issue requires an unassigned "
                "warp-issue resource")
        if inp.demand_per_resource["warp_issue"] != 0.0:
            raise ValueError(
                "legacy C500 RMSNorm program issue demand must start at zero")
        # ``warp_dispatch`` exposes the aggregate five-slot AP envelope. Five
        # generic dispatch operations represent one normalized Vector cycle
        # without inventing a second whole-program resource or allowing the
        # other four issue slots to raise this one-slot floor.
        issue_operations = float(
            5.0 * rows * legacy_program_vector_cycles_per_cta)
        demand_per_resource = dict(inp.demand_per_resource)
        demand_per_resource["warp_issue"] = issue_operations
        issue_phase_demand = PhaseComputeDemand(
            resource_role="warp_issue",
            operation="legacy_rmsnorm_program_vector_cycle_floor",
            operations=issue_operations,
        )
        phase_ledger = dataclasses.replace(
            inp.phase_ledger,
            unassigned_compute_demands=(
                inp.phase_ledger.unassigned_compute_demands
                + (issue_phase_demand,)),
            limitations=tuple(dict.fromkeys(
                inp.phase_ledger.limitations + (
                    "legacy_rmsnorm_program_issue_phase_partition_unavailable",
                ))),
        )
        inp = dataclasses.replace(
            inp,
            demand_per_resource=demand_per_resource,
            compute_demands=inp.compute_demands + (ComputeDemand(
                resource_role="warp_issue",
                operations=issue_operations,
                shape_efficiency=1.0,
                schedule_efficiency=1.0,
            ),),
            phase_ledger=phase_ledger,
        )
    typed_add_rms_rounds = (
        variant == "add_rmsnorm" and typed_add_rms_rounds)
    typed_rmsnorm_scalar_rounds = (
        variant == "rmsnorm" and typed_rmsnorm_scalar_rounds)
    typed_norm_rounds = (
        typed_add_rms_rounds or typed_rmsnorm_scalar_rounds)
    rmsnorm_phase_chain_excess_service = None
    if (typed_rmsnorm_scalar_rounds
            and inp.active_units
            in _C500_RMSNORM_PHASE_CHAIN_EXCESS_SERVICES):
        rmsnorm_phase_chain_excess_service = (
            _C500_RMSNORM_PHASE_CHAIN_EXCESS_SERVICES[inp.active_units])
    generic_add_rms_round_services = (
        typed_add_rms_rounds
        and hardware.add_rmsnorm_source_round_pipeline is None
        and hardware.add_rmsnorm_block256_source_round_pipeline is None)
    c500_grid64_fixed_input_service = (
        _c500_add_rmsnorm_grid64_fixed_input_service(kernel, hardware)
        if generic_add_rms_round_services else None)
    c500_grid64_fixed_input_selected = False
    block_threads = kernel.launch.block_size
    if (kernel.launch.grid != (rows, 1, 1)
            or kernel.launch.block != (block_threads, 1, 1)):
        raise ValueError(
            "source-bound GPU NORM CTA pipeline requires one CTA per row")
    if block_threads not in _BLOCK_REDUCTION_RESIDENT_CTAS:
        # The swept resident-CTA services cover only the block sizes they
        # were measured at.  Outside that domain the CTA pipeline does not
        # mount and the kernel keeps its aggregate representation, per the
        # repository rule that a mechanism falls back to the general model
        # beyond its measured domain.
        return inp
    resident_ctas = _thread_limited_resident_ctas(kernel, hardware)
    if inp.active_units != min(rows, hardware.num_compute_units):
        raise ValueError(
            "source-bound GPU NORM CTA pipeline active-unit count changed")

    phases = {phase.name: phase for phase in inp.phase_ledger.phases}
    expected_phases = {
        "load_reduce", "input_compute", "reduce_compute", "load_output",
        "output_compute", "store",
    }
    if variant == "add_rmsnorm":
        expected_phases.add("residual_store")
    if set(phases) != expected_phases:
        raise ValueError(
            "source-bound GPU NORM CTA pipeline phase schema changed: "
            f"{sorted(phases)}")

    compute_totals = {
        demand.resource_role: demand.operations
        for demand in inp.compute_demands
    }
    path_l2_totals = {
        path.name: next(
            node.bytes for node in path.nodes
            if node.resource_role == "l2_cache")
        for path in inp.memory_paths
    }
    def _phase_compute_operations(phase_name: str, role: str) -> float:
        return math.fsum(
            demand.operations
            for demand in phases[phase_name].compute_demands
            if demand.resource_role == role)

    def _phase_memory_bytes(
            phase_name: str, path_name: str, resource_role: str) -> float:
        return math.fsum(
            demand.bytes
            for demand in phases[phase_name].memory_demands
            if (demand.path_name == path_name
                and demand.resource_role == resource_role))

    if variant == "rmsnorm":
        vector_width = 1
        loop_elements = math.ceil(hidden / vector_width)
        loop_rounds = math.ceil(loop_elements / block_threads)
        add_rms_round_widths = ()
    elif variant == "add_rmsnorm":
        if typed_add_rms_scalar_rounds:
            loop_rounds = hidden // block_threads
            tail_threads = 0
            add_rms_round_widths = ()
        else:
            vector_width = 8
            loop_elements = math.ceil(hidden / vector_width)
            full_rounds, tail_threads = divmod(loop_elements, block_threads)
            round_widths = [((block_threads, 1.0),)] * full_rounds
            if tail_threads:
                additional_widths = _add_rmsnorm_additional_round_widths(
                    hardware, block_threads)
                if (typed_add_rms_rounds and block_threads == 256
                        and tail_threads in additional_widths):
                    if not round_widths:
                        raise ValueError(
                            "block-256 Add-RMSNorm full+tail service requires "
                            "one complete leading source round")
                    round_widths[-1] = ((tail_threads, 2.0),)
                else:
                    # Exact measured widths cover production source rounds.
                    # Only an unmeasured partial source iteration is
                    # interpolated between adjacent mechanism points.
                    round_widths.append(_bf16_vector_round_widths(
                        tail_threads, block_threads, additional_widths))
            add_rms_round_widths = tuple(round_widths)
            loop_rounds = len(add_rms_round_widths)
    else:
        raise AssertionError(f"unexpected GPU NORM variant: {variant!r}")
    if loop_rounds <= 0:
        raise ValueError("GPU NORM source loop must contain a round")
    c500_complete_phase_chain_service = None
    c500_complete_phase_chain_rounds = 0
    complete_phase_chain_candidate = (
        _C500_COMPLETE_NORM_PHASE_CHAIN_SERVICES[variant])
    complete_phase_chain_depth = hidden // 1024
    if (typed_norm_rounds
            and block_threads == 1024
            and hidden % 1024 == 0
            and rows == inp.active_units
            and _C500_COMPLETE_NORM_PHASE_CHAIN_MIN_DEPTH
            <= complete_phase_chain_depth
            <= _C500_COMPLETE_NORM_PHASE_CHAIN_MAX_DEPTH
            and hardware.resource_by_role(
                "compute_vector").has_typed_service(
                    complete_phase_chain_candidate)):
        # A CTA-local clock64 probe measures the complete source loop chain,
        # including the publication barrier but excluding task launch and the
        # production reduction.  Use it only when every active AP owns one
        # first CTA, so no resident/queued scheduling is collapsed.  This one
        # service replaces isolated input/output and residual terms; memory
        # traffic and the real block reduction remain independent bounds.
        c500_complete_phase_chain_service = (
            complete_phase_chain_candidate)
        c500_complete_phase_chain_rounds = complete_phase_chain_depth
        rmsnorm_phase_chain_excess_service = None
    rmsnorm_phase_entry_service = None
    if (c500_complete_phase_chain_service is None
            and typed_rmsnorm_scalar_rounds
            and rows == inp.active_units
            and loop_rounds >= _RMSNORM_PHASE_ENTRY_MIN_DEPTH):
        if inp.active_units in _RMSNORM_PHASE_ENTRY_SERVICES:
            rmsnorm_phase_entry_service = (
                _RMSNORM_PHASE_ENTRY_SERVICES[inp.active_units])
        elif (inp.active_units == 64
                and hardware.resource_by_role(
                    "compute_vector").has_typed_service(
                        _RMSNORM_PHASE_ENTRY_ACTIVE64_SERVICE)):
            rmsnorm_phase_entry_service = (
                _RMSNORM_PHASE_ENTRY_ACTIVE64_SERVICE)
    add_rmsnorm_phase_chain_excess_service = None
    add_rmsnorm_scalar_round_depth = 0
    if (c500_complete_phase_chain_service is None
            and typed_add_rms_rounds
            and generic_add_rms_round_services
            and block_threads == 1024
            and hidden % block_threads == 0
            and rows == inp.active_units
            and _C500_ADD_RMSNORM_PHASE_CHAIN_MIN_DEPTH
            <= hidden // block_threads
            <= _C500_ADD_RMSNORM_PHASE_CHAIN_MAX_DEPTH
            and inp.active_units
            in _C500_ADD_RMSNORM_PHASE_CHAIN_EXCESS_SERVICES):
        add_rmsnorm_phase_chain_excess_service = (
            _C500_ADD_RMSNORM_PHASE_CHAIN_EXCESS_SERVICES[inp.active_units])
        add_rmsnorm_scalar_round_depth = hidden // block_threads
    c500_add_rmsnorm_block256_scalar_input = (
        typed_add_rms_rounds
        and generic_add_rms_round_services
        and block_threads == 256
        and hidden % block_threads == 0
        and math.ceil(rows / inp.active_units) <= resident_ctas
        and inp.active_units
        in _C500_ADD_RMSNORM_BLOCK256_SCALAR_ACTIVE_UNITS
        and _C500_ADD_RMSNORM_BLOCK256_SCALAR_MIN_DEPTH
        <= hidden // block_threads
        <= _C500_ADD_RMSNORM_BLOCK256_SCALAR_MAX_DEPTH
        and all(
            hardware.resource_by_role("compute_vector").has_typed_service(
                service)
            for service
            in _C500_ADD_RMSNORM_BLOCK256_SCALAR_INPUT_SERVICES.values()))
    read_l2_total = path_l2_totals["global_read"]
    write_l2_total = path_l2_totals["global_write"]
    if read_l2_total <= 0 or write_l2_total <= 0:
        raise ValueError("GPU NORM CTA pipeline requires read and write work")
    if variant == "add_rmsnorm" and not typed_add_rms_rounds:
        load_service = "memory:global_read"
        store_service = "memory:global_write"
        include_first_request_latency = False
        load_reduce_fraction = (
            _phase_memory_bytes("load_reduce", "global_read", "l2_cache")
            / read_l2_total / rows / loop_rounds)
        load_output_fraction = (
            _phase_memory_bytes("load_output", "global_read", "l2_cache")
            / read_l2_total / rows / loop_rounds)
        residual_store_fraction = (
            _phase_memory_bytes(
                "residual_store", "global_write", "l2_cache")
            / write_l2_total / rows / loop_rounds)
        final_store_fraction = (
            _phase_memory_bytes("store", "global_write", "l2_cache")
            / write_l2_total / rows / loop_rounds)
    elif variant == "rmsnorm":
        load_service = "memory:global_read"
        store_service = "memory:global_write"
        include_first_request_latency = True
        load_reduce_fraction = (
            _phase_memory_bytes("load_reduce", "global_read", "l2_cache")
            / read_l2_total / rows / loop_rounds)
        load_output_fraction = (
            _phase_memory_bytes("load_output", "global_read", "l2_cache")
            / read_l2_total / rows / loop_rounds)
        residual_store_fraction = 0.0
        final_store_fraction = (
            _phase_memory_bytes("store", "global_write", "l2_cache")
            / write_l2_total / rows / loop_rounds)
    elif not typed_add_rms_rounds:
        raise AssertionError(f"unexpected GPU NORM variant: {variant!r}")

    # Each CTA retains the same static input payload after the grid exceeds
    # the SM count.  The output gamma path is local only when the static
    # launch program declares exactly one L2 weight copy per active SM; a
    # later resident CTA then reuses that SM-local gamma allocation instead
    # of receiving a fictitious ``total/rows`` slice.  Other source programs
    # retain their aggregate representation until the reload schedule is
    # known.
    local_memory_source_completion = (
        not typed_norm_rounds
        and hardware.l1_cache_enabled
        and "norm_weight_l2_copies" in kernel.launch.parameters
        and int(kernel.launch.parameters["norm_weight_l2_copies"])
        == inp.active_units)
    if local_memory_source_completion:
        read_path, = tuple(
            path for path in inp.memory_paths
            if path.name == "global_read")
        if (read_path.rate_unit != "transactions"
                or tuple(node.resource_role for node in read_path.nodes)
                != ("l2_cache", "dram")):
            raise ValueError(
                "GPU normalization local source completion requires one "
                "transaction-resolved L2-to-DRAM read path")
        load_reduce_l2_bytes = _phase_memory_bytes(
            "load_reduce", "global_read", "l2_cache")
        load_reduce_dram_bytes = _phase_memory_bytes(
            "load_reduce", "global_read", "dram")
        load_output_l2_bytes = _phase_memory_bytes(
            "load_output", "global_read", "l2_cache")
        load_output_dram_bytes = _phase_memory_bytes(
            "load_output", "global_read", "dram")
        # The audited local source-completion contract covers only the
        # all-or-nothing first-pass pattern -- the row stream either misses
        # L2 completely (every L2 byte also reaches DRAM) or is fully
        # L2-resident (no DRAM byte at all) -- with exactly one gamma L2
        # copy per active SM and one launch-shared DRAM copy.  A graded
        # operand-residency split sits outside that measured domain, so the
        # source keeps its aggregate representation instead of mounting the
        # local contract.
        local_memory_source_completion = (
            (math.isclose(
                load_reduce_l2_bytes, load_reduce_dram_bytes,
                rel_tol=1e-12, abs_tol=1e-6)
             or load_reduce_dram_bytes == 0.0)
            and math.isclose(
                load_output_l2_bytes,
                inp.active_units * load_output_dram_bytes,
                rel_tol=1e-12, abs_tol=1e-6))
    if local_memory_source_completion:

        def _local_read_round_demands(phase_name: str, regime: str):
            if phase_name == "load_reduce":
                local = (
                    PhaseMemoryDemand(
                        "global_read", "l2_cache",
                        load_reduce_l2_bytes / rows / loop_rounds),
                )
                if load_reduce_dram_bytes > 0.0:
                    local += (
                        PhaseMemoryDemand(
                            "global_read", "dram",
                            load_reduce_dram_bytes / rows / loop_rounds),
                    )
                shared = ()
            elif phase_name == "load_output":
                local = (
                    (PhaseMemoryDemand(
                        "global_read", "l2_cache",
                        load_output_l2_bytes / inp.active_units / loop_rounds),)
                    if regime == "first" else ())
                shared = (
                    (PhaseMemoryDemand(
                        "global_read", "dram",
                        load_output_dram_bytes / loop_rounds),)
                    if regime == "first" else ())
            else:
                raise ValueError(
                    "unknown GPU normalization local read phase: "
                    f"{phase_name!r}")
            return local, shared
    else:
        def _local_read_round_demands(phase_name: str, regime: str):
            raise ValueError(
                "GPU normalization local source completion is unavailable")

    reduction_fma_fraction = 0.0
    reduction_sfu_fraction = 0.0
    if typed_norm_rounds:
        if any(
                demand.resource_role in ("compute_fma", "compute_alu")
                for demand in inp.compute_demands):
            raise ValueError(
                "typed normalization rounds must replace generic FMA/ALU "
                "work")
        input_fma_fraction = 0.0
        output_fma_fraction = 0.0
        input_alu_fraction = 0.0
        output_alu_fraction = 0.0
    else:
        fma_total = compute_totals["compute_fma"]
        alu_total = 0.0
        if "compute_alu" in compute_totals:
            alu_total = compute_totals["compute_alu"]
        sfu_total = 0.0
        if "compute_sfu" in compute_totals:
            sfu_total = compute_totals["compute_sfu"]
        input_fma_operations = _phase_compute_operations(
            "input_compute", "compute_fma")
        reduction_fma_operations = _phase_compute_operations(
            "reduce_compute", "compute_fma")
        output_fma_operations = _phase_compute_operations(
            "output_compute", "compute_fma")
        input_alu_operations = _phase_compute_operations(
            "input_compute", "compute_alu")
        output_alu_operations = _phase_compute_operations(
            "output_compute", "compute_alu")
        reduction_sfu_operations = _phase_compute_operations(
            "reduce_compute", "compute_sfu")
        if not math.isclose(
                input_fma_operations + reduction_fma_operations
                + output_fma_operations,
                fma_total, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "GPU NORM CTA pipeline does not conserve FMA work")
        if not math.isclose(
                input_alu_operations + output_alu_operations,
                alu_total, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "GPU NORM CTA pipeline does not conserve ALU work")
        if not math.isclose(
                reduction_sfu_operations, sfu_total,
                rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "GPU NORM CTA pipeline does not conserve SFU work")
        input_fma_fraction = (
            input_fma_operations / fma_total / rows / loop_rounds)
        reduction_fma_fraction = (
            reduction_fma_operations / fma_total / rows)
        output_fma_fraction = (
            output_fma_operations / fma_total / rows / loop_rounds)
        if alu_total > 0:
            input_alu_fraction = (
                input_alu_operations / alu_total / rows / loop_rounds)
            output_alu_fraction = (
                output_alu_operations / alu_total / rows / loop_rounds)
        else:
            input_alu_fraction = 0.0
            output_alu_fraction = 0.0
        if sfu_total > 0:
            reduction_sfu_fraction = (
                reduction_sfu_operations / sfu_total / rows)
        else:
            reduction_sfu_fraction = 0.0

    if typed_block_reduction:
        reduction = next(
            demand for demand in inp.compute_demands
            if demand.resource_role == "compute_reduction")
        services = {
            demand.service: demand for demand in reduction.service_demands}
    else:
        services = {}

    base_ctas, extra_ctas = divmod(rows, inp.active_units)
    ctas_by_worker = tuple(
        base_ctas + (1 if worker < extra_ctas else 0)
        for worker in range(inp.active_units))
    if sum(ctas_by_worker) != rows or min(ctas_by_worker) <= 0:
        raise ValueError("GPU NORM CTA assignment does not conserve rows")
    completion_service = None
    if (legacy_program_vector_cycles_per_cta is not None
            and block_threads == 128):
        completion_service = _C500_LEGACY_RMSNORM_COMPLETION_SERVICE
        vector_resource = hardware.resource_by_role("compute_vector")
        if (vector_resource.exact_request_services is None
                or completion_service
                not in vector_resource.exact_request_services):
            raise ValueError(
                "legacy C500 block128 RMSNorm completion service is "
                f"unavailable: {completion_service!r}")
    def _round_service(function: str, regime: str, threads: int) -> str:
        if regime not in _BF16_VECTOR_ROUND_REGIMES:
            raise ValueError("unknown BF16 Vector round service identity")
        if typed_rmsnorm_scalar_rounds:
            if function == "phase_chain_excess":
                if (regime != "first"
                        or rmsnorm_phase_chain_excess_service is None):
                    raise ValueError(
                        "RMSNorm phase-chain excess is available only for "
                        "its exact first-CTA active-unit classes")
                return rmsnorm_phase_chain_excess_service
            if (function not in _RMSNORM_SCALAR_ROUND_FUNCTIONS
                    or threads not in (128, 1024)):
                raise ValueError(
                    "unknown RMSNorm scalar round service identity")
            return f"rmsnorm_{function}_{regime}_{threads}"
        if typed_add_rms_scalar_rounds:
            if (regime != "first" or threads != 1024
                    or function
                    not in _C500_ADD_RMSNORM_SCALAR_FIRST_SERVICES):
                raise ValueError(
                    "Add-RMSNorm scalar round service is available only for "
                    "its exact first-CTA block1024 domain")
            return _C500_ADD_RMSNORM_SCALAR_FIRST_SERVICES[function]
        if (function not in _BF16_VECTOR_ROUND_FUNCTIONS
                or threads not in (64, 128, 256, 320, 512, 640, 1024)):
            raise ValueError("unknown BF16 Vector round service identity")
        service = f"bf16_{function}_{regime}_{threads}"
        if (hardware.add_rmsnorm_source_round_pipeline is not None
                or hardware.add_rmsnorm_block256_source_round_pipeline
                is not None):
            service = f"{service}_block{block_threads}"
        return service

    def _round_demands(
            function: str, regime: str,
            round_index: int) -> Tuple[ComputeServiceDemand, ...]:
        if typed_rmsnorm_scalar_rounds:
            if round_index < 0 or round_index >= loop_rounds:
                raise ValueError("RMSNorm scalar round index is invalid")
            active_threads = min(
                block_threads,
                hidden - round_index * block_threads,
            )
            # The final scalar-loop iteration executes only its statically
            # active waves.  Scale the already measured complete-wave service
            # by that exact active-thread fraction; CTA residency and the
            # block reduction remain separate physical constraints.
            return (ComputeServiceDemand(
                _round_service(function, regime, block_threads),
                active_threads / block_threads),)
        if typed_add_rms_scalar_rounds:
            if round_index < 0 or round_index >= loop_rounds:
                raise ValueError("Add-RMSNorm scalar round index is invalid")
            return (ComputeServiceDemand(
                _round_service(function, regime, block_threads), 1.0),)
        if (c500_add_rmsnorm_block256_scalar_input
                and function == "add_square"):
            scalar_rounds = math.fsum(
                threads * width_weight * 8.0 / block_threads
                for threads, width_weight
                in add_rms_round_widths[round_index])
            return (ComputeServiceDemand(
                _C500_ADD_RMSNORM_BLOCK256_SCALAR_INPUT_SERVICES[regime],
                scalar_rounds),)
        demands = []
        for threads, width_weight in add_rms_round_widths[round_index]:
            if width_weight <= 0:
                continue
            if regime == "first":
                curve = _add_rmsnorm_active_unit_service_weights(
                    hardware, function, threads, inp.active_units,
                    block_threads)
            elif regime in {"resident", "queued"}:
                curve = ()
            else:
                raise ValueError(f"unknown scalar-round regime: {regime}")
            if curve:
                demands.extend(
                    ComputeServiceDemand(service, width_weight * weight)
                    for service, weight in curve if weight > 0)
            else:
                demands.append(ComputeServiceDemand(
                    _round_service(function, regime, threads), width_weight))
        return tuple(demands)

    if typed_norm_rounds:
        if "compute_vector" not in hardware._role_map:
            raise ValueError(
                "source-bound normalization rounds require compute_vector")
        vector_resource = hardware.resource_by_role("compute_vector")
        regime_counts = {
            "first": float(len(ctas_by_worker)),
            "resident": float(math.fsum(
                max(min(count, resident_ctas) - 1, 0)
                for count in ctas_by_worker)),
            "queued": float(math.fsum(
                max(count - resident_ctas, 0)
                for count in ctas_by_worker)),
        }
        if not math.isclose(
                math.fsum(regime_counts.values()), rows,
                rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "normalization round regimes do not conserve CTAs")
        round_service_demands = []
        if typed_rmsnorm_scalar_rounds:
            round_functions = _RMSNORM_SCALAR_ROUND_FUNCTIONS
        elif typed_add_rms_scalar_rounds:
            round_functions = tuple(
                _C500_ADD_RMSNORM_SCALAR_FIRST_SERVICES)
        else:
            round_functions = _BF16_VECTOR_ROUND_FUNCTIONS
        if rmsnorm_phase_chain_excess_service is not None:
            round_functions += ("phase_chain_excess",)
        for function in round_functions:
            for regime in _BF16_VECTOR_ROUND_REGIMES:
                if regime_counts[regime] == 0:
                    continue
                operations_by_service = {}
                for round_index in range(loop_rounds):
                    for demand in _round_demands(
                            function, regime, round_index):
                        if not vector_resource.has_typed_service(
                                demand.service):
                            raise ValueError(
                                "source-bound normalization round service is "
                                f"unavailable: {demand.service!r}")
                        contribution = (
                            regime_counts[regime] * demand.operations)
                        if demand.service in operations_by_service:
                            operations_by_service[demand.service] += (
                                contribution)
                        else:
                            operations_by_service[demand.service] = (
                                contribution)
                round_service_demands.extend(
                    ComputeServiceDemand(service, operations)
                    for service, operations in operations_by_service.items()
                    if operations > 0)
        if (c500_grid64_fixed_input_service is not None
                and c500_complete_phase_chain_service is None):
            if (regime_counts["first"] != float(rows)
                    or regime_counts["resident"] != 0.0
                    or regime_counts["queued"] != 0.0):
                raise ValueError(
                    "C500 grid64 fixed AddRMSNorm input service requires "
                    "one first CTA on every active AP")
            execution_domain_units = hardware.execution_units_for_task(
                inp.launch_task_kind)
            input_operations_by_service = {}
            for round_index in range(loop_rounds):
                for demand in _round_demands(
                        "add_square", "first", round_index):
                    if demand.service in input_operations_by_service:
                        input_operations_by_service[demand.service] += (
                            demand.operations)
                    else:
                        input_operations_by_service[demand.service] = (
                            demand.operations)
            current_input_duration_s = math.fsum(
                rows * operations
                / vector_resource.effective_service_capacity(
                    service,
                    active_units=inp.active_units,
                    execution_domain_units=execution_domain_units)
                for service, operations
                in input_operations_by_service.items())
            participating_warps = min(
                math.ceil(hidden / 256),
                block_threads // hardware.parallel_width)
            prologue_operations = float(
                _C500_ADD_RMSNORM_GENERIC_FIXED_INSTRUCTIONS_PER_WAVE
                * rows * participating_warps)
            prologue_duration_s = (
                prologue_operations
                / hardware.resource_by_role("warp_issue").effective_capacity(
                    inp.active_units, inp.sm_freq_hz, inp.dram_freq_hz))
            fixed_input_duration_s = (
                rows
                / vector_resource.effective_service_capacity(
                    c500_grid64_fixed_input_service,
                    active_units=inp.active_units,
                    execution_domain_units=execution_domain_units))
            # This is an analytical max, not an error-selected replacement.
            # Retain the ordinary prologue -> input branch unless the exact
            # fixed-address complete-loop floor is physically longer.  When
            # it wins, the prologue remains a parallel lower bound so neither
            # source setup nor CTA admission is serially charged twice.
            if fixed_input_duration_s > (
                    prologue_duration_s + current_input_duration_s):
                c500_grid64_fixed_input_selected = True
                round_service_demands = [
                    demand for demand in round_service_demands
                    if demand.service not in input_operations_by_service]
                round_service_demands.append(ComputeServiceDemand(
                    c500_grid64_fixed_input_service, float(rows)))
        if c500_complete_phase_chain_service is not None:
            round_service_demands = [ComputeServiceDemand(
                c500_complete_phase_chain_service,
                float(rows * c500_complete_phase_chain_rounds))]
        if rmsnorm_phase_entry_service is not None:
            if not vector_resource.has_typed_service(
                    rmsnorm_phase_entry_service):
                raise ValueError(
                    "RMSNorm fixed phase-entry service is unavailable: "
                    f"{rmsnorm_phase_entry_service!r}")
            round_service_demands.append(ComputeServiceDemand(
                rmsnorm_phase_entry_service, regime_counts["first"]))
        if add_rmsnorm_phase_chain_excess_service is not None:
            if not vector_resource.has_typed_service(
                    add_rmsnorm_phase_chain_excess_service):
                raise ValueError(
                    "AddRMSNorm scalar phase-chain service is unavailable: "
                    f"{add_rmsnorm_phase_chain_excess_service!r}")
            round_service_demands.append(ComputeServiceDemand(
                add_rmsnorm_phase_chain_excess_service,
                float(rows * add_rmsnorm_scalar_round_depth)))
        if typed_rmsnorm_scalar_rounds or typed_add_rms_scalar_rounds:
            round_weight_per_cta = hidden / block_threads
        else:
            round_weight_per_cta = math.fsum(
                weight
                for round_widths in add_rms_round_widths
                for _, weight in round_widths)
        input_round_weight_per_cta = (
            1.0 if c500_grid64_fixed_input_selected else (
                hidden / block_threads
                if c500_add_rmsnorm_block256_scalar_input
                else round_weight_per_cta))
        if c500_complete_phase_chain_service is not None:
            round_operations = float(
                rows * c500_complete_phase_chain_rounds)
        else:
            round_operations = float(
                rows * input_round_weight_per_cta
                + rows * round_weight_per_cta
                + (rows * round_weight_per_cta
                   if rmsnorm_phase_chain_excess_service is not None else 0.0)
                + (regime_counts["first"]
                   if rmsnorm_phase_entry_service is not None else 0.0)
                + (float(rows * add_rmsnorm_scalar_round_depth)
                   if add_rmsnorm_phase_chain_excess_service is not None
                   else 0.0))
        if not math.isclose(
                math.fsum(
                    demand.operations
                    for demand in round_service_demands),
                round_operations, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "typed normalization rounds do not conserve source rounds")
        if any(demand.resource_role == "compute_vector"
               for demand in inp.compute_demands):
            raise ValueError(
                "source-bound normalization rounds require an unassigned "
                "compute_vector resource")
        exact_completion_demands = ()
        if completion_service is not None:
            exact_completion_demands = (ComputeExactRequestDemand(
                completion_service,
                request_sizes_by_worker=tuple(
                    (1,) * count for count in ctas_by_worker),
                pipelined_completion=True,
            ),)
            round_operations += float(rows)
        typed_rounds = ComputeDemand(
            resource_role="compute_vector",
            operations=round_operations,
            shape_efficiency=1.0,
            schedule_efficiency=1.0,
            service_demands=tuple(round_service_demands),
            exact_request_demands=exact_completion_demands,
        )
        if typed_rmsnorm_scalar_rounds:
            input_phase_operation = "rmsnorm_input_square_round"
        elif c500_grid64_fixed_input_selected:
            input_phase_operation = "add_rmsnorm_grid64_fixed_input_loop"
        elif c500_add_rmsnorm_block256_scalar_input:
            input_phase_operation = (
                "add_rmsnorm_scalar_input_add_square_round")
        else:
            input_phase_operation = "bf16_add_square_round"
        if c500_complete_phase_chain_service is None:
            phase_rounds = {
                "input_compute": (PhaseComputeDemand(
                    "compute_vector", input_phase_operation,
                    float(rows * input_round_weight_per_cta)),),
                "output_compute": (PhaseComputeDemand(
                    "compute_vector", (
                        "rmsnorm_output_scale_round"
                        if typed_rmsnorm_scalar_rounds
                        else "bf16_scale_multiply_round"),
                    float(rows * round_weight_per_cta)),),
            }
        else:
            phase_rounds = {}
        if rmsnorm_phase_chain_excess_service is not None:
            phase_rounds["output_compute"] += (PhaseComputeDemand(
                "compute_vector", "rmsnorm_phase_chain_excess_round",
                float(rows * round_weight_per_cta)),)
        if rmsnorm_phase_entry_service is not None:
            phase_rounds["output_compute"] += (PhaseComputeDemand(
                "compute_vector", "rmsnorm_phase_entry",
                regime_counts["first"]),)
        if add_rmsnorm_phase_chain_excess_service is not None:
            phase_rounds["output_compute"] += (PhaseComputeDemand(
                "compute_vector", "add_rmsnorm_phase_chain_excess",
                float(rows * add_rmsnorm_scalar_round_depth)),)
        unassigned_compute_demands = inp.phase_ledger.unassigned_compute_demands
        if c500_complete_phase_chain_service is not None:
            unassigned_compute_demands += (PhaseComputeDemand(
                "compute_vector", "norm_complete_source_phase_chain",
                float(rows * c500_complete_phase_chain_rounds)),)
        if completion_service is not None:
            unassigned_compute_demands += (PhaseComputeDemand(
                "compute_vector", "rmsnorm_program_completion",
                float(rows)),)
        phase_ledger = dataclasses.replace(
            inp.phase_ledger,
            phases=tuple(
                dataclasses.replace(
                    phase,
                    compute_demands=(
                        phase.compute_demands + phase_rounds[phase.name]))
                if phase.name in phase_rounds else phase
                for phase in inp.phase_ledger.phases),
            unassigned_compute_demands=unassigned_compute_demands,
            limitations=(
                inp.phase_ledger.limitations + (
                    "c500_complete_source_phase_chain_cross_phase_"
                    "partition",
                )
                if c500_complete_phase_chain_service is not None
                else inp.phase_ledger.limitations),
        )
        demand_per_resource = dict(inp.demand_per_resource)
        if ("compute_vector" not in demand_per_resource
                or demand_per_resource["compute_vector"] != 0.0):
            raise ValueError(
                "source-bound normalization rounds require an explicit zero "
                "flat compute_vector demand")
        demand_per_resource["compute_vector"] = round_operations
        inp = dataclasses.replace(
            inp,
            demand_per_resource=demand_per_resource,
            compute_demands=inp.compute_demands + (typed_rounds,),
            phase_ledger=phase_ledger,
        )

    add_rms_prologue_fraction = None
    has_source_round_pipeline = any(
        mechanism is not None
        for mechanism in (
            hardware.add_rmsnorm_source_round_pipeline,
            hardware.add_rmsnorm_block256_source_round_pipeline,
        ))
    has_c500_generic_block1024_prologue = (
        generic_add_rms_round_services
        and block_threads == 1024)
    if (typed_add_rms_rounds
            and (has_source_round_pipeline
                 or has_c500_generic_block1024_prologue)):
        # The retained SM80 CFG and the pinned C500 generic specialization
        # each have a mandatory non-loop subset on every participating wave.
        # C500 has 22 non-SNOP instructions before the input loop and 24 more
        # between the completed reduction interval and the output loop.  The
        # independently measured add/square and scale/multiply services repeat
        # only their loop bodies, while the reduction service covers the
        # disjoint CUB interval, so these fixed CTA-local instructions are not
        # represented elsewhere.
        # C500 evidence: profiler_output/microbench/<hardware_id>/
        # add_rmsnorm_static_issue_4d5b5826_20260908/report.json.
        participating_warps = min(
            math.ceil(hidden / 256), block_threads // hardware.parallel_width)
        fixed_instructions = (
            _C500_ADD_RMSNORM_GENERIC_FIXED_INSTRUCTIONS_PER_WAVE
            if has_c500_generic_block1024_prologue else 22)
        prologue_operations = float(
            fixed_instructions * rows * participating_warps)
        if ("warp_issue" not in hardware._role_map
                or inp.demand_per_resource["warp_issue"] != 0.0
                or any(demand.resource_role == "warp_issue"
                       for demand in inp.compute_demands)):
            raise ValueError(
                "typed Add-RMSNorm prologue requires unassigned warp issue")
        demand_per_resource = dict(inp.demand_per_resource)
        demand_per_resource["warp_issue"] = prologue_operations
        prologue_demand = ComputeDemand(
            resource_role="warp_issue",
            operations=prologue_operations,
            shape_efficiency=1.0,
            schedule_efficiency=1.0,
        )
        prologue_phases = []
        for phase in inp.phase_ledger.phases:
            updated_phase = phase
            if phase.name == "input_compute":
                updated_phase = dataclasses.replace(
                    phase,
                    compute_demands=phase.compute_demands + (
                        PhaseComputeDemand(
                            "warp_issue", "add_rmsnorm_non_loop_prologue",
                            prologue_operations),))
            prologue_phases.append(updated_phase)
        phase_ledger = dataclasses.replace(
            inp.phase_ledger, phases=tuple(prologue_phases))
        inp = dataclasses.replace(
            inp,
            demand_per_resource=demand_per_resource,
            compute_demands=inp.compute_demands + (prologue_demand,),
            phase_ledger=phase_ledger,
        )
        add_rms_prologue_fraction = 1.0 / rows

    cta_stages_by_contract = {}

    def _cta_task(
            worker: int, regime: str,
            reduction_service_demands: Optional[
                Tuple[ComputeServiceDemand, ...]]) -> PipelineTaskDemand:
        contract = (regime, reduction_service_demands)
        if contract in cta_stages_by_contract:
            return PipelineTaskDemand(
                name=cta_stages_by_contract[contract][0], worker=worker,
                stages=cta_stages_by_contract[contract][1])
        stages = []
        if typed_norm_rounds:
            # Each typed service already executes the complete warm-cache
            # source round. Keep external L2/DRAM traffic as an aggregate lower
            # bound, but do not charge those bytes again as serial stages
            # inside the CTA dependency path.
            input_function = (
                "input_square" if typed_rmsnorm_scalar_rounds
                else "add_square")
            for round_index in range(loop_rounds):
                stages.append(PipelineStageDemand(
                    f"{input_function}_{round_index}",
                    "compute:compute_vector", None,
                    compute_service_demands=_round_demands(
                        input_function, regime, round_index)))
        else:
            for round_index in range(loop_rounds):
                load_predecessors = (
                    () if round_index == 0 else
                    (f"input_fma_{round_index - 1}",))
                load_start_predecessors = (
                    () if (variant != "add_rmsnorm"
                           or round_index == 0) else
                    (f"residual_store_{round_index - 1}",))
                if local_memory_source_completion:
                    local_demands, shared_demands = (
                        _local_read_round_demands("load_reduce", regime))
                    stages.append(PipelineStageDemand(
                        f"load_reduce_{round_index}", "memory:global_read",
                        None,
                        memory_node_demands=local_demands,
                        launch_shared_memory_node_demands=shared_demands,
                        predecessors=load_predecessors,
                        start_predecessors=load_start_predecessors,
                        local_memory_source_completion=(
                            "sequence_start"
                            if round_index == 0 else "steady")))
                else:
                    stages.append(PipelineStageDemand(
                        f"load_reduce_{round_index}", "memory:global_read",
                        load_reduce_fraction,
                        # A disabled per-AP L1 leaves every CTA's first source
                        # load at the shared L2/DRAM boundary.  Resident and
                        # queued CTAs therefore start independent request
                        # chains; later loop rounds retain the same CTA-local
                        # chain.  An enabled L1 keeps the historical
                        # one-launch cold edge.
                        include_first_request_latency=(
                            include_first_request_latency
                            and round_index == 0
                            and (regime == "first"
                                 or not hardware.l1_cache_enabled)),
                        predecessors=load_predecessors,
                        start_predecessors=load_start_predecessors))
                if input_alu_fraction > 0:
                    stages.append(PipelineStageDemand(
                        f"input_alu_{round_index}", "compute:compute_alu",
                        input_alu_fraction))
                if input_fma_fraction > 0:
                    stages.append(PipelineStageDemand(
                        f"input_fma_{round_index}", "compute:compute_fma",
                        input_fma_fraction))
                if variant == "add_rmsnorm":
                    stages.append(PipelineStageDemand(
                        f"residual_store_{round_index}",
                        store_service, residual_store_fraction))
        if typed_block_reduction:
            if reduction_service_demands is None:
                raise ValueError(
                    "typed block reduction requires a service identity")
            reduction_predecessors = None
            reduction_start_predecessors = ()
            if variant == "add_rmsnorm":
                if typed_add_rms_rounds:
                    reduction_predecessors = (
                        f"add_square_{loop_rounds - 1}",)
                else:
                    reduction_start_predecessors = (
                        f"residual_store_{loop_rounds - 1}",)
                    reduction_predecessors = (
                        f"input_fma_{loop_rounds - 1}",)
            elif variant == "rmsnorm":
                pass
            else:
                raise AssertionError(
                    f"unexpected GPU NORM variant: {variant!r}")
            stages.append(PipelineStageDemand(
                "block_reduction", "compute:compute_reduction", None,
                compute_service_demands=reduction_service_demands,
                predecessors=reduction_predecessors,
                start_predecessors=reduction_start_predecessors))
        else:
            if reduction_fma_fraction > 0:
                stages.append(PipelineStageDemand(
                    "block_reduction", "compute:compute_fma",
                    reduction_fma_fraction,
                    predecessors=(f"input_fma_{loop_rounds - 1}",),
                    start_predecessors=(
                        f"residual_store_{loop_rounds - 1}",)))
            if reduction_sfu_fraction > 0:
                stages.append(PipelineStageDemand(
                    "reciprocal_sqrt", "compute:compute_sfu",
                    reduction_sfu_fraction))
        if typed_norm_rounds:
            reduction_terminal = "block_reduction"
            output_function = (
                "output_scale" if typed_rmsnorm_scalar_rounds
                else "scale_multiply")
            for round_index in range(loop_rounds):
                stages.append(PipelineStageDemand(
                    f"{output_function}_{round_index}",
                    "compute:compute_vector", None,
                    compute_service_demands=_round_demands(
                        output_function, regime, round_index),
                    predecessors=(reduction_terminal,)
                    if round_index == 0 else None))
                if rmsnorm_phase_chain_excess_service is not None:
                    stages.append(PipelineStageDemand(
                        f"phase_chain_excess_{round_index}",
                        "compute:compute_vector", None,
                        compute_service_demands=_round_demands(
                            "phase_chain_excess", regime, round_index),
                        predecessors=(f"{output_function}_{round_index}",)))
        else:
            reduction_terminal = (
                "reciprocal_sqrt"
                if reduction_sfu_fraction > 0 else "block_reduction")
            for round_index in range(loop_rounds):
                predecessor_by_variant = {
                    "add_rmsnorm": (
                        reduction_terminal,
                        f"residual_store_{round_index}",
                    ),
                    "rmsnorm": (reduction_terminal,),
                }
                output_start_predecessors = (
                    (f"final_store_{round_index - 1}",)
                    if round_index > 0 else ())
                if local_memory_source_completion:
                    local_demands, shared_demands = (
                        _local_read_round_demands("load_output", regime))
                    if local_demands or shared_demands:
                        stages.append(PipelineStageDemand(
                            f"load_output_{round_index}",
                            "memory:global_read", None,
                            memory_node_demands=local_demands,
                            launch_shared_memory_node_demands=shared_demands,
                            predecessors=predecessor_by_variant[variant],
                            start_predecessors=output_start_predecessors,
                            local_memory_source_completion=(
                                "sequence_start"
                                if round_index == 0 else "steady")))
                else:
                    stages.append(PipelineStageDemand(
                        f"load_output_{round_index}", "memory:global_read",
                        load_output_fraction,
                        # The block reduction separates the two source loops,
                        # so the first reread of each CTA is another dependency
                        # head when no private L1 can retain that CTA's request
                        # state.
                        include_first_request_latency=(
                            include_first_request_latency
                            and round_index == 0
                            and (regime == "first"
                                 or not hardware.l1_cache_enabled)),
                        predecessors=predecessor_by_variant[variant],
                        start_predecessors=output_start_predecessors))
                if output_alu_fraction > 0:
                    stages.append(PipelineStageDemand(
                        f"output_alu_{round_index}", "compute:compute_alu",
                        output_alu_fraction))
                if output_fma_fraction > 0:
                    stages.append(PipelineStageDemand(
                        f"output_fma_{round_index}", "compute:compute_fma",
                        output_fma_fraction))
                stages.append(PipelineStageDemand(
                    f"final_store_{round_index}", "memory:global_write",
                    final_store_fraction))
        task_prefix = (
            reduction_service_demands[0].service
            if reduction_service_demands is not None
            else f"generic_norm_{regime}")
        task_name = f"{task_prefix}_cta"
        resolved_stages = tuple(stages)
        cta_stages_by_contract[contract] = (task_name, resolved_stages)
        return PipelineTaskDemand(
            name=task_name, worker=worker, stages=resolved_stages)

    entries = []
    typed_cta_classes = set()
    generic_cta_classes = set()
    for worker in range(inp.active_units):
        cta_count = ctas_by_worker[worker]
        regime_order = tuple(
            ["first"]
            + ["resident"]
            * (min(cta_count, resident_ctas) - 1)
            + ["queued"]
            * max(cta_count - resident_ctas, 0)
        )
        pattern_regimes = tuple(dict.fromkeys(regime_order))
        if typed_block_reduction:
            reduction_services = {}
            for regime in pattern_regimes:
                if regime == "first":
                    demands = tuple(
                        ComputeServiceDemand(service, weight)
                        for service, weight in
                        _block_reduction_first_service_weights(
                            hardware.resource_by_role("compute_reduction"),
                            block_threads, inp.active_units))
                elif regime in ("resident", "queued"):
                    demands = (ComputeServiceDemand(
                        f"block_reduce_{regime}_{block_threads}", 1.0),)
                else:
                    raise ValueError(
                        f"unknown block-reduction regime: {regime!r}")
                reduction_services[regime] = demands
            unknown_services = {
                demand.service
                for demands in reduction_services.values()
                for demand in demands
                if demand.service not in services}
            if unknown_services:
                raise ValueError(
                    "GPU NORM CTA reduction schedule references unavailable "
                    f"services: {sorted(unknown_services)}")
        else:
            reduction_services = {
                regime: None for regime in pattern_regimes}

        if typed_norm_rounds:
            if cta_count in typed_cta_classes:
                continue
            typed_cta_classes.add(cta_count)
            physical_workers = tuple(
                candidate_worker
                for candidate_worker, candidate_count in enumerate(
                    ctas_by_worker)
                if candidate_count == cta_count)
            stages = []
            reduction_terminals = {}
            output_terminals = {}

            def _append_input_and_reduction(
                    cta_index: int,
                    slot_predecessors: Tuple[str, ...]) -> None:
                regime = regime_order[cta_index]
                if legacy_program_vector_cycles_per_cta is not None:
                    # This independent branch is the whole-program occupied
                    # issue floor.  It intentionally has no phase dependency:
                    # dependencies use the source-stage branch below, while
                    # all CTAs contend for one physical issue slot per AP.
                    stages.append(PipelineStageDemand(
                        f"cta{cta_index}_program_issue_floor",
                        "compute:warp_issue", 1.0 / rows,
                        predecessors=()))
                    if completion_service is not None:
                        stages.append(PipelineStageDemand(
                            f"cta{cta_index}_program_completion",
                            "compute:compute_vector", None,
                            compute_exact_request_demands=(
                                ComputeExactRequestDemand(
                                    completion_service,
                                    request_sizes_by_worker=((1,),),
                                    pipelined_completion=True,
                                ),),
                            predecessors=(),
                            start_predecessors=(
                                f"cta{cta_index}_program_issue_floor",),
                        ))
                input_function = (
                    "input_square" if typed_rmsnorm_scalar_rounds
                    else "add_square")
                reduction_service = reduction_services[regime]
                if reduction_service is None:
                    raise ValueError(
                        "typed normalization rounds require typed reduction")
                early_reduction_name = None
                terminal_reduction_service = reduction_service
                if (generic_add_rms_round_services and block_threads == 1024
                        and not c500_grid64_fixed_input_selected
                        and tail_threads > 0
                        and tail_threads < block_threads):
                    # In the vec8 source loop, only ``tail_threads`` execute
                    # the final input round. The remaining physical warps
                    # reach CUB BlockReduce immediately and can execute their
                    # participant-local reduction prefix while the active
                    # warps finish the source round. Split the conserved
                    # reduction service by that static participant ratio;
                    # the active suffix still waits for both paths.
                    active_fraction = tail_threads / block_threads
                    early_fraction = 1.0 - active_fraction
                    early_reduction_name = (
                        f"cta{cta_index}_block_reduction_inactive_prefix")
                    stages.append(PipelineStageDemand(
                        early_reduction_name,
                        "compute:compute_reduction", None,
                        compute_service_demands=tuple(
                            ComputeServiceDemand(
                                demand.service,
                                demand.operations * early_fraction)
                            for demand in reduction_service),
                        predecessors=slot_predecessors))
                    terminal_reduction_service = tuple(
                        ComputeServiceDemand(
                            demand.service,
                            demand.operations * active_fraction)
                        for demand in reduction_service)
                prior_stage = None
                parallel_prologue_name = None
                if add_rms_prologue_fraction is not None:
                    prologue_name = f"cta{cta_index}_source_prologue"
                    stages.append(PipelineStageDemand(
                        prologue_name, "compute:warp_issue",
                        add_rms_prologue_fraction,
                        predecessors=(
                            slot_predecessors if prior_stage is None
                            else (prior_stage,))))
                    if c500_grid64_fixed_input_selected:
                        parallel_prologue_name = prologue_name
                    else:
                        prior_stage = prologue_name
                if c500_complete_phase_chain_service is not None:
                    if regime != "first":
                        raise ValueError(
                            "complete normalization phase-chain service is "
                            "available only for the first CTA cohort")
                    complete_name = (
                        f"cta{cta_index}_complete_source_phase_chain")
                    stages.append(PipelineStageDemand(
                        complete_name, "compute:compute_vector", None,
                        compute_service_demands=(ComputeServiceDemand(
                            c500_complete_phase_chain_service,
                            float(c500_complete_phase_chain_rounds)),),
                        predecessors=(
                            slot_predecessors if prior_stage is None
                            else (prior_stage,))))
                    prior_stage = complete_name
                elif c500_grid64_fixed_input_selected:
                    stage_name = f"cta{cta_index}_fixed_input_loop"
                    stages.append(PipelineStageDemand(
                        stage_name, "compute:compute_vector", None,
                        compute_service_demands=(ComputeServiceDemand(
                            c500_grid64_fixed_input_service, 1.0),),
                        predecessors=slot_predecessors))
                    prior_stage = stage_name
                else:
                    for round_index in range(loop_rounds):
                        stage_name = (
                            f"cta{cta_index}_{input_function}_{round_index}")
                        predecessors = (
                            slot_predecessors if prior_stage is None
                            else (prior_stage,))
                        stages.append(PipelineStageDemand(
                            stage_name, "compute:compute_vector", None,
                            compute_service_demands=_round_demands(
                                input_function, regime, round_index),
                            predecessors=predecessors))
                        prior_stage = stage_name
                if prior_stage is None:
                    raise RuntimeError(
                        "normalization CTA input contains no source round")
                reduction_name = f"cta{cta_index}_block_reduction"
                reduction_predecessors = (prior_stage,)
                if parallel_prologue_name is not None:
                    reduction_predecessors += (parallel_prologue_name,)
                if early_reduction_name is not None:
                    reduction_predecessors += (early_reduction_name,)
                stages.append(PipelineStageDemand(
                    reduction_name, "compute:compute_reduction", None,
                    compute_service_demands=terminal_reduction_service,
                    predecessors=reduction_predecessors))
                reduction_terminals[cta_index] = reduction_name

            def _append_output(cta_index: int) -> None:
                regime = regime_order[cta_index]
                if c500_complete_phase_chain_service is not None:
                    output_terminals[cta_index] = (
                        reduction_terminals[cta_index])
                    return
                output_function = (
                    "output_scale" if typed_rmsnorm_scalar_rounds
                    else "scale_multiply")
                prior_stage = reduction_terminals[cta_index]
                if (rmsnorm_phase_entry_service is not None
                        and regime == "first"):
                    entry_name = f"cta{cta_index}_phase_entry"
                    stages.append(PipelineStageDemand(
                        entry_name, "compute:compute_vector", None,
                        compute_service_demands=(ComputeServiceDemand(
                            rmsnorm_phase_entry_service, 1.0),),
                        predecessors=(prior_stage,)))
                    prior_stage = entry_name
                if (add_rmsnorm_phase_chain_excess_service is not None
                        and regime == "first"):
                    excess_name = f"cta{cta_index}_scalar_phase_chain_excess"
                    stages.append(PipelineStageDemand(
                        excess_name, "compute:compute_vector", None,
                        compute_service_demands=(ComputeServiceDemand(
                            add_rmsnorm_phase_chain_excess_service,
                            float(add_rmsnorm_scalar_round_depth)),),
                        predecessors=(prior_stage,)))
                    prior_stage = excess_name
                for round_index in range(loop_rounds):
                    stage_name = (
                        f"cta{cta_index}_{output_function}_{round_index}")
                    stages.append(PipelineStageDemand(
                        stage_name, "compute:compute_vector", None,
                        compute_service_demands=_round_demands(
                            output_function, regime, round_index),
                        predecessors=(prior_stage,)))
                    prior_stage = stage_name
                    if rmsnorm_phase_chain_excess_service is not None:
                        excess_name = (
                            f"cta{cta_index}_phase_chain_excess_"
                            f"{round_index}")
                        stages.append(PipelineStageDemand(
                            excess_name, "compute:compute_vector", None,
                            compute_service_demands=_round_demands(
                                "phase_chain_excess", regime, round_index),
                            predecessors=(prior_stage,)))
                        prior_stage = excess_name
                output_terminals[cta_index] = prior_stage

            initial_ctas = min(cta_count, resident_ctas)
            block256_mechanism = (
                hardware.add_rmsnorm_block256_source_round_pipeline)
            # A typed RMSNorm round and its typed block reduction both occupy
            # the same AP worker.  Their independent service probes expose
            # rates, not an additional issue lane, so resident CTA lifetimes
            # cannot overlap across the two logical resources.
            shared_issue_serial_ctas = (
                typed_rmsnorm_scalar_rounds
                or (block256_mechanism is not None
                    and block_threads == block256_mechanism["threads_per_cta"]
                    and block256_mechanism["model"] ==
                    "sm80_bf16_add_rmsnorm_block256_resident_v1"))
            if shared_issue_serial_ctas:
                prior_output = ()
                for cta_index in range(cta_count):
                    _append_input_and_reduction(cta_index, prior_output)
                    _append_output(cta_index)
                    prior_output = (output_terminals[cta_index],)
                next_cta = cta_count
                active_slots = []
            else:
                active_slots = list(range(initial_ctas))
                for cta_index in active_slots:
                    _append_input_and_reduction(cta_index, ())
                next_cta = initial_ctas
            while active_slots:
                refilled_slots = []
                for cta_index in active_slots:
                    _append_output(cta_index)
                    if next_cta >= cta_count:
                        continue
                    _append_input_and_reduction(
                        next_cta, (output_terminals[cta_index],))
                    refilled_slots.append(next_cta)
                    next_cta += 1
                active_slots = refilled_slots
            if next_cta != cta_count:
                raise RuntimeError(
                    "normalization resident CTA schedule lost queued work")
            resident_schedule = PipelineTaskDemand(
                name=f"resident_{cta_count}_cta_schedule",
                worker=worker,
                stages=tuple(stages))
            sequence = PipelineTaskSequenceDemand(
                name=f"resident_{cta_count}_cta_sequence",
                worker=worker,
                tasks=(resident_schedule,))
            entries.append(PipelineTaskProgramDemand(
                name=f"resident_{cta_count}_cta_worker_class",
                worker=worker,
                patterns=(sequence,),
                order=(0,),
                worker_multiplicity=len(physical_workers),
                physical_workers=physical_workers,
            ))
            continue

        if cta_count in generic_cta_classes:
            continue
        physical_workers = tuple(
            candidate_worker
            for candidate_worker, candidate_count in enumerate(ctas_by_worker)
            if candidate_count == cta_count)
        compact_worker = len(generic_cta_classes)
        generic_cta_classes.add(cta_count)
        patterns = tuple(
            PipelineTaskSequenceDemand(
                name=regime,
                worker=compact_worker,
                tasks=(_cta_task(
                    compact_worker, regime, reduction_services[regime]),),
            )
            for regime in pattern_regimes)
        pattern_index = {
            regime: index for index, regime in enumerate(pattern_regimes)}
        entries.append(PipelineTaskProgramDemand(
            name=f"worker{worker}_ctas",
            worker=compact_worker,
            patterns=patterns,
            order=tuple(pattern_index[regime] for regime in regime_order),
            worker_multiplicity=len(physical_workers),
            physical_workers=physical_workers,
        ))
    buffers = () if typed_norm_rounds else (PipelineBuffer(
        name="resident_cta_slots",
        producer_stage="load_reduce_0",
        consumer_stage=f"final_store_{loop_rounds - 1}",
        slots=resident_ctas,
        scope="worker",
    ),)
    plan = PipelineWorkPlan(
        worker_count=inp.active_units,
        tasks=tuple(entries),
        buffers=buffers,
    )
    attached = attach_pipeline_work_plan(inp, hardware, plan)
    return _attach_a800_block256_add_rms_resident_cohort_floor(
        attached, kernel, hardware, ctas_by_worker)


def _has_semantic_memory_dependency_task(kernel: KernelSpec) -> bool:
    """Return whether operator semantics require a non-empty load/store task."""

    variant = kernel.variant.lower()
    if variant in (
            "rmsnorm", "add_rmsnorm", "rotary", "rope",
            "activation", "silu",
            "kv_cache", "kvcache",
            "paged_kv_cache"):
        return True
    if variant == "elementwise_pytorch_vec4":
        return kernel.attributes["operation"] in ("add", "mul")
    if variant == "elementwise":
        return kernel.attributes["operation"] == "copy"
    return False


def _multiround_memory_dependency_source_supported(
        kernel: KernelSpec, hardware) -> bool:
    """Return whether source semantics match the measured multi-round task.

    The generic paired task remains a valid fixed setup mechanism for a
    single AP-resident launch.  Beyond one full-chip round, its serialized
    one-lane-per-wave load/add/store body is reusable only when launch/source
    facts explicitly expose the same dependency-limited structure.  A typed
    scalar KV loop provides that fact; throughput-oriented BF16 pipelines do
    not, and already carry their own resident/queued service decomposition.
    """

    model = hardware.memory_dependency_task_latency_model
    if (model is None
            or model["model"] not in (
                "paired_batched_multiround_warm_default_cache_"
                "memory_dependency_task_v1",
                "isolated_mcevent_multiround_warm_default_cache_"
                "memory_dependency_task_v1")
            or kernel.launch.grid_size <= hardware.num_compute_units):
        return True
    parameters = kernel.launch.parameters
    return (
        SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER in parameters
        or SOURCE_SCALAR_KV_LOOP_PARAMETER in parameters)


def _source_bound_memory_dependency_task_floor(kernel, hardware):
    """Bind the independent warm load-dependency-store task mechanism.

    Applicability follows the operator-level Load -> Compute/Address -> Store
    dependency and the benchmark's launch domain.  The optional launch marker
    remains an exact-source assertion for stronger transaction contracts, but
    is not required merely to consume this architecture-wide lower bound.
    """

    contract = WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR
    if not hardware.has_task_duration_floor_contract(contract):
        return None, ()
    if not _memory_dependency_timer_domain_matches_launch(hardware):
        return None, (
            "memory_dependency_task_timer_domain_mismatch",)
    parameters = kernel.launch.parameters
    if SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER in parameters:
        return None, ()
    if (SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER not in parameters
            and not _has_semantic_memory_dependency_task(kernel)):
        return None, ()
    if not _multiround_memory_dependency_source_supported(kernel, hardware):
        return None, (
            "memory_dependency_task_multiround_source_unavailable",)
    if hardware.task_duration_floor_contract_supports_launch(
            contract,
            kernel.launch.grid_size,
            kernel.launch.block_size,
            kernel.launch.grid,
            kernel.launch.block,
            hardware.nominal_sm_freq_hz):
        return contract, ()
    return None, (
        "memory_dependency_task_floor_outside_support",)


def _source_bound_tensor_iterator_copy_task_floor(kernel, hardware):
    """Bind a source-shaped non-contiguous TensorIterator copy floor."""

    parameter = SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER
    if parameter not in kernel.launch.parameters:
        return None, ()
    dimensions = kernel.launch.parameters[parameter]
    if kernel.dtype == "bf16" and dimensions == 4.0:
        contract = TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR
    elif kernel.dtype == "float32" and dimensions == 3.0:
        contract = TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR
    else:
        raise ValueError(
            "TensorIterator copy dtype and iterator rank are inconsistent")
    if not hardware.has_task_duration_floor_contract(contract):
        return None, (
            "tensor_iterator_copy_task_floor_unavailable",)
    if hardware.task_duration_floor_contract_supports_launch(
            contract,
            kernel.launch.grid_size,
            kernel.launch.block_size,
            kernel.launch.grid,
            kernel.launch.block,
            hardware.nominal_sm_freq_hz):
        return contract, ()
    return None, (
        "tensor_iterator_copy_task_floor_outside_support",)


def _attach_source_bound_gpu_l2_read_source(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Publish source-derived per-warp reads without guessing stores.

    The launch parameter is a static workload fact: independent source loads
    multiplied by their tail-free coalescing into 32-byte L2 sectors.  It may
    come from source semantics alone or from a stronger fixed-binary audit;
    production duration never supplies the value.  The write stream remains
    unbounded because load/store credit sharing is not yet proven.
    """

    transaction_parameter = (
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER)
    producer_parameter = SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
    has_transactions = transaction_parameter in kernel.launch.parameters
    has_producers = producer_parameter in kernel.launch.parameters
    if not has_transactions and not has_producers:
        return inp
    if (hardware.kernel_memory_model != "gpu_mshr"
            or inp.mem_model != "gpu_mshr"):
        raise ValueError(
            "source-bound L2 read transactions require gpu_mshr hardware")
    expected_paths = {"global_read", "global_write"}
    actual_paths = {path.name for path in inp.memory_paths}
    if actual_paths != expected_paths or len(inp.memory_paths) != 2:
        raise ValueError(
            "source-bound L2 read transactions require one read and one "
            "write path")
    transactions = (
        float(kernel.launch.parameters[transaction_parameter])
        if has_transactions else None)
    if has_producers:
        copy_warps_per_cta = float(
            kernel.launch.parameters[producer_parameter])
        # A per-CTA warp count establishes producers inside that source
        # program only. Both architectural residency and a version-bound
        # register residency are capacity facts; neither establishes that
        # peer CTAs issue their dependent loads in the same instant. Keep the
        # request-source completion to one CTA until a source admission or
        # independent co-issue mechanism explicitly proves otherwise.
        resident_ctas = 1
        producers_per_unit = copy_warps_per_cta
        physical_warps_per_unit = (
            math.ceil(kernel.launch.block_size / hardware.parallel_width)
            * resident_ctas)
        if producers_per_unit > physical_warps_per_unit:
            raise ValueError(
                "source-bound L2 read producers exceed resident warps")
    elif has_transactions:
        # A per-warp transaction contract proves that the active warps of at
        # least one dispatched CTA are independent request producers. Keep
        # the bound occupancy-free by using exactly one CTA per active unit;
        # additional resident CTAs require an explicit source/launch fact.
        producers_per_unit = float(math.ceil(
            kernel.launch.block_size / hardware.parallel_width))
    else:
        producers_per_unit = None
    paths = []
    for path in inp.memory_paths:
        if (has_transactions
                and path.source.independent_requests_per_producer is not None):
            raise ValueError(
                "source-bound L2 read transactions must replace an unknown "
                "producer-independence contract")
        if path.name == "global_read":
            source = dataclasses.replace(
                path.source,
                producers_per_unit=(
                    path.source.producers_per_unit
                    if producers_per_unit is None else producers_per_unit),
                independent_requests_per_producer=(
                    path.source.independent_requests_per_producer
                    if transactions is None else transactions),
            )
            paths.append(dataclasses.replace(path, source=source))
        elif path.name == "global_write":
            paths.append(path)
        else:
            raise AssertionError(
                f"unhandled source-bound Vector path: {path.name!r}")
    return dataclasses.replace(inp, memory_paths=tuple(paths))


def _attach_a800_add_rmsnorm_mixed_memory_service(
        inp: PredictionInput, kernel: KernelSpec, hardware, *,
        resident_footprint_bytes: float) -> PredictionInput:
    """Bind the exact 3:2 BF16 vec8 stream to its residency endpoint.

    Independent mechanism probes measure one fixed mixed stream on both sides
    of the physical L2-capacity boundary. Their read and write services
    publish the corresponding 60/40 shares of one coupled capacity.
    Add-RMSNorm has exactly the same aggregate ingress-byte ratio, so the
    maximum of the two directional service times is the measured lower bound
    without changing generic L2 or DRAM resources.
    """

    if hardware.l2_size_bytes <= 0:
        raise ValueError(
            "Add-RMSNorm mixed service requires positive L2 capacity")
    if resident_footprint_bytes > hardware.l2_size_bytes:
        service_names = {
            _A800_ADD_RMS_MIXED_DRAM_READ_SERVICE,
            _A800_ADD_RMS_MIXED_DRAM_WRITE_SERVICE,
        }
        service_scope = "DRAM"
    else:
        service_names = {
            _A800_ADD_RMS_MIXED_L2_READ_SERVICE,
            _A800_ADD_RMS_MIXED_L2_WRITE_SERVICE,
        }
        service_scope = "L2"
    services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service in service_names)
    if not services:
        return inp
    if ({service.service for service in services} != service_names
            or len(services) != 2):
        raise ValueError(
            f"Add-RMSNorm mixed {service_scope} hardware services must "
            "provide one exact read/write pair")
    service_by_kind = {service.access_kind: service for service in services}
    if set(service_by_kind) != {"read", "write"}:
        raise ValueError(
            f"Add-RMSNorm mixed {service_scope} services must cover read "
            "and write")

    rows = _shape(kernel, "rows")
    hidden = _shape(kernel, "hidden")
    parameters = kernel.launch.parameters
    source_matches = (
        kernel.variant.lower() == "add_rmsnorm"
        and kernel.dtype == "bf16"
        and kernel.launch.grid == (rows, 1, 1)
        and kernel.launch.block == (256, 1, 1)
        and hidden % 256 == 0
        and rows >= hardware.num_compute_units
        and rows <= (
            _A800_ADD_RMS_MIXED_L2_MAX_WAVES
            * hardware.num_compute_units)
        and SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
        in parameters
        and parameters[
            SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER]
        == 32.0
        and SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER in parameters
        and parameters[
            SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER]
        == 8.0
    )
    if not source_matches:
        return inp
    if inp.mem_model != "gpu_mshr":
        raise ValueError(
            "Add-RMSNorm mixed L2 service requires gpu_mshr memory")
    paths_by_name = {path.name: path for path in inp.memory_paths}
    if set(paths_by_name) != {"global_read", "global_write"}:
        raise ValueError(
            "Add-RMSNorm mixed L2 service requires split global paths")
    if any(path.access_class_demands for path in inp.memory_paths):
        raise ValueError(
            "Add-RMSNorm mixed L2 paths already have a typed service")

    l2_bytes = {}
    for path_name, expected_kind in (
            ("global_read", "read"), ("global_write", "write")):
        path = paths_by_name[path_name]
        if path.access_kind != expected_kind:
            raise ValueError("Add-RMSNorm global path direction changed")
        matching_nodes = tuple(
            node for node in path.nodes
            if node.resource_role == "l2_cache")
        if len(matching_nodes) != 1:
            raise ValueError(
                "Add-RMSNorm mixed service requires one L2 node per path")
        l2_bytes[expected_kind] = matching_nodes[0].bytes
    if not math.isclose(
            2.0 * l2_bytes["read"],
            3.0 * l2_bytes["write"],
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "Add-RMSNorm mixed L2 source must retain its exact 3:2 ratio")

    typed_paths = []
    for path in inp.memory_paths:
        service = service_by_kind[path.access_kind]
        expected_signature = MemoryAccessClassSignature(
            access_class=service.service,
            dtype="bf16",
            tile_m=256,
            tile_k=8,
            logical_row_bytes=16,
        )
        if service.signature != expected_signature:
            raise ValueError(
                f"Add-RMSNorm mixed {service_scope} service signature "
                "changed")
        typed_paths.append(dataclasses.replace(
            path,
            access_class_demands=(MemoryAccessClassDemand(
                service=service.service,
                ingress_resource_role="l2_cache",
                logical_bytes=l2_bytes[path.access_kind],
                signature=expected_signature,
            ),),
        ))
    return dataclasses.replace(inp, memory_paths=tuple(typed_paths))


def _stage_exact_memory_access_classes(
        inp: PredictionInput) -> PredictionInput:
    """Place aggregate typed memory services on source pipeline stages.

    Fractional memory stages already partition an aggregate path across the
    source-ordered worker tasks.  Once that path receives an exact access
    class, retain the partition while expressing its typed logical bytes
    absolutely.  The aggregate generic path remains an independent lower
    bound; source predecessors determine how the typed stages compose.
    """

    if inp.pipeline_plan is None:
        return inp
    typed_paths = {
        path.name: path for path in inp.memory_paths
        if path.access_class_demands
    }
    if not typed_paths:
        return inp

    def _stage(stage: PipelineStageDemand) -> PipelineStageDemand:
        if not stage.service.startswith("memory:"):
            return stage
        path_name = stage.service.removeprefix("memory:")
        if path_name not in typed_paths:
            return stage
        if (stage.work_fraction is None
                or stage.memory_node_demands
                or stage.launch_shared_memory_node_demands
                or stage.memory_access_class_demands
                or stage.memory_exact_request_demands):
            raise ValueError(
                "typed memory path requires one untyped fractional source "
                f"stage: {stage.name!r}")
        fraction = stage.work_fraction
        path = typed_paths[path_name]
        return dataclasses.replace(
            stage,
            work_fraction=None,
            memory_access_class_demands=tuple(
                dataclasses.replace(
                    demand,
                    logical_bytes=demand.logical_bytes * fraction,
                )
                for demand in path.access_class_demands),
        )

    def _task(task: PipelineTaskDemand) -> PipelineTaskDemand:
        return dataclasses.replace(
            task, stages=tuple(_stage(stage) for stage in task.stages))

    def _sequence(
            sequence: PipelineTaskSequenceDemand
            ) -> PipelineTaskSequenceDemand:
        return dataclasses.replace(
            sequence, tasks=tuple(_task(task) for task in sequence.tasks))

    entries = []
    for entry in inp.pipeline_plan.tasks:
        if isinstance(entry, PipelineTaskProgramDemand):
            entries.append(dataclasses.replace(
                entry,
                patterns=tuple(_sequence(pattern)
                               for pattern in entry.patterns),
            ))
        elif isinstance(entry, PipelineTaskSequenceDemand):
            entries.append(_sequence(entry))
        elif isinstance(entry, PipelineTaskDemand):
            entries.append(_task(entry))
        else:
            raise TypeError(
                "typed memory staging received an unknown pipeline entry")
    return dataclasses.replace(
        inp,
        pipeline_plan=dataclasses.replace(
            inp.pipeline_plan, tasks=tuple(entries)),
    )


def _attach_c500_elementwise_mixed_dram_service(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Bind the tail-free BF16 vec4 2R1W stream to its DRAM service."""

    grid_size = kernel.launch.grid_size
    if grid_size >= _C500_ELEMENTWISE_2R1W_MIN_FULL_RESIDENT_CTAS:
        service_names = {
            _C500_ELEMENTWISE_2R1W_DRAM_READ_SERVICE,
            _C500_ELEMENTWISE_2R1W_DRAM_WRITE_SERVICE,
        }
    elif grid_size >= _C500_ELEMENTWISE_2R1W_MIN_PARTIAL_RESIDENT_CTAS:
        service_names = {
            _C500_ELEMENTWISE_2R1W_PARTIAL_DRAM_READ_SERVICE,
            _C500_ELEMENTWISE_2R1W_PARTIAL_DRAM_WRITE_SERVICE,
        }
    else:
        return inp
    services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service in service_names)
    if not services:
        return inp
    if ({service.service for service in services} != service_names
            or len(services) != 2):
        raise ValueError(
            "C500 elementwise mixed DRAM services must provide one exact "
            "read/write pair")
    service_by_kind = {service.access_kind: service for service in services}
    if set(service_by_kind) != {"read", "write"}:
        raise ValueError(
            "C500 elementwise mixed DRAM services must cover read and write")

    elements = _shape(kernel, "element_count")
    source_matches = (
        kernel.variant.lower() == "elementwise"
        and kernel.dtype == "bf16"
        and kernel.attributes["reads"] == 2
        and kernel.attributes["writes"] == 1
        and kernel.attributes["operation"] in ("add", "mul")
        and kernel.launch.block == (256, 1, 1)
        and elements % 1024 == 0
        and kernel.launch.grid == (elements // 1024, 1, 1)
        and grid_size >= _C500_ELEMENTWISE_2R1W_MIN_PARTIAL_RESIDENT_CTAS)
    if not source_matches:
        return inp
    if inp.mem_model != "gpu_mshr":
        raise ValueError(
            "C500 elementwise mixed DRAM service requires gpu_mshr memory")
    paths_by_name = {path.name: path for path in inp.memory_paths}
    if set(paths_by_name) != {"global_read", "global_write"}:
        raise ValueError(
            "C500 elementwise mixed DRAM service requires split global paths")
    if any(path.access_class_demands for path in inp.memory_paths):
        raise ValueError(
            "C500 elementwise mixed DRAM paths already have a typed service")

    dram_bytes = {}
    for path_name, expected_kind in (
            ("global_read", "read"), ("global_write", "write")):
        path = paths_by_name[path_name]
        if path.access_kind != expected_kind:
            raise ValueError(
                "C500 elementwise global path direction changed")
        matching_nodes = tuple(
            node for node in path.nodes
            if node.resource_role == "dram")
        if len(matching_nodes) != 1:
            raise ValueError(
                "C500 elementwise mixed service requires one DRAM node per "
                "path")
        dram_bytes[expected_kind] = matching_nodes[0].bytes
    if not math.isclose(
            dram_bytes["read"], 2.0 * dram_bytes["write"],
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "C500 elementwise mixed DRAM source must retain its exact 2:1 "
            "ratio")

    typed_paths = []
    for path in inp.memory_paths:
        service = service_by_kind[path.access_kind]
        expected_signature = MemoryAccessClassSignature(
            access_class=service.service,
            dtype="bf16",
            tile_m=256,
            tile_k=4,
            logical_row_bytes=8,
        )
        if service.signature != expected_signature:
            raise ValueError(
                "C500 elementwise mixed DRAM service signature changed")
        typed_paths.append(dataclasses.replace(
            path,
            access_class_demands=(MemoryAccessClassDemand(
                service=service.service,
                ingress_resource_role="l2_cache",
                logical_bytes=dram_bytes[path.access_kind],
                signature=expected_signature,
            ),),
        ))
    return dataclasses.replace(inp, memory_paths=tuple(typed_paths))


def _input(kernel: KernelSpec, hardware, demand: Dict[str, float], *,
           write_bytes: float,
           dram_write_bytes: Optional[float] = None,
           phase_blueprint: Optional[_VectorPhaseBlueprint],
           include_dispatch_overhead: bool = True,
           dram_capacity_multiplier: float = 1.0,
           operator_aux: bool = False) -> PredictionInput:
    demand = dict(demand)
    if dram_write_bytes is None:
        normalized_dram_write_bytes = float(write_bytes)
    else:
        if (isinstance(dram_write_bytes, bool)
                or not isinstance(dram_write_bytes, (int, float))
                or not math.isfinite(dram_write_bytes)
                or dram_write_bytes < 0
                or dram_write_bytes > write_bytes):
            raise ValueError(
                "Vector DRAM write bytes must be finite and within the L2 "
                "write extent")
        normalized_dram_write_bytes = float(dram_write_bytes)
    normalized_phase_blueprint = phase_blueprint
    unassigned_phase_compute = ()
    forward_context = copy_context()
    if _FORWARD_CANN_ANALYTICAL_ENVELOPE in forward_context:
        use_cann_analytical_envelope = forward_context[
            _FORWARD_CANN_ANALYTICAL_ENVELOPE]
    else:
        use_cann_analytical_envelope = False
    launch_parameters = kernel.launch.parameters
    if ("cann_tiling_key" in launch_parameters
            and (hardware.kernel_memory_model != "npu_tile"
                 or not hardware.aic_aiv_separated)):
        raise ValueError(
            "cann_tiling_key requires a separated CANN npu_tile target")
    graph_padding = _has_graph_padding(kernel)
    present_swiglu_pipeline_parameters = tuple(
        name for name in _TYPED_SWIGLU_PIPELINE_PARAMETERS
        if name in launch_parameters)
    if present_swiglu_pipeline_parameters and not graph_padding:
        if present_swiglu_pipeline_parameters != (
                _TYPED_SWIGLU_PIPELINE_PARAMETERS):
            raise ValueError(
                "typed CANN SwiGLU pipeline parameters must be complete")
        if kernel.variant.lower() not in (
                "activation", "silu", "moe_swiglu"):
            raise ValueError(
                "typed CANN SwiGLU pipeline parameters require "
                "activation/silu or moe_swiglu")
        if normalized_phase_blueprint is None:
            raise ValueError(
                "typed CANN SwiGLU pipeline requires a phase blueprint")
    phase_vector_parameters = (
        ("lowered_vector_input_operations", "input_compute",
         "backend_lowered_vector_input"),
        ("lowered_vector_reduce_operations", "reduce_compute",
         "backend_lowered_vector_reduce"),
        ("lowered_vector_output_operations", "output_compute",
         "backend_lowered_vector_output"),
    )
    for parameter_name, phase_name, operation_name in phase_vector_parameters:
        if graph_padding or parameter_name not in launch_parameters:
            continue
        if "compute_vector" not in hardware._role_map:
            raise ValueError(
                "phase-assigned lowered Vector operations require a "
                "compute_vector hardware resource")
        if normalized_phase_blueprint is None:
            raise ValueError(
                "phase-assigned lowered Vector operations require a phase "
                "blueprint")
        operations = float(launch_parameters[parameter_name])
        demand["compute_vector"] += operations
        normalized_phase_blueprint = _append_phase_compute(
            normalized_phase_blueprint,
            phase_name,
            PhaseComputeDemand(
                resource_role="compute_vector",
                operation=operation_name,
                operations=operations,
            ),
        )
    if not graph_padding and "lowered_vector_operations" in launch_parameters:
        if "compute_vector" not in hardware._role_map:
            raise ValueError(
                "lowered Vector operation demand requires a compute_vector "
                "hardware resource")
        lowered_vector_operations = float(
            launch_parameters["lowered_vector_operations"])
        demand["compute_vector"] += lowered_vector_operations
        unassigned_phase_compute += (PhaseComputeDemand(
            resource_role="compute_vector",
            operation="backend_lowered_vector",
            operations=lowered_vector_operations,
        ),)
    if (not graph_padding
            and "lowered_sfu_operations" in launch_parameters
            and _SOURCE_BOUND_BLOCK_REDUCTION_PARAMETER
            not in launch_parameters):
        lowered_sfu_operations = float(
            launch_parameters["lowered_sfu_operations"])
        demand["compute_sfu"] += lowered_sfu_operations
        if normalized_phase_blueprint is not None:
            normalized_phase_blueprint = _append_phase_compute(
                normalized_phase_blueprint,
                normalized_phase_blueprint.lowered_sfu_phase,
                PhaseComputeDemand(
                    resource_role="compute_sfu",
                    operation="compiler_lowered_sfu",
                    operations=lowered_sfu_operations,
                ),
            )
    present_instruction_parameters = tuple(
        name for name in (
            "warp_instructions", "warp_instructions_lower_bound")
        if name in launch_parameters)
    if len(present_instruction_parameters) > 1:
        raise ValueError(
            "Vector launch cannot declare exact and lower-bound warp "
            "instruction demand together")
    if present_instruction_parameters and not graph_padding:
        issue_parameter_name = present_instruction_parameters[0]
        if issue_parameter_name == "warp_instructions":
            issue_operation = "whole_kernel_warp_instruction"
        elif issue_parameter_name == "warp_instructions_lower_bound":
            issue_operation = "whole_kernel_warp_instruction_lower_bound"
        else:
            raise AssertionError(
                f"unknown warp issue parameter: {issue_parameter_name!r}")
    else:
        issue_parameter_name = None
        issue_operation = None
    if issue_parameter_name is not None and not graph_padding:
        if "warp_issue" not in hardware._role_map:
            raise ValueError(
                "positive warp issue demand requires a warp_issue "
                "hardware resource")
        warp_issue_work = float(launch_parameters[issue_parameter_name])
        demand["warp_issue"] = warp_issue_work
        if normalized_phase_blueprint is not None:
            unassigned_phase_compute += (PhaseComputeDemand(
                resource_role="warp_issue",
                operation=issue_operation,
                operations=warp_issue_work,
            ),)
    if (not graph_padding
            and SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER
            in launch_parameters):
        if kernel.variant.lower() not in ("activation", "silu"):
            raise ValueError(
                "source-bound warp slot-2 instructions require "
                "activation/silu")
        if "compute_vector" not in hardware._role_map:
            raise ValueError(
                "source-bound warp slot-2 instructions require a "
                "compute_vector hardware resource")
        slot2_instructions = float(launch_parameters[
            SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER])
        if slot2_instructions <= 0:
            raise ValueError(
                "source-bound warp slot-2 instruction demand must be "
                "positive")
        demand["compute_vector"] += slot2_instructions
        if normalized_phase_blueprint is not None:
            unassigned_phase_compute += (PhaseComputeDemand(
                resource_role="compute_vector",
                operation="whole_kernel_warp_slot2_instruction_lower_bound",
                operations=slot2_instructions,
            ),)
    if hardware.aic_aiv_separated:
        # A pure CANN Vector kernel executes on the independent AIV domain.
        # The generic arithmetic roles describe GPU/AIC pipes; fold their
        # source-level work into the INI-declared AIV vector pipe instead of
        # pricing it on the 24 Cube cores.
        vector_work = sum(
            float(demand[role])
            for role in (
                "compute_vector", "compute_fma", "compute_sfu",
                "compute_alu"))
        demand["compute_vector"] = vector_work
        for role in ("compute_fma", "compute_sfu", "compute_alu"):
            demand[role] = 0.0
        if normalized_phase_blueprint is not None:
            normalized_phase_blueprint = _fold_phase_compute_to_aiv(
                normalized_phase_blueprint)
        execution_units = int(
            hardware.resource_by_role("compute_vector").total_units)
    else:
        execution_units = None
    eta, wave_efficiency, active_units = launch_wave_geometry(
        kernel, hardware,
        execution_units=execution_units)
    if graph_padding:
        task_duration_floor_contract, task_floor_limitations = None, ()
        tensor_iterator_floor_contract, tensor_iterator_floor_limitations = (
            None, ())
    else:
        task_duration_floor_contract, task_floor_limitations = (
            _source_bound_memory_dependency_task_floor(kernel, hardware))
        tensor_iterator_floor_contract, tensor_iterator_floor_limitations = (
            _source_bound_tensor_iterator_copy_task_floor(kernel, hardware))
    if (task_duration_floor_contract is not None
            and tensor_iterator_floor_contract is not None):
        raise ValueError(
            "Vector launch cannot select two complete task-floor contracts")
    if tensor_iterator_floor_contract is not None:
        task_duration_floor_contract = tensor_iterator_floor_contract
    task_floor_limitations += tensor_iterator_floor_limitations
    runtime_task_class = None
    runtime_task_elements = None
    runtime_task_limitations = ()
    runtime_markers = (() if graph_padding else tuple(
        (parameter, task_class, float(launch_parameters[parameter]))
        for parameter, task_class in (
            (SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER,
             PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS),
            (SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER,
             PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS),
        )
        if parameter in launch_parameters))
    # The dataset names this task class by its PyTorch kernel template,
    # at::native::vectorized_elementwise_kernel<4, ...> over bf16 binary
    # functors, which is the instantiation the task-floor probe drives.
    # "elementwise" is the generic spelling and "elementwise_pytorch_vec4"
    # the vectorised one; both denote that template.
    # The vectorised spelling is only recognised on a card that actually
    # publishes a floor for this task class; a card with no such measurement
    # keeps its previous classification untouched.
    accepted_elementwise_variants = ("elementwise",) + (
        ("elementwise_pytorch_vec4",)
        if PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS
        in hardware.runtime_task_latency_models else ())
    if (not graph_padding
            and not runtime_markers
            and kernel.variant.lower() in accepted_elementwise_variants):
        elements = _shape(kernel, "element_count")
        source_bound_binary_iterator = (
            kernel.dtype == "bf16"
            and kernel.attributes["reads"] == 2
            and kernel.attributes["writes"] == 1
            and kernel.attributes["operation"] in ("add", "mul")
            and kernel.launch.grid == (
                _ceil_div_positive(elements, 1024), 1, 1)
            and hardware.runtime_task_class_floor_supports(
                PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
                elements, kernel.launch.grid_size,
                kernel.launch.block_size))
        if source_bound_binary_iterator:
            # Canonical profiler observations retain physical launch facts but
            # not selector-only source markers.  The exact BF16 binary
            # TensorIterator geometry and semantic ABI are sufficient to bind
            # the same runtime task class without consuming a kernel name or
            # duration.
            runtime_markers = ((
                SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER,
                PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
                1.0,
            ),)
    if len(runtime_markers) > 1:
        raise ValueError("Vector launch cannot select two runtime task classes")
    if include_dispatch_overhead and runtime_markers:
        parameter, task_class, marker = runtime_markers[0]
        if marker != 1.0:
            raise ValueError(
                "runtime task-class marker must equal one")
        if task_class == PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS:
            elements = _shape(kernel, "element_count")
        elif task_class == (
                PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS):
            runtime_variant = kernel.variant.lower()
            if runtime_variant in ("rmsnorm", "add_rmsnorm"):
                elements = _shape(kernel, "rows") * _shape(kernel, "hidden")
            elif runtime_variant in ("activation", "silu"):
                elements = _shape(kernel, "tok") * _shape(
                    kernel, "intermediate")
            elif runtime_variant in ("rotary", "rope"):
                rotary_dim = _shape(kernel, "head_dim")
                if runtime_variant == "rope":
                    rotary_dim = _shape(kernel, "rotary_dim")
                elements = (
                    _shape(kernel, "seq_len")
                    * (_shape(kernel, "num_query_heads")
                       + _shape(kernel, "num_kv_heads"))
                    * rotary_dim)
            elif runtime_variant in (
                    "kv_cache", "kvcache", "paged_kv_cache"):
                elements = (
                    _shape(kernel, "seq_len")
                    * _shape(kernel, "num_kv_heads")
                    * _shape(kernel, "head_dim"))
            else:
                raise ValueError(
                    "custom-op runtime task has no logical element contract: "
                    f"{runtime_variant!r}")
        else:
            raise AssertionError(
                f"unhandled runtime task class: {task_class!r}")
        if hardware.runtime_task_class_floor_supports(
                task_class, elements,
                kernel.launch.grid_size, kernel.launch.block_size):
            runtime_task_class = task_class
            runtime_task_elements = elements
        else:
            runtime_task_limitations = (
                "runtime_task_class_floor_outside_support",)
    l2_bytes = float(demand["l2_cache"])
    dram_bytes = float(demand["dram"])
    if operator_aux:
        # Auxiliaries emitted by a fused parent operator share that operator's
        # L1-cached memory path.  Preserve those structured execution facts in
        # the normalized production input; they are launch/hardware properties,
        # not profiler observations.
        dram_capacity_multiplier = (
            hardware.dram_protocol_efficiency_l1cached)
    memory_model = hardware.kernel_memory_model
    if memory_model == "gpu_mshr":
        launch_task_kind = "generic"
        launch_floor_available = True
        support_level = "aggregate_only"
        limitations = gpu_partial_limitations(
            hardware,
            "vector_cta_to_sm_residency_assignment_unavailable",
            "vector_load_compute_store_buffer_depth_unavailable",
        ) + task_floor_limitations + runtime_task_limitations
    elif memory_model == "npu_tile":
        launch_task_kind = "aiv"
        launch_floor_available, launch_limitations = (
            npu_launch_floor_contract(hardware, launch_task_kind))
        support_level = "aggregate_only"
        limitations = (
            "mte_transaction_geometry_unavailable",
            "memory_first_request_latency_unavailable",
            "mte2_descriptor_and_credit_evidence_unavailable",
            "mte3_startup_and_credit_evidence_unavailable",
            "workload_producer_independence_unavailable",
        ) + launch_limitations
        if (kernel.variant.lower() == "elementwise"
                and kernel.attributes["operation"] == "copy"
                and "cann_tiling_key" not in launch_parameters):
            limitations += (
                "cann_contiguous_slice_tiling_payload_unavailable",)
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        launch_task_kind = "generic"
        launch_floor_available = True
        support_level = "compatibility"
        limitations = ("outside_a800_c500_910b_acceptance_scope",)
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")
    if (not graph_padding
            and "warp_instructions_lower_bound" in launch_parameters):
        limitations += (
            "warp_instruction_demand_is_static_lower_bound",)
    if (not graph_padding
            and SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER
            in launch_parameters):
        limitations += (
            "warp_slot2_instruction_demand_is_static_lower_bound",)
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
        mem_model=memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind=launch_task_kind,
        dram_capacity_multiplier=dram_capacity_multiplier,
        include_dispatch_overhead=(
            include_dispatch_overhead and launch_floor_available),
        support_level=support_level,
        limitations=limitations,
        task_duration_floor_contract=task_duration_floor_contract,
        runtime_task_class=runtime_task_class,
        runtime_task_elements=runtime_task_elements,
        # A card publishing a measured warm floor for this task class states a
        # whole-task lower bound at the geometry it was measured at: no launch
        # of the class completes faster.  Scheduler-model classes keep their
        # dispatch treatment; cards publishing no floor are untouched.
        execution_time_floor_ns=(
            hardware.runtime_task_class_duration_ns(
                runtime_task_class, runtime_task_elements,
                kernel.launch.grid_size, kernel.launch.block_size)
            if (runtime_task_class is not None
                and not hardware.runtime_task_class_is_launch_scheduler(
                    runtime_task_class))
            else None),
    )
    normalized, transaction_bytes = _architecture_input(inp, kernel, hardware)
    if memory_model == "gpu_mshr":
        producers, requests_per_producer = gpu_memory_source(hardware)
        path_name = "global_memory"
    elif memory_model == "npu_tile":
        producers, requests_per_producer = 1.0, None
        path_name = "mte2_read"
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        producers, requests_per_producer = 1.0, 1.0
        path_name = None
    else:
        raise AssertionError(
            f"unhandled Vector memory model: {memory_model!r}")
    structured = attach_structured_contract(
        normalized,
        hardware,
        transaction_bytes=transaction_bytes,
        path_name=path_name,
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=None,
        prepend_npu_aiv_mte2_ingress=(
            memory_model == "npu_tile" and write_bytes == 0),
    )
    if (not graph_padding
            and SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER
            in launch_parameters):
        slot2_service = "vector_alu_issue_slot_instruction"
        if not _has_compute_service(
                hardware, "compute_vector", slot2_service):
            raise ValueError(
                "source-bound warp slot-2 instructions require typed "
                f"service {slot2_service!r}")
        structured = _replace_vector_service_demands(
            structured,
            (ComputeServiceDemand(
                slot2_service,
                float(launch_parameters[
                    SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER])),),
        )
    if not graph_padding:
        structured = _attach_source_bound_block_reduction(
            structured, kernel, hardware)
    present_invocation_parameters = tuple(
        name for name in _TYPED_SWIGLU_INVOCATION_PARAMETERS
        if name in launch_parameters)
    if present_invocation_parameters and not graph_padding:
        if present_invocation_parameters != (
                _TYPED_SWIGLU_INVOCATION_PARAMETERS):
            raise ValueError(
                "typed SwiGLU launch must declare both Exp and Div "
                "serial invocation counts")
        if kernel.variant.lower() not in (
                "activation", "silu", "moe_swiglu"):
            raise ValueError(
                "typed SwiGLU launch parameters require activation/silu "
                "or moe_swiglu")
        _, _, logical_elements = _cann_swiglu_logical_shape(kernel)
        elements = float(logical_elements)
        vector_demands = tuple(
            item for item in structured.compute_demands
            if item.resource_role == "compute_vector")
        if len(vector_demands) != 1:
            raise ValueError(
                "typed SwiGLU requires exactly one compute_vector demand")
        vector_demand = vector_demands[0]
        serial_invocations = int(launch_parameters[
            "vector_exp_serial_invocations"])
        div_serial_invocations = int(launch_parameters[
            "vector_div_serial_invocations"])
        service_demands = _cann_swiglu_service_demands(
            hardware,
            (
                (
                    "cast_bf16_to_fp32_serialized", 2.0 * elements,
                    2 * serial_invocations,
                ),
                ("muls_serialized", elements, serial_invocations),
                ("exp", elements, serial_invocations),
                ("adds_serialized", elements, serial_invocations),
                ("div", elements, div_serial_invocations),
                ("mul_serialized", elements, serial_invocations),
                (
                    "cast_fp32_to_bf16_serialized", elements,
                    serial_invocations,
                ),
            ))
        if not math.isclose(
                math.fsum(
                    demand.operations for demand in service_demands),
                vector_demand.operations,
                rel_tol=0.0,
                abs_tol=1e-9):
            raise ValueError(
                "typed SwiGLU services do not conserve aggregate Vector work")
        typed_vector_demand = dataclasses.replace(
            vector_demand,
            service_demands=service_demands,
            exact_request_demands=(),
        )
        typed_compute_demands = list(structured.compute_demands)
        vector_demand_index = typed_compute_demands.index(vector_demand)
        typed_compute_demands[vector_demand_index] = typed_vector_demand
        structured = dataclasses.replace(
            structured, compute_demands=tuple(typed_compute_demands))
    if memory_model == "gpu_mshr":
        l2_total = float(demand["l2_cache"])
        dram_total = float(demand["dram"])
        if (write_bytes > l2_total
                or normalized_dram_write_bytes > dram_total):
            raise ValueError(
                "Vector write bytes cannot exceed L2 or DRAM traffic")
        read_l2_bytes = l2_total - write_bytes
        read_dram_bytes = dram_total - normalized_dram_write_bytes
        # Both sides are recovered by subtracting the write extent from a
        # total the caller formed by addition, so `(a + b) - b` can land one
        # ulp above `a` -- a few nanobytes.  The authoritative
        # DRAM-cannot-exceed-L2 invariant is already enforced exactly on the
        # unsubtracted operands in `_vector_demand`; clamp the reconstructed
        # read split to it rather than failing the downstream strict check on
        # float noise.
        if read_dram_bytes > read_l2_bytes:
            if not math.isclose(
                    read_dram_bytes, read_l2_bytes,
                    rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(
                    "Vector read DRAM traffic exceeds its L2 path extent")
            read_dram_bytes = read_l2_bytes
        structured = split_global_memory_paths(
            structured,
            read_l2_bytes=read_l2_bytes,
            read_dram_bytes=read_dram_bytes,
            write_l2_bytes=write_bytes,
            write_dram_bytes=normalized_dram_write_bytes,
        )
        if not graph_padding:
            structured = _attach_source_bound_gpu_l2_read_source(
                structured, kernel, hardware)
        if not structured.memory_paths and not structured.compute_demands:
            raise ValueError(
                "Vector kernel has neither memory nor compute work")
        # The aggregate GPU contract retains exact compute and traffic demand,
        # but no source-backed CTA worker assignment or Load/Compute/Store
        # buffer lifecycle is available to define a worker pipeline.
    elif memory_model == "npu_tile":
        if write_bytes > 0:
            # AIV stores leave UB through MTE3 and enter the shared L2/GM
            # path.  Keep that source-side L2 write explicit instead of
            # reporting every helper output only as a DRAM continuation.
            structured = append_npu_write_path(
                structured,
                hardware,
                l2_write_bytes=write_bytes,
                dram_write_bytes=normalized_dram_write_bytes,
                prepend_aiv_mte2_ingress=True,
            )
        if (kernel.variant.lower() == "elementwise"
                and kernel.attributes["operation"]
                in _CANN_ELEMENTWISE_COMPARE_CAST_SOURCE_ABI):
            # The routing-mask compare/cast services are selected by the
            # elementwise operation, not by a launch parameter: unlike the
            # Add/Mul tile DAG they carry no tiling-payload prologue and are
            # not suppressed under graph padding.
            structured = _attach_cann_elementwise_serialized_completion(
                structured, kernel, hardware)
        if "cann_tiling_key" in launch_parameters and not graph_padding:
            if kernel.variant.lower() == "elementwise":
                _validate_cann_elementwise_tiling_prologue_contract(
                    kernel, hardware)
                structured = _attach_cann_elementwise_pipeline(
                    structured,
                    kernel,
                    hardware,
                    analytical_compact=use_cann_analytical_envelope,
                )
                structured = dataclasses.replace(
                    structured,
                    task_prologue_contract=(
                        CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE),
                    task_prologue_repetitions=1,
                )
            elif kernel.variant.lower() in ("rmsnorm", "add_rmsnorm"):
                structured = _attach_cann_norm_exact_service(
                    structured, kernel, hardware)
                tiling_key = int(launch_parameters["cann_tiling_key"])
                has_single_n_dag = (
                    _CANN_ADD_RMS_SINGLE_N_DAG_PARAMETER
                    in launch_parameters)
                has_row_group_dag = (
                    _CANN_ADD_RMS_ROW_GROUP_DAG_PARAMETER
                    in launch_parameters)
                has_add_rms_split_d_dag = (
                    _CANN_ADD_RMS_SPLIT_D_DAG_PARAMETER
                    in launch_parameters)
                has_rms_single_row_dag = (
                    _CANN_RMS_SINGLE_ROW_DAG_PARAMETER
                    in launch_parameters)
                has_rms_row_group_dag = (
                    _CANN_RMS_ROW_GROUP_DAG_PARAMETER
                    in launch_parameters)
                has_rms_split_d_dag = (
                    _CANN_RMS_SPLIT_D_DAG_PARAMETER
                    in launch_parameters)
                has_rms_merge_n_dag = (
                    _CANN_RMS_MERGE_N_DAG_PARAMETER
                    in launch_parameters)
                if tiling_key == 0:
                    if has_rms_single_row_dag:
                        structured = _attach_cann_rms_single_row_pipeline(
                            structured, kernel, hardware,
                            analytical_compact=(
                                use_cann_analytical_envelope))
                    elif has_rms_row_group_dag:
                        structured = _attach_cann_rms_row_group_pipeline(
                            structured, kernel, hardware,
                            analytical_compact=use_cann_analytical_envelope)
                    elif has_single_n_dag:
                        raise ValueError(
                            "source-bound AddRmsNorm SingleN DAG identity is "
                            "valid only for CANN key 33")
                    elif has_row_group_dag:
                        raise ValueError(
                            "source-bound AddRmsNorm row-group DAG identity "
                            "is valid only for CANN key 30")
                    else:
                        raise ValueError(
                            "CANN RMSNorm key 0 requires a source-bound "
                            "single-row or row-group DAG identity")
                    if normalized_phase_blueprint is None:
                        raise ValueError(
                            "CANN key-0 RMSNorm requires a phase blueprint")
                    normalized_phase_blueprint = _remove_phase_compute_role(
                        normalized_phase_blueprint,
                        "compute_reduction",
                        "CANN key-0 RMSNorm half-interval reduction",
                    )
                elif tiling_key == 1:
                    if not has_rms_split_d_dag:
                        raise ValueError(
                            "CANN RMSNorm key 1 requires its source-bound "
                            "Split-D DAG identity")
                    structured = _attach_cann_rms_split_d_pipeline(
                        structured, kernel, hardware,
                        analytical_compact=use_cann_analytical_envelope)
                elif tiling_key == 30:
                    structured = _attach_cann_add_rms_row_group_pipeline(
                        structured, kernel, hardware,
                        analytical_compact=use_cann_analytical_envelope)
                    if normalized_phase_blueprint is None:
                        raise ValueError(
                            "CANN key-30 AddRMSNorm requires a phase "
                            "blueprint")
                    normalized_phase_blueprint = _remove_phase_compute_role(
                        normalized_phase_blueprint,
                        "compute_reduction",
                        "CANN key-30 AddRMSNorm ReduceSumFP32",
                    )
                elif tiling_key == 31:
                    if not has_add_rms_split_d_dag:
                        raise ValueError(
                            "CANN AddRMSNorm key 31 requires its "
                            "source-bound Split-D DAG identity")
                    structured = _attach_cann_add_rms_key31_completion_envelope(
                        structured, kernel, hardware,
                        analytical_compact=use_cann_analytical_envelope)
                    if normalized_phase_blueprint is None:
                        raise ValueError(
                            "CANN AddRMSNorm key-31 requires a phase "
                            "blueprint")
                    normalized_phase_blueprint = _remove_phase_compute_role(
                        normalized_phase_blueprint,
                        "compute_reduction",
                        "CANN key-31 AddRMSNorm BlockReduceSumFP32",
                    )
                    normalized_phase_blueprint = _append_phase_compute(
                        normalized_phase_blueprint,
                        "reduce_compute",
                        PhaseComputeDemand(
                            resource_role="compute_reduction",
                            operation=(
                                "cann_add_rmsnorm_split_d_block_reduce_"
                                "source_branches"),
                            operations=structured.demand_per_resource[
                                "compute_reduction"],
                        ),
                    )
                    scalar_operations = structured.demand_per_resource[
                        "scalar_aiv"]
                    if scalar_operations <= 0.0:
                        raise ValueError(
                            "CANN AddRMSNorm key-31 source program requires "
                            "its ComputeY scalar handoff")
                    # ``ComputeY`` performs the V->S rstd read and S->V
                    # return after the output-pass input cast and before
                    # output scaling.  It therefore belongs to the public
                    # output-compute phase rather than an unassigned scalar
                    # residual.
                    normalized_phase_blueprint = _append_phase_compute(
                        normalized_phase_blueprint,
                        "output_compute",
                        PhaseComputeDemand(
                            resource_role="scalar_aiv",
                            operation=(
                                "cann_add_rmsnorm_split_d_rstd_handoff"),
                            operations=scalar_operations,
                        ),
                    )
                elif tiling_key == 2:
                    if not has_rms_merge_n_dag:
                        raise ValueError(
                            "CANN RMSNorm key 2 requires its source-bound "
                            "MERGE_N DAG identity")
                    structured = _attach_cann_rms_merge_n_pipeline(
                        structured, kernel, hardware,
                        analytical_compact=use_cann_analytical_envelope)
                    if normalized_phase_blueprint is None:
                        raise ValueError(
                            "CANN key-2 RMSNorm requires a phase blueprint")
                    normalized_phase_blueprint = _remove_phase_compute_role(
                        normalized_phase_blueprint,
                        "compute_reduction",
                        "CANN key-2 RMSNorm ReduceSumMultiN",
                    )
                elif tiling_key == 33:
                    structured = _attach_cann_add_rms_single_n_pipeline(
                        structured, kernel, hardware,
                        analytical_compact=use_cann_analytical_envelope)
                elif (
                        has_single_n_dag
                        or has_row_group_dag
                        or has_add_rms_split_d_dag
                        or has_rms_single_row_dag
                        or has_rms_row_group_dag
                        or has_rms_split_d_dag
                        or has_rms_merge_n_dag):
                    raise ValueError(
                        "source-bound CANN NORM DAG identity does not match "
                        "the selected tiling key")
                # The fixed CANN 8.5.1 entry materializes one typed tiling
                # struct through ``GET_TILING_DATA`` before it dispatches the
                # selected key.  The source entry, rather than the shape or
                # key, fixes its payload: RmsNormTilingData is 28 B and
                # AddRmsNormTilingData is 80 B.  Retained AddRMS launch
                # reflection binds the latter to AddRmsNormTilingData; select
                # its independently measured materialization mechanism here
                # instead of silently reusing the RMSNorm one.
                payload_bytes = launch_parameters["cann_tiling_payload_bytes"]
                expected_payload_bytes = (
                    28.0 if kernel.variant.lower() == "rmsnorm" else 80.0)
                if payload_bytes != expected_payload_bytes:
                    raise ValueError(
                        "CANN normalization tiling payload does not match "
                        "the source-bound entry program: "
                        f"expected={expected_payload_bytes}, "
                        f"actual={payload_bytes}")
                structured = dataclasses.replace(
                    structured,
                    task_prologue_contract=(
                        CANN_AIV_TILING_MATERIALIZATION_PROLOGUE
                        if payload_bytes == 28.0
                        else CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE),
                    task_prologue_repetitions=1,
                    # Source-closed Norm programs compose after the common
                    # empty AIV task shell.  The legacy descriptor/prologue
                    # overlap is part of the complete-nonempty-task floor
                    # and cannot coexist with worker-entry fill.
                    aiv_descriptor_prologue_pipeline=False,
                )
            elif kernel.variant.lower() in ("rotary", "rope"):
                if _CANN_ATB_BF16_ROPE_DAG_PARAMETER in launch_parameters:
                    structured = _attach_cann_atb_bf16_rope_pipeline(
                        structured,
                        kernel,
                        hardware,
                        analytical_compact=use_cann_analytical_envelope,
                    )
                    # The frozen ATB entry point materializes one 48-byte
                    # RopeTilingData before dispatching either supported BF16
                    # tiling key.  Keep this source-mandatory phase outside
                    # the worker DAG, as for the other typed CANN prologues.
                    structured = dataclasses.replace(
                        structured,
                        task_prologue_contract=(
                            CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE),
                        task_prologue_repetitions=1,
                    )
            elif kernel.variant.lower() in (
                    "kv_cache", "kvcache", "paged_kv_cache"):
                tiling_key = int(launch_parameters["cann_tiling_key"])
                if tiling_key == _CANN_RESHAPE_CACHE_EQUAL_KEY:
                    structured = _attach_cann_equal_kv_pipeline(
                        structured,
                        kernel,
                        hardware,
                        analytical_compact=use_cann_analytical_envelope,
                    )
                elif tiling_key == _CANN_RESHAPE_CACHE_FALLBACK_KEY:
                    structured = _attach_cann_unequal_kv_pipeline(
                        structured,
                        kernel,
                        hardware,
                        analytical_compact=use_cann_analytical_envelope,
                    )
                else:
                    raise ValueError(
                        "unsupported CANN ReshapeAndCache tiling key: "
                        f"{tiling_key}")
                structured = dataclasses.replace(
                    structured,
                    task_prologue_contract=(
                        CANN_AIV_TILING_MATERIALIZATION_PROLOGUE),
                    task_prologue_repetitions=1,
                )
            elif (kernel.variant.lower()
                  == "moe_compute_expert_tokens"):
                # The key selects one of three source-audited traffic
                # programs.  Their DataCopyPad/InitOutput payloads are exact
                # above, while SyncAll and scalar phase lifetimes remain
                # explicit limitations rather than a guessed DAG.
                pass
            else:
                raise ValueError(
                    "cann_tiling_key is valid only for a supported CANN "
                    "Vector operator")
        present_swiglu_pipeline_parameters = tuple(
            name for name in _TYPED_SWIGLU_PIPELINE_PARAMETERS
            if name in launch_parameters)
        if present_swiglu_pipeline_parameters and not graph_padding:
            if present_swiglu_pipeline_parameters != (
                    _TYPED_SWIGLU_PIPELINE_PARAMETERS):
                raise ValueError(
                    "typed CANN SwiGLU pipeline parameters must be complete")
            # SwiGLU already reduces its source tile layout to equivalent AIV
            # worker classes before building the contract, so it is a compact
            # analytical worker envelope on both routes.
            structured = _attach_cann_swiglu_pipeline(
                structured, kernel, hardware)
            structured = dataclasses.replace(
                structured,
                task_prologue_contract=(
                    CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE),
                task_prologue_repetitions=(
                    _CANN_SWIGLU_TILING_MATERIALIZATION_PHASES),
            )
        # Other NPU Vector paths retain aggregate byte capacities until their
        # MTE2/AIV/MTE3 task order and buffer lifetime are source-proven.
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        pass
    else:
        raise AssertionError(
            "unhandled Vector memory model: "
            f"{memory_model!r}")
    if use_cann_analytical_envelope and normalized_phase_blueprint is not None:
        # A source worker DAG used to carry a few exact AIV services outside
        # the semantic per-phase compute tuple.  In the analytical form those
        # services remain conserved aggregate resource work; declare them as
        # explicitly unassigned rather than reintroducing a per-worker replay
        # merely to make the ledger complete.
        phase_roles = {
            demand.resource_role
            for phase in normalized_phase_blueprint.phases
            for demand in phase.compute_demands
        } | {
            demand.resource_role for demand in unassigned_phase_compute
        }
        unassigned_phase_compute += tuple(
            PhaseComputeDemand(
                resource_role=demand.resource_role,
                operation=(
                    "cann_source_dag_phase_assignment_unavailable"),
                operations=demand.operations,
            )
            for demand in structured.compute_demands
            if demand.operations > 0
            and demand.resource_role not in phase_roles
        )
    if normalized_phase_blueprint is not None:
        structured = _attach_phase_ledger(
            structured,
            normalized_phase_blueprint,
            unassigned_phase_compute,
        )
    if not graph_padding:
        structured = _attach_source_bound_gpu_norm_cta_pipeline(
            structured, kernel, hardware)
    return structured


def _shape(kernel: KernelSpec, name: str) -> int:
    return int(kernel.shape[name])


def _has_graph_padding(kernel: KernelSpec) -> bool:
    """Whether this Vector spec's logical shape understates its padded extent.

    Graph-attributed main specs keep a logical token shape while the captured
    graph executes the physical model-input extent; their padding gap is
    priced through the physical-extent paths.  Derived helper specs carry the
    same domain metadata, but their own shape already is that physical
    extent, so no padding gap remains to price.  KernelSpec normalization may
    add alias keys (``dtype_bytes``, ``elements``) to the spec shape, so the
    helper check matches on the model-input keys rather than dict equality.
    """

    names = (
        "graph_logical_tokens", "graph_model_input_tokens",
        "graph_runtime_mode", "model_input_shape",
    )
    present = tuple(name for name in names if name in kernel.attributes)
    if not present:
        return False
    if present != names:
        raise ValueError("VECTOR graph-token-domain fields must be complete")
    if (kernel.attributes["graph_logical_tokens"]
            == kernel.attributes["graph_model_input_tokens"]):
        return False
    shape = kernel.shape
    model_input_shape = dict(kernel.attributes["model_input_shape"])
    return any(
        name not in shape or shape[name] != value
        for name, value in model_input_shape.items())


def _metax_full_graph_norm_execution_kernel(
        kernel: KernelSpec, hardware) -> KernelSpec:
    """Use C500's captured extent for its source-backed Graph Vector programs.

    Logical rows remain the Serving-workload contract. A C500 Graph replay
    dispatches the pinned source programs over the resolved Graph tensor
    extent, though, so their source CTA pipeline must use the already-recorded
    model-input extent.  FULL replays the whole padded forward; PIECEWISE
    replays the captured segments, which hold these ops, over the same padded
    token tensor (``piecewise_graph_physical_extent_dispatch``).  The launch
    grid is the evidence of physical dispatch: it must span the model-input
    rows (norms, RoPE: one CTA per row/token; SwiGLU: grid.y per token), or
    the spec keeps its logical contract.  No observed duration participates.
    """

    if not _has_graph_padding(kernel):
        return kernel
    if not hardware.full_graph_physical_extent_dispatch:
        return kernel
    if kernel.dtype != "bf16":
        return kernel
    mode = kernel.attributes["graph_runtime_mode"]
    if mode == "PIECEWISE":
        if not hardware.piecewise_graph_physical_extent_dispatch:
            return kernel
    elif mode != "FULL":
        return kernel
    model_input_shape = kernel.attributes["model_input_shape"]
    if kernel.variant in ("rmsnorm", "add_rmsnorm"):
        if (SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER
                not in kernel.launch.parameters):
            return kernel
        if kernel.launch.parameters[
                SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] != 1.0:
            return kernel
        if mode == "PIECEWISE" and (
                kernel.launch.grid != (int(model_input_shape["rows"]), 1, 1)):
            return kernel
    elif kernel.variant == "rotary":
        if kernel.launch.grid != (int(model_input_shape["seq_len"]), 1, 1):
            return kernel
    elif kernel.variant == "silu":
        if kernel.launch.grid[1] != int(model_input_shape["tok"]):
            return kernel
    else:
        return kernel
    physical_shape = dict(kernel.shape)
    for name, value in model_input_shape.items():
        physical_shape[name] = value
    if "tok" in physical_shape and "rows" in model_input_shape:
        physical_shape["tok"] = model_input_shape["rows"]
    if "elements" in physical_shape and "element_count" in model_input_shape:
        physical_shape["elements"] = model_input_shape["element_count"]
    attributes = dict(kernel.attributes)
    attributes["graph_logical_tokens"] = attributes[
        "graph_model_input_tokens"]
    return dataclasses.replace(
        kernel, shape=physical_shape, attributes=attributes)


def _gpu_full_graph_source_program_execution_kernel(
        kernel: KernelSpec, hardware) -> KernelSpec:
    """Bind source programs to the physical extent of a FULL-graph launch.

    vLLM FULL-mode CUDA-graph capture pads the decode batch to the captured
    size and dispatches each of these Vector kernels over the padded
    model-input extent, one CTA per physical row; padding rows execute the
    same retained source program on captured buffers.  The launch grid
    recorded in the spec is the direct evidence of that dispatch: when it
    equals the padded row extent, the source-bound services below must bind
    to the physical shape instead of being rejected as graph padding.  When
    the grid disagrees, the launch does not demonstrate physical-extent
    dispatch and the spec keeps its logical padded contract.
    """

    if not _has_graph_padding(kernel):
        return kernel
    if not hardware.full_graph_source_program_physical_extent:
        return kernel
    if kernel.attributes["graph_runtime_mode"] != "FULL":
        return kernel
    variant = kernel.variant.lower()
    if variant in ("activation", "silu"):
        row_axis = "tok"
    elif variant in ("rotary", "rope"):
        row_axis = "seq_len"
    elif variant in ("rmsnorm", "add_rmsnorm", "layernorm"):
        row_axis = "rows"
    else:
        return kernel
    model_input_shape = kernel.attributes["model_input_shape"]
    if row_axis not in model_input_shape:
        raise ValueError(
            "FULL-graph source-program extent requires the model-input "
            f"{row_axis!r} extent")
    physical_rows = int(model_input_shape[row_axis])
    if kernel.launch.grid_size != physical_rows:
        return kernel
    physical_shape = dict(kernel.shape)
    for name, value in model_input_shape.items():
        physical_shape[name] = value
    if "tok" in physical_shape and "rows" in model_input_shape:
        physical_shape["tok"] = model_input_shape["rows"]
    if "elements" in physical_shape and "element_count" in model_input_shape:
        physical_shape["elements"] = model_input_shape["element_count"]
    attributes = dict(kernel.attributes)
    attributes["graph_logical_tokens"] = attributes[
        "graph_model_input_tokens"]
    return dataclasses.replace(
        kernel, shape=physical_shape, attributes=attributes)


def _cann_full_graph_source_program_execution_kernel(
        kernel: KernelSpec, hardware) -> KernelSpec:
    """Bind CANN source programs to the physical extent of a padded Graph launch.

    vLLM Ascend Graph replay pads the decode batch up to the captured size and
    re-issues retained CANN kernels over the padded model-input extent.  The
    Serving ``KernelSpec`` retains logical token accounting.  The recorded
    launch, ``realCoreNum``, and source tiling payload establish the physical
    dispatch used only while pricing that source program.  This applies to
    both FULL and PIECEWISE Graph modes: mode determines capture coverage,
    while the source-owned launch establishes the extent of this kernel.

    Padding rows and padding tiles run the same retained program over slots
    of the captured buffers that belong to no scheduled request: the NORM
    block still loads a hidden row through MTE2, reduces it on the AIV and
    stores it through MTE3, and the SwiGLU tile still streams its two operand
    columns and writes one.  Binding here prices that as the AIV and MTE
    occupancy it is.  It does not turn those slots into semantic work: the
    rebound spec is local to this pricing call, the Serving token accounting
    keeps the logical count, and the masked rows carry no per-request
    identity, because the physical-extent schedule assigns anonymous rows and
    tiles rather than replaying a per-request table onto masked slots.
    """

    if not _has_graph_padding(kernel):
        return kernel
    if (hardware.kernel_memory_model != "npu_tile"
            or not hardware.aic_aiv_separated):
        return kernel
    model_input_shape = kernel.attributes["model_input_shape"]
    parameters = kernel.launch.parameters
    worker_count = kernel.launch.grid_size
    if worker_count <= 0:
        return kernel
    variant = kernel.variant.lower()
    if variant in ("rmsnorm", "add_rmsnorm"):
        if ("cann_tiling_key" not in parameters
                or "norm_block_factor" not in parameters):
            return kernel
        if "rows" not in model_input_shape:
            raise ValueError(
                "CANN FULL-graph NORM extent requires the model-input "
                "'rows' extent")
        physical_rows = int(model_input_shape["rows"])
        # Every AIV owns a non-empty contiguous row range of blockFactor
        # rows.  The padded blockDim exceeds the logical row count, so only
        # the physical extent can distribute the recorded blockFactor.
        if worker_count > physical_rows:
            return kernel
        if (int(parameters["norm_block_factor"])
                != math.ceil(physical_rows / worker_count)):
            return kernel
    elif variant in ("activation", "silu"):
        if any(name not in parameters
               for name in _TYPED_SWIGLU_PIPELINE_PARAMETERS):
            return kernel
        for name in ("tok", "intermediate"):
            if name not in model_input_shape:
                raise ValueError(
                    "CANN FULL-graph SwiGLU extent requires the model-input "
                    f"{name!r} extent")
        base_rows = int(parameters["swiglu_base_row_len"])
        base_columns = int(parameters["swiglu_base_col_len"])
        if base_rows <= 0 or base_columns <= 0:
            raise ValueError("typed CANN SwiGLU base tile must be positive")
        physical_tiles = (
            math.ceil(int(model_input_shape["tok"]) / base_rows)
            * math.ceil(
                int(model_input_shape["intermediate"]) / base_columns))
        # The reflected tiling payload counts tiles over the dispatched
        # operand.  It matches the padded operand and not the logical one.
        if int(parameters["swiglu_total_tiles"]) != physical_tiles:
            return kernel
        if worker_count > physical_tiles:
            return kernel
    elif variant in ("rotary", "rope"):
        required = (
            SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
            SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
            "cann_rope_gather_v3_real_core_num",
            "cann_rope_gather_v3_selected_row_bytes",
            "cann_rope_gather_v3_tiling_payload_bytes",
        )
        if any(name not in parameters for name in required):
            return kernel
        for name in ("seq_len", "head_dim"):
            if name not in model_input_shape:
                raise ValueError(
                    "CANN FULL-graph RoPE extent requires the model-input "
                    f"{name!r} extent")
        physical_tokens = int(model_input_shape["seq_len"])
        physical_head_dim = int(model_input_shape["head_dim"])
        physical_dtype_bytes = _shape(kernel, "dtype_bytes")
        vector_cores = int(
            hardware.resource_by_role("compute_vector").total_units)
        if int(parameters["cann_rope_gather_v3_real_core_num"]) != min(
                physical_tokens, vector_cores):
            return kernel
        if int(parameters["cann_rope_gather_v3_selected_row_bytes"]) != (
                physical_tokens * physical_head_dim * physical_dtype_bytes):
            return kernel
        if int(parameters[
                "cann_rope_gather_v3_tiling_payload_bytes"]) != (
                    _CANN_PUBLIC_ROPE_GATHER_V3_TILING_PAYLOAD_BYTES):
            return kernel
    else:
        return kernel
    physical_shape = dict(kernel.shape)
    for name, value in model_input_shape.items():
        physical_shape[name] = value
    if "tok" in physical_shape and "rows" in model_input_shape:
        physical_shape["tok"] = model_input_shape["rows"]
    if "elements" in physical_shape and "element_count" in model_input_shape:
        physical_shape["elements"] = model_input_shape["element_count"]
    attributes = dict(kernel.attributes)
    attributes["graph_logical_tokens"] = attributes[
        "graph_model_input_tokens"]
    return dataclasses.replace(
        kernel, shape=physical_shape, attributes=attributes)


def _serving_vector_index_bytes(hardware) -> Optional[int]:
    """Return the index width for a supported serving Vector ABI."""

    memory_model = hardware.kernel_memory_model
    if memory_model == "gpu_mshr":
        # The CUDA and MetaX serving calls consume int64 positions/slots.
        return 8
    elif memory_model == "npu_tile":
        if not hardware.aic_aiv_separated:
            raise ValueError(
                "npu_tile Vector ABI requires separated AIC/AIV execution")
        # The CANN AIV serving calls consume int32 positions/slots.
        return 4
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        # Compatibility models do not implement one of the three serving
        # runner ABIs; their architecture adapters replace regular traffic.
        return None
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")


def _vector_demand(hardware, *, l2_read_bytes: float,
                   dram_read_bytes: float, write_bytes: float,
                   flops: Dict[str, float],
                   l1_read_bytes: Optional[float] = None,
                   dram_write_bytes: Optional[float] = None
                   ) -> Dict[str, float]:
    """Build Vector traffic with an explicit CANN MTE2/MTE3 split.

    GPU-like paths use one cache/DRAM traffic envelope.  On a separated CANN
    AIV, GM->UB reads and UB->GM writes use independent INI-declared engines,
    so they are distinct resource paths rather than an empirical overlap
    factor.
    """

    if l1_read_bytes is None:
        l1_read_bytes = l2_read_bytes
    if dram_write_bytes is None:
        dram_write_bytes = write_bytes
    values = {
        "l1_read_bytes": l1_read_bytes,
        "l2_read_bytes": l2_read_bytes,
        "dram_read_bytes": dram_read_bytes,
        "write_bytes": write_bytes,
        "dram_write_bytes": dram_write_bytes,
    }
    invalid = {
        name: value for name, value in values.items()
        if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0)
    }
    if invalid:
        raise ValueError(
            f"Vector traffic must be finite non-negative bytes: {invalid}")
    if dram_read_bytes > l2_read_bytes:
        raise ValueError("Vector DRAM reads cannot exceed L2-path reads")
    if l2_read_bytes > l1_read_bytes:
        raise ValueError("Vector L2 reads cannot exceed L1-path reads")
    if dram_write_bytes > write_bytes:
        raise ValueError("Vector DRAM writes cannot exceed L2-path writes")

    l1_total = float(l1_read_bytes + write_bytes)
    l2_total = float(l2_read_bytes + write_bytes)
    dram_total = float(dram_read_bytes + dram_write_bytes)
    demand = build_forward_demand(l2_total, dram_total, True, flops)
    demand["l1_cache"] = l1_total
    if hardware.aic_aiv_separated:
        # AIV data movement bypasses the AIC-local L1/smem/xbar resources.
        # The npu_tile L2 path represents MTE2 GM->UB reads; MTE3 represents
        # the independent UB->GM write path.
        for role in ("l1_cache", "smem", "xbar"):
            demand[role] = 0.0
        demand["l2_cache"] = float(l2_read_bytes)
        demand["dram"] = float(dram_read_bytes)
        demand["mte3"] = float(write_bytes)
    return demand


def _cann_moe_aux_aggregate_components(
        kernel: KernelSpec, hardware, demand: Dict[str, float], *,
        additional_limitations: Tuple[str, ...] = (),
        launch_task_kind: str = "aiv"):
    """Return source-aggregate CANN MoE helper fields before final assembly.

    These helpers have no source-proven per-worker queue lifecycle.  Their
    common model is therefore one AIV resource envelope, not a replay of
    generic Vector intermediate states.  The fields stay unvalidated only
    until a caller provides its final analytical contract, avoiding a
    complete validation pass for an intermediate that the formula never sees.
    """

    if kernel.variant.lower() not in (_CANN_MOE_AUX_VARIANTS - {"moe_swiglu"}):
        raise ValueError(
            "CANN MoE aggregate input requires a non-SwiGLU CANN helper")
    if (hardware.kernel_memory_model != "npu_tile"
            or not hardware.aic_aiv_separated):
        raise ValueError(
            "CANN MoE aggregate input requires separated npu_tile hardware")
    if not isinstance(additional_limitations, tuple):
        raise TypeError("CANN MoE aggregate limitations must be a tuple")
    if any(not isinstance(value, str) or not value for value
           in additional_limitations):
        raise TypeError("CANN MoE aggregate limitations must be non-empty strings")
    if launch_task_kind not in ("aiv", "mix_aiv"):
        raise ValueError(
            "CANN MoE aggregate launch task kind must be 'aiv' or "
            f"'mix_aiv', got {launch_task_kind!r}")
    unsupported_lowered_parameters = (
        "lowered_sfu_operations", "warp_instructions",
        "warp_instructions_lower_bound")
    if ("lowered_vector_operations" in kernel.launch.parameters
            or any(parameter in kernel.launch.parameters
                   for parameter in unsupported_lowered_parameters)):
        raise ValueError(
            "CANN MoE aggregate helper does not support extra lowered "
            "instruction work")

    normalized_demand = dict(demand)
    vector_work = sum(
        float(normalized_demand[role])
        for role in (
            "compute_vector", "compute_fma", "compute_sfu", "compute_alu"))
    normalized_demand["compute_vector"] = vector_work
    for role in ("compute_fma", "compute_sfu", "compute_alu",
                 "l1_cache", "smem", "xbar"):
        normalized_demand[role] = 0.0

    eta, wave_efficiency, active_units = launch_wave_geometry(
        kernel,
        hardware,
        execution_units=int(
            hardware.resource_by_role("compute_vector").total_units),
    )
    launch_floor_available, launch_limitations = npu_launch_floor_contract(
        hardware, launch_task_kind)
    limitations = (
        "mte_transaction_geometry_unavailable",
        "memory_first_request_latency_unavailable",
        "mte2_descriptor_and_credit_evidence_unavailable",
        "mte3_startup_and_credit_evidence_unavailable",
        "workload_producer_independence_unavailable",
    ) + launch_limitations + additional_limitations

    read_l2_bytes = float(normalized_demand["l2_cache"])
    read_dram_bytes = float(normalized_demand["dram"])
    write_bytes = float(normalized_demand["mte3"])
    if read_dram_bytes > read_l2_bytes:
        raise ValueError("CANN MoE aggregate DRAM reads exceed L2 reads")

    paths = []
    if read_l2_bytes > 0:
        read_nodes = [MemoryNodeDemand(
            resource_role="mte2",
            transactions=None,
            bytes=read_l2_bytes,
            transaction_bytes=None,
        ), MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=None,
            bytes=read_l2_bytes,
            transaction_bytes=None,
        )]
        if read_dram_bytes > 0:
            read_nodes.append(MemoryNodeDemand(
                resource_role="dram",
                transactions=None,
                bytes=read_dram_bytes,
                transaction_bytes=None,
            ))
        paths.append(MemoryPathDemand(
            name="mte2_read",
            access_kind="read",
            initial_transactions=None,
            nodes=tuple(read_nodes),
            source=MemorySourceDemand(
                active_units=active_units,
                producers_per_unit=1.0,
                independent_requests_per_producer=None,
                frontend_rate_per_unit_s=None,
            ),
        ))
    if write_bytes > 0:
        if active_units > hardware.resource_by_role("mte3").total_units:
            raise ValueError(
                "CANN MoE aggregate MTE3 active units exceed the AIV domain")
        write_nodes = (
            MemoryNodeDemand(
                resource_role="mte3",
                transactions=None,
                bytes=write_bytes,
                transaction_bytes=None,
            ),
            MemoryNodeDemand(
                resource_role="l2_cache",
                transactions=None,
                bytes=write_bytes,
                transaction_bytes=None,
            ),
            MemoryNodeDemand(
                resource_role="dram",
                transactions=None,
                bytes=write_bytes,
                transaction_bytes=None,
            ),
        )
        paths.append(MemoryPathDemand(
            name="mte3_write",
            access_kind="write",
            initial_transactions=None,
            nodes=write_nodes,
            source=MemorySourceDemand(
                active_units=active_units,
                producers_per_unit=1.0,
                independent_requests_per_producer=None,
                frontend_rate_per_unit_s=None,
            ),
        ))
        normalized_demand["l2_cache"] += write_bytes
        normalized_demand["dram"] += write_bytes

    return (
        normalized_demand,
        eta,
        wave_efficiency,
        active_units,
        launch_floor_available,
        limitations,
        tuple(paths),
        vector_work,
    )


def _cann_moe_aux_aggregate_input(
        kernel: KernelSpec, hardware, demand: Dict[str, float], *,
        additional_limitations: Tuple[str, ...] = ()) -> PredictionInput:
    """Build the source-aggregate CANN MoE helper contract in one pass."""

    if kernel.variant in ("moe_init_routing", "moe_init_routing_v2",
                          "moe_init_routing_v3"):
        topology_parameter = "cann_moe_init_routing_mix_aiv_1_0"
        if topology_parameter not in kernel.launch.parameters:
            raise ValueError(
                "CANN MoeInitRouting launch is missing its MIX_AIV 1:0 "
                "source topology")
        if kernel.launch.parameters[topology_parameter] != 1.0:
            raise ValueError(
                "CANN MoeInitRouting launch MIX_AIV 1:0 source topology "
                "must be exactly one")
        launch_task_kind = "mix_aiv"
    elif kernel.variant == "moe_compute_expert_tokens":
        topology_parameter = "cann_moe_compute_expert_tokens_mix_aiv_1_0"
        if topology_parameter not in kernel.launch.parameters:
            raise ValueError(
                "CANN MoeComputeExpertTokens launch is missing its MIX_AIV "
                "1:0 source topology")
        if kernel.launch.parameters[topology_parameter] != 1.0:
            raise ValueError(
                "CANN MoeComputeExpertTokens launch MIX_AIV 1:0 source "
                "topology must be exactly one")
        launch_task_kind = "mix_aiv"
    else:
        launch_task_kind = "aiv"
    (normalized_demand, eta, wave_efficiency, active_units,
     launch_floor_available, limitations, paths,
     vector_work) = _cann_moe_aux_aggregate_components(
         kernel,
         hardware,
         demand,
         additional_limitations=additional_limitations,
         launch_task_kind=launch_task_kind,
     )
    compute_demands = ()
    if vector_work > 0:
        compute_demands = (ComputeDemand(
            resource_role="compute_vector",
            operations=vector_work,
            shape_efficiency=1.0,
            schedule_efficiency=eta * wave_efficiency,
        ),)
    return PredictionInput(
        demand_per_resource=normalized_demand,
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
        pipeline_plan=None,
        launch_task_kind=launch_task_kind,
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        include_dispatch_overhead=launch_floor_available,
        support_level="aggregate_only",
        limitations=limitations,
    )


def _l2_dram_served_byte_fraction(hardware, footprint_bytes: float) -> float:
    """Return the measured DRAM-served byte fraction at this footprint.

    The A800 residency probe shows the L2-to-DRAM transition of a repeated
    same-operand single-pass stream is graded, not a knee: full residency
    holds only to about 0.68 of L2 capacity, the stream is fully DRAM-served
    from about 1.50 of capacity, and at capacity itself barely half of the L2
    benefit is realised.  Interpolate linearly between the measured
    footprints; clamp to the first and last measured fraction rather than
    extrapolating outside the probe's domain.
    """

    profile = hardware.l2_operand_residency_curve
    if profile is None:
        raise ValueError(
            "graded L2 operand residency requires a measured curve")
    curve = profile["residency_curve"]
    if footprint_bytes <= curve[0][0]:
        return curve[0][1]
    if footprint_bytes >= curve[-1][0]:
        return curve[-1][1]
    index = bisect_right([point[0] for point in curve], footprint_bytes)
    lower_bytes, lower_fraction = curve[index - 1]
    upper_bytes, upper_fraction = curve[index]
    span = float(upper_bytes - lower_bytes)
    weight = (footprint_bytes - lower_bytes) / span
    return lower_fraction + weight * (upper_fraction - lower_fraction)


def _completion_dependent_dram_write_bytes(
        hardware, *, resident_footprint_bytes: float,
        write_bytes: float) -> float:
    """Return stores that must drain beyond L2 before kernel completion."""

    values = {
        "resident_footprint_bytes": resident_footprint_bytes,
        "write_bytes": write_bytes,
    }
    invalid = {
        name: value for name, value in values.items()
        if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0)
    }
    if invalid:
        raise ValueError(
            "GPU completion traffic must be finite and non-negative: "
            f"{invalid}")
    if hardware.kernel_memory_model != "gpu_mshr":
        return float(write_bytes)
    if hardware.gpu_store_completion_scope == "l2":
        if hardware.l2_operand_residency_curve is None:
            if resident_footprint_bytes <= hardware.l2_size_bytes:
                return 0.0
            return float(write_bytes)
        # The measured residency curve replaces the binary capacity threshold:
        # only the DRAM-served share of the footprint forces a dirty line past
        # L2 before completion.
        return float(write_bytes) * _l2_dram_served_byte_fraction(
            hardware, resident_footprint_bytes)
    elif hardware.gpu_store_completion_scope == "dram":
        return float(write_bytes)
    else:
        raise ValueError(
            "gpu_mshr hardware requires an explicit store completion scope")


def _resident_operand_dram_read_bytes(
        hardware, *, resident_footprint_bytes: float,
        read_bytes: float, compulsory_dram_read_bytes: float) -> float:
    """Return operand reads that must be served beyond L2.

    This is the load-side counterpart of
    ``_completion_dependent_dram_write_bytes`` and uses the same physical
    condition.  On a target whose stores complete in a coherent L2, the
    producer of a single-pass vector kernel's operands completed those stores
    in L2; if the kernel's complete operand and result footprint fits in that
    cache, no line of the operand stream can have been evicted before this
    kernel reads it, so the loads resolve in L2.  Once the footprint exceeds
    L2 capacity the operand stream cannot stay resident and its loads are
    compulsory DRAM misses.  ``compulsory_dram_read_bytes`` are bytes the
    caller has already established as unavoidable DRAM traffic (for instance a
    launch's first touch of a shared weight row) and are retained in both
    regimes.
    """

    values = {
        "resident_footprint_bytes": resident_footprint_bytes,
        "read_bytes": read_bytes,
        "compulsory_dram_read_bytes": compulsory_dram_read_bytes,
    }
    invalid = {
        name: value for name, value in values.items()
        if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0)
    }
    if invalid:
        raise ValueError(
            "GPU operand read traffic must be finite and non-negative: "
            f"{invalid}")
    if compulsory_dram_read_bytes > read_bytes:
        raise ValueError(
            "compulsory operand DRAM reads cannot exceed the operand reads")
    if hardware.kernel_memory_model != "gpu_mshr":
        return float(read_bytes)
    if hardware.gpu_store_completion_scope == "l2":
        if hardware.l2_operand_residency_curve is None:
            if resident_footprint_bytes <= hardware.l2_size_bytes:
                return float(compulsory_dram_read_bytes)
            return float(read_bytes)
        # Only the measured DRAM-served share of the resident operand stream
        # misses; the compulsory bytes miss in either regime.
        residency_bytes = float(read_bytes) - float(compulsory_dram_read_bytes)
        return float(compulsory_dram_read_bytes) + (
            residency_bytes * _l2_dram_served_byte_fraction(
                hardware, resident_footprint_bytes))
    elif hardware.gpu_store_completion_scope == "dram":
        return float(read_bytes)
    else:
        raise ValueError(
            "gpu_mshr hardware requires an explicit store completion scope")


def _flash_mla_metadata_bytes(kernel: KernelSpec) -> Tuple[float, float]:
    """Return exact source-defined metadata input and output traffic."""

    dtype_bytes = _shape(kernel, "dtype_bytes")
    batch = _shape(kernel, "batch")
    scheduler_metadata_ints = _shape(kernel, "scheduler_metadata_ints")
    split_metadata_ints = _shape(kernel, "split_metadata_ints")
    read_bytes = float(dtype_bytes * batch)
    write_bytes = float(
        dtype_bytes * (scheduler_metadata_ints + split_metadata_ints))
    return read_bytes, write_bytes


def _flash_mla_metadata_dependency_cycles(
        kernel: KernelSpec, hardware) -> float:
    """Return the source-enumerated serial scheduler dependency chain.

    The fixed metadata kernel launches one wave.  Thread zero serially emits
    all partition records, while request transitions walk shared scheduler
    state.  Each partition has one global visibility dependency and one
    shared-state dependency; each scheduled segment contributes two shared
    dependencies; completing a request contributes six, while continuing to
    another request in the same partition elides three of those dependencies.
    Global and shared latencies come from the C500 architecture contract.
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise ValueError("FlashMLA metadata requires a GPU memory hierarchy")
    batch = _shape(kernel, "batch")
    partitions = _shape(kernel, "scheduler_partitions")
    segments = _shape(kernel, "scheduler_segment_count")
    worker_runs = _shape(kernel, "scheduler_worker_run_count")
    transitions = segments - worker_runs
    if transitions < 0 or worker_runs > partitions:
        raise ValueError(
            "FlashMLA same-worker transitions exceed source schedule")
    cycles_per_ns = hardware.nominal_sm_freq_hz * 1.0e-9
    dram_dependency_cycles = hardware.dram_latency_ns * cycles_per_ns
    shared = _METAX_C500_SHARED_DEPENDENCY_CYCLES
    return float(
        partitions * (dram_dependency_cycles + shared)
        + 2 * shared * segments
        + 6 * shared * batch
        - 3 * shared * transitions)


def _attach_a800_fp16_rmsnorm_single_cta_completion_floor(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Keep one static RMSNorm CTA completion from shrinking with grid width.

    The vLLM scalar RMSNorm source assigns one identical CTA to each row.  A
    wider launch can increase aggregate memory parallelism, but cannot make
    the completion of that unchanged CTA shorter.  Re-evaluate that one-CTA
    source program through the same hardware-only analytical path and retain
    it as a competing execution lower bound.  This is a source-composition
    rule: it consumes neither profiler timing nor a shape-specific constant.
    """

    if _A800_FP16_RMSNORM_SINGLE_CTA_FLOOR.get():
        return inp
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.num_compute_units != 108
            or hardware.parallel_width != 32
            or kernel.variant.lower() != "rmsnorm"
            or kernel.dtype != "fp16"
            or kernel.launch.grid[1:] != (1, 1)
            or kernel.launch.block != (1024, 1, 1)
            or kernel.launch.grid[0] != _shape(kernel, "rows")
            or "input_reread_count" not in kernel.launch.parameters
            or kernel.launch.parameters["input_reread_count"] != 1.0
            or SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
            in kernel.launch.parameters
            or hardware.dispatch_overhead_ns != 0.0):
        return inp

    hidden = _shape(kernel, "hidden")
    single_shape = dict(kernel.shape)
    single_shape["rows"] = 1
    if "tok" in single_shape:
        single_shape["tok"] = 1
    if "element_count" in single_shape:
        single_shape["element_count"] = hidden
    if "elements" in single_shape:
        single_shape["elements"] = hidden
    single_parameters = dict(kernel.launch.parameters)
    single_parameters["norm_reduction_tree_flops"] = float(
        kernel.launch.block_size - 1)
    single_parameters["norm_epsilon_add_flops"] = 1.0
    single_launch = dataclasses.replace(
        kernel.launch, grid=(1, 1, 1), parameters=single_parameters)
    single_kernel = dataclasses.replace(
        kernel, shape=single_shape, launch=single_launch)

    token = _A800_FP16_RMSNORM_SINGLE_CTA_FLOOR.set(True)
    try:
        single_input, _ = _regular_model(single_kernel, hardware)
    finally:
        _A800_FP16_RMSNORM_SINGLE_CTA_FLOOR.reset(token)
    single_prediction = PerformanceFormula(hardware).predict_kernel(
        single_input)
    if single_prediction.t_launch_s != 0.0:
        raise ValueError(
            "A800 FP16 RMSNorm single-CTA completion floor requires a "
            "zero-dispatch analytical path")
    floor_ns = single_prediction.t_total_ns
    if inp.execution_time_floor_ns is not None:
        floor_ns = max(floor_ns, inp.execution_time_floor_ns)
    return dataclasses.replace(inp, execution_time_floor_ns=floor_ns)


def _regular_model(kernel: KernelSpec, hardware) -> Tuple[PredictionInput, float]:
    inp, compulsory = _regular_model_body(kernel, hardware)
    if (SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER in kernel.launch.parameters
            and hardware.memory_dependency_task_latency_model is None):
        inp = dataclasses.replace(
            inp,
            limitations=inp.limitations + (
                "memory_dependency_task_model_unavailable",),
        )
    return inp, compulsory


def _regular_model_body(kernel: KernelSpec, hardware) -> Tuple[PredictionInput, float]:
    variant = kernel.variant.lower()
    dtype_bytes = _shape(kernel, "dtype_bytes")
    graph_padding = _has_graph_padding(kernel)
    if variant == "qkv_rmsnorm_rope":
        marker = SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER
        if kernel.launch.parameters[marker] != 1.0:
            raise ValueError(
                "vLLM Ascend split-QKV RMSNorm RoPE marker must equal one")
        if (hardware.kernel_memory_model != "npu_tile"
                or not hardware.aic_aiv_separated):
            raise ValueError(
                "split-QKV RMSNorm RoPE requires a separated CANN AIV "
                "execution target")
        tokens = _shape(kernel, "seq_len")
        query_heads = _shape(kernel, "num_query_heads")
        kv_heads = _shape(kernel, "num_kv_heads")
        head_dim = _shape(kernel, "head_dim")
        rotary_dim = _shape(kernel, "rotary_dim")
        q_hidden = query_heads * head_dim
        kv_hidden = kv_heads * head_dim
        qk_elements = tokens * (query_heads + kv_heads) * head_dim
        rotary_elements = tokens * (query_heads + kv_heads) * rotary_dim
        input_elements = tokens * (q_hidden + 2 * kv_hidden)
        if input_elements != _shape(kernel, "element_count"):
            raise ValueError(
                "split-QKV RMSNorm RoPE structured input extent changed")
        # The pinned fusion pattern constructs ``positions`` as torch.int64;
        # this program therefore has an explicit eight-byte index ABI rather
        # than inheriting the generic CANN serving-index convention.
        index_bytes = 8

        # The pinned vLLM-Ascend Triton program loads the fused QKV input once,
        # keeps each Q/K tile in UB through normalization and rotation, then
        # copies V in a second source loop. Q and K share the two gamma rows,
        # position value, and selected cos/sin row. These are compulsory GM
        # bytes; repeated weight loads by the AIV programs are local-L2 work.
        input_bytes = float(input_elements * dtype_bytes)
        output_bytes = input_bytes
        weight_dram_bytes = float(2 * head_dim * dtype_bytes)
        weight_l2_bytes = float(
            kernel.launch.grid_size * 2 * head_dim * dtype_bytes)
        position_bytes = float(tokens * index_bytes)
        cos_sin_bytes = float(tokens * rotary_dim * dtype_bytes)
        l2_read = input_bytes + weight_l2_bytes + position_bytes + cos_sin_bytes
        dram_read = (
            input_bytes + weight_dram_bytes + position_bytes + cos_sin_bytes)

        # Static source semantics per Q/K element: BF16->FP32 conversion,
        # square, reduction add, normalization multiply, gamma multiply, and
        # BF16 output conversion. RoPE contributes two multiplies plus one
        # add/subtract per rotated output element. Mean, epsilon, and rsqrt
        # add one dependency operation each per Q/K row. The separated-AIV
        # adapter folds these source roles onto the physical Vector pipe.
        qk_rows = tokens * (query_heads + kv_heads)
        bias_operations = qk_elements if kernel.attributes["bias"] else 0
        vector_operations = float(
            6 * qk_elements + bias_operations
            + 3 * rotary_elements + 3 * qk_rows)
        flops = {
            "compute_tensor": 0.0,
            "compute_vector": vector_operations,
            "compute_fma": 0.0,
            "compute_reduction": 0.0,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
        }
        demand = _vector_demand(
            hardware,
            l2_read_bytes=l2_read,
            dram_read_bytes=dram_read,
            write_bytes=output_bytes,
            flops=flops,
        )
        inp = _input(
            kernel,
            hardware,
            demand,
            write_bytes=output_bytes,
            # Rows are the steady-state iterations of one 48-worker program,
            # not independent serialized kernels. The aggregate envelope lets
            # MTE2, Vector, and MTE3 overlap across successive source rows.
            phase_blueprint=None,
        )
        inp = dataclasses.replace(
            inp,
            limitations=inp.limitations + (
                "split_qkv_rmsnorm_rope_tile_fill_drain_unavailable",
            ),
        )
        return inp, dram_read + output_bytes
    if variant == "layernorm":
        rows = _shape(kernel, "rows")
        hidden = _shape(kernel, "hidden")
        elements = rows * hidden
        affine = kernel.attributes["affine"]
        input_bytes = float(elements * dtype_bytes)
        parameter_bytes = float(2 * hidden * dtype_bytes if affine else 0)
        write = float(elements * dtype_bytes)
        read = input_bytes + parameter_bytes
        # LayerNorm has two row reductions (mean and centered square sum),
        # followed by normalization and optional affine scale/bias.  These
        # are semantic operation counts, not duration-calibrated weights.
        flops = {
            "compute_tensor": 0,
            "compute_reduction": float(2 * rows),
            "compute_fma": float((7 if affine else 5) * elements),
            "compute_alu": 0,
            "compute_sfu": float(rows),
        }
        inp = _input(
            kernel,
            hardware,
            _vector_demand(
                hardware,
                l2_read_bytes=read,
                dram_read_bytes=read,
                write_bytes=write,
                flops=flops,
            ),
            write_bytes=write,
            phase_blueprint=None,
        )
        return dataclasses.replace(
            inp,
            limitations=inp.limitations + (
                "layernorm_multi_pass_cta_schedule_unavailable",
                "production_duration_calibration_not_used",
            ),
        ), read + write
    if variant == "mla_rope":
        elements = _shape(kernel, "element_count")
        tokens = _shape(kernel, "seq_len")
        head_dim = _shape(kernel, "head_dim")
        index_bytes = _serving_vector_index_bytes(hardware)
        position_bytes = 0 if index_bytes is None else tokens * index_bytes
        cos_sin_bytes = tokens * head_dim * dtype_bytes
        tensor_bytes = elements * dtype_bytes
        read = float(tensor_bytes + position_bytes + cos_sin_bytes)
        write = float(tensor_bytes)
        flops = {
            "compute_tensor": 0,
            "compute_fma": float(3 * elements),
            "compute_alu": float(4 * elements),
            "compute_sfu": 0,
        }
        inp = _input(
            kernel,
            hardware,
            _vector_demand(
                hardware,
                l2_read_bytes=read,
                dram_read_bytes=read,
                write_bytes=write,
                flops=flops,
            ),
            write_bytes=write,
            phase_blueprint=None,
        )
        return dataclasses.replace(
            inp,
            limitations=inp.limitations + (
                "mla_rope_cta_pipeline_schedule_unavailable",
                "production_duration_calibration_not_used",
            ),
        ), read + write
    if variant in ("rmsnorm", "add_rmsnorm"):
        rows = _shape(kernel, "rows")
        hidden = _shape(kernel, "hidden")
        elements = rows * hidden
        bf16_conversion_operations = (
            _source_bf16_conversion_operations(kernel))
        reduction_tree_flops = 0.0
        if "norm_reduction_tree_flops" in kernel.launch.parameters:
            reduction_tree_flops = float(
                kernel.launch.parameters["norm_reduction_tree_flops"])
        epsilon_add_flops = 0.0
        if "norm_epsilon_add_flops" in kernel.launch.parameters:
            epsilon_add_flops = float(
                kernel.launch.parameters["norm_epsilon_add_flops"])
        source_bound_block_reduction = (
            _SOURCE_BOUND_BLOCK_REDUCTION_PARAMETER
            in kernel.launch.parameters)
        # The typed source-bound service is one complete scalar-normalization
        # dependency: block reduction, variance division, epsilon addition,
        # reciprocal square root, and scalar publication.  Keep the ordinary
        # per-element square/add stream, but do not charge any of those
        # enclosed row-level operations again on the generic FMA/SFU pipes.
        reduction_semantic_flops = (
            0.0 if source_bound_block_reduction else
            reduction_tree_flops + epsilon_add_flops)
        reciprocal_sqrt_operations = (
            0.0 if source_bound_block_reduction else float(rows))
        reduction_extra_cycles = 0.0
        if hardware.vector_reduce_sum_latency_lut is not None:
            reduction = hardware.resource_by_role("compute_reduction")
            vector = hardware.resource_by_role("compute_vector")
            if reduction.total_units <= 0 or vector.total_units <= 0:
                raise ValueError(
                    "Vector reduction resources require physical units")
            reduction_latency_ns = hardware.vector_reduce_sum_latency_ns(
                hidden)
            reduction_cycles_per_second = (
                reduction.capacity / reduction.total_units)
            vector_ops_per_second = vector.capacity / vector.total_units
            # The ordinary arithmetic contract below already includes one
            # reduction add per element.  ReduceSum measures that arithmetic
            # plus its tree/barrier dependency.  Price only the excess service
            # time here, otherwise the elementwise adds are counted twice.
            measured_cycles = (
                reduction_latency_ns * 1e-9
                * reduction_cycles_per_second)
            ideal_add_cycles = (
                hidden * reduction_cycles_per_second
                / vector_ops_per_second)
            reduction_extra_cycles_per_row = (
                measured_cycles - ideal_add_cycles)
            if reduction_extra_cycles_per_row <= 0:
                raise ValueError(
                    "measured ReduceSum service must exceed its ideal vector "
                    "add time")
            reduction_extra_cycles = rows * reduction_extra_cycles_per_row
        reduction_resource_demand = (
            float(rows)
            if source_bound_block_reduction else reduction_extra_cycles)
        if "norm_weight_l2_copies" in kernel.launch.parameters:
            weight_l2_copies = int(
                kernel.launch.parameters["norm_weight_l2_copies"])
        else:
            weight_l2_copies = rows
        weight_l2_bytes = weight_l2_copies * hidden * dtype_bytes
        weight_dram_bytes = hidden * dtype_bytes
        if variant == "rmsnorm":
            data_bytes = 2 * elements * dtype_bytes
        elif variant == "add_rmsnorm":
            # Fused add-RMSNorm reads x and residual, then writes both the
            # normalized output and the updated residual in place.
            data_bytes = 4 * elements * dtype_bytes
        else:
            raise AssertionError(
                f"unhandled normalization variant: {variant!r}")
        if "input_reread_count" in kernel.launch.parameters:
            reread_count = int(
                kernel.launch.parameters["input_reread_count"])
        else:
            reread_count = 0
        if "rstd_output_bytes" in kernel.launch.parameters:
            rstd_logical_output_bytes = float(
                kernel.launch.parameters["rstd_output_bytes"])
        else:
            rstd_logical_output_bytes = 0.0
        if "cann_rmsnorm_rstd_transfer_bytes" in kernel.launch.parameters:
            if variant != "rmsnorm":
                raise ValueError(
                    "CANN RMSNorm rstd transfer geometry has an invalid "
                    "variant contract")
            rstd_transfer_bytes = float(kernel.launch.parameters[
                "cann_rmsnorm_rstd_transfer_bytes"])
        else:
            rstd_transfer_bytes = rstd_logical_output_bytes
        reread_bytes = reread_count * elements * dtype_bytes
        # A source-defined reread is issued at the local cache, but it only
        # belongs to the downstream structured path when that hardware does
        # not expose an enabled L1 for this GPU memory model.  The retained
        # A800 source and physical L2-sector observations show that the
        # post-reduction input/residual pass is locally resident.  C500 has
        # L1 disabled and CANN uses its MTE2 ingress, so both retain the
        # existing downstream reread without a hardware-ID branch.
        reread_l1_only_bytes = (
            float(reread_bytes)
            if (hardware.kernel_memory_model == "gpu_mshr"
                and hardware.l1_cache_enabled)
            else 0.0
        )
        reread_l2_bytes = float(reread_bytes - reread_l1_only_bytes)
        reread_compulsory_dram_bytes = 0.0
        if (hardware.kernel_memory_model == "gpu_mshr"
                and not hardware.l1_cache_enabled):
            if hardware.l2_size_bytes <= 0:
                raise ValueError(
                    "GPU normalization reread requires positive L2 capacity")
            # At the start of the source-defined second pass, no more than one
            # L2 capacity of the reread tensor can remain resident.  Price only
            # the compulsory capacity overflow; the unknown CTA order may evict
            # more data but cannot evict less than this lower bound.
            reread_compulsory_dram_bytes = max(
                0.0, reread_l2_bytes - hardware.l2_size_bytes)
        # CANN 8.5.1 AddRmsNormSplitD (key 31) issues ``CopyInGamma`` once
        # for every row group, then ``CopyInX`` for every latter-path row.
        # Both are explicit Gm->UB DataCopyCustom operations in the fixed
        # source program.  Unlike the generic normalization contract, that
        # program contains no source-level L2-residency or reuse guarantee,
        # so neither stream may be silently demoted to an L2-only read.  This
        # is a versioned source-program fact, not a shape or timing rule, and
        # is shared by Dataset and Serving through the same launch identity.
        cann_add_rms_split_d_source = (
            variant == "add_rmsnorm"
            and _CANN_ADD_RMS_SPLIT_D_DAG_PARAMETER
            in kernel.launch.parameters
            and kernel.launch.parameters[
                _CANN_ADD_RMS_SPLIT_D_DAG_PARAMETER] == 1.0)
        if cann_add_rms_split_d_source:
            if (hardware.kernel_memory_model != "npu_tile"
                    or kernel.launch.algorithm != "aclnn_add_rmsnorm"
                    or int(kernel.launch.parameters["cann_tiling_key"]) != 31
                    or reread_l1_only_bytes != 0.0):
                raise ValueError(
                    "CANN AddRMSNorm Split-D source identity disagrees with "
                    "its global-read program")
            weight_dram_bytes = weight_l2_bytes
            reread_compulsory_dram_bytes = reread_l2_bytes
        l2_total = float(
            data_bytes + weight_l2_bytes + reread_l2_bytes
            + rstd_transfer_bytes)
        # Both variants are priced by the same per-element arithmetic tally;
        # only the residual stream they carry differs, and that difference is
        # a byte account plus the one extra add the tally already contains.
        arithmetic_per_element = _norm_arithmetic_operations_per_element(
            variant)
        residual_add_operations = float(
            _NORM_RESIDUAL_ADD_OPERATIONS_PER_ELEMENT[variant] * elements)
        square_accumulate_operations = float(
            _NORM_SQUARE_ACCUMULATE_OPERATIONS_PER_ELEMENT * elements)
        output_multiply_operations = float(
            _NORM_OUTPUT_MULTIPLY_OPERATIONS_PER_ELEMENT * elements)
        if math.fsum((
                residual_add_operations,
                square_accumulate_operations,
                output_multiply_operations,
        )) != arithmetic_per_element * elements:
            raise AssertionError(
                "normalization per-element arithmetic decomposition drifted "
                "from its source tally")
        if variant == "rmsnorm":
            reduction_input_streams = 1
            reduction_write_bytes = 0.0
            final_write_bytes = float(
                elements * dtype_bytes + rstd_transfer_bytes)
        elif variant == "add_rmsnorm":
            reduction_input_streams = 2
            reduction_write_bytes = float(elements * dtype_bytes)
            final_write_bytes = float(
                elements * dtype_bytes + rstd_transfer_bytes)
        else:
            raise AssertionError("unexpected normalized RMSNorm variant")
        typed_add_rms_rounds = (
            variant == "add_rmsnorm"
            and _has_typed_add_rms_round_services(kernel, hardware))
        typed_rmsnorm_scalar_rounds = (
            variant == "rmsnorm"
            and _has_typed_rmsnorm_scalar_round_services(kernel, hardware))
        typed_norm_rounds = (
            typed_add_rms_rounds or typed_rmsnorm_scalar_rounds)
        generic_fma_operations = (
            0.0 if typed_norm_rounds else
            arithmetic_per_element * elements + reduction_semantic_flops)
        generic_alu_operations = (
            0.0 if typed_norm_rounds else bf16_conversion_operations)
        flops = {
            "compute_tensor": 0,
            "compute_reduction": reduction_resource_demand,
            # ``_norm_arithmetic_operations_per_element`` carries the
            # shared source tally: one multiply-add square accumulation and
            # two output multiplies for both variants, plus the fused
            # variant's residual add.  GPU CUB reduction tree and epsilon-add
            # work are source-mandatory row terms outside that tally.  A
            # typed source round already executes these instructions, so the
            # generic tally is withheld there rather than charged twice.
            "compute_fma": generic_fma_operations,
            "compute_alu": generic_alu_operations,
            # A generic reduction computes one reciprocal square root per row;
            # the typed source-bound reduction service already contains it.
            "compute_sfu": reciprocal_sqrt_operations,
        }
        if variant == "rmsnorm":
            read_l2 = float(
                elements * dtype_bytes + weight_l2_bytes
                + reread_l2_bytes)
            read_dram = float(
                elements * dtype_bytes + weight_dram_bytes
                + reread_compulsory_dram_bytes)
            write = float(elements * dtype_bytes + rstd_transfer_bytes)
        elif variant == "add_rmsnorm":
            read_l2 = float(
                2 * elements * dtype_bytes + weight_l2_bytes
                + reread_l2_bytes)
            read_dram = float(
                2 * elements * dtype_bytes + weight_dram_bytes
                + reread_compulsory_dram_bytes)
            write = float(2 * elements * dtype_bytes + rstd_transfer_bytes)
        else:
            raise AssertionError(
                f"unhandled normalization traffic: {variant!r}")
        # RMSNorm writes a distinct output, while fused Add-RMSNorm updates
        # its two input buffers in place. Both therefore retain two row-sized
        # tensors plus the shared weight and any scalar rstd output.
        resident_footprint_bytes = float(
            (2 * elements + hidden) * dtype_bytes + rstd_transfer_bytes)
        dram_write = _completion_dependent_dram_write_bytes(
            hardware,
            resident_footprint_bytes=resident_footprint_bytes,
            write_bytes=write,
        )
        # ``compulsory`` stays the cold traffic envelope the roofline baseline
        # consumes; only the served demand follows the residency curve.
        compulsory_dram_total = float(read_dram + write)
        read_dram = _resident_operand_dram_read_bytes(
            hardware,
            resident_footprint_bytes=resident_footprint_bytes,
            read_bytes=read_dram,
            compulsory_dram_read_bytes=float(
                weight_dram_bytes + reread_compulsory_dram_bytes),
        )
        dram_total = float(read_dram + dram_write)
        if reduction_write_bytes + final_write_bytes != write:
            raise AssertionError(
                "normalization phase write decomposition drifted")
        if (read_l2 + write != l2_total
                or read_dram + dram_write != dram_total):
            raise AssertionError("normalization traffic decomposition drifted")
        demand = _vector_demand(
            hardware,
            l2_read_bytes=read_l2,
            dram_read_bytes=read_dram,
            write_bytes=write,
            dram_write_bytes=dram_write,
            flops=flops,
            l1_read_bytes=read_l2 + reread_l1_only_bytes)
        load_reduce_l2 = float(
            reduction_input_streams * elements * dtype_bytes)
        load_compute_l2 = float(weight_l2_bytes + reread_l2_bytes)
        load_compute_dram = float(
            weight_dram_bytes + reread_compulsory_dram_bytes)
        # The row operand stream is the only read that L2 residency can
        # absorb; the shared weight row's first touch and any compulsory
        # re-read remain DRAM traffic in both regimes.
        load_reduce_dram = float(read_dram - load_compute_dram)
        if (load_reduce_l2 + load_compute_l2 != read_l2
                or load_reduce_dram + load_compute_dram != read_dram):
            raise AssertionError(
                "normalization phase read decomposition drifted")
        phase_blueprint = _target_phase_blueprint(
            hardware,
            _VectorPhaseBlueprint(
                phases=(
                    _VectorPhaseSpec(
                        name="load_reduce",
                        predecessors=(),
                        completion_barrier=False,
                        l2_read_bytes=load_reduce_l2,
                        dram_read_bytes=load_reduce_dram,
                        write_bytes=0.0,
                        compute_demands=(),
                    ),
                    _VectorPhaseSpec(
                        name="input_compute",
                        predecessors=("load_reduce",),
                        completion_barrier=False,
                        l2_read_bytes=0.0,
                        dram_read_bytes=0.0,
                        write_bytes=0.0,
                        compute_demands=_phase_compute_demands(
                            ("compute_fma", "residual_add",
                             0.0 if typed_norm_rounds
                             else residual_add_operations),
                            ("compute_fma", "square_accumulate",
                             0.0 if typed_norm_rounds
                             else square_accumulate_operations),
                            ("compute_alu", "bf16_input_conversions",
                             0.0 if typed_norm_rounds else (
                                 float(elements)
                                 if bf16_conversion_operations else 0.0)),
                        ),
                    ),
                ) + ((
                    _VectorPhaseSpec(
                        name="residual_store",
                        predecessors=("input_compute",),
                        completion_barrier=False,
                        l2_read_bytes=0.0,
                        dram_read_bytes=0.0,
                        write_bytes=reduction_write_bytes,
                        compute_demands=(),
                        dram_write_bytes=(
                            dram_write * reduction_write_bytes / write),
                    ),
                ) if reduction_write_bytes > 0 else ()) + (
                    _VectorPhaseSpec(
                        name="reduce_compute",
                        predecessors=((
                            "residual_store" if reduction_write_bytes > 0
                            else "input_compute"),),
                        completion_barrier=True,
                        l2_read_bytes=0.0,
                        dram_read_bytes=0.0,
                        write_bytes=0.0,
                        compute_demands=_phase_compute_demands(
                            ("compute_fma", "block_reduction_and_epsilon",
                             reduction_semantic_flops),
                            ("compute_sfu", "reciprocal_sqrt",
                             reciprocal_sqrt_operations),
                            (
                                "compute_reduction",
                                "source_bound_block_reduction"
                                if source_bound_block_reduction
                                else "reduction_dependency",
                                reduction_resource_demand,
                            ),
                        ),
                    ),
                    _VectorPhaseSpec(
                        name="load_output",
                        predecessors=("reduce_compute",),
                        completion_barrier=False,
                        l2_read_bytes=load_compute_l2,
                        dram_read_bytes=load_compute_dram,
                        write_bytes=0.0,
                        compute_demands=(),
                    ),
                    _VectorPhaseSpec(
                        name="output_compute",
                        predecessors=("load_output",),
                        completion_barrier=False,
                        l2_read_bytes=0.0,
                        dram_read_bytes=0.0,
                        write_bytes=0.0,
                        compute_demands=_phase_compute_demands(
                            ("compute_fma", "output_multiply",
                             0.0 if typed_norm_rounds
                             else output_multiply_operations),
                            ("compute_alu", "bf16_output_conversions",
                             0.0 if typed_norm_rounds else (
                                 bf16_conversion_operations - elements
                                 if bf16_conversion_operations else 0.0)),
                        ),
                    ),
                    _VectorPhaseSpec(
                        name="store",
                        predecessors=("output_compute",),
                        completion_barrier=False,
                        l2_read_bytes=0.0,
                        dram_read_bytes=0.0,
                        write_bytes=final_write_bytes,
                        compute_demands=(),
                        dram_write_bytes=(
                            dram_write * final_write_bytes / write),
                    ),
                ),
                lowered_sfu_phase="reduce_compute",
            ),
        )
        inp = _input(
            kernel, hardware, demand, write_bytes=write,
            dram_write_bytes=dram_write,
            phase_blueprint=phase_blueprint)
        inp = _attach_a800_add_rmsnorm_mixed_memory_service(
            inp, kernel, hardware,
            resident_footprint_bytes=resident_footprint_bytes)
        # A second vec8 source round establishes the recurrent input/output
        # phase order carried by the CTA pipeline.  A single-round CTA keeps
        # the full-chip service as an aggregate bound, while a footprint over
        # L2 already selects the cold-stream DRAM endpoint.
        if (resident_footprint_bytes <= hardware.l2_size_bytes
                and hidden > kernel.launch.block_size * 8):
            inp = _stage_exact_memory_access_classes(inp)
        if hardware.analytical_vector_dependency_paths:
            has_source_rmsnorm_reduction = any(
                service.service == "rmsnorm_half_interval_reduce"
                for compute in inp.compute_demands
                for service in compute.service_demands)
            if (inp.demand_per_resource["compute_reduction"] <= 0
                    and not has_source_rmsnorm_reduction):
                inp = dataclasses.replace(
                    inp,
                    limitations=inp.limitations + (
                        "norm_reduction_dependency_service_unavailable",),
                )
            if (variant == "rmsnorm"
                    and SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                    in kernel.launch.parameters):
                row_iterations = math.ceil(
                    hidden / kernel.launch.block_size)
                aggregate_read_rounds = math.ceil(
                    2 * hidden / kernel.launch.block_size)
                dependent_read_rounds = (
                    2 * row_iterations - aggregate_read_rounds)
            else:
                dependent_read_rounds = 1
            if (SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
                    in kernel.launch.parameters):
                cta_batches = _source_bound_memory_dependency_stages(
                    inp, kernel, hardware)
                if (variant == "rmsnorm"
                        and kernel.launch.grid_size == 1
                        and inp.task_duration_floor_contract
                        == WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR):
                    # A one-CTA RMSNorm is one complete task.  Its typed task
                    # floor is already max-composed with the full source path;
                    # adding the benchmark body's non-empty-minus-empty
                    # increment to that same CTA would count its dependent
                    # Load -> Compute -> Store body a second time.  Multi-CTA
                    # launches retain the explicit per-active-task increment.
                    dependent_task_stages = 0
                else:
                    # RMS has one post-reduction load->normalize->store stage.
                    # Add-RMS additionally has a source-defined first-pass
                    # load->add->residual-store stage before BlockReduce.
                    dependent_task_stages = cta_batches * {
                        "rmsnorm": 1,
                        "add_rmsnorm": 2,
                    }[variant]
            else:
                generic_task_batches = (
                    _generic_memory_dependency_stage_batches(
                        inp, kernel, hardware))
                # Fused Add-RMS has two source-ordered complete memory tasks:
                # update and publish residual before the collective, then
                # reread/scale/store the normalized output.  Plain RMS has no
                # first-pass global store and retains one complete task.
                dependent_task_stages = generic_task_batches * {
                    "rmsnorm": 1,
                    "add_rmsnorm": 2,
                }[variant]
            additional_concurrent_service_groups = ()
            if (inp.mem_model == "npu_tile"
                    and "cann_tiling_key" in kernel.launch.parameters):
                cann_tiling_key = int(
                    kernel.launch.parameters["cann_tiling_key"])
                if cann_tiling_key in (1, 31):
                    additional_concurrent_service_groups = ((
                        "memory:mte2_read",
                        "compute:compute_vector",
                        "memory:mte3_write",
                    ),)
            inp = _attach_linear_vector_dependency_path(
                inp, hardware, name=variant,
                compute_roles=(
                    "compute_vector", "compute_fma",
                    "compute_reduction", "compute_sfu", "compute_alu"),
                # For a typed RMS source, the finite-source aggregate already
                # contains both source passes.  Only a partially populated
                # final iteration can lose a round when their transactions are
                # pooled.  Other backends retain the explicit post-reduction
                # reread latency until they expose an equivalent source model.
                additional_dependent_reads=dependent_read_rounds,
                dependent_task_stages=dependent_task_stages,
                # CANN split-D keys 1/31 necessarily stream multiple column
                # tiles because one row exceeds the full-row UB capacity.
                # MTE2, Vector, and MTE3 are independent hardware endpoints,
                # so their steady-state tile services overlap.  The row
                # reduction remains outside this group and therefore stays
                # serialized between the input and output passes.
                additional_concurrent_service_groups=(
                    additional_concurrent_service_groups))
        inp = _attach_a800_fp16_rmsnorm_single_cta_completion_floor(
            inp, kernel, hardware)
        return inp, compulsory_dram_total
    if variant in ("elementwise", "elementwise_pytorch_vec4"):
        elements = _shape(kernel, "element_count")
        reads = int(kernel.attributes["reads"])
        writes = int(kernel.attributes["writes"])
        operation = kernel.attributes["operation"]
        if operation in _CANN_ELEMENTWISE_COMPARE_CAST_SOURCE_ABI:
            # Routing-mask compare/cast pricing: the source-level element ABI
            # (int32->bool compare, bool->BF16 cast), not the BF16 spec dtype,
            # fixes the byte envelope; the serialized completion service
            # attached inside ``_input`` owns the Vector work on its critical
            # worker.
            if (hardware.kernel_memory_model != "npu_tile"
                    or not hardware.aic_aiv_separated):
                raise ValueError(
                    "CANN elementwise compare/cast is 910B-only: the source "
                    "ABI family requires separated npu_tile AIVs")
            read_bytes_per_element, write_bytes_per_element = (
                _CANN_ELEMENTWISE_COMPARE_CAST_SOURCE_ABI[operation])
            read = float(elements * read_bytes_per_element)
            write = float(elements * write_bytes_per_element)
            total = read + write
            # One vector-lane comparison/conversion per element.
            flops = {
                "compute_tensor": 0,
                "compute_vector": float(elements),
                "compute_fma": 0,
                "compute_alu": 0,
                "compute_sfu": 0,
            }
            dram_write = _completion_dependent_dram_write_bytes(
                hardware,
                resident_footprint_bytes=read + write,
                write_bytes=write,
            )
            dram_read = _resident_operand_dram_read_bytes(
                hardware,
                resident_footprint_bytes=read + write,
                read_bytes=read,
                compulsory_dram_read_bytes=0.0,
            )
            demand = _vector_demand(
                hardware, l2_read_bytes=read, dram_read_bytes=dram_read,
                write_bytes=write, dram_write_bytes=dram_write,
                flops=flops)
            inp = _input(
                kernel, hardware, demand, write_bytes=write,
                dram_write_bytes=dram_write, phase_blueprint=None)
            return inp, total
        # A BF16 -> FP32 cast reads two bytes and writes four per element.
        read_bytes_per_element, write_bytes_per_element = (
            (2, 4) if operation == "cast_bf16_float32"
            else (dtype_bytes, dtype_bytes))
        total = float(elements * (
            reads * read_bytes_per_element
            + writes * write_bytes_per_element))
        if operation in ("add", "mul", "unary"):
            arithmetic = float(elements)
        elif operation in ("copy", "fill", "read", "cast_bf16_float32"):
            arithmetic = 0.0
        else:
            raise AssertionError(
                f"KernelSpec allowed unknown elementwise operation "
                f"{operation!r}")
        bf16_conversion_operations = (
            _source_bf16_conversion_operations(kernel))
        # The independent add-square/scale-multiply round characterizes the
        # fused Add-RMSNorm stages, not plain elementwise Add or Mul. Keep
        # elementwise work on its source-level arithmetic and conversion
        # services instead of borrowing a heavier fused-task service.
        vector_round_operations = 0.0
        flops = {"compute_tensor": 0,
                 "compute_vector": vector_round_operations,
                 "compute_fma": arithmetic,
                 "compute_alu": bf16_conversion_operations,
                 "compute_sfu": 0}
        read = float(elements * reads * read_bytes_per_element)
        write = float(elements * writes * write_bytes_per_element)
        # A single-pass elementwise kernel completes once its stores are
        # accepted by coherent L2. If the complete read/write footprint fits
        # in L2, no dirty output line is forced to DRAM before completion.
        dram_write = _completion_dependent_dram_write_bytes(
            hardware,
            resident_footprint_bytes=read + write,
            write_bytes=write,
        )
        dram_read = _resident_operand_dram_read_bytes(
            hardware,
            resident_footprint_bytes=read + write,
            read_bytes=read,
            compulsory_dram_read_bytes=0.0,
        )
        demand = _vector_demand(
            hardware, l2_read_bytes=read, dram_read_bytes=dram_read,
            write_bytes=write, dram_write_bytes=dram_write, flops=flops)
        phase_blueprint = _target_phase_blueprint(
            hardware,
            _linear_phase_blueprint(
                l2_read_bytes=read,
                dram_read_bytes=dram_read,
                write_bytes=write,
                dram_write_bytes=dram_write,
                compute_demands=_phase_compute_demands(
                    ("compute_vector", "bf16_vector_memory_round",
                     vector_round_operations),
                    ("compute_fma", f"elementwise_{operation}", arithmetic),
                    ("compute_alu", "bf16_fp32_conversions",
                     bf16_conversion_operations),
                ),
            ),
        )
        inp = _input(
            kernel, hardware, demand, write_bytes=write,
            dram_write_bytes=dram_write,
            phase_blueprint=phase_blueprint)
        inp = _attach_c500_elementwise_mixed_dram_service(
            inp, kernel, hardware)
        c500_typed_mixed_dram_stream = (
            {
                demand.service
                for path in inp.memory_paths
                for demand in path.access_class_demands
            } in ({
                _C500_ELEMENTWISE_2R1W_DRAM_READ_SERVICE,
                _C500_ELEMENTWISE_2R1W_DRAM_WRITE_SERVICE,
            }, {
                _C500_ELEMENTWISE_2R1W_PARTIAL_DRAM_READ_SERVICE,
                _C500_ELEMENTWISE_2R1W_PARTIAL_DRAM_WRITE_SERVICE,
            }))
        forward_context = copy_context()
        if _FORWARD_CANN_ANALYTICAL_ENVELOPE in forward_context:
            use_cann_analytical_envelope = forward_context[
                _FORWARD_CANN_ANALYTICAL_ENVELOPE]
        else:
            use_cann_analytical_envelope = False
        if ((hardware.analytical_vector_dependency_paths
             or c500_typed_mixed_dram_stream)
                and (inp.mem_model != "npu_tile"
                     or use_cann_analytical_envelope)):
            # A plain elementwise map has no whole-tensor barrier between its
            # input, arithmetic, and output stages: independent elements may
            # stream through distinct memory and Vector pipes.  GPU targets
            # either opt into the general dependency-path contract or bind
            # the exact C500 tail-free 2R1W stream.  The latter still needs
            # the independently decomposed submit prefix before its device
            # stream, while read/compute/write remain concurrent.  The
            # current NPU contract has neither the CANN tile/queue schedule
            # nor MTE startup/credit evidence needed to price pipeline fill
            # and drain, so retain the aggregate resource envelope instead
            # of imposing the physically false ordering ``all reads -> all
            # compute -> all writes``.
            if (SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
                    in kernel.launch.parameters):
                task_model = hardware.memory_dependency_task_latency_model
                if task_model is None:
                    raise ValueError(
                        "source-bound elementwise task requires a hardware "
                        "task model")
                if (task_model["model"] == (
                        "isolated_warm_default_cache_"
                        "memory_dependency_task_v1")
                        and inp.task_duration_floor_contract
                        == WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR):
                    # This mechanism is a complete Load -> dependent op ->
                    # Store task floor.  Formula already takes its maximum
                    # with the analytical source body, so serially appending
                    # its non-empty increment would count that body twice.
                    dependent_task_stages = 0
                else:
                    dependent_task_stages = (
                        _source_bound_memory_dependency_stages(
                            inp, kernel, hardware))
            else:
                dependent_task_stages = (
                    _generic_memory_dependency_stage_batches(
                        inp, kernel, hardware))
            inp = _attach_linear_vector_dependency_path(
                inp, hardware, name=f"elementwise_{operation}",
                compute_roles=(
                    "compute_vector", "compute_fma", "compute_alu",
                    "warp_issue"),
                dependent_task_stages=dependent_task_stages,
                stream_data_services=(
                    inp.mem_model != "npu_tile"
                    or sum((
                        int(any(path.name == "mte2_read"
                                for path in inp.memory_paths)),
                        int(any(path.name == "mte3_write"
                                for path in inp.memory_paths)),
                        int(any(
                            inp.demand_per_resource[role] > 0
                            for role in (
                                "compute_vector", "compute_fma",
                                "compute_alu"))),
                    )) >= 2))
        return inp, total
    if variant == "flash_mla_metadata":
        read_bytes, write_bytes = _flash_mla_metadata_bytes(kernel)
        total = read_bytes + write_bytes
        control_work = float(
            _shape(kernel, "kv_block_count")
            + _shape(kernel, "scheduler_metadata_ints")
            + _shape(kernel, "split_metadata_ints"))
        flops = {
            "compute_tensor": 0,
            "compute_fma": 0,
            "compute_reduction": _flash_mla_metadata_dependency_cycles(
                kernel, hardware),
            "compute_alu": control_work,
            "compute_sfu": 0,
        }
        demand = _vector_demand(
            hardware, l2_read_bytes=read_bytes,
            dram_read_bytes=read_bytes, write_bytes=write_bytes,
            flops=flops)
        return _input(
            kernel, hardware, demand, write_bytes=write_bytes,
            phase_blueprint=None,
            operator_aux=True), total
    if variant in ("rotary", "rope"):
        elements = _shape(kernel, "element_count")
        tokens = _shape(kernel, "seq_len")
        head_dim = _shape(kernel, "head_dim")
        if variant == "rotary":
            rotary_dim = head_dim
        elif variant == "rope":
            rotary_dim = _shape(kernel, "rotary_dim")
        else:
            raise AssertionError(f"unhandled RoPE variant: {variant!r}")
        query_pairs = (
            _shape(kernel, "num_query_heads") * rotary_dim // 2)
        if elements % (2 * tokens):
            raise ValueError(
                "RoPE active elements must form complete per-token pairs")
        total_pairs = elements // (2 * tokens)
        key_pairs = total_pairs - query_pairs
        if key_pairs < 0:
            raise ValueError("RoPE active elements are smaller than Q extent")
        serial_rotation_rounds = (
            math.ceil(query_pairs / kernel.launch.block_size)
            + (math.ceil(key_pairs / kernel.launch.block_size)
               if key_pairs else 0))
        index_bytes = _serving_vector_index_bytes(hardware)
        # element_count is the active rotary extent of Q plus the optional K.
        # RoPE reads and writes those elements once, reads one position per
        # token, and reads only the selected rows/rotary width from the
        # pre-built cos/sin cache.
        tensor_read = elements * dtype_bytes
        if index_bytes is None:
            # PIM/Gemmini are compatibility models without one of the three
            # serving ABIs.  Their architecture adapters retain the original
            # tensor-only contract rather than borrowing GPU/NPU metadata.
            position_read = 0
            cos_sin_read = 0
        else:
            position_read = tokens * index_bytes
            cos_sin_read = tokens * rotary_dim * dtype_bytes
        write = float(tensor_read)
        # RoPE updates the rotated tensor in place, so the same lines are both
        # read and written.  The measured residency curve characterizes a
        # stream over distinct read and write buffers, and its forbidden_use
        # excludes extrapolation outside that stream contract, so this variant
        # keeps its compulsory operand read until an in-place stream is
        # measured.
        dram_read = float(tensor_read + position_read + cos_sin_read)
        if (not graph_padding
                and "rotary_cache_path_bytes" in kernel.launch.parameters):
            cache_path_bytes = float(
                kernel.launch.parameters["rotary_cache_path_bytes"])
        else:
            cache_path_bytes = float(cos_sin_read)
        if (not graph_padding
                and "rotary_position_path_bytes" in kernel.launch.parameters):
            position_cache_path_bytes = float(
                kernel.launch.parameters["rotary_position_path_bytes"])
        else:
            position_cache_path_bytes = float(position_read)
        # Source-issued repeated cos/sin loads are physical local-cache
        # requests.  Only the unique cache row footprint continues to L2 and
        # DRAM; charging every head-loop load to L2 would treat logical loads
        # as downstream physical transactions.
        l1_read = float(
            tensor_read + position_cache_path_bytes + cache_path_bytes)
        if hardware.l1_cache_enabled:
            l2_read = dram_read
        else:
            # C500 disables Vector L1.  Its source-issued repeated cos/sin
            # loads and per-wave position loads therefore reach L2, while L2
            # residency still keeps their compulsory DRAM footprint unique.
            l2_read = l1_read
        total = dram_read + write
        # Each pair produces two outputs; each output uses two multiplies and
        # one add/subtract: six scalar operations per two elements.
        if (not graph_padding
                and SOURCE_QK_ROPE_LOOP_PARAMETER
                in kernel.launch.parameters):
            if kernel.launch.parameters[
                    SOURCE_QK_ROPE_LOOP_PARAMETER] != 1.0:
                raise ValueError("source Q/K RoPE loop marker must equal one")
            # The versioned C500 ABI10 and ABI11 NeoX sources retain the same
            # K-present, contiguous-head Q/K loop and address expression:
            # nine ordinary index/address operators per active pair plus
            # seven setup/control operators per launched thread.  The ABI10
            # binary SHA256 is
            # a8715e7f7fe1790b46586fd8780036177c083504b24556ae1d661d0f0b569955:
            # rot_dim / 2 lowers to shift/add at 0xb8c0..0xb8d0, while each Q/K
            # quotient and remainder pair shares one reciprocal/correction
            # sequence at 0xb9c0..0xbab8 and 0xbde0..0xbed8.  The measured
            # remainder service contains that quotient-producing sequence, so
            # charging a separate division service would duplicate work.
            active_pairs = total_pairs * tokens
            address_operations = float(
                9 * active_pairs + 7 * tokens * kernel.launch.block_size)
            expected_key_pairs = (
                _shape(kernel, "num_kv_heads") * rotary_dim // 2)
            if key_pairs != expected_key_pairs:
                raise ValueError(
                    "source Q/K RoPE loop disagrees with key rotary extent")
            if kernel.launch.block_size != min(query_pairs, 512):
                raise ValueError(
                    "source Q/K RoPE loop disagrees with source block size")
            qk_dependency_rounds = (
                _source_wave_dependency_layers(query_pairs, hardware)
                + _source_wave_dependency_layers(key_pairs, hardware))
            integer_rem_dependency_rounds = float(
                tokens * qk_dependency_rounds)
            source_pair_wave_rounds = float(tokens * qk_dependency_rounds)
            # The source address expression contains one dependent 64-bit
            # multiply-add per active pair. The remaining seven scalar
            # address/control operations stay on the generic ALU envelope.
            # One pair also performs four BF16-to-FP32 and two FP32-to-BF16
            # conversions, equal to 1.5 independently measured two-way pair
            # round trips per active wave.
            address_multiply_add_dependency_rounds = (
                source_pair_wave_rounds)
            bf16_roundtrip_pair_dependency_rounds = (
                1.5 * source_pair_wave_rounds)
            generic_address_operations = float(
                address_operations - 2 * active_pairs)
            if generic_address_operations < 0:
                raise ValueError(
                    "source Q/K RoPE typed address work exceeds source work")
            source_qk_rope_loop = True
        else:
            # Other source contracts retain the architecture-neutral semantic
            # address work used before a source loop is published.
            # The source-bound ATB program instead exposes scalar address
            # calculation once per token/slice.  Do not retain this generic
            # per-element Vector placeholder on top of its exact DAG.
            address_operations = (
                0.0
                if (not graph_padding
                    and _CANN_ATB_BF16_ROPE_DAG_PARAMETER
                    in kernel.launch.parameters)
                else 4.0 * elements)
            integer_rem_dependency_rounds = 0.0
            address_multiply_add_dependency_rounds = 0.0
            bf16_roundtrip_pair_dependency_rounds = 0.0
            generic_address_operations = address_operations
            source_qk_rope_loop = False
        bf16_conversion_operations = (
            _source_bf16_conversion_operations(kernel))
        if source_qk_rope_loop:
            if bf16_conversion_operations != 6.0 * active_pairs:
                raise ValueError(
                    "source Q/K RoPE BF16 conversion work disagrees with "
                    "the source pair loop")
            generic_alu_operations = generic_address_operations
        else:
            generic_alu_operations = (
                generic_address_operations + bf16_conversion_operations)
        flops = {"compute_tensor": 0, "compute_fma": 3.0 * elements,
                 "compute_vector": (
                     integer_rem_dependency_rounds
                     + address_multiply_add_dependency_rounds
                     + bf16_roundtrip_pair_dependency_rounds),
                 "compute_alu": generic_alu_operations,
                 "compute_sfu": 0}
        # RoPE rewrites the exact query/key lines it just loaded.  The store
        # therefore lands on a line this kernel itself brought into L2, so it
        # completes there for the same reason the store-completion scope
        # gives any resident store -- and without needing the residency
        # curve, whose distinct-buffer stream contract correctly excludes the
        # in-place *read* above.  The resident footprint is the tensor alone,
        # not read plus write, because the two alias.  Every A800 RoPE case
        # measures exactly its read extent as DRAM traffic and none of its
        # write extent.
        #
        # The 910B AIC/AIV L2 is the same kind of coherent, capacity-bounded
        # write-back cache: a store only has to be accepted into L2 before
        # kernel completion, not drained to DRAM, provided nothing evicts
        # that line first.  Every measured 910B RoPE case -- 28 shapes up to
        # the largest captured tensor, 35.92 MB, all well under the
        # independently measured L2 replacement capacity -- reports
        # dram__bytes_write == 0, the same in-place-alias reasoning gated by
        # the replacement-capacity field npu_tile hardware already publishes
        # (gpu_mshr's l2_size_bytes/gpu_store_completion_scope pair has no
        # npu_tile counterpart, so this stays local to the RoPE branch
        # rather than widening the shared GPU completion-scope helper to
        # every other npu_tile Vector variant without their own evidence).
        if hardware.kernel_memory_model == "npu_tile":
            dram_write = (
                0.0
                if float(tensor_read) <= hardware.l2_replacement_capacity_bytes
                else write)
        else:
            dram_write = _completion_dependent_dram_write_bytes(
                hardware,
                resident_footprint_bytes=float(tensor_read),
                write_bytes=write,
            )
        demand = _vector_demand(
            hardware, l2_read_bytes=l2_read, dram_read_bytes=dram_read,
            write_bytes=write, dram_write_bytes=dram_write, flops=flops,
            l1_read_bytes=l1_read)
        phase_blueprint = _target_phase_blueprint(
            hardware,
            _linear_phase_blueprint(
                l2_read_bytes=l2_read,
                dram_read_bytes=dram_read,
                write_bytes=write,
                dram_write_bytes=dram_write,
                compute_demands=_phase_compute_demands(
                    ("compute_fma", "complex_rotation", 3.0 * elements),
                    ("compute_alu", "remaining_pair_addressing",
                     generic_alu_operations),
                    ("compute_vector", "integer_rotary_remainder",
                     integer_rem_dependency_rounds),
                    ("compute_vector", "address_multiply_add_u64",
                     address_multiply_add_dependency_rounds),
                    ("compute_vector", "bf16_roundtrip_pair",
                     bf16_roundtrip_pair_dependency_rounds),
                ),
            ),
        )
        inp = _input(
            kernel, hardware, demand, write_bytes=write,
            dram_write_bytes=dram_write,
            phase_blueprint=phase_blueprint)
        if source_qk_rope_loop:
            for service in (
                    _INTEGER_REM_U32_DEPENDENCY_SERVICE,
                    _BF16_ROUNDTRIP_PAIR_DEPENDENCY_SERVICE,
                    _ADDRESS_MULTIPLY_ADD_U64_DEPENDENCY_SERVICE):
                if not _has_compute_service(
                        hardware, "compute_vector", service):
                    raise ValueError(
                        "source Q/K RoPE loop requires typed uint32 "
                        f"service {service!r}")
            inp = _replace_vector_service_demands(
                inp,
                (
                    ComputeServiceDemand(
                        _INTEGER_REM_U32_DEPENDENCY_SERVICE,
                        integer_rem_dependency_rounds),
                    ComputeServiceDemand(
                        _ADDRESS_MULTIPLY_ADD_U64_DEPENDENCY_SERVICE,
                        address_multiply_add_dependency_rounds),
                    ComputeServiceDemand(
                        _BF16_ROUNDTRIP_PAIR_DEPENDENCY_SERVICE,
                        bf16_roundtrip_pair_dependency_rounds),
                ),
            )
        if source_qk_rope_loop:
            # The source executes separate Q and K loops inside each CTA.  A
            # loop iteration consumes at most one block of pairs, and its
            # loads must complete before that iteration's dependent address,
            # conversion, rotation, and store instructions can retire.
            round_pairs = tuple(
                min(kernel.launch.block_size, query_pairs - offset)
                for offset in range(0, query_pairs, kernel.launch.block_size)
            ) + tuple(
                min(kernel.launch.block_size, key_pairs - offset)
                for offset in range(0, key_pairs, kernel.launch.block_size)
            )
            if (len(round_pairs) != serial_rotation_rounds
                    or sum(round_pairs) != total_pairs):
                raise ValueError("source Q/K RoPE rounds do not conserve pairs")
            round_waves = tuple(
                _source_wave_dependency_layers(pairs, hardware)
                for pairs in round_pairs)
            if sum(round_waves) != qk_dependency_rounds:
                raise ValueError(
                    "source Q/K RoPE rounds do not conserve wave layers")
            worker_count = min(tokens, inp.active_units)
            if worker_count != inp.active_units:
                raise ValueError(
                    "source Q/K RoPE requires one worker per active token")
            full_chip_source_latency_overlap = (
                worker_count == hardware.num_compute_units)
            per_token_read_bytes = l2_read / tokens
            per_token_write_bytes = write / tokens
            per_token_address_operations = generic_address_operations / tokens
            fixed_address_operations = float(
                7 * kernel.launch.block_size)
            expected_read_bytes = float(
                4 * dtype_bytes * total_pairs + index_bytes)
            expected_write_bytes = float(
                2 * dtype_bytes * total_pairs)
            expected_address_operations = float(
                7 * total_pairs + fixed_address_operations)
            if (not math.isclose(
                    per_token_read_bytes, expected_read_bytes,
                    rel_tol=1e-12, abs_tol=1e-6)
                    or not math.isclose(
                        per_token_write_bytes, expected_write_bytes,
                        rel_tol=1e-12, abs_tol=1e-6)
                    or not math.isclose(
                        per_token_address_operations,
                        expected_address_operations,
                        rel_tol=1e-12, abs_tol=1e-6)):
                raise ValueError(
                    "source Q/K RoPE round decomposition changed")
            stages = []
            for round_index, (pairs, waves) in enumerate(zip(
                    round_pairs, round_waves)):
                read_bytes = float(4 * dtype_bytes * pairs)
                address_ops = float(7 * pairs)
                if round_index == 0:
                    read_bytes += float(index_bytes)
                    address_ops += fixed_address_operations
                load_name = (
                    "load" if round_index == 0
                    else f"round{round_index}_load")
                store_name = (
                    "store" if round_index == len(round_pairs) - 1
                    else f"round{round_index}_store")
                prefix = f"round{round_index}"
                stages.extend((
                    PipelineStageDemand(
                        load_name, "memory:global_read",
                        read_bytes / l2_read,
                        # Once every AP has a source CTA, later contiguous Q/K
                        # rounds inherit the same in-flight request stream.
                        # Keep the cold completion only on the first round,
                        # matching the full-chip C500 attention pipeline;
                        # partial-chip launches retain each source edge.
                        include_first_request_latency=(
                            round_index == 0
                            or not full_chip_source_latency_overlap)),
                    PipelineStageDemand(
                        f"{prefix}_source_dependency",
                        "compute:compute_vector",
                        waves / integer_rem_dependency_rounds),
                    PipelineStageDemand(
                        f"{prefix}_remaining_address",
                        "compute:compute_alu",
                        address_ops / generic_address_operations),
                    PipelineStageDemand(
                        f"{prefix}_rotation", "compute:compute_fma",
                        pairs / total_pairs / tokens),
                    PipelineStageDemand(
                        store_name, "memory:global_write",
                        (2 * dtype_bytes * pairs) / write),
                ))
            steady_stages = (
                PipelineStageDemand(
                    "load", "memory:global_read", 1.0 / tokens,
                    include_first_request_latency=True),
                PipelineStageDemand(
                    "source_dependency", "compute:compute_vector",
                    1.0 / tokens),
                PipelineStageDemand(
                    "remaining_address", "compute:compute_alu",
                    1.0 / tokens),
                PipelineStageDemand(
                    "rotation", "compute:compute_fma", 1.0 / tokens),
                PipelineStageDemand(
                    "store", "memory:global_write", 1.0 / tokens),
            )
            # Only the first CTA admitted to an AP exposes every cold
            # per-loop completion edge.  Later resident/queued CTAs retain
            # the aggregate source path: multiplying cold completions by CTA
            # depth would discard the hardware's inter-CTA latency hiding.
            base_ctas, extra_ctas = divmod(tokens, worker_count)
            ctas_by_worker = tuple(
                base_ctas + (1 if worker < extra_ctas else 0)
                for worker in range(worker_count))
            entries = []
            for cta_count in tuple(dict.fromkeys(ctas_by_worker)):
                physical_workers = tuple(
                    worker for worker, count in enumerate(ctas_by_worker)
                    if count == cta_count)
                compact_worker = physical_workers[0]
                first = PipelineTaskSequenceDemand(
                    name="first_cta",
                    worker=compact_worker,
                    tasks=(PipelineTaskDemand(
                        name="first_source_qk_rope_cta",
                        worker=compact_worker,
                        stages=tuple(stages)),),
                )
                patterns = (first,)
                order = (0,)
                if cta_count > 1:
                    steady = PipelineTaskSequenceDemand(
                        name="steady_cta",
                        worker=compact_worker,
                        tasks=(PipelineTaskDemand(
                            name="steady_source_qk_rope_cta",
                            worker=compact_worker,
                            stages=steady_stages),),
                    )
                    patterns += (steady,)
                    order += (1,) * (cta_count - 1)
                entries.append(PipelineTaskProgramDemand(
                    name=f"source_qk_rope_{cta_count}_cta_worker_class",
                    worker=compact_worker,
                    patterns=patterns,
                    order=order,
                    worker_multiplicity=len(physical_workers),
                    physical_workers=physical_workers,
                ))
            plan = PipelineWorkPlan(
                worker_count=worker_count,
                tasks=tuple(entries),
                buffers=(PipelineBuffer(
                    name="resident_cta_slots",
                    producer_stage="load",
                    consumer_stage="store",
                    slots=_thread_limited_resident_ctas(kernel, hardware),
                    scope="worker",
                ),),
            )
            inp = attach_pipeline_work_plan(inp, hardware, plan)
        elif hardware.analytical_vector_dependency_paths:
            if (SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
                    in kernel.launch.parameters):
                dependent_task_stages = _source_bound_memory_dependency_stages(
                    inp, kernel, hardware)
            else:
                dependent_task_stages = (
                    _generic_memory_dependency_stage_batches(
                        inp, kernel, hardware))
            inp = _attach_linear_vector_dependency_path(
                inp, hardware, name="rope",
                compute_roles=(
                    "compute_vector", "compute_alu", "compute_fma",
                    "compute_sfu"),
                additional_dependent_reads=(
                    serial_rotation_rounds),
                dependent_task_stages=dependent_task_stages)
        return inp, total
    if variant in ("activation", "silu"):
        elements = _shape(kernel, "element_count")
        total = float(elements * 3 * dtype_bytes)
        source_bf16_conversion_operations = (
            _source_bf16_conversion_operations(kernel))
        uses_nonrecurrent_cta_pipeline = (
            not graph_padding
            and SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER
            in kernel.launch.parameters)
        has_scalar_round_service = _has_compute_service(
            hardware, "compute_vector", _SILU_SCALAR_ROUND_SERVICE)
        canonical_scalar_round_launch = (
            kernel.launch.grid_size == _shape(kernel, "tok")
            and kernel.launch.block_size == 1024)
        uses_scalar_round_service = (
            not graph_padding
            and has_scalar_round_service
            and canonical_scalar_round_launch
            and not uses_nonrecurrent_cta_pipeline
            and kernel.launch.grid_size <= hardware.num_compute_units)
        has_exp_dependency_service = _has_compute_service(
            hardware, "compute_sfu", _EXP_DEPENDENCY_SERVICE)
        has_reciprocal_dependency_service = _has_compute_service(
            hardware, "compute_sfu",
            _RECIPROCAL_ONE_PLUS_DEPENDENCY_SERVICE)
        has_source_math_layout = (
            not graph_padding
            and SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER
            in kernel.launch.parameters)
        uses_math_dependency_services = (
            has_exp_dependency_service
            and has_reciprocal_dependency_service
            and has_source_math_layout
            # The source-bound non-recurrent CTA schedule is the more
            # specific path for large, one-CTA-per-token launches.  It
            # subsumes the generic exp/reciprocal service pair there.
            and not uses_nonrecurrent_cta_pipeline)
        uses_exp_dependency_service = uses_math_dependency_services
        uses_reciprocal_dependency_service = uses_math_dependency_services
        if (int(uses_nonrecurrent_cta_pipeline)
                + int(uses_scalar_round_service)
                + int(uses_math_dependency_services) > 1):
            raise ValueError(
                "SiLU cannot use multiple semantic dependency services")
        if uses_nonrecurrent_cta_pipeline:
            scalar_rounds = _silu_scalar_rounds(kernel)
            entry_operations = float(_shape(kernel, "tok"))
            flops = {
                "compute_tensor": 0,
                "compute_vector": scalar_rounds + entry_operations,
                "compute_fma": float(elements),
                "compute_alu": source_bf16_conversion_operations,
                "compute_sfu": 0,
            }
            phase_compute = _phase_compute_demands(
                ("compute_vector", "source_silu_cta_entry",
                 entry_operations),
                ("compute_vector", "source_silu_rounds", scalar_rounds),
                ("compute_fma", "gate_multiply", elements),
                ("compute_alu", "bf16_fp32_conversions",
                 source_bf16_conversion_operations),
            )
        elif uses_scalar_round_service:
            scalar_rounds = _silu_scalar_rounds(kernel)
            # The typed service covers x/(1 + exp(-x)); the independent gate
            # multiplication remains ordinary FMA-pipe work.  This separates
            # semantic dependency depth from instruction-count modeling.
            flops = {
                "compute_tensor": 0,
                "compute_vector": scalar_rounds,
                "compute_fma": float(elements),
                "compute_alu": source_bf16_conversion_operations,
                "compute_sfu": 0,
            }
            phase_compute = _phase_compute_demands(
                ("compute_vector", "exp_add_reciprocal", scalar_rounds),
                ("compute_fma", "gate_multiply", elements),
                ("compute_alu", "bf16_fp32_conversions",
                 source_bf16_conversion_operations),
            )
        elif uses_math_dependency_services:
            exp_dependency_rounds = _silu_exp_dependency_rounds(kernel)
            typed_service_count = (
                int(uses_exp_dependency_service)
                + int(uses_reciprocal_dependency_service))
            remaining_conversion_operations = (
                source_bf16_conversion_operations)
            remaining_fma_operations = float(
                (2 if uses_reciprocal_dependency_service else 3)
                * elements)
            # The typed reciprocal service includes the source's +1. Both
            # source-math mechanisms are required together so the SFU phase
            # remains source-closed rather than mixing typed and generic work.
            flops = {
                "compute_tensor": 0,
                "compute_fma": remaining_fma_operations,
                "compute_alu": remaining_conversion_operations,
                "compute_sfu": (
                    typed_service_count * exp_dependency_rounds),
            }
            phase_compute_items = []
            if not uses_exp_dependency_service:
                phase_compute_items.append(
                    ("compute_sfu", "exp", elements))
            if uses_exp_dependency_service:
                phase_compute_items.append((
                    "compute_sfu", "exp_dependency",
                    exp_dependency_rounds))
            if uses_reciprocal_dependency_service:
                phase_compute_items.append((
                    "compute_sfu", "reciprocal_one_plus_dependency",
                    exp_dependency_rounds))
            else:
                phase_compute_items.append(
                    ("compute_sfu", "reciprocal", elements))
            phase_compute_items.extend((
                ("compute_fma", "activation_arithmetic",
                 flops["compute_fma"]),
                ("compute_alu", "bf16_fp32_conversions",
                 remaining_conversion_operations),
            ))
            phase_compute = _phase_compute_demands(*phase_compute_items)
        else:
            # vLLM act_and_mul evaluates x/(1+exp(-x))*gate.  Negation is an
            # operand modifier; the chain has three ordinary arithmetic
            # operations and two special functions (exp and reciprocal).
            flops = {"compute_tensor": 0, "compute_fma": 3.0 * elements,
                     "compute_alu": source_bf16_conversion_operations,
                     "compute_sfu": 2.0 * elements}
            phase_compute = _phase_compute_demands(
                ("compute_fma", "activation_arithmetic", 3.0 * elements),
                ("compute_sfu", "exp_and_reciprocal", 2.0 * elements),
                ("compute_alu", "bf16_fp32_conversions",
                 source_bf16_conversion_operations),
            )
        read = float(2 * elements * dtype_bytes)
        write = float(elements * dtype_bytes)
        # act_and_mul reads two operand halves it did not produce and writes
        # one result.  The read-side residency credit assumes a producer left
        # those operands in L2; this kernel's own L2ExecutionContext declares
        # no same-program operand replay, and the canonical capture evicts
        # cache before the measured launch, so no producer did.  The launch's
        # own non-duration L2 hit-rate counter settles it independently: a
        # 2:1 read:write stream measures 35.1% (T64) and 40.2% (T256) L2 hit
        # against the 33.3% that "every read misses, every write is absorbed"
        # predicts, and against the 66.7% the opposite credit would require.
        # Charge the reads as the compulsory misses they are, and give the
        # result stream the store-completion scope the hardware already
        # publishes.
        if uses_nonrecurrent_cta_pipeline:
            dram_read = read
            dram_write = _completion_dependent_dram_write_bytes(
                hardware,
                resident_footprint_bytes=read + write,
                write_bytes=write,
            )
        else:
            dram_read = _resident_operand_dram_read_bytes(
                hardware,
                resident_footprint_bytes=read + write,
                read_bytes=read,
                compulsory_dram_read_bytes=0.0,
            )
            dram_write = write
        demand = _vector_demand(
            hardware, l2_read_bytes=read, dram_read_bytes=dram_read,
            write_bytes=write, dram_write_bytes=dram_write, flops=flops)
        phase_blueprint = _target_phase_blueprint(
            hardware,
            _linear_phase_blueprint(
                l2_read_bytes=read,
                dram_read_bytes=dram_read,
                write_bytes=write,
                dram_write_bytes=dram_write,
                compute_demands=phase_compute,
            ),
        )
        inp = _input(
            kernel, hardware, demand, write_bytes=write,
            dram_write_bytes=dram_write,
            phase_blueprint=phase_blueprint)
        if uses_nonrecurrent_cta_pipeline:
            inp = _attach_a800_silu_nonrecurrent_cta_pipeline(
                inp, kernel, hardware)
        elif uses_scalar_round_service:
            inp = _attach_silu_semantic_dependency_path(
                inp, kernel, hardware)
        elif uses_math_dependency_services:
            inp = _attach_silu_source_cta_pipeline(
                inp, kernel, hardware,
                use_exp_dependency=uses_exp_dependency_service,
                use_reciprocal_dependency=(
                    uses_reciprocal_dependency_service))
        elif has_scalar_round_service:
            inp = dataclasses.replace(
                inp,
                limitations=inp.limitations + (
                    "silu_scalar_dependency_launch_outside_support",),
            )
            if (canonical_scalar_round_launch
                    and hardware.analytical_vector_dependency_paths):
                dependent_task_stages = (
                    _source_bound_memory_dependency_stages(
                        inp, kernel, hardware)
                    if SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
                    in kernel.launch.parameters
                    else _generic_memory_dependency_stage_batches(
                        inp, kernel, hardware))
                inp = _attach_linear_vector_dependency_path(
                    inp, hardware, name="silu",
                    compute_roles=(
                        "compute_vector", "compute_fma", "compute_sfu",
                        "compute_alu"),
                    dependent_task_stages=dependent_task_stages)
        elif hardware.analytical_vector_dependency_paths:
            inp = _attach_linear_vector_dependency_path(
                inp, hardware, name="silu",
                compute_roles=(
                    "compute_vector", "compute_fma", "compute_sfu",
                    "compute_alu"),
                dependent_task_stages=(
                    _generic_memory_dependency_stage_batches(
                        inp, kernel, hardware)))
        return inp, total
    if variant == "paged_kv_cache":
        base_elements = _shape(kernel, "element_count")
        tokens = _shape(kernel, "seq_len")
        index_bytes = _serving_vector_index_bytes(hardware)
        # The canonical count describes one logical K or V tensor. The serving
        # op reads K and V plus one slot per token, then writes both tensors
        # into the paged cache.  The GPU runners pass ``cache_dtype=auto``;
        # that compile-time path does not dereference their scale pointers.
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": 0, "compute_sfu": 0}
        kv_compute_demands = ()
        source_scalar_kv_loop = False
        if SOURCE_SCALAR_KV_LOOP_PARAMETER in kernel.launch.parameters:
            if kernel.launch.parameters[
                    SOURCE_SCALAR_KV_LOOP_PARAMETER] != 1.0:
                raise ValueError("source scalar KV loop marker must equal one")
            elements_per_token = (
                _shape(kernel, "num_kv_heads")
                * _shape(kernel, "head_dim"))
            if (base_elements != tokens * elements_per_token
                    or kernel.launch.grid_size != tokens
                    or kernel.launch.block_size
                    != min(elements_per_token, 512)):
                raise ValueError(
                    "source scalar KV loop disagrees with shape or launch")
            # Frozen C500 source: one slot guard plus two page-coordinate
            # terms per launched thread, and twelve ordinary address/control
            # operators per K/V body element.  Integer div/rem remains an
            # explicit limitation until a matching typed hardware service is
            # available.
            alu_operations = float(tokens * (
                3 * kernel.launch.block_size + 12 * elements_per_token))
            flops["compute_alu"] = alu_operations
            loop_rounds = math.ceil(
                elements_per_token / kernel.launch.block_size)
            # The typed integer service is characterized by one physical
            # wave64.  Slot/page coordinates execute once in every launched
            # wave; head coordinates execute once in every active scalar-loop
            # wave layer, including the final partial layer.  Independent
            # waves occupy the four architectural PEUs in parallel.
            u64_dependency_rounds = float(
                tokens * _source_wave_dependency_layers(
                    kernel.launch.block_size, hardware))
            u32_dependency_rounds = float(
                tokens * _source_wave_dependency_layers(
                    elements_per_token, hardware))
            flops["compute_vector"] = float(
                2.0 * u64_dependency_rounds
                + 2.0 * u32_dependency_rounds)
            kv_compute_demands = _phase_compute_demands(
                ("compute_alu", "address_and_control", alu_operations),
                ("compute_vector", "slot_page_div_u64",
                 u64_dependency_rounds),
                ("compute_vector", "slot_page_rem_u64",
                 u64_dependency_rounds),
                ("compute_vector", "head_div_u32",
                 u32_dependency_rounds),
                ("compute_vector", "head_rem_u32",
                 u32_dependency_rounds),
            )
            source_scalar_kv_loop = True
        elif SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER in (
                kernel.launch.parameters):
            if kernel.dtype != "bf16":
                raise ValueError(
                    "source-bound KV-cache program work requires BF16")
            elements_per_token = (
                _shape(kernel, "num_kv_heads")
                * _shape(kernel, "head_dim"))
            if base_elements != tokens * elements_per_token:
                raise ValueError(
                    "source-bound KV-cache element count is inconsistent")
            vector_packs_per_cta = elements_per_token // 8
            # Static source operators in reshape_and_cache_flash_kernel and
            # vectorize_with_alignment's aligned path.  Every launched thread
            # performs the slot guard, extent formation, four source/dest
            # pointer expressions, the layout test, and both fast-path setup
            # and initial loop tests: 29 integer/address operators.  Each
            # participating vec8 thread additionally performs an increment
            # and loop test in both the K and V calls: four operators.  The
            # slot page quotient and remainder are two source-level integer
            # div/rem operations per launched thread.  These counts describe
            # C++ semantics, not compiler instructions or SASS.
            alu_operations = float(tokens * (
                29 * kernel.launch.block_size
                + 4 * vector_packs_per_cta))
            divrem_operations = float(
                2 * tokens * kernel.launch.block_size)
            flops["compute_alu"] = alu_operations
            flops["compute_sfu"] = divrem_operations
            kv_compute_demands = _phase_compute_demands(
                ("compute_alu", "address_and_control", alu_operations),
                ("compute_sfu", "slot_page_divrem", divrem_operations),
            )
        tensor_read = 2 * base_elements * dtype_bytes
        slot_reads_per_token = 1
        scalar_get_value_requests = 0
        if "cann_tiling_key" in kernel.launch.parameters:
            tiling_key = int(kernel.launch.parameters["cann_tiling_key"])
            if tiling_key == _CANN_RESHAPE_CACHE_EQUAL_KEY:
                slot_reads_per_token = 1
                scalar_get_value_requests = tokens
            elif tiling_key == _CANN_RESHAPE_CACHE_FALLBACK_KEY:
                # ProcessKVUnequal calls the K and V queue paths separately;
                # each path obtains its own slot mapping from global memory.
                slot_reads_per_token = 2
                scalar_get_value_requests = 2 * tokens
            else:
                raise ValueError(
                    "unsupported CANN ReshapeAndCache tiling key: "
                    f"{tiling_key}")
        if scalar_get_value_requests:
            if any(flops[role] != 0 for role in flops):
                raise ValueError(
                    "CANN scalar GetValue cannot mix with another KV "
                    "compute contract")
            flops["compute_vector"] = float(scalar_get_value_requests)
            kv_compute_demands = _phase_compute_demands((
                "compute_vector", "scalar_global_get_value",
                scalar_get_value_requests,
            ))
        slot_read = (
            0 if index_bytes is None
            else tokens * index_bytes * slot_reads_per_token)
        if "kv_cache_slot_l1_bytes" in kernel.launch.parameters:
            slot_l1_read = float(
                kernel.launch.parameters["kv_cache_slot_l1_bytes"])
        else:
            slot_l1_read = float(slot_read)
        slot_l2_read = (
            float(slot_read)
            if hardware.l1_cache_enabled
            else slot_l1_read)
        l2_read = float(tensor_read + slot_l2_read)
        l1_read = float(tensor_read + slot_l1_read)
        write = float(tensor_read)
        # The cache write targets a distinct paged allocation, so the resident
        # footprint is the source tensor plus its copy.  Slot-mapping indices
        # are launch metadata and keep their compulsory DRAM bytes.
        compulsory_read = float(tensor_read + slot_read)
        dram_read = _resident_operand_dram_read_bytes(
            hardware,
            resident_footprint_bytes=float(tensor_read + write),
            read_bytes=compulsory_read,
            compulsory_dram_read_bytes=float(slot_read),
        )
        total = compulsory_read + write
        # The paged-cache store completes once coherent L2 accepts it, the
        # same store-completion scope act_and_mul and the elementwise path
        # already use; this path simply never applied it.  The measured
        # counters agree that it should: every A800 KV_CACHE case reports
        # exactly its read extent as DRAM traffic and none of its write
        # extent.
        dram_write = _completion_dependent_dram_write_bytes(
            hardware,
            resident_footprint_bytes=float(tensor_read + write),
            write_bytes=write,
        )
        demand = _vector_demand(
            hardware, l2_read_bytes=l2_read, dram_read_bytes=dram_read,
            write_bytes=write, dram_write_bytes=dram_write, flops=flops,
            l1_read_bytes=l1_read)
        if ("cann_tiling_key" in kernel.launch.parameters
                and int(kernel.launch.parameters["cann_tiling_key"])
                == _CANN_RESHAPE_CACHE_EQUAL_KEY):
            tensor_bytes = float(base_elements * dtype_bytes)
            phase_blueprint = _target_phase_blueprint(
                hardware,
                _VectorPhaseBlueprint(
                    phases=(
                        _VectorPhaseSpec(
                            name="key_load",
                            predecessors=(),
                            completion_barrier=True,
                            l2_read_bytes=tensor_bytes + slot_read,
                            dram_read_bytes=tensor_bytes + slot_read,
                            write_bytes=0.0,
                            compute_demands=kv_compute_demands,
                        ),
                        _VectorPhaseSpec(
                            name="key_store",
                            predecessors=("key_load",),
                            completion_barrier=True,
                            l2_read_bytes=0.0,
                            dram_read_bytes=0.0,
                            write_bytes=tensor_bytes,
                            compute_demands=(),
                        ),
                        _VectorPhaseSpec(
                            name="value_load",
                            predecessors=("key_store",),
                            completion_barrier=True,
                            l2_read_bytes=tensor_bytes,
                            dram_read_bytes=tensor_bytes,
                            write_bytes=0.0,
                            compute_demands=(),
                        ),
                        _VectorPhaseSpec(
                            name="value_store",
                            predecessors=("value_load",),
                            completion_barrier=True,
                            l2_read_bytes=0.0,
                            dram_read_bytes=0.0,
                            write_bytes=tensor_bytes,
                            compute_demands=(),
                        ),
                    ),
                    lowered_sfu_phase="value_load",
                ),
            )
        else:
            phase_blueprint = _target_phase_blueprint(
                hardware,
                _linear_phase_blueprint(
                    l2_read_bytes=l2_read,
                    dram_read_bytes=dram_read,
                    write_bytes=write,
                    dram_write_bytes=dram_write,
                    compute_demands=kv_compute_demands,
                ),
            )
        inp = _input(
            kernel, hardware, demand, write_bytes=write,
            dram_write_bytes=dram_write,
            phase_blueprint=phase_blueprint)
        if scalar_get_value_requests:
            inp = _attach_cann_kv_scalar_get_value_service(
                inp, kernel, hardware, slot_reads_per_token)
        if source_scalar_kv_loop:
            typed_integer_services = (
                ComputeServiceDemand(
                    _INTEGER_DIV_U64_DEPENDENCY_SERVICE,
                    u64_dependency_rounds),
                ComputeServiceDemand(
                    _INTEGER_REM_U64_DEPENDENCY_SERVICE,
                    u64_dependency_rounds),
                ComputeServiceDemand(
                    _INTEGER_DIV_U32_DEPENDENCY_SERVICE,
                    u32_dependency_rounds),
                ComputeServiceDemand(
                    _INTEGER_REM_U32_DEPENDENCY_SERVICE,
                    u32_dependency_rounds),
            )
            for service_demand in typed_integer_services:
                if not _has_compute_service(
                        hardware, "compute_vector",
                        service_demand.service):
                    raise ValueError(
                        "source scalar KV loop requires typed integer "
                        f"service {service_demand.service!r}")
            inp = _replace_vector_service_demands(
                inp, typed_integer_services)
        if source_scalar_kv_loop:
            inp = attach_uniform_pipeline(
                inp, hardware,
                task_count=kernel.launch.grid_size,
                stages=(
                    ("load", "memory:global_read", True),
                    ("integer_address", "compute:compute_vector", False),
                    ("address_control", "compute:compute_alu", False),
                    ("store", "memory:global_write", False),
                ),
                buffer_slots=_thread_limited_resident_ctas(
                    kernel, hardware),
            )
        elif (hardware.analytical_vector_dependency_paths
              or SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
              in kernel.launch.parameters):
            source_bound_copy = (
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
                in kernel.launch.parameters)
            if source_bound_copy:
                if hardware.memory_dependency_task_latency_model is None:
                    # One complete copy program runs per CTA (vec8 and the
                    # pinned scalar body alike).  Its dependency-chain depth
                    # is the same "how many sequential CTA-batches does the
                    # busiest AP traverse" quantity every other source-bound
                    # copy path on this hardware already derives from the
                    # architectural (or source-measured) resident-CTA count
                    # via _source_bound_cta_batches; dividing by active-unit
                    # count alone -- as if only one CTA could ever be
                    # resident per AP -- ignored that admission concurrency
                    # and made the predicted chain grow linearly with grid
                    # size even though a resident AP can advance several
                    # CTAs' independent load latencies at once.
                    cta_batches = _source_bound_cta_batches(
                        inp, kernel, hardware)
                    dependent_task_stages = 0
                else:
                    cta_batches = _source_bound_cta_batches(
                        inp, kernel, hardware)
                    # Every source CTA executes its K copy before its V copy.
                    # Across C resident/queued cohorts those two ordered
                    # regions form an ideal two-stage pipeline of depth C+1.
                    dependent_task_stages = cta_batches + 1
                source_copy_rounds = 1
                if (SOURCE_VECTORIZED_KV_COPY_PARAMETER
                        in kernel.launch.parameters):
                    # The source-selected C500 BF16 NHD fast path lowers
                    # vectorize_with_alignment into one scalar K/V loop whose
                    # induction step is blockDim.x.  Each iteration performs
                    # the K and V loads/stores in the same loop body.  H4D128
                    # has one 512-element round; H8D128 has two ordered
                    # rounds.  Aggregate traffic already prices both rounds,
                    # but the prior critical path included only one read
                    # completion per CTA batch.  Preserve the same generic
                    # latency service and derive its exact source depth from
                    # static elements-per-token and block geometry.
                    source_copy_rounds = math.ceil(
                        elements_per_token / kernel.launch.block_size)
                    if source_copy_rounds <= 0:
                        raise ValueError(
                            "C500 source KV copy must have a positive "
                            "scalar-loop round count")
                dependent_reads = cta_batches * source_copy_rounds
            else:
                dependent_reads = 2
                dependent_task_stages = (
                    _generic_memory_dependency_stage_batches(
                        inp, kernel, hardware))
            inp = _attach_linear_vector_dependency_path(
                inp, hardware, name="kv_cache",
                compute_roles=(
                    "compute_vector", "compute_alu", "compute_sfu"),
                additional_dependent_reads=dependent_reads,
                dependent_task_stages=dependent_task_stages,
                stream_data_services=(
                    SOURCE_VECTORIZED_KV_COPY_PARAMETER
                    in kernel.launch.parameters
                    and elements_per_token <= kernel.launch.block_size
                    and kernel.launch.grid_size <= inp.active_units))
        return inp, total
    if variant == "reduce":
        elements = _shape(kernel, "element_count")
        factor = int(kernel.attributes["reduction_factor"])
        total = float(elements * dtype_bytes * factor)
        flops = {"compute_tensor": 0, "compute_fma": 3.0 * elements,
                 "compute_alu": float(elements), "compute_sfu": float(elements)}
        write = float(elements * dtype_bytes)
        read = total - write
        return _input(
            kernel, hardware,
            _vector_demand(
                hardware, l2_read_bytes=read, dram_read_bytes=read,
                write_bytes=write, flops=flops),
            write_bytes=write, phase_blueprint=None), total
    if variant == "gather":
        elements = _shape(kernel, "element_count")
        factor = int(kernel.attributes["gather_factor"])
        index_count = int(kernel.attributes["index_count"])
        index_bytes = int(kernel.attributes["index_bytes"])
        total = float(elements * dtype_bytes * factor + index_count * index_bytes)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(elements), "compute_sfu": 0}
        efficiency = (
            hardware.l2_conflict_efficiency_moe
            if kernel.attributes["scattered"] else 1.0)
        # The gather contract counts source/index reads.  It does not expose a
        # distinct materialized-output count, so retain the complete traffic on
        # the read path rather than inventing an MTE3 split.
        return _input(
            kernel, hardware,
            _vector_demand(
                hardware, l2_read_bytes=total, dram_read_bytes=total,
                write_bytes=0.0, flops=flops),
            write_bytes=0.0,
            phase_blueprint=None,
            dram_capacity_multiplier=efficiency), total
    if variant == "fp8_group_quant":
        # Dynamic 1xG FP8 activation quantisation (vLLM
        # per_token_group_quant_8bit): read the BF16 tensor once, reduce each
        # group's absolute maximum, scale, convert, and write the e4m3 tensor
        # plus one FP32 scale per group.
        elements = _shape(kernel, "element_count")
        groups = elements // int(kernel.attributes["group_size"])
        read = float(elements * dtype_bytes)
        write = float(elements + groups * 4)
        flops = {"compute_tensor": 0, "compute_fma": float(elements),
                 "compute_alu": float(2 * elements),
                 "compute_sfu": float(groups)}
        return _input(
            kernel, hardware,
            _vector_demand(
                hardware, l2_read_bytes=read, dram_read_bytes=read,
                write_bytes=write, flops=flops),
            write_bytes=write, phase_blueprint=None), read + write
    if variant == "softmax":
        elements = _shape(kernel, "element_count")
        total = float(elements * 2 * dtype_bytes)
        flops = {"compute_tensor": 0, "compute_fma": float(elements),
                 "compute_alu": 0, "compute_sfu": float(elements)}
        read = float(elements * dtype_bytes)
        write = read
        return _input(
            kernel, hardware,
            _vector_demand(
                hardware, l2_read_bytes=read, dram_read_bytes=read,
                write_bytes=write, flops=flops),
            write_bytes=write, phase_blueprint=None), total
    if variant in ("gemm_reduce", "gemm_elementwise"):
        elements = _shape(kernel, "element_count")
        if variant == "gemm_reduce":
            if kernel.launch.algorithm == "mcblas_gemm_reduce_unresolved_abi":
                raise NotImplementedError(
                    "analytically selected mcBLAS split-K reducer remains "
                    "unsupported without exact reducer launch geometry")
            supported_reducers = {
                "observed_gemm_aux:gemm_reduce",
                "observed_mcblas_gemm_aux:gemm_reduce",
                "logical_gemm_aux:gemm_reduce",
            }
            if kernel.launch.algorithm not in supported_reducers:
                raise ValueError(
                    "unknown split-K reducer launch contract: "
                    f"{kernel.launch.algorithm!r}")
            split_k = int(kernel.attributes["split_k"])
            if split_k <= 1:
                raise ValueError(
                    "split-K reducer requires split_k greater than one")
            partial_element_bytes = int(
                kernel.launch.parameters[
                    "split_k_partial_bytes_per_element"])
            partial_dram_element_bytes = int(
                kernel.launch.parameters[
                    "split_k_partial_dram_bytes_per_element"])
            read = float(elements * split_k * partial_element_bytes)
            dram_read = float(
                elements * split_k * partial_dram_element_bytes)
            write = float(elements * dtype_bytes)
            arithmetic = float(elements * (split_k - 1))
            total = dram_read + write
        elif variant == "gemm_elementwise":
            read = float(elements * dtype_bytes)
            dram_read = read
            write = float(elements * dtype_bytes)
            arithmetic = float(elements)
            total = read + write
        else:
            raise AssertionError(f"unhandled GEMM auxiliary variant: {variant!r}")
        flops = {"compute_tensor": 0, "compute_fma": arithmetic,
                 "compute_alu": 0, "compute_sfu": 0}
        phase_blueprint = _target_phase_blueprint(
            hardware,
            _linear_phase_blueprint(
                l2_read_bytes=read,
                dram_read_bytes=dram_read,
                write_bytes=write,
                compute_demands=_phase_compute_demands(
                    ("compute_fma", f"{variant}_add", arithmetic),
                ),
            ),
        )
        resource_input = _input(
            kernel, hardware,
            _vector_demand(
                hardware, l2_read_bytes=read, dram_read_bytes=dram_read,
                write_bytes=write, flops=flops),
            write_bytes=write, phase_blueprint=phase_blueprint,
            operator_aux=True)
        source_reducer_parameters = {
            SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
            SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
            "warp_instructions_lower_bound",
        }
        present_source_reducer_parameters = (
            source_reducer_parameters & set(kernel.launch.parameters))
        if present_source_reducer_parameters:
            if (variant != "gemm_reduce"
                    or present_source_reducer_parameters
                    != source_reducer_parameters):
                raise ValueError(
                    "source-bound GEMM reducer program must be complete")
            if (hardware.kernel_memory_model != "gpu_mshr"
                    or hardware.parallel_width != 32
                    or kernel.launch.block != (32, 16, 1)
                    or partial_element_bytes != 2
                    or partial_dram_element_bytes != 0):
                raise ValueError(
                    "source-bound GEMM reducer requires the CUDA warp32 FP16 "
                    "output-dtype ABI")
            if elements % 32 != 0:
                raise ValueError(
                    "source-bound GEMM reducer requires complete warp "
                    "segments")
            output_warp_programs = elements // 32
            physical_warps_per_cta = 16
            row_iterations_per_warp = math.ceil(
                output_warp_programs
                / (kernel.launch.grid_size * physical_warps_per_cta))
            active_warps_per_cta = math.ceil(
                output_warp_programs
                / (kernel.launch.grid_size * row_iterations_per_warp))
            expected_transactions = (
                2 * split_k * row_iterations_per_warp)
            if split_k == 3:
                expected_warp_instructions = output_warp_programs * 82
            else:
                expected_warp_instructions = (
                    output_warp_programs * (3 * split_k + 64))
            parameters = kernel.launch.parameters
            if (parameters[
                    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER]
                    != float(active_warps_per_cta)
                    or parameters[
                        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER]
                    != float(expected_transactions)
                    or parameters["warp_instructions_lower_bound"]
                    != float(expected_warp_instructions)):
                raise ValueError(
                    "source-bound GEMM reducer program does not conserve "
                    "output warps, L2 reads, and warp issue work")
            remaining_splits = split_k
            split_read_groups = 0
            for group_width in (16, 8, 4):
                groups, remaining_splits = divmod(
                    remaining_splits, group_width)
                split_read_groups += groups
            if remaining_splits:
                split_read_groups += 1
            dependent_reads = (
                split_read_groups * row_iterations_per_warp - 1)
            resource_input = _attach_linear_vector_dependency_path(
                resource_input,
                hardware,
                name="cublaslt_output_dtype_split_k_reduce",
                compute_roles=("compute_fma", "warp_issue"),
                additional_dependent_reads=dependent_reads,
                support_level="partial",
            )
        if (variant == "gemm_reduce"
                and kernel.launch.algorithm
                == "observed_mcblas_gemm_aux:gemm_reduce"
                and kernel.launch.grid_size <= hardware.num_compute_units):
            # DevDstElementwiseAdd loads one FP32 partial from every split,
            # folds those values through one dependent add chain, and only
            # then publishes the BF16 output.  The aggregate memory service
            # already contains the first completion; retain one additional
            # first-request latency for each remaining source partial without
            # expanding CTAs or output elements.
            resource_input = _attach_linear_vector_dependency_path(
                resource_input,
                hardware,
                name="mcblas_split_k_reduce",
                compute_roles=("compute_fma",),
                additional_dependent_reads=split_k - 1,
                stream_data_services=True,
                support_level="partial",
            )
        if (variant == "gemm_reduce"
                and "split_k_count_observed" in kernel.launch.parameters):
            if kernel.launch.parameters["split_k_count_observed"] != 0.0:
                raise ValueError(
                    "split_k_count_observed only marks an unobserved count")
            if split_k != 2:
                raise ValueError(
                    "an unobserved split count is priced at its lower bound 2")
            resource_input = dataclasses.replace(
                resource_input,
                limitations=resource_input.limitations + (
                    "split_k_count_unobserved_priced_at_lower_bound",
                ),
            )
        if variant == "gemm_reduce" and dram_read == 0:
            resource_input = dataclasses.replace(
                resource_input,
                limitations=resource_input.limitations + (
                    "split_k_partial_dram_residency_unavailable",
                ),
            )
        return resource_input, total
    raise ValueError(f"unknown VECTOR variant: {kernel.variant!r}")


def _validate_moe_aux_execution(variant: str, hardware) -> None:
    memory_model = hardware.kernel_memory_model
    if variant in _CANN_MOE_AUX_VARIANTS:
        if memory_model != "npu_tile":
            raise ValueError(
                f"CANN MoE auxiliary variant {variant!r} requires the "
                "npu_tile execution path")
    elif variant in _GPU_MOE_AUX_VARIANTS:
        if memory_model != "gpu_mshr":
            raise ValueError(
                f"GPU MoE auxiliary variant {variant!r} requires the "
                "gpu_mshr execution path")
    elif variant in _SHARED_MOE_AUX_VARIANTS:
        return
    else:
        raise ValueError(f"unknown MoE auxiliary VECTOR variant: {variant!r}")


def _ceil_div_positive(value: int, divisor: int) -> int:
    if value <= 0 or divisor <= 0:
        raise ValueError("ceil-div operands must be positive")
    return (value + divisor - 1) // divisor


def _cann_int32_aligned_copy_bytes(elements: int) -> int:
    if elements <= 0:
        raise ValueError("CANN int32 copy element count must be positive")
    return 32 * _ceil_div_positive(elements, 8)


def _cann_partitioned_int32_copy_bytes(
        elements: int, cores: int, *, per_core_multiple: int,
        copy_limit: int) -> tuple[int, int]:
    """Return active cores and GM->UB bytes from source DataCopyPad loops."""

    per_core = _ceil_div_positive(elements, cores)
    if per_core_multiple > 1:
        per_core = (
            _ceil_div_positive(per_core, per_core_multiple)
            * per_core_multiple)
    used_cores = min(cores, _ceil_div_positive(elements, per_core))
    tail_elements = elements - (used_cores - 1) * per_core

    def worker_bytes(worker_elements: int) -> int:
        full_loops, tail = divmod(worker_elements, copy_limit)
        copied = full_loops * copy_limit * 4
        if tail:
            copied += _cann_int32_aligned_copy_bytes(tail)
        return copied

    total_bytes = (
        (used_cores - 1) * worker_bytes(per_core)
        + worker_bytes(tail_elements))
    return used_cores, total_bytes


def _cann_moe_compute_expert_tokens_traffic(
        kernel: KernelSpec, hardware) -> tuple[float, float]:
    """Derive source-visible GM traffic for CANN's three int32 programs.

    The selection boundaries and loops come from the public
    ``MoeComputeExpertTokens`` host tiler and Int32SS/M/L kernels.  This counts
    explicit DataCopyPad and InitOutput payloads only.  SyncAll lowering and
    physical write-transaction amplification remain separate evidence gaps.
    """

    parameters = kernel.launch.parameters
    tiling_key = int(parameters["cann_tiling_key"])
    routed_items = _shape(kernel, "tokens") * _shape(kernel, "topk")
    experts = _shape(kernel, "experts")
    physical_cores = int(
        hardware.resource_by_role("compute_vector").total_units)
    if (kernel.launch.grid != (physical_cores, 1, 1)
            or kernel.launch.block != (1, 1, 1)):
        raise ValueError(
            "CANN MoeComputeExpertTokens requires the source-defined full-AIV "
            "launch")

    if tiling_key == 1001:
        used_cores, input_read_bytes = (
            _cann_partitioned_int32_copy_bytes(
                routed_items, physical_cores,
                per_core_multiple=1, copy_limit=routed_items))
        workspace_read_bytes = 0
        write_bytes = 4 * experts * (1 + used_cores)
    elif tiling_key == 1002:
        used_cores, input_read_bytes = (
            _cann_partitioned_int32_copy_bytes(
                routed_items, physical_cores,
                per_core_multiple=1, copy_limit=320))
        workspace_read_bytes = (
            experts * _cann_int32_aligned_copy_bytes(used_cores))
        write_bytes = 4 * experts * (physical_cores + used_cores + 1)
    elif tiling_key == 1003:
        used_cores, one_scan_read_bytes = (
            _cann_partitioned_int32_copy_bytes(
                routed_items, physical_cores,
                per_core_multiple=320, copy_limit=320))
        expert_segments = _ceil_div_positive(experts, 256)
        input_read_bytes = expert_segments * one_scan_read_bytes
        workspace_read_bytes = (
            experts * _cann_int32_aligned_copy_bytes(used_cores))
        write_bytes = 4 * experts * (physical_cores + used_cores + 1)
    else:
        raise ValueError(
            "CANN MoeComputeExpertTokens tiling key must be 1001, 1002, or "
            f"1003, got {tiling_key}")
    return (
        float(input_read_bytes + workspace_read_bytes),
        float(write_bytes),
    )


def _cann_moe_compute_expert_tokens_key1001_compute_service(
        kernel: KernelSpec, hardware) -> Optional[ComputeServiceDemand]:
    """Return key-1001's source-visible vector body by physical AIV.

    ``MoeComputeExpertTokensInt32SS::Compute`` is not an elementwise
    ``routed_items + experts`` helper.  Each input-owning AIV casts its local
    sorted-expert span, initializes its private result vector, then visits
    every integer target between that span's first and last expert.  Every
    visited target executes Duplicate, Compare, Select, and ReduceMax; a
    target present in that worker's span additionally executes ReduceSum.

    These are static source lane counts, not a duration-derived service.  The
    shared InitOutput/SyncAll phase and scalar GetValue/SetValue control stay
    outside this demand: the former has its own exact service and the latter
    remains explicitly unresolved.
    """

    parameters = kernel.launch.parameters
    if int(parameters["cann_tiling_key"]) != 1001:
        return None
    routed_items = _shape(kernel, "tokens") * _shape(kernel, "topk")
    experts = _shape(kernel, "experts")
    counts = tuple(kernel.attributes["tokens_per_expert"])
    if len(counts) != experts or sum(counts) != routed_items:
        raise ValueError(
            "CANN key-1001 routing histogram disagrees with shape")
    physical_cores = int(
        hardware.resource_by_role("compute_vector").total_units)
    if (kernel.launch.grid != (physical_cores, 1, 1)
            or kernel.launch.block != (1, 1, 1)):
        raise ValueError(
            "CANN key-1001 Compute requires the source-defined full-AIV "
            "launch")

    # CalcSortedExpertTiling partitions the sorted stream by this unaligned
    # normal span.  The Int32SS source then assigns the tail span only to the
    # final active AIV; all remaining launched AIVs return from Process.
    normal_items = _ceil_div_positive(routed_items, physical_cores)
    used_cores = min(
        physical_cores, _ceil_div_positive(routed_items, normal_items))
    prefix_ends = []
    prefix = 0
    for count in counts:
        prefix += count
        prefix_ends.append(prefix)

    def expert_at(sorted_index: int) -> int:
        if not 0 <= sorted_index < routed_items:
            raise ValueError(
                "CANN key-1001 sorted expert index is out of range")
        expert = bisect_right(prefix_ends, sorted_index)
        if expert >= experts:
            raise AssertionError(
                "CANN key-1001 routing histogram did not cover its stream")
        return expert

    operations_by_worker = []
    for worker in range(physical_cores):
        if worker >= used_cores:
            operations_by_worker.append(0.0)
            continue
        start = worker * normal_items
        handle_items = min(normal_items, routed_items - start)
        if handle_items <= 0:
            raise AssertionError("CANN key-1001 active AIV has no input")
        first_expert = expert_at(start)
        last_expert = expert_at(start + handle_items - 1)

        # One Cast(handle_items), Duplicate(oneBuf, 8), and
        # Duplicate(output, experts) precede the target loop.
        operations = handle_items + 8 + experts
        for target in range(first_expert, last_expert + 1):
            # Duplicate(targetBuf, 8), Compare, Select, and ReduceMax always
            # execute.  ReduceSum is guarded by the source's isFind branch.
            operations += 8 + 3 * handle_items
            target_start = prefix_ends[target - 1] if target else 0
            target_end = prefix_ends[target]
            if max(start, target_start) < min(
                    start + handle_items, target_end):
                operations += handle_items
        operations_by_worker.append(float(operations))

    operations = math.fsum(operations_by_worker)
    if operations <= 0:
        raise AssertionError("CANN key-1001 source Compute has no vector work")
    return ComputeServiceDemand(
        "generic",
        operations,
        operations_by_worker=tuple(operations_by_worker),
        serial_invocations_by_worker=(0,) * physical_cores,
    )


@lru_cache(maxsize=320)
def _cann_upper_bound_comparison_depths(elements: int) -> Tuple[int, ...]:
    """Return source binary-search depth for every upper-bound position."""

    if not 0 < elements <= 320:
        raise ValueError(
            "CANN key-1003 search round must contain 1..320 elements")
    depths = []
    for upper_bound in range(elements + 1):
        low = 0
        high = elements - 1
        comparisons = 0
        while low <= high:
            comparisons += 1
            middle = (low + high) // 2
            if middle >= upper_bound:
                high = middle - 1
            else:
                low = middle + 1
        depths.append(comparisons)
    return tuple(depths)


def _cann_moe_compute_expert_tokens_search_service(
        kernel: KernelSpec, hardware) -> Optional[ComputeServiceDemand]:
    """Return the key-1003 source search chain assigned to physical AIVs.

    ``MoeComputeExpertTokensInt32L`` partitions the sorted expert-id stream
    into 320-element UB rounds and experts into 256-target segments.  The
    parent FusedMoE routing histogram reconstructs that sorted stream without
    materializing it.  For each round this function mirrors the source's
    segment early-return and counts the exact binary-search comparison tree;
    the independent mechanism service then prices comparisons plus one fixed
    V/S-and-Duplicate setup per executed target.  Neither count uses duration.
    """

    parameters = kernel.launch.parameters
    if int(parameters["cann_tiling_key"]) != 1003:
        return None
    routed_items = _shape(kernel, "tokens") * _shape(kernel, "topk")
    experts = _shape(kernel, "experts")
    counts = tuple(kernel.attributes["tokens_per_expert"])
    if len(counts) != experts or sum(counts) != routed_items:
        raise ValueError(
            "CANN key-1003 routing histogram disagrees with shape")
    prefix_ends = []
    routed_prefix = 0
    for count in counts:
        routed_prefix += count
        prefix_ends.append(routed_prefix)

    prefix_ends = tuple(prefix_ends)

    def expert_at(sorted_index: int) -> int:
        if not 0 <= sorted_index < routed_items:
            raise ValueError(
                "CANN key-1003 sorted expert index is out of range")
        expert = bisect_right(prefix_ends, sorted_index)
        if expert >= experts:
            raise AssertionError(
                "CANN key-1003 routing histogram did not cover its stream")
        return expert

    def comparison_count(
            chunk_start: int, chunk_elements: int,
            target_start: int, target_end: int) -> int:
        """Count a target interval from compressed expert-run boundaries."""

        chunk_end = chunk_start + chunk_elements
        first_expert = expert_at(chunk_start)
        last_expert = expert_at(chunk_end - 1)
        depths = _cann_upper_bound_comparison_depths(chunk_elements)
        if target_start == 0:
            upper_bound = 0
        else:
            upper_bound = min(
                chunk_elements,
                max(0, prefix_ends[target_start - 1] - chunk_start),
            )
        next_target = target_start
        comparisons = 0
        for expert in range(
                max(first_expert, target_start),
                min(last_expert, target_end) + 1):
            expert_start = prefix_ends[expert - 1] if expert else 0
            local_count = max(
                0,
                min(chunk_end, prefix_ends[expert])
                - max(chunk_start, expert_start),
            )
            if local_count == 0:
                continue
            comparisons += (expert - next_target) * depths[upper_bound]
            upper_bound += local_count
            comparisons += depths[upper_bound]
            next_target = expert + 1
        comparisons += (
            target_end - next_target + 1) * depths[upper_bound]
        return comparisons

    physical_cores = int(
        hardware.resource_by_role("compute_vector").total_units)
    handle_per_core = (
        _ceil_div_positive(
            _ceil_div_positive(routed_items, physical_cores), 320)
        * 320)
    used_cores = min(
        physical_cores, _ceil_div_positive(routed_items, handle_per_core))
    tail_elements = routed_items - (used_cores - 1) * handle_per_core

    operations_by_worker = []
    invocations_by_worker = []
    for worker in range(physical_cores):
        if worker >= used_cores:
            worker_elements = 0
        elif worker == used_cores - 1:
            worker_elements = tail_elements
        else:
            worker_elements = handle_per_core
        worker_comparisons = 0
        worker_invocations = 0
        worker_start = worker * handle_per_core
        for chunk_offset in range(0, worker_elements, 320):
            chunk_elements = min(320, worker_elements - chunk_offset)
            chunk_start = worker_start + chunk_offset
            first_expert = expert_at(chunk_start)
            for target_start in range(0, experts, 256):
                target_end = min(experts, target_start + 256) - 1
                if first_expert > target_end:
                    continue
                target_count = target_end - target_start + 1
                worker_invocations += target_count
                worker_comparisons += comparison_count(
                    chunk_start, chunk_elements,
                    target_start, target_end)
        operations_by_worker.append(float(worker_comparisons))
        invocations_by_worker.append(worker_invocations)
    operations = math.fsum(operations_by_worker)
    if operations <= 0 or max(invocations_by_worker) <= 0:
        raise ValueError(
            "CANN key-1003 search schedule must contain positive work")
    vector = hardware.resource_by_role("compute_vector")
    generic_rate = vector.service_rate_for_one_unit("generic")
    lane_operations_per_cycle = (
        generic_rate / hardware.nominal_sm_freq_hz)
    if (not math.isfinite(lane_operations_per_cycle)
            or lane_operations_per_cycle <= 0):
        raise ValueError(
            "CANN key-1003 search requires positive generic AIV lanes")
    lane_operations_by_worker = tuple(
        lane_operations_per_cycle * (
            comparisons
            * _CANN_MOE_COMPUTE_EXPERT_TOKENS_SEARCH_COMPARISON_AIV_CYCLES
            + invocations
            * _CANN_MOE_COMPUTE_EXPERT_TOKENS_TARGET_SETUP_AIV_CYCLES)
        for comparisons, invocations in zip(
            operations_by_worker, invocations_by_worker))
    return ComputeServiceDemand(
        "generic",
        math.fsum(lane_operations_by_worker),
        operations_by_worker=lane_operations_by_worker,
        serial_invocations_by_worker=(0,) * physical_cores,
    )


def _cann_moe_compute_expert_tokens_source_service(
        kernel: KernelSpec, hardware) -> Optional[ComputeServiceDemand]:
    """Return the closed key-1003 source sequence on its critical AIV.

    The search service above prices the source binary-search loop.  For the
    independently covered E512/one-input-worker domain, the source then runs
    two 256-expert scatter phases, the 2048-byte and 32-byte InitOutput plus
    SyncAll phases, and ProcessAfter, in that strict order.  Their measured
    mechanism spans are converted to equivalent generic-AIV lane work so the
    existing worker schedule serializes them with the search chain without a
    duration residual or a production-shaped lookup.
    """

    search = _cann_moe_compute_expert_tokens_search_service(kernel, hardware)
    if search is None:
        return None
    routed_items = _shape(kernel, "tokens") * _shape(kernel, "topk")
    experts = _shape(kernel, "experts")
    physical_cores = int(
        hardware.resource_by_role("compute_vector").total_units)
    handle_per_core = (
        _ceil_div_positive(
            _ceil_div_positive(routed_items, physical_cores), 320)
        * 320)
    used_cores = min(
        physical_cores, _ceil_div_positive(routed_items, handle_per_core))
    if experts != 512 or used_cores != 1:
        return search

    expert_segments = _ceil_div_positive(experts, 256)
    tail_ns = math.fsum((
        expert_segments
        * _CANN_MOE_COMPUTE_EXPERT_TOKENS_SCATTER_MTE3_NS,
        _CANN_MOE_COMPUTE_EXPERT_TOKENS_FIRST_SYNC_NS,
        _CANN_MOE_COMPUTE_EXPERT_TOKENS_SECOND_SYNC_NS,
        _CANN_MOE_COMPUTE_EXPERT_TOKENS_PROCESS_AFTER_NS,
    ))
    generic_rate = hardware.resource_by_role(
        "compute_vector").service_rate_for_one_unit("generic")
    tail_lane_operations = tail_ns * 1e-9 * generic_rate
    lane_operations_by_worker = list(search.operations_by_worker)
    lane_operations_by_worker[0] += tail_lane_operations
    return dataclasses.replace(
        search,
        operations=math.fsum(lane_operations_by_worker),
        operations_by_worker=tuple(lane_operations_by_worker),
    )


def _cann_moe_compute_expert_tokens_key1001_init_sync_request(
        kernel: KernelSpec, hardware) -> Optional[ComputeExactRequestDemand]:
    """Return the exact key-1001 shared InitOutput-to-SyncAll source phase.

    MoeComputeExpertTokensInt32SS.Init assigns one shared int32 output
    initialization to AIV zero and then synchronizes all 48 AIVs before its
    input copy and scalar/vector body. This is a standalone source phase,
    measured independently for its exact shared-output geometries; it is not
    a complete operator or a duration-derived residual.
    """

    parameters = kernel.launch.parameters
    if int(parameters["cann_tiling_key"]) != 1001:
        return None
    experts = _shape(kernel, "experts")
    if experts not in _CANN_MOE_COMPUTE_EXPERT_TOKENS_KEY1001_INIT_SYNC_EXPERT_COUNTS:
        return None
    vector = hardware.resource_by_role("compute_vector")
    workers = int(vector.total_units)
    if workers != 48:
        raise ValueError(
            "CANN key-1001 InitOutput/SyncAll evidence requires 48 AIVs")
    if (kernel.launch.grid != (workers, 1, 1)
            or kernel.launch.block != (1, 1, 1)):
        raise ValueError(
            "CANN key-1001 InitOutput/SyncAll requires its full-AIV launch")
    exact_services = vector.exact_request_services
    if (exact_services is None
            or _CANN_MOE_COMPUTE_EXPERT_TOKENS_KEY1001_INIT_SYNC_SERVICE
            not in exact_services):
        raise ValueError(
            "CANN key-1001 InitOutput/SyncAll exact service is unavailable")
    published_sizes = exact_services[
        _CANN_MOE_COMPUTE_EXPERT_TOKENS_KEY1001_INIT_SYNC_SERVICE][
            "service_time_s_by_request_size"]
    if experts not in published_sizes:
        raise ValueError(
            "CANN key-1001 InitOutput/SyncAll geometry is not published")
    return ComputeExactRequestDemand(
        _CANN_MOE_COMPUTE_EXPERT_TOKENS_KEY1001_INIT_SYNC_SERVICE,
        ((experts,),) + ((),) * (workers - 1),
    )


def _attach_cann_moe_compute_expert_tokens_key1001_init_sync(
        inp: PredictionInput, request: ComputeExactRequestDemand
        ) -> PredictionInput:
    """Serialize the exact source phase before the remaining generic body."""

    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN key-1001 InitOutput/SyncAll requires one vector demand")
    vector_demand = vector_demands[0]
    if vector_demand.service_demands or vector_demand.exact_request_demands:
        raise ValueError(
            "CANN key-1001 InitOutput/SyncAll conflicts with another "
            "vector source schedule")
    generic_operations = vector_demand.operations - request.operations
    if generic_operations <= 0:
        raise ValueError(
            "CANN key-1001 InitOutput/SyncAll must leave positive generic "
            "Vector work")
    typed_demand = dataclasses.replace(
        vector_demand,
        service_demands=(ComputeServiceDemand("generic", generic_operations),),
        exact_request_demands=(request,),
    )
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            typed_demand if demand is vector_demand else demand
            for demand in inp.compute_demands),
    )


def _metax_moe_sum_source_service(
        kernel: KernelSpec, hardware) -> Optional[str]:
    """Return the typed TOK=1 MACA imcontinuous-reduction program."""

    mechanism = hardware.imcontinuous_vec4_reduction_source_service
    if mechanism is None:
        return None
    if kernel.variant.lower() != "moe_sum" or kernel.dtype != "bf16":
        raise ValueError(
            "C500 imcontinuous reduction service requires its frozen BF16 "
            "source launch")
    if kernel.launch.algorithm not in {
            "pytorch_2_4_metax_bf16_sum_imcontinuous_reduce",
            "observed_moe_aux:moe_sum",
    }:
        return None
    tokens = _shape(kernel, "tokens")
    if tokens != 1:
        # The 3-D TensorIterator path maps all 512 threads to independent
        # outputs and does not execute the shared block-y reduction measured
        # by this mechanism.
        return None
    topk = _shape(kernel, "topk")
    programs = mechanism["programs"]
    if topk not in programs:
        raise NotImplementedError(
            f"C500 TOK=1 moe_sum has no source service for top-k {topk}")
    program = programs[topk]
    block_x = program["block_x"]
    block_y = program["block_y"]
    if (block_x * block_y != mechanism["threads_per_cta"]
            or kernel.launch.block_size != mechanism["threads_per_cta"]):
        raise ValueError(
            "C500 imcontinuous reduction block geometry changed")
    hidden = _shape(kernel, "hidden")
    output_vector_width = 4
    expected_grid = _ceil_div_positive(
        hidden, output_vector_width * block_x)
    if kernel.launch.grid != (expected_grid, 1, 1):
        raise ValueError(
            "C500 imcontinuous reduction grid differs from source mapping")
    reduction = hardware.resource_by_role("compute_reduction")
    service = program["service"]
    if not reduction.has_typed_service(service):
        raise ValueError(
            f"C500 compute_reduction lacks source service {service!r}")
    return service


def _moe_aux_model(kernel: KernelSpec, hardware) -> Tuple[PredictionInput, float]:
    variant = kernel.variant.lower()
    _validate_moe_aux_execution(variant, hardware)
    tokens = _shape(kernel, "tokens")
    experts = _shape(kernel, "experts")
    topk = _shape(kernel, "topk")
    hidden = _shape(kernel, "hidden")
    intermediate = _shape(kernel, "intermediate")
    source_read_bytes = None
    source_write_bytes = None
    phase_blueprint = None
    expert_tokens_source_service = None
    expert_tokens_key1001_init_sync_request = None
    moe_sum_source_service = None
    if variant == "moe_swiglu":
        output_elements = int(kernel.attributes["routed_rows"]) * intermediate
        elements = output_elements
        source_read_bytes = float(2 * output_elements * 2)
        source_write_bytes = float(output_elements * 2)
        total = source_read_bytes + source_write_bytes
        # CANN FusedMoe calls the same BF16
        # SwiGlu_3_high_performance_27 program as standalone SwiGLU.  Keep
        # its source arithmetic in the common roles; the three cast passes
        # are supplied by the launch contract's lowered_vector_operations.
        flops = {"compute_tensor": 0, "compute_fma": 3.0 * elements,
                 "compute_alu": 0, "compute_sfu": 2.0 * elements}
        phase_blueprint = _target_phase_blueprint(
            hardware,
            _linear_phase_blueprint(
                l2_read_bytes=source_read_bytes,
                dram_read_bytes=source_read_bytes,
                write_bytes=source_write_bytes,
                compute_demands=_phase_compute_demands(
                    ("compute_fma", "activation_arithmetic",
                     3.0 * elements),
                    ("compute_sfu", "exp_and_reciprocal",
                     2.0 * elements),
                ),
            ),
        )
    elif variant in ("moe_init_routing", "moe_init_routing_v2",
                     "moe_init_routing_v3"):
        # Public CANN ABI.  Retain the historical standalone V1-like helper
        # separately, while the public FusedMoE call uses V2/V3.  Both public
        # variants consume hidden[T,H] BF16 plus expert_ids[T,K] int32.
        # V2 emits int32 expert metadata; V3 emits int64 expert metadata and
        # a float expansion scale.  These bytes come directly from the public
        # tensor signatures, not a duration-derived correction.
        routed_items = tokens * topk
        source_read_bytes = float(tokens * hidden * 2 + routed_items * 4)
        if variant == "moe_init_routing_v2":
            source_write_bytes = float(
                routed_items * hidden * 2 + routed_items * 4 + experts * 8)
        elif variant == "moe_init_routing_v3":
            source_write_bytes = float(
                routed_items * hidden * 2 + routed_items * 8 + experts * 8)
        else:
            # The historical standalone helper owns two int32 metadata
            # streams.  Keep its existing contract intact for diagnostics.
            source_read_bytes = float(
                tokens * hidden * 2 + 2 * routed_items * 4)
            source_write_bytes = float(
                routed_items * hidden * 2 + 2 * routed_items * 4)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(routed_items * hidden), "compute_sfu": 0}
    elif variant == "moe_gating_topk":
        # MoeGatingTopK input: float[T,E].  Outputs are routed weights
        # float[T,K], expert ids int32[T,K], and the full float[T,E] router
        # output.  The source keeps these public outputs materialized.
        routed_items = tokens * topk
        source_read_bytes = float(4 * tokens * experts)
        source_write_bytes = float(
            4 * tokens * experts + 8 * routed_items)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0, "compute_fma": float(tokens * experts),
                 "compute_alu": 0, "compute_sfu": 0}
    elif variant == "moe_abs":
        # Abs<int32> transforms only the T*K routing-row metadata.
        elements = tokens * topk
        source_read_bytes = float(4 * elements)
        source_write_bytes = float(4 * elements)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(elements), "compute_sfu": 0}
    elif variant == "moe_token_unpermute":
        # Every output token accumulates K routed BF16 rows using FP32 route
        # probabilities, then writes one BF16 output row.  Source reads the
        # permuted rows, int32 sorted indices, and float probabilities.
        routed_items = tokens * topk
        source_read_bytes = float(
            2 * routed_items * hidden + 4 * routed_items + 4 * routed_items)
        source_write_bytes = float(2 * tokens * hidden)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(3 * routed_items * hidden),
                 "compute_sfu": 0}
    elif variant == "moe_finalize_routing":
        # CANN 8.5.1 key-20036 MoeFinalizeRoutingV2 BF16 no-bias ABI:
        #   inputs:  down[T*K,H] bf16, scales[T,K] bf16,
        #            expanded_src_to_dst_row[T*K] int32
        #   output:  out[T,H] bf16
        # The conditional expertForSourceRow/export_for_source_row input is
        # not read by this no-bias source specialization.  Per output value,
        # its K routed values each cast BF16 to FP32 and multiply by a scale;
        # K - 1 additions reduce them, then the final FP32 value casts to
        # BF16.  That is exactly 3 * T * K * H Vector operations, independent
        # of duration or profiler counters.
        parameters = kernel.launch.parameters
        if "cann_tiling_key" not in parameters:
            raise ValueError(
                "CANN MoeFinalizeRoutingV2 requires its source tiling key")
        if (int(parameters["cann_tiling_key"])
                != _CANN_MOE_FINALIZE_ROUTING_BF16_NO_BIAS_TILING_KEY):
            raise ValueError(
                "CANN MoeFinalizeRoutingV2 requires BF16 no-bias key 20036")
        routed_items = tokens * topk
        source_read_bytes = float(
            routed_items * hidden * 2 + routed_items * 2
            + routed_items * 4)
        source_write_bytes = float(tokens * hidden * 2)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(3 * tokens * topk * hidden),
                 "compute_sfu": 0}
    elif variant == "moe_compute_expert_tokens":
        source_read_bytes, source_write_bytes = (
            _cann_moe_compute_expert_tokens_traffic(kernel, hardware))
        total = source_read_bytes + source_write_bytes
        expert_tokens_source_service = (
            _cann_moe_compute_expert_tokens_key1001_compute_service(
                kernel, hardware))
        if expert_tokens_source_service is None:
            expert_tokens_source_service = (
                _cann_moe_compute_expert_tokens_source_service(
                    kernel, hardware))
        expert_tokens_key1001_init_sync_request = (
            _cann_moe_compute_expert_tokens_key1001_init_sync_request(
                kernel, hardware))
        compute_work = (
            expert_tokens_source_service.operations
            if expert_tokens_source_service is not None
            else float(tokens * topk + experts))
        if expert_tokens_key1001_init_sync_request is not None:
            # The exact InitOutput request replaces its logical output-size
            # work when it is attached below.  Keep that size in the broad
            # demand so _attach_* can remove it without touching the source
            # Compute service above.
            compute_work += float(experts)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": compute_work, "compute_sfu": 0}
    elif variant == "moe_transpose":
        total = float(tokens * topk * 4 * 2)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(tokens * topk), "compute_sfu": 0}
    elif variant == "moe_cast":
        cast_role = kernel.attributes["cast_role"]
        if cast_role == "topk_ids":
            elements = tokens * topk
        elif cast_role == "expert_tokens":
            elements = experts
        else:
            raise ValueError(
                f"unknown VECTOR/moe_cast cast_role: {cast_role!r}")
        source_read_bytes = float(elements * 4)
        source_write_bytes = float(elements * 4)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(elements), "compute_sfu": 0}
    elif variant == "moe_range":
        total = float((experts + tokens * topk) * 4 * 2)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(tokens * topk + experts), "compute_sfu": 0}
    elif variant in ("moe_topk", "moe_softmax", "moe_gating"):
        total = float(tokens * experts * 2 * 2)
        flops = {"compute_tensor": 0, "compute_fma": float(tokens * experts),
                 "compute_alu": 0, "compute_sfu": 0}
    elif variant == "moe_act_and_mul":
        # vLLM's fused activation consumes the two BF16 halves of the W1
        # projection and materializes one BF16 output half.  Each output
        # element evaluates x / (1 + exp(-x)) * y: three ordinary arithmetic
        # operations plus exp and reciprocal/division.  Vector packing changes
        # the instruction representation, not the semantic element count.
        output_elements = tokens * topk * intermediate
        source_read_bytes = float(2 * output_elements * 2)
        source_write_bytes = float(output_elements * 2)
        total = source_read_bytes + source_write_bytes
        uses_scalar_round_service = (
            _has_compute_service(
                hardware, "compute_vector", _SILU_SCALAR_ROUND_SERVICE)
            and kernel.launch.grid_size == tokens * topk
            and kernel.launch.block_size == 1024)
        if uses_scalar_round_service:
            scalar_rounds = _silu_scalar_rounds(kernel)
            flops = {
                "compute_tensor": 0,
                "compute_vector": scalar_rounds,
                "compute_fma": float(output_elements),
                "compute_alu": 0,
                "compute_sfu": 0,
            }
        else:
            flops = {
                "compute_tensor": 0,
                "compute_fma": 3.0 * output_elements,
                "compute_alu": 0,
                "compute_sfu": 2.0 * output_elements,
            }
    elif variant == "moe_silu":
        # The MetaX standalone serving path lowers
        # ``silu(gate) * up`` into two native kernels.  This first kernel reads
        # and writes the gate tensor once: 2 BF16 transfers per element.
        work_fraction = kernel.attributes["work_fraction"]
        elements = tokens * topk * intermediate * work_fraction
        source_read_bytes = float(elements * 2)
        source_write_bytes = float(elements * 2)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0, "compute_fma": 2.0 * elements,
                 "compute_alu": 0, "compute_sfu": float(elements)}
    elif variant == "moe_mul":
        # The second native kernel reads the SiLU result and ``up`` tensor and
        # writes their product: 3 BF16 transfers per element.
        work_fraction = kernel.attributes["work_fraction"]
        elements = tokens * topk * intermediate * work_fraction
        source_read_bytes = float(2 * elements * 2)
        source_write_bytes = float(elements * 2)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0, "compute_fma": float(elements),
                 "compute_alu": 0, "compute_sfu": 0}
    elif variant == "moe_sum":
        # Reduce the w2 output [TOK, TOPK, H] to [TOK, H].  Read every routed
        # value and write one result per output element.  Some backends split
        # one logical TensorIterator reduction into multiple physical
        # invocations; work_fraction is the explicit per-invocation contract.
        work_fraction = kernel.attributes["work_fraction"]
        output_elements = tokens * hidden
        input_elements = output_elements * topk
        source_read_bytes = float(input_elements * 2 * work_fraction)
        source_write_bytes = float(output_elements * 2 * work_fraction)
        total = source_read_bytes + source_write_bytes
        moe_sum_source_service = _metax_moe_sum_source_service(
            kernel, hardware)
        flops = {"compute_tensor": 0,
                 "compute_reduction": (
                     float(kernel.launch.grid_size)
                     if moe_sum_source_service is not None else 0.0),
                 "compute_fma": float(
                     output_elements * (topk - 1) * work_fraction),
                 "compute_alu": 0, "compute_sfu": 0}
    elif variant.startswith("moe_align"):
        # The alignment kernel scans every routed token id once, then visits
        # every expert counter to round the per-expert offsets to block
        # boundaries.  Count those source-level loop iterations directly;
        # do not infer an ALU multiplier from profiler pipe counters.
        routed_items = tokens * topk
        total = float(experts * 4 * 2 + tokens * topk * 4 * 2 * 2)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(routed_items + experts),
                 "compute_sfu": 0}
    elif variant == "moe_routing_weight_norm":
        # The captured topkGatingSoftmax does not fuse renormalisation, so
        # vLLM divides the routed weights by their own row sum.  This is the
        # sum: it reads the [tokens, topk] FP32 routing weights once and
        # writes one FP32 partial per token.
        total = float(tokens * topk * 4 + tokens * 4)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(tokens * max(topk - 1, 1)),
                 "compute_sfu": 0}
    elif variant == "moe_routing_weight_scale":
        # The division that follows the row sum: every routed weight is
        # rescaled by its token's total.  One FP32 read of the weights, one
        # FP32 read of the per-token sum, one FP32 write.
        routed = tokens * topk
        total = float(routed * 4 + tokens * 4 + routed * 4)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(routed), "compute_sfu": 0}
    elif variant == "moe_routed_id_extent":
        # torch's bincount establishes its output range by reducing the
        # routed id tensor for its minimum and then its maximum before
        # launching the histogram.  Each pass reads the ids once and writes
        # a single scalar.
        routed = tokens * topk
        total = float(routed * 8 + 8)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(routed), "compute_sfu": 0}
    elif variant == "moe_expert_map_lookup":
        # Expert-parallel translation of the alignment kernel's global expert
        # ids into this rank's local range, one gathered int32 per padded
        # block.  Reads the block's expert id and the map entry it selects,
        # writes the translated id.
        block_m = 16 if tokens <= experts else 64
        padded = tokens * topk + experts * (block_m - 1)
        blocks = -(-padded // block_m)
        total = float(blocks * 4 * 3)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(blocks), "compute_sfu": 0}
    elif variant == "moe_router_logit_cast":
        # The router projection emits BF16 logits, but the captured
        # topkGatingSoftmax takes `float const*`, so the logits are widened
        # to FP32 before routing.  One BF16 read and one FP32 write per
        # logit.
        logits = tokens * experts
        total = float(logits * 2 + logits * 4)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(logits), "compute_sfu": 0}
    elif variant == "moe_routed_id_widen":
        # The captured routing kernel emits int32 ids, and the bincount that
        # follows reduces int64, so the ids are widened in between.  One
        # int32 read and one int64 write per routed id.
        routed = tokens * topk
        total = float(routed * 4 + routed * 8)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(routed), "compute_sfu": 0}
    elif variant == "moe_count_and_sort":
        # One classification/counting step per routed token id.  The previous
        # 0.85 multiplier was an NCU-derived median and is intentionally not a
        # production Forward parameter.
        routed_items = tokens * topk
        total = float(experts * 4 * 2 + tokens * topk * 4 * 2 * 2)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(routed_items), "compute_sfu": 0}
    elif variant == "moe_reduce":
        # PyTorch's sum reduction consumes [TOK, TOPK, H] and materializes
        # [TOK, H].  PyTorch 2.8 may split one logical TensorIterator into
        # physical TensorIterator output slices to preserve signed 32-bit
        # indexing (whole-token slices for top-k > 1, flat slices for top-k 1);
        # work_fraction is derived from that source algorithm, not duration.
        output_elements = tokens * hidden
        input_elements = output_elements * topk
        work_fraction = kernel.attributes["work_fraction"]
        source_read_bytes = float(input_elements * 2 * work_fraction)
        source_write_bytes = float(output_elements * 2 * work_fraction)
        total = source_read_bytes + source_write_bytes
        flops = {"compute_tensor": 0,
                 "compute_fma": float(
                     output_elements * (topk - 1) * work_fraction),
                 "compute_alu": 0,
                 "compute_sfu": 0}
    elif variant == "moe_fill":
        block_m = 16 if tokens <= experts else 64
        max_padded = tokens * topk + experts * (block_m - 1)
        fill_role = kernel.attributes["fill_role"]
        if fill_role == "sorted_token_ids":
            elements = max_padded
        elif fill_role == "expert_ids":
            elements = (max_padded + block_m - 1) // block_m
        elif fill_role == "num_tokens_post_padded":
            elements = 1
        else:
            raise ValueError(
                f"unknown VECTOR/moe_fill fill_role: {fill_role!r}")
        # torch.full/zeros materialize one int32 output and consume no input
        # tensor.  FillFunctor's scalar assignment is not logical tensor ALU.
        total = float(elements * 4)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": 0, "compute_sfu": 0}
    elif variant == "moe_enable_if":
        total = float(experts * 4 * 2 + tokens * topk * 4 * 2 * 2)
        flops = {"compute_tensor": 0, "compute_fma": 0,
                 "compute_alu": float(tokens * topk), "compute_sfu": 0}
    else:
        raise ValueError(f"unknown MoE auxiliary VECTOR variant: {kernel.variant!r}")
    # Keep unresolved native helpers aggregate-read-only, but preserve the
    # direction of helpers whose public tensor ABI fixes input and output
    # ownership.  This split is source semantics, not a duration-derived
    # overlap or bandwidth correction.
    if (source_read_bytes is None) != (source_write_bytes is None):
        raise AssertionError(
            "MoE auxiliary directional traffic must define both directions")
    if source_read_bytes is None:
        l2_read_bytes = total
        dram_read_bytes = total
        write_bytes = 0.0
    else:
        if not math.isclose(
                source_read_bytes + source_write_bytes, total,
                rel_tol=0.0, abs_tol=0.0):
            raise AssertionError(
                "MoE auxiliary directional traffic does not conserve bytes")
        l2_read_bytes = source_read_bytes
        dram_read_bytes = source_read_bytes
        write_bytes = source_write_bytes
    if variant == "moe_fill":
        l2_read_bytes = 0.0
        dram_read_bytes = 0.0
        write_bytes = total
    demand = _vector_demand(
        hardware, l2_read_bytes=l2_read_bytes,
        dram_read_bytes=dram_read_bytes,
        write_bytes=write_bytes, flops=flops)
    expert_tokens_limitations = (
        "moe_compute_expert_tokens_public_source_binary_binding_evidence_pending",
    )
    has_closed_expert_tokens_tail = (
        variant == "moe_compute_expert_tokens"
        and int(kernel.launch.parameters["cann_tiling_key"]) == 1003
        and experts == 512
        and tokens * topk <= 320
    )
    if not has_closed_expert_tokens_tail:
        if expert_tokens_key1001_init_sync_request is None:
            expert_tokens_limitations += (
                "moe_compute_expert_tokens_syncall_lowering_evidence_pending",
            )
        expert_tokens_limitations += (
            "moe_compute_expert_tokens_write_transaction_evidence_pending",
        )
    if (expert_tokens_source_service is None
            or (variant == "moe_compute_expert_tokens"
                and int(kernel.launch.parameters["cann_tiling_key"]) == 1001)):
        expert_tokens_limitations += (
            "moe_compute_expert_tokens_scalar_control_work_evidence_pending",
        )
    limitation_extensions = {
        "moe_init_routing": (
            "moe_init_routing_internal_read_traffic_evidence_pending",
        ),
        "moe_init_routing_v2": (
            "moe_init_routing_internal_read_traffic_evidence_pending",
        ),
        "moe_init_routing_v3": (
            "moe_init_routing_internal_read_traffic_evidence_pending",
        ),
        "moe_compute_expert_tokens": expert_tokens_limitations,
    }
    additional_limitations = (
        limitation_extensions[variant]
        if variant in limitation_extensions else ())
    if (hardware.kernel_memory_model == "npu_tile"
          and variant in (_CANN_MOE_AUX_VARIANTS - {"moe_swiglu"})):
        resource_input = _cann_moe_aux_aggregate_input(
            kernel,
            hardware,
            demand,
            additional_limitations=additional_limitations,
        )
    else:
        resource_input = _input(
            kernel, hardware, demand, write_bytes=write_bytes,
            phase_blueprint=phase_blueprint,
            operator_aux=True)
    if expert_tokens_key1001_init_sync_request is not None:
        resource_input = _attach_cann_moe_compute_expert_tokens_key1001_init_sync(
            resource_input, expert_tokens_key1001_init_sync_request)
        resource_input = _attach_linear_vector_dependency_path(
            resource_input,
            hardware,
            name="cann_moe_compute_expert_tokens_key1001_init_sync",
            compute_roles=("compute_vector",),
        )
        if resource_input.launch_task_kind == "aiv":
            resource_input = dataclasses.replace(
                resource_input,
                task_duration_floor_contract=(
                    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM),
            )
    if expert_tokens_source_service is not None:
        resource_input = _replace_vector_service_demands(
            resource_input, (expert_tokens_source_service,))
    if moe_sum_source_service is not None:
        resource_input = _replace_compute_service_demands(
            resource_input,
            "compute_reduction",
            (ComputeServiceDemand(
                moe_sum_source_service,
                float(kernel.launch.grid_size)),),
        )
        resource_input = _attach_linear_vector_dependency_path(
            resource_input,
            hardware,
            name="metax_imcontinuous_vec4_reduction",
            compute_roles=("compute_reduction",),
            include_memory_services=False,
            support_level="partial",
        )
    if (variant == "moe_act_and_mul"
            and uses_scalar_round_service):
        resource_input = _attach_silu_semantic_dependency_path(
            resource_input, kernel, hardware)
    atomic_service = _moe_routing_global_atomic_service(kernel, hardware)
    if atomic_service is not None:
        resource_input = _replace_compute_service_demands(
            resource_input,
            "compute_alu",
            (ComputeServiceDemand(
                atomic_service,
                float(routed_items)),),
        )
    if (additional_limitations
            and hardware.kernel_memory_model != "npu_tile"):
        resource_input = dataclasses.replace(
            resource_input,
            limitations=(
                resource_input.limitations + additional_limitations),
        )
    return resource_input, total


def _model(kernel: KernelSpec, hardware):
    kernel = _gpu_full_graph_source_program_execution_kernel(
        kernel, hardware)
    kernel = _metax_full_graph_norm_execution_kernel(kernel, hardware)
    kernel = _cann_full_graph_source_program_execution_kernel(
        kernel, hardware)
    if kernel.variant.startswith("moe_"):
        return _moe_aux_model(kernel, hardware)
    elif kernel.variant in {
        "rmsnorm", "add_rmsnorm", "layernorm", "mla_rope", "elementwise",
        "elementwise_pytorch_vec4", "rotary", "rope",
        "qkv_rmsnorm_rope",
        "activation", "silu",
        "reduce", "gather", "softmax",
        "gemm_reduce", "gemm_elementwise", "flash_mla_metadata",
        "fp8_group_quant",
    }:
        return _regular_model(kernel, hardware)
    else:
        raise ValueError(f"unknown VECTOR variant: {kernel.variant!r}")


def paged_kv_cache_resource_input(kernel: KernelSpec, hardware):
    """Reuse the established paged-cache physical model under its own kind."""

    if (kernel.kind is not KernelKind.KV_CACHE
            or kernel.variant != "paged_kv_cache"):
        raise ValueError(
            "paged KV cache resource input requires "
            "KV_CACHE/paged_kv_cache")
    return _regular_model(kernel, hardware)


def forward_analytical_resource_input(kernel: KernelSpec, hardware):
    """Build Forward's final CANN analytical envelope.

    The public resource-input route intentionally keeps the source-DAG form
    available for contract validation.  This helper evaluates the same static
    workload with the production Forward representation: phase ledger,
    physical services, analytical dependency envelope, and launch floor.
    """

    token = _FORWARD_CANN_ANALYTICAL_ENVELOPE.set(True)
    try:
        if (kernel.kind is KernelKind.KV_CACHE
                and kernel.variant == "paged_kv_cache"):
            return _regular_model(kernel, hardware)
        return _model(kernel, hardware)
    finally:
        _FORWARD_CANN_ANALYTICAL_ENVELOPE.reset(token)


class VectorPredictor(KernelPredictor):
    kind = KernelKind.VECTOR

    def __init__(self):
        self._cached_forward_kernel = None
        self._cached_forward_hardware = None
        self._cached_forward_value = None

    def resource_input(self, kernel: KernelSpec, hardware):
        return _model(kernel, hardware)

    def forward_resource_input(self, kernel: KernelSpec, hardware):
        return self._forward_resource_input(kernel, hardware)

    def _forward_resource_input(self, kernel: KernelSpec, hardware):
        """Return this exact spec's compact analytical input for one caller."""

        if (self._cached_forward_kernel is kernel
                and self._cached_forward_hardware is hardware):
            if self._cached_forward_value is None:
                raise RuntimeError("Vector forward resource-input cache is incomplete")
            cached_input, cached_compulsory = self._cached_forward_value
            return dataclasses.replace(
                cached_input,
                demand_per_resource=dict(cached_input.demand_per_resource),
                resource_time_overrides_ns=dict(
                    cached_input.resource_time_overrides_ns),
            ), cached_compulsory
        resource_input, compulsory = forward_analytical_resource_input(
            kernel, hardware)
        self._cached_forward_kernel = kernel
        self._cached_forward_hardware = hardware
        self._cached_forward_value = (resource_input, compulsory)
        return resource_input, compulsory

    def oracle_resource_input(self, kernel: KernelSpec, hardware):
        # Keep resource_input's expanded source program for contract callers.
        # Oracle only consumes its normalized resource/dependency envelope, so
        # CANN Vector can use the same exact worker-class representation as
        # Forward instead of materializing one identical tile program per
        # physical AIV.
        if (kernel.variant.lower() in {
                    "elementwise", "rmsnorm", "add_rmsnorm"}
                and hardware.kernel_memory_model == "npu_tile"
                and hardware.aic_aiv_separated
                and "cann_tiling_key" in kernel.launch.parameters):
            return self._forward_resource_input(kernel, hardware)
        return self.resource_input(kernel, hardware)

    def midlayer(self, kernel: KernelSpec, hardware):
        # The vLLM-Ascend QKNormRope fusion pass rewrites this pattern at
        # torchair compile time, and capture-limit fallback (NONE) steps
        # still execute the codegen'd fused fx graph: msprof serving traces
        # show one split_qkv_rmsnorm_rope_kernel in both lanes and never a
        # standalone [seq*heads, head_dim] RmsNorm sequence.  Both lanes
        # therefore price through the single fused streaming envelope.
        inp, _ = self._forward_resource_input(kernel, hardware)
        prediction = PerformanceFormula(hardware).predict_kernel(
            _with_graph_regime(inp, kernel))
        return _append_cann_public_rope_index_selects(
            prediction, kernel, hardware)

    def roofline(self, kernel: KernelSpec, hardware):
        inp, compulsory = self._forward_resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory), "roofline")

    def pipeweave(self, kernel: KernelSpec, hardware, *, model_hardware=None):
        variant = kernel.variant.lower()
        if variant in ("rmsnorm", "add_rmsnorm"):
            from src.kernel.predictors.pipeweave_adapter import predict_norm
            return predict_norm(
                kernel, hardware, model_hardware=model_hardware)
        # PipeWeave's siluandmul model covers the SiLU-and-multiply operator
        # itself, so both the dense launch and the fused-MoE launch of that
        # same operator belong in its domain.  Routing, align, sort and reduce
        # launches have no PipeWeave model and fall through to the base class.
        if variant not in (
                "activation", "silu", "moe_act_and_mul", "moe_swiglu"):
            return super().pipeweave(
                kernel, hardware, model_hardware=model_hardware)
        from src.kernel.predictors.pipeweave_adapter import predict_vector_activation
        return predict_vector_activation(
            kernel, hardware, model_hardware=model_hardware)
