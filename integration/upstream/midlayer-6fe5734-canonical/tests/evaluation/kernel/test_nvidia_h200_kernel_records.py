"""Contract tests for H200 NSYS launch-only kernel records in evaluation."""

import pytest

from evaluation.kernel.dataset import get_hardware_config_path
from evaluation.kernel.evaluation_runner import (
    _process_op,
    _validate_case_executor_args,
)
from evaluation.kernel.observation_normalizer import (
    observed_flash_attention_splits,
    observed_gemm_split_k,
    observed_gemm_split_k_partial_bytes,
    observed_nvjet_gemm_split_k,
)
from evaluation.kernel.profiler_observation import (
    extract_ncu_metrics,
    require_hardware_kernel_record_kind,
)
from src.kernel.launch_config.tiling_spec import (
    parse_complete_gemm_tile,
    parse_nvjet_gemm_symbol,
)


_RMSNORM = ("void vllm::rms_norm_kernel<c10::BFloat16, (int)8, (int)2, "
            "(bool)1>(T1 *, const T1 *, long, long, long, long, long, "
            "const T1 *, long, float, int, int)")
_FA3_MAIN = ("void cutlass::device_kernel<flash::enable_sm90_or_later<"
             "flash::FlashAttnFwdSm90<...>>>(T1::Params)")
_FA3_COMBINE = ("void cutlass::device_kernel<flash::FlashAttnFwdCombine<"
                "...>>(T1::Params)")


def _record(name, grid=(1, 1, 1), block=(256, 1, 1)):
    grid_total = grid[0] * grid[1] * grid[2]
    block_total = block[0] * block[1] * block[2]
    return {
        "Metadata": {
            "Block Size": f"({block[0]}, {block[1]}, {block[2]})",
            "CC": "9.0",
            "Grid Size": f"({grid[0]}, {grid[1]}, {grid[2]})",
            "Kernel Name": name,
            "Source": "nsys_cupti_kernel_activity",
        },
        "Sections": {"Launch Statistics": {"Header": {
            "Block Size": str(block_total),
            "Dynamic Shared Memory Per Block [byte/block]": "0",
            "Grid Size": str(grid_total),
            "Registers Per Thread [register/thread]": "30",
            "Shared Memory Configuration Size [byte]": "32768",
            "Static Shared Memory Per Block [byte/block]": "168",
            "Threads [thread]": str(grid_total * block_total),
        }}},
    }


def test_nsys_launch_only_record_exposes_launch_facts_but_no_counters():
    metrics = extract_ncu_metrics(_record(_RMSNORM, grid=(4, 2, 1)))
    assert metrics["record_kind"] == "nsys_launch_only"
    assert (metrics["grid_x"], metrics["grid_y"], metrics["grid_size"]) == (
        4, 2, 8)
    assert metrics["regs_per_thread"] == 30
    for counter in ("duration_ns", "dram_bytes", "l2_total", "tensor_pct"):
        assert counter not in metrics
    require_hardware_kernel_record_kind(metrics, "nvidia_h200")
    with pytest.raises(ValueError, match="ncu_sections"):
        require_hardware_kernel_record_kind(metrics, "nvidia_a800")
    bad = _record(_RMSNORM)
    bad["Sections"]["Occupancy"] = {"Header": {}}
    with pytest.raises(ValueError, match="only Launch Statistics"):
        extract_ncu_metrics(bad)
    bad = _record(_RMSNORM)
    bad["Metadata"]["Source"] = "ncu"
    with pytest.raises(ValueError, match="unknown profiler record source"):
        extract_ncu_metrics(bad)


def test_h200_is_a_kernel_evaluation_hardware():
    _validate_case_executor_args([], 1, "nvidia_h200")


def test_h200_vector_case_runs_forward_and_refuses_oracle():
    entry = {
        "op_info": {
            "Type": "NORM",
            "Attr": {"op": "rmsnorm", "op2": "add", "tok": 1,
                     "hidden": 2048, "num_heads": 32, "head_dim": 128,
                     "intermediate": 6144, "num_kv_heads": 4, "batch": 1,
                     "element_count": 2048, "dtype_bytes": 2},
            "evaluation": {"timing": {
                "granularity": "kernel", "source": "nsys_nvtx_kernel_median",
                "total_duration_ns": 1888.0,
                "kernels": [{"kernel_idx": 0, "kernel_name": _RMSNORM,
                             "duration_ns": 1888}]}},
        },
        "kernels": [_record(_RMSNORM)],
    }
    result = _process_op(
        "H200_RMSNORM", entry, get_hardware_config_path("nvidia_h200"),
        True, True)
    assert result[2] == "ERR"
    prediction_ns, actual_ns = result[3]
    assert prediction_ns > 0 and actual_ns == 1888.0


def _fa_attributes(sp, per_request, combine):
    return {"SP": sp, "B": len(per_request), "HKV": 8, "Q_LEN": 1,
            "kv_layout": "paged", "SP_PER_REQUEST": per_request,
            "FA3_SPLIT_COMBINE": combine}


def test_h200_fa3_split_comes_from_scheduler_metadata_attributes():
    split = [_record(_FA3_MAIN, grid=(132, 1, 1)),
             _record(_FA3_COMBINE, grid=(640, 1, 1))]
    assert observed_flash_attention_splits(
        _fa_attributes(3, [3, 1], True), split, "nvidia_h200") == 3
    # Static bound > 1 launches the combine even when every request runs 1.
    assert observed_flash_attention_splits(
        _fa_attributes(1, [1, 1], True), split, "nvidia_h200") == 1
    with pytest.raises(ValueError, match="combine launch disagrees"):
        observed_flash_attention_splits(
            _fa_attributes(1, [1, 1], False), split, "nvidia_h200")
    with pytest.raises(ValueError, match="largest per-request split"):
        observed_flash_attention_splits(
            _fa_attributes(2, [3, 1], True), split, "nvidia_h200")


def test_h200_fa3_split_check_accepts_mla_verify_queries():
    split = [_record(_FA3_MAIN, grid=(132, 1, 1)),
             _record(_FA3_COMBINE, grid=(640, 1, 1))]
    attributes = {"SP": 3, "B": 2, "HKV": 1, "Q_LEN": 3, "kv_layout": "mla",
                  "SP_PER_REQUEST": [3, 2], "FA3_SPLIT_COMBINE": True}
    assert observed_flash_attention_splits(
        attributes, split, "nvidia_h200") == 3


_NVJET_SPLIT = "nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT"
_SPLIT_REDUCE = (
    "void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, "
    "__nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, "
    "__nv_bfloat16, (bool)1, (bool)0, (bool)0, (bool)0>"
    "(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, "
    "T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, "
    "long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, "
    "const T6 *, const T6 *)")


def test_nvjet_symbol_states_tile_stages_cluster_and_split():
    symbol = parse_nvjet_gemm_symbol(
        "nvjet_hsh_256x128_64x4_1x2_h_bz_coopA_splitK_TNT")
    # The cuBLAS tile is (output features, tokens) for a PyTorch linear.
    assert (symbol.tile_m, symbol.tile_n, symbol.tile_k) == (128, 256, 64)
    assert symbol.pipeline_depth == 4
    assert (symbol.cluster_n_tiles, symbol.cluster_m_tiles) == (1, 2)
    assert symbol.cooperative == "coopA" and symbol.split_k
    assert parse_complete_gemm_tile(_NVJET_SPLIT) == (8, 64, 64, 16)
    assert parse_nvjet_gemm_symbol("ampere_bf16_s16816gemm") is None
    with pytest.raises(ValueError, match="unrecognized cuBLASLt nvjet"):
        parse_nvjet_gemm_symbol("nvjet_tst_64x8_bogus")


def test_nvjet_split_is_the_cluster_padded_tile_multiplicity():
    attributes = {"B": 1, "M": 1, "N": 2624, "K": 6144, "dtype": "bf16"}
    metrics = extract_ncu_metrics(_record(
        "nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT", grid=(2, 63, 1),
        block=(384, 1, 1)))
    # 41 N tiles pad to 42 under the 2x1 cluster: 126 CTAs = 3 K slices.
    assert observed_nvjet_gemm_split_k(attributes, metrics) == 3
    persistent = extract_ncu_metrics(_record(
        "nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN", grid=(2, 66, 1)))
    assert observed_nvjet_gemm_split_k(
        {"B": 1, "M": 1536, "N": 6144, "K": 256}, persistent) == 1
    ragged = extract_ncu_metrics(_record(
        "nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT", grid=(2, 61, 1)))
    with pytest.raises(NotImplementedError, match="integer multiple"):
        observed_nvjet_gemm_split_k(attributes, ragged)


def test_h200_nvjet_split_k_gemm_case_runs_forward():
    kernels = [_record(_NVJET_SPLIT, grid=(4, 5, 1), block=(384, 1, 1)),
               _record(_SPLIT_REDUCE, grid=(1, 7, 1), block=(64, 1, 1))]
    attributes = {"B": 1, "M": 7, "K": 7168, "N": 256, "dtype": "bf16"}
    assert observed_gemm_split_k(kernels, attributes) == 5
    assert observed_gemm_split_k_partial_bytes(kernels, attributes) == 4
    entry = {
        "op_info": {
            "Type": "GEMM", "Attr": attributes,
            "evaluation": {"timing": {
                "granularity": "kernel", "source": "nsys_nvtx_kernel_median",
                "total_duration_ns": 6400.0,
                "kernels": [
                    {"kernel_idx": 0, "kernel_name": _NVJET_SPLIT,
                     "duration_ns": 4864},
                    {"kernel_idx": 1, "kernel_name": _SPLIT_REDUCE,
                     "duration_ns": 1536}]}},
        },
        "kernels": kernels,
    }
    result = _process_op(
        "H200_NVJET_SPLIT", entry, get_hardware_config_path("nvidia_h200"),
        False, True)
    prediction_ns, actual_ns = result[3]
    assert prediction_ns > 0 and actual_ns == 6400.0
