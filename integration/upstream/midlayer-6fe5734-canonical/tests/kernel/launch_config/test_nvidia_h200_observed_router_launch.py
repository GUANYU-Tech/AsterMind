"""H200 SGLang cuBLAS router: grid-block-only launch rows and their use."""

import json

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.gemm_launch_selection import (
    compress_gemm_launch_csv,
    decompress_gemm_launch_csv,
    load_gemm_launch_selection,
    load_gemm_launch_selection_files,
)
from src.kernel.launch_config.select_launch_config import (
    select_gemm_reducer_launch,
    select_launch_config,
)


_ROUTER = {
    "score_function": "sigmoid", "topk_method": "noaux_tc",
    "expert_group_count": 8, "topk_group_count": 4,
    "normalize_topk_probability": True,
    "weight_dtype": "bf16", "weight_group_size": 0,
}


def _hardware():
    return HWSpec("configs/hardware/nvidia_h200.yaml")


def _node_attributes(runner):
    attributes = dict(
        _ROUTER, has_bias=False, position="router",
        launch_policy="analytical_launch")
    if runner:
        attributes["router_linear_runner"] = "sglang_linear_bf16_fp32_cublas"
    return attributes


def _launch(m, runner, hardware):
    return select_launch_config(
        "nvidia_h200", KernelKind.GEMM, {"b": 1, "m": m, "n": 256, "k": 7168},
        "bf16", "moe_router", _node_attributes(runner), hardware)


def test_rows_keep_tile_and_split_null_and_state_the_observed_grid():
    lut = load_gemm_launch_selection("nvidia_h200", software_revision="cuda_12.9")
    assert sorted(key[3] for key in lut) == [17, 24, 32, 48, 64, 96, 128, 192]
    row = lut[("bf16", "gemm", 1, 24, 256, 7168, "row_major", "row_major",
               "row_major", False, True, "none", False)]
    assert (row.tile_m, row.tile_n, row.tile_k, row.pipeline_depth,
            row.split_k) == (None, None, None, None, None)
    assert row.reduction == "fp32_external"
    assert row.observed_grid.grid == (8, 10, 1)
    assert row.observed_grid.reducer_grid == (8, 2)
    assert row.execution_variant == "gemm"


def test_a_row_that_claims_a_tile_fails_closed(tmp_path):
    source = __import__("pathlib").Path("launch_data/nvidia_h200")
    csv = decompress_gemm_launch_csv(
        (source / "gemm_launch_selection.csv.zst").read_bytes()).decode()
    lines = csv.splitlines()
    fields = lines[1].split(",")
    fields[13:16] = ["64", "8", "64"]
    tampered = "\n".join([lines[0], ",".join(fields), *lines[2:]]) + "\n"
    compressed = compress_gemm_launch_csv(tampered.encode())
    manifest = json.loads(
        (source / "gemm_launch_selection.manifest.json").read_text())
    import hashlib
    manifest["csv_sha256"] = hashlib.sha256(tampered.encode()).hexdigest()
    manifest["csv_compressed_sha256"] = hashlib.sha256(compressed).hexdigest()
    (tmp_path / "rows.csv.zst").write_bytes(compressed)
    (tmp_path / "rows.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="grid-block-only"):
        load_gemm_launch_selection_files(
            tmp_path / "rows.csv.zst", tmp_path / "rows.json",
            hardware_id="nvidia_h200", software_revision="cuda_12.9")


def test_only_the_sglang_runner_on_an_observed_row_uses_the_observation():
    hardware = _hardware()
    assert _launch(24, True, hardware).algorithm == (
        "cublaslt_observed_grid:moe_router")
    assert _launch(24, True, hardware).grid == (8, 10, 1)
    # The vLLM stack, and SGLang outside the observed rows, keep the
    # analytical launch; <= 16 tokens stays the DSV3 router kernel.
    assert _launch(24, False, hardware).algorithm == (
        "cublas_analytical_gemm:moe_router")
    assert _launch(50, True, hardware).algorithm == (
        "cublas_analytical_gemm:moe_router")
    assert _launch(16, True, hardware).algorithm == (
        "vllm_dsv3_router_gemm_float_out")


def test_observed_launch_prices_with_declared_limitations():
    hardware = _hardware()
    shape = {"b": 1, "m": 96, "n": 256, "k": 7168}
    launch = _launch(96, True, hardware)
    main = predict(KernelSpec(
        kind=KernelKind.GEMM, hardware_id="nvidia_h200", shape=shape,
        dtype="bf16", variant="moe_router", launch=launch,
        attributes=dict(_ROUTER)), hardware)
    assert main.t_total_s > 0
    reducer_launch = select_gemm_reducer_launch(
        "nvidia_h200", shape, "bf16", "moe_router",
        _node_attributes(True), hardware)
    assert reducer_launch.grid == (8, 6, 1)
    reducer = predict(KernelSpec(
        kind=KernelKind.VECTOR, hardware_id="nvidia_h200",
        shape={"element_count": 96 * 256}, dtype="bf16",
        variant="gemm_reduce", launch=reducer_launch,
        attributes={"split_k": 2}), hardware)
    assert reducer.t_total_s > 0
    main_limitations = main.diagnostics.limitations
    assert "launch_observed_on_cuda_12_9_not_serving_cuda_13_0" in (
        main_limitations)
    assert "split_k_count_unobserved_priced_at_lower_bound" in (
        reducer.diagnostics.limitations)
