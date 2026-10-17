"""H200 vLLM 0.24 FP8 block-quantized Triton fused-MoE launch and template."""

import dataclasses
import random

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config


H200 = HWSpec("configs/hardware/nvidia_h200.yaml")
A800 = HWSpec("configs/hardware/nvidia_a800.yaml")
DSV3_TP4 = {"experts": 256, "topk": 8, "hidden": 7168, "intermediate": 512}


def _counts(tokens, experts=256, topk=8, seed=7):
    rng = random.Random(seed + tokens)
    counts = [0] * experts
    for _ in range(tokens):
        for expert in rng.sample(range(experts), topk):
            counts[expert] += 1
    return tuple(counts)


def _attributes(counts, routed_weight_dtype="fp8", expert_parallel_size=1):
    return {
        "tokens_per_expert": counts,
        "local_tokens_per_expert": counts,
        "expert_parallel_size": expert_parallel_size,
        "expert_rank": 0,
        "owned_experts": tuple(range(len(counts))),
        "expert_weight_tile_rows": None,
        "routed_scaling_factor": 2.5,
        "weight_group_size": 0,
        "routed_weight_dtype": routed_weight_dtype,
    }


def _spec(hardware, hardware_id, tokens, variant, dims=DSV3_TP4, **kwargs):
    shape = {"tokens": tokens, **dims, "dtype_bytes": 2}
    counts = _counts(tokens, dims["experts"], dims["topk"])
    attributes = _attributes(counts, **kwargs)
    launch = select_launch_config(
        hardware_id, KernelKind.FUSED_MOE, shape, "bf16", variant,
        attributes, hardware)
    return KernelSpec(
        kind=KernelKind.FUSED_MOE, hardware_id=hardware_id, shape=shape,
        dtype="bf16", variant=variant, launch=launch, attributes=attributes)


# Launch geometry of every in-scope fused_moe_kernel in the rank-0 nsys
# windows of the DSV3-L32 TP4 vLLM 0.24 deployment (decode C=1/4/16/64 and
# prefill steps of one, three and four 7373-token prompts).
@pytest.mark.parametrize(("tokens", "w1_grid", "w2_grid", "tile_m", "shared"), (
    (1, 64, 448, 16, 37376),
    (4, 256, 1792, 16, 37376),
    (16, 1024, 7168, 16, 37376),
    (64, 2176, 15232, 16, 37376),
    (7373, 9392, 65744, 64, 98816),
    (22119, 24136, 168952, 64, 98816),
    (29492, 31512, 220584, 64, 98816),
))
def test_h200_fp8_selector_reproduces_trace_launch_geometry(
        tokens, w1_grid, w2_grid, tile_m, shared):
    for variant, grid in (("w1", w1_grid), ("w2", w2_grid)):
        launch = _spec(H200, "nvidia_h200", tokens, variant).launch
        assert launch.algorithm.startswith("vllm_triton_fused_moe_fp8_block128")
        assert launch.grid == (grid, 1, 1)
        assert launch.block == (128, 1, 1)
        assert dict(launch.tiling.tile_shape) == {
            "m": tile_m, "n": 128, "k": 128}
        assert launch.parameters["dynamic_shared_memory_bytes"] == shared


def test_h200_fp8_selector_marks_naive_assignment_from_source_predicate():
    assert "_naive_" in _spec(H200, "nvidia_h200", 8, "w1").launch.algorithm
    assert "_aligned_" in _spec(H200, "nvidia_h200", 9, "w1").launch.algorithm


@pytest.mark.parametrize(("tokens", "variant", "dims", "kwargs"), (
    (64, "w1", DSV3_TP4, {"routed_weight_dtype": "bf16"}),
    (4096, "w2", {"experts": 8, "topk": 2, "hidden": 4096,
                  "intermediate": 14336}, {}),
    (64, "w1", {"experts": 256, "topk": 8, "hidden": 6144,
                "intermediate": 256}, {}),
    (2048, "w2", {"experts": 128, "topk": 8, "hidden": 2048,
                  "intermediate": 768}, {}),
))
def test_h200_out_of_domain_launch_keeps_projected_rule(
        tokens, variant, dims, kwargs):
    h200 = _spec(H200, "nvidia_h200", tokens, variant, dims, **kwargs).launch
    a800 = _spec(A800, "nvidia_a800", tokens, variant, dims, **kwargs).launch
    assert h200 == a800


def test_h200_routed_experts_keeps_projected_marlin_rule():
    dims = {"experts": 384, "topk": 8, "hidden": 7168, "intermediate": 2048}
    shape = {"tokens": 32, **dims, "dtype_bytes": 2}
    counts = _counts(32, 384, 8)
    attributes = {
        **_attributes(counts, routed_weight_dtype="int4"),
        "weight_group_size": 32,
    }
    launches = [
        select_launch_config(
            hardware_id, KernelKind.FUSED_MOE, shape, "bf16",
            "routed_experts", attributes, hardware)
        for hardware_id, hardware in (("nvidia_h200", H200),
                                      ("nvidia_a800", A800))]
    assert launches[0] == launches[1]


@pytest.mark.parametrize(("tokens", "variant"), (
    (1, "w1"), (4, "w2"), (64, "w1"), (29492, "w2")))
def test_h200_fp8_launch_uses_measured_tile_pipeline(tokens, variant):
    spec = _spec(H200, "nvidia_h200", tokens, variant)
    prediction = predict(spec, H200, "midlayer")
    pipeline = prediction.diagnostics.pipeline
    assert pipeline is not None
    assert "gpu_grouped_gemm/fixed_span" in pipeline.critical_path
    assert prediction.t_total_s > 0.0


def test_fp8_launch_without_measured_program_fails_closed():
    spec = _spec(H200, "nvidia_h200", 64, "w1")
    tampered = dataclasses.replace(
        spec.launch,
        parameters={**spec.launch.parameters, "num_stages": 6.0})
    with pytest.raises(NotImplementedError, match="no measured pipeline"):
        predict(dataclasses.replace(spec, launch=tampered), H200, "midlayer")
    a800_spec = dataclasses.replace(spec, hardware_id="nvidia_a800")
    with pytest.raises(NotImplementedError, match="no measured pipeline"):
        predict(a800_spec, A800, "midlayer")


def test_fp8_tile_pipeline_rows_are_content_addressed(tmp_path):
    import shutil
    import yaml

    from src.kernel.hardware_spec import (
        _normalize_grouped_gemm_fp8_block_tile_pipelines,
    )

    config = yaml.safe_load(open("configs/hardware/nvidia_h200.yaml"))
    rows = config["scheduling"]["grouped_gemm_fp8_block_tile_pipelines"]
    reference = (
        "microbench/nvidia_h200/references/"
        "fp8_block_grouped_gemm_tile_pipeline_sm90.json")
    refill_reference = (
        "microbench/nvidia_h200/references/"
        "fp8_block_grouped_gemm_tile_wave_refill_sm90.json")
    dram_reference = (
        "microbench/nvidia_h200/references/"
        "fp8_block_grouped_gemm_tile_wave_refill_dram_sm90.json")
    (tmp_path / reference).parent.mkdir(parents=True)
    shutil.copy(reference, tmp_path / reference)
    shutil.copy(refill_reference, tmp_path / refill_reference)
    shutil.copy(dram_reference, tmp_path / dram_reference)
    normalized = _normalize_grouped_gemm_fp8_block_tile_pipelines(
        rows, tmp_path)
    assert set(normalized) == {row["program"] for row in rows}
    # The DRAM-fed later-wave service is optional, all-or-none, bound to its
    # reference, and limited to the probed K depth (K <= 1024: 8 tiles).
    dram_row = next(row for row in rows if "dram_wave_refill_ns" in row)
    assert normalized[dram_row["program"]]["dram_wave_max_k_tiles"] == 8
    partial = [{name: value for name, value in dram_row.items()
                if name != "dram_wave_k_round_interval_ns"}]
    with pytest.raises(ValueError, match="together"):
        _normalize_grouped_gemm_fp8_block_tile_pipelines(partial, tmp_path)
    wrong_dram = [dict(dram_row, dram_wave_refill_ns=1.0)]
    with pytest.raises(ValueError, match="dram_wave_refill_ns"):
        _normalize_grouped_gemm_fp8_block_tile_pipelines(wrong_dram, tmp_path)
    wrong_depth = [dict(rows[0], pipeline_depth=rows[0]["num_stages"])]
    with pytest.raises(ValueError, match="pipeline_depth"):
        _normalize_grouped_gemm_fp8_block_tile_pipelines(wrong_depth, tmp_path)
    wrong_refill = [dict(rows[0], wave_refill_ns=1.0)]
    with pytest.raises(ValueError, match="wave_refill_ns"):
        _normalize_grouped_gemm_fp8_block_tile_pipelines(wrong_refill, tmp_path)
    (tmp_path / reference).write_text("{}\n")
    with pytest.raises(ValueError, match="reference_sha256"):
        _normalize_grouped_gemm_fp8_block_tile_pipelines(rows, tmp_path)
