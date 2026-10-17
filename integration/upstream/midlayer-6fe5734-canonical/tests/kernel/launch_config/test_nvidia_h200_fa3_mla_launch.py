"""H200 FlashAttention-3 MLA launch derivation and template contracts (wp10).

The expected geometries are the source-derived launches of the pinned
vllm-flash-attention revision for the DeepSeek-V3 TP4 MLA shapes (32 local
heads); they match the recorded H200 launch geometry of those kernels.
"""

import functools

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
    fa3_num_splits_heuristic,
    owns_select_flash_attn_launch,
    select_fa3_prepare_varlen_launch,
    select_target_mla_launch,
)
from src.kernel.launch_config.select_launch_config import select_launch_config


# The audited vLLM 0.24 deployment: FULL_AND_PIECEWISE, capture size 256.
_FULL_256 = (True, 256)


@functools.lru_cache(maxsize=None)
def _hardware(name):
    return HWSpec(f"configs/hardware/{name}.yaml")


def _mla_spec(hardware_id, variant, q_lens, kv_lens, *, graph):
    hardware = _hardware(hardware_id)
    decode = variant == "mla_decode"
    shape = {
        "batch": len(q_lens), "q_len": max(q_lens), "kv_len": max(kv_lens),
        "qk_head_dim": 576 if decode else 192,
        "v_head_dim": 512 if decode else 128,
        "num_query_heads": 32, "kv_lora_rank": 512,
    }
    # ``graph``: None = eager deployment, else (full graphs, capture size).
    full, capture = (False, 0) if graph is None else graph
    attributes = {
        "causal": True, "kv_layout": "mla", "kv_dist": "uniform",
        "q_lens": tuple(q_lens), "kv_lens": tuple(kv_lens),
        "full_cuda_graphs": full, "max_cudagraph_capture_size": capture,
    }
    launch = select_launch_config(
        hardware_id, KernelKind.FLASH_ATTN, shape, "bf16", variant,
        attributes, hardware)
    return KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id=hardware_id, shape=shape,
        dtype="bf16", variant=launch.execution_variant, launch=launch,
        attributes=attributes)


def _combine_spec(batch, kv_len, splits):
    hardware = _hardware("nvidia_h200")
    shape = {"batch": batch, "q_len": 1, "kv_len": kv_len, "head_dim": 512,
             "num_query_heads": 32, "num_kv_heads": 1}
    attributes = {
        "causal": True, "kv_layout": "mla", "kv_dist": "uniform",
        "q_lens": (1,) * batch, "kv_lens": (kv_len,) * batch,
        "fa_ksteps": None, "fa_eff_mfrac": 1.0, "num_kv_splits": splits,
    }
    launch = select_launch_config(
        "nvidia_h200", KernelKind.FLASH_ATTN, shape, "bf16", "combine",
        attributes, hardware)
    return KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id="nvidia_h200", shape=shape,
        dtype="bf16", variant="combine", launch=launch,
        attributes={name: attributes[name] for name in (
            "causal", "kv_layout", "kv_dist", "fa_ksteps", "fa_eff_mfrac",
            "q_lens", "kv_lens")})


@pytest.mark.parametrize("batch", (1, 4, 16, 64))
def test_decode_launch_is_the_persistent_fa3_hdim64_512_kernel(batch):
    spec = _mla_spec("nvidia_h200", "mla_decode", (1,) * batch,
                     (8000,) * batch, graph=_FULL_256)
    launch = spec.launch
    assert launch.algorithm == "vllm_fa3_sm90_mla_decode"
    assert launch.grid == (132, 1, 1) and launch.block == (384, 1, 1)
    assert launch.cluster == (1, 1, 1)
    assert dict(launch.tiling.tile_shape) == {
        "q": 64, "kv": 64, "qk_head": 64, "value_head": 512}
    assert launch.tiling.pipeline_depth == 2
    # vLLM's full-CUDA-graph split bound.
    assert launch.tiling.num_kv_splits == 32


@pytest.mark.parametrize(("batch", "grid"), (
    (1, (4, 4, 1)), (4, (20, 4, 1)), (16, (80, 4, 1)), (64, (320, 4, 1))))
def test_combine_grid_follows_the_varlen_tile_scheduler(batch, grid):
    launch = _combine_spec(batch, 8000, 32).launch
    assert launch.algorithm == "vllm_fa3_sm90_combine"
    assert launch.grid == grid and launch.block == (256, 1, 1)


def test_prefill_launch_is_the_persistent_fa3_hdim192_128_kernel():
    spec = _mla_spec("nvidia_h200", "mla_prefill", (7373,) * 4,
                     (7373,) * 4, graph=None)
    launch = spec.launch
    assert launch.algorithm == "vllm_fa3_sm90_mla_prefill"
    assert launch.grid == (132, 1, 1) and launch.block == (384, 1, 1)
    assert dict(launch.tiling.tile_shape) == {
        "q": 128, "kv": 128, "qk_head": 192, "value_head": 128}
    assert launch.tiling.num_kv_splits == 1


def test_prepare_varlen_launch_matches_the_warp_switch():
    assert select_fa3_prepare_varlen_launch(1).block == (32, 1, 1)
    assert select_fa3_prepare_varlen_launch(31).block == (32, 1, 1)
    assert select_fa3_prepare_varlen_launch(64).block == (128, 1, 1)
    assert select_fa3_prepare_varlen_launch(64).grid == (1, 1, 1)
    assert select_fa3_prepare_varlen_launch(993).grid == (2, 1, 1)


def test_split_heuristic_matches_the_source_cases():
    # batch * heads = 48 on 108 SMs chooses 2 splits (heuristics.h comment).
    assert fa3_num_splits_heuristic(48, 108, 64, 1, 0, False, 128) == 2
    assert fa3_num_splits_heuristic(200, 132, 64, 4, 0, False, 128) == 1
    assert fa3_num_splits_heuristic(1, 132, 4, 1, 0, False, 128) == 1


def test_owned_domains_route_everything_else_to_the_projected_package():
    hardware = _hardware("nvidia_h200")
    dense = {"batch": 1, "q_len": 1, "kv_len": 128, "head_dim": 128,
             "num_query_heads": 8, "num_kv_heads": 8}
    assert not owns_select_flash_attn_launch(
        dense, "bf16", "combine", {"kv_layout": "paged"}, hardware)
    assert not owns_select_flash_attn_launch(
        dense, "bf16", "decode", {"kv_layout": "paged"}, hardware)
    # Chunked-context MLA prefill (q < kv) is not the single FA3 call.
    context = _mla_spec("nvidia_h200", "mla_prefill", (512,), (4096,),
                        graph=None)
    assert context.launch.algorithm == "vllm_flash_attn_v2_mla_prefill"
    with pytest.raises(ValueError, match="owns only the derived FA3 domain"):
        select_target_mla_launch(
            dict(context.shape), "bf16", "mla_prefill",
            dict(context.attributes), hardware)


def test_a800_target_mla_launch_is_not_routed_to_h200():
    spec = _mla_spec("nvidia_a800", "mla_prefill", (512,), (512,),
                     graph=None)
    assert spec.launch.algorithm == "vllm_flash_attn_v2_mla_prefill"


def test_fa3_template_prices_every_in_scope_launch_with_mechanisms():
    hardware = _hardware("nvidia_h200")
    one = predict(_mla_spec("nvidia_h200", "mla_prefill", (7373,),
                            (7373,), graph=None), hardware, "midlayer")
    four = predict(_mla_spec("nvidia_h200", "mla_prefill", (7373,) * 4,
                             (7373,) * 4, graph=None), hardware, "midlayer")
    assert one.bottleneck_group == "fa3_tensor_softmax_chain"
    # Four identical causal prompts are four times the tile work.
    assert 3.5 * one.t_total_s < four.t_total_s < 4.5 * one.t_total_s
    small = predict(_mla_spec("nvidia_h200", "mla_decode", (1,), (8000,),
                              graph=_FULL_256), hardware, "midlayer")
    large = predict(_mla_spec("nvidia_h200", "mla_decode", (1,) * 64,
                              (8000,) * 64, graph=_FULL_256), hardware,
                    "midlayer")
    assert 0 < small.t_total_s < large.t_total_s
    combine = predict(_combine_spec(4, 8000, 32), hardware, "midlayer")
    assert combine.t_total_s > 0
    # Roofline stays available through the aggregate FA3 demand.
    assert predict(_mla_spec("nvidia_h200", "mla_decode", (1,), (8000,),
                             graph=_FULL_256), hardware, "roofline").t_total_s > 0


def test_fa3_template_requires_the_mechanism_section(monkeypatch):
    hardware = _hardware("nvidia_h200")
    spec = _mla_spec("nvidia_h200", "mla_decode", (1,), (8000,), graph=_FULL_256)
    monkeypatch.setattr(hardware, "fa3_mla_mechanisms", None)
    with pytest.raises(NotImplementedError, match="fa3_mla_mechanisms"):
        predict(spec, hardware, "midlayer")


def test_decode_split_bound_follows_the_deployment_graph_rule():
    full = _mla_spec("nvidia_h200", "mla_decode", (1,), (8000,),
                     graph=_FULL_256)
    eager = _mla_spec("nvidia_h200", "mla_decode", (1,), (8000,),
                      graph=None)
    piecewise_only = _mla_spec("nvidia_h200", "mla_decode", (1,), (8000,),
                               graph=(False, 256))
    beyond_capture = _mla_spec("nvidia_h200", "mla_decode", (1,) * 257,
                               (512,) * 257, graph=_FULL_256)
    assert full.launch.tiling.num_kv_splits == 32
    # Without full graphs: the FA3 heuristic over 125 KV blocks, 132 SMs.
    assert eager.launch.tiling.num_kv_splits == 107
    assert piecewise_only.launch.tiling.num_kv_splits == 107
    # Beyond the capture size: the heuristic over 8 KV blocks.
    assert beyond_capture.launch.tiling.num_kv_splits == 7


def test_decode_selector_requires_the_deployment_graph_facts():
    hardware = _hardware("nvidia_h200")
    shape = {"batch": 1, "q_len": 1, "kv_len": 8000, "qk_head_dim": 576,
             "v_head_dim": 512, "num_query_heads": 32, "kv_lora_rank": 512}
    attributes = {"causal": True, "kv_layout": "mla", "kv_dist": "uniform",
                  "q_lens": (1,), "kv_lens": (8000,)}
    with pytest.raises(KeyError, match="full_cuda_graphs"):
        select_launch_config(
            "nvidia_h200", KernelKind.FLASH_ATTN, shape, "bf16",
            "mla_decode", attributes, hardware)


def test_padded_full_graph_combine_grid_counts_the_padded_requests():
    hardware = _hardware("nvidia_h200")
    shape = {"batch": 3, "q_len": 1, "kv_len": 8000, "head_dim": 512,
             "num_query_heads": 32, "num_kv_heads": 1}
    attributes = {
        "causal": True, "kv_layout": "mla", "kv_dist": "uniform",
        "q_lens": (1,) * 3, "kv_lens": (8000,) * 3,
        "fa_ksteps": None, "fa_eff_mfrac": 1.0, "num_kv_splits": 32,
        "graph_logical_tokens": 3, "graph_model_input_tokens": 4,
        "graph_runtime_mode": "FULL",
    }
    launch = select_launch_config(
        "nvidia_h200", KernelKind.FLASH_ATTN, shape, "bf16", "combine",
        attributes, hardware)
    # LINEARIZE: ceil(4 * 32 / 8) + 4 padded requests, 512 / 128 columns.
    assert launch.grid == (20, 4, 1)


def test_uniform_verify_queries_run_the_fa3_decode_kernel():
    # 32 local heads: q_len 2 packs 64 rows into one kBlockM = 64 M block
    # (vLLM FlashAttnMLA runs uniform speculative verify as decode); q_len 3
    # would need two M blocks and a mixed-length batch is not uniform.
    verify = _mla_spec("nvidia_h200", "mla_decode", (2,) * 4, (8000,) * 4,
                       graph=_FULL_256)
    assert verify.launch.algorithm == "vllm_fa3_sm90_mla_decode"
    for q_lens in ((3,) * 4, (1, 2, 1, 2)):
        other = _mla_spec("nvidia_h200", "mla_decode", q_lens, (8000,) * 4,
                          graph=_FULL_256)
        assert other.launch.algorithm != "vllm_fa3_sm90_mla_decode"


def _absorbed_prefill_spec(q_lens, kv_lens):
    hardware = _hardware("nvidia_h200")
    shape = {
        "batch": len(q_lens), "q_len": max(q_lens), "kv_len": max(kv_lens),
        "qk_head_dim": 576, "v_head_dim": 512,
        "num_query_heads": 16, "kv_lora_rank": 512,
    }
    attributes = {
        "causal": True, "kv_layout": "mla", "kv_dist": "uniform",
        "q_lens": tuple(q_lens), "kv_lens": tuple(kv_lens),
        "full_cuda_graphs": False, "max_cudagraph_capture_size": 0,
    }
    launch = select_launch_config(
        "nvidia_h200", KernelKind.FLASH_ATTN, shape, "bf16",
        "mla_absorbed_prefill", attributes, hardware)
    return hardware, KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id="nvidia_h200", shape=shape,
        dtype="bf16", variant=launch.execution_variant, launch=launch,
        attributes=attributes)


def test_sglang_absorbed_mla_prefill_uses_flashinfer_hopper_tile():
    hardware, spec = _absorbed_prefill_spec((4097, 7373), (7373, 7373))
    assert spec.launch.algorithm == "flashinfer_mla_hopper_sm90_paged"
    assert spec.launch.grid == (132, 1, 1)
    assert spec.launch.block == (256, 1, 1)
    assert spec.launch.tiling.tile_shape == {
        "q": 64, "kv": 64, "qk_head": 64, "value_head": 512}
    assert predict(spec, hardware, "midlayer").t_total_s > 0


def test_sglang_absorbed_mla_prefill_grows_with_prefix():
    hardware, short = _absorbed_prefill_spec((4096,), (4096 + 64,))
    _, long = _absorbed_prefill_spec((4096,), (4096 + 4096,))
    assert (predict(long, hardware, "midlayer").t_total_s
            > predict(short, hardware, "midlayer").t_total_s)


def _flashinfer_decode_spec(batch, kv_len):
    hardware = _hardware("nvidia_h200")
    shape = {
        "batch": batch, "q_len": 1, "kv_len": kv_len,
        "qk_head_dim": 576, "v_head_dim": 512,
        "num_query_heads": 16, "kv_lora_rank": 512,
    }
    attributes = {
        "causal": True, "kv_layout": "mla", "kv_dist": "uniform",
        "q_lens": (1,) * batch, "kv_lens": (kv_len,) * batch,
        "full_cuda_graphs": False, "max_cudagraph_capture_size": 0,
        "attention_backend": "flashinfer",
    }
    launch = select_launch_config(
        "nvidia_h200", KernelKind.FLASH_ATTN, shape, "bf16", "mla_decode",
        attributes, hardware)
    return hardware, KernelSpec(
        kind=KernelKind.FLASH_ATTN, hardware_id="nvidia_h200", shape=shape,
        dtype="bf16", variant=launch.execution_variant, launch=launch,
        attributes=attributes)


def test_sglang_decode_routes_to_flashinfer_and_prices_the_split_merge():
    hardware, split = _flashinfer_decode_spec(1, 8192)
    assert split.launch.algorithm == "flashinfer_mla_hopper_sm90_paged"
    assert split.launch.tiling.num_kv_splits == 1
    groups = predict(split, hardware, "midlayer").t_per_group
    # One request of 8192 tokens over 132 CTAs is split into 192-token
    # chunks, so the kernel crosses grid.sync and merges 43 chunks serially.
    assert groups["flashinfer_mla_grid_sync_and_merge"] == pytest.approx(
        hardware.cooperative_grid_sync_ns(132) * 1e-9
        + 43 * hardware.split_merge_chunk_s())
    # Two dependent DRAM-cold plan loads precede the first KV tile.
    dram = next(entry for entry in
                hardware.fa2_oaccum_load_use_cycles["cache_entries"]
                if entry["cache_tier"] == "dram_cg")
    assert groups["flashinfer_mla_plan_prologue"] == pytest.approx(
        2 * dram["first_load_ready_cycles"] / 1.98e9, rel=1e-3)


def test_vllm_decode_without_backend_attribute_keeps_fa3():
    hardware = _hardware("nvidia_h200")
    spec = _mla_spec("nvidia_h200", "mla_decode", (1,), (8192,),
                     graph=_FULL_256)
    assert spec.launch.algorithm == "vllm_fa3_sm90_mla_decode"
