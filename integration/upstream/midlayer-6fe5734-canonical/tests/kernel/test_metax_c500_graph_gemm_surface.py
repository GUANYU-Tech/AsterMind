"""Contract tests for the C500 serving graph-replay GEMM surface overlay."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.launch_config.metax_c500 import gemm_launch


@pytest.fixture(autouse=True)
def _clear_caches():
    gemm_launch._data.cache_clear()
    gemm_launch._graph_surface.cache_clear()
    yield
    gemm_launch._data.cache_clear()
    gemm_launch._graph_surface.cache_clear()


def _attributes(graph_runtime_mode=None, logical_tokens=16,
                model_input_tokens=16):
    attributes = {
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": "gate_up_proj",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "launch_policy": "injected_launch",
    }
    if graph_runtime_mode is not None:
        attributes["graph_runtime_mode"] = graph_runtime_mode
        attributes["graph_logical_tokens"] = logical_tokens
        attributes["graph_model_input_tokens"] = model_input_tokens
    return attributes


def _hardware():
    return HWSpec("configs/hardware/metax_c500.yaml")


def test_overlay_never_consulted_without_graph_token_domain(monkeypatch):
    def _explode():
        raise AssertionError("graph surface loaded outside the gated domain")
    monkeypatch.setattr(gemm_launch, "_graph_surface", _explode)
    shape = {"b": 1, "m": 16, "n": 19456, "k": 2560}
    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", _attributes(), _hardware())
    assert launch.tiling.tile_shape == {"m": 32, "n": 256, "k": 128}


def test_overlay_hit_replaces_csv_selection_for_full_graph_buckets():
    # The deployed stack replays qwen3-32b's TP2 o_proj at bucket 16 with
    # 48x64x256/256t; the canonical dry-launch CSV picks a different tile.
    shape = {"b": 1, "m": 16, "n": 5120, "k": 4096}
    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", _attributes("FULL"), _hardware())
    plain = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", _attributes(), _hardware())
    assert launch.tiling.tile_shape == {"m": 48, "n": 64, "k": 256}
    assert launch.block == (256, 1, 1)
    assert launch.tiling.split_k == 1
    assert plain.tiling.tile_shape != launch.tiling.tile_shape


def test_overlay_hit_applies_to_piecewise_buckets():
    shape = {"b": 1, "m": 16, "n": 5120, "k": 4096}
    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", _attributes("PIECEWISE"), _hardware())
    assert launch.tiling.tile_shape == {"m": 48, "n": 64, "k": 256}


def test_overlay_piecewise_keys_by_exact_logical_tokens():
    # The deployment launches mixed-step GEMMs at the exact scheduled token
    # count: (2560,4096) at exact 113 selects 64x64x256, while the padded
    # bucket 128 selects 128x128x128/4stage.
    shape = {"b": 1, "m": 128, "n": 2560, "k": 4096}
    overlay = gemm_launch._graph_surface()
    assert overlay[(2560, 4096, 113)].tile_k == 256
    assert overlay[(2560, 4096, 128)].tile_k == 128
    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm",
        _attributes("PIECEWISE", logical_tokens=113, model_input_tokens=128),
        _hardware())
    assert launch.tiling.tile_shape == {"m": 64, "n": 64, "k": 256}


def test_overlay_full_keeps_padded_bucket_keying():
    # FULL decode graphs are captured at the padded bucket, so the physical
    # m stays the key even when fewer tokens are logical.
    shape = {"b": 1, "m": 128, "n": 2560, "k": 4096}
    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm",
        _attributes("FULL", logical_tokens=113, model_input_tokens=128),
        _hardware())
    assert launch.tiling.tile_shape == {"m": 128, "n": 128, "k": 128}


def test_overlay_piecewise_requires_logical_tokens():
    shape = {"b": 1, "m": 128, "n": 2560, "k": 4096}
    attributes = _attributes(
        "PIECEWISE", logical_tokens=113, model_input_tokens=128)
    del attributes["graph_logical_tokens"]
    with pytest.raises(ValueError, match="graph_logical_tokens"):
        gemm_launch.select_gemm_launch(
            shape, "bf16", "gemm", attributes, _hardware())


def test_overlay_split_k_drives_main_and_reducer_consistently():
    shape = {"b": 1, "m": 16, "n": 2560, "k": 9728}
    attributes = _attributes("FULL")
    main = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", attributes, _hardware())
    assert main.tiling.split_k == 5
    assert main.tiling.split_k == main.grid[2]
    reducer = gemm_launch.select_gemm_reducer_launch(
        shape, "bf16", "gemm", attributes, _hardware())
    assert reducer.execution_variant == "gemm_reduce"


def test_overlay_miss_inside_gated_domain_keeps_csv_selection():
    # The lm_head vocab cell is excluded from the overlay; a gated query on
    # it must fall through to the canonical CSV selection unchanged.
    shape = {"b": 1, "m": 16, "n": 151936, "k": 2560}
    overlay = gemm_launch._graph_surface()
    assert (151936, 2560, 16) not in overlay
    gated = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", _attributes("FULL"), _hardware())
    plain = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", _attributes(), _hardware())
    assert gated.tiling.tile_shape == plain.tiling.tile_shape
    assert gated.grid == plain.grid


def test_overlay_does_not_apply_to_gemv():
    shape = {"b": 1, "m": 1, "n": 151936, "k": 2560}
    gated = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemv", _attributes("FULL"), _hardware())
    plain = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemv", _attributes(), _hardware())
    assert gated.execution_variant == "gemv"
    assert gated.grid == plain.grid and gated.block == plain.block


def test_overlay_rejects_unknown_graph_runtime_mode():
    shape = {"b": 1, "m": 16, "n": 19456, "k": 2560}
    with pytest.raises(ValueError, match="graph runtime mode"):
        gemm_launch.select_gemm_launch(
            shape, "bf16", "gemm", _attributes("WEIRD"), _hardware())


def test_overlay_reference_tamper_fails_closed(monkeypatch):
    monkeypatch.setattr(gemm_launch, "_GRAPH_SURFACE_SHA256", "0" * 64)
    gemm_launch._graph_surface.cache_clear()
    with pytest.raises(ValueError, match="pinned artifact"):
        gemm_launch._graph_surface()
