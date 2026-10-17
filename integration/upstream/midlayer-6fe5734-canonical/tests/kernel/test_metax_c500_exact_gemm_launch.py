import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.launch_config.gemm_launch_selection import GemmLaunchSelection
from src.kernel.launch_config.gemm_workload import gemm_launch_selection_key
from src.kernel.launch_config.metax_c500 import gemm_launch


@pytest.fixture(autouse=True)
def _clear_launch_data_cache():
    gemm_launch._data.cache_clear()
    yield
    gemm_launch._data.cache_clear()


def _attributes(position="qkv_proj", *, launch_policy="injected_launch"):
    return {
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": position,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "launch_policy": launch_policy,
    }


def _selection(shape, variant, attributes, *, tile, split_k=1,
               reduction="none"):
    key = gemm_launch_selection_key(shape, "bf16", variant, attributes)
    tile_values = (None, None, None) if tile is None else tuple(tile)
    return key, {key: GemmLaunchSelection(
        *tile_values, None, split_k, reduction)}


def _load_selection(monkeypatch, selection):
    monkeypatch.setattr(gemm_launch, "_EXPECTED_SEMANTIC_KEYS", 1)
    monkeypatch.setattr(
        gemm_launch, "load_gemm_launch_selection",
        lambda *_args, **_kwargs: selection)
    gemm_launch._data.cache_clear()


def test_injected_m1_can_select_tensor_gemm_without_observed_geometry(
        monkeypatch):
    shape = {"b": 1, "m": 1, "n": 151936, "k": 2560}
    attributes = _attributes("lm_head")
    _, selection = _selection(
        shape, "gemv", attributes, tile=[64, 128, 256])
    _load_selection(monkeypatch, selection)

    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemv", attributes,
        HWSpec("configs/hardware/metax_c500.yaml"))

    assert launch.algorithm == "mcblas_injected_gemm:gemm"
    assert launch.execution_variant == "gemm"
    assert launch.grid == (1, 1187, 1)
    assert launch.block == (256, 1, 1)


def test_injected_split_k_uses_logical_main_and_reducer_geometry(monkeypatch):
    shape = {"b": 1, "m": 930, "n": 2560, "k": 9728}
    attributes = _attributes("down_proj")
    _, selection = _selection(
        shape, "gemm", attributes, tile=[192, 256, 64], split_k=2,
        reduction="fp32_external")
    _load_selection(monkeypatch, selection)
    hardware = HWSpec("configs/hardware/metax_c500.yaml")

    main_launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", attributes, hardware)
    reducer = gemm_launch.select_gemm_reducer_launch(
        shape, "bf16", "gemm", attributes, hardware)

    assert main_launch.grid == (5, 10, 2)
    assert main_launch.block == (512, 1, 1)
    assert reducer.algorithm == "logical_gemm_aux:gemm_reduce"
    assert reducer.grid == (80, 59, 1)
    assert reducer.block == (512, 1, 1)
    assert reducer.parameters == {
        "split_k_partial_bytes_per_element": 4.0,
        "split_k_partial_dram_bytes_per_element": 4.0,
    }


def test_injected_gemv_uses_logical_geometry_without_tile(monkeypatch):
    shape = {"b": 1, "m": 1, "n": 4096, "k": 4096}
    attributes = _attributes()
    _, selection = _selection(shape, "gemv", attributes, tile=None)
    _load_selection(monkeypatch, selection)

    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemv", attributes,
        HWSpec("configs/hardware/metax_c500.yaml"))

    assert launch.execution_variant == "gemv"
    assert launch.tiling is None
    assert launch.grid == (256, 1, 1)
    assert launch.block == (256, 1, 1)


def test_analytical_path_does_not_touch_injected_data(monkeypatch):
    monkeypatch.setattr(
        gemm_launch, "_data",
        lambda: (_ for _ in ()).throw(
            AssertionError("analytical path loaded injected data")))

    launch = gemm_launch.select_gemm_launch(
        {"b": 1, "m": 16, "n": 4096, "k": 4096},
        "bf16", "gemm", {"launch_policy": "analytical_launch"},
        HWSpec("configs/hardware/metax_c500.yaml"))

    assert launch.algorithm == "mcblas_analytical_gemm"


def test_semantic_context_does_not_change_physical_lookup(monkeypatch):
    shape = {"b": 1, "m": 8, "n": 4096, "k": 4096}
    attributes = _attributes("qkv_proj")
    _, selection = _selection(
        shape, "gemm", attributes, tile=[64, 128, 256])
    _load_selection(monkeypatch, selection)

    changed_attributes = {
        **attributes,
        "position": "down_proj",
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
    }
    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", changed_attributes,
        HWSpec("configs/hardware/metax_c500.yaml"))

    assert dict(launch.tiling.tile_shape) == {"m": 64, "n": 128, "k": 256}
