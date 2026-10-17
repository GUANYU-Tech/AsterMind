"""Contract tests for the C500 vLLM-MetaX FlashAttention split forward."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.kernel.launch_config.metax_c500 import flash_attn_launch
from src.kernel.launch_config.select_launch_config import (
    select_flash_attn_forward_partition,
)


_SHAPE = {
    "batch": 4, "q_len": 37, "kv_len": 900, "head_dim": 128,
    "num_query_heads": 32, "num_kv_heads": 8,
}


def _attributes(q_lens, kv_lens, mode="PIECEWISE", kv_layout="paged_varlen"):
    return {
        "causal": True,
        "kv_layout": kv_layout,
        "kv_dist": "uniform",
        "q_lens": tuple(q_lens),
        "kv_lens": tuple(kv_lens),
        "launch_policy": "injected_launch",
        "split_policy": "serving",
        "graph_runtime_mode": mode,
    }


def test_mixed_call_splits_into_prefill_copy_and_decode_pathways():
    q_lens = (1, 1, 37, 5)
    parts = select_flash_attn_forward_partition(
        "metax_c500", _SHAPE, "prefill",
        _attributes(q_lens, (300, 900, 400, 60)))

    assert [part.role for part in parts] == ["prefill", "decode"]
    prefill, decode = parts
    assert prefill.request_indexes == (2, 3)
    assert prefill.output_copy_elements == (37 + 5) * 32 * 128
    assert prefill.output_copy_launch.algorithm == (
        "native_d2d_flash_attention_copy")
    assert decode.variant == "decode"
    assert decode.request_indexes == (0, 1)
    # The decode write-back copy belongs to the decode copy selector.
    assert decode.output_copy_elements == 0
    assert decode.output_copy_launch is None


def test_pure_prefill_call_keeps_one_pathway_with_its_slice_copy():
    parts = select_flash_attn_forward_partition(
        "metax_c500", {**_SHAPE, "batch": 1}, "prefill",
        _attributes((37,), (900,), mode="NONE"))

    assert len(parts) == 1
    assert parts[0].role == "prefill"
    assert parts[0].output_copy_elements == 37 * 32 * 128


def test_decode_and_mla_calls_are_not_partitioned():
    assert select_flash_attn_forward_partition(
        "metax_c500", {**_SHAPE, "q_len": 1}, "decode",
        _attributes((1, 1, 1, 1), (300, 900, 400, 60), mode="FULL")) == ()
    assert select_flash_attn_forward_partition(
        "metax_c500", _SHAPE, "prefill",
        _attributes((1, 1, 37, 5), (300, 900, 400, 60),
                    kv_layout="mla")) == ()


def test_full_graph_call_partitions_its_padded_request_abi():
    # Inside a FULL replay the backend sees the padded model-input requests,
    # which the Serving lowering passes as q_lens; the rule itself is the
    # same decode-threshold split.
    parts = select_flash_attn_forward_partition(
        "metax_c500", {**_SHAPE, "batch": 2, "q_len": 15}, "prefill",
        _attributes((1, 15), (33, 15), mode="FULL"))

    assert [(part.role, part.request_indexes) for part in parts] == [
        ("prefill", (1,)), ("decode", (0,))]
    assert parts[0].output_copy_elements == 15 * 32 * 128


@pytest.mark.parametrize("hardware_id", ("nvidia_a800", "ascend_910b"))
def test_other_hardware_keeps_one_fused_call(hardware_id):
    assert select_flash_attn_forward_partition(
        hardware_id, _SHAPE, "prefill",
        _attributes((1, 1, 37, 5), (300, 900, 400, 60))) == ()


def test_split_forward_decode_pathway_uses_native_split_heuristic():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {**_SHAPE, "batch": 2, "q_len": 1, "kv_len": 4000}
    attributes = {
        **_attributes((1, 1), (4000, 3000)),
        "is_decode": True,
    }
    fused = flash_attn_launch.select_flash_attn_launch(
        shape, "bf16", "decode", attributes, hardware)
    split = flash_attn_launch.select_flash_attn_launch(
        shape, "bf16", "decode",
        {**attributes, "flash_attn_forward_role": "decode"}, hardware)

    assert fused.tiling.num_kv_splits == 1
    assert split.tiling.num_kv_splits > 1


def test_unknown_forward_role_fails_closed():
    with pytest.raises(ValueError, match="flash_attn_forward_role"):
        flash_attn_launch.normalize_flash_attn_execution_attributes(
            _SHAPE, {**_attributes((37,), (900,)),
                     "flash_attn_forward_role": "chunk"})


def test_native_copy_plateau_covers_measured_four_mebibyte_probe():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")

    def native(byte_count):
        return predict(KernelSpec(
            kind=KernelKind.MEMCPY,
            hardware_id=hardware.hardware_id,
            transfer=PhysicalTransferConfig(
                algorithm="native_d2d_flash_attention_copy",
                bytes=byte_count,
                direction="d2d",
                link="dram",
            ),
        ), hardware, "midlayer").t_total_ns

    # The plateau floor holds across the measured 256-KiB--4-MiB range and
    # the price stays monotone in bytes inside and at the edge of it.
    assert native(2 * 1024 * 1024) > 35735.33924058255
    assert native(1024 * 1024) < native(2 * 1024 * 1024) < native(
        4 * 1024 * 1024)


@pytest.mark.parametrize("mode", ("FULL", "PIECEWISE"))
@pytest.mark.parametrize("variant, logical, physical, model_input", (
    ("rmsnorm", {"hidden": 128, "rows": 264 * 32},
     {"hidden": 128, "rows": 272 * 32}, {"hidden": 128, "rows": 272 * 32}),
    ("add_rmsnorm", {"hidden": 2560, "rows": 264},
     {"hidden": 2560, "rows": 272}, {"hidden": 2560, "rows": 272}),
    ("silu",
     {"element_count": 264 * 9728, "intermediate": 9728, "tok": 264},
     {"element_count": 272 * 9728, "intermediate": 9728, "tok": 272},
     {"element_count": 272 * 9728, "intermediate": 9728, "tok": 272}),
    ("rotary",
     {"seq_len": 264, "num_query_heads": 32, "num_kv_heads": 8,
      "head_dim": 128},
     {"seq_len": 272, "num_query_heads": 32, "num_kv_heads": 8,
      "head_dim": 128},
     {"seq_len": 272, "num_query_heads": 32, "num_kv_heads": 8,
      "head_dim": 128, "element_count": 272 * 40 * 128}),
))
def test_graph_padded_vector_prices_its_physical_dispatch(
        mode, variant, logical, physical, model_input):
    # A C500 Graph replay (FULL, and PIECEWISE segments) runs these source
    # programs over the padded token tensor; the launch grid is that
    # evidence, so the padded spec prices exactly like the physical one.
    from src.kernel.launch_config.select_launch_config import (
        select_launch_config,
    )

    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    launch = select_launch_config(
        "metax_c500", KernelKind.VECTOR, physical, "bf16", variant,
        {"launch_policy": "injected_launch"}, hardware=hardware)
    padded = KernelSpec(
        kind=KernelKind.VECTOR, hardware_id="metax_c500", shape=logical,
        dtype="bf16", variant=variant, launch=launch, attributes={
            "graph_logical_tokens": 264, "graph_model_input_tokens": 272,
            "graph_runtime_mode": mode, "model_input_shape": model_input})
    standalone = KernelSpec(
        kind=KernelKind.VECTOR, hardware_id="metax_c500", shape=physical,
        dtype="bf16", variant=variant, launch=launch, attributes={})
    assert predict(padded, hardware, "midlayer").t_total_ns == pytest.approx(
        predict(standalone, hardware, "midlayer").t_total_ns)


def test_piecewise_physical_extent_requires_the_full_graph_binding(tmp_path):
    import yaml

    document = yaml.safe_load(open("configs/hardware/metax_c500.yaml"))
    document["features"]["full_graph_physical_extent_dispatch"] = False
    path = tmp_path / "c500.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="requires it"):
        HWSpec(str(path))
