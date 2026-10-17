import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.launch_config.ascend_910b.vector_launch import (
    select_cann_public_rope_gather_v3_tiling,
)
from src.kernel.model_contract import (
    SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
    SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
)
from src.kernel.predictors.vector import (
    _cann_public_rope_index_select_duration_s,
)


HARDWARE = HWSpec("configs/hardware/ascend_910b.yaml")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _rope_kernel(tokens, head_dim=128):
    shape = {
        "seq_len": tokens,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": head_dim,
        "element_count": tokens * 40 * head_dim,
        "dtype_bytes": 2,
    }
    launch = select_launch_config(
        "ascend_910b", KernelKind.VECTOR, shape, "bf16", "rotary",
        {"launch_policy": "injected_launch"}, HARDWARE)
    return KernelSpec(
        KernelKind.VECTOR, HARDWARE.hardware_id, shape, dtype="bf16",
        variant="rotary", launch=launch)


def _graph_padded_rope_kernel(logical_tokens, physical_tokens):
    """Keep logical work in the spec and its captured launch physical."""

    physical = _rope_kernel(physical_tokens)
    logical_shape = {
        **physical.shape,
        "seq_len": logical_tokens,
        "element_count": logical_tokens * 40 * physical.shape["head_dim"],
        "elements": logical_tokens * 40 * physical.shape["head_dim"],
    }
    return KernelSpec(
        KernelKind.VECTOR, HARDWARE.hardware_id, logical_shape, dtype="bf16",
        variant="rotary", launch=physical.launch,
        attributes={
            "graph_logical_tokens": logical_tokens,
            "graph_model_input_tokens": physical_tokens,
            "graph_runtime_mode": "FULL",
            "model_input_shape": {
                name: value for name, value in physical.shape.items()
                if name not in {"dtype_bytes", "elements"}
            },
        })


def test_gather_v3_two_tque_lifecycle_reference_is_source_bound():
    reference_path = (
        REPOSITORY_ROOT / "microbench" / "ascend_910b" / "references"
        / "aiv_gather_v3_two_tque_lifecycle_cann_8_5_1.json")
    reference = json.loads(reference_path.read_text())
    source_path = (
        REPOSITORY_ROOT / "microbench" / "ascend_910b" / "operators"
        / "vector" / "gather_v3_two_tque_lifecycle_kernel.cpp")

    assert reference["service"] == "aiv_gather_v3_two_tque_lifecycle"
    assert reference["request_size_unit"] == (
        "source_two_tque_lifecycles")
    assert reference["service_time_ns_by_request_size"] == {"1": 50.0}
    assert reference["source_domain"] == {
        "tiling_key": 10330,
        "data_queue_type": "TQueBind<VECIN,VECOUT,1>",
        "data_queue_depth": 2,
        "data_queue_buffer_bytes": 512,
        "index_queue_type": "TQue<VECIN,1>",
        "index_queue_depth": 2,
        "index_queue_buffer_bytes": 8,
    }
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        reference["benchmark_source_sha256"][
            "gather_v3_two_tque_lifecycle_kernel.cpp"])
    assert reference["campaign"][
        "maximum_run_relative_median_spread"] <= 0.20
    assert reference["campaign"][
        "maximum_concurrency_relative_median_spread"] <= 0.20
    assert reference["production_or_evaluation_duration_consumed"] is False
    assert reference["mechanism_publishable"] is True


def test_public_rope_source_declares_exactly_two_index_selects():
    kernel = _rope_kernel(64)

    assert kernel.launch.parameters[
        SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER] == 2.0
    assert kernel.launch.parameters[
        SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER] == 10_330.0
    assert kernel.launch.parameters["cann_rope_gather_v3_real_core_num"] == 48.0
    assert (kernel.launch.parameters[
        "cann_rope_gather_v3_selected_row_bytes"] == 16_384.0)
    assert (kernel.launch.parameters[
        "cann_rope_gather_v3_tiling_payload_bytes"] == 192.0)

    invalid_launch = dataclasses.replace(
        kernel.launch,
        parameters={
            **kernel.launch.parameters,
            SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER: 1.0,
        },
    )
    with pytest.raises(ValueError, match="exactly two GatherV3"):
        dataclasses.replace(kernel, launch=invalid_launch)
    prediction = predict(kernel, HARDWARE)
    gather_s, gather_launch_s, gather_count = (
        _cann_public_rope_index_select_duration_s(kernel, HARDWARE))
    assert gather_count == 2
    assert (prediction.t_per_group["public_op_index_select_sequence"]
            == pytest.approx(2 * gather_s))
    assert prediction.t_launch_s >= 2 * gather_launch_s


def test_full_graph_padding_keeps_public_rope_index_selects():
    exact = _rope_kernel(10)
    padded = _graph_padded_rope_kernel(10, 16)

    exact_prediction = predict(exact, HARDWARE)
    padded_prediction = predict(padded, HARDWARE)

    assert "public_op_index_select_sequence" in padded_prediction.t_per_group
    assert (padded_prediction.t_per_group["public_op_index_select_sequence"]
            >= exact_prediction.t_per_group[
                "public_op_index_select_sequence"])


@pytest.mark.parametrize(
    "tokens", (1820, 1821, 2048, 2963, 3869, 8192),
)
def test_public_rope_uses_captured_direct_row_program(tokens):
    tiling = select_cann_public_rope_gather_v3_tiling(
        tokens, 128, 2, 48)

    assert tiling["tiling_key"] == 10_330
    assert tiling["real_core_num"] == 48
    assert tiling["selected_row_bytes"] == tokens * 128 * 2
    assert set(tiling) == {
        "tiling_key", "real_core_num", "selected_row_bytes",
        "tiling_payload_bytes",
    }
    kernel = _rope_kernel(tokens)
    assert kernel.launch.parameters[
        SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER] == 10_330.0
    assert (kernel.launch.parameters[
        "cann_rope_gather_v3_selected_row_bytes"] == tokens * 128 * 2)


@pytest.mark.parametrize(
    "tokens, head_dim",
    (
        (1, 64),
        (256, 128),
        (8192, 128),
    ),
)
def test_public_rope_prices_the_selected_gather_program(
        tokens, head_dim):
    kernel = _rope_kernel(tokens, head_dim)
    prediction = predict(kernel, HARDWARE)

    assert kernel.launch.parameters[
        SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER] == 10_330.0
    assert prediction.t_total_s > 0
    assert "public_op_index_select_sequence" in prediction.t_per_group
    assert (prediction.t_per_group["public_op_index_select_sequence"] > 0)


def test_public_rope_rejects_incomplete_gather_contract():
    kernel = _rope_kernel(64)
    missing_program = dataclasses.replace(
        kernel.launch,
        parameters={
            name: value
            for name, value in kernel.launch.parameters.items()
            if name != SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER
        },
    )
    with pytest.raises(ValueError, match="launch parameter schema mismatch"):
        dataclasses.replace(kernel, launch=missing_program)
