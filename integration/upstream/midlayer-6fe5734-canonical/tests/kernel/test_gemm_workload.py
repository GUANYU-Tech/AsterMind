import json

import pytest

from src.kernel.launch_config.gemm_workload import (
    GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    GEMM_WORKLOAD_KEY_FIELDS,
    gemm_launch_selection_key,
    gemm_launch_selection_key_from_workload_key,
    gemm_physical_operation_key,
    gemm_workload_key,
    validate_gemm_workload_key,
)


def _attributes(**overrides):
    values = {
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": "qkv_proj",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    values.update(overrides)
    return values


def test_gemm_workload_key_is_canonical_and_keeps_batch_separate():
    shape = {"b": 2, "m": 3, "n": 4, "k": 5}
    key = gemm_workload_key(shape, "bf16", "gemm", _attributes())

    assert len(json.loads(key)) == len(GEMM_WORKLOAD_KEY_FIELDS) == 16
    validate_gemm_workload_key(key)
    assert key != gemm_workload_key(
        {"b": 1, "m": 6, "n": 4, "k": 5},
        "bf16", "gemm", _attributes())


def test_gemm_physical_operation_key_projects_selector_inputs():
    key = gemm_workload_key(
        {"b": 1, "m": 16, "n": 64, "k": 32},
        "bf16", "gemm", _attributes(has_bias=True, epilogue="bias"))

    assert gemm_physical_operation_key(key) == '[16,64,32,"bf16",true]'


def test_gemm_launch_selection_key_excludes_semantic_context():
    shape = {"b": 1, "m": 16, "n": 64, "k": 32}
    baseline_attributes = _attributes()
    changed_attributes = _attributes(
        position="down_proj", tensor_parallel_size=2,
        expert_parallel_size=2)

    baseline = gemm_launch_selection_key(
        shape, "bf16", "gemm", baseline_attributes)
    changed = gemm_launch_selection_key(
        shape, "bf16", "gemm", changed_attributes)
    semantic = gemm_workload_key(
        shape, "bf16", "gemm", baseline_attributes)

    assert len(baseline) == len(GEMM_LAUNCH_SELECTION_KEY_FIELDS) == 13
    assert baseline == changed
    assert baseline == gemm_launch_selection_key_from_workload_key(semantic)


@pytest.mark.parametrize(
    "field,value",
    [
        ("position", "o_proj"),
        ("trans_b", False),
        ("tensor_parallel_size", 2),
        ("expert_parallel_size", 2),
    ],
)
def test_gemm_workload_key_separates_semantic_fields(field, value):
    shape = {"b": 1, "m": 16, "n": 64, "k": 32}
    baseline = gemm_workload_key(
        shape, "bf16", "gemm", _attributes())
    changed = gemm_workload_key(
        shape, "bf16", "gemm", _attributes(**{field: value}))
    assert changed != baseline


def test_gemm_workload_key_requires_every_attribute():
    attributes = _attributes()
    del attributes["trans_b"]
    with pytest.raises(ValueError, match="missing attributes.*trans_b"):
        gemm_workload_key(
            {"b": 1, "m": 16, "n": 64, "k": 32},
            "bf16", "gemm", attributes)


def test_gemm_workload_key_rejects_bias_epilogue_disagreement():
    with pytest.raises(ValueError, match="has_bias conflicts"):
        gemm_workload_key(
            {"b": 1, "m": 16, "n": 64, "k": 32},
            "bf16", "gemm", _attributes(has_bias=True))


def test_gemm_workload_key_rejects_generic_position():
    with pytest.raises(ValueError, match="identify the logical operation"):
        gemm_workload_key(
            {"b": 1, "m": 16, "n": 64, "k": 32},
            "bf16", "gemm", _attributes(position="generic"))
