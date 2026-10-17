"""Fail-closed mirror contract for the helper/glue launch emission seam.

Every test here mirrors one rule of the helper-launch contract: counts come
from a vLLM source predicate or the versioned discrete emission observation,
explicit zeros stay authorized, and every uncovered or malformed input
fails closed instead of falling back.
"""

from pathlib import Path

import pytest
import yaml

from src.serving.workload import helper_launch
from src.serving.workload.helper_launch import (
    BOUNDARY_LOGITS_GATHER_SOURCES,
    MOE_ROUTING_MASK_EMISSIONS,
    OBSERVATION_PATH,
    RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED,
    ROUTER_MASK_RUN_CLASS,
    ROUTER_MASK_SOURCES,
    ROUTER_MASK_STRING_CALL_POINTS,
    TOKEN_EMBEDDING_SOURCES,
    _load_observation,
    boundary_logits_gather_source,
    helper_emission_count,
    moe_routing_mask_emissions,
    moe_routing_mask_string_complete,
    router_mask_source,
)


def _entry(hardware_id, call_point, phase, model_type, observed_count,
           authorization="authorized for the fail-closed mirror contract"):
    return {
        "key": {
            "hardware_id": hardware_id,
            "call_point": call_point,
            "phase": phase,
            "model_type": model_type,
        },
        "observed_count": observed_count,
        "authorization": authorization,
        "evidence": "tests/serving/workload/test_helper_launch.py",
    }


def _document(items, version=1):
    return {"version": version, "items": list(items)}


@pytest.fixture
def forged_observation(tmp_path, monkeypatch):
    """Point the loader at a forged document and reset its cache."""

    def install(items, version=1):
        path = tmp_path / "forged_observation.yaml"
        path.write_text(yaml.safe_dump(_document(items, version=version)))
        monkeypatch.setattr(helper_launch, "OBSERVATION_PATH", path)
        monkeypatch.setattr(helper_launch, "_observation_cache", None)
        return path

    yield install
    monkeypatch.setattr(helper_launch, "_observation_cache", None)


def test_registered_observation_file_validates_and_covers_its_entries():
    table = _load_observation()
    assert set(table) == {
        ("nvidia_a800", "token_embedding_gather", "prefill"),
        ("ascend_910b", "attention_output_writeback", "decode"),
        ("ascend_910b", "attention_output_writeback", "prefill"),
        ("ascend_910b", "qkv_v_materialization", "prefill"),
    }
    for scoped in table.values():
        for item in scoped.values():
            assert isinstance(item["observed_count"], int)
            assert item["authorization"].strip()


def test_observation_counts_win_over_the_source_predicate():
    assert helper_emission_count(
        "nvidia_a800", "token_embedding_gather", "prefill",
        source_count=1, model_type="qwen3") == 0
    assert helper_emission_count(
        "ascend_910b", "attention_output_writeback", "decode",
        source_count=1, model_type="qwen3_moe") == 0
    assert helper_emission_count(
        "ascend_910b", "attention_output_writeback", "prefill",
        source_count=1, model_type="qwen3_moe") == 0
    assert helper_emission_count(
        "ascend_910b", "qkv_v_materialization", "prefill",
        source_count=0, model_type="qwen3") == 1


def test_model_scoped_entry_overrides_only_its_model_type():
    # The 910B prefill write-back zero is scoped to qwen3_moe; the dense
    # qwen3 model type keeps the source predicate count.
    assert helper_emission_count(
        "ascend_910b", "attention_output_writeback", "prefill",
        source_count=1, model_type="qwen3") == 1
    assert helper_emission_count(
        "ascend_910b", "attention_output_writeback", "prefill",
        source_count=3, model_type="qwen3") == 3


def test_uncovered_call_point_keeps_the_source_predicate_count():
    assert helper_emission_count(
        "metax_c500", "token_embedding_gather", "prefill",
        source_count=1, model_type="qwen3") == 1
    assert helper_emission_count(
        "nvidia_a800", "token_embedding_gather", "decode",
        source_count=1, model_type="llama") == 1


def test_observation_required_call_point_fails_without_an_entry(
        forged_observation):
    forged_observation([])
    with pytest.raises(ValueError, match="requires a versioned observation"):
        helper_emission_count(
            "nvidia_a800", "token_embedding_gather", "prefill",
            source_count=1, model_type="qwen3")
    # The same call point in an unregistered phase stays on the source
    # predicate; only the registered divergence is mandatory.
    assert helper_emission_count(
        "nvidia_a800", "token_embedding_gather", "decode",
        source_count=1, model_type="qwen3") == 1


def test_any_plus_model_scoped_entry_is_ambiguous(forged_observation):
    forged_observation([
        _entry("ascend_910b", "attention_output_writeback", "prefill",
               "any", 0),
        _entry("ascend_910b", "attention_output_writeback", "prefill",
               "qwen3_moe", 0),
    ])
    with pytest.raises(ValueError, match="ambiguous"):
        helper_emission_count(
            "ascend_910b", "attention_output_writeback", "prefill",
            source_count=1, model_type="qwen3_moe")


@pytest.mark.parametrize(
    "items,version,match",
    (
        (
            [_entry("nvidia_a800", "token_embedding_gather", "prefill",
                    "any", 0, authorization=" ")],
            1,
            "authorization must be a non-empty string",
        ),
        (
            [_entry("nvidia_a800", "token_embedding_gather", "prefill",
                    "any", -1)],
            1,
            "observed_count must be a non-negative integer",
        ),
        (
            [_entry("nvidia_a800", "token_embedding_gather", "prefill",
                    "any", True)],
            1,
            "observed_count must be a non-negative integer",
        ),
        (
            [_entry("nvidia_a800", "token_embedding_gather", "monsoon",
                    "any", 0)],
            1,
            "unknown helper emission observation phase",
        ),
        (
            [
                _entry("nvidia_a800", "token_embedding_gather", "prefill",
                       "any", 0),
                _entry("nvidia_a800", "token_embedding_gather", "prefill",
                       "any", 1),
            ],
            1,
            "duplicate helper emission observation item",
        ),
        ([], 2, "unsupported helper emission observation schema version"),
    ),
)
def test_observation_document_validates_exactly(
        forged_observation, items, version, match):
    forged_observation(items, version=version)
    with pytest.raises(ValueError, match=match):
        helper_emission_count(
            "nvidia_a800", "token_embedding_gather", "prefill",
            source_count=1, model_type="qwen3")


def test_observation_document_rejects_unknown_and_missing_fields(
        forged_observation):
    base = {
        "key": {
            "hardware_id": "nvidia_a800",
            "call_point": "token_embedding_gather",
            "phase": "prefill",
            "model_type": "any",
        },
        "observed_count": 0,
        "authorization": "authorized",
        "evidence": "mirror contract",
    }
    cases = (
        {**base, "extra_field": 1},
        {key: value for key, value in base.items() if key != "authorization"},
        {**base, "key": {**base["key"], "extra_key": "x"}},
        {**base, "key": {
            field: value for field, value in base["key"].items()
            if field != "phase"
        }},
    )
    for item in cases:
        forged_observation([item])
        with pytest.raises(ValueError, match="must be exactly"):
            helper_emission_count(
                "nvidia_a800", "token_embedding_gather", "prefill",
                source_count=1, model_type="qwen3")


@pytest.mark.parametrize(
    "hardware_id,phase,model_type,source_count,match",
    (
        ("metax_c501", "prefill", "qwen3", 1, "unknown hardware id"),
        ("nvidia_a800", "monsoon", "qwen3", 1, "unknown emission phase"),
        ("nvidia_a800", "prefill", "", 1, "non-empty string"),
        ("nvidia_a800", "prefill", "qwen3", -1, "non-negative integer"),
        ("nvidia_a800", "prefill", "qwen3", True, "non-negative integer"),
    ),
)
def test_lookup_arguments_validate_before_any_resolution(
        hardware_id, phase, model_type, source_count, match):
    with pytest.raises(ValueError, match=match):
        helper_emission_count(
            hardware_id, "token_embedding_gather", phase,
            source_count=source_count, model_type=model_type)


def test_routing_mask_emission_dispatch_is_exhaustive_per_card():
    assert moe_routing_mask_emissions("nvidia_a800") == (
        "moe_expert_map_index",)
    assert moe_routing_mask_emissions("metax_c500") == ()
    assert moe_routing_mask_emissions("ascend_910b") == (
        "router_mask_index", "router_mask_notequal", "router_mask_cast",
        "router_mask_mul")
    with pytest.raises(ValueError, match="unknown hardware id"):
        moe_routing_mask_emissions("metax_c501")
    assert set(MOE_ROUTING_MASK_EMISSIONS) == set(TOKEN_EMBEDDING_SOURCES)


def test_string_complete_is_the_910b_four_kernel_string_only():
    # Helper contract §2.2: only the 910B emission set forms the whole
    # Index->NotEqual->Cast->Mul string the trace-side counter observes, so
    # only that card may claim a router run count.
    assert moe_routing_mask_string_complete("ascend_910b")
    assert not moe_routing_mask_string_complete("nvidia_a800")
    assert not moe_routing_mask_string_complete("metax_c500")
    with pytest.raises(ValueError, match="unknown hardware id"):
        moe_routing_mask_string_complete("metax_c501")


def test_string_call_points_are_exactly_the_910b_string_members():
    assert ROUTER_MASK_STRING_CALL_POINTS == frozenset((
        "router_mask_index", "router_mask_notequal", "router_mask_cast",
        "router_mask_mul"))
    assert ROUTER_MASK_STRING_CALL_POINTS == frozenset(
        moe_routing_mask_emissions("ascend_910b"))


def test_router_run_class_literal_is_the_stable_single_literal():
    # The G1 harness counts the trace-side string run under this exact
    # class, so the literal is the contract.
    assert ROUTER_MASK_RUN_CLASS == "class2_router_run"


def test_source_tables_cover_exactly_the_three_cards_and_h200():
    # H200 runs the in-tree CUDA vLLM path and carries A800's sources.
    hardware_ids = {"nvidia_a800", "metax_c500", "ascend_910b", "nvidia_h200"}
    for table in (BOUNDARY_LOGITS_GATHER_SOURCES, ROUTER_MASK_SOURCES):
        assert set(table) == hardware_ids
        assert table["nvidia_h200"] == table["nvidia_a800"]
    for hardware_id in hardware_ids:
        for resolver in (
                boundary_logits_gather_source, router_mask_source):
            assert resolver(hardware_id)
    for resolver in (
            boundary_logits_gather_source, router_mask_source):
        with pytest.raises(ValueError, match="unknown hardware id"):
            resolver("metax_c501")


def test_residual_anchor_marker_is_a_stable_single_literal():
    # The lowering-side AF boundary validation admits a no-predecessor
    # residual only with this exact marker, so the literal is the contract.
    assert RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED == (
        "observation_suppressed_step_input")
