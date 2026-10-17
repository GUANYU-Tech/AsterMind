"""Per-MoE-layer realization of an imported expert-frequency profile."""

import dataclasses

import pytest

from src.serving.workload.expert_frequency import load_expert_frequency_profile
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    validate_expert_routing_provenance,
)

_SHA = "0" * 64


def _profile():
    return load_expert_frequency_profile(
        "serving_dataset/moe_expert_freq/manifest.json",
        "Qwen3-30B-A3B",
        expected_model_id="Qwen3-30B-A3B",
        expected_moe_layers=48,
        expected_num_experts=128,
    )


def _resolve(routing, *, step_id, layer_index, ordinal, tokens=64):
    invocation = (f"{step_id}:layer-{layer_index}:moe"
                  if layer_index is not None
                  else f"{step_id}:mtp-pass-0:moe")
    return routing.resolve(
        step_id=step_id, layer_index=layer_index,
        moe_invocation_id=invocation, schedule_step_sha256=_SHA,
        tokens=tokens, num_experts=128, topk=8, model_config_sha256=_SHA,
        pool_name="main", tensor_parallel_size=1, expert_parallel_size=1,
        expert_rank_group=(0,), expert_placement_strategy="linear",
        moe_layer_ordinal=ordinal)


def test_each_moe_layer_draws_from_its_own_row():
    profile = _profile()
    routing = ExpertRoutingInput.profile_observation(
        profile, per_moe_layer=True)
    first = _resolve(routing, step_id="s0", layer_index=0, ordinal=0)
    last = _resolve(routing, step_id="s0", layer_index=47, ordinal=47)
    assert sum(first.tokens_per_expert) == 64 * 8
    assert first.tokens_per_expert != last.tokens_per_expert
    # Experts with zero probability in a layer's row are never selected.
    for resolution, row in ((first, 0), (last, 47)):
        for count, weight in zip(resolution.tokens_per_expert,
                                 profile.layer_probabilities[row]):
            assert weight > 0 or count == 0
    provenance = validate_expert_routing_provenance(
        first.routing_provenance, tokens_per_expert=first.tokens_per_expert,
        tokens=64, num_experts=128, topk=8,
        moe_invocation_id="s0:layer-0:moe")
    assert provenance["distribution"] == "model_profile_per_moe_layer_v1"
    assert provenance["sharing_scope"] == "model_moe_layer_token_count"


def test_histogram_is_shared_across_steps_with_equal_signature():
    routing = ExpertRoutingInput.profile_observation(
        _profile(), per_moe_layer=True)
    a = _resolve(routing, step_id="s0", layer_index=3, ordinal=3)
    b = _resolve(routing, step_id="s1", layer_index=3, ordinal=3)
    assert a.tokens_per_expert == b.tokens_per_expert


def test_layer_mean_realization_is_unchanged_by_the_ordinal():
    routing = ExpertRoutingInput.profile_observation(_profile())
    a = _resolve(routing, step_id="s0", layer_index=3, ordinal=3)
    b = _resolve(routing, step_id="s0", layer_index=3, ordinal=None)
    assert a.tokens_per_expert == b.tokens_per_expert
    assert a.routing_provenance["distribution"] == "model_profile_layer_mean_v1"


def test_per_layer_requires_an_ordinal_in_range():
    routing = ExpertRoutingInput.profile_observation(
        _profile(), per_moe_layer=True)
    with pytest.raises(TypeError):
        _resolve(routing, step_id="s0", layer_index=3, ordinal=None)
    with pytest.raises(ValueError, match="exceeds the profile layers"):
        _resolve(routing, step_id="s0", layer_index=3, ordinal=48)
    mtp = _resolve(routing, step_id="s0", layer_index=None, ordinal=None)
    assert sum(mtp.tokens_per_expert) == 64 * 8


def test_mismatched_algorithm_fails_closed():
    routing = ExpertRoutingInput.profile_observation(
        _profile(), per_moe_layer=True)
    with pytest.raises(ValueError, match="sampling algorithm"):
        dataclasses.replace(
            routing,
            algorithm="sha256_profile_weighted_topk_without_replacement_"
                      "model_step_v1")
