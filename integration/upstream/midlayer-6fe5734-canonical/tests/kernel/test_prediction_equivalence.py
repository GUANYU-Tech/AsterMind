"""Contract for the physical prediction classes a predictor may declare."""

import itertools
import random

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import (
    create_predictor,
    prediction_equivalence_attributes,
)
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel import lru_cache
from src.kernel.predictors.fused_moe import (
    canonical_expert_axis,
    prediction_equivalence_attributes as _declared_class,
)


def _moe_kernel(hardware, *, counts, experts, topk, tokens, variant,
                expert_parallel_size=1, expert_rank=0):
    owned_count = experts // expert_parallel_size
    owned = tuple(range(
        expert_rank * owned_count, expert_rank * owned_count + owned_count))
    local_counts = tuple(
        count if expert in owned else 0
        for expert, count in enumerate(counts))
    shape = {
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "hidden": 2048,
        "intermediate": 768,
        "dtype_bytes": 2,
    }
    attributes = {
        "tokens_per_expert": counts,
        "local_tokens_per_expert": local_counts,
        "expert_parallel_size": expert_parallel_size,
        "expert_rank": expert_rank,
        "owned_experts": owned,
        "expert_weight_tile_rows": None,
    }
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.FUSED_MOE, shape, "bf16", variant,
        attributes, hardware=hardware,
    )
    return KernelSpec(
        KernelKind.FUSED_MOE, hardware.hardware_id, shape, dtype="bf16",
        variant=variant, launch=launch, attributes=attributes,
    )


def _relabelled(counts, owned_positions, generator):
    """Permute the expert axis within the shard and within its complement."""

    shard = [counts[expert] for expert in sorted(owned_positions)]
    other = [
        counts[expert] for expert in range(len(counts))
        if expert not in owned_positions
    ]
    # Only empty experts may move inside the shard: the populated ones keep
    # the launch order that becomes the grouped M-tile program.
    populated = [count for count in shard if count]
    empty = [0] * (len(shard) - len(populated))
    positions = list(range(len(shard)))
    generator.shuffle(positions)
    shard_out = [None] * len(shard)
    empty_slots = sorted(generator.sample(
        positions, len(empty))) if empty else []
    for slot in empty_slots:
        shard_out[slot] = 0
    remaining = [slot for slot in range(len(shard)) if shard_out[slot] is None]
    for slot, count in zip(remaining, populated):
        shard_out[slot] = count
    generator.shuffle(other)
    result = []
    shard_iterator = iter(shard_out)
    other_iterator = iter(other)
    for expert in range(len(counts)):
        result.append(
            next(shard_iterator) if expert in owned_positions
            else next(other_iterator))
    return tuple(result)


@pytest.mark.parametrize("hardware_id", ("nvidia_a800", "metax_c500"))
def test_relabelled_expert_axis_shares_one_prediction(hardware_id):
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    generator = random.Random(20260921)
    experts, topk, tokens = 128, 8, 8
    owned_positions = frozenset(range(64))
    checked = 0
    for variant in ("w1", "w2"):
        for _ in range(12):
            counts = [0] * experts
            for _route in range(tokens * topk):
                counts[generator.randrange(experts)] += 1
            counts = tuple(counts)
            kernel = _moe_kernel(
                hardware, counts=counts, experts=experts, topk=topk,
                tokens=tokens, variant=variant, expert_parallel_size=2,
                expert_rank=0)
            relabelled_counts = _relabelled(
                counts, owned_positions, generator)
            if relabelled_counts == counts:
                continue
            relabelled = _moe_kernel(
                hardware, counts=relabelled_counts, experts=experts,
                topk=topk, tokens=tokens, variant=variant,
                expert_parallel_size=2, expert_rank=0)
            assert (prediction_equivalence_attributes(kernel)
                    == prediction_equivalence_attributes(relabelled))
            predictor = create_predictor(KernelKind.FUSED_MOE)
            assert (predictor.midlayer(kernel, hardware)
                    == predictor.midlayer(relabelled, hardware))
            checked += 1
    assert checked >= 8


def test_distinct_shard_routing_keeps_distinct_classes():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    base = _moe_kernel(
        hardware, counts=(4, 4, 0, 0, 8, 0, 0, 0), experts=8, topk=2,
        tokens=8, variant="w1", expert_parallel_size=2, expert_rank=0)
    reordered_shard = _moe_kernel(
        hardware, counts=(4, 0, 4, 0, 8, 0, 0, 0), experts=8, topk=2,
        tokens=8, variant="w1", expert_parallel_size=2, expert_rank=0)
    different_rows = _moe_kernel(
        hardware, counts=(5, 3, 0, 0, 8, 0, 0, 0), experts=8, topk=2,
        tokens=8, variant="w1", expert_parallel_size=2, expert_rank=0)
    different_outside = _moe_kernel(
        hardware, counts=(4, 4, 0, 0, 5, 3, 0, 0), experts=8, topk=2,
        tokens=8, variant="w1", expert_parallel_size=2, expert_rank=0)
    equivalence = prediction_equivalence_attributes
    assert equivalence(base) == equivalence(reordered_shard)
    assert equivalence(base) != equivalence(different_rows)
    assert equivalence(base) != equivalence(different_outside)


def test_equivalence_is_declared_only_where_a_predictor_owns_one():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"b": 1, "m": 64, "k": 128, "n": 256, "dtype_bytes": 2}
    gemm = KernelSpec(
        KernelKind.GEMM, "nvidia_a800", shape, dtype="bf16", variant="gemm",
        launch=select_launch_config(
            "nvidia_a800", KernelKind.GEMM, shape, "bf16", "gemm",
            {"launch_policy": "analytical_launch"}, hardware=hardware),
    )
    assert prediction_equivalence_attributes(gemm) is None
    with pytest.raises(TypeError):
        prediction_equivalence_attributes(object())


def test_empty_experts_do_not_change_the_moe_tile_replay():
    """The replay walks populated M tiles, so empty experts cannot move it."""

    geometry = dict(
        n_tiles=6, group_size_m=4, k_extent=2048, block_k=64,
        n_extent=768, block_n=128, element_bytes=2, c_row_bytes=256,
        block_m=32, wave_size=64, capacity_bytes=40 * 1024 * 1024,
        residency_profile=(),
    )
    generator = random.Random(4242)
    checked = 0
    for _ in range(24):
        populated = tuple(
            generator.choice((1, 7, 32, 33, 96))
            for _ in range(generator.randrange(2, 7)))
        padded = list(populated)
        for _ in range(generator.randrange(1, 40)):
            padded.insert(generator.randrange(len(padded) + 1), 0)
        m_tiles = sum(
            -(-count // geometry["block_m"]) for count in populated)
        lru_cache._moe_tile_program_cache.clear()
        dense = lru_cache.simulate_moe_tile_program(
            m_tiles=m_tiles, tokens_per_expert=populated, **geometry)
        lru_cache._moe_tile_program_cache.clear()
        padded_result = lru_cache.simulate_moe_tile_program(
            m_tiles=m_tiles, tokens_per_expert=tuple(padded), **geometry)
        assert dense == padded_result
        checked += 1
    assert checked == 24


def test_moe_tile_replay_cache_collapses_empty_expert_padding():
    geometry = dict(
        m_tiles=4, n_tiles=6, group_size_m=4, k_extent=2048, block_k=64,
        n_extent=768, block_n=128, element_bytes=2, c_row_bytes=256,
        block_m=32, wave_size=64, capacity_bytes=40 * 1024 * 1024,
        residency_profile=(),
    )
    lru_cache._moe_tile_program_cache.clear()
    for padding in itertools.product((0,), repeat=3):
        lru_cache.simulate_moe_tile_program(
            tokens_per_expert=(32, 32, 32, 32) + padding, **geometry)
        lru_cache.simulate_moe_tile_program(
            tokens_per_expert=padding + (32, 32, 32, 32), **geometry)
    assert len(lru_cache._moe_tile_program_cache) == 1


def test_canonical_expert_axis_conserves_the_routing_it_relabels():
    generator = random.Random(90210)
    experts, owned = 32, tuple(range(16, 32))
    for _ in range(60):
        counts = [0] * experts
        for _route in range(generator.randrange(1, 90)):
            counts[generator.randrange(experts)] += 1
        counts = tuple(counts)
        canonical, canonical_local = canonical_expert_axis(counts, owned)
        assert len(canonical) == experts
        assert sum(canonical) == sum(counts)
        assert sorted(canonical) == sorted(counts)
        assert canonical_local == tuple(
            count if expert in owned else 0
            for expert, count in enumerate(canonical))
        assert sorted(canonical_local) == sorted(
            count if expert in owned else 0
            for expert, count in enumerate(counts))
        # The shard's populated counts keep the launch order that becomes the
        # grouped M-tile program.
        assert (
            tuple(counts[expert] for expert in owned if counts[expert])
            == tuple(count for count in canonical_local if count))
        assert canonical_expert_axis(canonical, owned) == (
            canonical, canonical_local)


def test_canonical_axis_and_declared_class_agree_on_the_partition():
    generator = random.Random(511)
    experts, owned = 32, tuple(range(16, 32))
    by_axis = {}
    by_class = {}
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    for index in range(80):
        counts = [0] * experts
        for _route in range(generator.randrange(1, 40)):
            counts[generator.randrange(experts)] += 1
        counts = tuple(counts)
        axis = canonical_expert_axis(counts, owned)
        kernel = _moe_kernel(
            hardware, counts=counts, experts=experts, topk=1,
            tokens=sum(counts), variant="w1", expert_parallel_size=2,
            expert_rank=1)
        declared = prediction_equivalence_attributes(kernel)
        assert declared is not None
        token = repr(sorted(declared.items(), key=lambda item: item[0]))
        by_axis.setdefault(axis, set()).add(index)
        by_class.setdefault(token, set()).add(index)
    assert sorted(by_axis.values(), key=sorted) == sorted(
        by_class.values(), key=sorted)
