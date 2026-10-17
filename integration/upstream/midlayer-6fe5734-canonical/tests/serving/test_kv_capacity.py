import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.kernel.hardware_spec import HWSpec
from src.serving.backend.vllm_scheduler_process import (
    vllm_scheduler_source_identity,
)
from src.serving.kv_cache.kv_capacity import (
    RuntimeMemoryPolicy,
    RuntimeMemoryPolicyBinding,
    kv_capacity_manifest,
    load_runtime_memory_policy,
    resolve_analytical_kv_capacity_plan,
    resolve_debug_kv_capacity_plan,
)
from src.serving.kv_cache.model_residency import model_rank_residencies
from src.serving.kv_cache.vllm_allocation import (
    VLLMKVAllocation,
    allocate_vllm_kv_cache,
)
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.workload.model_config import load_model_config


_MODEL_FIXTURE = Path("tests/serving/fixtures/qwen3_scheduler")
_HARDWARE_FIXTURE = Path("configs/hardware/nvidia_a800.yaml")
_SOURCE_SHA256 = vllm_scheduler_source_identity()[
    "package_python_manifest_sha256"]
_SOURCE_COMMIT = vllm_scheduler_source_identity()["source_commit"]
_RESERVE_KINDS = (
    "activation",
    "cuda_graph",
    "communication",
    "allocator",
    "fragmentation",
    "framework",
)


def _provenance():
    return {
        "source_kind": "runtime_configuration",
        "source_reference": "unit-test runtime policy",
        "source_sha256": None,
    }


def _reserve_bytes_for_kind(kind, activation_bytes):
    if kind == "activation":
        return activation_bytes
    if kind in set(_RESERVE_KINDS) - {"activation"}:
        return 0
    raise ValueError(f"unknown runtime reserve kind: {kind!r}")


def _policy_entry(pool_name="main", reserve_bytes=0):
    model = load_model_config(str(_MODEL_FIXTURE))
    return {
        "pool_name": pool_name,
        "binding": {
            "hardware_id": "nvidia_a800",
            "hardware_config_sha256": hashlib.sha256(
                _HARDWARE_FIXTURE.read_bytes()).hexdigest(),
            "model_config_sha256": model.config_sha256,
            "deployment": "colocated",
            "worker_role": "colocated",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "max_model_len": 128,
            "max_batched_tokens": 128,
            "max_requests": 1,
            "kv_block_size_tokens": 16,
        },
        "gpu_memory_utilization": {
            "numerator": 9,
            "denominator": 10,
            "provenance": _provenance(),
        },
        "non_kv_reserves": [
            {
                "kind": kind,
                "bytes": _reserve_bytes_for_kind(kind, reserve_bytes),
                "provenance": _provenance(),
            }
            for kind in _RESERVE_KINDS
        ],
        "reserve_scope": "verified_uniform_across_pool_ranks",
        "completeness_assertion": "all_non_kv_reserves_for_bound_runtime",
    }


def _write_policy(tmp_path, entries):
    path = tmp_path / "runtime-memory-policy.json"
    payload = json.dumps({
        "schema_version": 2,
        "scheduler_pools": entries,
    }, sort_keys=True).encode()
    path.write_bytes(payload)
    return path, payload


def _fake_allocator(
    model,
    parallelism,
    *,
    kv_block_size_tokens,
    max_model_len,
    available_memory_by_rank,
):
    page_bytes = (
        kv_block_size_tokens
        * 2
        * model.kv_heads_per_tensor_parallel_rank(
            parallelism.tensor_parallel_size)
        * model.head_dim
        * model.dtype_bytes
    )
    local_layers = model.num_hidden_layers // parallelism.pipeline_parallel_size
    num_blocks = min(available_memory_by_rank) // (page_bytes * local_layers)
    return VLLMKVAllocation(
        version="0.11.0",
        source_commit=_SOURCE_COMMIT,
        package_python_manifest_sha256=_SOURCE_SHA256,
        allocator_symbol=(
            "vllm.v1.core.kv_cache_utils.get_kv_cache_configs"),
        scheduler_config_symbol=(
            "vllm.v1.core.kv_cache_utils."
            "generate_scheduler_kv_cache_config"),
        kv_spec_type="FullAttentionSpec",
        kv_block_size_tokens=kv_block_size_tokens,
        max_model_len=max_model_len,
        kv_page_size_bytes_per_layer=page_bytes,
        num_kv_blocks=num_blocks,
        worker_num_kv_blocks=tuple(
            num_blocks for unused in available_memory_by_rank),
        worker_allocated_tensor_bytes=tuple(
            num_blocks * page_bytes * local_layers
            for unused in available_memory_by_rank),
        scheduler_tensor_owners=tuple(
            (f"model.layers.{index}.self_attn.attn",)
            for index in range(local_layers)
        ),
    )


def test_analytical_capacity_uses_hwspec_model_policy_and_native_result(tmp_path):
    path, payload = _write_policy(tmp_path, [_policy_entry(reserve_bytes=4096)])
    policy_config = load_runtime_memory_policy(path)
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config(str(_MODEL_FIXTURE))

    plan = resolve_analytical_kv_capacity_plan(
        "main",
        hardware=hardware,
        hardware_config_sha256=hashlib.sha256(
            _HARDWARE_FIXTURE.read_bytes()).hexdigest(),
        model=model,
        parallelism=ParallelConfig(),
        deployment="colocated",
        worker_role="colocated",
        kv_block_size_tokens=16,
        max_model_len=128,
        max_batched_tokens=128,
        max_requests=1,
        policy=policy_config.policy_for_pool("main"),
        allocator=_fake_allocator,
    )
    manifest = kv_capacity_manifest((plan,), policy_config)

    rank = plan.rank_budgets[0]
    assert rank.physical_hbm_bytes == 80 * 1024**3
    assert rank.utilized_hbm_bytes == 72 * 1024**3
    assert rank.non_kv_reserve_bytes == 4096
    assert rank.model_weight_resident_bytes > 0
    assert plan.kv_block_bytes == 4096
    assert plan.num_kv_blocks == rank.available_kv_bytes // 4096
    assert manifest["source"] == {
        "mode": "analytical",
        "runtime_memory_policy_path": str(path.resolve()),
        "runtime_memory_policy_sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert manifest["scheduler_pools"][0]["native_vllm_allocation"][
        "allocator_symbol"
    ] == "vllm.v1.core.kv_cache_utils.get_kv_cache_configs"


def test_changing_only_hbm_capacity_changes_derived_block_count(tmp_path):
    path, unused_payload = _write_policy(tmp_path, [_policy_entry()])
    del unused_payload
    policy = load_runtime_memory_policy(path).policy_for_pool("main")
    model = load_model_config(str(_MODEL_FIXTURE))
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    first = resolve_analytical_kv_capacity_plan(
        "main",
        hardware=hardware,
        hardware_config_sha256=hashlib.sha256(
            _HARDWARE_FIXTURE.read_bytes()).hexdigest(),
        model=model,
        parallelism=ParallelConfig(),
        deployment="colocated",
        worker_role="colocated",
        kv_block_size_tokens=16,
        max_model_len=128,
        max_batched_tokens=128,
        max_requests=1,
        policy=policy,
        allocator=_fake_allocator,
    )
    hardware.resource_by_role("dram").size_bytes -= 1024**3
    second = resolve_analytical_kv_capacity_plan(
        "main",
        hardware=hardware,
        hardware_config_sha256=hashlib.sha256(
            _HARDWARE_FIXTURE.read_bytes()).hexdigest(),
        model=model,
        parallelism=ParallelConfig(),
        deployment="colocated",
        worker_role="colocated",
        kv_block_size_tokens=16,
        max_model_len=128,
        max_batched_tokens=128,
        max_requests=1,
        policy=policy,
        allocator=_fake_allocator,
    )

    assert second.num_kv_blocks < first.num_kv_blocks


def test_static_residency_tracks_model_size_tp_pp_and_ep():
    pipeline_model = load_model_config(
        "tests/serving/fixtures/qwen3_pp2_scheduler")
    dense = load_model_config(
        "tests/serving/fixtures/qwen3_tp2_scheduler")
    tp1 = model_rank_residencies(
        dense, ParallelConfig(), worker_role="colocated")
    tp2 = model_rank_residencies(
        dense,
        ParallelConfig(tensor_parallel_size=2),
        worker_role="colocated",
    )
    attention_only = model_rank_residencies(
        dense,
        ParallelConfig(tensor_parallel_size=2),
        worker_role="attention",
    )
    pp2 = model_rank_residencies(
        pipeline_model,
        ParallelConfig(pipeline_parallel_size=2),
        worker_role="colocated",
    )
    assert tp2[0].weight_resident_bytes < tp1[0].weight_resident_bytes
    assert attention_only[0].weight_resident_bytes < tp2[0].weight_resident_bytes
    assert pp2[0].weight_resident_bytes != pp2[-1].weight_resident_bytes

    moe = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    without_ep = model_rank_residencies(
        moe,
        ParallelConfig(tensor_parallel_size=2),
        worker_role="colocated",
    )
    with_ep = model_rank_residencies(
        moe,
        ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
        ),
        worker_role="colocated",
    )
    assert with_ep[0].expert_parallel_rank == 0
    assert with_ep[1].expert_parallel_rank == 1
    assert with_ep[0].weight_resident_bytes >= without_ep[0].weight_resident_bytes


def test_static_residency_matches_exact_qwen3_weights_and_vocab_padding():
    model = load_model_config(str(_MODEL_FIXTURE))
    rank = model_rank_residencies(
        model, ParallelConfig(), worker_role="colocated")[0]
    assert (
        rank.embedding_bytes,
        rank.decoder_layer_bytes,
        rank.final_norm_bytes,
        rank.lm_head_bytes,
        rank.weight_resident_bytes,
    ) == (8192, 82432, 128, 8192, 98944)

    padded_model = replace(model, vocab_size=65)
    padded_rank = model_rank_residencies(
        padded_model, ParallelConfig(), worker_role="colocated")[0]
    assert padded_rank.embedding_bytes == 128 * model.hidden_size * 2
    assert padded_rank.lm_head_bytes == 128 * model.hidden_size * 2


def test_qwen3_moe_residency_rejects_unmodeled_parallel_layouts():
    model = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    with pytest.raises(NotImplementedError, match="does not yet model"):
        model_rank_residencies(
            model,
            ParallelConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                enable_expert_parallel=True,
            ),
            worker_role="colocated",
        )
    with pytest.raises(NotImplementedError, match="TP <= num_experts"):
        model_rank_residencies(
            replace(model, num_experts=1),
            ParallelConfig(tensor_parallel_size=2),
            worker_role="colocated",
        )
    with pytest.raises(NotImplementedError, match="divisible by TP"):
        model_rank_residencies(
            replace(model, moe_intermediate_size=127),
            ParallelConfig(tensor_parallel_size=2),
            worker_role="colocated",
        )


def test_qwen3_moe_loader_rejects_noncanonical_sparse_layer_policy(tmp_path):
    source = Path(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler/config.json")
    config = json.loads(source.read_text())
    for field, value, message in (
            ("decoder_sparse_step", 2, "decoder_sparse_step=1"),
            ("mlp_only_layers", [0], "empty mlp_only_layers")):
        changed = dict(config)
        changed[field] = value
        model_path = tmp_path / field
        model_path.mkdir()
        (model_path / "config.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            load_model_config(str(model_path))


def test_native_allocation_contract_rejects_unpinned_identity():
    model = load_model_config(str(_MODEL_FIXTURE))
    allocation = _fake_allocator(
        model,
        ParallelConfig(),
        kv_block_size_tokens=16,
        max_model_len=128,
        available_memory_by_rank=(4096 * 9,),
    )
    with pytest.raises(ValueError, match="version disagrees"):
        replace(allocation, version="0.11.1")


def test_runtime_policy_requires_all_dimensioned_reserves_and_provenance(tmp_path):
    missing = _policy_entry()
    missing["non_kv_reserves"].pop()
    path, unused_payload = _write_policy(tmp_path, [missing])
    del unused_payload
    with pytest.raises(ValueError, match="each non-KV reserve kind"):
        load_runtime_memory_policy(path)

    bad_provenance = _policy_entry()
    del bad_provenance["non_kv_reserves"][0]["provenance"]["source_sha256"]
    path, unused_payload = _write_policy(tmp_path, [bad_provenance])
    del unused_payload
    with pytest.raises(ValueError, match="fields must be exactly"):
        load_runtime_memory_policy(path)


def test_runtime_policy_rejects_duplicate_json_fields(tmp_path):
    path = tmp_path / "runtime-memory-policy.json"
    path.write_text('{"schema_version":2,"schema_version":2,"scheduler_pools":[]}')
    with pytest.raises(ValueError, match="duplicate runtime memory policy"):
        load_runtime_memory_policy(path)


def test_debug_capacity_is_explicitly_separate_from_analytical_path():
    plan = resolve_debug_kv_capacity_plan(
        "main",
        kv_block_size_tokens=16,
        kv_bytes_per_token=128,
        num_kv_blocks=4,
    )
    assert plan.source_mode == "debug_explicit_block_count"
    assert plan.num_kv_blocks == 4
    assert kv_capacity_manifest((plan,), None)["source"]["mode"] == (
        "debug_explicit_block_count")


def test_target_model_residency_supports_unsharded_experts_for_pure_pp():
    model = load_model_config(
        "tests/serving/fixtures/kimi_k2_6_official")
    residencies = model_rank_residencies(
        model,
        ParallelConfig(pipeline_parallel_size=16),
        worker_role="colocated",
    )

    assert len(residencies) == 16
    assert max(rank.weight_resident_bytes for rank in residencies) == (
        39_234_181_120)
    assert all(rank.expert_parallel_rank == 0 for rank in residencies)


def test_standard_gqa_kv_bytes_preserve_replicated_heads_for_large_tp():
    model = load_model_config(str(_MODEL_FIXTURE))
    assert model.kv_heads_per_tensor_parallel_rank(1) == 1
    assert model.kv_heads_per_tensor_parallel_rank(2) == 1
    assert model.kv_heads_per_tensor_parallel_rank(3) == 1
    assert model.kv_bytes_per_token_per_tensor_parallel_rank(1) == 256
    assert model.kv_bytes_per_token_per_tensor_parallel_rank(2) == 256


def test_pp2_kv_bytes_are_stage_local_for_uniform_layer_partition():
    model = load_model_config(
        "tests/serving/fixtures/qwen3_pp2_scheduler")
    assert model.kv_bytes_per_token_per_tensor_parallel_rank(1) == 1024
    assert model.kv_bytes_per_token_per_parallel_rank(1, 2) == 512
    with pytest.raises(ValueError, match="divisible by pipeline"):
        model.kv_bytes_per_token_per_parallel_rank(1, 3)


def test_kimi_k2_6_mla_capacity_uses_one_replicated_compressed_vector():
    model = load_model_config(
        "tests/serving/fixtures/kimi_k2_6_official")
    expected_per_layer = (model.kv_lora_rank + model.qk_rope_head_dim) * 2
    assert expected_per_layer == 1_152
    assert model.kv_bytes_per_token_per_parallel_rank(1, 1) == 70_272
    assert model.kv_bytes_per_token_per_parallel_rank(2, 1) == 70_272
    assert model.kv_bytes_per_token_per_parallel_rank(1, 61) == 1_152


def test_glm_5_2_dsa_capacity_combines_compressed_and_full_index_caches():
    model = load_model_config(
        "tests/serving/fixtures/glm_5_2_official")
    compressed_per_layer = (
        model.kv_lora_rank + model.qk_rope_head_dim) * model.dtype_bytes
    index_per_full_layer = model.index_head_dim + 4

    assert model.indexer_types.count("full") == 21
    assert model.kv_bytes_per_token_per_parallel_rank(1, 1) == (
        model.num_hidden_layers * compressed_per_layer
        + 21 * index_per_full_layer
    )
    assert model.kv_bytes_per_token_per_parallel_rank(2, 1) == 92_628
    with pytest.raises(NotImplementedError, match="PP>1"):
        model.kv_bytes_per_token_per_parallel_rank(1, 2)


def test_glm_5_2_mtp_residency_is_complete_and_last_stage_only():
    model = load_model_config(
        "tests/serving/fixtures/glm_5_2_official")
    parallelism = ParallelConfig(
        tensor_parallel_size=8,
        pipeline_parallel_size=4,
        enable_expert_parallel=True,
    )
    target_only = model_rank_residencies(
        model, parallelism, worker_role="colocated")
    with_mtp = model_rank_residencies(
        model,
        parallelism,
        worker_role="colocated",
        include_mtp=True,
    )

    assert all(rank.mtp_bytes == 0 for rank in target_only)
    assert all(rank.mtp_bytes == 0 for rank in with_mtp[:-8])
    assert {rank.mtp_bytes for rank in with_mtp[-8:]} == {3_143_567_872}
    assert all(
        enabled.weight_resident_bytes - disabled.weight_resident_bytes
        == enabled.mtp_bytes
        for disabled, enabled in zip(target_only, with_mtp, strict=True)
    )


def test_dsv3_h200_tp8_ep1_weight_residency_and_mtp_bytes():
    model = load_model_config("configs/model/deepseek-v3.json")
    parallelism = ParallelConfig(tensor_parallel_size=8)
    target_only = model_rank_residencies(
        model, parallelism, worker_role="colocated")
    with_mtp = model_rank_residencies(
        model, parallelism, worker_role="colocated", include_mtp=True)

    # FP8 weights TP8-sharded fit one H200 per rank with ample KV headroom.
    assert all(rank.mtp_bytes == 0 for rank in target_only)
    assert {rank.weight_resident_bytes for rank in target_only} == {
        85_013_078_016}
    assert {rank.mtp_bytes for rank in with_mtp} == {1_903_602_688}
    assert all(
        enabled.weight_resident_bytes - disabled.weight_resident_bytes
        == enabled.mtp_bytes
        for disabled, enabled in zip(target_only, with_mtp, strict=True)
    )
