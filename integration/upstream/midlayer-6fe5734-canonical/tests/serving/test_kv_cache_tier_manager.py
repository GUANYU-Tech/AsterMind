from dataclasses import replace

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import KernelKind
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    WorkerPoolConfig,
)
from src.serving.kernel_lowering.execution_timeline import (
    ExecutionTimeline,
    build_execution_timeline,
)
from src.serving.kernel_lowering.kv_offload_completion import (
    build_kv_cache_transfer_completion,
)
from src.serving.kernel_lowering.kernel_predict import predict_kernel_plan
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.kv_cache import (
    KVCacheAction,
    KVCacheConfig,
    KVCacheEvent,
    KVCachePhysicalTransferCompletion,
    KVCacheTierConfig,
    KVCacheTierManager,
    KVCacheTransferCompletion,
)
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
from src.serving.workload.build_workload import (
    build_workload,
    is_validated_colocated_tp2_host_kv_scope,
)
from src.serving.workload.model_config import (
    is_validated_qwen3_tp2_host_kv_model_scope,
    load_model_config,
)
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    RuntimeInputs,
)
from src.serving.workload.workload_graph import WorkloadGraph, WorkloadKind


def _config(*, hbm_blocks=1, host_blocks=1, policy="lru"):
    block_tokens = 16
    bytes_per_token = 64
    block_bytes = block_tokens * bytes_per_token
    return KVCacheConfig(
        tiers=(
            KVCacheTierConfig(
                "hbm", hbm_blocks * block_bytes, block_tokens),
            KVCacheTierConfig(
                "host", host_blocks * block_bytes, block_tokens, "pcie"),
        ),
        bytes_per_token=bytes_per_token,
        offload_policy=policy,
    )


def _test_hardware():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    # These tests exercise Host-KV lifecycle, fan-out, completion, and
    # timeline composition rather than PCIe setup calibration.  Production
    # A800 deliberately leaves the unmeasured setup latency unavailable.
    hardware.interconnect["pcie_latency_ns"] = 100.0
    return hardware


def test_host_tier_capacity_changes_only_when_transfer_completes():
    manager = KVCacheTierManager(_config())

    allocated_a = manager.allocate("a", 0, 16)
    assert allocated_a.action is KVCacheAction.ALLOCATE
    evict_a = manager.prepare_evict(pool_name="main", ranks=(0,))
    assert evict_a.action is KVCacheAction.EVICT
    assert evict_a.source_tier == "hbm"
    assert evict_a.destination_tier == "host"
    assert evict_a.bytes == manager.block_bytes
    before_evict_completion = manager.manifest()
    assert before_evict_completion["tiers"]["hbm"] == {
        "capacity_bytes": manager.block_bytes,
        "used_bytes": manager.block_bytes,
        "reserved_bytes": 0,
        "resident_blocks": 1,
    }
    assert before_evict_completion["tiers"]["host"] == {
        "capacity_bytes": manager.block_bytes,
        "used_bytes": 0,
        "reserved_bytes": manager.block_bytes,
        "resident_blocks": 0,
    }
    with pytest.raises(ValueError, match="no free block capacity"):
        manager.allocate("b", 0, 16)

    assert manager.complete_transfer(_unit_completion(evict_a)) == evict_a
    after_evict_completion = manager.manifest()
    assert after_evict_completion["tiers"]["hbm"]["used_bytes"] == 0
    assert after_evict_completion["tiers"]["host"]["used_bytes"] == (
        manager.block_bytes)
    allocated_b = manager.allocate("b", 0, 16)
    assert allocated_b.action is KVCacheAction.ALLOCATE
    released_b = manager.release("b", 0, 16)
    assert released_b.action is KVCacheAction.RELEASE

    load_a = manager.prepare_load(
        "a", 0, 16, pool_name="main", ranks=(0,))
    assert load_a.action is KVCacheAction.LOAD
    assert load_a.source_tier == "host"
    assert load_a.destination_tier == "hbm"
    before_load_completion = manager.manifest()
    assert before_load_completion["tiers"]["host"]["used_bytes"] == (
        manager.block_bytes)
    assert before_load_completion["tiers"]["hbm"]["reserved_bytes"] == (
        manager.block_bytes)
    assert manager.complete_transfer(_unit_completion(load_a)) == load_a
    assert manager.release("a", 0, 16).tier == "hbm"
    final = manager.manifest()
    assert final["tiers"]["hbm"]["used_bytes"] == 0
    assert final["tiers"]["host"]["used_bytes"] == 0
    assert final["residency"] == []
    assert final["inflight_ready_events"] == ()


def test_capacity_events_apply_multi_block_ranges_atomically():
    config = _config(hbm_blocks=3, host_blocks=1)
    manager = KVCacheTierManager(config)
    allocate = KVCacheEvent(
        "a", KVCacheAction.ALLOCATE, "hbm",
        0, 32, 2 * manager.block_bytes,
    )
    release = KVCacheEvent(
        "a", KVCacheAction.RELEASE, "hbm",
        0, 32, 2 * manager.block_bytes,
    )

    assert manager.apply_capacity_event(allocate) == allocate
    allocated = manager.manifest()
    assert allocated["schema_version"] == 2
    assert "policy" not in allocated
    assert allocated["configured_policy"] == "lru"
    assert allocated["victim_policy_validation"] == (
        "external_transfer_intent_not_validated")
    assert allocated["tiers"]["hbm"]["used_bytes"] == (
        2 * manager.block_bytes)
    assert allocated["tiers"]["hbm"]["resident_blocks"] == 2
    assert manager.apply_capacity_event(release) == release
    released = manager.manifest()
    assert released["tiers"]["hbm"]["used_bytes"] == 0
    assert released["residency"] == []

    insufficient = KVCacheTierManager(_config(hbm_blocks=1, host_blocks=1))
    with pytest.raises(ValueError, match="no free block capacity"):
        insufficient.apply_capacity_event(allocate)
    state = insufficient.manifest()
    assert state["tiers"]["hbm"]["used_bytes"] == 0
    assert state["residency"] == []

def test_explicit_lifecycle_intents_validate_capacity_and_quiescence():
    manager = KVCacheTierManager(_config())
    allocate = KVCacheEvent(
        "a",
        KVCacheAction.ALLOCATE,
        "hbm",
        0,
        16,
        manager.block_bytes,
    )
    evict = KVCacheEvent(
        "a",
        KVCacheAction.EVICT,
        "host",
        0,
        16,
        manager.block_bytes,
        source_tier="hbm",
        destination_tier="host",
        ready_event="framework-evict-a",
    )

    assert manager.apply_capacity_event(allocate) == allocate
    assert manager.register_transfer_intent(
        evict, pool_name="main", ranks=(0,)) == evict
    with pytest.raises(RuntimeError, match="in-flight transfers"):
        manager.assert_quiescent()
    assert manager.manifest()["tiers"]["host"]["reserved_bytes"] == (
        manager.block_bytes)

    assert manager.complete_transfer(_unit_completion(evict)) == evict
    manager.assert_quiescent()
    release = KVCacheEvent(
        "a",
        KVCacheAction.RELEASE,
        "host",
        0,
        16,
        manager.block_bytes,
    )
    assert manager.apply_capacity_event(release) == release
    assert manager.manifest()["residency"] == []


@pytest.mark.parametrize(
    "case,error",
    (
        ("wrong_bytes", "bytes disagree"),
        ("wrong_source", "source disagrees"),
        ("wrong_direction", "direction disagrees"),
        ("unknown_block", "not resident"),
    ),
)
def test_explicit_transfer_intents_fail_loud(case, error):
    manager = KVCacheTierManager(_config())
    manager.allocate("a", 0, 16)
    if case == "wrong_bytes":
        request_id = "a"
        byte_count = 2 * manager.block_bytes
        source = "hbm"
        destination = "host"
    elif case == "wrong_source":
        first_evict = manager.prepare_evict(pool_name="main", ranks=(0,))
        manager.complete_transfer(_unit_completion(first_evict))
        request_id = "a"
        byte_count = manager.block_bytes
        source = "hbm"
        destination = "host"
    elif case == "wrong_direction":
        request_id = "a"
        byte_count = manager.block_bytes
        source = "host"
        destination = "hbm"
    elif case == "unknown_block":
        request_id = "b"
        byte_count = manager.block_bytes
        source = "hbm"
        destination = "host"
    else:
        raise AssertionError(f"unknown explicit intent case: {case!r}")
    event = KVCacheEvent(
        request_id,
        KVCacheAction.EVICT,
        "host",
        0,
        16,
        byte_count,
        source_tier=source,
        destination_tier=destination,
        ready_event=f"framework-{case}",
    )

    with pytest.raises(ValueError, match=error):
        manager.register_transfer_intent(
            event, pool_name="main", ranks=(0,))


def test_explicit_transfer_ready_event_cannot_be_reused():
    manager = KVCacheTierManager(_config())
    manager.allocate("a", 0, 16)
    evict = KVCacheEvent(
        "a",
        KVCacheAction.EVICT,
        "host",
        0,
        16,
        manager.block_bytes,
        source_tier="hbm",
        destination_tier="host",
        ready_event="framework-transfer-a",
    )
    manager.register_transfer_intent(
        evict, pool_name="main", ranks=(0,))
    manager.complete_transfer(_unit_completion(evict))
    load = replace(
        evict,
        action=KVCacheAction.LOAD,
        source_tier="host",
        destination_tier="hbm",
    )

    with pytest.raises(ValueError, match="must be unique"):
        manager.register_transfer_intent(
            load, pool_name="main", ranks=(0,))


def test_host_tier_evicts_the_least_recently_used_full_block():
    manager = KVCacheTierManager(_config(hbm_blocks=2))
    manager.allocate("a", 0, 16)
    manager.allocate("b", 0, 16)
    manager.touch("a", 0, 16)

    event = manager.prepare_evict(pool_name="main", ranks=(0,))

    assert event.request_id == "b"
    assert (event.token_start, event.token_end) == (0, 16)


@pytest.mark.parametrize(
    "config,error",
    (
        (_config(policy="native"), "policy 'lru'"),
        (KVCacheConfig(
            tiers=(
                KVCacheTierConfig("hbm", 1024, 16),
                KVCacheTierConfig("host", 2048, 32, "pcie"),
            ),
            bytes_per_token=64,
            offload_policy="lru",
        ), "equal HBM and Host block sizes"),
        (KVCacheConfig(
            tiers=(
                KVCacheTierConfig("hbm", 1024, 16),
                KVCacheTierConfig("host", 1024, 16, "nvlink"),
            ),
            bytes_per_token=64,
            offload_policy="lru",
        ), "PCIe"),
        (KVCacheConfig(
            tiers=(
                KVCacheTierConfig("HBM", 1024, 16),
                KVCacheTierConfig("Host", 1024, 16, "pcie"),
            ),
            bytes_per_token=64,
            offload_policy="lru",
        ), "lowercase tier names"),
    ),
)
def test_host_tier_manager_rejects_unsupported_contracts(config, error):
    with pytest.raises((ValueError, NotImplementedError), match=error):
        KVCacheTierManager(config)


def test_transfer_events_require_positive_work_and_completion_identity():
    with pytest.raises(ValueError, match="positive transfer"):
        KVCacheEvent(
            "a", KVCacheAction.EVICT, "host", 0, 0, 0,
            source_tier="hbm", destination_tier="host",
            ready_event="evict-a",
        )
    with pytest.raises(ValueError, match="ready_event"):
        KVCacheEvent(
            "a", KVCacheAction.LOAD, "host", 0, 16, 1024,
            source_tier="host", destination_tier="hbm",
        )
    with pytest.raises(ValueError, match="cannot carry a ready_event"):
        KVCacheEvent(
            "a", KVCacheAction.ALLOCATE, "hbm", 0, 16, 1024,
            ready_event="not-a-transfer",
        )


def test_host_tier_events_lower_and_predict_through_the_canonical_path():
    model = load_model_config("tests/serving/fixtures/qwen3_scheduler")
    bytes_per_token = (
        2 * model.num_hidden_layers * model.num_key_value_heads
        * model.head_dim * model.dtype_bytes
    )
    block_tokens = 16
    block_bytes = block_tokens * bytes_per_token
    cache_config = KVCacheConfig(
        tiers=(
            KVCacheTierConfig("hbm", block_bytes, block_tokens),
            KVCacheTierConfig(
                "host", block_bytes, block_tokens, "pcie"),
        ),
        bytes_per_token=bytes_per_token,
        offload_policy="lru",
    )
    manager = KVCacheTierManager(cache_config)
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main", "colocated", "nvidia_a800",
                kv_cache=cache_config,
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    hardware = _test_hardware()
    evict_step = ScheduleStep(
        "host-tier-evict-step",
        "main",
        (ScheduledRequest("compute", "decode", 0, 1, 16),),
        attention_kv_layout="paged_varlen",
    )
    load_step = ScheduleStep(
        "host-tier-load-step",
        "main",
        (ScheduledRequest("a", "decode", 0, 1, 16),),
        attention_kv_layout="paged_varlen",
    )
    manager.allocate("a", 0, 16)
    evict = manager.prepare_evict(pool_name="main", ranks=(0,))
    evict_plan, evict_timeline, evict_completion = _lower_predict_event(
        evict, evict_step, model, deployment, hardware)
    manager.complete_transfer(evict_completion)
    load = manager.prepare_load(
        "a", 0, 16, pool_name="main", ranks=(0,))
    load_plan, load_timeline, load_completion = _lower_predict_event(
        load, load_step, model, deployment, hardware)

    evict_call = next(
        call for call in evict_plan.calls
        if call.kernel_spec.kind is KernelKind.KV_OFFLOAD)
    load_call = next(
        call for call in load_plan.calls
        if call.kernel_spec.kind is KernelKind.KV_OFFLOAD)
    assert evict_call.kernel_spec.transfer.direction == "evict"
    assert load_call.kernel_spec.transfer.direction == "load"
    for call in (evict_call, load_call):
        transfer = call.kernel_spec.transfer
        assert transfer.algorithm == "dma"
        assert transfer.bytes == block_bytes
        assert transfer.link == "pcie"
        assert transfer.tier == "host"
    assert evict_timeline.duration_s > 0
    assert load_timeline.duration_s > 0
    evict_interval = next(
        interval for interval in evict_timeline.intervals
        if interval.call_id == evict_call.call_id)
    evict_lm_head = next(
        interval for interval in evict_timeline.intervals
        if interval.call_id == "host-tier-evict-step:lm-head")
    assert evict_lm_head.end_time_s <= evict_interval.start_time_s
    load_interval = next(
        interval for interval in load_timeline.intervals
        if interval.call_id == load_call.call_id)
    following = next(
        interval for interval in load_timeline.intervals
        if interval.call_id == "host-tier-load-step:token-embedding")
    assert load_interval.end_time_s <= following.start_time_s
    assert manager.complete_transfer(load_completion) == load


def test_tp2_host_tier_event_fans_out_and_completes_atomically():
    model = load_model_config(
        "tests/serving/fixtures/qwen3_tp2_scheduler")
    bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(2, 1)
    assert bytes_per_token == 512
    block_tokens = 16
    block_bytes = block_tokens * bytes_per_token
    assert block_bytes == 8192
    cache_config = KVCacheConfig(
        tiers=(
            KVCacheTierConfig("hbm", block_bytes, block_tokens),
            KVCacheTierConfig("host", block_bytes, block_tokens, "pcie"),
        ),
        bytes_per_token=bytes_per_token,
        offload_policy="lru",
    )
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            "nvlink:main:tensor-fabric",
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                (0, 1),
                cache_config,
                collectives,
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    assert is_validated_colocated_tp2_host_kv_scope(deployment, model)
    assert not is_validated_colocated_tp2_host_kv_scope(
        deployment, replace(model, config_sha256="0" * 64))
    manager = KVCacheTierManager(cache_config)
    manager.allocate("a", 0, block_tokens)
    evict = manager.prepare_evict(pool_name="main", ranks=(0, 1))
    evict_step = ScheduleStep(
        "tp2-host-evict",
        "main",
        (ScheduledRequest("compute", "decode", 0, 1, 16),),
        attention_kv_layout="paged_varlen",
    )
    hardware = _test_hardware()
    evict_plan, evict_timeline, evict_completion = _lower_predict_event(
        evict,
        evict_step,
        model,
        deployment,
        hardware,
        expected_ranks=(0, 1),
    )

    assert evict_completion.ranks == (0, 1)
    assert len(evict_completion.physical_transfers) == 2
    assert all(
        physical.bytes == block_bytes
        for physical in evict_completion.physical_transfers)
    evict_calls = tuple(
        call for call in evict_plan.calls
        if call.kernel_spec.kind is KernelKind.KV_OFFLOAD)
    assert tuple(call.rank for call in evict_calls) == (0, 1)
    assert all(
        call.kernel_spec.transfer.bytes == block_bytes
        for call in evict_calls)
    assert {
        interval.reserved_resource_ids
        for interval in evict_timeline.intervals
        if interval.call_id in {call.call_id for call in evict_calls}
    } == {
        ("pool:main:rank:0:pcie",),
        ("pool:main:rank:1:pcie",),
    }
    assert manager.complete_transfer(evict_completion) == evict
    after_evict = manager.manifest()
    assert after_evict["tiers"]["hbm"]["used_bytes"] == 0
    assert after_evict["tiers"]["host"]["used_bytes"] == block_bytes

    load = manager.prepare_load(
        "a", 0, block_tokens, pool_name="main", ranks=(0, 1))
    load_step = ScheduleStep(
        "tp2-host-load",
        "main",
        (ScheduledRequest("a", "decode", 0, 1, 16),),
        attention_kv_layout="paged_varlen",
    )
    load_plan, load_timeline, load_completion = _lower_predict_event(
        load,
        load_step,
        model,
        deployment,
        hardware,
        expected_ranks=(0, 1),
    )
    load_intervals = {
        interval.rank: interval
        for interval in load_timeline.intervals
        if interval.call_id in {
            physical.call_id
            for physical in load_completion.physical_transfers
        }
    }
    embedding_intervals = {
        interval.rank: interval
        for interval in load_timeline.intervals
        if ":token-embedding:tp-rank-" in interval.call_id
    }
    assert set(load_intervals) == set(embedding_intervals) == {0, 1}
    assert all(
        load_intervals[rank].end_time_s
        <= embedding_intervals[rank].start_time_s
        for rank in (0, 1))
    assert manager.complete_transfer(load_completion) == load
    manager.release("a", 0, block_tokens)
    final_state = manager.manifest()
    assert final_state["residency"] == []
    assert all(
        final_state["tiers"][tier]["used_bytes"] == 0
        and final_state["tiers"][tier]["reserved_bytes"] == 0
        for tier in ("hbm", "host"))


def test_tp2_host_kv_scope_rejects_same_geometry_without_source_identity():
    model = load_model_config(
        "tests/serving/fixtures/qwen3_tp2_scheduler")

    assert is_validated_qwen3_tp2_host_kv_model_scope(model)
    assert not is_validated_qwen3_tp2_host_kv_model_scope(replace(
        model,
        intermediate_size=3 * model.intermediate_size,
        vocab_size=model.vocab_size + 17,
        config_sha256="0" * 64,
    ))


def test_tp2_host_tier_rejects_partial_or_wrong_aggregate_atomically():
    manager = KVCacheTierManager(_config())
    manager.allocate("a", 0, 16)
    event = manager.prepare_evict(pool_name="main", ranks=(0, 1))
    before = manager.manifest()

    partial = _completion(
        event,
        (_physical_completion(event, 0),),
    )
    with pytest.raises(ValueError, match="owner ranks"):
        manager.complete_transfer(partial)
    assert manager.manifest() == before

    wrong_rank = _completion(
        event,
        (
            _physical_completion(event, 0),
            _physical_completion(event, 2),
        ),
    )
    with pytest.raises(ValueError, match="owner ranks"):
        manager.complete_transfer(wrong_rank)
    assert manager.manifest() == before

    wrong_pool = _completion(
        event,
        (
            _physical_completion(event, 0, pool_name="other"),
            _physical_completion(event, 1, pool_name="other"),
        ),
    )
    with pytest.raises(ValueError, match="owner ranks"):
        manager.complete_transfer(wrong_pool)
    assert manager.manifest() == before

    aggregate_bytes = 2 * event.bytes
    wrong_aggregate = KVCacheTransferCompletion(
        ready_event=event.ready_event,
        action=event.action,
        bytes=aggregate_bytes,
        source_tier=event.source_tier,
        destination_tier=event.destination_tier,
        link="pcie",
        physical_transfers=(
            _physical_completion(event, 0, bytes_=aggregate_bytes),
            _physical_completion(event, 1, bytes_=aggregate_bytes),
        ),
    )
    with pytest.raises(ValueError, match="provenance"):
        manager.complete_transfer(wrong_aggregate)
    assert manager.manifest() == before

    with pytest.raises(ValueError, match="unique ranks"):
        _completion(
            event,
            (
                _physical_completion(event, 0),
                replace(
                    _physical_completion(event, 1),
                    rank=0,
                ),
            ),
        )
    assert manager.manifest() == before

    complete = _completion(
        event,
        (
            _physical_completion(event, 0),
            _physical_completion(event, 1),
        ),
    )
    assert manager.complete_transfer(complete) == event
    after = manager.manifest()
    assert after["tiers"]["hbm"]["used_bytes"] == 0
    assert after["tiers"]["host"]["used_bytes"] == event.bytes
    assert after["tiers"]["host"]["reserved_bytes"] == 0


@pytest.mark.parametrize(
    "case,error",
    (
        ("missing_rank", "one workload node per expected rank"),
        ("duplicate_rank", "pool/ranks disagree"),
        ("extra_rank", "one workload node per expected rank"),
        ("null_rank", "explicit physical ranks"),
        ("wrong_node_pool", "pool/ranks disagree"),
        ("aggregate_node_bytes", "workload provenance"),
        ("missing_evict_dependency", "dependency successor"),
        ("wrong_call_rank", "node/call/interval ownership"),
        ("wrong_source_node", "node/call/interval ownership"),
        ("aggregate_call_bytes", "physical provenance"),
        ("wrong_interval_rank", "node/call/interval ownership"),
        ("wrong_prediction", "canonical prediction"),
    ),
)
def test_tp2_completion_builder_rejects_noncanonical_materialization(
        case, error):
    event, graph, plan, predictions, timeline = _tp2_event_execution()
    offload_nodes = tuple(
        node for node in graph.nodes
        if node.kind is WorkloadKind.KV_OFFLOAD)
    assert tuple(node.rank for node in offload_nodes) == (0, 1)
    rank_one_node = offload_nodes[1]

    if case == "missing_rank":
        graph = WorkloadGraph(
            tuple(
                node for node in graph.nodes
                if node.node_id != rank_one_node.node_id),
            tuple(
                edge for edge in graph.edges
                if (edge.source != rank_one_node.node_id
                    and edge.destination != rank_one_node.node_id)),
        )
    elif case == "duplicate_rank":
        graph = WorkloadGraph(
            tuple(
                replace(node, rank=0)
                if node.node_id == rank_one_node.node_id else node
                for node in graph.nodes),
            graph.edges,
        )
    elif case == "extra_rank":
        graph = WorkloadGraph(
            graph.nodes + (
                replace(
                    rank_one_node,
                    node_id=f"{rank_one_node.node_id}:unexpected",
                    rank=2,
                ),
            ),
            graph.edges,
        )
    elif case == "null_rank":
        graph = WorkloadGraph(
            tuple(
                replace(node, rank=None)
                if node.node_id == rank_one_node.node_id else node
                for node in graph.nodes),
            graph.edges,
        )
    elif case == "wrong_node_pool":
        graph = WorkloadGraph(
            tuple(
                replace(node, pool_name="other")
                if node.node_id == rank_one_node.node_id else node
                for node in graph.nodes),
            graph.edges,
        )
    elif case == "aggregate_node_bytes":
        graph = WorkloadGraph(
            tuple(
                replace(
                    node,
                    attributes={
                        **node.attributes,
                        "bytes": 2 * event.bytes,
                    },
                ) if node.kind is WorkloadKind.KV_OFFLOAD else node
                for node in graph.nodes),
            graph.edges,
        )
    elif case == "missing_evict_dependency":
        graph = WorkloadGraph(
            graph.nodes,
            tuple(
                edge for edge in graph.edges
                if edge.destination != rank_one_node.node_id),
        )
    elif case in (
            "wrong_call_rank", "wrong_source_node", "aggregate_call_bytes"):
        replacement_calls = []
        for call in plan.calls:
            if call.call_id != rank_one_node.node_id:
                replacement_calls.append(call)
            elif case == "wrong_call_rank":
                replacement_calls.append(replace(call, rank=0))
            elif case == "wrong_source_node":
                replacement_calls.append(replace(
                    call, source_node_ids=(offload_nodes[0].node_id,)))
            elif case == "aggregate_call_bytes":
                replacement_calls.append(replace(
                    call,
                    kernel_spec=replace(
                        call.kernel_spec,
                        transfer=replace(
                            call.kernel_spec.transfer,
                            bytes=2 * event.bytes,
                        ),
                    ),
                ))
            else:
                raise AssertionError(f"unknown call mutation: {case!r}")
        plan = KernelPlan(tuple(replacement_calls))
    elif case == "wrong_interval_rank":
        timeline = ExecutionTimeline(
            tuple(
                replace(interval, rank=0)
                if interval.call_id == rank_one_node.node_id else interval
                for interval in timeline.intervals),
            timeline.duration_s,
        )
    elif case == "wrong_prediction":
        predictions = dict(predictions)
        prediction = predictions[rank_one_node.node_id]
        predictions[rank_one_node.node_id] = replace(
            prediction, t_total_s=2 * prediction.t_total_s)
    else:
        raise AssertionError(f"unknown materialization mutation: {case!r}")

    with pytest.raises(ValueError, match=error):
        build_kv_cache_transfer_completion(
            event,
            graph,
            plan,
            predictions,
            timeline,
            expected_pool_name="main",
            expected_ranks=(0, 1),
        )


def test_tp2_load_requires_a_dependency_edge_on_every_rank():
    event, graph, plan, predictions, timeline = _tp2_event_execution(
        action="load")
    rank_one_load = next(
        node for node in graph.nodes
        if node.kind is WorkloadKind.KV_OFFLOAD and node.rank == 1)
    corrupted_graph = WorkloadGraph(
        graph.nodes,
        tuple(
            edge for edge in graph.edges
            if edge.source != rank_one_load.node_id),
    )

    with pytest.raises(ValueError, match="dependency predecessor"):
        build_kv_cache_transfer_completion(
            event,
            corrupted_graph,
            plan,
            predictions,
            timeline,
            expected_pool_name="main",
            expected_ranks=(0, 1),
        )


@pytest.mark.parametrize(
    "case,error",
    (
        ("multi_rank", "does not yet support KV offload"),
        ("wrong_bytes", "bytes_per_token disagrees"),
        ("unequal_blocks", "equal HBM and Host block sizes"),
        ("non_host", "only the Host tier"),
        ("inactive_load", "demand load must target an active request"),
    ),
)
def test_workload_builder_rejects_unsupported_offload_boundaries(case, error):
    model = load_model_config("tests/serving/fixtures/qwen3_scheduler")
    expected_bytes_per_token = (
        2 * model.num_hidden_layers * model.num_key_value_heads
        * model.head_dim * model.dtype_bytes
    )
    block_tokens = 16
    if case == "multi_rank":
        parallelism = ParallelConfig(tensor_parallel_size=2)
        bytes_per_token = expected_bytes_per_token
        host_block_tokens = block_tokens
        offload_tier_name = "host"
    elif case == "wrong_bytes":
        parallelism = ParallelConfig()
        bytes_per_token = expected_bytes_per_token + 1
        host_block_tokens = block_tokens
        offload_tier_name = "host"
    elif case == "unequal_blocks":
        parallelism = ParallelConfig()
        bytes_per_token = expected_bytes_per_token
        host_block_tokens = 2 * block_tokens
        offload_tier_name = "host"
    elif case == "non_host":
        parallelism = ParallelConfig()
        bytes_per_token = expected_bytes_per_token
        host_block_tokens = block_tokens
        offload_tier_name = "ssd"
    elif case == "inactive_load":
        parallelism = ParallelConfig()
        bytes_per_token = expected_bytes_per_token
        host_block_tokens = block_tokens
        offload_tier_name = "host"
    else:
        raise AssertionError(f"unknown offload boundary case: {case!r}")
    event_bytes = block_tokens * bytes_per_token
    cache_config = KVCacheConfig(
        tiers=(
            KVCacheTierConfig("hbm", event_bytes, block_tokens),
            KVCacheTierConfig(
                offload_tier_name,
                host_block_tokens * bytes_per_token,
                host_block_tokens,
                "pcie",
            ),
        ),
        bytes_per_token=bytes_per_token,
        offload_policy="lru",
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main", "colocated", "nvidia_a800",
                parallelism=parallelism,
                kv_cache=cache_config,
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    step = ScheduleStep(
        "unsupported-offload",
        "main",
        (ScheduledRequest("a", "decode", 0, 1, 16),),
        attention_kv_layout="paged_varlen",
    )
    if case == "inactive_load":
        event_request_id = "b"
        event_action = KVCacheAction.LOAD
        source_tier = "host"
        destination_tier = "hbm"
    else:
        event_request_id = "a"
        event_action = KVCacheAction.EVICT
        source_tier = "hbm"
        destination_tier = offload_tier_name
    event = KVCacheEvent(
        event_request_id,
        event_action,
        offload_tier_name,
        0,
        block_tokens,
        event_bytes,
        source_tier=source_tier,
        destination_tier=destination_tier,
        ready_event="unsupported-transfer",
    )

    with pytest.raises((ValueError, NotImplementedError), match=error):
        build_workload(
            step,
            model,
            deployment,
            launch_policy="analytical_launch",
            kv_cache_events=(event,),
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "test_seeded_routing_v1", 0),
                "paged_varlen", "uniform"),
        )


def _tp2_event_execution(*, action="evict"):
    model = load_model_config(
        "tests/serving/fixtures/qwen3_tp2_scheduler")
    bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(2, 1)
    block_tokens = 16
    block_bytes = block_tokens * bytes_per_token
    cache_config = KVCacheConfig(
        tiers=(
            KVCacheTierConfig("hbm", block_bytes, block_tokens),
            KVCacheTierConfig(
                "host", block_bytes, block_tokens, "pcie"),
        ),
        bytes_per_token=bytes_per_token,
        offload_policy="lru",
    )
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            "nvlink:main:tensor-fabric",
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                (0, 1),
                cache_config,
                collectives,
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    manager = KVCacheTierManager(cache_config)
    manager.allocate("a", 0, block_tokens)
    evict = manager.prepare_evict(pool_name="main", ranks=(0, 1))
    if action == "evict":
        event = evict
        request_id = "compute"
    elif action == "load":
        manager.complete_transfer(_completion(
            evict,
            (
                _physical_completion(evict, 0),
                _physical_completion(evict, 1),
            ),
        ))
        event = manager.prepare_load(
            "a", 0, block_tokens, pool_name="main", ranks=(0, 1))
        request_id = "a"
    else:
        raise AssertionError(f"unknown TP2 Host-KV action: {action!r}")
    step = ScheduleStep(
        f"tp2-host-negative-{action}",
        "main",
        (ScheduledRequest(request_id, "decode", 0, 1, 16),),
        attention_kv_layout="paged_varlen",
    )
    graph, plan, predictions, timeline = _build_event_execution(
        event,
        step,
        model,
        deployment,
        _test_hardware(),
    )
    return event, graph, plan, predictions, timeline


def _build_event_execution(event, step, model, deployment, hardware):
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=(event,),
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )
    offload = next(
        node for node in graph.nodes
        if node.kind is WorkloadKind.KV_OFFLOAD)
    assert offload.attributes["request_id"] == event.request_id
    assert offload.attributes["token_start"] == event.token_start
    assert offload.attributes["token_end"] == event.token_end
    assert offload.attributes["ready_event"] == event.ready_event
    plan = lower_workload(
        graph, deployment, {"nvidia_a800": hardware})
    predictions = predict_kernel_plan(
        plan, {"nvidia_a800": hardware}, "midlayer")
    timeline = build_execution_timeline(plan, predictions)
    return graph, plan, predictions, timeline


def _lower_predict_event(
        event, step, model, deployment, hardware, *, expected_ranks=(0,)):
    graph, plan, predictions, timeline = _build_event_execution(
        event, step, model, deployment, hardware)
    offload_call = next(
        call for call in plan.calls
        if call.kernel_spec.kind is KernelKind.KV_OFFLOAD)
    corrupted_predictions = dict(predictions)
    corrupted_predictions[offload_call.call_id] = replace(
        predictions[offload_call.call_id],
        t_total_s=2 * predictions[offload_call.call_id].t_total_s,
    )
    with pytest.raises(ValueError, match="canonical prediction"):
        build_kv_cache_transfer_completion(
            event, graph, plan, corrupted_predictions, timeline,
            expected_pool_name="main", expected_ranks=expected_ranks)
    completion = build_kv_cache_transfer_completion(
        event, graph, plan, predictions, timeline,
        expected_pool_name="main", expected_ranks=expected_ranks)
    return plan, timeline, completion


def _physical_completion(event, rank, *, pool_name="main", bytes_=None):
    node_id = (
        f"unit:kv-offload:{event.ready_event}:tp-rank-{rank}")
    return KVCachePhysicalTransferCompletion(
        node_id=node_id,
        call_id=node_id,
        pool_name=pool_name,
        rank=rank,
        bytes=event.bytes if bytes_ is None else bytes_,
        link="pcie",
        start_time_s=float(rank),
        end_time_s=float(rank + 1),
    )


def _completion(event, physical_transfers):
    return KVCacheTransferCompletion(
        ready_event=event.ready_event,
        action=event.action,
        bytes=event.bytes,
        source_tier=event.source_tier,
        destination_tier=event.destination_tier,
        link="pcie",
        physical_transfers=physical_transfers,
    )


def _unit_completion(event):
    return KVCacheTransferCompletion(
        ready_event=event.ready_event,
        action=event.action,
        bytes=event.bytes,
        source_tier=event.source_tier,
        destination_tier=event.destination_tier,
        link="pcie",
        physical_transfers=(KVCachePhysicalTransferCompletion(
            node_id=f"unit:kv-offload:{event.ready_event}",
            call_id=f"unit:kv-offload:{event.ready_event}",
            pool_name="main",
            rank=0,
            bytes=event.bytes,
            link="pcie",
            start_time_s=0.0,
            end_time_s=1.0,
        ),),
    )
