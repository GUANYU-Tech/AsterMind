from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from src.serving.deployment.af_disagg import build_af_disagg_plan
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import (
    DeploymentConfig,
    RouteConfig,
    WorkerPoolConfig,
)
from src.serving.kv_cache.kv_cache_config import (
    KVCacheConfig,
    KVCacheTierConfig,
)
from src.serving.kv_cache.kv_lifecycle_replay import (
    load_kv_lifecycle_replay,
)
from src.serving.request_gen.request_stream import (
    RequestSpec,
    RequestStream,
    request_stream_sha256,
)
from src.serving.schedule.backend_protocol import BackendConfig, backend_config_sha256
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
    schedule_step_sha256,
)
from src.serving.workload.model_config import load_model_config


_MODEL_FIXTURE = (
    Path(__file__).parent / "fixtures" / "qwen3_scheduler"
)


def _contract():
    model = load_model_config(str(_MODEL_FIXTURE))
    requests = RequestStream((RequestSpec(0.0, "a", 16, 1),))
    block_tokens = 16
    bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(1, 1)
    block_bytes = block_tokens * bytes_per_token
    cache = KVCacheConfig(
        tiers=(
            KVCacheTierConfig("hbm", 2 * block_bytes, block_tokens),
            KVCacheTierConfig(
                "host", block_bytes, block_tokens, "pcie"),
        ),
        bytes_per_token=bytes_per_token,
        offload_policy="lru",
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main", "colocated", "nvidia_a800", kv_cache=cache),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    backend = BackendConfig(
        "main", "colocated", 128, 1, 2,
        kv_block_size_tokens=block_tokens,
        kv_bytes_per_token=bytes_per_token,
    scheduler_pipeline_depth=1)
    payload = {
        "schema_version": 1,
        "artifact_id": "kv-lifecycle-test-v1",
        "provenance": {
            "source_kind": "synthetic_fixture",
            "producer": "pytest",
            "producer_version": "1",
            "source_reference": "unit-test",
        },
        "context": {
            "request_stream_sha256": request_stream_sha256(requests),
            "model_config_sha256": model.config_sha256,
            "deployment_id": deployment.deployment_id,
            "scheduler_config_sha256": backend_config_sha256(backend),
            "scheduler_runtime_sha256": "5" * 64,
            "scheduler_step_count": 3,
            "pool_name": "main",
            "backend_role": "colocated",
            "kv_cache": asdict(cache),
        },
        "steps": [
            {
                "step_index": 0,
                "step_id": "main:vllm:0",
                "schedule_step_sha256": schedule_step_sha256(
                    _backend_step(0).schedule),
                "events": [],
            },
            {
                "step_index": 1,
                "step_id": "main:vllm:1",
                "schedule_step_sha256": schedule_step_sha256(
                    _backend_step(1).schedule),
                "events": [{
                    "request_id": "a",
                    "action": "evict",
                    "tier": "host",
                    "token_start": 0,
                    "token_end": block_tokens,
                    "bytes": block_bytes,
                    "source_tier": "hbm",
                    "destination_tier": "host",
                    "ready_event": "evict-a-0",
                }],
            },
            {
                "step_index": 2,
                "step_id": "main:vllm:2",
                "schedule_step_sha256": schedule_step_sha256(
                    _backend_step(2).schedule),
                "events": [{
                    "request_id": "a",
                    "action": "load",
                    "tier": "host",
                    "token_start": 0,
                    "token_end": block_tokens,
                    "bytes": block_bytes,
                    "source_tier": "host",
                    "destination_tier": "hbm",
                    "ready_event": "load-a-0",
                }],
            },
        ],
    }
    return model, requests, deployment, backend, payload


def _write_replay(tmp_path, payload):
    path = tmp_path / "kv_lifecycle.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _backend_step(step_index):
    return BackendStep(
        ScheduleStep(
            f"main:vllm:{step_index}",
            "main",
            (ScheduledRequest("a", "prefill", 16, 0),),
            attention_kv_layout="paged_varlen",
        ),
        generated_tokens={"a": 0},
    )


def test_replay_binds_and_consumes_native_scheduler_steps(tmp_path):
    model, requests, deployment, backend, payload = _contract()
    path = _write_replay(tmp_path, payload)
    replay = load_kv_lifecycle_replay(path)

    assert replay.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    cursor = replay.bind(
        model, deployment, requests, backend, "5" * 64)
    assert cursor.events_for_step(_backend_step(0)) == ()
    assert tuple(
        event.action.value
        for event in cursor.events_for_step(_backend_step(1))
    ) == ("evict",)
    assert tuple(
        event.action.value
        for event in cursor.events_for_step(_backend_step(2))
    ) == ("load",)
    cursor.assert_exhausted()
    manifest = replay.manifest()
    assert manifest["source_sha256"] == replay.source_sha256
    assert manifest["context"]["scheduler_config_sha256"] == (
        backend_config_sha256(backend))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["steps"][1]["events"][0].__setitem__(
                "token_start", 1),
            "aligned full Host KV block",
        ),
        (
            lambda payload: payload["steps"][2]["events"][0].__setitem__(
                "ready_event", "evict-a-0"),
            "ready_event values must be unique",
        ),
        (
            lambda payload: payload["steps"][1].__setitem__(
                "step_id", "main:vllm:7"),
            "step IDs disagree",
        ),
        (
            lambda payload: payload["steps"][1]["events"][0].__setitem__(
                "action", "release"),
            "accepts only LOAD/EVICT",
        ),
    ),
)
def test_loader_rejects_noncanonical_replay(tmp_path, mutate, message):
    _, _, _, _, payload = _contract()
    mutate(payload)
    path = _write_replay(tmp_path, payload)

    with pytest.raises((TypeError, ValueError), match=message):
        load_kv_lifecycle_replay(path)


def test_binding_rejects_scheduler_context_mismatch(tmp_path):
    model, requests, deployment, backend, payload = _contract()
    payload["context"]["scheduler_config_sha256"] = "0" * 64
    replay = load_kv_lifecycle_replay(_write_replay(tmp_path, payload))

    with pytest.raises(ValueError, match="scheduler_config_sha256 mismatch"):
        replay.bind(model, deployment, requests, backend, "5" * 64)


def test_binding_rejects_scheduler_runtime_mismatch(tmp_path):
    model, requests, deployment, backend, payload = _contract()
    replay = load_kv_lifecycle_replay(_write_replay(tmp_path, payload))

    with pytest.raises(ValueError, match="scheduler_runtime_sha256 mismatch"):
        replay.bind(model, deployment, requests, backend, "6" * 64)


def test_binding_rejects_non_attention_af_backend_role(tmp_path):
    model, requests, _, _, payload = _contract()
    cache_payload = payload["context"]["kv_cache"]
    cache = KVCacheConfig(
        tiers=tuple(
            KVCacheTierConfig(**tier)
            for tier in cache_payload["tiers"]
        ),
        bytes_per_token=cache_payload["bytes_per_token"],
        offload_policy=cache_payload["offload_policy"],
    )
    deployment = build_af_disagg_plan(
        DeploymentConfig(
            "af_disagg",
            (
                WorkerPoolConfig(
                    "attention", "attention", "nvidia_a800",
                    ranks=(0,), kv_cache=cache),
                WorkerPoolConfig(
                    "ffn", "ffn", "nvidia_a800", ranks=(1,)),
            ),
            (
                RouteConfig(
                    "attention", "ffn", "activation",
                    "send_recv", "nvlink",
                    resource_id="nvlink:attention--ffn",
                    backend="analytical_roofline_v1"),
                RouteConfig(
                    "ffn", "attention", "activation",
                    "send_recv", "nvlink",
                    resource_id="nvlink:attention--ffn",
                    backend="analytical_roofline_v1"),
            ),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    invalid_backend = BackendConfig(
        "attention", "ffn", 128, 1, 2,
        kv_block_size_tokens=16,
        kv_bytes_per_token=cache.bytes_per_token,
    scheduler_pipeline_depth=1)
    payload["context"]["deployment_id"] = deployment.deployment_id
    payload["context"]["scheduler_config_sha256"] = (
        backend_config_sha256(invalid_backend))
    payload["context"]["pool_name"] = "attention"
    payload["context"]["backend_role"] = "ffn"
    for index, step in enumerate(payload["steps"]):
        schedule = ScheduleStep(
            f"attention:vllm:{index}",
            "attention",
            (ScheduledRequest("a", "prefill", 16, 0),),
            attention_kv_layout="paged_varlen",
        )
        step["step_id"] = schedule.step_id
        step["schedule_step_sha256"] = schedule_step_sha256(schedule)
    replay = load_kv_lifecycle_replay(_write_replay(tmp_path, payload))

    with pytest.raises(
            ValueError, match="backend role must be colocated Attention"):
        replay.bind(
            model,
            deployment,
            requests,
            invalid_backend,
            "5" * 64,
        )


def test_cursor_rejects_step_order_and_unconsumed_events(tmp_path):
    model, requests, deployment, backend, payload = _contract()
    replay = load_kv_lifecycle_replay(_write_replay(tmp_path, payload))
    cursor = replay.bind(
        model, deployment, requests, backend, "5" * 64)

    with pytest.raises(ValueError, match="Scheduler step order mismatch"):
        cursor.events_for_step(_backend_step(1))

    cursor = replay.bind(
        model, deployment, requests, backend, "5" * 64)
    cursor.events_for_step(_backend_step(0))
    with pytest.raises(RuntimeError, match="unconsumed steps"):
        cursor.assert_exhausted()


def test_cursor_rejects_changed_schedule_decision_and_extra_tail(tmp_path):
    model, requests, deployment, backend, payload = _contract()
    replay = load_kv_lifecycle_replay(_write_replay(tmp_path, payload))
    cursor = replay.bind(
        model, deployment, requests, backend, "5" * 64)
    cursor.events_for_step(_backend_step(0))
    changed = BackendStep(
        ScheduleStep(
            "main:vllm:1",
            "main",
            (ScheduledRequest("a", "prefill", 15, 0),),
            attention_kv_layout="paged_varlen",
        ),
        generated_tokens={"a": 0},
    )
    with pytest.raises(ValueError, match="ScheduleStep SHA256 mismatch"):
        cursor.events_for_step(changed)

    cursor = replay.bind(
        model, deployment, requests, backend, "5" * 64)
    for index in range(3):
        cursor.events_for_step(_backend_step(index))
    cursor.assert_exhausted()
    with pytest.raises(ValueError, match="more Scheduler steps"):
        cursor.events_for_step(_backend_step(3))


def test_loader_requires_explicit_empty_step_coverage(tmp_path):
    _, _, _, _, payload = _contract()
    del payload["steps"][0]

    with pytest.raises(ValueError, match="cover every Scheduler step"):
        load_kv_lifecycle_replay(_write_replay(tmp_path, payload))


@pytest.mark.parametrize(
    ("needle", "duplicate", "key"),
    (
        (
            '"artifact_id":"kv-lifecycle-test-v1"',
            ('"artifact_id":"kv-lifecycle-test-v1",'
             '"artifact_id":"kv-lifecycle-test-v1"'),
            "artifact_id",
        ),
        (
            '"backend_role":"colocated"',
            ('"backend_role":"colocated",'
             '"backend_role":"colocated"'),
            "backend_role",
        ),
        (
            '"step_id":"main:vllm:0"',
            ('"step_id":"main:vllm:0",'
             '"step_id":"main:vllm:0"'),
            "step_id",
        ),
        (
            '"ready_event":"evict-a-0"',
            ('"ready_event":"evict-a-0",'
             '"ready_event":"evict-a-0"'),
            "ready_event",
        ),
    ),
)
def test_loader_rejects_duplicate_json_keys(
        tmp_path, needle, duplicate, key):
    _, _, _, _, payload = _contract()
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert serialized.count(needle) == 1
    serialized = serialized.replace(needle, duplicate, 1)
    path = tmp_path / "duplicate_kv_lifecycle.json"
    path.write_text(serialized, encoding="utf-8")

    with pytest.raises(ValueError, match=f"duplicate key '{key}'"):
        load_kv_lifecycle_replay(path)
