import json
from pathlib import Path

import pytest

from dataset_pipeline.serving_dataset.compressed_artifact import (
    read_compressed_canonical_bytes,
)


_CASE_ROOT = Path("serving_dataset/production")


def _token_count(phase):
    if phase == "prefill":
        return 3
    if phase == "decode":
        return 1
    raise AssertionError(f"unsupported target phase: {phase}")


def _samples(model_id, phase):
    case_dir = _CASE_ROOT / model_id / "debug_workload_1s"
    manifest = json.loads((case_dir / "manifest.json").read_text())
    entry = manifest["alignment_steps"]
    payload = json.loads(read_compressed_canonical_bytes(
        case_dir / entry["path"],
        compressed_sha256=entry["compressed_sha256"],
        content_sha256=entry["sha256"],
        label="target-model alignment",
    ))
    return payload["phases"][phase]["samples"]


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_glm_alignment_preserves_dsa_cache_and_moe_routed_rows(phase):
    dense, moe = _samples("glm-5.2", phase)
    dense_specs = [
        call["kernel_spec"] for call in dense["midlayer"]["kernel_calls"]
    ]
    moe_specs = [
        call["kernel_spec"] for call in moe["midlayer"]["kernel_calls"]
    ]

    assert {spec["variant"] for spec in dense_specs} >= {
        "dsa_fp8_index_cache",
        "mla_kv_cache",
        f"dsa_indexer_{phase}",
        f"dsa_sparse_{phase}",
    }
    assert {spec["variant"] for spec in moe_specs} >= {
        "dsa_index_reuse",
        "mla_kv_cache",
        f"dsa_sparse_{phase}",
        "moe_topk",
        "moe_align",
        "moe_count_and_sort",
        "w1",
        "moe_act_and_mul",
        "w2",
        "moe_reduce",
    }
    actual_expert = next(
        event for event in moe["actual"]["events"]
        if len(event["inputs"]) == 3
    )
    token_count = _token_count(phase)
    assert actual_expert["inputs"][1]["shape"] == [token_count, 2]
    assert any(
        boundary["capability"] == "mtp_actual_upstream_execution"
        for boundary in moe["unsupported"]
    )


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_kimi_alignment_preserves_mla_int4_moe_and_routed_rows(phase):
    dense, moe = _samples("kimi-k2.6", phase)
    dense_specs = [
        call["kernel_spec"] for call in dense["midlayer"]["kernel_calls"]
    ]
    moe_specs = [
        call["kernel_spec"] for call in moe["midlayer"]["kernel_calls"]
    ]

    assert {spec["variant"] for spec in dense_specs} >= {
        "mla_kv_cache",
        f"mla_{phase}",
    }
    routed = [spec for spec in moe_specs if spec["kind"] == "FUSED_MOE"]
    assert len(routed) == 1
    assert routed[0]["variant"] == "routed_experts"
    assert routed[0]["attributes"]["routed_weight_dtype"] == "int4"
    token_count = _token_count(phase)
    assert routed[0]["shape"]["tokens"] == token_count
    actual_expert = next(
        event for event in moe["actual"]["events"]
        if len(event["inputs"]) == 3
    )
    assert actual_expert["inputs"][0]["shape"][0] == token_count
    assert actual_expert["inputs"][1]["shape"] == [token_count, 2]
