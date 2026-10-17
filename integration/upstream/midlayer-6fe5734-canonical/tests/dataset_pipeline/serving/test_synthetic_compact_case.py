"""Synthetic requests are immutable inputs, never fabricated observations."""

import json
from pathlib import Path
import shutil

import pytest

from dataset_pipeline.serving_dataset.compact_case import load_compact_case


CASE = Path("serving_dataset/synthetic/inferencex_style/"
            "dsv3_h200_8k1024_closedloop_tp8_ep1/c4")


def test_synthetic_case_has_requests_and_explicitly_absent_observations():
    case = load_compact_case(CASE)
    assert len(case.requests) == 12
    assert len(case.warmup_requests) == 4
    assert case.metrics is None
    assert case.scheduler_steps is None
    assert all(value is None for value in case.alignment_steps.values())
    assert all(row.cached_hit_tokens == 0 for row in case.requests)
    assert case.manifest["generation"]["cached_hit_provenance"] == (
        "assumed_zero_prefix_cache_disabled")


@pytest.mark.parametrize("filename", (
    "requests.csv", "warmup_requests.csv", "runtime_memory_policy_colocated.json", "runtime_memory_policy_pd_disagg.json"))
def test_synthetic_payload_tampering_fails_closed(tmp_path, filename):
    root = tmp_path / "case"
    shutil.copytree(CASE, root)
    path = root / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256|sha256|SHA-256"):
        load_compact_case(root)


@pytest.mark.parametrize("change", ("observed_metrics", "protocol", "extra", "cache"))
def test_synthetic_manifest_rejects_false_evidence_and_inconsistent_inputs(tmp_path, change):
    root = tmp_path / "case"
    shutil.copytree(CASE, root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "observed_metrics":
        manifest["metrics"]["status"] = "available"
    elif change == "protocol":
        manifest["runtime"]["requests_per_client"] = 10
    elif change == "extra":
        manifest["unknown"] = 1
    elif change == "cache":
        manifest["generation"]["cached_hit_provenance"] = "observed_zero"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_compact_case(root)
