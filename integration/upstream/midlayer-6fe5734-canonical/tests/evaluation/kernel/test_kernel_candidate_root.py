"""Contract tests for kernel evaluation candidate roots."""

import json

import pytest

import evaluation.kernel.dataset as dataset
from src.kernel.dataset_storage import compress_kernel_dataset_content


_KIND = "nvidia_h200_kcampaign_kernel_dataset_candidate"


def _write_family(root, campaign, family, cases, *, supported=True,
                  stack="vllm030"):
    plan_dir = root / campaign
    plan_dir.mkdir(parents=True, exist_ok=True)
    payload_path = plan_dir / f"{family}.json.zst"
    if cases:
        payload_path.write_bytes(compress_kernel_dataset_content(
            json.dumps(cases).encode("utf-8")))
    (plan_dir / f"{family}.manifest.json").write_text(json.dumps({
        "artifact_kind": _KIND, "hardware_id": "nvidia_h200",
        "campaign": campaign, "family": family,
        "evaluation_supported": supported, "vllm_stack": stack,
        "candidate_path": str(payload_path) if cases else None,
        "measured_case_count": len(cases),
    }))


def test_candidate_root_prefixes_cases_by_plan_and_skips_declared_families(
        tmp_path):
    _write_family(tmp_path, "plan_a", "gemm", {"G1": {"x": 1}})
    _write_family(tmp_path, "plan_b", "gemm", {"G1": {"x": 2}})
    _write_family(tmp_path, "plan_b", "kda", {"K1": {"x": 3}},
                  supported=False)
    _write_family(tmp_path, "plan_c", "vector", {})

    merged = dataset.load_kernel_candidate_root("nvidia_h200", str(tmp_path))

    assert merged == {"plan_a/G1": {"x": 1}, "plan_b/G1": {"x": 2}}


def test_candidate_root_rejects_mixed_stacks_and_foreign_manifests(tmp_path):
    _write_family(tmp_path, "plan_a", "gemm", {"G1": {}}, stack="vllm030")
    _write_family(tmp_path, "plan_b", "gemm", {"G2": {}}, stack="vllm024")
    with pytest.raises(ValueError, match="mixes software stacks"):
        dataset.load_kernel_candidate_root("nvidia_h200", str(tmp_path))
    with pytest.raises(ValueError, match="no kernel candidate contract"):
        dataset.load_kernel_candidate_root("nvidia_a800", str(tmp_path))


def test_candidate_root_config_must_not_point_into_canonical_data(
        monkeypatch, tmp_path):
    config = tmp_path / "roots.yaml"
    config.write_text(
        "schema_version: 1\n"
        "candidate_roots:\n  nvidia_h200: kernel_dataset/nvidia_h200\n")
    monkeypatch.setattr(dataset, "KERNEL_CANDIDATE_ROOTS_CONFIG", str(config))
    with pytest.raises(ValueError, match="must not be canonical"):
        dataset.kernel_evaluation_dataset_location("nvidia_h200")
    config.write_text(
        "schema_version: 1\n"
        "candidate_roots:\n  nvidia_h200: profiler_output/h200\n")
    kind, _ = dataset.kernel_evaluation_dataset_location("nvidia_h200")
    assert kind == "candidate"
    assert dataset.kernel_evaluation_dataset_location("nvidia_a800")[0] == (
        "canonical")
