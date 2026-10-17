"""H200 Host profile assembly: substitution records and fitter guards."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset_pipeline.serving_dataset import scheduler_host_profile

_ROOT = Path(__file__).resolve().parents[3]
_REFERENCES = _ROOT / "microbench/nvidia_h200/runtime"
_SCHEDULER = _ROOT / "configs/host_timing/scheduler_host_profiles"


def _reference(relative: str) -> dict:
    return json.loads((_REFERENCES / relative).read_text())


def test_vllm030_mechanism_references_declare_their_vllm011_source():
    for relative in (
            "host_request_path/references/host_request_path_vllm030.json",
            "launch/references/host_runtime_prepare_vllm030.json"):
        substitution = _reference(relative)["substitution"]
        assert substitution["target_vllm_version"] == "0.30.0"
        assert substitution["measured_vllm_version"] == "0.11.0"
        assert substitution["reason"]


def test_same_version_references_carry_no_substitution():
    assert _reference("launch/references/host_launch_vllm030.json")["substitution"] is None
    for relative in (
            "launch/references/host_launch_vllm011.json",
            "host_request_path/references/host_request_path_vllm011.json",
            "launch/references/host_runtime_prepare_vllm011.json"):
        assert _reference(relative)["substitution"] is None


def test_vllm030_scheduler_profile_names_its_substituted_prefill_line():
    profile = json.loads((_SCHEDULER / "nvidia_h200_vllm030.json").read_text())
    source = json.loads((_SCHEDULER / "nvidia_h200_vllm011.json").read_text())
    assert profile["scheduler"]["vllm_version"] == "0.30.0"
    assert "prefill_line_substitute=vllm=0.11.0" in profile["scheduler"]["source_identity"]
    assert profile["service_time_s"]["prefill"] == source["service_time_s"]["prefill"]
    assert profile["service_time_s"]["plan"] != source["service_time_s"]["plan"]


def _substitute_args(path, reason):
    return SimpleNamespace(
        prefill_line_substitute=path, prefill_line_substitution_reason=reason,
        hardware_id="nvidia_h200")


def test_prefill_substitution_requires_a_reason_and_another_version():
    path = _SCHEDULER / "nvidia_h200_vllm011.json"
    with pytest.raises(ValueError, match="requires a reason"):
        scheduler_host_profile._prefill_line_substitute(
            _substitute_args(path, None), "0.30.0")
    with pytest.raises(ValueError, match="must be fitted, not substituted"):
        scheduler_host_profile._prefill_line_substitute(
            _substitute_args(path, "reason"), "0.11.0")
    with pytest.raises(ValueError, match="needs a substitute"):
        scheduler_host_profile._prefill_line_substitute(
            _substitute_args(None, "reason"), "0.30.0")
    assert scheduler_host_profile._prefill_line_substitute(
        _substitute_args(None, None), "0.30.0") is None
