import json

import pytest

from evaluation.serving.experiments import pd_capture_executor


def _plan(report_path):
    return {
        "identity": {
            "case_id": "tp1_qps4",
            "hardware_id": "nvidia_a800",
            "model_id": "qwen3-8b",
        },
        "artifacts": {"setting_parity": str(report_path)},
    }


def _report(passed=True, differences=None):
    return {
        "schema_version": 1,
        "kind": "sharegpt_pd_setting_parity",
        "identity": {
            "case_id": "tp1_qps4",
            "hardware_id": "nvidia_a800",
            "model_id": "qwen3-8b",
        },
        "source": {},
        "allowed_differences": [],
        "engine_fields": [],
        "differences": {
            "prefill": [] if differences is None else differences,
            "decode": [],
        },
        "passed": passed,
    }


def test_executor_requires_a_passing_setting_parity_report(tmp_path):
    report_path = tmp_path / "setting_parity.json"
    report_path.write_text(json.dumps(_report()))

    pd_capture_executor._validate_setting_parity(_plan(report_path))


def test_executor_rejects_setting_parity_difference(tmp_path):
    report_path = tmp_path / "setting_parity.json"
    report_path.write_text(json.dumps(_report(passed=False, differences=["seed"])))

    with pytest.raises(ValueError, match="did not pass"):
        pd_capture_executor._validate_setting_parity(_plan(report_path))
