from types import SimpleNamespace

import pytest

from evaluation.serving.metrics.pd_disaggregation_study import (
    _is_discarded_async_terminal_artifact_request,
    _scheduler_pipeline_depth_from_pd_case,
)


def _cell():
    return SimpleNamespace(
        hardware=SimpleNamespace(hardware_id="ascend_910b"),
        model=SimpleNamespace(dataset_model_id="llama-3.1-8b"),
        case_id="tp1_qps64",
        requests_sha256="a" * 64,
    )


def _pd_case(source_kind):
    return SimpleNamespace(manifest={
        "schema_version": 6,
        "identity": {
            "hardware_id": "ascend_910b",
            "model_id": "llama-3.1-8b",
            "case_id": "tp1_qps64",
            "source_kind": source_kind,
        },
        "requests": {"sha256": "a" * 64},
        "runtime": {
            "instances": {
                "prefill": {"scheduler_pipeline_depth": 2},
                "decode": {"scheduler_pipeline_depth": 2},
            },
        },
    })


def test_pd_study_accepts_canonical_pd_source_kind():
    assert _scheduler_pipeline_depth_from_pd_case(
        _pd_case("sharegpt_pd_disaggregated"), _cell()) == 2


def test_pd_study_rejects_colocated_source_kind():
    with pytest.raises(ValueError, match="source_kind mismatch"):
        _scheduler_pipeline_depth_from_pd_case(_pd_case("sharegpt"), _cell())


def test_pd_conservation_recognizes_discarded_async_terminal_tail():
    scheduled = {
        "request_id": "request",
        "phase": "decode",
        "input_tokens": 0,
        "output_tokens": 1,
        "recomputed_tokens": 0,
        "prefix_replay_tokens": 0,
        "resumed_from_preemption": False,
        "raw_model_output_tokens": 1,
        "mtp_proposer_mode": "disabled",
        "mtp_acceptance_source_kind": "disabled",
        "mtp_acceptance_source_id": "disabled",
        "mtp_proposed_tokens": 0,
        "mtp_verified_tokens": 0,
        "mtp_accepted_tokens": 0,
        "mtp_proposer_tokens": 0,
        "mtp_next_proposed_tokens": 0,
    }

    assert _is_discarded_async_terminal_artifact_request(
        scheduled, {"request": 0}, ())
    assert not _is_discarded_async_terminal_artifact_request(
        scheduled, {"request": 1}, ())
