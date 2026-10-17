import json
import math

import pytest

from dataset_pipeline.kernel_dataset.dataset_writer import (
    DatasetCompletenessError,
    build_kernel_evaluation_timing,
    publish_complete_dataset,
    write_complete_candidate,
)
from dataset_pipeline.profiler.nsys_capture import (
    get_capture_output_dir,
    require_capture_output_dir,
)
from src.kernel.dataset_storage import load_kernel_dataset_document


def test_complete_dataset_publishes_compressed_canonical_artifact(tmp_path):
    candidate = tmp_path / "candidate.json"
    canonical = tmp_path / "canonical.json.zst"

    result = publish_complete_dataset(
        {"case_b": {"value": 2}, "case_a": {"value": 1}},
        ("case_a", "case_b"),
        candidate,
        canonical,
    )

    assert result == canonical
    assert list(json.loads(candidate.read_text())) == ["case_b", "case_a"]
    assert list(load_kernel_dataset_document(canonical)) == ["case_a", "case_b"]


def test_partial_candidate_does_not_publish_canonical_dataset(tmp_path):
    candidate = tmp_path / "candidate.json"
    canonical = tmp_path / "canonical.json"
    canonical.write_text('{"existing": true}')

    with pytest.raises(DatasetCompletenessError) as raised:
        publish_complete_dataset(
            {"case_a": {"value": 1}},
            ("case_a", "case_b"),
            candidate,
            canonical,
        )

    assert raised.value.missing == ("case_b",)
    assert json.loads(candidate.read_text()) == {"case_a": {"value": 1}}
    assert json.loads(canonical.read_text()) == {"existing": True}


def test_candidate_with_unexpected_case_does_not_publish(tmp_path):
    candidate = tmp_path / "candidate.json"
    canonical = tmp_path / "canonical.json"
    results = {
        "case_a": {"value": 1},
        "case_b": {"value": 2},
        "obsolete_case": {"value": 3},
    }

    with pytest.raises(DatasetCompletenessError) as raised:
        publish_complete_dataset(
            results, ("case_a", "case_b"), candidate, canonical)

    assert raised.value.missing == ()
    assert raised.value.unexpected == ("obsolete_case",)
    assert not canonical.exists()


def test_complete_staging_candidate_is_ordered_without_canonical_publish(
        tmp_path):
    candidate = tmp_path / "candidate.json"
    canonical = tmp_path / "canonical.json"
    result = write_complete_candidate(
        {"case_b": {"value": 2}, "case_a": {"value": 1}},
        ("case_a", "case_b"),
        candidate,
    )

    assert result == candidate
    assert list(json.loads(candidate.read_text())) == ["case_a", "case_b"]
    assert not canonical.exists()


def test_incomplete_staging_candidate_fails_loud(tmp_path):
    candidate = tmp_path / "candidate.json"
    with pytest.raises(DatasetCompletenessError) as raised:
        write_complete_candidate(
            {"case_a": {"value": 1}},
            ("case_a", "case_b"),
            candidate,
        )
    assert raised.value.missing == ("case_b",)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_candidate_is_never_serialized(tmp_path, value):
    candidate = tmp_path / "candidate.json"
    canonical = tmp_path / "canonical.json"

    with pytest.raises(ValueError, match="JSON compliant"):
        publish_complete_dataset(
            {"case_a": {"value": value}},
            ("case_a",),
            candidate,
            canonical,
        )

    assert not candidate.exists()
    assert not canonical.exists()


def test_serving_profiler_output_is_confined_to_canonical_root(tmp_path):
    root = get_capture_output_dir("nvidia_a800").resolve()
    assert require_capture_output_dir(root, "nvidia_a800") == root
    assert require_capture_output_dir(
        root / "case", "nvidia_a800") == root / "case"
    with pytest.raises(ValueError, match="must stay under"):
        require_capture_output_dir(tmp_path, "nvidia_a800")
    with pytest.raises(NotImplementedError, match="not implemented"):
        get_capture_output_dir("metax_c500")


@pytest.mark.parametrize(
    "hardware_id,source",
    (
        ("nvidia_a800", "nsys_nvtx_kernel_median"),
        ("metax_c500", "mcTracer"),
        ("ascend_910b", "msprof_task_duration"),
        ("samsung_hbm_pim", "unindp_simulator_latency"),
        ("groq_lpu", "timeloop_lpu_scaled_hybrid_latency"),
        ("ucb_gemmini", "timeloop_mapper_simulated_latency"),
    ),
)
def test_kernel_evaluation_timing_is_self_identifying(hardware_id, source):
    kernels = [
        {"Metadata": {"Kernel Name": "first"}},
        {"Metadata": {"Kernel Name": "second"}},
    ]

    assert build_kernel_evaluation_timing(
        hardware_id, kernels, [11, 22]) == {
            "timing": {
                "granularity": "kernel",
                "source": source,
                "total_duration_ns": 33,
                "kernels": [
                    {
                        "kernel_idx": 0,
                        "kernel_name": "first",
                        "duration_ns": 11,
                    },
                    {
                        "kernel_idx": 1,
                        "kernel_name": "second",
                        "duration_ns": 22,
                    },
                ],
            },
        }


def test_kernel_evaluation_timing_rejects_count_mismatch():
    with pytest.raises(ValueError, match="count does not match"):
        build_kernel_evaluation_timing(
            "metax_c500",
            [{"Metadata": {"Kernel Name": "kernel"}}],
            [11, 22],
        )
