from pathlib import Path
import shutil

import pytest

from dataset_pipeline.serving_dataset.scheduler_capture_publisher import (
    publish_scheduler_capture_candidates,
)


_ROOT = Path(__file__).resolve().parents[3]
_CASE = ("qwen3-4b", "tp1_qps4")


def _roots(tmp_path):
    candidates = tmp_path / "profiler_output" / "serving_dataset_compaction"
    shutil.copytree(
        _ROOT / "serving_dataset/sharegpt/nvidia_h200" / Path(*_CASE),
        candidates / "nvidia_h200" / Path(*_CASE))
    canonical = tmp_path / "sharegpt"
    shutil.copytree(
        _ROOT / "serving_dataset/sharegpt/nvidia_a800" / Path(*_CASE),
        canonical / "nvidia_a800" / Path(*_CASE))
    return candidates, canonical


def test_new_hardware_case_requires_a_reference_with_the_same_requests(
        tmp_path):
    candidates, canonical = _roots(tmp_path)
    case = "nvidia_h200/" + "/".join(_CASE)
    with pytest.raises(ValueError, match="no reference hardware"):
        publish_scheduler_capture_candidates(
            candidate_root=candidates, dataset_root=canonical, cases=[case])
    with pytest.raises(ValueError, match="reference hardware must differ"):
        publish_scheduler_capture_candidates(
            candidate_root=candidates, dataset_root=canonical, cases=[case],
            new_case_reference_hardware="nvidia_h200")
    published = publish_scheduler_capture_candidates(
        candidate_root=candidates, dataset_root=canonical, cases=[case],
        new_case_reference_hardware="nvidia_a800")
    assert published == (canonical / case,)
    source = candidates / case
    assert sorted(path.name for path in (canonical / case).iterdir()) == sorted(
        path.name for path in source.iterdir())
    for path in source.iterdir():
        assert (canonical / case / path.name).read_bytes() == path.read_bytes()
