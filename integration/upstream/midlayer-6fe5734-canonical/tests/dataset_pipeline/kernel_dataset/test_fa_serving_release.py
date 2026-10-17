from pathlib import Path

import pytest

from dataset_pipeline.kernel_dataset.fa_serving_release import (
    _repository_relative_paths,
    merge_release_into_canonical,
)
from src.kernel.dataset_storage import compress_kernel_dataset_content


def test_release_normalizes_nested_repository_paths(tmp_path):
    raw = tmp_path / "profiler_output" / "capture.json"
    value = {
        "candidate_path": str(raw),
        "windows": [{"path": str(raw), "sha256": "a" * 64}],
    }

    assert _repository_relative_paths(value, tmp_path) == {
        "candidate_path": "profiler_output/capture.json",
        "windows": [{
            "path": "profiler_output/capture.json",
            "sha256": "a" * 64,
        }],
    }


def test_release_rejects_absolute_paths_outside_repository(tmp_path):
    outside = tmp_path.parent / "outside.json"

    with pytest.raises(ValueError, match="escapes repository root"):
        _repository_relative_paths(str(outside), tmp_path)


def _write_zstd_json(path, payload):
    import json

    path.write_bytes(compress_kernel_dataset_content(
        json.dumps(payload).encode("utf-8")))


def test_release_merges_exactly_40_records_into_canonical(tmp_path):
    canonical = tmp_path / "flash_attn.json.zst"
    release = tmp_path / "release.json.zst"
    _write_zstd_json(canonical, {"old": {"op_info": {}, "kernels": []}})
    records = {
        f"case_{index:02d}": {
            "op_info": {},
            "capture_evidence": {"counter": {"status": "available"}},
        }
        for index in range(40)
    }
    _write_zstd_json(release, {
        "schema_version": 1,
        "kind": "fa_serving_capture_release",
        "hardware_id": "nvidia_a800",
        "capture_manifest": "manifest.json",
        "capture_manifest_sha256": "a" * 64,
        "case_count": 40,
        "counter_available_case_count": 40,
        "counter_unavailable_case_count": 0,
        "records": records,
    })

    merge_release_into_canonical(
        canonical_path=canonical, release_path=release,
        hardware_id="nvidia_a800")

    from src.kernel.dataset_storage import load_kernel_dataset_document
    merged = load_kernel_dataset_document(canonical)
    assert len(merged) == 41
    assert merged["case_00"]["capture_evidence"]["release"] == {
        "schema_version": 1,
        "kind": "fa_serving_capture_release",
        "hardware_id": "nvidia_a800",
        "capture_manifest": "manifest.json",
        "capture_manifest_sha256": "a" * 64,
    }


def test_release_merge_rejects_case_collision(tmp_path):
    canonical = tmp_path / "flash_attn.json.zst"
    release = tmp_path / "release.json.zst"
    records = {
        f"case_{index:02d}": {
            "op_info": {},
            "capture_evidence": {"counter": {"status": "available"}},
        }
        for index in range(40)
    }
    _write_zstd_json(canonical, {"case_00": {}})
    _write_zstd_json(release, {
        "schema_version": 1,
        "kind": "fa_serving_capture_release",
        "hardware_id": "nvidia_a800",
        "capture_manifest": "manifest.json",
        "capture_manifest_sha256": "a" * 64,
        "case_count": 40,
        "counter_available_case_count": 40,
        "counter_unavailable_case_count": 0,
        "records": records,
    })

    with pytest.raises(ValueError, match="collides"):
        merge_release_into_canonical(
            canonical_path=canonical, release_path=release,
            hardware_id="nvidia_a800")


@pytest.mark.parametrize(
    "hardware_id,available,unavailable",
    (
        ("nvidia_a800", 40, 0),
        ("metax_c500", 39, 1),
        ("ascend_910b", 40, 0),
    ),
)
def test_canonical_fa_contains_complete_serving_capture_release(
        hardware_id, available, unavailable):
    from src.kernel.dataset_storage import load_kernel_dataset_document

    dataset = load_kernel_dataset_document(
        Path("kernel_dataset") / hardware_id / "flash_attn.json.zst")
    records = [
        record for record in dataset.values()
        if "capture_evidence" in record
    ]
    statuses = [
        record["capture_evidence"]["counter"]["status"]
        for record in records
    ]

    assert len(dataset) == 334
    assert len(records) == 40
    assert statuses.count("available") == available
    assert statuses.count("unavailable") == unavailable
    assert all(
        record["capture_evidence"]["release"]["hardware_id"] == hardware_id
        for record in records)
