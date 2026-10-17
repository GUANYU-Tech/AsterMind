import hashlib
import json
from pathlib import Path

import pytest

import dataset_pipeline.serving_dataset.sharegpt_candidate_builder as builder
from dataset_pipeline.serving_dataset.compact_candidate_writer import (
    AvailabilityPayload,
    CompactCasePayload,
)
from dataset_pipeline.serving_dataset.compact_case import CanonicalRequestRow


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _write_legacy_official_truth_matrix(root):
    request_bytes = (
        b"Query Time Stamp [ms],In Token [tokens],Out Token [tokens],Cached hit [tokens]\n"
        b"0,8,2,0\n"
    )
    request_sha = hashlib.sha256(request_bytes).hexdigest()
    source = {
        "archive_id": "production-ground-truth",
        "role": "production-ground-truth",
        "relative_path": "ground_truth.json",
        "bytes": 17,
        "sha256": "a" * 64,
        "tool": "vllm-serving-runtime",
        "tool_version": "0.11.0",
    }
    for spec in (value for value in builder.SHAREGPT_CASE_SPECS if value.publish_metrics):
        case = root / spec.hardware_id / spec.raw_model_id / spec.case_id
        case.mkdir(parents=True)
        (case / "requests.csv").write_bytes(request_bytes)
        metrics = {
            "schema_version": 1,
            "requests_sha256": request_sha,
            "metric_provenance": {
                "source_kind": "serving_runtime",
                "source_id": "production-ground-truth",
                "source_sha256": "a" * 64,
            },
            "per_request": [{
                "request_idx": 0,
                "actual_output_tokens": 2,
                "ttft_ms": 1.0,
                "tpot_ms": 1.0,
                "e2e_latency_ms": 2.0,
                "inference_time_ms": None,
                "generation_speed_tokens_per_s": None,
            }],
            "summary": {
                "request_count": 1,
                "request_throughput_per_s": 1.0,
                "output_token_throughput_per_s": 2.0,
                "total_token_throughput_per_s": 10.0,
            },
        }
        metrics_bytes = _json_bytes(metrics)
        alignment = case / "alignment"
        alignment.mkdir()
        (alignment / "metrics.json").write_bytes(metrics_bytes)
        manifest = {
            "schema_version": 1,
            "identity": {
                "source_kind": "sharegpt",
                "hardware_id": spec.hardware_id,
                "model_id": spec.raw_model_id,
                "case_id": spec.case_id,
            },
            "requests": {
                "path": "requests.csv",
                "requests_sha256": request_sha,
                "row_count": 1,
            },
            "alignment": {
                "metrics": {
                    "status": "available",
                    "path": "alignment/metrics.json",
                    "sha256": hashlib.sha256(metrics_bytes).hexdigest(),
                },
            },
            "raw_sources": [source],
        }
        (case / "manifest.json").write_bytes(_json_bytes(manifest))


def test_frozen_matrix_has_51_lowercase_cases():
    specs = builder.SHAREGPT_CASE_SPECS
    assert len(specs) == 51
    assert len({(s.hardware_id, s.model_id, s.case_id) for s in specs}) == 51
    assert all(s.model_id == s.model_id.lower() for s in specs)
    assert sum(s.publish_metrics for s in specs) == 45


@pytest.mark.parametrize("legacy_flag", ("--phase", "--target-qps", "--inspect-source-set"))
def test_candidate_cli_rejects_removed_partial_build_paths(legacy_flag):
    with pytest.raises(SystemExit):
        builder.main((legacy_flag, "first"))


def test_all_case_builder_rejects_invalid_commit_before_writing(
    tmp_path, monkeypatch
):
    candidate_root = tmp_path / "candidates"
    monkeypatch.setattr(
        builder,
        "validate_official_truth_snapshot",
        lambda _root: pytest.fail("invalid commit must fail before snapshot parsing"),
    )
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        builder.build_all_sharegpt_candidates(
            raw_root=tmp_path / "raw",
            converter_source_commit="deadbeef",
            candidate_root=candidate_root,
            canonical_dataset_root=tmp_path / "canonical",
        )
    assert not candidate_root.exists()


def test_all_case_selection_supports_hardware_model_and_case_filters():
    selected = builder._select_case_specs(
        ("metax_c500",),
        ("qwen3-32b",),
        ("tp2_ep1_qps4",),
    )
    assert [(spec.hardware_id, spec.model_id, spec.case_id) for spec in selected] == [
        ("metax_c500", "qwen3-32b", "tp2_ep1_qps4")
    ]
    with pytest.raises(ValueError, match="duplicates"):
        builder._select_case_specs(("metax_c500", "metax_c500"), None, None)
    with pytest.raises(ValueError, match="matches no frozen case"):
        builder._select_case_specs(
            ("ascend_910b",), ("llama-3.1-70b",), ("tp4_qps1",)
        )


def test_all_case_cli_emits_machine_readable_summary(monkeypatch, capsys, tmp_path):
    candidate = tmp_path / "candidate" / "nvidia_a800/qwen3-4b/tp1_qps4"
    candidate.mkdir(parents=True)
    (candidate / "manifest.json").write_text("{}\n")
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return (candidate,)

    monkeypatch.setattr(builder, "build_all_sharegpt_candidates", fake_build)
    exit_code = builder.main((
        "--raw-root", str(tmp_path / "raw"),
        "--candidate-root", str(
            tmp_path / "profiler_output" / "serving_dataset_compaction"
        ),
        "--canonical-dataset-root", str(tmp_path / "canonical"),
        "--hardware-id", "nvidia_a800",
        "--model-id", "qwen3-4b",
        "--case-id", "tp1_qps4",
        "--converter-source-commit", "a" * 40,
    ))
    assert exit_code == 0
    assert calls == [{
        "raw_root": tmp_path / "raw",
        "converter_source_commit": "a" * 40,
        "candidate_root": tmp_path / "profiler_output" / "serving_dataset_compaction",
        "canonical_dataset_root": tmp_path / "canonical",
        "hardware_ids": ["nvidia_a800"],
        "model_ids": ["qwen3-4b"],
        "case_ids": ["tp1_qps4"],
        "jobs": 1,
    }]
    summary = __import__("json").loads(capsys.readouterr().out)
    assert summary["status"] == "ok"
    assert summary["successful_count"] == 1
    assert summary["cases"][0]["files"] == {"manifest.json": 3}


def test_all_case_cli_returns_nonzero_with_structured_failures(
    monkeypatch, capsys, tmp_path
):
    failure = {
        "successful_count": 0,
        "failure_count": 1,
        "failures": [{
            "cell_id": "metax_c500/qwen3-4b/tp1_qps4",
            "error_type": "ValueError",
            "message": "bad raw",
        }],
    }
    monkeypatch.setattr(
        builder,
        "build_all_sharegpt_candidates",
        lambda **_: (_ for _ in ()).throw(RuntimeError(__import__("json").dumps(failure))),
    )
    exit_code = builder.main((
        "--raw-root", str(tmp_path / "raw"),
        "--canonical-dataset-root", str(tmp_path / "snapshot"),
        "--converter-source-commit", "a" * 40,
    ))
    assert exit_code == 1
    summary = __import__("json").loads(capsys.readouterr().out)
    assert summary["status"] == "failed"
    assert summary["failure_count"] == 1


def test_all_case_cli_reports_invalid_snapshot_as_json(capsys, tmp_path):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    exit_code = builder.main((
        "--raw-root", str(tmp_path / "raw"),
        "--canonical-dataset-root", str(snapshot_root),
        "--converter-source-commit", "a" * 40,
    ))

    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["failures"][0]["cell_id"] is None
    assert report["failures"][0]["error_type"] == "ValueError"


def test_official_truth_snapshot_contains_exact_45_case_inventory(tmp_path):
    canonical_root = tmp_path / "canonical"
    snapshot_root = tmp_path / "snapshot"
    _write_legacy_official_truth_matrix(canonical_root)

    output = builder.build_official_truth_snapshot(
        source_dataset_root=canonical_root,
        snapshot_root=snapshot_root,
    )
    index = builder.validate_official_truth_snapshot(output)

    assert len(index) == 45
    assert (
        "nvidia_a800/qwen3-4b/tp1_qps4" in index
        and "ascend_910b/qwen3-32b/tp2_ep1_qps16" in index
    )
    files = tuple(path for path in output.rglob("*") if path.is_file())
    assert len(files) == 91
    assert not any("alignment" in path.parts for path in files)
    metrics = json.loads((
        output / "nvidia_a800/qwen3-4b/tp1_qps4/metrics.json"
    ).read_text())
    assert metrics["schema_version"] == 2
    assert metrics["metric_provenance"]["profiler_attached"] is False


def test_official_truth_snapshot_rejects_changed_payload(tmp_path):
    canonical_root = tmp_path / "canonical"
    snapshot_root = tmp_path / "snapshot"
    _write_legacy_official_truth_matrix(canonical_root)
    builder.build_official_truth_snapshot(
        source_dataset_root=canonical_root,
        snapshot_root=snapshot_root,
    )
    requests = snapshot_root / "metax_c500/qwen3-8b/tp1_qps16/requests.csv"
    requests.write_bytes(requests.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="byte count is inconsistent"):
        builder.validate_official_truth_snapshot(snapshot_root)


def test_official_truth_snapshot_rejects_duplicate_json_keys(tmp_path):
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "official_truth_snapshot.json").write_text(
        '{"schema_version":1,"schema_version":1}'
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        builder.validate_official_truth_snapshot(snapshot_root)


def test_snapshot_cli_reports_every_file_size(tmp_path, capsys):
    canonical_root = tmp_path / "canonical"
    snapshot_root = tmp_path / "snapshot"
    _write_legacy_official_truth_matrix(canonical_root)

    exit_code = builder.main((
        "--build-official-truth-snapshot",
        "--canonical-dataset-root", str(canonical_root),
        "--snapshot-root", str(snapshot_root),
    ))

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["case_count"] == 45
    assert len(report["files"]) == 91
    assert report["total_bytes"] == sum(report["files"].values())


def test_all_case_builder_validates_snapshot_once_before_workers(
    tmp_path, monkeypatch
):
    snapshot_index = {
        f"{spec.hardware_id}/{spec.model_id}/{spec.case_id}": {
            "hardware_id": spec.hardware_id,
            "model_id": spec.model_id,
            "case_id": spec.case_id,
        }
        for spec in builder.SHAREGPT_CASE_SPECS if spec.publish_metrics
    }
    validations = []
    worker_arguments = []
    monkeypatch.setattr(
        builder,
        "validate_official_truth_snapshot",
        lambda root: validations.append(Path(root)) or snapshot_index,
    )
    monkeypatch.setattr(
        builder,
        "_build_one_sharegpt_candidate",
        lambda arguments: worker_arguments.append(arguments) or Path("candidate"),
    )
    snapshot_root = tmp_path / "snapshot"

    outputs = builder.build_all_sharegpt_candidates(
        raw_root=tmp_path / "raw",
        converter_source_commit="a" * 40,
        candidate_root=tmp_path / "profiler_output/serving_dataset_compaction",
        canonical_dataset_root=snapshot_root,
        hardware_ids=("nvidia_a800",),
        model_ids=("qwen3-4b",),
        case_ids=("tp1_qps4",),
    )

    assert outputs == (Path("candidate"),)
    assert validations == [snapshot_root]
    assert worker_arguments[0][4] == snapshot_index[
        "nvidia_a800/qwen3-4b/tp1_qps4"
    ]


def test_snapshot_truth_binding_replaces_requests_and_extends_raw_set(tmp_path):
    canonical_root = tmp_path / "canonical"
    snapshot_root = tmp_path / "snapshot"
    _write_legacy_official_truth_matrix(canonical_root)
    builder.build_official_truth_snapshot(
        source_dataset_root=canonical_root,
        snapshot_root=snapshot_root,
    )
    index = builder.validate_official_truth_snapshot(snapshot_root)
    spec = next(
        value for value in builder.SHAREGPT_CASE_SPECS
        if (value.hardware_id, value.model_id, value.case_id)
        == ("nvidia_a800", "qwen3-4b", "tp1_qps4")
    )
    raw_source = {
        "archive_id": "raw-capture",
        "role": "operators",
        "relative_path": "operators.jsonl",
        "bytes": 1,
        "sha256": "b" * 64,
        "tool": "trace",
        "tool_version": "1",
    }
    payload = CompactCasePayload(
        identity={
            "source_kind": "sharegpt",
            "hardware_id": spec.hardware_id,
            "model_id": spec.model_id,
            "raw_model_id": spec.raw_model_id,
            "case_id": spec.case_id,
            "model_revision": None,
            "tokenizer_revision": None,
        },
        runtime={},
        requests=(CanonicalRequestRow(1.0, 1, 1, 0),),
        request_field_provenance={},
        cached_hit_value_evidence={
            "raw_source_set_sha256": "0" * 64,
        },
        metrics=AvailabilityPayload("unavailable", "not-bound", None),
        alignment_steps={
            "operator_signatures": {},
            "launch_signatures": {},
            "phases": {
                phase: {"status": "not_observed", "sample": None}
                for phase in ("prefill", "decode", "mixed")
            },
        },
        raw_sources=(raw_source,),
        provenance_gaps=(),
        converter={},
    )

    bound = builder._bind_existing_official_truth(
        payload,
        spec,
        snapshot_root=snapshot_root,
        snapshot_entry=index["nvidia_a800/qwen3-4b/tp1_qps4"],
    )

    assert bound.requests == (CanonicalRequestRow(0.0, 8, 2, 0),)
    assert bound.metrics.status == "available"
    assert bound.metrics.payload["per_request"][0]["ttft_ms"] == 1.0
    assert [source["archive_id"] for source in bound.raw_sources] == [
        "raw-capture", "production-ground-truth",
    ]
