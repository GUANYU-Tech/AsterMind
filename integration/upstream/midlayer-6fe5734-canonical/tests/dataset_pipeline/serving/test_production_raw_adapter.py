import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

import dataset_pipeline.serving_dataset.compact_candidate_writer as writer
import dataset_pipeline.serving_dataset.production_raw_adapter as production_adapter
from dataset_pipeline.serving_dataset import (
    audit_production_raw_csv_bytes,
    build_production_compact_candidate,
    build_production_compact_payload,
    load_compact_case,
)


FIELDS = (
    "Query Time Stamp [ms]",
    "In Token [tokens]",
    "Out Token [tokens]",
    "Normal Input Tokens [tokens]",
    "Cached hit [tokens]",
    "Total Token [tokens]",
    "Inference Time [ms]",
    "Ttft [ms]",
    "gen_speed [token/s]",
)
ROWS = (
    ("10", "8", "2", "5", "3", "10", "9", "8", "0.25"),
    ("1", "4", "1", "4", "0", "5", "3", "2", "1"),
)


def _load_cli_module():
    path = Path("scripts/evaluation/serving/build_production_candidates.py")
    spec = importlib.util.spec_from_file_location("production_candidate_cli", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load Production candidate CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_bytes(rows=ROWS):
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows((FIELDS, *rows))
    return buffer.getvalue().encode()


def _write_raw_csv(tmp_path, rows=ROWS):
    path = tmp_path / "debug.csv"
    path.write_bytes(_raw_bytes(rows))
    return path


def _build_payload(source_bytes=None):
    if source_bytes is None:
        source_bytes = _raw_bytes()
    return build_production_compact_payload(
        source_bytes,
        source_name="debug.csv",
        model_id="glm-5.2",
        case_id="debug_workload_1s",
        schema_profile="glm_5_2_debug_csv_v1",
        upstream_success_provenance="synthetic successful-row export v1",
        converter_source_commit="a" * 40,
    )


def test_production_raw_audit_uses_the_strict_build_parser():
    source_bytes = _raw_bytes()
    audit = audit_production_raw_csv_bytes(
        source_bytes,
        schema_profile="glm_5_2_debug_csv_v1",
        source_name="debug.csv",
        upstream_success_provenance="synthetic successful-row export v1",
    )

    assert audit["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert audit["source_bytes"] == len(source_bytes)
    assert audit["statistics"] == {
        "request_count": 2,
        "arrival_time_total_ms": 11.0,
        "arrival_time_min_ms": 1.0,
        "arrival_time_max_ms": 10.0,
        "input_tokens_total": 12,
        "output_tokens_total": 3,
        "prefix_hit_tokens_total": 3,
        "logical_miss_tokens_total": 9,
    }
    assert audit["aggregate_consistency"]["logical_miss_identity"][
        "matching_rows"
    ] == 2
    assert "9.0" not in repr(audit)
    assert "8.0" not in repr(audit)
    assert "0.25" not in repr(audit)


def test_production_builders_reject_invalid_commit_before_parsing_or_writing(
    tmp_path, monkeypatch
):
    def unexpected_parse(*_args, **_kwargs):
        raise AssertionError("raw parser must not run for an invalid converter commit")

    monkeypatch.setattr(
        production_adapter, "_parse_production_raw_csv_bytes", unexpected_parse
    )
    kwargs = {
        "source_name": "debug.csv",
        "model_id": "glm-5.2",
        "case_id": "debug_workload_1s",
        "schema_profile": "glm_5_2_debug_csv_v1",
        "upstream_success_provenance": "synthetic successful-row export v1",
        "converter_source_commit": "deadbeef",
    }
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        build_production_compact_payload(_raw_bytes(), **kwargs)
    destination = tmp_path / "candidate"
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        build_production_compact_candidate(_raw_bytes(), destination, **kwargs)
    assert not destination.exists()


@pytest.mark.parametrize(
    "source_bytes",
    (
        b"secret-request-0,8,2,5,3,10,9,8,0.25\n",
        (
            b"secret-header-0,secret-header-1,secret-header-2,secret-header-3,"
            b"secret-header-4,secret-header-5,secret-header-6,secret-header-7,"
            b"secret-header-8\n0,8,2,5,3,10,9,8,0.25\n"
        ),
    ),
)
def test_production_raw_header_errors_do_not_echo_supplied_values(source_bytes):
    with pytest.raises(ValueError, match="header does not match") as captured:
        audit_production_raw_csv_bytes(
            source_bytes,
            schema_profile="glm_5_2_debug_csv_v1",
            source_name="debug.csv",
            upstream_success_provenance="synthetic successful-row export v1",
        )
    assert "secret" not in str(captured.value)
    assert "0.25" not in str(captured.value)


def test_production_raw_row_error_does_not_echo_supplied_values():
    source_bytes = _raw_bytes(
        (("0", "8", "2", "5", "3", "10", "secret-observation", "8", "1"),)
    )
    with pytest.raises(TypeError, match="inference_time_ms") as captured:
        _build_payload(source_bytes)
    assert "secret-observation" not in str(captured.value)


@pytest.mark.parametrize(
    ("row", "error"),
    (
        (("0", "8", "2", "4", "3", "10", "9", "8", "1"), "normal_input_tokens"),
        (("0", "8", "2", "5", "3", "9", "9", "8", "1"), "total_tokens"),
        (("0", "8", "2", "5", "3", "10", "nan", "8", "1"), "inference_time_ms"),
    ),
)
def test_production_raw_rejects_inconsistent_or_nonfinite_rows(row, error):
    with pytest.raises((TypeError, ValueError), match=error):
        _build_payload(_raw_bytes((row,)))


def test_production_raw_audit_rejects_nonfinite_arrival_total():
    rows = (
        ("1e308", "8", "2", "5", "3", "10", "9", "8", "1"),
        ("1e308", "4", "1", "4", "0", "5", "3", "2", "1"),
    )
    with pytest.raises(ValueError, match="arrival_time_total_ms must be finite"):
        audit_production_raw_csv_bytes(
            _raw_bytes(rows),
            schema_profile="glm_5_2_debug_csv_v1",
            source_name="debug.csv",
            upstream_success_provenance="synthetic successful-row export v1",
        )


def test_production_payload_preserves_rows_and_separates_observations():
    payload = _build_payload()

    assert [row.arrival_time_ms for row in payload.requests] == [10.0, 1.0]
    assert [row.input_tokens for row in payload.requests] == [8, 4]
    assert payload.identity["source_kind"] == "production"
    assert payload.identity["hardware_id"] is None
    assert payload.metrics.status == "available"
    assert all(
        payload.alignment_steps["phases"][phase]
        == {"status": "not_observed", "sample": None}
        for phase in ("prefill", "decode", "mixed")
    )
    metrics = payload.metrics.payload
    assert metrics["field_coverage"] == {
        "inference_time_ms": {
            "coverage": "complete",
            "unavailable_request_count": 0,
            "reason": None,
        },
        "ttft_ms": {
            "coverage": "complete",
            "unavailable_request_count": 0,
            "reason": None,
        },
        "generation_speed_tokens_per_s": {
            "coverage": "complete",
            "unavailable_request_count": 0,
            "reason": None,
        },
    }
    assert metrics["per_request"] == [
        {
            "request_idx": 0,
            "actual_output_tokens": None,
            "ttft_ms": 8.0,
            "tpot_ms": None,
            "e2e_latency_ms": None,
            "inference_time_ms": 9.0,
            "generation_speed_tokens_per_s": 0.25,
        },
        {
            "request_idx": 1,
            "actual_output_tokens": None,
            "ttft_ms": 2.0,
            "tpot_ms": None,
            "e2e_latency_ms": None,
            "inference_time_ms": 3.0,
            "generation_speed_tokens_per_s": 1.0,
        },
    ]
    assert metrics["summary"] == {
        "request_count": 2,
        "request_throughput_per_s": None,
        "output_token_throughput_per_s": None,
        "total_token_throughput_per_s": None,
    }


def test_production_payload_preserves_audited_ttft_sentinel_as_explained_null():
    row = (*ROWS[0][:7], "-1.0", ROWS[0][8])
    payload = _build_payload(_raw_bytes((row, ROWS[1])))

    metrics_availability = payload.metrics
    first, second = metrics_availability.payload["per_request"]
    assert first["ttft_ms"] is None
    assert metrics_availability.payload["field_coverage"]["ttft_ms"] == {
        "coverage": "partial",
        "unavailable_request_count": 1,
        "reason": "source_unavailable_sentinel_minus_one",
    }
    assert second["ttft_ms"] == 2.0


def test_production_raw_audit_reports_partial_observations_without_source_values():
    source_bytes = _raw_bytes(
        (
            (*ROWS[0][:7], "-1.0", ""),
            ROWS[1],
        )
    )
    audit = audit_production_raw_csv_bytes(
        source_bytes,
        schema_profile="glm_5_2_debug_csv_v1",
        source_name="debug.csv",
        upstream_success_provenance="synthetic successful-row export v1",
    )

    assert audit["observation_coverage"]["ttft_ms"] == {
        "coverage": "partial",
        "available_request_count": 1,
        "unavailable_request_count": 1,
        "unavailable_reasons": ["source_unavailable_sentinel_minus_one"],
    }
    assert audit["observation_coverage"]["generation_speed_tokens_per_s"] == {
        "coverage": "partial",
        "available_request_count": 1,
        "unavailable_request_count": 1,
        "unavailable_reasons": ["source_unavailable_sentinel_empty"],
    }


def test_production_partial_metrics_round_trip_records_manifest_reason(
    tmp_path, monkeypatch
):
    candidate_root = tmp_path / "profiler_output" / "serving_dataset_compaction"
    candidate_root.mkdir(parents=True)
    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    destination = candidate_root / "production" / "glm-5.2" / "partial"
    destination.parent.mkdir(parents=True)
    row = (*ROWS[0][:7], "-1.0", ROWS[0][8])

    output = build_production_compact_candidate(
        _raw_bytes((row, ROWS[1])),
        destination,
        source_name="debug.csv",
        model_id="glm-5.2",
        case_id="partial",
        schema_profile="glm_5_2_debug_csv_v1",
        upstream_success_provenance="synthetic successful-row export v1",
        converter_source_commit="a" * 40,
    )
    case = load_compact_case(output)

    assert case.manifest["metrics"]["status"] == "available"
    assert case.metrics["per_request"][0]["ttft_ms"] is None
    assert case.metrics["field_coverage"]["ttft_ms"] == {
        "coverage": "partial",
        "unavailable_request_count": 1,
        "reason": "source_unavailable_sentinel_minus_one",
    }


@pytest.mark.parametrize(
    "raw_ttft", ("-1", " -1.0 ", "-2", "nan", "inf", "bad", "")
)
def test_production_payload_rejects_unknown_invalid_ttft(raw_ttft):
    row = (*ROWS[0][:7], raw_ttft, ROWS[0][8])
    with pytest.raises((TypeError, ValueError), match="ttft_ms"):
        _build_payload(_raw_bytes((row,)))


@pytest.mark.parametrize("raw_speed", (" ", "-1", "nan", "inf", "bad"))
def test_production_payload_rejects_unknown_invalid_generation_speed(raw_speed):
    row = (*ROWS[0][:8], raw_speed)
    with pytest.raises((TypeError, ValueError), match="generation_speed_tokens_per_s"):
        _build_payload(_raw_bytes((row,)))


def test_production_payload_preserves_audited_empty_generation_speed_sentinel():
    row = (*ROWS[0][:8], "")
    payload = _build_payload(_raw_bytes((row, ROWS[1])))

    metrics_availability = payload.metrics
    first = metrics_availability.payload["per_request"][0]
    assert first["generation_speed_tokens_per_s"] is None
    assert metrics_availability.payload["field_coverage"][
        "generation_speed_tokens_per_s"
    ] == {
        "coverage": "partial",
        "unavailable_request_count": 1,
        "reason": "source_unavailable_sentinel_empty",
    }


def test_production_payload_derives_identity_and_records_from_exact_bytes():
    original = _raw_bytes()
    modified = _raw_bytes(((*ROWS[0][:6], "10", *ROWS[0][7:]), ROWS[1]))
    original_payload = _build_payload(original)
    modified_payload = _build_payload(modified)

    assert original_payload.raw_sources[0]["bytes"] == len(original)
    assert original_payload.raw_sources[0]["sha256"] == hashlib.sha256(
        original
    ).hexdigest()
    assert modified_payload.raw_sources[0]["sha256"] == hashlib.sha256(
        modified
    ).hexdigest()
    assert original_payload.raw_sources[0]["sha256"] != (
        modified_payload.raw_sources[0]["sha256"]
    )
    assert original_payload.metrics.payload["per_request"][0][
        "inference_time_ms"
    ] == 9.0
    assert modified_payload.metrics.payload["per_request"][0][
        "inference_time_ms"
    ] == 10.0


def test_production_candidate_round_trip_accepts_bom_and_preserves_physical_order(
    tmp_path, monkeypatch
):
    candidate_root = tmp_path / "profiler_output" / "serving_dataset_compaction"
    candidate_root.mkdir(parents=True)
    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    source_bytes = b"\xef\xbb\xbf" + _raw_bytes()
    destination = candidate_root / "production" / "glm-5.2" / "debug_workload_1s"
    destination.parent.mkdir(parents=True)

    output = build_production_compact_candidate(
        source_bytes,
        destination,
        source_name="debug.csv",
        model_id="glm-5.2",
        case_id="debug_workload_1s",
        schema_profile="glm_5_2_debug_csv_v1",
        upstream_success_provenance="synthetic successful-row export v1",
        converter_source_commit="a" * 40,
    )
    case = load_compact_case(output)

    assert [row.arrival_time_ms for row in case.requests] == [10.0, 1.0]
    assert case.manifest["raw_sources"][0]["bytes"] == len(source_bytes)
    assert case.manifest["raw_sources"][0]["sha256"] == hashlib.sha256(
        source_bytes
    ).hexdigest()
    assert [row["request_idx"] for row in case.metrics["per_request"]] == [0, 1]


def test_production_candidate_round_trip_is_valid_and_never_overwrites(
    tmp_path, monkeypatch
):
    candidate_root = tmp_path / "profiler_output" / "serving_dataset_compaction"
    candidate_root.mkdir(parents=True)
    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    destination = candidate_root / "production" / "glm-5.2" / "debug_workload_1s"
    destination.parent.mkdir(parents=True)
    kwargs = {
        "source_name": "debug.csv",
        "model_id": "glm-5.2",
        "case_id": "debug_workload_1s",
        "schema_profile": "glm_5_2_debug_csv_v1",
        "upstream_success_provenance": "synthetic successful-row export v1",
        "converter_source_commit": "a" * 40,
    }

    output = build_production_compact_candidate(
        _raw_bytes(), destination, **kwargs)
    case = load_compact_case(output)

    assert [row.arrival_time_ms for row in case.requests] == [10.0, 1.0]
    assert all(value is None for value in case.alignment_steps.values())
    assert case.metrics["requests_sha256"] == case.manifest["requests"]["sha256"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["identity"]["hardware_id"] is None
    assert manifest["metrics"]["status"] == "available"
    assert manifest["alignment_steps"]["phases"] == {
        phase: {"status": "not_observed", "rank_ids": []}
        for phase in ("prefill", "decode", "mixed")
    }
    with pytest.raises(FileExistsError, match="already exists"):
        build_production_compact_candidate(_raw_bytes(), destination, **kwargs)


def test_production_payload_requires_matching_identity_and_explicit_provenance():
    kwargs = {
        "source_name": "debug.csv",
        "model_id": "glm-5.2",
        "case_id": "debug_workload_1s",
        "schema_profile": "glm_5_2_debug_csv_v1",
        "upstream_success_provenance": "synthetic successful-row export v1",
        "converter_source_commit": "a" * 40,
    }
    with pytest.raises(ValueError, match="requires model_id"):
        build_production_compact_payload(
            _raw_bytes(), **{**kwargs, "model_id": "kimi-k2.6"})
    with pytest.raises(ValueError, match="upstream_success_provenance"):
        build_production_compact_payload(
            _raw_bytes(), **{**kwargs, "upstream_success_provenance": ""})
    with pytest.raises(ValueError, match="one path segment"):
        build_production_compact_payload(
            _raw_bytes(), **{**kwargs, "case_id": "../escape"})


def test_production_candidate_cli_reads_once_and_passes_exact_bytes(
    tmp_path, monkeypatch
):
    cli = _load_cli_module()
    source_path = _write_raw_csv(tmp_path)
    source_bytes = source_path.read_bytes()
    candidate_root = tmp_path / "profiler_output" / "serving" / "production"
    calls = []

    def fake_build(payload, output_dir, **kwargs):
        calls.append((payload, output_dir, kwargs))
        return output_dir

    monkeypatch.setattr(cli, "_CANDIDATE_ROOT", candidate_root)
    monkeypatch.setattr(cli, "build_production_compact_candidate", fake_build)
    output = cli.run_cli(
        (
            "--source",
            str(source_path),
            "--model-id",
            "glm-5.2",
            "--case-id",
            "debug_workload_1s",
            "--schema-profile",
            "glm_5_2_debug_csv_v1",
            "--upstream-success-provenance",
            "synthetic successful-row export v1",
            "--converter-source-commit",
            "a" * 40,
        )
    )

    expected = candidate_root / "glm-5.2" / "debug_workload_1s"
    assert output == expected
    assert calls == [
        (
            source_bytes,
            expected,
            {
                "source_name": "debug.csv",
                "model_id": "glm-5.2",
                "case_id": "debug_workload_1s",
                "schema_profile": "glm_5_2_debug_csv_v1",
                "upstream_success_provenance": (
                    "synthetic successful-row export v1"
                ),
                "converter_source_commit": "a" * 40,
            },
        )
    ]
