from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from dataset_pipeline.kernel_dataset.fa_serving_capture_audit import (
    _load_a800_ncu_raw,
)


ROOT = Path(__file__).resolve().parents[3]


def _write_raw(path, *, body_grid="(1, 5, 120)"):
    fields = [
        "ID", "Kernel Name", "Grid Size", "Block Size", "Device",
        "dram__bytes_read.sum", "dram__bytes_write.sum",
        "smsp__inst_executed.sum", "launch__registers_per_thread",
    ]
    rows = [
        {
            "ID": "", "Kernel Name": "", "Grid Size": "", "Block Size": "",
            "Device": "", "dram__bytes_read.sum": "Mbyte",
            "dram__bytes_write.sum": "Kbyte", "smsp__inst_executed.sum": "inst",
            "launch__registers_per_thread": "register/thread",
        },
        {
            "ID": "0", "Kernel Name": "body", "Grid Size": body_grid,
            "Block Size": "(128, 1, 1)", "Device": "0",
            "dram__bytes_read.sum": "1", "dram__bytes_write.sum": "2",
            "smsp__inst_executed.sum": "3", "launch__registers_per_thread": "4",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_a800_raw_parser_retains_only_explicit_non_duration_metrics(tmp_path):
    raw = tmp_path / "counter.raw.csv"
    _write_raw(raw)

    records = _load_a800_ncu_raw(raw)

    assert records == [{
        "ordinal": 0,
        "identity": "body",
        "grid": (1, 5, 120),
        "block": (128, 1, 1),
        "device": 0.0,
        "counters": {
            "dram__bytes_read.sum": float(1024 ** 2),
            "dram__bytes_write.sum": float(2 * 1024),
            "smsp__inst_executed.sum": 3.0,
            "launch__registers_per_thread": 4.0,
        },
    }]


def test_a800_raw_parser_rejects_non_contiguous_ordinals(tmp_path):
    raw = tmp_path / "counter.raw.csv"
    _write_raw(raw)
    text = raw.read_text(encoding="utf-8").replace("\n0,body,", "\n1,body,")
    raw.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="ordinal"):
        _load_a800_ncu_raw(raw)


def test_closure_cli_does_not_require_unused_source_snapshot(tmp_path):
    case_id = "case"
    digest = "a" * 64
    source = "b" * 64
    timing = tmp_path / "timing.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "closure.json"
    timing.write_text(
        "{" +
        f'"status":"pass","aggregation":{{"pass":true}},"hardware":{{"hardware_id":"nvidia_a800"}},"case_id":"{case_id}","workload_digest":"{digest}","source_snapshot":{{"sha256":"{source}"}}' +
        "}", encoding="utf-8")
    candidate.write_text(
        "{" +
        f'"hardware_id":"nvidia_a800","case_id":"{case_id}","workload_digest":"{digest}","source_snapshot_sha256":"{source}","counter_contract":{{"duration_consumed":false}},"kernels":[{{}}],"timing_launch_join":{{"windows":[{{"launch_count":1}}]}}' +
        "}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/kernel_dataset/fa_serving_capture_audit_runner.py"),
            "--hardware", "nvidia_a800",
            "--case-id", case_id,
            "--workload-digest", digest,
            "--candidate-input", str(candidate),
            "--timing-audit", str(timing),
            "--closure-output", str(output),
        ],
        check=True, capture_output=True, text=True, cwd=ROOT)

    assert completed.stdout.strip() == str(output.resolve())
    assert output.is_file()
