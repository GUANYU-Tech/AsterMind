import csv
import importlib.util
import json

import pytest


def _module():
    path = "scripts/evaluation/serving/build_slow_trace_candidate.py"
    spec = importlib.util.spec_from_file_location("slow_trace_candidate", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load slow-trace candidate builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trace(path, rows):
    fields = (
        "Query Time Stamp [ms]",
        "In Token [tokens]",
        "Out Token [tokens]",
        "Cached hit [tokens]",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_slow_trace_provenance_records_selected_token_conservation(tmp_path):
    module = _module()
    source = tmp_path / "source.csv"
    output = tmp_path / "candidate.csv"
    provenance = tmp_path / "candidate.provenance.json"
    _write_trace(source, [
        {
            "Query Time Stamp [ms]": "0",
            "In Token [tokens]": "4",
            "Out Token [tokens]": "2",
            "Cached hit [tokens]": "1",
        },
        {
            "Query Time Stamp [ms]": "1000",
            "In Token [tokens]": "8",
            "Out Token [tokens]": "3",
            "Cached hit [tokens]": "0",
        },
    ])

    module.build(source, output, provenance, window_s=0.5, anchor_qps=1.0)

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["source_token_totals"] == {
        "In Token [tokens]": 12,
        "Out Token [tokens]": 5,
        "Cached hit [tokens]": 1,
    }
    assert payload["selected_token_totals"] == {
        "In Token [tokens]": 4,
        "Out Token [tokens]": 2,
        "Cached hit [tokens]": 1,
    }
    assert payload["token_conservation"]["selection_preserves_row_values"]


def test_slow_trace_rejects_negative_token_values(tmp_path):
    module = _module()
    source = tmp_path / "source.csv"
    _write_trace(source, [{
        "Query Time Stamp [ms]": "0",
        "In Token [tokens]": "-1",
        "Out Token [tokens]": "2",
        "Cached hit [tokens]": "0",
    }, {
        "Query Time Stamp [ms]": "1000",
        "In Token [tokens]": "1",
        "Out Token [tokens]": "2",
        "Cached hit [tokens]": "0",
    }])

    with pytest.raises(ValueError, match="cannot be negative"):
        module.build(source, tmp_path / "candidate.csv", tmp_path / "p.json",
                     window_s=1.0, anchor_qps=1.0)
