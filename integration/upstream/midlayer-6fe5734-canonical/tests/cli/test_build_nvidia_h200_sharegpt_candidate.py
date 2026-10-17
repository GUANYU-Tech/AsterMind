import importlib.util
import json
from pathlib import Path
import sys

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/evaluation/serving/build_nvidia_h200_sharegpt_candidate.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "build_nvidia_h200_sharegpt_candidate", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
builder = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(builder)

_GATE = "scheduler_trace_observer_effect.json"
_RUN_FILES = ("ground_truth.json", "scheduler_steps.raw.jsonl", _GATE)


def _case(root: Path, case: str = "qwen3-4b/tp1_qps4", runtime: str = "0.30.0") -> Path:
    case_dir = root / runtime / case
    case_dir.mkdir(parents=True)
    (case_dir / "case_capture.json").write_text(json.dumps({
        "schema_version": 1, "case": case, "hardware_id": "nvidia_h200",
        "runtime_version": runtime,
    }))
    return case_dir


def _trace_on(case_dir: Path, name: str, files=_RUN_FILES, **record_overrides) -> Path:
    directory = case_dir / name
    directory.mkdir()
    for filename in files:
        (directory / filename).write_text("{}\n")
    if name != "trace_on":
        original = json.loads((case_dir / "case_capture.json").read_text())
        record = {
            "schema_version": 1, "case": original["case"],
            "hardware_id": "nvidia_h200",
            "runtime_version": original["runtime_version"],
            "rerun_of": "trace_on", "rerun_dir": name,
            "started": "2026-09-27T00:00:00Z", "ended": "2026-09-27T00:10:00Z",
            "stages": [],
        }
        record.update(record_overrides)
        record = {key: value for key, value in record.items() if value is not None}
        (case_dir / f"case_capture.{name}.json").write_text(json.dumps(record))
    return directory


def test_default_and_finished_tagged_rerun_directories_are_accepted(tmp_path):
    case_dir = _case(tmp_path)
    default = _trace_on(case_dir, "trace_on")
    assert builder._trace_on_dir(case_dir, "trace_on", _RUN_FILES) == default
    tagged = _trace_on(case_dir, "trace_on_fix1")
    assert builder._trace_on_dir(case_dir, "trace_on_fix1", _RUN_FILES) == tagged


@pytest.mark.parametrize("name", (
    "trace_off", "trace_on_", "trace_on-fix1", "trace_on_fix1/..", "../trace_on",
    "trace_on_fix 1", "trace_onfix1", "",
))
def test_malformed_directory_name_fails_before_touching_files(tmp_path, name):
    case_dir = _case(tmp_path)
    with pytest.raises(ValueError, match="trace-on directory name"):
        builder._trace_on_dir(case_dir, name, _RUN_FILES)


@pytest.mark.parametrize("missing", _RUN_FILES)
def test_missing_directory_or_required_file_fails(tmp_path, missing):
    case_dir = _case(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        builder._trace_on_dir(case_dir, "trace_on_fix1", _RUN_FILES)
    _trace_on(case_dir, "trace_on_fix1",
              files=tuple(name for name in _RUN_FILES if name != missing))
    with pytest.raises(FileNotFoundError, match=f"lacks {missing}"):
        builder._trace_on_dir(case_dir, "trace_on_fix1", _RUN_FILES)


def test_tagged_directory_requires_its_rerun_record(tmp_path):
    case_dir = _case(tmp_path)
    _trace_on(case_dir, "trace_on_fix1")
    (case_dir / "case_capture.trace_on_fix1.json").unlink()
    with pytest.raises(FileNotFoundError, match="rerun trace-on record"):
        builder._trace_on_dir(case_dir, "trace_on_fix1", _RUN_FILES)


@pytest.mark.parametrize("override", (
    {"rerun_dir": "trace_on_lag1"},
    {"case": "qwen3-8b/tp1_qps4"},
    {"runtime_version": "0.11.0"},
    {"rerun_of": "trace_off"},
))
def test_rerun_record_must_describe_this_case_and_directory(tmp_path, override):
    case_dir = _case(tmp_path)
    _trace_on(case_dir, "trace_on_fix1", **override)
    with pytest.raises(ValueError, match="does not record a rerun"):
        builder._trace_on_dir(case_dir, "trace_on_fix1", _RUN_FILES)


def test_unfinished_rerun_is_rejected(tmp_path):
    case_dir = _case(tmp_path)
    _trace_on(case_dir, "trace_on_fix1", ended=None)
    with pytest.raises(ValueError, match="did not finish"):
        builder._trace_on_dir(case_dir, "trace_on_fix1", _RUN_FILES)


def _run_main(monkeypatch, tmp_path, case_dir: Path, *extra: str) -> dict:
    calls: dict = {}

    def fail_alignment(*_args, **_kwargs):
        raise AssertionError("options must be validated before stage 1 runs")

    def fake_candidate(**kwargs):
        calls.update(kwargs)
        return tmp_path / "candidate"

    if "--reuse-alignment-stage" in extra:
        monkeypatch.setattr(builder, "load_compact_case", lambda _path: None)
    monkeypatch.setattr(builder, "_write_alignment_stage", fail_alignment)
    monkeypatch.setattr(builder, "_raw_consistency", lambda *_args: {})
    monkeypatch.setattr(builder, "build_scheduler_capture_candidate", fake_candidate)
    monkeypatch.setattr(builder, "_summary", lambda _output: {})
    monkeypatch.setattr(sys, "argv", [
        "build_nvidia_h200_sharegpt_candidate.py", "--case-dir", str(case_dir),
        "--work-root", str(tmp_path / "work"), "--converter-source-commit", "b" * 40,
        *extra,
    ])
    assert builder.main() == 0
    return calls


def test_cli_binds_the_selected_rerun_and_its_own_gate(monkeypatch, tmp_path):
    case_dir = _case(tmp_path)
    _trace_on(case_dir, "trace_on")
    tagged = _trace_on(case_dir, "trace_on_fix1")
    calls = _run_main(monkeypatch, tmp_path, case_dir,
                      "--trace-on-dir-name", "trace_on_fix1", "--reuse-alignment-stage")
    # The v5 writer records ``<parent dir>/<file>`` of these paths as the
    # manifest capture run_id and raw_sources relative_path.
    assert calls["trace_on_result_path"] == tagged / "ground_truth.json"
    assert calls["scheduler_trace_path"] == tagged / "scheduler_steps.raw.jsonl"
    assert calls["observer_effect_path"] == tagged / _GATE
    assert calls["observer_validation_status"] == "measured_for_this_case"


def test_cli_default_binds_trace_on(monkeypatch, tmp_path):
    case_dir = _case(tmp_path)
    default = _trace_on(case_dir, "trace_on")
    calls = _run_main(monkeypatch, tmp_path, case_dir, "--reuse-alignment-stage")
    assert calls["trace_on_result_path"] == default / "ground_truth.json"
    assert calls["observer_effect_path"] == default / _GATE


def test_cli_selects_representative_gate_directory_independently(monkeypatch, tmp_path):
    case_dir = _case(tmp_path)
    tagged = _trace_on(case_dir, "trace_on_rankfix1",
                       files=("ground_truth.json", "scheduler_steps.raw.jsonl"))
    representative = _case(tmp_path, case="qwen3-8b/tp1_qps4")
    _trace_on(representative, "trace_on")
    representative_gate = _trace_on(representative, "trace_on_lag1")
    calls = _run_main(
        monkeypatch, tmp_path, case_dir, "--reuse-alignment-stage",
        "--trace-on-dir-name", "trace_on_rankfix1",
        "--observer-representative-case-dir", str(representative),
        "--observer-representative-trace-on-dir-name", "trace_on_lag1")
    assert calls["trace_on_result_path"] == tagged / "ground_truth.json"
    assert calls["observer_effect_path"] == representative_gate / _GATE
    assert calls["observer_validation_status"] == "inherited_from_representative_case"
    default_calls = _run_main(
        monkeypatch, tmp_path, case_dir, "--reuse-alignment-stage",
        "--trace-on-dir-name", "trace_on_rankfix1",
        "--observer-representative-case-dir", str(representative))
    assert default_calls["observer_effect_path"] == representative / "trace_on" / _GATE


def test_cli_rejects_representative_directory_without_representative(
    monkeypatch, tmp_path
):
    case_dir = _case(tmp_path)
    _trace_on(case_dir, "trace_on")
    with pytest.raises(ValueError, match="requires --observer-representative-case-dir"):
        _run_main(monkeypatch, tmp_path, case_dir,
                  "--observer-representative-trace-on-dir-name", "trace_on")


def test_cli_validates_the_directory_before_stage_one(monkeypatch, tmp_path):
    case_dir = _case(tmp_path)
    _trace_on(case_dir, "trace_on")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _run_main(monkeypatch, tmp_path, case_dir, "--trace-on-dir-name", "trace_on_fix1")
    # The case's own gate is required only when it is used.
    _trace_on(case_dir, "trace_on_fix1", files=("ground_truth.json",
                                                "scheduler_steps.raw.jsonl"))
    with pytest.raises(FileNotFoundError, match=f"lacks {_GATE}"):
        _run_main(monkeypatch, tmp_path, case_dir, "--trace-on-dir-name", "trace_on_fix1")
