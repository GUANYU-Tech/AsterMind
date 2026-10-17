import importlib.util
import json
from pathlib import Path
import sys

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/evaluation/serving/validate_sharegpt_compact_alignment.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "validate_sharegpt_compact_alignment", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
validate_sharegpt_compact_alignment = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(validate_sharegpt_compact_alignment)


@pytest.mark.parametrize(
    ("mismatched_count", "unsupported_count", "expected_exit_code"),
    ((0, 0, 0), (1, 0, 1), (0, 1, 1)),
)
def test_alignment_cli_prints_report_and_enforces_gate(
    monkeypatch,
    capsys,
    mismatched_count,
    unsupported_count,
    expected_exit_code,
):
    passed = mismatched_count == 0 and unsupported_count == 0
    result = {
        "alignment_checked": passed,
        "formal_conclusion_allowed": passed,
        "mismatched_count": mismatched_count,
        "unsupported_count": unsupported_count,
    }
    monkeypatch.setattr(
        validate_sharegpt_compact_alignment,
        "run_compact_alignment_suite",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        validate_sharegpt_compact_alignment,
        "repository_source_commit",
        lambda _root: "0" * 40,
    )
    monkeypatch.setattr(sys, "argv", ["validate_sharegpt_compact_alignment.py"])

    exit_code = validate_sharegpt_compact_alignment.main()

    assert exit_code == expected_exit_code
    assert json.loads(capsys.readouterr().out) == result
