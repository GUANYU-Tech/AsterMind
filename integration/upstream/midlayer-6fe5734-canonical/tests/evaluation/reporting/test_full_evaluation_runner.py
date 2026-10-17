import pytest

from pathlib import Path

from evaluation.reporting import full_evaluation_runner


def test_kernel_card_forwards_case_worker_timing_and_semantic_contract(
        monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        full_evaluation_runner, "kernel_evaluation_dataset_available",
        lambda hardware_id: True)
    monkeypatch.setattr(
        full_evaluation_runner, "_run",
        lambda command, root, quiet: (calls.append(command) or (0, "ok")))
    monkeypatch.setattr(
        full_evaluation_runner, "_parse_kernel_table",
        lambda text: [("ALL", 1, (1.0, 1.0, 1.0),
                       (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), None)])

    status, _, _, coverage = full_evaluation_runner._run_one_kernel_card(
        "nvidia_a800", Path("/repo"), case_jobs=45,
        case_timing_dir=tmp_path / "case_timing",
        semantic_output_dir=tmp_path / "semantic")

    assert status == full_evaluation_runner.PASS
    assert coverage == {"scored": [], "unsupported_cases": {}}
    assert calls == [[
        full_evaluation_runner.sys.executable,
        "scripts/evaluation/kernel/run_kernel_evaluation.py",
        "--hardware-id", "nvidia_a800", "--mode", "standard",
        "--jobs", "45", "--case-timing-output",
        str(tmp_path / "case_timing" / "nvidia_a800.json"),
        "--semantic-output", str(tmp_path / "semantic" / "nvidia_a800.json"),
    ]]


def test_forward_coverage_sections_bind_to_their_table_rows():
    rows = [("ALL", 5, None, (1.0, 1.0, 1.0), None, None),
            ("├─ GEMM", 3, None, (1.0, 1.0, 1.0), None, None)]
    text = "\n".join((
        "═" * 10,
        "  Forward rows scored over a subset of their N (unsupported cases "
        "excluded)",
        "═" * 10,
        "  row   0 scored     4 / 5     ALL",
        "═" * 10,
        "  Evaluation timing targets",
        "═" * 10,
        "  Forward unsupported cases (explicit, excluded from scoring)",
        "═" * 10,
        "  ALL                          cases=    1 / 5",
        "  FLASHATTN                    cases=    1",
        "    reason cases=    1: unsupported template",
        "═" * 10,
    ))

    coverage = full_evaluation_runner._parse_forward_coverage(text, rows)

    assert coverage == {"scored": [[0, 4, 5]],
                        "unsupported_cases": {"ALL": 1, "FLASHATTN": 1}}
    with pytest.raises(ValueError, match="does not match its row"):
        full_evaluation_runner._parse_forward_coverage(
            text.replace("row   0", "row   1"), rows)
