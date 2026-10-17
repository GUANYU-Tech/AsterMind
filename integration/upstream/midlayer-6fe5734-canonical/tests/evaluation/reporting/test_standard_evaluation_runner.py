from argparse import Namespace
from pathlib import Path

from evaluation.reporting import standard_evaluation_runner


def test_all_runs_kernel_before_serving_on_the_shared_cpu_set(
        monkeypatch, tmp_path):
    output_dir = tmp_path / "standard"
    calls = []
    args = Namespace(
        all=True,
        kernel_only=False,
        serving_only=False,
        output_dir=output_dir,
        jobs=45,
        kernel_jobs=45,
        case=[],
        reuse_serving_results=[],
        serving_python=Path("/test/python"),
        nvidia_h200_vllm_version=None,
        nvidia_h200_dataset_root=None,
    )

    monkeypatch.setattr(standard_evaluation_runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        standard_evaluation_runner, "_arguments", lambda argv: args)
    monkeypatch.setattr(
        standard_evaluation_runner, "_run_kernel",
        lambda output, stamp, jobs: (
            calls.append(("kernel", output, stamp, jobs))
            or {"kind": "kernel"}))
    monkeypatch.setattr(
        standard_evaluation_runner, "_run_serving",
        lambda output, stamp, jobs, cases, reuse, serving_python,
        h200_version, h200_root: (
            calls.append(
                ("serving", output, stamp, jobs, cases, reuse, serving_python,
                 h200_version, h200_root))
            or {"kind": "serving"}))
    monkeypatch.setattr(standard_evaluation_runner, "_source_commit", lambda: "commit")
    monkeypatch.setattr(standard_evaluation_runner, "timestamp_now", lambda: "now")

    def write_report(path, state):
        path.write_text(state["update_mode"] + "\n")
        return 0.0

    monkeypatch.setattr(
        standard_evaluation_runner, "write_kernel_evaluation_report", write_report)
    monkeypatch.setattr(
        standard_evaluation_runner, "write_serving_evaluation_report", write_report)

    reports = standard_evaluation_runner.run([])

    assert [call[0] for call in calls] == ["kernel", "serving"]
    assert calls[0][3] == 45
    assert calls[1][3:] == (45, (), (), Path("/test/python"), None, None)
    assert reports == (
        output_dir / "kernel_report.md",
        output_dir / "serving_report.md",
    )
