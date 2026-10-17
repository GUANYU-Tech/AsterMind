"""The capture CLI brackets a capture with the host delivery-region probe.

The host delivery path drifts between days by much more than it varies inside
a session, so a probe value measured days away from a capture cannot be
compared against it.  Bracketing the capture is what makes the two share a
machine state; these tests pin the contract that makes the bracket reliable,
not the probe's numbers.
"""

from pathlib import Path
import importlib.util
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "experiments" / "run_sharegpt_ground_truth.py"


def _cli():
    spec = importlib.util.spec_from_file_location("sharegpt_gt_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["run_sharegpt_ground_truth.py"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def test_probe_runner_is_published_where_the_cli_looks_for_it():
    assert _cli().PROBE.is_file()


def test_probe_is_on_by_default_and_targets_the_capture_directory():
    plan = _cli()._probe_plan(["--model", "/models/m", "--output", "/out/gt.json"])
    assert plan == ("/models/m", Path("/out"))


def test_probe_can_be_turned_off():
    argv = ["--model", "/models/m", "--output", "/out/gt.json", "--no-host-path-probe"]
    assert _cli()._probe_plan(argv) is None


def test_probe_is_skipped_when_the_capture_arguments_are_incomplete():
    # The capture's own parser reports the missing argument; the probe must not
    # guess a destination from a half-specified command line.
    assert _cli()._probe_plan(["--model", "/models/m"]) is None
    assert _cli()._probe_plan(["--output", "/out/gt.json"]) is None


def test_probe_refuses_to_overwrite_an_existing_artifact(tmp_path):
    (tmp_path / "host_request_path_probe_before.json").write_text("{}\n")
    with pytest.raises(FileExistsError):
        _cli()._run_probe("/models/m", tmp_path, "before")


def _runner():
    spec = importlib.util.spec_from_file_location("delivery_region_runner", _cli().PROBE)
    # The module runs a probe at import, so only its pure helper is loaded here.
    source = _cli().PROBE.read_text()
    start = source.index("def derive_terms(")
    end = source.index("ic, sl = derive_terms(res)")
    namespace: dict = {}
    exec(compile(source[start:end], str(_cli().PROBE), "exec"), namespace)
    return namespace


def test_terms_are_derived_from_the_two_fanouts_the_estimator_reads():
    derive = _runner()["derive_terms"]
    # slope 1 ms per output between n=2 and n=64, intercept one step below v2
    intercept, slope = derive({1: 0.5, 2: 3.0, 64: 65.0})
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(2.0)


def test_a_non_positive_term_is_refused_rather_than_reported():
    derive = _runner()["derive_terms"]
    with pytest.raises(SystemExit) as caught:
        derive({2: 0.78, 8: 0.62})          # inverted: the region did not resolve
    assert "not resolved" in str(caught.value)


def test_a_single_usable_fanout_is_refused():
    derive = _runner()["derive_terms"]
    with pytest.raises(SystemExit):
        derive({1: 0.2, 2: 0.3})
