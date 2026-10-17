"""Boundary tests for the MsprofActiveCyclesDropout / NA-outlier tolerance
added in c66bfdf8c.

Both tolerance paths are narrow: they drop exactly one isolated glitch only
when the surviving evidence is an unambiguous strict majority, and re-raise
unchanged in every other split (tie, multiple glitches, a non-"NA" outlier).
These tests exercise both sides of that boundary directly against
`_aggregate_parsed` and `_parse_kernel_dir`, using fully synthetic
placeholder data -- no real msprof capture or SHA is pinned here.
"""
import os

import pytest

from dataset_pipeline.kernel_dataset import ascend_910b


def _invocation(raw):
    return {"duration_us": 1.0, "op_name": "synthetic_op", "op_type": "vector", "raw": raw}


# --- _aggregate_parsed: raw-field NA/numeric boundary --------------------


def test_aggregate_parsed_tolerates_single_na_outlier():
    parsed = [
        _invocation({"f": 10.0}),
        _invocation({"f": 12.0}),
        _invocation({"f": "NA"}),
    ]
    with pytest.warns(UserWarning, match="dropping a single 'NA' outlier"):
        agg = ascend_910b._aggregate_parsed(parsed, "synthetic_op", "vector")
    assert agg["raw"]["f"] == pytest.approx(11.0)


def test_aggregate_parsed_raises_on_tied_na_split():
    parsed = [
        _invocation({"f": 10.0}),
        _invocation({"f": "NA"}),
    ]
    with pytest.raises(TypeError, match="mixed numeric/NA raw field"):
        ascend_910b._aggregate_parsed(parsed, "synthetic_op", "vector")


def test_aggregate_parsed_raises_on_multiple_na():
    parsed = [
        _invocation({"f": 10.0}),
        _invocation({"f": 12.0}),
        _invocation({"f": "NA"}),
        _invocation({"f": "NA"}),
    ]
    with pytest.raises(TypeError, match="mixed numeric/NA raw field"):
        ascend_910b._aggregate_parsed(parsed, "synthetic_op", "vector")


def test_aggregate_parsed_raises_on_non_na_non_numeric_outlier():
    parsed = [
        _invocation({"f": 10.0}),
        _invocation({"f": 12.0}),
        _invocation({"f": "ERR"}),
    ]
    with pytest.raises(TypeError, match="mixed numeric/NA raw field"):
        ascend_910b._aggregate_parsed(parsed, "synthetic_op", "vector")


# --- _parse_kernel_dir: per-invocation dropout-count boundary ------------


def _make_inv_dirs(tmp_path, n):
    kernel_dir = tmp_path / "kernel_dir"
    for i in range(n):
        (kernel_dir / str(i)).mkdir(parents=True)
    return kernel_dir


def _fake_invocation_factory(dropout_indices):
    def fake_parse_kernel_invocation(inv_dir):
        if os.path.basename(inv_dir) in dropout_indices:
            raise ascend_910b.MsprofActiveCyclesDropout(
                f"synthetic dropout for {inv_dir}")
        return _invocation({})
    return fake_parse_kernel_invocation


def test_parse_kernel_dir_tolerates_strict_minority_dropout(tmp_path, monkeypatch):
    kernel_dir = _make_inv_dirs(tmp_path, 3)
    monkeypatch.setattr(
        ascend_910b, "_parse_kernel_invocation",
        _fake_invocation_factory({"1"}))

    with pytest.warns(UserWarning, match="dropping invocation"):
        result = ascend_910b._parse_kernel_dir(str(kernel_dir))

    assert len(result) == 1
    assert len(result[0]["invocations"]) == 2


def test_parse_kernel_dir_reraises_on_tied_dropout(tmp_path, monkeypatch):
    kernel_dir = _make_inv_dirs(tmp_path, 2)
    monkeypatch.setattr(
        ascend_910b, "_parse_kernel_invocation",
        _fake_invocation_factory({"1"}))

    with pytest.raises(ascend_910b.MsprofActiveCyclesDropout, match="synthetic dropout"):
        ascend_910b._parse_kernel_dir(str(kernel_dir))


def test_parse_kernel_dir_reraises_when_dropouts_outnumber_successes(tmp_path, monkeypatch):
    kernel_dir = _make_inv_dirs(tmp_path, 3)
    monkeypatch.setattr(
        ascend_910b, "_parse_kernel_invocation",
        _fake_invocation_factory({"1", "2"}))

    with pytest.raises(ascend_910b.MsprofActiveCyclesDropout, match="synthetic dropout"):
        ascend_910b._parse_kernel_dir(str(kernel_dir))
