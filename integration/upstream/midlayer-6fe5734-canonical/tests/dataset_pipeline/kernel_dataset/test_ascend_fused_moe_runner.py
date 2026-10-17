"""Pure contracts for the 910B FusedMoE launch-observation runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "scripts/kernel_dataset/ascend_910b/fused_moe_runner.py"


def _runner_module(monkeypatch: pytest.MonkeyPatch):
    # The route constructor is deliberately device-independent.  Stub the
    # device-only imports so this contract stays runnable on non-Ascend CI.
    torch_module = ModuleType("torch")
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    spec = importlib.util.spec_from_file_location(
        "ascend_fused_moe_runner_contract", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_census_route_has_distinct_topk_ids(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner_module(monkeypatch)
    counts = [3, 2, 2, 1, 0, 0]

    rows = runner._topk_rows_from_tokens_per_expert(
        counts, tokens=4, experts=6, topk=2)

    assert len(rows) == 4
    assert all(len(set(row)) == 2 for row in rows)
    assert [sum(expert in row for row in rows) for expert in range(6)] == counts


@pytest.mark.parametrize(
    ("counts", "tokens", "experts", "topk", "error"),
    [
        ((1, 1), 1, 2, 2, TypeError),
        ([1, 1], 1, 3, 2, ValueError),
        ([2, 0], 1, 2, 2, ValueError),
        ([True, 1], 1, 2, 2, TypeError),
    ],
)
def test_exact_census_route_rejects_invalid_contract(
        monkeypatch: pytest.MonkeyPatch, counts, tokens, experts, topk,
        error) -> None:
    runner = _runner_module(monkeypatch)

    with pytest.raises(error):
        runner._topk_rows_from_tokens_per_expert(
            counts, tokens=tokens, experts=experts, topk=topk)
