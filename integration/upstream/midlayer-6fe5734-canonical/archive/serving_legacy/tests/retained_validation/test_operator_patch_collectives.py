"""Unit tests for collective-communication capture in the in-EngineCore
operator patch (src/serve_trace/_operator_patch.py).

These exercise the patch functions directly against fake module/class objects
(no real vLLM process needed), verifying that:
  - TP all-reduce is captured as an `all_reduce` record with real msg_bytes;
  - EP all-to-all is captured as an `all_to_all` record (dispatch/combine
    positions) WHEN AND ONLY WHEN GroupCoordinator.dispatch/combine is actually
    invoked. The naive EP backend does not call them, so nothing is fabricated.
"""
import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

import src.serve_trace._operator_patch as op


def _reset_emit(tmp_path):
    """Point the patch's emit sink at a fresh temp file, reset step counters."""
    path = str(tmp_path / "ops.jsonl")
    op._GLOBAL["path"] = path
    op._GLOBAL["step_id"] = 0
    op._GLOBAL["op_idx"] = 0
    return path


def _records(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def test_all_reduce_wrapper_emits_record(tmp_path):
    path = _reset_emit(tmp_path)
    called = {}

    def _orig(x):
        called["x"] = x
        return x

    wrapped = op._make_all_reduce_wrapper(_orig)
    t = torch.zeros((8, 16), dtype=torch.float16)
    out = wrapped(t)
    assert out is t and called["x"] is t  # original still invoked, passthrough

    recs = _records(path)
    assert len(recs) == 1
    r = recs[0]
    assert r["op_type"] == "all_reduce" and r["position"] == "tp"
    assert r["attrs"]["msg_bytes"] == 8 * 16 * 2  # fp16 = 2 bytes
    assert r["attrs"]["seq_len"] == 8


def _fake_group_coordinator():
    """A minimal stand-in for vllm GroupCoordinator with dispatch/combine."""
    class GroupCoordinator:
        def dispatch(self, hidden_states, router_logits,
                     is_sequence_parallel=False):
            return hidden_states, router_logits

        def combine(self, hidden_states, is_sequence_parallel=False):
            return hidden_states

    return GroupCoordinator


def test_ep_dispatch_combine_captured_when_called(tmp_path):
    path = _reset_emit(tmp_path)
    cls = _fake_group_coordinator()
    op._patch_group_coordinator(cls)
    inst = cls()

    hs = torch.zeros((8, 32), dtype=torch.float16)
    rl = torch.zeros((8, 4), dtype=torch.float16)
    out_hs, out_rl = inst.dispatch(hs, rl)
    assert out_hs is hs and out_rl is rl  # passthrough preserved
    out2 = inst.combine(hs)
    assert out2 is hs

    recs = _records(path)
    assert [r["op_type"] for r in recs] == ["all_to_all", "all_to_all"]
    assert [r["position"] for r in recs] == ["dispatch", "combine"]
    assert recs[0]["attrs"]["msg_bytes"] == 8 * 32 * 2


def test_ep_not_fabricated_when_not_called(tmp_path):
    # If the EP collective is never invoked (the naive backend path), no
    # all_to_all record is emitted -- matching nsys (zero all-to-all kernels).
    path = _reset_emit(tmp_path)
    cls = _fake_group_coordinator()
    op._patch_group_coordinator(cls)
    # ... patched, but never called.
    assert _records(path) == []


def test_patch_is_idempotent(tmp_path):
    # Double-patching must not stack wrappers (would double-count records).
    path = _reset_emit(tmp_path)
    cls = _fake_group_coordinator()
    op._patch_group_coordinator(cls)
    op._patch_group_coordinator(cls)
    inst = cls()
    inst.combine(torch.zeros((4, 8), dtype=torch.float16))
    recs = _records(path)
    assert len(recs) == 1  # exactly one, not two
