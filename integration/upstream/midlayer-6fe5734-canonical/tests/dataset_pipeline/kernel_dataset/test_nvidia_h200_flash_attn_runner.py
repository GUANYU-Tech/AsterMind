"""CPU contracts for the H200 FA3 MLA decode branch (Q_LEN=1 and MTP verify)."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "scripts/kernel_dataset/nvidia_h200/flash_attn_runner.py"


def _runner(monkeypatch):
    calls = {"metadata": [], "attention": []}

    def get_scheduler_metadata(**kwargs):
        calls["metadata"].append(kwargs)
        return torch.zeros(9, dtype=torch.int32)

    def flash_attn_varlen_func(**kwargs):
        calls["attention"].append(kwargs)

    interface = ModuleType("vllm.vllm_flash_attn")
    interface.get_scheduler_metadata = get_scheduler_metadata
    interface.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", interface)
    spec = importlib.util.spec_from_file_location(
        "h200_flash_attn_runner_contract", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "installed_vllm_stack", lambda: "vllm030")

    def run(fn, *, mapping, split_evidence, **_):
        fn()
        calls["mapping"] = mapping
        calls["evidence"] = split_evidence()

    monkeypatch.setattr(module, "_run", run)
    return module, calls


def _mla(module, *, batch, query, q_lens=None, kv=64, splits=4,
         layout="mla"):
    module.run_flash_attn(
        batch, kv, 16, 1, 512, "bf16", query, splits, layout, "uniform",
        True, q_lens if q_lens is not None else [query] * batch,
        [kv] * batch, "H200_KERNEL::flash_attn::X", None, device="cpu")


def test_mla_verify_runs_the_vllm_decode_call_with_max_seqlen_q(monkeypatch):
    module, calls = _runner(monkeypatch)
    _mla(module, batch=2, query=3)
    (attention,) = calls["attention"]
    (metadata,) = calls["metadata"]
    assert attention["max_seqlen_q"] == 3
    assert attention["q"].shape == (6, 16, 64)
    assert attention["q_v"].shape == (6, 16, 512)
    assert attention["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert attention["seqused_k"].tolist() == [64, 64]
    assert (attention["causal"], attention["num_splits"]) == (True, 0)
    assert attention["scheduler_metadata"] is not None
    assert (metadata["max_seqlen_q"], metadata["causal"],
            metadata["num_splits"], metadata["headdim"],
            metadata["headdim_v"]) == (3, True, 0, 64, 512)
    arguments = calls["evidence"]["get_scheduler_metadata_arguments"]
    assert (arguments["max_seqlen_q"], arguments["causal"]) == (3, True)
    assert calls["mapping"]["max_seqlen_q"] == 3


def test_mla_single_token_decode_call_is_unchanged(monkeypatch):
    module, calls = _runner(monkeypatch)
    _mla(module, batch=2, query=1)
    (attention,) = calls["attention"]
    assert attention["max_seqlen_q"] == 1
    assert attention["q"].shape == (2, 16, 64)
    assert calls["metadata"][0]["max_seqlen_q"] == 1
    assert calls["mapping"] == {
        "a800_call": "vllm triton_mla decode_attention_fwd(num_kv_splits=4)",
        "h200_call": ("FLASH_ATTN_MLA decode: vllm FA3 flash_attn_varlen_func("
                      "q=q_pe, k=k_pe_cache, v=kv_c_cache, q_v=q_nope, "
                      "causal=True, fa_version=3, num_splits=0)"),
        "kv_lora_rank": 512, "qk_rope_head_dim": 64, "causal": True,
    }


def test_mla_rejects_non_uniform_or_prefill_length_queries(monkeypatch):
    module, calls = _runner(monkeypatch)
    with pytest.raises(ValueError, match="uniform Q lengths"):
        _mla(module, batch=2, query=3, q_lens=[3, 2])
    with pytest.raises(ValueError, match="Q_LEN <= 512"):
        _mla(module, batch=1, query=513, kv=1024)
    assert calls["attention"] == []


@pytest.mark.parametrize("layout", ["paged", "contiguous", "chunked_prefill"])
def test_multi_token_split_guard_still_applies_to_other_layouts(
        monkeypatch, layout):
    module, calls = _runner(monkeypatch)
    with pytest.raises(ValueError, match="prefill requires num_splits=1"):
        _mla(module, batch=1, query=3, layout=layout)
    assert calls["attention"] == []
