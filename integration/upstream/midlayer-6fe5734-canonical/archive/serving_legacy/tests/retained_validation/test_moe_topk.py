from pathlib import Path

import pytest
import torch


def test_emulation_fused_topk_samples_without_replacement_and_preserves_contract():
    from src.serve_trace.moe_topk import emulation_fused_topk

    hidden_states = torch.empty((8, 4), device="cpu")
    gating_output = torch.zeros((8, 4), device=hidden_states.device)

    weights, ids, token_expert_indices = emulation_fused_topk(
        hidden_states,
        gating_output,
        topk=3,
        renormalize=True,
        indices_type=torch.int64,
    )

    assert weights.shape == (8, 3)
    assert ids.shape == (8, 3)
    assert token_expert_indices.shape == (8, 3)
    assert weights.device == hidden_states.device
    assert ids.device == hidden_states.device
    assert token_expert_indices.device == hidden_states.device
    assert ids.dtype == torch.int64
    assert token_expert_indices.dtype == torch.int64
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8))
    assert torch.equal(token_expert_indices, ids)
    for row in ids.tolist():
        assert len(set(row)) == len(row)


def test_emulation_fused_topk_is_deterministic_and_defaults_to_int32():
    from src.serve_trace.moe_topk import emulation_fused_topk

    hidden_states = torch.empty((6, 2))
    gating_output = torch.zeros((6, 5))

    first = emulation_fused_topk(hidden_states, gating_output, topk=2, renormalize=False)
    second = emulation_fused_topk(hidden_states, gating_output, topk=2, renormalize=False)

    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert first[1].dtype == torch.int32
    assert first[2].dtype == torch.int32


def test_emulation_fused_topk_rejects_topk_larger_than_num_experts():
    from src.serve_trace.moe_topk import emulation_fused_topk

    hidden_states = torch.empty((2, 2))
    gating_output = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="topk"):
        emulation_fused_topk(hidden_states, gating_output, topk=4, renormalize=True)


def test_vllm_emulation_entrypoints_use_shared_moe_topk_helper():
    root = Path(__file__).resolve().parents[2]
    for rel in [
        "scripts/serve_trace/trace_eager.py",
        "scripts/serve_trace/trace_serving.py",
        "scripts/serve_trace/_disagg_instance.py",
    ]:
        text = (root / rel).read_text()
        assert "emulation_fused_topk" in text
        assert "torch.randint" not in text
