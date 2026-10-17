"""Shared MoE top-k emulation helpers."""


def _topk_seed(num_tokens: int, num_experts: int, topk: int) -> int:
    return (
        (num_tokens * 73_856_093)
        ^ (num_experts * 19_349_663)
        ^ (topk * 83_492_791)
    ) & 0xFFFFFFFF


def emulation_fused_topk(hidden_states, gating_output, topk, renormalize,
                         indices_type=None):
    """Return deterministic no-replacement expert routing for emulated MoE."""
    import torch

    num_tokens = hidden_states.size(0)
    num_experts = gating_output.size(1)
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk > num_experts:
        raise ValueError(
            f"topk ({topk}) must be less than or equal to num_experts ({num_experts})"
        )

    ids_dtype = indices_type if indices_type is not None else torch.int32
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_topk_seed(num_tokens, num_experts, topk))

    if num_tokens == 0:
        topk_ids_cpu = torch.empty((0, topk), dtype=ids_dtype)
    else:
        rows = [
            torch.randperm(num_experts, generator=gen, dtype=ids_dtype)[:topk]
            for _ in range(num_tokens)
        ]
        topk_ids_cpu = torch.stack(rows, dim=0)

    topk_weights_cpu = torch.ones((num_tokens, topk), dtype=torch.float32) / topk
    if renormalize:
        topk_weights_cpu = topk_weights_cpu / topk_weights_cpu.sum(
            dim=-1, keepdim=True)

    topk_ids = topk_ids_cpu.to(device=hidden_states.device)
    topk_weights = topk_weights_cpu.to(device=hidden_states.device)
    token_expert_indices = topk_ids.clone()
    return topk_weights, topk_ids, token_expert_indices
