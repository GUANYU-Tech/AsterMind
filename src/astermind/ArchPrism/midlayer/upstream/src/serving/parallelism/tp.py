"""Tensor-parallel plan rules."""

from dataclasses import replace


def apply(plan, config):
    if config.tensor_parallel_size == 1:
        return plan
    return replace(
        plan,
        enabled_strategies=plan.enabled_strategies + ("tensor_parallel",),
        communication_rules=plan.communication_rules + (
            "row_parallel_all_reduce", "vocab_parallel_all_gather"),
    )
