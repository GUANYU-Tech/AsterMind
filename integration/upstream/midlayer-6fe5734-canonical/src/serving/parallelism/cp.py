"""Context-parallel plan rules."""


def apply(plan, config):
    if config.context_parallel_size == 1:
        return plan
    raise NotImplementedError("Context Parallel is not implemented")
