"""Pipeline-parallel plan rules matching pinned vLLM placement."""

from dataclasses import replace


def pipeline_layer_ranges(
        num_layers, pipeline_parallel_size, layer_partition=None):
    """Return vLLM's default contiguous decoder-layer partitions."""

    for name, value in (
            ("num_layers", num_layers),
            ("pipeline_parallel_size", pipeline_parallel_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if pipeline_parallel_size > num_layers:
        raise ValueError(
            "pipeline_parallel_size cannot exceed num_layers")
    if layer_partition is None:
        layers_per_partition = num_layers // pipeline_parallel_size
        partitions = [layers_per_partition] * pipeline_parallel_size
        remaining_layers = num_layers % pipeline_parallel_size
        for offset in range(2, remaining_layers + 2):
            partitions[-offset] += 1
    else:
        if (not isinstance(layer_partition, (tuple, list))
                or len(layer_partition) != pipeline_parallel_size
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 1 for value in layer_partition)
                or sum(layer_partition) != num_layers):
            raise ValueError(
                "pipeline layer partition must contain one positive layer "
                "count per stage and conserve num_layers")
        partitions = list(layer_partition)
    ranges = []
    layer_start = 0
    for layer_count in partitions:
        layer_end = layer_start + layer_count
        ranges.append((layer_start, layer_end))
        layer_start = layer_end
    if layer_start != num_layers:
        raise RuntimeError("pipeline layer placement does not conserve layers")
    return tuple(ranges)


def pipeline_layer_ranges_for_model(model, pipeline_parallel_size):
    """Return the source-backed PP partition for a model/runtime pair."""

    if model.model_type == "glm_moe_dsa" and pipeline_parallel_size == 2:
        # Explicit source candidate VLLM_PP_LAYER_PARTITION=38,40. Default
        # 39,39 starts stage1 at a shared-indexer layer without local top-k.
        # This is a declared partition, not an observed runtime placement.
        layer_partition = (38, 40)
    elif (model.model_type == "glm_moe_dsa"
            and pipeline_parallel_size == 4):
        layer_partition = (18, 20, 20, 20)
    else:
        layer_partition = None
    return pipeline_layer_ranges(
        model.num_hidden_layers,
        pipeline_parallel_size,
        layer_partition,
    )


def apply(plan, config, layer_partition=None):
    if config.pipeline_parallel_size == 1:
        return plan
    ranges = pipeline_layer_ranges(
        plan.num_layers, config.pipeline_parallel_size, layer_partition)
    stages = tuple(
        stage
        for stage, (layer_start, layer_end) in enumerate(ranges)
        for unused_layer in range(layer_start, layer_end)
    )
    return replace(
        plan,
        layer_to_pipeline_stage=stages,
        enabled_strategies=plan.enabled_strategies + ("pipeline_parallel",),
        communication_rules=plan.communication_rules + ("pipeline_activation_send",),
    )
