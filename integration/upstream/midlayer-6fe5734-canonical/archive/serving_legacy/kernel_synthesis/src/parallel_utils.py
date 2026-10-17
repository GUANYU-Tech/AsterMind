"""Utilities for GPU and parallelism configuration validation."""
from __future__ import annotations

from typing import Any


class ParallelConfigError(ValueError):
    pass


def parallel_worker_count(parallelism: dict[str, Any]) -> int:
    """Return total GPU worker count: TP * PP * DP."""
    tp = _coerce_positive_int(parallelism.get("tensor_parallel_size", 1), "tensor_parallel_size")
    pp = _coerce_positive_int(parallelism.get("pipeline_parallel_size", 1), "pipeline_parallel_size")
    dp = _coerce_positive_int(parallelism.get("data_parallel_size", 1), "data_parallel_size")
    return tp * pp * dp


def validate_gpu_capacity(
    label: str,
    gpus: list[int],
    parallelism: dict[str, Any],
) -> None:
    """Raise ParallelConfigError if gpus is insufficient for the parallelism config."""
    required = parallel_worker_count(parallelism)
    if len(gpus) < required:
        raise ParallelConfigError(
            f"{label}: requires {required} GPUs "
            f"(TP={parallelism.get('tensor_parallel_size', 1)} x "
            f"PP={parallelism.get('pipeline_parallel_size', 1)} x "
            f"DP={parallelism.get('data_parallel_size', 1)}) "
            f"but only {len(gpus)} provided: {gpus}"
        )


def validate_parallel_semantics(engine_args: dict[str, Any], role: str) -> None:
    """Validate decode_context_parallel_size constraints against tensor_parallel_size."""
    tp = engine_args.get("tensor_parallel_size", 1)
    dcp = engine_args.get("decode_context_parallel_size", 1)
    if dcp < 1:
        raise ParallelConfigError(
            f"{role}: decode_context_parallel_size must be >= 1, got {dcp}"
        )
    if dcp > tp:
        raise ParallelConfigError(
            f"{role}: decode_context_parallel_size ({dcp}) must be "
            f"<= tensor_parallel_size ({tp})"
        )
    if tp % dcp != 0:
        raise ParallelConfigError(
            f"{role}: tensor_parallel_size ({tp}) must be divisible "
            f"by decode_context_parallel_size ({dcp})"
        )


def _coerce_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ParallelConfigError(f"{label} must be an integer >= 1")
    return value
