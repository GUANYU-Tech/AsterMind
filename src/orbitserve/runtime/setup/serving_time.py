"""Length-table interpolation used by Runtime service scheduling."""

from __future__ import annotations

import math
from typing import Any


def prefill_full_time_s(table: tuple[dict[str, Any], ...], prompt_length: int) -> float:
    """Return cumulative full-prompt time, interpolating cumulative time."""

    length = int(prompt_length)
    if length < 0:
        raise ValueError("prompt_length must be non-negative")
    if length == 0:
        return 0.0
    points = _time_points(table, "prefill_tokens_per_s")
    return _interpolate(points, float(length))


def prefill_suffix_time_s(
    table: tuple[dict[str, Any], ...],
    *,
    prompt_length: int,
    cached_prefix_tokens: int,
) -> float:
    """Derive suffix compute time from the one-dimensional cumulative table."""

    prompt = int(prompt_length)
    cached = max(0, min(prompt, int(cached_prefix_tokens)))
    return max(0.0, prefill_full_time_s(table, prompt) - prefill_full_time_s(table, cached))


def decode_time_s(
    table: tuple[dict[str, Any], ...],
    *,
    prompt_length: int,
    output_tokens: int,
) -> float:
    """Integrate the piecewise-linear per-token time over generated context."""

    start = float(prompt_length)
    count = int(output_tokens)
    if start < 0 or count < 0:
        raise ValueError("decode lengths must be non-negative")
    if count == 0:
        return 0.0
    points = _time_points(table, "decode_tokens_per_s")
    end = start + count
    if start < 1.0:
        start = 1.0
        end = start + count
    if end > points[-1][0]:
        raise ValueError("decode_context_exceeds_serving_table")
    knots = [start]
    knots.extend(x for x, _ in points if start < x < end)
    knots.append(end)
    total = 0.0
    for left, right in zip(knots, knots[1:]):
        total += (right - left) * (
            _interpolate(points, left) + _interpolate(points, right)
        ) / 2.0
    return total


def _time_points(
    table: tuple[dict[str, Any], ...],
    rate_key: str,
) -> tuple[tuple[float, float], ...]:
    if not table:
        raise ValueError("serving_time_table_missing")
    points: list[tuple[float, float]] = []
    for row in table:
        length = float(row["prompt_length"])
        rate = float(row[rate_key])
        if not math.isfinite(length) or length <= 0.0 or not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("serving_time_table_invalid")
        time_s = length / rate if rate_key == "prefill_tokens_per_s" else 1.0 / rate
        points.append((length, time_s))
    result = tuple(points)
    if tuple(sorted({x for x, _ in result})) != tuple(x for x, _ in result):
        raise ValueError("serving_time_table_lengths_not_strictly_increasing")
    return result


def _interpolate(points: tuple[tuple[float, float], ...], value: float) -> float:
    if value < points[0][0] or value > points[-1][0]:
        raise ValueError("serving_time_lookup_out_of_range")
    for x, y in points:
        if value == x:
            return y
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 < value < x1:
            fraction = (value - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    raise ValueError("serving_time_lookup_out_of_range")


__all__ = ["decode_time_s", "prefill_full_time_s", "prefill_suffix_time_s"]
