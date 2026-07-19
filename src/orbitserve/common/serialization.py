"""Small JSON-safe serialization helpers for framework dataclasses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import math
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, paths, tuples, and nested containers to JSON-safe values."""

    if is_dataclass(value):
        # ``dataclasses.asdict`` recursively deep-copies the complete object
        # graph before ``to_jsonable`` walks it a second time.  Runtime results
        # can contain tens of megabytes of request-level audit records, so that
        # double traversal makes otherwise small summary operations extremely
        # expensive.  Convert fields directly in one pass instead.
        return {
            item.name: to_jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
