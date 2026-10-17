"""Cross-framework facts emitted by cache managers."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class KVCacheAction(str, Enum):
    HIT = "hit"
    MISS = "miss"
    ALLOCATE = "allocate"
    RELEASE = "release"
    LOAD = "load"
    EVICT = "evict"


@dataclass(frozen=True)
class KVCacheEvent:
    request_id: str
    action: KVCacheAction
    tier: str
    token_start: int
    token_end: int
    bytes: int
    source_tier: Optional[str] = None
    destination_tier: Optional[str] = None
    ready_event: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("request_id", "tier"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"KV cache event {name} must be a non-empty string")
        object.__setattr__(self, "action", KVCacheAction(self.action))
        for name in ("token_start", "token_end", "bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"KV cache event {name} must be an integer")
        if self.token_start < 0 or self.token_end < self.token_start:
            raise ValueError("invalid KV cache token range")
        if self.bytes < 0:
            raise ValueError("KV cache event bytes cannot be negative")
        for name in ("source_tier", "destination_tier", "ready_event"):
            value = getattr(self, name)
            if value is not None and (
                    not isinstance(value, str) or not value):
                raise ValueError(
                    f"KV cache event {name} must be non-empty or null")
        if self.action in (KVCacheAction.LOAD, KVCacheAction.EVICT):
            if not self.source_tier or not self.destination_tier:
                raise ValueError(f"{self.action.value} requires source and destination tiers")
            if self.source_tier.lower() == self.destination_tier.lower():
                raise ValueError(
                    f"{self.action.value} requires distinct tier endpoints")
            if self.token_end == self.token_start or self.bytes == 0:
                raise ValueError(
                    f"{self.action.value} requires a positive transfer")
            if self.ready_event is None:
                raise ValueError(
                    f"{self.action.value} requires a ready_event")
        elif self.source_tier is not None or self.destination_tier is not None:
            raise ValueError(
                f"{self.action.value} cannot carry source/destination tiers")
        elif self.ready_event is not None:
            raise ValueError(
                f"{self.action.value} cannot carry a ready_event")
        if self.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE):
            if self.token_end == self.token_start or self.bytes == 0:
                raise ValueError(
                    f"{self.action.value} requires positive cache capacity")
