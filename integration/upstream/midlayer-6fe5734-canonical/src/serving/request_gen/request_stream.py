"""Common output contract for synthetic and production request sources."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Iterator, Tuple


@dataclass(frozen=True)
class RequestSpec:
    arrival_time_s: float
    request_id: str
    input_tokens: int
    output_tokens: int
    prefix_hit_tokens: int = 0

    def __post_init__(self) -> None:
        if (isinstance(self.arrival_time_s, bool)
                or not isinstance(self.arrival_time_s, (int, float))
                or not math.isfinite(self.arrival_time_s)
                or self.arrival_time_s < 0):
            raise ValueError(
                "request arrival time must be finite and non-negative")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        for name in ("input_tokens", "output_tokens", "prefix_hit_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"request {name} must be an integer")
        if self.input_tokens < 1:
            raise ValueError("request input_tokens must be positive")
        if self.output_tokens < 0:
            raise ValueError("request output_tokens cannot be negative")
        if not 0 <= self.prefix_hit_tokens <= self.input_tokens:
            raise ValueError(
                "request prefix_hit_tokens must be between zero and "
                "input_tokens")


@dataclass(frozen=True)
class RequestStream:
    requests: Tuple[RequestSpec, ...]

    def __init__(self, requests: Iterable[RequestSpec]):
        supplied = tuple(requests)
        if not supplied:
            raise ValueError("RequestStream requires at least one request")
        if any(not isinstance(request, RequestSpec) for request in supplied):
            raise TypeError("RequestStream requires RequestSpec objects")
        # ``sorted`` is stable, so equal-arrival requests retain source order.
        # That order is observable because the router reserves load after every
        # decision; using request_id as a secondary key would rewrite a
        # production trace's same-time ordering.
        ordered = tuple(sorted(
            supplied,
            key=lambda request: request.arrival_time_s))
        ids = [request.request_id for request in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("request IDs must be unique")
        object.__setattr__(self, "requests", ordered)

    def __iter__(self) -> Iterator[RequestSpec]:
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)


def request_stream_sha256(requests: RequestStream) -> str:
    """Hash the canonical ordered request-stream contract."""

    if not isinstance(requests, RequestStream):
        raise TypeError("request stream hash requires RequestStream")
    payload = [
        {
            "arrival_time_s": request.arrival_time_s,
            "request_id": request.request_id,
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "prefix_hit_tokens": request.prefix_hit_tokens,
        }
        for request in requests
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
