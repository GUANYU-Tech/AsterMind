from __future__ import annotations

from typing import Annotated, Literal
from pydantic import Field
from astermind.common.contracts import Contract, Identifier, Nanoseconds


class TransferRequest(Contract):
    id: Identifier
    source: Identifier
    target: Identifier
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
    ready_time_ns: Nanoseconds = 0
    origin: Literal["multi_gpu", "orbitserve", "midlayer", "test"] = "test"
    operation: str = "transfer"
    correlation_id: str | None = None


class FlowStatus(Contract):
    id: str
    state: Literal[
        "scheduled",
        "waiting_link",
        "transmitting",
        "propagating",
        "completed",
        "failed",
    ]
    remaining_bytes: float
    rate_gbps: float
    transmission_end_ns: int | None = None
    completion_ns: int | None = None


class CompletionEvent(Contract):
    kind: Literal["completed"] = "completed"
    id: str
    origin: str
    operation: str
    correlation_id: str | None
    source: str
    target: str
    size_bytes: int
    completion_time_ns: int
    path: tuple[str, ...]


class FailureEvent(Contract):
    kind: Literal["failed"] = "failed"
    id: str
    origin: str
    operation: str
    correlation_id: str | None
    source: str
    target: str
    size_bytes: int
    failure_time_ns: int
    reason: str
    remaining_bytes: float
    path: tuple[str, ...]
