"""Cell event contracts backed by the platform-wide GlobalTimeline."""

import math
from dataclasses import dataclass

from astermind.engine.timeline import GlobalTimeline

ARRIVAL = 0
STEP_COMPLETE = 1
NOTIFY = 2
DISPATCH = 3


@dataclass(frozen=True)
class CellRequest:
    request_id: str
    cell_ready_time_s: float
    input_tokens: int
    output_tokens: int
    prefix_hit_tokens: int = 0
    session_id: str | None = None
    context_id: str | None = None
    parent_context_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be nonempty")
        if (
            isinstance(self.cell_ready_time_s, bool)
            or not isinstance(self.cell_ready_time_s, (int, float))
            or not math.isfinite(self.cell_ready_time_s)
            or self.cell_ready_time_s < 0
        ):
            raise ValueError("invalid cell_ready_time_s")
        for name in ("input_tokens", "output_tokens", "prefix_hit_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid {name}")
        if self.input_tokens < 1 or self.prefix_hit_tokens > self.input_tokens:
            raise ValueError("invalid prompt/prefix length")
        for name in ("session_id", "context_id", "parent_context_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"invalid {name}")
        if self.session_id is None and (
            self.context_id is not None or self.parent_context_id is not None
        ):
            raise ValueError("context identity requires a session")


@dataclass(frozen=True)
class CellTokenOutput:
    cell_id: str
    request_id: str
    token_start: int
    token_count: int
    produced_time_s: float
    is_final: bool

    def __post_init__(self):
        if not self.cell_id or not self.request_id:
            raise ValueError("output requires Cell and request IDs")
        if type(self.token_start) is not int or self.token_start < 0:
            raise ValueError("invalid output token start")
        if type(self.token_count) is not int or self.token_count < 1:
            raise ValueError("output requires positive token count")
        if not math.isfinite(self.produced_time_s) or self.produced_time_s < 0:
            raise ValueError("invalid output time")
        if type(self.is_final) is not bool:
            raise TypeError("invalid final-output flag")


@dataclass(frozen=True)
class CellCompletion:
    cell_id: str
    request_id: str
    cell_ready_time_s: float
    first_execution_s: float
    completion_time_s: float
    first_token_time_s: float | None
    last_token_time_s: float | None
    generated_tokens: int
    prefill_active_s: float
    decode_active_s: float


@dataclass(frozen=True)
class CellWaitInterval:
    """Reported request residence that does not overlap its active batches."""

    kind: str
    start_s: float
    end_s: float

    def __post_init__(self):
        if self.kind != "pd_kv_transfer":
            raise ValueError("unsupported Cell wait interval kind")
        for value in (self.start_s, self.end_s):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("finite Cell wait timestamps required")
        if self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError("invalid Cell wait interval")


class CellEventLoop(GlobalTimeline):
    """Compatibility name for code that already receives the global timeline.

    This class intentionally adds no state and owns no local queue.  New code
    should depend on :class:`orbitserve.runtime.timeline.GlobalTimeline`.
    """
