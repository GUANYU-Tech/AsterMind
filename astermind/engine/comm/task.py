"""Source-independent contract for synchronous communication estimates."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CommunicationTask:
    operation: str
    participants: tuple[str, ...]
    bytes: int
    bytes_semantics: str
    algorithm: str | None = None
    ring_order: tuple[str, ...] = ()
    reduction: str | None = None
    traffic_matrix_bytes: tuple[tuple[int, ...], ...] = ()
    partition_bytes: tuple[int, ...] = ()
    endpoint_kind: Literal["gpu", "satellite"] = "gpu"
    cell_id: str | None = None
    task_id: str | None = None


class SatelliteTaskAdapter:
    @staticmethod
    def from_transfer(source, target, size_bytes):
        return CommunicationTask("send_recv", (source, target), size_bytes,
                                 "payload", endpoint_kind="satellite")
