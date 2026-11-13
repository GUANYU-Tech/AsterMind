from dataclasses import dataclass

@dataclass(frozen=True)
class CommunicationOperator:
    operation: str
    algorithm: str
    participants: tuple[str, ...]
    bytes: int
    bytes_semantics: str
    ring_order: tuple[str, ...] = ()
    reduction: str | None = None
    partition_bytes: tuple[int, ...] = ()
    traffic_matrix_bytes: tuple[tuple[int, ...], ...] = ()
    cell_id: str | None = None
    task_id: str | None = None


@dataclass(frozen=True)
class OperatorEstimate:
    duration_ns: int
    rounds: int
    flow_count: int
    cross_satellite_flows: int
    cross_satellite_bytes: int



