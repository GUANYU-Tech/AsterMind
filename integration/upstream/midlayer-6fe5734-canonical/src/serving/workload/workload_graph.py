"""Logical workload DAG containing compute, communication, and transfers."""

from collections import deque

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple


class WorkloadKind(str, Enum):
    LINEAR = "linear"
    LM_HEAD = "lm_head"
    ROUTER = "router"
    FLASH_ATTN = "flash_attn"
    EAGER_ATTN = "eager_attn"
    FUSED_MOE = "fused_moe"
    NORM = "norm"
    ACTIVATION = "activation"
    ROTARY = "rotary"
    ELEMENTWISE = "elementwise"
    REDUCE = "reduce"
    GATHER = "gather"
    VECTOR = "vector"
    KV_UPDATE = "kv_update"
    COLLECTIVE = "collective"
    P2P = "p2p"
    PD_KV_TRANSFER = "pd_kv_transfer"
    AF_ACTIVATION_TRANSFER = "af_activation_transfer"
    MEMCPY = "memcpy"
    KV_OFFLOAD = "kv_offload"


@dataclass(frozen=True)
class WorkloadNode:
    node_id: str
    kind: WorkloadKind
    pool_name: str
    shape: Mapping[str, int]
    dtype: Optional[str] = None
    variant: Optional[str] = None
    rank: Optional[int] = None
    attributes: Mapping[str, object] = None

    def __post_init__(self) -> None:
        for name in ("node_id", "pool_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"workload {name} must be a non-empty string")
        object.__setattr__(self, "kind", WorkloadKind(self.kind))
        if not isinstance(self.shape, Mapping):
            raise TypeError("workload shape must be a mapping")
        invalid_shape_keys = [
            name for name in self.shape if not isinstance(name, str) or not name
        ]
        if invalid_shape_keys:
            raise TypeError(
                "workload shape keys must be non-empty strings: "
                f"{invalid_shape_keys}")
        invalid_shape_types = {
            name: value for name, value in self.shape.items()
            if isinstance(value, bool) or not isinstance(value, int)
        }
        if invalid_shape_types:
            raise TypeError(
                f"workload shape dimensions must be integers: "
                f"{invalid_shape_types}")
        normalized_shape = dict(self.shape)
        if any(value < 0 for value in normalized_shape.values()):
            raise ValueError("workload shape dimensions cannot be negative")
        object.__setattr__(self, "shape", MappingProxyType(normalized_shape))
        for name in ("dtype", "variant"):
            value = getattr(self, name)
            if value is not None and (
                    not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"workload {name} must be a non-empty string or null")
        if self.rank is not None:
            if isinstance(self.rank, bool) or not isinstance(self.rank, int):
                raise TypeError("workload rank must be an integer or null")
            if self.rank < 0:
                raise ValueError("workload rank cannot be negative")
        if self.attributes is not None and not isinstance(
                self.attributes, Mapping):
            raise TypeError("workload attributes must be a mapping or null")
        attributes = {} if self.attributes is None else dict(self.attributes)
        object.__setattr__(self, "attributes", MappingProxyType(attributes))


@dataclass(frozen=True)
class WorkloadEdge:
    source: str
    destination: str
    kind: str = "data"

    def __post_init__(self) -> None:
        for name in ("source", "destination", "kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"workload edge {name} must be a non-empty string")
        if self.kind not in ("data", "barrier", "ready", "completion"):
            raise ValueError(f"unknown workload edge kind: {self.kind!r}")


@dataclass(frozen=True)
class WorkloadBlockRepeat:
    """One materialized block template plus exact recurrence edges."""

    block_id: str
    node_ids: Tuple[str, ...]
    repeat_count: int
    feedback_edges: Tuple[WorkloadEdge, ...]
    trace_node_ids: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("workload block repeat ID must be non-empty")
        if not isinstance(self.node_ids, (list, tuple)):
            raise TypeError("workload block repeat node IDs must be a sequence")
        node_ids = tuple(self.node_ids)
        if (not node_ids
                or any(not isinstance(node_id, str) or not node_id
                       for node_id in node_ids)):
            raise ValueError(
                "workload block repeat node IDs must be non-empty strings")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("workload block repeat node IDs must be unique")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)
                or self.repeat_count < 2):
            raise ValueError(
                "workload block repeat count must be an integer >= 2")
        if not isinstance(self.feedback_edges, (list, tuple)):
            raise TypeError(
                "workload block repeat feedback edges must be a sequence")
        feedback_edges = tuple(self.feedback_edges)
        if any(not isinstance(edge, WorkloadEdge) for edge in feedback_edges):
            raise TypeError(
                "workload block repeat feedback requires WorkloadEdge objects")
        known = set(node_ids)
        if any(
                edge.source not in known or edge.destination not in known
                for edge in feedback_edges):
            raise ValueError(
                "workload block feedback edges must stay inside the template")
        if not isinstance(self.trace_node_ids, (list, tuple)):
            raise TypeError(
                "workload block repeat trace node IDs must be a sequence")
        trace_node_ids = tuple(
            (template_id, tuple(logical_ids))
            for template_id, logical_ids in self.trace_node_ids
        )
        template_ids = []
        logical_ids = []
        for template_id, occurrences in trace_node_ids:
            if template_id not in known:
                raise ValueError(
                    "workload block trace node is outside the template")
            if (len(occurrences) != self.repeat_count
                    or occurrences[0] != template_id
                    or any(not isinstance(node_id, str) or not node_id
                           for node_id in occurrences)):
                raise ValueError(
                    "workload block trace node occurrences must cover every "
                    "logical repeat")
            template_ids.append(template_id)
            logical_ids.extend(occurrences)
        if len(template_ids) != len(set(template_ids)):
            raise ValueError(
                "workload block trace template node IDs must be unique")
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError(
                "workload block logical trace node IDs must be unique")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "feedback_edges", feedback_edges)
        object.__setattr__(self, "trace_node_ids", trace_node_ids)


@dataclass(frozen=True)
class WorkloadGraph:
    nodes: Tuple[WorkloadNode, ...]
    edges: Tuple[WorkloadEdge, ...]
    block_repeats: Tuple[WorkloadBlockRepeat, ...]
    _nodes_by_id: Mapping[str, WorkloadNode] = field(
        init=False, repr=False, compare=False)
    _predecessors_by_id: Mapping[str, Tuple[str, ...]] = field(
        init=False, repr=False, compare=False)
    _topological_node_ids: Tuple[str, ...] = field(
        init=False, repr=False, compare=False)

    def __init__(self, nodes: Iterable[WorkloadNode],
                 edges: Iterable[WorkloadEdge] = (),
                 block_repeats: Iterable[WorkloadBlockRepeat] = ()):
        node_tuple = tuple(nodes)
        edge_tuple = tuple(edges)
        block_repeat_tuple = tuple(block_repeats)
        if not node_tuple:
            raise ValueError("WorkloadGraph requires at least one node")
        ids = [node.node_id for node in node_tuple]
        if len(ids) != len(set(ids)):
            raise ValueError("workload node IDs must be unique")
        known = set(ids)
        for edge in edge_tuple:
            if edge.source not in known or edge.destination not in known:
                raise ValueError(f"workload edge references an unknown node: {edge}")
            if edge.source == edge.destination:
                raise ValueError("workload self-edges are not allowed")
        if any(
                not isinstance(repeat, WorkloadBlockRepeat)
                for repeat in block_repeat_tuple):
            raise TypeError(
                "WorkloadGraph block repeats require WorkloadBlockRepeat objects")
        block_ids = [repeat.block_id for repeat in block_repeat_tuple]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("workload block repeat IDs must be unique")
        repeated_node_ids = []
        for repeat in block_repeat_tuple:
            unknown = set(repeat.node_ids) - known
            if unknown:
                raise ValueError(
                    "workload block repeat references unknown nodes: "
                    f"{sorted(unknown)}")
            repeated_node_ids.extend(repeat.node_ids)
        if len(repeated_node_ids) != len(set(repeated_node_ids)):
            raise ValueError(
                "workload nodes cannot belong to multiple block repeats")
        predecessors = {node_id: [] for node_id in ids}
        for edge in edge_tuple:
            predecessors[edge.destination].append(edge.source)
        object.__setattr__(self, "nodes", node_tuple)
        object.__setattr__(self, "edges", edge_tuple)
        object.__setattr__(self, "block_repeats", block_repeat_tuple)
        object.__setattr__(
            self, "_nodes_by_id", MappingProxyType({
                node.node_id: node for node in node_tuple
            }))
        object.__setattr__(
            self, "_predecessors_by_id", MappingProxyType({
                node_id: tuple(source_ids)
                for node_id, source_ids in predecessors.items()
            }))
        object.__setattr__(
            self, "_topological_node_ids", self._compute_topological_order())

    @property
    def logical_node_count(self) -> int:
        return len(self.nodes) + sum(
            (repeat.repeat_count - 1) * len(repeat.node_ids)
            for repeat in self.block_repeats
        )

    @property
    def logical_edge_count(self) -> int:
        repeated_edges = 0
        for repeat in self.block_repeats:
            node_ids = set(repeat.node_ids)
            internal_edge_count = sum(
                1 for edge in self.edges
                if edge.source in node_ids and edge.destination in node_ids
            )
            repeated_edges += (repeat.repeat_count - 1) * (
                internal_edge_count + len(repeat.feedback_edges))
        return len(self.edges) + repeated_edges

    def node(self, node_id: str) -> WorkloadNode:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as exc:
            raise KeyError(node_id) from exc

    def predecessors(self, node_id: str) -> Tuple[str, ...]:
        if node_id in self._predecessors_by_id:
            return self._predecessors_by_id[node_id]
        return ()

    def topological_order(self) -> Tuple[str, ...]:
        return self._topological_node_ids

    def _compute_topological_order(self) -> Tuple[str, ...]:
        indegree = {node.node_id: 0 for node in self.nodes}
        successors = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            indegree[edge.destination] += 1
            successors[edge.source].append(edge.destination)
        ready = deque(
            node.node_id for node in self.nodes if indegree[node.node_id] == 0)
        order = []
        while ready:
            current = ready.popleft()
            order.append(current)
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if len(order) != len(self.nodes):
            raise ValueError("WorkloadGraph must be acyclic")
        return tuple(order)
