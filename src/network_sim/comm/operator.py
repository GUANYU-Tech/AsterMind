"""Estimate one communication operator on a fixed physical topology.

Transfers are isolated. Each round takes the longest isolated path time and
successive rounds are serial. No global clock or in-flight traffic is needed.
"""

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from math import ceil, inf


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


@dataclass(frozen=True)
class OperatorEstimate:
    duration_ns: int
    rounds: int
    flow_count: int
    cross_satellite_flows: int
    cross_satellite_bytes: int


class OperatorCommunication:
    """Use configured routes and capacity for synchronous, isolated estimates."""

    def __init__(self, topology):
        self.topology = topology
        self._outgoing = defaultdict(list)
        self._routes = {}
        for edge in topology.native_edges:
            self._outgoing[edge.source].append(edge)

    def _path_limits(self, source, target):
        key = source, target
        if key not in self._routes:
            start, end = self.topology.endpoints(source, target)
            queue = [(0, (start,), inf)]
            best = {start: (0, (start,))}
            while queue:
                delay, path, bandwidth = heappop(queue)
                node = path[-1]
                if best[node] != (delay, path):
                    continue
                if node == end:
                    self._routes[key] = bandwidth, delay
                    break
                for edge in self._outgoing[node]:
                    if edge.target in path:
                        continue
                    candidate = delay + edge.latency_ns, path + (edge.target,)
                    if edge.target not in best or candidate < best[edge.target]:
                        best[edge.target] = candidate
                        heappush(queue, (*candidate, min(bandwidth, edge.bandwidth_gbps)))
            else:
                raise ValueError(f"no configured route from {source} to {target}")
        return self._routes[key]

    def estimate(self, operator: CommunicationOperator) -> OperatorEstimate:
        participants = operator.participants
        if len(participants) < 2 or len(set(participants)) != len(participants):
            raise ValueError("communication requires distinct participants")
        if any(p not in self.topology.devices or self.topology.devices[p].kind != "gpu"
               for p in participants):
            raise ValueError("communication operator participants must be physical GPUs")
        if type(operator.bytes) is not int or operator.bytes < 0:
            raise ValueError("operator bytes must be a nonnegative integer")
        duration = count = cross_count = cross_bytes = 0
        rounds = self._rounds(operator)
        for flows in rounds:
            round_duration = 0
            for source, target, size in flows:
                bandwidth, delay = self._path_limits(source, target)
                round_duration = max(round_duration, ceil(size * 8 / bandwidth) + delay)
                count += 1
                if self.topology.devices[source].satellite != self.topology.devices[target].satellite:
                    cross_count += 1
                    cross_bytes += size
            duration += round_duration
        return OperatorEstimate(duration, len(rounds), count, cross_count, cross_bytes)

    @staticmethod
    def _rounds(op):
        members, size, operation = op.participants, op.bytes, op.operation
        n = len(members)
        if operation == "send_recv":
            if n != 2 or op.bytes_semantics != "payload":
                raise ValueError("send_recv requires an ordered source and destination and a payload")
            return (((members[0], members[1], size),),) if size else ()
        if operation == "broadcast":
            if op.algorithm != "direct" or op.bytes_semantics != "payload":
                raise ValueError("broadcast requires direct fanout and a payload")
            return (tuple((members[0], target, size) for target in members[1:]),) if size else ()
        if operation in ("all_reduce", "all_gather"):
            if op.algorithm != "ring" or op.bytes_semantics != "per_rank_input":
                raise ValueError("equal-input collectives require a ring and per-rank bytes")
            order = op.ring_order
            if len(order) != n or set(order) != set(members):
                raise ValueError("ring order must contain every participant exactly once")
            chunk, count = size, n - 1
            if operation == "all_reduce":
                if op.reduction != "sum" or size % n:
                    raise ValueError("ring AllReduce requires sum and equally divisible input")
                chunk, count = size // n, 2 * count
            return (tuple((rank, order[(i + 1) % n], chunk) for i, rank in enumerate(order)),) * count if chunk else ()
        if operation in ("all_gatherv", "reduce_scatterv"):
            sizes = op.partition_bytes
            semantics = "aggregate_input" if operation == "all_gatherv" else "per_rank_input"
            if (op.algorithm not in ("direct", "pairwise") or op.bytes_semantics != semantics or len(sizes) != n
                    or any(type(b) is not int or b < 0 for b in sizes) or sum(sizes) != size):
                raise ValueError("variable collective requires one nonnegative payload per participant")
            flows = tuple((owner, peer, b) if operation == "all_gatherv" else (peer, owner, b)
                          for owner, b in zip(members, sizes, strict=True) if b
                          for peer in members if peer != owner)
            return (flows,) if flows else ()
        if operation == "all_to_all" and op.algorithm == "pairwise":
            matrix = op.traffic_matrix_bytes
            if matrix:
                if (op.bytes_semantics != "aggregate_input" or len(matrix) != n
                        or any(len(row) != n or row[i] != 0 or any(type(b) is not int or b < 0 for b in row)
                               for i, row in enumerate(matrix))
                        or sum(map(sum, matrix)) != size):
                    raise ValueError("AllToAll matrix must conserve aggregate nonlocal bytes")
            else:
                if op.bytes_semantics != "per_rank_input" or size % n:
                    raise ValueError("equal AllToAll requires equally divisible per-rank input")
                matrix = tuple(tuple(size // n if i != j else 0 for j in range(n)) for i in range(n))
            rounds = []
            for offset in range(1, n):
                flows = []
                for i, source in enumerate(members):
                    j = (i + offset) % n
                    if matrix[i][j]:
                        flows.append((source, members[j], matrix[i][j]))
                if flows:
                    rounds.append(tuple(flows))
            return tuple(rounds)
        raise NotImplementedError(f"unsupported communication operator: {operation}/{op.algorithm}")
