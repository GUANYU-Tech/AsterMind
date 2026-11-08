"""Resource-constrained critical-path schedule for a KernelPlan."""

from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass, replace
from heapq import heappop, heappush
import hashlib
import math
from numbers import Real
import struct
from typing import Mapping, NamedTuple, Tuple

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import KernelKind, KernelPrediction

from .kernel_plan import KernelBlockRepeat, KernelPlan
from src.serving.host_timing import (
    HostLaunchProfile,
    HostRuntimePrepareProfile,
    RuntimePrepareEvent,
    RuntimeGraphSegmentDescriptor,
)
from src.serving.workload.workload_graph import WorkloadGraph, WorkloadKind


@dataclass(frozen=True)
class ExecutionInterval:
    call_id: str
    pool_name: str
    rank: int
    resource: str
    start_time_s: float
    end_time_s: float
    reserved_resource_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.reserved_resource_ids, (list, tuple)):
            raise TypeError(
                "execution interval reserved resources must be a list or tuple")
        resources = tuple(self.reserved_resource_ids)
        if (not resources
                or any(not isinstance(resource_id, str) or not resource_id
                       for resource_id in resources)):
            raise ValueError(
                "execution interval requires non-empty reserved resource IDs")
        if len(resources) != len(set(resources)):
            raise ValueError(
                "execution interval reserved resource IDs must be unique")
        object.__setattr__(self, "reserved_resource_ids", resources)


@dataclass(frozen=True)
class HostExecutionInterval:
    """One predicted Host event on a concrete worker lane."""

    event_id: str
    event_kind: str
    pool_name: str
    rank: int
    call_id: str
    call_occurrence_index: int
    covered_call_occurrences: Tuple[Tuple[str, int], ...]
    device_predecessor_occurrences: Tuple[Tuple[str, int], ...]
    host_predecessor_event_id: str | None
    runtime_graph_mode: str
    graph_segment_id: str | None
    start_time_s: float
    end_time_s: float
    exposed_time_s: float
    overlapped_time_s: float
    mechanism: str | None = None
    event_count: int = 1

    def __post_init__(self) -> None:
        if self.event_kind not in (
                "eager_submit", "graph_replay", "runtime_prepare"):
            raise ValueError(f"unknown Host event kind: {self.event_kind!r}")
        for name in ("event_id", "pool_name", "call_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Host event {name} must be non-empty")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("Host event rank must be a non-negative integer")
        if (isinstance(self.call_occurrence_index, bool)
                or not isinstance(self.call_occurrence_index, int)
                or self.call_occurrence_index < -1):
            raise ValueError(
                "Host event call occurrence index must be an integer >= -1")
        covered = tuple(
            tuple(occurrence) for occurrence in self.covered_call_occurrences)
        if not covered:
            raise ValueError("Host event must cover at least one call occurrence")
        for call_id, occurrence_index in covered:
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Host event covered call ID must be non-empty")
            if (isinstance(occurrence_index, bool)
                    or not isinstance(occurrence_index, int)
                    or occurrence_index < -1):
                raise ValueError("Host event covered occurrence index must be >= -1")
        if len(covered) != len(set(covered)):
            raise ValueError("Host event covered occurrences must be unique")
        if (self.call_id, self.call_occurrence_index) not in covered:
            raise ValueError("Host event trigger call must be covered")
        if self.event_kind in ("eager_submit", "runtime_prepare") and covered != ((
                self.call_id, self.call_occurrence_index),):
            raise ValueError(
                "single-call Host event must cover exactly its trigger call")
        object.__setattr__(self, "covered_call_occurrences", covered)
        predecessors = tuple(
            tuple(predecessor)
            for predecessor in self.device_predecessor_occurrences
        )
        for predecessor in predecessors:
            if len(predecessor) != 2:
                raise ValueError(
                    "Host event Device predecessor must be a call/index pair")
            call_id, occurrence_index = predecessor
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    "Host event Device predecessor call ID must be non-empty")
            if (isinstance(occurrence_index, bool)
                    or not isinstance(occurrence_index, int)
                    or occurrence_index < -1):
                raise ValueError(
                    "Host event Device predecessor index must be >= -1")
        if len(predecessors) != len(set(predecessors)):
            raise ValueError("Host event Device predecessors must be unique")
        object.__setattr__(
            self, "device_predecessor_occurrences", predecessors)
        if (self.host_predecessor_event_id is not None
                and (not isinstance(self.host_predecessor_event_id, str)
                     or not self.host_predecessor_event_id)):
            raise ValueError(
                "Host predecessor event ID must be non-empty or null")
        if self.runtime_graph_mode not in ("NONE", "FULL", "PIECEWISE"):
            raise ValueError("Host event runtime Graph mode is invalid")
        if (self.event_kind == "eager_submit"
                and self.runtime_graph_mode == "FULL"):
            raise ValueError(
                "eager Host event is invalid in FULL runtime Graph mode")
        if (self.event_kind == "graph_replay"
                and self.runtime_graph_mode == "NONE"):
            raise ValueError(
                "Graph replay Host event requires a Graph runtime mode")
        if self.event_kind == "runtime_prepare":
            if not isinstance(self.mechanism, str) or not self.mechanism:
                raise ValueError(
                    "runtime prepare Host event requires a mechanism")
        elif self.mechanism is not None:
            raise ValueError(
                "only runtime prepare Host events may carry a mechanism")
        if (isinstance(self.event_count, bool)
                or not isinstance(self.event_count, int)
                or self.event_count < 1):
            raise ValueError("Host event count must be positive")
        if self.event_kind != "runtime_prepare" and self.event_count != 1:
            raise ValueError(
                "only runtime prepare Host events may have multiplicity")
        if (self.event_kind != "graph_replay") != (
                self.graph_segment_id is None):
            raise ValueError("Host event Graph segment identity is inconsistent")
        if (self.graph_segment_id is not None
                and (not isinstance(self.graph_segment_id, str)
                     or not self.graph_segment_id)):
            raise ValueError("Host event Graph segment ID must be non-empty or null")
        for name in (
                "start_time_s", "end_time_s", "exposed_time_s",
                "overlapped_time_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Host event {name} must be finite and non-negative")
        if self.end_time_s < self.start_time_s:
            raise ValueError("Host event ends before it starts")
        if not math.isclose(
                self.exposed_time_s + self.overlapped_time_s,
                self.end_time_s - self.start_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15):
            raise ValueError("Host event exposed/overlapped split does not conserve time")


@dataclass(frozen=True)
class ExecutionTimeline:
    intervals: Tuple[ExecutionInterval, ...]
    duration_s: float
    block_repeats: Tuple["ExecutionBlockRepeat", ...] = ()
    pool_activity: Tuple["ExecutionActivity", ...] = ()
    reserved_resource_activity: Tuple["ExecutionActivity", ...] = ()
    call_occurrence_counts: Tuple[Tuple[str, int], ...] = ()
    critical_path_contributions: Tuple[Tuple[str, float], ...] = ()
    critical_call_count: int = 0
    critical_path_unique: bool = True
    predicted_device_time_s: float = 0.0
    launch_cpu_time_s: float = 0.0
    launch_critical_path_s: float = 0.0
    host_device_overlap_s: float = 0.0
    eager_dispatch_count: int = 0
    graph_replay_count: int = 0
    launch_kinds: Tuple[str, ...] = ()
    host_intervals: Tuple[HostExecutionInterval, ...] = ()
    runtime_graph_mode: str | None = None

    @property
    def host_exposed_critical_path_s(self) -> float:
        """Net step extension caused by Host dispatch on the critical path."""

        return max(
            0.0,
            self.launch_critical_path_s - self.host_device_overlap_s,
        )


@dataclass(frozen=True)
class ExecutionBlockRepeat:
    block_id: str
    repeat_count: int
    first_iteration_start_time_s: float
    last_iteration_end_time_s: float
    call_intervals: Tuple[
        Tuple[str, Tuple[Tuple[float, float], ...]], ...
    ] = ()
    call_interval_sha256: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("execution block repeat ID must be non-empty")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)
                or self.repeat_count < 2):
            raise ValueError(
                "execution block repeat count must be an integer >= 2")
        for name, value in (
                ("first iteration start", self.first_iteration_start_time_s),
                ("last iteration end", self.last_iteration_end_time_s)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(
                    f"execution block {name} must be finite and non-negative")
        if self.last_iteration_end_time_s < self.first_iteration_start_time_s:
            raise ValueError(
                "execution block repeat cannot end before it starts")
        if not isinstance(self.call_intervals, (list, tuple)):
            raise TypeError(
                "execution block repeat call intervals must be a sequence")
        normalized_call_intervals = []
        call_ids = []
        for call_id, raw_intervals in self.call_intervals:
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    "execution block repeat call ID must be non-empty")
            if not isinstance(raw_intervals, (list, tuple)):
                raise TypeError(
                    "execution block repeat intervals must be a sequence")
            call_intervals = tuple(tuple(interval) for interval in raw_intervals)
            if len(call_intervals) != self.repeat_count:
                raise ValueError(
                    "execution block repeat call interval count disagrees "
                    "with repeat count")
            for start_time_s, end_time_s in call_intervals:
                if (isinstance(start_time_s, bool)
                        or not isinstance(start_time_s, Real)
                        or not math.isfinite(start_time_s)
                        or start_time_s < 0
                        or isinstance(end_time_s, bool)
                        or not isinstance(end_time_s, Real)
                        or not math.isfinite(end_time_s)
                        or end_time_s < start_time_s):
                    raise ValueError(
                        "execution block repeat call intervals must be "
                        "finite, non-negative, and ordered")
            call_ids.append(call_id)
            normalized_call_intervals.append((call_id, call_intervals))
        if len(call_ids) != len(set(call_ids)):
            raise ValueError(
                "execution block repeat call interval IDs must be unique")
        if not isinstance(self.call_interval_sha256, (list, tuple)):
            raise TypeError(
                "execution block repeat interval hashes must be a sequence")
        normalized_hashes = tuple(
            tuple(item) for item in self.call_interval_sha256)
        hash_call_ids = []
        for call_id, digest in normalized_hashes:
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    "execution block repeat hashed call ID must be non-empty")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    "execution block repeat interval digest must be SHA256")
            hash_call_ids.append(call_id)
        if len(hash_call_ids) != len(set(hash_call_ids)):
            raise ValueError(
                "execution block repeat hashed call IDs must be unique")
        if bool(call_ids) == bool(hash_call_ids):
            raise ValueError(
                "execution block repeat requires exactly one interval "
                "representation")
        object.__setattr__(
            self, "call_intervals", tuple(normalized_call_intervals))
        object.__setattr__(
            self, "call_interval_sha256", normalized_hashes)


@dataclass(frozen=True)
class ExecutionActivity:
    activity_id: str
    interval_count: int
    intervals: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.activity_id, str) or not self.activity_id:
            raise ValueError("execution activity ID must be non-empty")
        if (isinstance(self.interval_count, bool)
                or not isinstance(self.interval_count, int)
                or self.interval_count < 0):
            raise ValueError(
                "execution activity interval count must be non-negative")
        if not isinstance(self.intervals, (list, tuple)):
            raise TypeError("execution activity intervals must be a sequence")
        intervals = tuple(tuple(interval) for interval in self.intervals)
        previous_end = None
        for interval in intervals:
            if len(interval) != 2:
                raise ValueError(
                    "execution activity interval requires start and end")
            start_time_s, end_time_s = interval
            if any(
                    isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0
                    for value in interval):
                raise ValueError(
                    "execution activity times must be finite and non-negative")
            if end_time_s < start_time_s:
                raise ValueError(
                    "execution activity interval cannot end before it starts")
            if previous_end is not None and start_time_s < previous_end:
                raise ValueError(
                    "execution activity intervals must be sorted and disjoint")
            previous_end = end_time_s
        object.__setattr__(self, "intervals", intervals)


@dataclass(frozen=True)
class ResourceReservation:
    resource_id: str
    start_time_s: float
    end_time_s: float


class ExecutionResourceCalendar:
    """Reserve non-overlapping intervals on deployment-wide transfer resources."""

    def __init__(self) -> None:
        self._reservations_by_resource: dict[
            str, list[ResourceReservation]
        ] = {}
        self._reservation_starts_by_resource: dict[str, list[float]] = {}
        self._reservation_history: list[ResourceReservation] = []

    @property
    def reservations(self) -> Tuple[ResourceReservation, ...]:
        return tuple(sorted(
            self._reservation_history,
            key=lambda reservation: (
                reservation.start_time_s,
                reservation.resource_id,
                reservation.end_time_s,
            ),
        ))

    def reserve(
        self,
        resource_id: str,
        earliest_start_time_s: float,
        duration_s: float,
    ) -> Tuple[float, float]:
        return self.reserve_many(
            (resource_id,), earliest_start_time_s, duration_s)

    def reserve_many(
        self,
        resource_ids: Tuple[str, ...],
        earliest_start_time_s: float,
        duration_s: float,
    ) -> Tuple[float, float]:
        if not isinstance(resource_ids, (list, tuple)):
            raise TypeError("resource calendar IDs must be a list or tuple")
        resources = tuple(resource_ids)
        if (not resources
                or any(not isinstance(resource_id, str) or not resource_id
                       for resource_id in resources)):
            raise ValueError("resource calendar IDs must be non-empty strings")
        if len(resources) != len(set(resources)):
            raise ValueError("resource calendar IDs must be unique")
        for name, value in (
                ("earliest start", earliest_start_time_s),
                ("duration", duration_s)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(
                    f"resource calendar {name} must be finite and non-negative")
        start_time_s = float(earliest_start_time_s)
        duration_s = float(duration_s)

        while True:
            next_start_time_s = max(
                self._first_available_start(
                    resource_id, start_time_s, duration_s)
                for resource_id in resources
            )
            if next_start_time_s == start_time_s:
                break
            start_time_s = next_start_time_s

        end_time_s = start_time_s + duration_s
        for resource_id in resources:
            reservation = ResourceReservation(
                resource_id, start_time_s, end_time_s)
            self._insert_reservation(reservation)
            self._reservation_history.append(reservation)
        return start_time_s, end_time_s

    def _first_available_start(
        self,
        resource_id: str,
        start_time_s: float,
        duration_s: float,
    ) -> float:
        if resource_id not in self._reservations_by_resource:
            return start_time_s
        reservations = self._reservations_by_resource[resource_id]
        starts = self._reservation_starts_by_resource[resource_id]
        index = bisect_left(starts, start_time_s)
        if index and start_time_s < reservations[index - 1].end_time_s:
            start_time_s = reservations[index - 1].end_time_s
        for reservation in reservations[index:]:
            if start_time_s + duration_s <= reservation.start_time_s:
                break
            if start_time_s < reservation.end_time_s:
                start_time_s = reservation.end_time_s
        return start_time_s

    def _insert_reservation(self, reservation: ResourceReservation) -> None:
        resource_id = reservation.resource_id
        if resource_id in self._reservations_by_resource:
            reservations = self._reservations_by_resource[resource_id]
            starts = self._reservation_starts_by_resource[resource_id]
        else:
            reservations = []
            starts = []
            self._reservations_by_resource[resource_id] = reservations
            self._reservation_starts_by_resource[resource_id] = starts
        insertion_index = bisect_left(starts, reservation.start_time_s)
        reservations.insert(insertion_index, reservation)
        starts.insert(insertion_index, reservation.start_time_s)


def call_reserved_resource_ids(call) -> Tuple[str, ...]:
    """Return every local resource occupied for the full call interval."""

    transfer = call.kernel_spec.transfer
    if (HWSpec.source_contract_supported("programmed_mxfp4_fp8_expert_service", call.kernel_spec.hardware_id)
            and call.kernel_spec.kind in (KernelKind.GEMM, KernelKind.VECTOR)):
        return (f"pool:{call.pool_name}:rank:{call.rank}:compute",
            f"pool:{call.pool_name}:rank:{call.rank}:hbm")
    if call.kernel_spec.kind is KernelKind.MEMCPY and transfer.algorithm == "fpga_staging":
        resources = (f"pool:{call.pool_name}:rank:{call.rank}:ddr",)
        if transfer.direction == "hbm_to_ddr":
            resources += (f"pool:{call.pool_name}:rank:{call.rank}:hbm",)
        return resources
    if (call.kernel_spec.kind is KernelKind.COMMUNICATION
            and transfer is not None):
        if transfer.link == "roce":
            # The declared nonblocking switch is not one serialized global
            # fabric lane. Physical NIC IDs share capacity across ranks and
            # pools; TX/RX lanes are independent on a full-duplex port.
            nic_by_rank = dict(zip(transfer.participants, transfer.participant_nic_ids))
            if transfer.operation == "send_recv":
                endpoints = ((nic_by_rank[transfer.source_rank], "tx"),
                    (nic_by_rank[transfer.destination_rank], "rx"))
            else:
                endpoints = tuple((nic, direction)
                    for nic in sorted(set(transfer.participant_nic_ids))
                    for direction in ("tx", "rx"))
            return tuple(f"communication:roce:nic:{nic}:{direction}"
                for nic, direction in endpoints)
        endpoint_resources = tuple(
            f"communication:endpoint:{call.kernel_spec.hardware_id}:"
            f"{rank}:{transfer.link}"
            for rank in transfer.participants
        )
        return endpoint_resources + (
            f"communication:fabric:{transfer.resource_id}",)
    return (
        f"pool:{call.pool_name}:rank:{call.rank}:{call.resource}",)


def with_earliest_ready_resource_order(plan, predictions):
    """Declare a non-delay device resource order using already priced calls.

    This opt-in preparation preserves every source dependency and kernel.
    It supports expanded device-only plans at time zero, without Host
    launch/runtime events, shared calendars or block recurrence. It does not
    change a duration or select a hardware rate. The caller must retain that
    same scope when building the resulting execution timeline.

    Ready calls sharing a resource tuple share one queue head, avoiding a
    rekey of every pending call whenever that resource advances. Within a
    tuple, calls whose dependencies have finished compete by original call
    position; the global queue compares earliest start then that position.
    """
    if not isinstance(plan, KernelPlan):
        raise TypeError("earliest-ready resource order requires a KernelPlan")
    if plan.block_repeats:
        raise NotImplementedError("earliest-ready resource order requires expanded calls without block repeats")
    if not isinstance(predictions, Mapping) or set(predictions) != {call.call_id for call in plan.calls}:
        raise ValueError("predictions must cover every KernelPlan call exactly once")
    for prediction in predictions.values():
        if not isinstance(prediction, KernelPrediction):
            raise TypeError("earliest-ready resource order requires KernelPrediction values")
        if (not math.isfinite(prediction.t_total_s) or prediction.t_total_s <= 0):
            raise ValueError("earliest-ready resource order requires positive finite call durations")

    calls = plan.calls
    positions = {call.call_id: index for index, call in enumerate(calls)}
    successors = [[] for _ in calls]
    indegree = [len(set(call.dependencies)) for call in calls]
    dependency_end = [0.0] * len(calls)
    resources = tuple(call_reserved_resource_ids(call) for call in calls)
    ready_time = {resource: 0.0 for group in resources for resource in group}
    last_call = {}
    group_index, group_resources = {}, []
    call_groups = []
    for group in resources:
        if group not in group_index:
            group_index[group] = len(group_resources)
            group_resources.append(group)
        call_groups.append(group_index[group])
    waiting = [[] for _ in group_resources]
    eligible = [[] for _ in group_resources]
    versions = [0] * len(group_resources)
    ready = []

    def group_head(group):
        resource_time = max(ready_time[resource] for resource in group_resources[group])
        while waiting[group] and waiting[group][0][0] <= resource_time:
            _, position = heappop(waiting[group])
            heappush(eligible[group], position)
        if eligible[group]:
            return resource_time, eligible[group][0], True
        if waiting[group]:
            time, position = waiting[group][0]
            return time, position, False
        return None

    def enqueue_group(group):
        versions[group] += 1
        head = group_head(group)
        if head is not None:
            time, position, _ = head
            heappush(ready, (time, position, group, versions[group]))

    for index, call in enumerate(calls):
        for parent in set(call.dependencies):
            successors[positions[parent]].append(index)
        if indegree[index] == 0:
            heappush(waiting[call_groups[index]], (0.0, index))
    for group in range(len(group_resources)):
        enqueue_group(group)

    ordered = [None] * len(calls)
    scheduled = 0
    while ready:
        time, position, group, version = heappop(ready)
        if version != versions[group]:
            continue
        head = group_head(group)
        if head is None:
            continue
        start, current_position, was_eligible = head
        if (start, current_position) != (time, position):
            enqueue_group(group)
            continue
        if was_eligible:
            heappop(eligible[group])
        else:
            heappop(waiting[group])
        call = calls[position]
        end = start + predictions[call.call_id].t_total_s
        if not math.isfinite(end):
            raise ValueError("earliest-ready resource schedule exceeds finite time")
        predecessors = {last_call[resource] for resource in resources[position] if resource in last_call}
        predecessors.difference_update(call.dependencies)
        ordered[position] = replace(call, dependencies=call.dependencies + tuple(sorted(predecessors, key=positions.__getitem__)))
        for resource in resources[position]:
            ready_time[resource], last_call[resource] = end, call.call_id
        scheduled += 1
        enqueue_group(group)
        for successor in successors[position]:
            dependency_end[successor] = max(dependency_end[successor], end)
            indegree[successor] -= 1
            if indegree[successor] == 0:
                successor_group = call_groups[successor]
                heappush(waiting[successor_group], (dependency_end[successor], successor))
                enqueue_group(successor_group)
    if scheduled != len(calls):
        raise ValueError("earliest-ready resource order did not schedule every call")
    return KernelPlan(ordered)


def call_host_lanes(call) -> Tuple[Tuple[str, int], ...]:
    """Return every worker lane that must submit this logical call."""

    transfer = call.kernel_spec.transfer
    if (call.kernel_spec.kind is KernelKind.COMMUNICATION
            and transfer is not None):
        return tuple(
            (call.pool_name, rank) for rank in transfer.participants)
    return ((call.pool_name, call.rank),)


def _transfer_structure_key(transfer):
    if transfer is None:
        return None
    return (
        tuple(transfer.participants),
        transfer.link,
        transfer.resource_id,
        transfer.operation,
        transfer.participant_nic_ids,
        transfer.source_rank,
        transfer.destination_rank,
    )


def _common_id_prefix(plan):
    """Longest common string prefix over every call and block identifier.

    Workload node IDs embed the schedule-step ID (``<step_id>:<suffix>``),
    so per-step plans are structurally identical except for this prefix.
    Stripping it yields a step-independent structure space; identifiers
    without the prefix (request-scoped nodes) simply shrink it.
    """

    ids = [call.call_id for call in plan.calls]
    ids.extend(repeat.block_id for repeat in plan.block_repeats)
    prefix = ids[0]
    for value in ids[1:]:
        while not value.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _normalize_id(value, prefix):
    if prefix and value.startswith(prefix):
        return value[len(prefix):]
    return value


def _plan_structure_key(calls, block_repeats, prefix):
    """Hashable fingerprint of every plan attribute the cached schedule
    structure depends on, in prefix-normalized identifier space.

    KernelSpec identity varies across memo-missed lowerings, so specs
    contribute only the fields that drive Host lanes, reserved resources,
    and the runtime prepare occurrence builder (kind/variant plus the
    transfer participants/link/resource/operation); prediction values
    never participate."""

    return (
        tuple(
            (
                _normalize_id(call.call_id, prefix),
                call.pool_name,
                call.rank,
                call.resource,
                call.kernel_spec.kind,
                call.kernel_spec.variant,
                call.kernel_spec.hardware_id,
                _transfer_structure_key(call.kernel_spec.transfer),
                tuple(
                    _normalize_id(dependency, prefix)
                    for dependency in call.dependencies),
            )
            for call in calls
        ),
        tuple(
            (
                _normalize_id(repeat.block_id, prefix),
                tuple(
                    _normalize_id(call_id, prefix)
                    for call_id in repeat.call_ids),
                repeat.repeat_count,
                tuple(
                    (
                        _normalize_id(destination, prefix),
                        tuple(
                            _normalize_id(source, prefix)
                            for source in sources),
                    )
                    for destination, sources in repeat.feedback_dependencies),
            )
            for repeat in block_repeats
        ),
    )


class _LogicalCall(NamedTuple):
    call_id: str
    dependencies: Tuple[str, ...]


class _PlanScheduleContext(NamedTuple):
    """Structure-only per-plan schedule artifacts in normalized ID space.

    Every entry is derived from the plan structure fingerprinted by
    ``_plan_structure_key`` and contains no float state, so reuse across
    steps is bit-exact: per-step durations still flow through the live
    recurrence in ``build_execution_timeline``.  Identifiers are
    prefix-normalized; callers translate at the boundary.
    """

    logical_schedule: tuple
    occurrence_indices_by_call: Mapping[str, frozenset]
    lane_occurrence_keys: frozenset
    lanes_by_call_id: Mapping[str, tuple]


# Each entry owns a complete call-structure graph. Retaining thousands of
# unique decode structures can consume more RAM than the current forward.
# Eviction only rebuilds metadata; live durations and recurrence are unchanged.
_PLAN_CONTEXT_CACHE_LIMIT = 16
_PLAN_CONTEXTS: OrderedDict = OrderedDict()
_GRAPH_SEGMENT_MAPPING_CACHE_LIMIT = 16
_GRAPH_SEGMENT_MAPPINGS: OrderedDict = OrderedDict()
_RUNTIME_PREPARE_MAPPING_CACHE_LIMIT = 16
_RUNTIME_PREPARE_MAPPINGS: OrderedDict = OrderedDict()
_RUNTIME_GRAPH_SEGMENTS_CACHE_LIMIT = 16
_RUNTIME_GRAPH_SEGMENTS: OrderedDict = OrderedDict()
_NORMALIZED_GRAPH_SEGMENTS_CACHE_LIMIT = 16
_NORMALIZED_GRAPH_SEGMENTS: OrderedDict = OrderedDict()
_INTERVAL_PAIR = struct.Struct("!dd")
_HOST_LANE_PROGRAM_CACHE_LIMIT = 16
_HOST_LANE_PROGRAMS: OrderedDict = OrderedDict()


def _cache_put(cache: OrderedDict, limit: int, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _plan_schedule_context(plan, calls, prefix):
    """Return the cached structure-only schedule context for one plan."""

    structure_key = _plan_structure_key(calls, plan.block_repeats, prefix)
    cached = (
        _PLAN_CONTEXTS[structure_key]
        if structure_key in _PLAN_CONTEXTS else None)
    if cached is not None:
        _PLAN_CONTEXTS.move_to_end(structure_key)
        return structure_key, cached
    norm_calls = tuple(
        _LogicalCall(
            _normalize_id(call.call_id, prefix),
            tuple(
                _normalize_id(dependency, prefix)
                for dependency in call.dependencies),
        )
        for call in calls
    )
    norm_calls_by_id = {call.call_id: call for call in norm_calls}
    norm_repeat_by_call = {}
    norm_repeat_call_ids_by_block = {}
    norm_feedback_by_destination_by_block = {}
    norm_block_repeats = []
    for repeat in plan.block_repeats:
        norm_block_id = _normalize_id(repeat.block_id, prefix)
        norm_call_ids = tuple(
            _normalize_id(call_id, prefix) for call_id in repeat.call_ids)
        norm_feedback = tuple(
            (
                _normalize_id(destination, prefix),
                tuple(
                    _normalize_id(source, prefix) for source in sources),
            )
            for destination, sources in repeat.feedback_dependencies)
        norm_repeat = KernelBlockRepeat(
            norm_block_id,
            norm_call_ids,
            repeat.repeat_count,
            norm_feedback,
        )
        norm_block_repeats.append(norm_repeat)
        norm_repeat_call_ids_by_block[norm_block_id] = set(norm_call_ids)
        norm_feedback_by_destination_by_block[norm_block_id] = dict(
            norm_feedback)
        ordered_call_ids = tuple(
            call.call_id for call in norm_calls
            if call.call_id in norm_repeat_call_ids_by_block[norm_block_id])
        if ordered_call_ids != norm_call_ids:
            raise ValueError(
                "kernel block repeat calls must follow topological order")
        for call_id in norm_call_ids:
            norm_repeat_by_call[call_id] = norm_repeat
    logical_schedule = tuple(_logical_schedule(
        plan,
        norm_calls,
        norm_calls_by_id,
        norm_repeat_by_call,
        norm_repeat_call_ids_by_block,
        norm_feedback_by_destination_by_block,
        block_repeats=tuple(norm_block_repeats),
    ))
    occurrence_indices_by_call, lane_occurrence_keys = (
        _occurrence_context_data(
            plan,
            {
                call.call_id: norm_repeat_by_call[
                    _normalize_id(call.call_id, prefix)]
                for call in plan.calls
                if _normalize_id(call.call_id, prefix) in norm_repeat_by_call
            },
            normalize=lambda call_id: _normalize_id(call_id, prefix),
        ))
    context = _PlanScheduleContext(
        logical_schedule,
        occurrence_indices_by_call,
        lane_occurrence_keys,
        {
            _normalize_id(call.call_id, prefix): call_host_lanes(call)
            for call in plan.calls
        },
    )
    _cache_put(
        _PLAN_CONTEXTS, _PLAN_CONTEXT_CACHE_LIMIT, structure_key, context)
    return structure_key, context


def _occurrence_context_data(plan, repeat_by_call, normalize=None):
    """Occurrence index sets and lane/occurrence keys for one plan.

    ``repeat_by_call`` is keyed by raw call ID; ``normalize`` maps the IDs
    emitted into the returned structures."""

    occurrence_indices_by_call = {}
    lane_occurrence_keys = set()
    for call in plan.calls:
        indices = (
            frozenset(range(repeat_by_call[call.call_id].repeat_count))
            if call.call_id in repeat_by_call else frozenset((-1,))
        )
        call_id = call.call_id if normalize is None else normalize(
            call.call_id)
        occurrence_indices_by_call[call_id] = indices
        for occurrence_index in indices:
            for pool_name, rank in call_host_lanes(call):
                lane_occurrence_keys.add((
                    pool_name, rank, call_id, occurrence_index))
    return occurrence_indices_by_call, frozenset(lane_occurrence_keys)


def _plan_schedule_inputs(plan):
    """Validate and return the compact inputs used by logical expansion."""

    if not isinstance(plan, KernelPlan):
        raise TypeError("execution schedule requires a KernelPlan")
    calls = plan.topological_order()
    calls_by_id = {call.call_id: call for call in calls}
    repeat_by_call = {}
    repeat_call_ids_by_block = {}
    feedback_by_destination_by_block = {}
    for repeat in plan.block_repeats:
        repeat_call_ids = set(repeat.call_ids)
        repeat_call_ids_by_block[repeat.block_id] = repeat_call_ids
        feedback_by_destination_by_block[repeat.block_id] = dict(
            repeat.feedback_dependencies)
        ordered_call_ids = tuple(
            call.call_id for call in calls if call.call_id in repeat_call_ids
        )
        if ordered_call_ids != repeat.call_ids:
            raise ValueError(
                "kernel block repeat calls must follow topological order")
        for call_id in repeat.call_ids:
            repeat_by_call[call_id] = repeat
    return (
        calls,
        calls_by_id,
        repeat_by_call,
        repeat_call_ids_by_block,
        feedback_by_destination_by_block,
    )


def _runtime_graph_segment_mapping(
    context,
    calls_by_id,
    runtime_graph_mode,
    runtime_graph_segments,
):
    """Validate explicit Graph coverage and index it by call occurrence/lane."""

    if runtime_graph_mode not in ("NONE", "FULL", "PIECEWISE"):
        raise ValueError(
            f"unknown runtime Graph mode: {runtime_graph_mode!r}")
    if runtime_graph_segments is None:
        segments = ()
    elif not isinstance(runtime_graph_segments, (list, tuple)):
        raise TypeError("runtime Graph segments must be a sequence")
    else:
        segments = tuple(runtime_graph_segments)
    if any(
            not isinstance(segment, RuntimeGraphSegmentDescriptor)
            for segment in segments):
        raise TypeError(
            "runtime Graph segments require RuntimeGraphSegmentDescriptor")
    if runtime_graph_mode == "NONE":
        if segments:
            raise ValueError("runtime Graph mode NONE cannot carry segments")
        return {}, ()
    if not segments:
        raise NotImplementedError(
            f"{runtime_graph_mode} Host timing requires an explicit runtime "
            "Graph segment descriptor")

    valid_occurrence_indices = context.occurrence_indices_by_call
    expected_lane_occurrences = context.lane_occurrence_keys

    segment_identity = set()
    segment_by_lane_occurrence = {}
    for segment in segments:
        identity = (segment.pool_name, segment.rank, segment.segment_id)
        if identity in segment_identity:
            raise ValueError(
                "runtime Graph segment identity must be unique per Host lane")
        segment_identity.add(identity)
        segment_hardware_ids = set()
        for call_id, occurrence_index in segment.call_occurrences:
            if call_id not in calls_by_id:
                raise ValueError(
                    f"runtime Graph segment references unknown call {call_id!r}")
            if occurrence_index not in valid_occurrence_indices[call_id]:
                raise ValueError(
                    "runtime Graph segment references an invalid call "
                    f"occurrence: {(call_id, occurrence_index)!r}")
            call = calls_by_id[call_id]
            segment_hardware_ids.add(call.kernel_spec.hardware_id)
            lane = (segment.pool_name, segment.rank)
            if lane not in call_host_lanes(call):
                raise ValueError(
                    "runtime Graph segment lane does not submit call "
                    f"occurrence: {identity!r}, {(call_id, occurrence_index)!r}")
            key = (
                segment.pool_name,
                segment.rank,
                call_id,
                occurrence_index,
            )
            if key in segment_by_lane_occurrence:
                raise ValueError(
                    "runtime Graph segments overlap on one call occurrence/lane")
            segment_by_lane_occurrence[key] = segment
        if len(segment_hardware_ids) != 1:
            raise ValueError(
                "runtime Graph segment cannot span hardware profiles")

    covered = set(segment_by_lane_occurrence)
    if runtime_graph_mode == "FULL" and covered != expected_lane_occurrences:
        missing = expected_lane_occurrences - covered
        extra = covered - expected_lane_occurrences
        raise ValueError(
            "FULL runtime Graph segments must cover every call occurrence/lane: "
            f"missing={len(missing)}, extra={len(extra)}")
    if runtime_graph_mode == "PIECEWISE" and covered == expected_lane_occurrences:
        raise ValueError(
            "PIECEWISE runtime Graph descriptor must retain eager occurrences")
    return segment_by_lane_occurrence, segments


def _runtime_prepare_mapping(
    context,
    plan,
    calls_by_id,
    runtime_graph_mode,
    graph_segment_by_lane_occurrence,
    host_runtime_prepare_profiles,
    runtime_prepare_events,
):
    """Validate source-authored eager prepare events for each call occurrence.

    Occurrence rules are authored against the eager source program and are
    therefore runtime-mode agnostic.  A call occurrence captured inside a
    runtime Graph segment replays without Host preparation, so its declared
    prepare events are discarded here; only occurrences submitted eagerly
    (every occurrence in NONE mode, the uncovered remainder in PIECEWISE
    mode) execute and are charged for them.
    """

    if host_runtime_prepare_profiles is None:
        if runtime_prepare_events:
            raise ValueError(
                "runtime prepare events require runtime prepare profiles")
        return {}
    if (not isinstance(host_runtime_prepare_profiles, Mapping)
            or not host_runtime_prepare_profiles):
        raise TypeError(
            "runtime prepare profiles must be a non-empty mapping")
    hardware_ids = {call.kernel_spec.hardware_id for call in plan.calls}
    if set(host_runtime_prepare_profiles) != hardware_ids:
        raise ValueError(
            "runtime prepare profiles must cover KernelPlan hardware exactly")
    for hardware_id, profile in host_runtime_prepare_profiles.items():
        if not isinstance(profile, HostRuntimePrepareProfile):
            raise TypeError(
                "runtime prepare profile mapping contains a non-profile")
        if profile.hardware_id != hardware_id:
            raise ValueError(
                "runtime prepare profile key disagrees with hardware_id")
    if runtime_prepare_events is None:
        return {}
    if not isinstance(runtime_prepare_events, Mapping):
        raise TypeError("runtime prepare events must be a mapping")
    valid_keys = context.lane_occurrence_keys
    normalized = {}
    for key, mechanisms in runtime_prepare_events.items():
        if not isinstance(key, tuple) or len(key) != 4:
            raise ValueError(
                "runtime prepare event key must be lane/call/occurrence")
        if key not in valid_keys:
            raise ValueError(
                f"runtime prepare event references unknown occurrence: {key!r}")
        if not isinstance(mechanisms, (list, tuple)) or not mechanisms:
            raise ValueError(
                "runtime prepare event must contain at least one mechanism")
        call = calls_by_id[key[2]]
        profile = host_runtime_prepare_profiles[call.kernel_spec.hardware_id]
        normalized_mechanisms = tuple(mechanisms)
        if any(not isinstance(mechanism, RuntimePrepareEvent)
               for mechanism in normalized_mechanisms):
            raise ValueError(
                "runtime prepare events must be RuntimePrepareEvent objects")
        names = tuple(mechanism.mechanism for mechanism in normalized_mechanisms)
        if len(set(names)) != len(names):
            raise ValueError("runtime prepare mechanisms cannot repeat")
        for mechanism in normalized_mechanisms:
            profile.service_time_s(
                mechanism.mechanism,
                pool_name=key[0],
                rank=key[1],
            )
        normalized[key] = normalized_mechanisms
    if runtime_graph_mode is not None and runtime_graph_mode != "NONE":
        normalized = {
            key: (
                tuple(
                    mechanism for mechanism in mechanisms
                    if not mechanism.graph_capturable
                )
                if key in graph_segment_by_lane_occurrence else mechanisms
            )
            for key, mechanisms in normalized.items()
        }
        normalized = {
            key: mechanisms for key, mechanisms in normalized.items()
            if mechanisms
        }
    return normalized


def _translate_segment_ids(segments, translate):
    """Map every call identifier in validated segment descriptors.

    The translation is a bijection on already-validated identifiers
    (prefix strip or prefix prepend), so every invariant checked by
    ``RuntimeGraphSegmentDescriptor.__post_init__`` is preserved and the
    clones are built directly.
    """

    translated = []
    for segment in segments:
        clone = object.__new__(RuntimeGraphSegmentDescriptor)
        object.__setattr__(clone, "segment_id", segment.segment_id)
        object.__setattr__(clone, "pool_name", segment.pool_name)
        object.__setattr__(clone, "rank", segment.rank)
        object.__setattr__(clone, "sync_before_replay", segment.sync_before_replay)
        object.__setattr__(clone, "call_occurrences", tuple(
            (translate(call_id), occurrence_index)
            for call_id, occurrence_index in segment.call_occurrences))
        translated.append(clone)
    return tuple(translated)


def _segments_match_normalized(segments, raw_segments, prefix):
    """True when raw_segments normalize exactly to the cached segments.

    Pure comparison against the prefix-normalized canonical tuple; avoids
    re-translating (and re-hashing) descriptor clones every step."""

    if len(segments) != len(raw_segments):
        return False
    for segment, raw_segment in zip(segments, raw_segments):
        if (segment.segment_id != raw_segment.segment_id
                or segment.pool_name != raw_segment.pool_name
                or segment.rank != raw_segment.rank
                or segment.sync_before_replay != raw_segment.sync_before_replay):
            return False
        occurrences = segment.call_occurrences
        raw_occurrences = raw_segment.call_occurrences
        if len(occurrences) != len(raw_occurrences):
            return False
        for (call_id, occurrence_index), (raw_call_id, raw_index) in zip(
                occurrences, raw_occurrences):
            if occurrence_index != raw_index:
                return False
            if _normalize_id(raw_call_id, prefix) != call_id:
                return False
    return True


def _cached_graph_segment_mapping(
    structure_key,
    context,
    calls_by_id,
    runtime_graph_mode,
    runtime_graph_segments,
):
    """Reuse the structure-only Graph coverage mapping for identical inputs.

    ``runtime_graph_segments`` must already be in the same normalized
    identifier space as ``structure_key``; the descriptors are frozen
    dataclasses, so the cache key hashes them by value.
    """

    cache_key = (structure_key, runtime_graph_mode, runtime_graph_segments)
    cached = (
        _GRAPH_SEGMENT_MAPPINGS[cache_key]
        if cache_key in _GRAPH_SEGMENT_MAPPINGS else None)
    if cached is not None:
        _GRAPH_SEGMENT_MAPPINGS.move_to_end(cache_key)
        return cached
    result = _runtime_graph_segment_mapping(
        context,
        calls_by_id,
        runtime_graph_mode,
        runtime_graph_segments,
    )
    _cache_put(
        _GRAPH_SEGMENT_MAPPINGS,
        _GRAPH_SEGMENT_MAPPING_CACHE_LIMIT,
        cache_key,
        result,
    )
    return result


def _cached_runtime_prepare_mapping(
    structure_key,
    context,
    plan,
    calls_by_id,
    runtime_graph_mode,
    graph_segment_by_lane_occurrence,
    host_runtime_prepare_profiles,
    runtime_prepare_events,
    events_token,
):
    """Reuse the structure-only prepare-event mapping for identical inputs.

    ``runtime_prepare_events`` must already be in the same normalized
    identifier space as ``structure_key``; ``events_token`` is a caller-
    retained object whose identity attests that the events are the same
    structure-determined content seen before.  The cached value also
    carries the per-event Host service times: they are pure functions of
    the (cached) events and the constant prepare profiles, so recomputing
    them once keeps every downstream float bit-identical."""

    cache_key = (
        structure_key,
        id(events_token),
        id(graph_segment_by_lane_occurrence),
        runtime_graph_mode,
        id(host_runtime_prepare_profiles),
    )
    cached = (
        _RUNTIME_PREPARE_MAPPINGS[cache_key]
        if cache_key in _RUNTIME_PREPARE_MAPPINGS else None)
    if (
            cached is not None
            and cached[0] is events_token
            and cached[1] is graph_segment_by_lane_occurrence
            and cached[2] is host_runtime_prepare_profiles):
        _RUNTIME_PREPARE_MAPPINGS.move_to_end(cache_key)
        return cached[3], cached[4]
    result = _runtime_prepare_mapping(
        context,
        plan,
        calls_by_id,
        runtime_graph_mode,
        graph_segment_by_lane_occurrence,
        host_runtime_prepare_profiles,
        runtime_prepare_events,
    )
    services = {}
    if host_runtime_prepare_profiles is not None:
        for lane_occurrence, mechanisms in result.items():
            call = calls_by_id[lane_occurrence[2]]
            profile = host_runtime_prepare_profiles[
                call.kernel_spec.hardware_id]
            services[lane_occurrence] = tuple(
                profile.service_time_s(
                    event.mechanism,
                    pool_name=lane_occurrence[0],
                    rank=lane_occurrence[1],
                ) * event.count
                for event in mechanisms
            )
    _cache_put(
        _RUNTIME_PREPARE_MAPPINGS,
        _RUNTIME_PREPARE_MAPPING_CACHE_LIMIT,
        cache_key,
        (
            events_token,
            graph_segment_by_lane_occurrence,
            host_runtime_prepare_profiles,
            result,
            services,
        ),
    )
    return result, services


def _host_lane_program(
    context,
    graph_segment_by_lane_occurrence,
    prepare_services_by_lane_occurrence,
    normalized_graph_segments,
):
    """Resolve each occurrence's Host lanes, segment, and prepare services.

    Which lanes one occurrence dispatches on, whether a runtime Graph segment
    covers it, and how long its runtime-prepare events take are all functions
    of the plan structure and the structure-only mappings above -- no float
    state from the live recurrence reaches them.  Resolving them once per
    structure therefore hands the per-step recurrence the same lane program it
    would have rebuilt, without re-forming a lane/occurrence key and hashing
    it twice for every lane of every occurrence of every step.  The segment
    traversal is a fail-closed check over that same static order, so it runs
    here, once, for the same reason.
    """

    traversal = _GraphSegmentTraversal(normalized_graph_segments)
    lanes_by_call_id = context.lanes_by_call_id
    program = {}
    for logical_key, _dependency_keys, _position, _successors in (
            context.logical_schedule):
        entries = []
        for lane in lanes_by_call_id[logical_key[0]]:
            occurrence_lane = (*lane, *logical_key)
            segment = (
                graph_segment_by_lane_occurrence[occurrence_lane]
                if occurrence_lane in graph_segment_by_lane_occurrence
                else None)
            traversal.observe(lane, logical_key, segment)
            service_times = (
                prepare_services_by_lane_occurrence[occurrence_lane]
                if occurrence_lane in prepare_services_by_lane_occurrence
                else None)
            entries.append((lane, segment, service_times))
        program[logical_key] = tuple(entries)
    traversal.finish()
    return program


class _DeviceLaunchGapTracker:
    """Resolve each occurrence's device-side idle before it starts.

    Occurrences are observed in schedule order with the ``(lane, segment)``
    pairs they launch on.  A launch pays the measured gap of the boundary it
    crosses from the previous launch on the same lane (next Graph node, Graph
    entry, Graph exit or eager-to-eager); the first launch on a lane is bound
    by its own Host submission and pays none.  The gap is a structure-only
    function of launch order and Graph membership, never of duration.
    """

    def __init__(self, host_launch_profiles):
        self._profiles = host_launch_profiles
        self._previous_by_lane = {}

    def observe(self, hardware_id, lanes):
        profile = self._profiles[hardware_id]
        gap_s = 0.0
        for lane, segment in lanes:
            current = (
                None if segment is None
                else (segment.pool_name, segment.rank, segment.segment_id))
            if lane in self._previous_by_lane:
                previous = self._previous_by_lane[lane]
                if current is None:
                    gap_class = "eager" if previous is None else "graph_exit"
                elif previous == current:
                    gap_class = "graph_node"
                else:
                    gap_class = "graph_entry"
                gap_s = max(gap_s, profile.device_launch_gap_s(gap_class))
            self._previous_by_lane[lane] = current
        return gap_s


def _cached_host_lane_program(
    structure_key,
    context,
    graph_segment_by_lane_occurrence,
    prepare_services_by_lane_occurrence,
    normalized_graph_segments,
):
    """Reuse the structure-only Host lane program for identical inputs."""

    cache_key = (
        structure_key,
        id(graph_segment_by_lane_occurrence),
        id(prepare_services_by_lane_occurrence),
        id(normalized_graph_segments),
    )
    cached = (
        _HOST_LANE_PROGRAMS[cache_key]
        if cache_key in _HOST_LANE_PROGRAMS else None)
    if (cached is not None
            and cached[0] is graph_segment_by_lane_occurrence
            and cached[1] is prepare_services_by_lane_occurrence
            and cached[2] is normalized_graph_segments):
        _HOST_LANE_PROGRAMS.move_to_end(cache_key)
        return cached[3]
    program = _host_lane_program(
        context,
        graph_segment_by_lane_occurrence,
        prepare_services_by_lane_occurrence,
        normalized_graph_segments,
    )
    _cache_put(
        _HOST_LANE_PROGRAMS,
        _HOST_LANE_PROGRAM_CACHE_LIMIT,
        cache_key,
        (
            graph_segment_by_lane_occurrence,
            prepare_services_by_lane_occurrence,
            normalized_graph_segments,
            program,
        ),
    )
    return program


class _GraphSegmentTraversal:
    """Fail closed when declared segment members are not lane-contiguous."""

    def __init__(self, segments):
        self._segment_by_identity = {
            (segment.pool_name, segment.rank, segment.segment_id): segment
            for segment in segments
        }
        self._next_index_by_identity = {
            identity: 0 for identity in self._segment_by_identity
        }
        self._active_identity_by_lane = {}

    def observe(self, lane, logical_key, segment):
        active_identity = (
            self._active_identity_by_lane[lane]
            if lane in self._active_identity_by_lane else None)
        if segment is None:
            if active_identity is not None:
                raise ValueError(
                    "runtime Graph segment members must be contiguous on "
                    f"Host lane {lane!r}")
            return
        identity = (lane[0], lane[1], segment.segment_id)
        if active_identity is None:
            self._active_identity_by_lane[lane] = identity
        elif active_identity != identity:
            raise ValueError(
                "runtime Graph segments overlap or interleave on Host lane "
                f"{lane!r}")
        next_index = self._next_index_by_identity[identity]
        if segment.call_occurrences[next_index] != logical_key:
            raise ValueError(
                "runtime Graph segment call occurrences disagree with "
                f"execution order for {identity!r}")
        next_index += 1
        self._next_index_by_identity[identity] = next_index
        if next_index == len(segment.call_occurrences):
            del self._active_identity_by_lane[lane]

    def finish(self):
        incomplete = tuple(
            identity for identity, next_index in self._next_index_by_identity.items()
            if next_index != len(
                self._segment_by_identity[identity].call_occurrences)
        )
        if self._active_identity_by_lane or incomplete:
            raise ValueError(
                "runtime Graph segment traversal did not consume every member")


def _logical_schedule(
    plan,
    calls,
    calls_by_id,
    repeat_by_call,
    repeat_call_ids_by_block,
    feedback_by_destination_by_block,
    block_repeats=None,
):
    """Yield the expanded Kahn order from compact dependency templates.

    ``block_repeats`` defaults to ``plan.block_repeats``; normalized
    callers pass the repeats matching their translated identifier space.
    """

    if block_repeats is None:
        block_repeats = plan.block_repeats
    nonrepeated_position_by_call = {}
    repeat_layout_by_call = {}
    inserted_repeat_ids = set()
    next_position = 0
    for call in calls:
        if call.call_id not in repeat_by_call:
            nonrepeated_position_by_call[call.call_id] = next_position
            next_position += 1
            continue
        repeat = repeat_by_call[call.call_id]
        if repeat.block_id in inserted_repeat_ids:
            continue
        for call_index, call_id in enumerate(repeat.call_ids):
            repeat_layout_by_call[call_id] = (
                next_position,
                call_index,
                len(repeat.call_ids),
            )
        next_position += repeat.repeat_count * len(repeat.call_ids)
        inserted_repeat_ids.add(repeat.block_id)
    if len(inserted_repeat_ids) != len(block_repeats):
        raise RuntimeError("kernel block repeat was not placed in logical plan")

    def iter_logical_keys():
        inserted = set()
        for call in calls:
            if call.call_id not in repeat_by_call:
                yield call.call_id, -1
                continue
            repeat = repeat_by_call[call.call_id]
            if repeat.block_id in inserted:
                continue
            for occurrence_index in range(repeat.repeat_count):
                for call_id in repeat.call_ids:
                    yield call_id, occurrence_index
            inserted.add(repeat.block_id)

    def logical_position(logical_key):
        call_id, occurrence_index = logical_key
        if occurrence_index < 0:
            return nonrepeated_position_by_call[call_id]
        base_position, call_index, block_width = repeat_layout_by_call[call_id]
        return base_position + occurrence_index * block_width + call_index

    def dependency_key(dependency_call_id):
        if dependency_call_id not in repeat_by_call:
            return dependency_call_id, -1
        dependency_repeat = repeat_by_call[dependency_call_id]
        return dependency_call_id, dependency_repeat.repeat_count - 1

    def dependencies_for(logical_key):
        call_id, occurrence_index = logical_key
        call = calls_by_id[call_id]
        if occurrence_index < 0:
            dependencies = tuple(
                dependency_key(dependency)
                for dependency in call.dependencies
            )
        else:
            repeat = repeat_by_call[call_id]
            block_call_ids = repeat_call_ids_by_block[repeat.block_id]
            internal_dependencies = tuple(
                (dependency, occurrence_index)
                for dependency in call.dependencies
                if dependency in block_call_ids
            )
            if occurrence_index == 0:
                dependencies = internal_dependencies + tuple(
                    dependency_key(dependency)
                    for dependency in call.dependencies
                    if dependency not in block_call_ids
                )
            else:
                feedback_by_destination = (
                    feedback_by_destination_by_block[repeat.block_id])
                feedback_sources = (
                    feedback_by_destination[call_id]
                    if call_id in feedback_by_destination else ()
                )
                dependencies = internal_dependencies + tuple(
                    (source, occurrence_index - 1)
                    for source in feedback_sources
                )
        return tuple(dict.fromkeys(dependencies))

    successors_by_call = {call.call_id: [] for call in calls}
    for call in calls:
        for dependency in dict.fromkeys(call.dependencies):
            successors_by_call[dependency].append(call.call_id)
    feedback_successors_by_block = {
        repeat.block_id: {call_id: [] for call_id in repeat.call_ids}
        for repeat in block_repeats
    }
    for repeat in block_repeats:
        feedback_successors = feedback_successors_by_block[repeat.block_id]
        for destination, sources in repeat.feedback_dependencies:
            for source in sources:
                feedback_successors[source].append(destination)

    def first_occurrence_key(call_id):
        return (
            (call_id, 0)
            if call_id in repeat_by_call else (call_id, -1)
        )

    def successors_for(logical_key):
        call_id, occurrence_index = logical_key
        if occurrence_index < 0:
            return tuple(
                first_occurrence_key(successor)
                for successor in successors_by_call[call_id]
            )
        repeat = repeat_by_call[call_id]
        block_call_ids = repeat_call_ids_by_block[repeat.block_id]
        internal_successors = tuple(
            (successor, occurrence_index)
            for successor in successors_by_call[call_id]
            if successor in block_call_ids
        )
        feedback_successors = (
            tuple(
                (successor, occurrence_index + 1)
                for successor in feedback_successors_by_block[
                    repeat.block_id][call_id]
            )
            if occurrence_index + 1 < repeat.repeat_count else ()
        )
        external_successors = (
            tuple(
                first_occurrence_key(successor)
                for successor in successors_by_call[call_id]
                if successor not in block_call_ids
            )
            if occurrence_index == repeat.repeat_count - 1 else ()
        )
        return tuple(dict.fromkeys(
            internal_successors
            + feedback_successors
            + external_successors
        ))

    ready = (
        logical_key for logical_key in iter_logical_keys()
        if not dependencies_for(logical_key)
    )
    remaining_indegree = {}
    scheduled_count = 0
    while True:
        next_ready = []
        found_ready = False
        for logical_key in ready:
            found_ready = True
            dependencies = dependencies_for(logical_key)
            successors = successors_for(logical_key)
            yield (
                logical_key,
                dependencies,
                logical_position(logical_key),
                len(successors),
            )
            scheduled_count += 1
            for successor in successors:
                remaining = (
                    remaining_indegree[successor]
                    if successor in remaining_indegree
                    else len(dependencies_for(successor))
                ) - 1
                if remaining == 0:
                    if successor in remaining_indegree:
                        del remaining_indegree[successor]
                    next_ready.append(successor)
                else:
                    remaining_indegree[successor] = remaining
        if not found_ready:
            break
        ready = sorted(next_ready, key=logical_position)
    if scheduled_count != plan.logical_call_count:
        raise ValueError("logical KernelPlan dependencies must be acyclic")


def iter_call_occurrences(plan):
    """Yield template-call occurrence keys in actual dependency execution order.

    This shares the timeline scheduler without materializing expanded calls or
    rebinding their memory, so source-boundary adapters can retain compaction.
    """
    for key, _dependencies, _position, _successors in _logical_schedule(
        plan, *_plan_schedule_inputs(plan)
    ):
        yield key


def build_runtime_graph_segments(
    workload: WorkloadGraph,
    plan: KernelPlan,
    runtime_graph_mode: str,
    piecewise_eager_workload_kinds: Tuple[str, ...],
    *,
    sync_before_replay: bool = False,
) -> Tuple[RuntimeGraphSegmentDescriptor, ...]:
    """Derive exact per-lane Graph replay segments in this plan's ID space."""

    segments, prefix = build_plan_structure_runtime_graph_segments(
        workload,
        plan,
        runtime_graph_mode,
        piecewise_eager_workload_kinds,
        sync_before_replay=sync_before_replay,
    )
    if prefix is None:
        return segments
    return _translate_segment_ids(
        segments,
        lambda call_id: prefix + call_id if prefix else call_id)


def build_plan_structure_runtime_graph_segments(
    workload: WorkloadGraph,
    plan: KernelPlan,
    runtime_graph_mode: str,
    piecewise_eager_workload_kinds: Tuple[str, ...],
    *,
    sync_before_replay: bool = False,
):
    """Derive exact per-lane Graph replay segments and name their ID space.

    FULL captures every logical call occurrence in one replay per Host lane.
    PIECEWISE leaves calls lowered from configured split-op workload kinds eager
    and captures each lane-contiguous run between those eager boundaries.

    Returns ``(segments, prefix)``.  ``prefix`` is the common plan identifier
    prefix the segments were normalized against, so a caller that works in the
    same plan-structure space consumes them without translating; it is
    ``None`` when normalized workload node IDs collided and the segments
    therefore stayed in the plan's own identifier space.
    """

    if not isinstance(workload, WorkloadGraph):
        raise TypeError("runtime Graph segment production requires WorkloadGraph")
    if runtime_graph_mode not in ("NONE", "FULL", "PIECEWISE"):
        raise ValueError(
            f"unknown runtime Graph mode: {runtime_graph_mode!r}")
    if not isinstance(piecewise_eager_workload_kinds, (list, tuple)):
        raise TypeError("piecewise eager workload kinds must be a sequence")
    if not isinstance(sync_before_replay, bool):
        raise TypeError("sync_before_replay must be a bool")
    try:
        eager_kinds = tuple(
            WorkloadKind(kind) for kind in piecewise_eager_workload_kinds)
    except ValueError as error:
        raise ValueError(
            "piecewise eager workload kind is unsupported") from error
    if len(eager_kinds) != len(set(eager_kinds)):
        raise ValueError("piecewise eager workload kinds must be unique")
    if runtime_graph_mode == "PIECEWISE":
        if not eager_kinds:
            raise ValueError(
                "PIECEWISE runtime Graph requires eager workload kinds")
    if runtime_graph_mode == "NONE":
        # No segment carries an identifier, so there is nothing to normalize.
        return (), None

    calls = plan.topological_order()
    hardware_ids = {call.kernel_spec.hardware_id for call in calls}
    if len(hardware_ids) != 1:
        raise ValueError(
            "runtime Graph segment production requires one hardware")
    # Some runtime bindings synchronize the current stream immediately before
    # every replay on the ordinary main-model path.  This is a barrier, not an
    # additive duration: the timeline accounts for the wait against the
    # already-scheduled device predecessor, while the independent runtime
    # microbenchmark only supplies the small API-call service baseline.  The
    # binding capability is supplied by the hardware/runtime owner rather than
    # inferred from a canonical hardware ID in this generic lowering layer.
    prefix = _common_id_prefix(plan)
    node_kind_by_id = {}
    collision = False
    for node in workload.nodes:
        normalized_id = _normalize_id(node.node_id, prefix)
        if (normalized_id in node_kind_by_id
                and node_kind_by_id[normalized_id] is not node.kind):
            collision = True
            break
        node_kind_by_id[normalized_id] = node.kind
    eager_call_ids = set()
    if not collision:
        for call in calls:
            normalized_source_ids = tuple(
                _normalize_id(node_id, prefix)
                for node_id in call.source_node_ids)
            unknown_source_ids = tuple(
                node_id for node_id in normalized_source_ids
                if node_id not in node_kind_by_id
            )
            if unknown_source_ids:
                raise ValueError(
                    "KernelPlan call references workload nodes absent from the "
                    f"source graph: {call.call_id!r}, {unknown_source_ids!r}")
            if any(
                    node_kind_by_id[node_id] in eager_kinds
                    for node_id in normalized_source_ids):
                eager_call_ids.add(call.call_id)

    if collision:
        # Normalized workload node IDs collide; schedule in raw ID space and
        # skip the structure cache rather than risk a wrong eager boundary.
        raw_node_kind_by_id = {
            node.node_id: node.kind for node in workload.nodes}
        for call in calls:
            unknown_source_ids = tuple(
                node_id for node_id in call.source_node_ids
                if node_id not in raw_node_kind_by_id)
            if unknown_source_ids:
                raise ValueError(
                    "KernelPlan call references workload nodes absent from the "
                    f"source graph: {call.call_id!r}, {unknown_source_ids!r}")
            if any(
                    raw_node_kind_by_id[node_id] in eager_kinds
                    for node_id in call.source_node_ids):
                eager_call_ids.add(call.call_id)
        (
            raw_calls,
            raw_calls_by_id,
            repeat_by_call,
            repeat_call_ids_by_block,
            feedback_by_destination_by_block,
        ) = _plan_schedule_inputs(plan)
        return _runtime_graph_segments_from_schedule(
            raw_calls_by_id,
            _logical_schedule(
                plan,
                raw_calls,
                raw_calls_by_id,
                repeat_by_call,
                repeat_call_ids_by_block,
                feedback_by_destination_by_block,
            ),
            runtime_graph_mode,
            frozenset(eager_call_ids),
            sync_before_replay=sync_before_replay,
        ), None

    structure_key, context = _plan_schedule_context(plan, calls, prefix)
    normalized_eager_call_ids = frozenset(
        _normalize_id(call_id, prefix) for call_id in eager_call_ids)
    segments_key = (
        structure_key,
        runtime_graph_mode,
        tuple(eager_kinds),
        normalized_eager_call_ids,
        sync_before_replay,
    )
    normalized_segments = (
        _RUNTIME_GRAPH_SEGMENTS[segments_key]
        if segments_key in _RUNTIME_GRAPH_SEGMENTS else None)
    if normalized_segments is None:
        normalized_segments = _runtime_graph_segments_from_schedule(
            {
                _normalize_id(call.call_id, prefix): call for call in calls
            },
            context.logical_schedule,
            runtime_graph_mode,
            normalized_eager_call_ids,
            sync_before_replay=sync_before_replay,
            normalize=lambda call_id: _normalize_id(call_id, prefix),
        )
        _cache_put(
            _RUNTIME_GRAPH_SEGMENTS,
            _RUNTIME_GRAPH_SEGMENTS_CACHE_LIMIT,
            segments_key,
            normalized_segments,
        )
    else:
        _RUNTIME_GRAPH_SEGMENTS.move_to_end(segments_key)
    return normalized_segments, prefix


def _runtime_graph_segments_from_schedule(
    calls_by_id,
    logical_schedule,
    runtime_graph_mode,
    eager_call_ids,
    *,
    sync_before_replay=False,
    normalize=lambda call_id: call_id,
):
    segment_members_by_lane = {}
    segment_counts_by_lane = {}
    segments = []

    def flush_lane(lane):
        members = (
            segment_members_by_lane.pop(lane)
            if lane in segment_members_by_lane else ())
        if not members:
            return
        segment_index = (
            segment_counts_by_lane[lane]
            if lane in segment_counts_by_lane else 0)
        segment_counts_by_lane[lane] = segment_index + 1
        segments.append(RuntimeGraphSegmentDescriptor(
            f"graph-segment-{segment_index}",
            lane[0],
            lane[1],
            tuple(members),
            sync_before_replay,
        ))

    for logical_key, _, _, _ in logical_schedule:
        call = calls_by_id[logical_key[0]]
        for lane in call_host_lanes(call):
            if runtime_graph_mode == "PIECEWISE" and (
                    normalize(call.call_id) in eager_call_ids):
                flush_lane(lane)
            else:
                if lane not in segment_members_by_lane:
                    segment_members_by_lane[lane] = []
                segment_members_by_lane[lane].append(logical_key)
    for lane in sorted(segment_members_by_lane):
        flush_lane(lane)
    return tuple(sorted(
        segments,
        key=lambda segment: (
            segment.pool_name,
            segment.rank,
            int(segment.segment_id.rsplit("-", 1)[1]),
        ),
    ))


class _SummarySchedulePlan(NamedTuple):
    """Integer-indexed static form of one plan structure's summary schedule.

    The summary recurrence visits the same occurrences, in the same order,
    reserving the same resources and reading the same predecessors on every
    step; only the durations change.  Everything that does not change is
    resolved here once per structure into dense indices, so the per-step
    recurrence indexes lists instead of re-hashing an identifier string or an
    occurrence tuple several times per occurrence.
    """

    occurrence_count: int
    call_index: tuple
    dependency_slots: tuple
    resource_slots: tuple
    resource_ids_by_call: tuple
    resource_count: int
    is_transfer: tuple
    occurrence_counts: tuple
    repeat_slot: tuple
    repeat_first: tuple
    repeat_last: tuple
    digest_slot: tuple
    digest_groups: tuple
    digest_count: int
    logical_keys: tuple


_SUMMARY_SCHEDULE_PLAN_CACHE_LIMIT = 16
_SUMMARY_SCHEDULE_PLANS: OrderedDict = OrderedDict()
_STATIC_HOST_TIMELINE_CACHE_LIMIT = 16
_STATIC_HOST_TIMELINES: OrderedDict = OrderedDict()


def _summary_schedule_plan(
    context,
    norm_call_ids,
    reserved_resources_by_norm_call,
    transfer_by_norm_call,
    norm_repeats,
):
    """Resolve the structure-only summary schedule into dense indices."""

    call_index_by_id = {
        call_id: index for index, call_id in enumerate(norm_call_ids)}
    resource_index = {}
    resource_slots_by_call = []
    resource_ids_by_call = []
    for call_id in norm_call_ids:
        resource_ids = reserved_resources_by_norm_call[call_id]
        slots = []
        for resource_id in resource_ids:
            if resource_id not in resource_index:
                resource_index[resource_id] = len(resource_index)
            slots.append(resource_index[resource_id])
        resource_slots_by_call.append(tuple(slots))
        resource_ids_by_call.append(resource_ids)

    repeat_position_by_call = {}
    repeat_count_by_call = {}
    digest_index_by_call = {}
    digest_groups = []
    digest_count = 0
    for position, (repeat_call_ids, repeat_count) in enumerate(norm_repeats):
        group = []
        for call_id in repeat_call_ids:
            repeat_position_by_call[call_id] = position
            repeat_count_by_call[call_id] = repeat_count
            digest_index_by_call[call_id] = digest_count
            group.append(digest_count)
            digest_count += 1
        digest_groups.append(tuple(group))

    slot_by_logical_key = {}
    call_index = []
    dependency_slots = []
    resource_slots = []
    is_transfer = []
    repeat_slot = []
    repeat_first = []
    repeat_last = []
    digest_slot = []
    logical_keys = []
    occurrence_counts = [0] * len(norm_call_ids)
    for slot, entry in enumerate(context.logical_schedule):
        logical_key, dependency_keys, _position, _successors = entry
        call_id, occurrence_index = logical_key
        index = call_index_by_id[call_id]
        slots = []
        for dependency_key in dependency_keys:
            if dependency_key not in slot_by_logical_key:
                raise ValueError(
                    "logical kernel occurrence has unscheduled dependencies: "
                    f"{(dependency_key,)!r}")
            slots.append(slot_by_logical_key[dependency_key])
        slot_by_logical_key[logical_key] = slot
        call_index.append(index)
        dependency_slots.append(tuple(slots))
        resource_slots.append(resource_slots_by_call[index])
        is_transfer.append(transfer_by_norm_call[call_id])
        logical_keys.append(logical_key)
        occurrence_counts[index] += 1
        if occurrence_index < 0:
            repeat_slot.append(-1)
            repeat_first.append(False)
            repeat_last.append(False)
            digest_slot.append(-1)
            continue
        repeat_slot.append(repeat_position_by_call[call_id])
        repeat_first.append(occurrence_index == 0)
        repeat_last.append(
            occurrence_index == repeat_count_by_call[call_id] - 1)
        digest_slot.append(digest_index_by_call[call_id])
    return _SummarySchedulePlan(
        occurrence_count=len(call_index),
        call_index=tuple(call_index),
        dependency_slots=tuple(dependency_slots),
        resource_slots=tuple(resource_slots),
        resource_ids_by_call=tuple(resource_ids_by_call),
        resource_count=len(resource_index),
        is_transfer=tuple(is_transfer),
        occurrence_counts=tuple(occurrence_counts),
        repeat_slot=tuple(repeat_slot),
        repeat_first=tuple(repeat_first),
        repeat_last=tuple(repeat_last),
        digest_slot=tuple(digest_slot),
        digest_groups=tuple(digest_groups),
        digest_count=digest_count,
        logical_keys=tuple(logical_keys),
    )


def _cached_summary_schedule_plan(
    structure_key,
    context,
    norm_call_ids,
    reserved_resources_by_norm_call,
    transfer_by_norm_call,
    norm_repeats,
):
    """Reuse the dense summary schedule for one plan structure."""

    cached = (
        _SUMMARY_SCHEDULE_PLANS[structure_key]
        if structure_key in _SUMMARY_SCHEDULE_PLANS else None)
    if cached is not None:
        _SUMMARY_SCHEDULE_PLANS.move_to_end(structure_key)
        return cached
    schedule_plan = _summary_schedule_plan(
        context,
        norm_call_ids,
        reserved_resources_by_norm_call,
        transfer_by_norm_call,
        norm_repeats,
    )
    _cache_put(
        _SUMMARY_SCHEDULE_PLANS,
        _SUMMARY_SCHEDULE_PLAN_CACHE_LIMIT,
        structure_key,
        schedule_plan,
    )
    return schedule_plan


class _StaticHostTimeline(NamedTuple):
    """Host launch timeline of a plan structure that no device state gates."""

    ready_time_s: tuple
    launch_cpu_time_s: float
    launch_critical_path_s: float
    eager_dispatch_count: int
    graph_replay_count: int
    launch_kinds: tuple


def _static_host_timeline(
    schedule_plan,
    host_lane_program,
    hardware_by_call_index,
    eager_service_s_by_hardware,
    graph_replay_service_s_by_hardware,
):
    """Resolve the Host launch timeline when no replay waits on the device.

    Every Host service time is a constant of the launch profile or of the
    structure-only prepare mapping, and each lane simply queues them in
    schedule order.  The Host timeline of such a structure is therefore the
    same on every step, and only its per-occurrence completion enters the
    device recurrence.  ``None`` means a segment synchronizes on the device
    before replay, which couples the two and keeps the per-step walk.
    """

    host_ready_by_lane = {}
    host_service_by_lane = {}
    graph_ready_by_identity = {}
    launch_cpu_time_s = 0.0
    eager_dispatch_count = 0
    graph_replay_count = 0
    launch_kinds = set()
    ready_time_s = []
    for slot, logical_key in enumerate(schedule_plan.logical_keys):
        hardware_id = hardware_by_call_index[schedule_plan.call_index[slot]]
        host_ends = []
        for lane, segment, service_times in host_lane_program[logical_key]:
            if service_times is not None:
                for service_s in service_times:
                    host_start_s = (
                        host_ready_by_lane[lane]
                        if lane in host_ready_by_lane else 0.0)
                    host_end_s = host_start_s + service_s
                    host_ready_by_lane[lane] = host_end_s
                    host_service_by_lane[lane] = (
                        host_service_by_lane[lane] + service_s
                        if lane in host_service_by_lane else service_s)
                    launch_cpu_time_s += service_s
                launch_kinds.add("runtime_prepare")
            if segment is None:
                service_s = eager_service_s_by_hardware[hardware_id]
                host_start_s = (
                    host_ready_by_lane[lane]
                    if lane in host_ready_by_lane else 0.0)
                host_end_s = host_start_s + service_s
                host_ready_by_lane[lane] = host_end_s
                host_service_by_lane[lane] = (
                    host_service_by_lane[lane] + service_s
                    if lane in host_service_by_lane else service_s)
                launch_cpu_time_s += service_s
                eager_dispatch_count += 1
                host_ends.append(host_end_s)
                launch_kinds.add("eager")
                continue
            if segment.sync_before_replay:
                return None
            identity = (lane[0], lane[1], segment.segment_id)
            if identity in graph_ready_by_identity:
                graph_end_s = graph_ready_by_identity[identity]
            else:
                service_s = graph_replay_service_s_by_hardware[hardware_id]
                host_start_s = (
                    host_ready_by_lane[lane]
                    if lane in host_ready_by_lane else 0.0)
                host_end_s = host_start_s + service_s
                host_ready_by_lane[lane] = host_end_s
                host_service_by_lane[lane] = (
                    host_service_by_lane[lane] + service_s
                    if lane in host_service_by_lane else service_s)
                graph_ready_by_identity[identity] = host_end_s
                launch_cpu_time_s += service_s
                graph_replay_count += 1
                launch_kinds.add("graph_replay")
                graph_end_s = host_end_s
            host_ends.append(graph_end_s)
        ready_time_s.append(
            max(host_ends) if host_ends else -math.inf)
    return _StaticHostTimeline(
        ready_time_s=tuple(ready_time_s),
        launch_cpu_time_s=launch_cpu_time_s,
        launch_critical_path_s=max(host_service_by_lane.values()),
        eager_dispatch_count=eager_dispatch_count,
        graph_replay_count=graph_replay_count,
        launch_kinds=tuple(sorted(launch_kinds)),
    )


def _cached_static_host_timeline(
    structure_key,
    schedule_plan,
    host_lane_program,
    hardware_by_call_index,
    host_launch_profiles,
    eager_service_s_by_hardware,
    graph_replay_service_s_by_hardware,
):
    """Reuse the structure-only Host launch timeline for identical inputs."""

    cache_key = (
        structure_key,
        id(host_lane_program),
        id(host_launch_profiles),
    )
    cached = (
        _STATIC_HOST_TIMELINES[cache_key]
        if cache_key in _STATIC_HOST_TIMELINES else None)
    if (cached is not None
            and cached[0] is host_lane_program
            and cached[1] is host_launch_profiles):
        _STATIC_HOST_TIMELINES.move_to_end(cache_key)
        return cached[2]
    timeline = _static_host_timeline(
        schedule_plan,
        host_lane_program,
        hardware_by_call_index,
        eager_service_s_by_hardware,
        graph_replay_service_s_by_hardware,
    )
    _cache_put(
        _STATIC_HOST_TIMELINES,
        _STATIC_HOST_TIMELINE_CACHE_LIMIT,
        cache_key,
        (host_lane_program, host_launch_profiles, timeline),
    )
    return timeline


_LAUNCH_GAP_CACHE_LIMIT = 16
_LAUNCH_GAPS: OrderedDict = OrderedDict()


def _cached_launch_gap_by_slot(
    structure_key,
    schedule_plan,
    host_lane_program,
    hardware_by_call_index,
    host_launch_profiles,
):
    """Reuse the structure-only per-slot device launch gaps."""

    cache_key = (
        structure_key,
        id(host_lane_program),
        id(host_launch_profiles),
    )
    cached = (
        _LAUNCH_GAPS[cache_key] if cache_key in _LAUNCH_GAPS else None)
    if (cached is not None
            and cached[0] is host_lane_program
            and cached[1] is host_launch_profiles):
        _LAUNCH_GAPS.move_to_end(cache_key)
        return cached[2]
    tracker = _DeviceLaunchGapTracker(host_launch_profiles)
    by_slot = tuple(
        tracker.observe(
            hardware_by_call_index[schedule_plan.call_index[slot]],
            tuple(
                (lane, segment)
                for lane, segment, _ in host_lane_program[logical_key]),
        )
        for slot, logical_key in enumerate(schedule_plan.logical_keys)
    )
    _cache_put(
        _LAUNCH_GAPS,
        _LAUNCH_GAP_CACHE_LIMIT,
        cache_key,
        (host_lane_program, host_launch_profiles, by_slot),
    )
    return by_slot


def _build_summary_execution_timeline(
    plan,
    predictions,
    calls,
    calls_by_id,
    origin_time_s,
    host_launch_profiles,
    runtime_graph_mode,
    host_lane_program,
    schedule_plan,
    static_host_timeline,
    hardware_by_call_index,
    eager_service_s_by_hardware,
    graph_replay_service_s_by_hardware,
    shared_resource_calendar,
    launch_gap_by_slot,
):
    """Schedule exactly while retaining only report-visible timing state."""

    occurrence_count = schedule_plan.occurrence_count
    call_index = schedule_plan.call_index
    dependency_slots = schedule_plan.dependency_slots
    resource_slots = schedule_plan.resource_slots
    is_transfer = schedule_plan.is_transfer
    repeat_slot = schedule_plan.repeat_slot
    repeat_first = schedule_plan.repeat_first
    repeat_last = schedule_plan.repeat_last
    digest_slot = schedule_plan.digest_slot
    logical_keys = schedule_plan.logical_keys
    durations = [
        predictions[call.call_id].t_total_s for call in calls]
    resource_end_time_s = [origin_time_s] * schedule_plan.resource_count
    end_by_slot = [0.0] * occurrence_count
    host_timing = host_launch_profiles is not None
    if host_timing:
        device_resource_end_time_s = (
            [origin_time_s] * schedule_plan.resource_count)
        device_end_by_slot = [0.0] * occurrence_count
    maximum_end_time_s = origin_time_s
    maximum_device_end_time_s = origin_time_s
    repeat_bounds = [[None, None] for _ in plan.block_repeats]
    # One occurrence contributes 16 bytes to its call's interval digest.  The
    # digest is a stream, so collecting a step's contributions and feeding
    # them in one update is the same hash as feeding each occurrence alone.
    interval_records = [[] for _ in range(schedule_plan.digest_count)]
    pack_interval = _INTERVAL_PAIR.pack
    host_ready_by_lane = {}
    host_service_by_lane = {}
    launch_cpu_time_s = 0.0
    eager_dispatch_count = 0
    graph_replay_count = 0
    launch_kinds = set()
    graph_ready_by_identity = {}
    static_host_ready_time_s = (
        None if static_host_timeline is None
        else static_host_timeline.ready_time_s)

    for slot in range(occurrence_count):
        index = call_index[slot]
        duration_s = durations[index]
        resources = resource_slots[slot]
        dependencies = dependency_slots[slot]
        ready_time_s = resource_end_time_s[resources[0]]
        for resource_id in resources[1:]:
            end_s = resource_end_time_s[resource_id]
            if end_s > ready_time_s:
                ready_time_s = end_s
        for dependency_slot in dependencies:
            end_s = end_by_slot[dependency_slot]
            if end_s > ready_time_s:
                ready_time_s = end_s
        if host_timing:
            launch_gap_s = launch_gap_by_slot[slot]
            ready_time_s += launch_gap_s
            device_ready_time_s = device_resource_end_time_s[resources[0]]
            for resource_id in resources[1:]:
                end_s = device_resource_end_time_s[resource_id]
                if end_s > device_ready_time_s:
                    device_ready_time_s = end_s
            for dependency_slot in dependencies:
                end_s = device_end_by_slot[dependency_slot]
                if end_s > device_ready_time_s:
                    device_ready_time_s = end_s
            device_ready_time_s += launch_gap_s
            device_end_time_s = device_ready_time_s + duration_s
            device_end_by_slot[slot] = device_end_time_s
            if device_end_time_s > maximum_device_end_time_s:
                maximum_device_end_time_s = device_end_time_s
            for resource_id in resources:
                device_resource_end_time_s[resource_id] = device_end_time_s
            if static_host_ready_time_s is not None:
                host_end_s = static_host_ready_time_s[slot]
                if host_end_s > ready_time_s:
                    ready_time_s = host_end_s
            else:
                hardware_id = hardware_by_call_index[index]
                for lane, segment, service_times in host_lane_program[
                        logical_keys[slot]]:
                    if service_times is not None:
                        for service_s in service_times:
                            host_start_s = (
                                host_ready_by_lane[lane]
                                if lane in host_ready_by_lane
                                else origin_time_s)
                            host_end_s = host_start_s + service_s
                            host_ready_by_lane[lane] = host_end_s
                            host_service_by_lane[lane] = (
                                host_service_by_lane[lane] + service_s
                                if lane in host_service_by_lane else service_s)
                            launch_cpu_time_s += service_s
                        launch_kinds.add("runtime_prepare")
                    if segment is None:
                        service_s = eager_service_s_by_hardware[hardware_id]
                        host_start_s = (
                            host_ready_by_lane[lane]
                            if lane in host_ready_by_lane else origin_time_s)
                        host_end_s = host_start_s + service_s
                        host_ready_by_lane[lane] = host_end_s
                        host_service_by_lane[lane] = (
                            host_service_by_lane[lane] + service_s
                            if lane in host_service_by_lane else service_s)
                        launch_cpu_time_s += service_s
                        eager_dispatch_count += 1
                        launch_kinds.add("eager")
                        if host_end_s > ready_time_s:
                            ready_time_s = host_end_s
                        continue
                    identity = (lane[0], lane[1], segment.segment_id)
                    if identity in graph_ready_by_identity:
                        graph_end_s = graph_ready_by_identity[identity]
                    else:
                        service_s = graph_replay_service_s_by_hardware[
                            hardware_id]
                        host_start_s = (
                            host_ready_by_lane[lane]
                            if lane in host_ready_by_lane else origin_time_s)
                        if segment.sync_before_replay:
                            # ACLGraphWrapper calls
                            # current_stream().synchronize() before replay.
                            # This is a dependency barrier, not a serving-
                            # duration correction: wait until the device
                            # predecessor is complete, then pay only the
                            # measured replay enqueue service below.
                            host_start_s = max(
                                host_start_s, device_ready_time_s)
                        host_end_s = host_start_s + service_s
                        host_ready_by_lane[lane] = host_end_s
                        host_service_by_lane[lane] = (
                            host_service_by_lane[lane] + service_s
                            if lane in host_service_by_lane else service_s)
                        graph_ready_by_identity[identity] = host_end_s
                        launch_cpu_time_s += service_s
                        graph_replay_count += 1
                        launch_kinds.add("graph_replay")
                        graph_end_s = host_end_s
                    if graph_end_s > ready_time_s:
                        ready_time_s = graph_end_s
        if shared_resource_calendar is not None and is_transfer[slot]:
            start_time_s, end_time_s = shared_resource_calendar.reserve_many(
                schedule_plan.resource_ids_by_call[index],
                ready_time_s,
                duration_s,
            )
        else:
            start_time_s = ready_time_s
            end_time_s = start_time_s + duration_s
        end_by_slot[slot] = end_time_s
        if end_time_s > maximum_end_time_s:
            maximum_end_time_s = end_time_s
        for resource_id in resources:
            resource_end_time_s[resource_id] = end_time_s
        digest_index = digest_slot[slot]
        if digest_index < 0:
            continue
        bounds = repeat_bounds[repeat_slot[slot]]
        start_interval = start_time_s - origin_time_s
        end_interval = end_time_s - origin_time_s
        if repeat_first[slot]:
            bounds[0] = (
                start_interval if bounds[0] is None
                else min(bounds[0], start_interval))
        if repeat_last[slot]:
            bounds[1] = (
                end_interval if bounds[1] is None
                else max(bounds[1], end_interval))
        interval_records[digest_index].append(
            pack_interval(start_interval, end_interval))

    block_repeats = []
    for position, repeat in enumerate(plan.block_repeats):
        first_start_time_s, last_end_time_s = repeat_bounds[position]
        if first_start_time_s is None or last_end_time_s is None:
            raise RuntimeError("kernel block repeat did not establish bounds")
        block_repeats.append(ExecutionBlockRepeat(
            repeat.block_id,
            repeat.repeat_count,
            first_start_time_s,
            last_end_time_s,
            (),
            tuple(
                (
                    call_id,
                    hashlib.sha256(
                        b"".join(interval_records[digest_index])).hexdigest(),
                )
                for call_id, digest_index in zip(
                    repeat.call_ids, schedule_plan.digest_groups[position])
            ),
        ))
    duration_s = maximum_end_time_s - origin_time_s
    if not host_timing:
        predicted_device_time_s = duration_s
        launch_critical_path_s = 0.0
        host_device_overlap_s = 0.0
    else:
        predicted_device_time_s = maximum_device_end_time_s - origin_time_s
        if static_host_timeline is None:
            launch_critical_path_s = max(host_service_by_lane.values())
        else:
            launch_cpu_time_s = static_host_timeline.launch_cpu_time_s
            launch_critical_path_s = (
                static_host_timeline.launch_critical_path_s)
            eager_dispatch_count = static_host_timeline.eager_dispatch_count
            graph_replay_count = static_host_timeline.graph_replay_count
            launch_kinds = set(static_host_timeline.launch_kinds)
        host_device_overlap_s = max(
            0.0,
            predicted_device_time_s + launch_critical_path_s - duration_s,
        )
    return ExecutionTimeline(
        (),
        duration_s,
        tuple(block_repeats),
        (),
        (),
        tuple(
            (call.call_id, count)
            for call, count in zip(calls, schedule_plan.occurrence_counts)
        ),
        (),
        0,
        False,
        predicted_device_time_s,
        launch_cpu_time_s,
        launch_critical_path_s,
        host_device_overlap_s,
        eager_dispatch_count,
        graph_replay_count,
        tuple(sorted(launch_kinds)),
        (),
        runtime_graph_mode,
    )


def build_execution_timeline(
    plan: KernelPlan,
    predictions: Mapping[str, KernelPrediction],
    *,
    origin_time_s: float = 0.0,
    shared_resource_calendar: ExecutionResourceCalendar | None = None,
    host_launch_profiles: Mapping[str, HostLaunchProfile] | None = None,
    host_runtime_prepare_profiles: Mapping[
        str, HostRuntimePrepareProfile] | None = None,
    runtime_prepare_events: Mapping[
        tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]] | None = None,
    runtime_graph_mode: str | None = None,
    runtime_graph_segments: Tuple[RuntimeGraphSegmentDescriptor, ...] | None = None,
    materialize_timeline_details: bool = True,
    normalized_runtime_prepare_events: Mapping[
        tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]] | None = None,
    runtime_prepare_events_token: object | None = None,
    plan_structure_runtime_graph_segments: object | None = None,
) -> ExecutionTimeline:
    if not isinstance(materialize_timeline_details, bool):
        raise TypeError("timeline detail materialization flag must be boolean")
    if plan_structure_runtime_graph_segments is not None:
        if runtime_graph_segments is not None:
            raise ValueError(
                "plan-structure runtime Graph segments conflict with "
                "plan-space runtime Graph segments")
        if materialize_timeline_details:
            raise ValueError(
                "plan-structure runtime Graph segments require the summary "
                "execution timeline")
        if (not isinstance(plan_structure_runtime_graph_segments, tuple)
                or len(plan_structure_runtime_graph_segments) != 2):
            raise TypeError(
                "plan-structure runtime Graph segments must be a "
                "(segments, prefix) pair")
    if (normalized_runtime_prepare_events is None) != (
            runtime_prepare_events_token is None):
        raise ValueError(
            "normalized runtime prepare events require their cache token")
    if normalized_runtime_prepare_events is not None:
        if runtime_prepare_events is not None:
            raise ValueError(
                "explicit runtime prepare events conflict with normalized "
                "runtime prepare events")
        if materialize_timeline_details:
            raise ValueError(
                "normalized runtime prepare events require the summary "
                "execution timeline")
    if set(predictions) != {call.call_id for call in plan.calls}:
        raise ValueError("predictions must cover every KernelPlan call exactly once")
    if (isinstance(origin_time_s, bool) or not isinstance(origin_time_s, Real)
            or not math.isfinite(origin_time_s) or origin_time_s < 0):
        raise ValueError("execution timeline origin must be finite and non-negative")
    origin_time_s = float(origin_time_s)
    if host_launch_profiles is None:
        if runtime_graph_mode is not None:
            raise ValueError(
                "runtime_graph_mode requires Host launch profiles")
        if runtime_graph_segments is not None:
            raise ValueError(
                "runtime_graph_segments require Host launch profiles")
        if plan_structure_runtime_graph_segments is not None:
            raise ValueError(
                "runtime_graph_segments require Host launch profiles")
    else:
        if not isinstance(host_launch_profiles, Mapping) or not host_launch_profiles:
            raise TypeError("Host launch profiles must be a non-empty mapping")
        if runtime_graph_mode is None:
            raise ValueError(
                "Host launch timing requires an explicit runtime Graph mode")
        hardware_ids = {call.kernel_spec.hardware_id for call in plan.calls}
        if set(host_launch_profiles) != hardware_ids:
            raise ValueError(
                "Host launch profiles must cover KernelPlan hardware exactly: "
                f"required={sorted(hardware_ids)}, "
                f"supplied={sorted(host_launch_profiles)}")
        for hardware_id, profile in host_launch_profiles.items():
            if not isinstance(profile, HostLaunchProfile):
                raise TypeError("Host launch profile mapping contains a non-profile")
            if profile.hardware_id != hardware_id:
                raise ValueError("Host launch profile key disagrees with hardware_id")
    if host_launch_profiles is None and (
            host_runtime_prepare_profiles is not None
            or runtime_prepare_events is not None
            or normalized_runtime_prepare_events is not None):
        raise ValueError(
            "runtime prepare timing requires Host launch profiles")
    calls = plan.topological_order()
    calls_by_id = {call.call_id: call for call in calls}
    if not materialize_timeline_details:
        # Per-step plans are structurally identical except for the embedded
        # schedule-step ID prefix, so all structure-only artifacts are
        # computed once per distinct normalized structure.  The float
        # recurrence itself still runs per step inside the summary builder.
        prefix = _common_id_prefix(plan)
        prefix_length = len(prefix)
        structure_key, schedule_context = _plan_schedule_context(
            plan, calls, prefix)
        norm_calls_by_id = {
            call.call_id[prefix_length:]: call for call in calls}
        norm_repeat_by_call = {}
        for repeat in plan.block_repeats:
            for call_id in repeat.call_ids:
                norm_repeat_by_call[call_id[prefix_length:]] = repeat
        if host_launch_profiles is not None:
            if plan_structure_runtime_graph_segments is not None:
                # The producer already emitted this plan's segments in the
                # normalized identifier space this builder works in, against
                # the same common prefix.  Translating them into plan space
                # only to strip the prefix again would rebuild every
                # descriptor once per step for an identical result.
                (
                    plan_structure_segments, plan_structure_prefix,
                ) = plan_structure_runtime_graph_segments
                if plan_structure_prefix != prefix:
                    raise ValueError(
                        "plan-structure runtime Graph segments were "
                        "normalized against a different plan prefix")
                if not isinstance(plan_structure_segments, tuple):
                    raise TypeError(
                        "runtime Graph segments must be a sequence")
                normalized_graph_segments = plan_structure_segments
            elif runtime_graph_segments is None:
                normalized_graph_segments = ()
            elif not isinstance(runtime_graph_segments, (list, tuple)):
                raise TypeError("runtime Graph segments must be a sequence")
            else:
                # The normalized descriptor tuple is canonical per plan
                # structure; reusing the cached object keeps the downstream
                # mapping-cache lookup hash-free once validated.
                segments_cache_key = (structure_key, runtime_graph_mode)
                cached_segments = (
                    _NORMALIZED_GRAPH_SEGMENTS[segments_cache_key]
                    if segments_cache_key in _NORMALIZED_GRAPH_SEGMENTS
                    else None)
                if cached_segments is not None and _segments_match_normalized(
                        cached_segments, runtime_graph_segments, prefix):
                    _NORMALIZED_GRAPH_SEGMENTS.move_to_end(segments_cache_key)
                    normalized_graph_segments = cached_segments
                else:
                    normalized_graph_segments = _translate_segment_ids(
                        tuple(runtime_graph_segments),
                        lambda call_id: _normalize_id(call_id, prefix))
                    if cached_segments is None:
                        _cache_put(
                            _NORMALIZED_GRAPH_SEGMENTS,
                            _NORMALIZED_GRAPH_SEGMENTS_CACHE_LIMIT,
                            segments_cache_key,
                            normalized_graph_segments,
                        )
            graph_segment_by_lane_occurrence, normalized_graph_segments = (
                _cached_graph_segment_mapping(
                    structure_key,
                    schedule_context,
                    norm_calls_by_id,
                    runtime_graph_mode,
                    normalized_graph_segments,
                )
            )
        else:
            graph_segment_by_lane_occurrence = {}
            normalized_graph_segments = ()
        if normalized_runtime_prepare_events is not None:
            norm_prepare_events = normalized_runtime_prepare_events
            prepare_events_token = runtime_prepare_events_token
        else:
            norm_prepare_events = (
                None
                if runtime_prepare_events is None
                else {
                    (pool_name, rank, _normalize_id(call_id, prefix),
                     occurrence): mechanisms
                    for (pool_name, rank, call_id, occurrence), mechanisms
                    in runtime_prepare_events.items()
                }
            )
            prepare_events_token = runtime_prepare_events
        if host_launch_profiles is not None:
            runtime_prepare_by_lane_occurrence, prepare_services = (
                _cached_runtime_prepare_mapping(
                    structure_key,
                    schedule_context,
                    plan,
                    norm_calls_by_id,
                    runtime_graph_mode,
                    graph_segment_by_lane_occurrence,
                    host_runtime_prepare_profiles,
                    norm_prepare_events,
                    prepare_events_token,
                )
            )
        norm_call_ids = tuple(
            call.call_id[prefix_length:] for call in calls)
        schedule_plan = _cached_summary_schedule_plan(
            structure_key,
            schedule_context,
            norm_call_ids,
            {
                call.call_id[prefix_length:]: call_reserved_resource_ids(call)
                for call in calls
            },
            {
                call.call_id[prefix_length:]:
                    call.kernel_spec.transfer is not None
                for call in calls
            },
            tuple(
                (
                    tuple(
                        call_id[prefix_length:]
                        for call_id in repeat.call_ids),
                    repeat.repeat_count,
                )
                for repeat in plan.block_repeats
            ),
        )
        hardware_by_call_index = tuple(
            call.kernel_spec.hardware_id for call in calls)
        # A device-only summary never reads a Host program. Constructing it
        # anyway retained another whole graph for every decode structure.
        host_lane_program = None
        eager_service_s_by_hardware = None
        graph_replay_service_s_by_hardware = None
        static_host_timeline = None
        launch_gap_by_slot = None
        if host_launch_profiles is not None:
            host_lane_program = _cached_host_lane_program(
                structure_key,
                schedule_context,
                graph_segment_by_lane_occurrence,
                prepare_services,
                normalized_graph_segments,
            )
            launch_gap_by_slot = _cached_launch_gap_by_slot(
                structure_key,
                schedule_plan,
                host_lane_program,
                hardware_by_call_index,
                host_launch_profiles,
            )
            eager_service_s_by_hardware = {
                hardware_id: profile.service_time_s("eager")
                for hardware_id, profile in host_launch_profiles.items()
            }
            graph_replay_service_s_by_hardware = {
                hardware_id: profile.service_time_s("graph_replay")
                for hardware_id, profile in host_launch_profiles.items()
            }
            if origin_time_s == 0.0:
                # The Host timeline is measured from the step origin, so it
                # only detaches from the device recurrence when that origin is
                # the zero the structure-only walk resolves it against.
                static_host_timeline = _cached_static_host_timeline(
                    structure_key,
                    schedule_plan,
                    host_lane_program,
                    hardware_by_call_index,
                    host_launch_profiles,
                    eager_service_s_by_hardware,
                    graph_replay_service_s_by_hardware,
                )
        return _build_summary_execution_timeline(
            plan,
            predictions,
            calls,
            norm_calls_by_id,
            origin_time_s,
            host_launch_profiles,
            runtime_graph_mode,
            host_lane_program,
            schedule_plan,
            static_host_timeline,
            hardware_by_call_index,
            eager_service_s_by_hardware,
            graph_replay_service_s_by_hardware,
            shared_resource_calendar,
            launch_gap_by_slot,
        )
    (
        calls,
        calls_by_id,
        repeat_by_call,
        repeat_call_ids_by_block,
        feedback_by_destination_by_block,
    ) = _plan_schedule_inputs(plan)
    raw_occurrence_indices, raw_lane_occurrence_keys = (
        _occurrence_context_data(plan, repeat_by_call))
    raw_context = _PlanScheduleContext(
        (),
        raw_occurrence_indices,
        raw_lane_occurrence_keys,
        {call.call_id: call_host_lanes(call) for call in plan.calls},
    )
    graph_segment_by_lane_occurrence, normalized_graph_segments = (
        _runtime_graph_segment_mapping(
            raw_context,
            calls_by_id,
            runtime_graph_mode,
            runtime_graph_segments,
        )
        if host_launch_profiles is not None else ({}, ())
    )
    runtime_prepare_by_lane_occurrence = _runtime_prepare_mapping(
        raw_context,
        plan,
        calls_by_id,
        runtime_graph_mode,
        graph_segment_by_lane_occurrence,
        host_runtime_prepare_profiles,
        runtime_prepare_events,
    )
    logical_schedule = _logical_schedule(
        plan,
        calls,
        calls_by_id,
        repeat_by_call,
        repeat_call_ids_by_block,
        feedback_by_destination_by_block,
    )
    launch_gap_tracker = (
        _DeviceLaunchGapTracker(host_launch_profiles)
        if host_launch_profiles is not None else None)
    reserved_resources_by_call = {
        call.call_id: call_reserved_resource_ids(call)
        for call in plan.calls
    }
    resource_predecessor = {
        resource_id: None
        for resources in reserved_resources_by_call.values()
        for resource_id in resources
    }
    device_resource_predecessor = {
        resource_id: None
        for resources in reserved_resources_by_call.values()
        for resource_id in resources
    }
    interval_times_by_nonrepeated_call = {}
    repeated_interval_times = (
        {
            repeat.block_id: {
                call_id: [] for call_id in repeat.call_ids
            }
            for repeat in plan.block_repeats
        }
        if materialize_timeline_details else None
    )
    repeated_interval_digests = (
        None if materialize_timeline_details else {
            repeat.block_id: {
                call_id: hashlib.sha256()
                for call_id in repeat.call_ids
            }
            for repeat in plan.block_repeats
        }
    )
    latest_repeated_interval_times = {
        repeat.block_id: {} for repeat in plan.block_repeats
    }
    repeat_bounds = {
        repeat.block_id: [None, None] for repeat in plan.block_repeats
    }
    pool_activity_records = {}
    reserved_resource_activity_records = {}
    call_occurrence_counts = {
        call.call_id: 0 for call in calls
    }
    state_by_key = {}
    device_end_by_key = {}
    remaining_dependency_uses = {}
    terminal_candidates = []
    maximum_end_time_s = origin_time_s
    maximum_device_end_time_s = origin_time_s
    host_ready_by_lane = {}
    host_service_by_lane = {}
    eager_dispatch_count = 0
    graph_replay_count = 0
    launch_cpu_time_s = 0.0
    launch_kinds = set()
    host_intervals = []
    last_host_event_by_lane = {}
    graph_ready_by_identity = {}
    graph_segment_by_identity = {
        (segment.pool_name, segment.rank, segment.segment_id): segment
        for segment in normalized_graph_segments
    }
    graph_segment_traversal = _GraphSegmentTraversal(
        normalized_graph_segments)

    def record_activity(records, activity_id, start_time_s, end_time_s):
        if activity_id not in records:
            records[activity_id] = {
                "interval_count": 0,
                "intervals": [],
            }
        record = records[activity_id]
        record["interval_count"] += 1
        record["intervals"].append((start_time_s, end_time_s))

    def merged_activity_intervals(raw_intervals):
        intervals = sorted(raw_intervals)
        if not intervals:
            return ()
        merged = [intervals[0]]
        for start_time_s, end_time_s in intervals[1:]:
            previous_start, previous_end = merged[-1]
            if previous_end < start_time_s:
                merged.append((start_time_s, end_time_s))
            else:
                merged[-1] = (
                    previous_start,
                    max(previous_end, end_time_s),
                )
        return tuple(merged)

    def schedule_call(
        call,
        logical_key,
        position,
        dependency_states,
        device_dependency_ends,
    ):
        nonlocal eager_dispatch_count, graph_replay_count, launch_cpu_time_s
        nonlocal maximum_end_time_s
        nonlocal maximum_device_end_time_s
        reserved_resource_ids = reserved_resources_by_call[call.call_id]
        duration_s = predictions[call.call_id].t_total_s
        if host_launch_profiles is not None:
            launch_gap_s = launch_gap_tracker.observe(
                call.kernel_spec.hardware_id,
                tuple(
                    (
                        lane,
                        graph_segment_by_lane_occurrence[
                            (*lane, *logical_key)]
                        if (*lane, *logical_key)
                        in graph_segment_by_lane_occurrence else None,
                    )
                    for lane in call_host_lanes(call)
                ),
            )
            device_predecessor_ends = list(device_dependency_ends)
            device_predecessor_ends.extend(
                predecessor_end
                for resource_id in reserved_resource_ids
                if (predecessor_end := device_resource_predecessor[
                    resource_id]) is not None
            )
            device_ready_time_s = (
                max(device_predecessor_ends)
                if device_predecessor_ends else origin_time_s
            ) + launch_gap_s
            device_end_time_s = device_ready_time_s + duration_s
            device_end_by_key[logical_key] = device_end_time_s
            maximum_device_end_time_s = max(
                maximum_device_end_time_s, device_end_time_s)
            for resource_id in reserved_resource_ids:
                device_resource_predecessor[resource_id] = device_end_time_s
        predecessor_causes = {}
        predecessor_states = {}
        for dependency_state in dependency_states:
            dependency_key = dependency_state["logical_key"]
            predecessor_causes[dependency_key] = "dependency"
            predecessor_states[dependency_key] = dependency_state
        for resource_id in reserved_resource_ids:
            predecessor = resource_predecessor[resource_id]
            if predecessor is None:
                continue
            predecessor_key = predecessor["logical_key"]
            predecessor_states[predecessor_key] = predecessor
            if predecessor_key in predecessor_causes:
                predecessor_causes[predecessor_key] = "dependency_and_resource"
            else:
                predecessor_causes[predecessor_key] = "resource"
        if predecessor_causes:
            ready_time_s = max(
                predecessor_states[predecessor]["end_time_s"]
                for predecessor in predecessor_causes
            )
        else:
            ready_time_s = origin_time_s
        if host_launch_profiles is None:
            launch_gap_s = 0.0
        else:
            ready_time_s += launch_gap_s
            profile = host_launch_profiles[call.kernel_spec.hardware_id]
            host_ends = []
            ordered_device_predecessors = tuple(
                predecessor_states[predecessor]["logical_key"]
                for predecessor in sorted(
                    predecessor_states,
                    key=lambda value: predecessor_states[value]["position"],
                )
            )
            for lane in call_host_lanes(call):
                occurrence_lane = (*lane, *logical_key)
                segment = (
                    graph_segment_by_lane_occurrence[occurrence_lane]
                    if occurrence_lane in graph_segment_by_lane_occurrence
                    else None
                )
                graph_segment_traversal.observe(lane, logical_key, segment)
                prepare_mechanisms = (
                    runtime_prepare_by_lane_occurrence[occurrence_lane]
                    if occurrence_lane in runtime_prepare_by_lane_occurrence
                    else ())
                prepare_profile = (
                    host_runtime_prepare_profiles[call.kernel_spec.hardware_id]
                    if prepare_mechanisms else None)
                for prepare_event in prepare_mechanisms:
                    service_s = (
                        prepare_profile.service_time_s(
                            prepare_event.mechanism,
                            pool_name=lane[0],
                            rank=lane[1])
                        * prepare_event.count
                    )
                    host_start_s = (
                        host_ready_by_lane[lane]
                        if lane in host_ready_by_lane else origin_time_s)
                    host_end_s = host_start_s + service_s
                    host_ready_by_lane[lane] = host_end_s
                    host_service_by_lane[lane] = (
                        host_service_by_lane[lane] + service_s
                        if lane in host_service_by_lane else service_s)
                    launch_cpu_time_s += service_s
                    relative_host_start_s = host_start_s - origin_time_s
                    relative_host_end_s = host_end_s - origin_time_s
                    represented_service_s = max(
                        0.0,
                        relative_host_end_s - relative_host_start_s,
                    )
                    exposed_s = min(
                        represented_service_s,
                        max(
                            0.0,
                            host_end_s - max(host_start_s, ready_time_s),
                        ),
                    )
                    occurrence_label = (
                        "single" if logical_key[1] < 0
                        else str(logical_key[1]))
                    event_id = (
                        f"runtime-prepare:{prepare_event.mechanism}:"
                        f"x{prepare_event.count}:{call.call_id}:"
                        f"{occurrence_label}:rank-{lane[1]}")
                    host_intervals.append(HostExecutionInterval(
                        event_id=event_id,
                        event_kind="runtime_prepare",
                        pool_name=lane[0],
                        rank=lane[1],
                        call_id=call.call_id,
                        call_occurrence_index=logical_key[1],
                        covered_call_occurrences=(logical_key,),
                        device_predecessor_occurrences=(
                            ordered_device_predecessors),
                        host_predecessor_event_id=(
                            last_host_event_by_lane[lane]
                            if lane in last_host_event_by_lane else None),
                        runtime_graph_mode=runtime_graph_mode,
                        graph_segment_id=None,
                        start_time_s=relative_host_start_s,
                        end_time_s=relative_host_end_s,
                        exposed_time_s=exposed_s,
                        overlapped_time_s=represented_service_s - exposed_s,
                        mechanism=prepare_event.mechanism,
                        event_count=prepare_event.count,
                    ))
                    last_host_event_by_lane[lane] = event_id
                    launch_kinds.add("runtime_prepare")
                if segment is None:
                    launch_kind = "eager"
                    service_s = profile.service_time_s(launch_kind)
                    host_start_s = (
                        host_ready_by_lane[lane]
                        if lane in host_ready_by_lane else origin_time_s)
                    host_end_s = host_start_s + service_s
                    host_ready_by_lane[lane] = host_end_s
                    host_service_by_lane[lane] = (
                        host_service_by_lane[lane] + service_s
                        if lane in host_service_by_lane else service_s)
                    launch_cpu_time_s += service_s
                    eager_dispatch_count += 1
                    host_ends.append(host_end_s)
                    relative_host_start_s = host_start_s - origin_time_s
                    relative_host_end_s = host_end_s - origin_time_s
                    represented_service_s = max(
                        0.0,
                        relative_host_end_s - relative_host_start_s,
                    )
                    exposed_s = min(
                        represented_service_s,
                        max(
                            0.0,
                            host_end_s - max(host_start_s, ready_time_s),
                        ),
                    )
                    occurrence_label = (
                        "single" if logical_key[1] < 0
                        else str(logical_key[1]))
                    event_id = (
                        f"eager-submit:{call.call_id}:"
                        f"{occurrence_label}:rank-{lane[1]}")
                    host_intervals.append(HostExecutionInterval(
                        event_id=event_id,
                        event_kind="eager_submit",
                        pool_name=lane[0],
                        rank=lane[1],
                        call_id=call.call_id,
                        call_occurrence_index=logical_key[1],
                        covered_call_occurrences=(logical_key,),
                        device_predecessor_occurrences=(
                            ordered_device_predecessors),
                        host_predecessor_event_id=(
                            last_host_event_by_lane[lane]
                            if lane in last_host_event_by_lane else None),
                        runtime_graph_mode=runtime_graph_mode,
                        graph_segment_id=None,
                        start_time_s=relative_host_start_s,
                        end_time_s=relative_host_end_s,
                        exposed_time_s=exposed_s,
                        overlapped_time_s=represented_service_s - exposed_s,
                    ))
                    last_host_event_by_lane[lane] = event_id
                    launch_kinds.add(launch_kind)
                    continue
                launch_kind = "graph_replay"
                identity = (lane[0], lane[1], segment.segment_id)
                if identity not in graph_ready_by_identity:
                    service_s = profile.service_time_s(launch_kind)
                    host_start_s = (
                        host_ready_by_lane[lane]
                        if lane in host_ready_by_lane else origin_time_s)
                    if segment.sync_before_replay:
                        # See the summary scheduler above: the source-level
                        # current-stream synchronize is modeled as a barrier.
                        host_start_s = max(host_start_s, device_ready_time_s)
                    host_end_s = host_start_s + service_s
                    host_ready_by_lane[lane] = host_end_s
                    host_service_by_lane[lane] = (
                        host_service_by_lane[lane] + service_s
                        if lane in host_service_by_lane else service_s)
                    graph_ready_by_identity[identity] = host_end_s
                    launch_cpu_time_s += service_s
                    graph_replay_count += 1
                    host_ends.append(host_end_s)
                    relative_host_start_s = host_start_s - origin_time_s
                    relative_host_end_s = host_end_s - origin_time_s
                    represented_service_s = max(
                        0.0,
                        relative_host_end_s - relative_host_start_s,
                    )
                    exposed_s = min(
                        represented_service_s,
                        max(
                            0.0,
                            host_end_s - max(host_start_s, ready_time_s),
                        ),
                    )
                    event_id = (
                        f"graph-replay:{segment.segment_id}:"
                        f"{lane[0]}:rank-{lane[1]}")
                    host_intervals.append(HostExecutionInterval(
                        event_id=event_id,
                        event_kind="graph_replay",
                        pool_name=lane[0],
                        rank=lane[1],
                        call_id=call.call_id,
                        call_occurrence_index=logical_key[1],
                        covered_call_occurrences=segment.call_occurrences,
                        device_predecessor_occurrences=(
                            ordered_device_predecessors),
                        host_predecessor_event_id=(
                            last_host_event_by_lane[lane]
                            if lane in last_host_event_by_lane else None),
                        runtime_graph_mode=runtime_graph_mode,
                        graph_segment_id=segment.segment_id,
                        start_time_s=relative_host_start_s,
                        end_time_s=relative_host_end_s,
                        exposed_time_s=exposed_s,
                        overlapped_time_s=represented_service_s - exposed_s,
                    ))
                    last_host_event_by_lane[lane] = event_id
                    launch_kinds.add(launch_kind)
                else:
                    host_ends.append(graph_ready_by_identity[identity])
                    if graph_segment_by_identity[identity] != segment:
                        raise RuntimeError(
                            "runtime Graph segment identity became ambiguous")
            ready_time_s = max(ready_time_s, *host_ends)
        def predecessor_sort_key(predecessor):
            cause = predecessor_causes[predecessor]
            if cause in ("dependency", "dependency_and_resource"):
                cause_priority = 0
            elif cause == "resource":
                cause_priority = 1
            else:
                raise ValueError(
                    f"unknown logical predecessor cause: {cause!r}")
            return cause_priority, predecessor_states[predecessor]["position"]

        ready_predecessors = sorted(
            (
                predecessor
                for predecessor in predecessor_causes
                if math.isclose(
                    predecessor_states[predecessor]["end_time_s"]
                    + launch_gap_s,
                    ready_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ),
            key=predecessor_sort_key,
        )
        predecessor = (
            ready_predecessors[0] if ready_predecessors else None)
        host_launch_critical_s = 0.0
        launch_gap_critical_s = 0.0
        if host_launch_profiles is not None:
            critical_prefix_end_s = (
                origin_time_s
                if predecessor is None
                else predecessor_states[predecessor]["end_time_s"]
            )
            if predecessor is not None:
                launch_gap_critical_s = launch_gap_s
            host_launch_critical_s = (
                ready_time_s - critical_prefix_end_s - launch_gap_critical_s)
        earliest_start = ready_time_s
        if shared_resource_calendar is not None and call.kernel_spec.transfer is not None:
            start, end = shared_resource_calendar.reserve_many(
                reserved_resource_ids, earliest_start, duration_s)
        else:
            start = earliest_start
            end = start + duration_s
        interval_start_time_s = start - origin_time_s
        interval_end_time_s = end - origin_time_s
        if predecessor is None:
            contributions = {}
            critical_call_count = 0
            critical_unique = True
        else:
            contributions = dict(
                predecessor_states[predecessor]["contributions"])
            critical_call_count = predecessor_states[predecessor][
                "critical_call_count"]
            critical_unique = predecessor_states[predecessor][
                "critical_unique"]
        if len(ready_predecessors) > 1:
            critical_unique = False
        if host_launch_critical_s > 0.0:
            contributions["host:launch"] = (
                contributions["host:launch"] + host_launch_critical_s
                if "host:launch" in contributions
                else host_launch_critical_s)
        if launch_gap_critical_s > 0.0:
            contributions["device:launch_gap"] = (
                contributions["device:launch_gap"] + launch_gap_critical_s
                if "device:launch_gap" in contributions
                else launch_gap_critical_s)
        wait_s = start - ready_time_s
        if wait_s > 1e-15:
            wait_category = "shared_transfer_resource_wait"
            if wait_category in contributions:
                contributions[wait_category] += wait_s
            else:
                contributions[wait_category] = wait_s
        category = f"kernel:{predictions[call.call_id].bottleneck_group}"
        interval_duration_s = interval_end_time_s - interval_start_time_s
        if category in contributions:
            contributions[category] += interval_duration_s
        else:
            contributions[category] = interval_duration_s
        state = {
            "logical_key": logical_key,
            "position": position,
            "end_time_s": end,
            "contributions": contributions,
            "critical_call_count": critical_call_count + 1,
            "critical_unique": critical_unique,
        }
        for resource_id in reserved_resource_ids:
            resource_predecessor[resource_id] = state
        if not terminal_candidates:
            maximum_end_time_s = end
            terminal_candidates.append(state)
        elif end > maximum_end_time_s:
            maximum_end_time_s = end
            terminal_candidates[:] = [
                candidate for candidate in terminal_candidates
                if math.isclose(
                    candidate["end_time_s"] - origin_time_s,
                    end - origin_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ]
            terminal_candidates.append(state)
        elif math.isclose(
                end - origin_time_s,
                maximum_end_time_s - origin_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15):
            terminal_candidates.append(state)
        call_occurrence_counts[call.call_id] += 1
        record_activity(
            pool_activity_records,
            call.pool_name,
            interval_start_time_s,
            interval_end_time_s,
        )
        for resource_id in reserved_resource_ids:
            record_activity(
                reserved_resource_activity_records,
                resource_id,
                interval_start_time_s,
                interval_end_time_s,
            )
        return state, (interval_start_time_s, interval_end_time_s)

    def require_states(states, logical_keys, label):
        missing = tuple(
            logical_key for logical_key in logical_keys
            if logical_key not in states)
        if missing:
            raise ValueError(
                f"logical kernel occurrence has unscheduled {label}: "
                f"{missing!r}")
        return tuple(states[logical_key] for logical_key in logical_keys)

    for (
        logical_key,
        dependency_keys,
        position,
        successor_count,
    ) in logical_schedule:
        call_id, occurrence_index = logical_key
        call = calls_by_id[call_id]
        dependency_states = require_states(
            state_by_key, dependency_keys, "dependencies")
        device_dependency_ends = (
            tuple(device_end_by_key[key] for key in dependency_keys)
            if host_launch_profiles is not None else ()
        )
        state, interval_times = schedule_call(
            call,
            logical_key,
            position,
            dependency_states,
            device_dependency_ends,
        )
        if successor_count:
            state_by_key[logical_key] = state
            remaining_dependency_uses[logical_key] = successor_count
        elif host_launch_profiles is not None:
            del device_end_by_key[logical_key]
        for dependency_key_value in dependency_keys:
            remaining_uses = (
                remaining_dependency_uses[dependency_key_value] - 1)
            if remaining_uses:
                remaining_dependency_uses[
                    dependency_key_value] = remaining_uses
            else:
                del remaining_dependency_uses[dependency_key_value]
                del state_by_key[dependency_key_value]
                if host_launch_profiles is not None:
                    del device_end_by_key[dependency_key_value]
        if occurrence_index < 0:
            interval_times_by_nonrepeated_call[call.call_id] = interval_times
            continue
        repeat = repeat_by_call[call_id]
        latest_repeated_interval_times[
            repeat.block_id][call_id] = interval_times
        bounds = repeat_bounds[repeat.block_id]
        if occurrence_index == 0:
            start_time_s = interval_times[0]
            bounds[0] = (
                start_time_s if bounds[0] is None
                else min(bounds[0], start_time_s)
            )
        if occurrence_index == repeat.repeat_count - 1:
            end_time_s = interval_times[1]
            bounds[1] = (
                end_time_s if bounds[1] is None
                else max(bounds[1], end_time_s)
            )
        repeated_interval_times[
            repeat.block_id][call_id].append(interval_times)

    graph_segment_traversal.finish()

    intervals = []
    for call in calls:
        if call.call_id not in repeat_by_call:
            start_time_s, end_time_s = (
                interval_times_by_nonrepeated_call[call.call_id])
        else:
            repeat = repeat_by_call[call.call_id]
            start_time_s, end_time_s = latest_repeated_interval_times[
                repeat.block_id][call.call_id]
        intervals.append(ExecutionInterval(
            call.call_id,
            call.pool_name,
            call.rank,
            call.resource,
            start_time_s,
            end_time_s,
            reserved_resources_by_call[call.call_id],
        ))
    block_repeat_timelines = []
    for repeat in plan.block_repeats:
        interval_groups = (
            repeated_interval_times[repeat.block_id]
            if materialize_timeline_details else None
        )
        if (materialize_timeline_details and any(
                len(interval_groups[call_id]) != repeat.repeat_count
                for call_id in repeat.call_ids)):
            raise RuntimeError(
                "kernel block repeat did not schedule every occurrence")
        first_iteration_start_time_s, last_iteration_end_time_s = (
            repeat_bounds[repeat.block_id])
        if (first_iteration_start_time_s is None
                or last_iteration_end_time_s is None):
            raise RuntimeError("kernel block repeat did not establish bounds")
        block_repeat_timelines.append(ExecutionBlockRepeat(
            repeat.block_id,
            repeat.repeat_count,
            first_iteration_start_time_s,
            last_iteration_end_time_s,
            (
                tuple(
                    (
                        call_id,
                        tuple(interval_groups[call_id]),
                    )
                    for call_id in repeat.call_ids
                )
                if materialize_timeline_details else ()
            ),
            (
                () if materialize_timeline_details else tuple(
                    (
                        call_id,
                        repeated_interval_digests[
                            repeat.block_id][call_id].hexdigest(),
                    )
                    for call_id in repeat.call_ids
                )
            ),
        ))
    duration = maximum_end_time_s - origin_time_s
    terminal_candidates = tuple(
        candidate for candidate in terminal_candidates
        if math.isclose(
            candidate["end_time_s"] - origin_time_s,
            duration,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    )
    if not terminal_candidates:
        raise RuntimeError("execution timeline has no terminal call")
    terminal_state = min(
        terminal_candidates, key=lambda state: state["position"])
    critical_path_unique = (
        len(terminal_candidates) == 1
        and terminal_state["critical_unique"]
    )
    pool_activity = tuple(
        ExecutionActivity(
            pool_name,
            pool_activity_records[pool_name]["interval_count"],
            merged_activity_intervals(
                pool_activity_records[pool_name]["intervals"]),
        )
        for pool_name in sorted(pool_activity_records)
    )
    reserved_resource_activity = tuple(
        ExecutionActivity(
            resource_id,
            reserved_resource_activity_records[resource_id][
                "interval_count"],
            merged_activity_intervals(
                reserved_resource_activity_records[resource_id][
                    "intervals"]),
        )
        for resource_id in sorted(reserved_resource_activity_records)
    )
    if host_launch_profiles is None:
        predicted_device_time_s = duration
        launch_critical_path_s = 0.0
        host_device_overlap_s = 0.0
    else:
        predicted_device_time_s = (
            maximum_device_end_time_s - origin_time_s
        )
        launch_critical_path_s = max(host_service_by_lane.values())
        host_device_overlap_s = max(
            0.0,
            predicted_device_time_s + launch_critical_path_s - duration,
        )
    return ExecutionTimeline(
        tuple(intervals),
        duration,
        tuple(block_repeat_timelines),
        pool_activity,
        reserved_resource_activity,
        tuple(
            (call.call_id, call_occurrence_counts[call.call_id])
            for call in calls
        ),
        tuple(sorted(
            terminal_state["contributions"].items()
        )),
        terminal_state["critical_call_count"],
        critical_path_unique,
        predicted_device_time_s,
        launch_cpu_time_s,
        launch_critical_path_s,
        host_device_overlap_s,
        eager_dispatch_count,
        graph_replay_count,
        tuple(sorted(launch_kinds)),
        tuple(host_intervals),
        runtime_graph_mode,
    )
