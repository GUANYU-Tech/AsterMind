"""Structured kernel calls and dependencies after workload lowering."""

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Tuple

from src.kernel.kernel_spec import KernelSpec




@dataclass(frozen=True)
class KernelCall:
    call_id: str
    kernel_spec: KernelSpec
    pool_name: str
    rank: int
    resource: str
    source_node_ids: Tuple[str, ...]
    dependencies: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("call_id", "pool_name", "resource"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"kernel call {name} must be a non-empty string")
        if not isinstance(self.kernel_spec, KernelSpec):
            raise TypeError("kernel call requires a KernelSpec")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("kernel call rank must be an integer")
        if self.rank < 0:
            raise ValueError("kernel call rank cannot be negative")
        if not isinstance(self.source_node_ids, (list, tuple)):
            raise TypeError(
                "kernel call source_node_ids must be a list or tuple")
        object.__setattr__(self, "source_node_ids", tuple(self.source_node_ids))
        if (not self.source_node_ids
                or any(not isinstance(node_id, str) or not node_id
                       for node_id in self.source_node_ids)):
            raise ValueError(
                "kernel call source_node_ids must contain non-empty strings")
        if len(self.source_node_ids) != len(set(self.source_node_ids)):
            raise ValueError("kernel call source_node_ids must be unique")
        if not isinstance(self.dependencies, (list, tuple)):
            raise TypeError("kernel call dependencies must be a list or tuple")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if any(not isinstance(dependency, str) or not dependency
               for dependency in self.dependencies):
            raise ValueError(
                "kernel call dependencies must contain non-empty strings")


@dataclass(frozen=True)
class KernelBlockRepeat:
    """One predicted block plan with recurrence dependencies."""

    block_id: str
    call_ids: Tuple[str, ...]
    repeat_count: int
    feedback_dependencies: Tuple[Tuple[str, Tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("kernel block repeat ID must be non-empty")
        if not isinstance(self.call_ids, (list, tuple)):
            raise TypeError("kernel block repeat call IDs must be a sequence")
        call_ids = tuple(self.call_ids)
        if (not call_ids
                or any(not isinstance(call_id, str) or not call_id
                       for call_id in call_ids)):
            raise ValueError(
                "kernel block repeat call IDs must be non-empty strings")
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("kernel block repeat call IDs must be unique")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)
                or self.repeat_count < 2):
            raise ValueError("kernel block repeat count must be an integer >= 2")
        if not isinstance(self.feedback_dependencies, (list, tuple)):
            raise TypeError(
                "kernel block feedback dependencies must be a sequence")
        feedback_dependencies = tuple(
            (destination, tuple(sources))
            for destination, sources in self.feedback_dependencies
        )
        known = set(call_ids)
        destinations = []
        for destination, sources in feedback_dependencies:
            if destination not in known:
                raise ValueError(
                    "kernel block feedback destination is outside the block")
            if (not sources
                    or any(source not in known for source in sources)):
                raise ValueError(
                    "kernel block feedback sources must be inside the block")
            destinations.append(destination)
        if len(destinations) != len(set(destinations)):
            raise ValueError(
                "kernel block feedback destinations must be unique")
        object.__setattr__(self, "call_ids", call_ids)
        object.__setattr__(
            self, "feedback_dependencies", feedback_dependencies)


@dataclass(frozen=True)
class KernelPlan:
    calls: Tuple[KernelCall, ...]
    block_repeats: Tuple[KernelBlockRepeat, ...]
    _topological_calls: Tuple[KernelCall, ...] = field(
        init=False, repr=False, compare=False)

    def __init__(self, calls: Iterable[KernelCall],
                 block_repeats: Iterable[KernelBlockRepeat] = ()):
        call_tuple = tuple(calls)
        block_repeat_tuple = tuple(block_repeats)
        if not call_tuple:
            raise ValueError("KernelPlan requires at least one kernel call")
        if any(not isinstance(call, KernelCall) for call in call_tuple):
            raise TypeError("KernelPlan requires KernelCall objects")
        ids = [call.call_id for call in call_tuple]
        if len(ids) != len(set(ids)):
            raise ValueError("kernel call IDs must be unique")
        known = set(ids)
        calls_by_id = {
            call.call_id: call for call in call_tuple
        }
        for call in call_tuple:
            unknown = set(call.dependencies) - known
            if unknown:
                raise ValueError(f"kernel call has unknown dependencies: {sorted(unknown)}")
            if call.call_id in call.dependencies:
                raise ValueError("kernel calls cannot depend on themselves")
        if any(
                not isinstance(repeat, KernelBlockRepeat)
                for repeat in block_repeat_tuple):
            raise TypeError(
                "KernelPlan block repeats require KernelBlockRepeat objects")
        block_ids = [repeat.block_id for repeat in block_repeat_tuple]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("kernel block repeat IDs must be unique")
        repeated_call_ids = []
        for repeat in block_repeat_tuple:
            unknown = set(repeat.call_ids) - known
            if unknown:
                raise ValueError(
                    "kernel block repeat references unknown calls: "
                    f"{sorted(unknown)}")
            repeated_call_ids.extend(repeat.call_ids)
            feedback_by_destination = {
                destination: sources
                for destination, sources in repeat.feedback_dependencies
            }
            repeated = set(repeat.call_ids)
            for call_id in repeat.call_ids:
                call = calls_by_id[call_id]
                external_dependencies = tuple(
                    dependency for dependency in call.dependencies
                    if dependency not in repeated
                )
                if external_dependencies and call_id not in feedback_by_destination:
                    raise ValueError(
                        "kernel block external dependency lacks feedback: "
                        f"{call_id!r}")
                if (not external_dependencies
                        and call_id in feedback_by_destination):
                    raise ValueError(
                        "kernel block feedback has no first-iteration "
                        f"dependency: {call_id!r}")
        if len(repeated_call_ids) != len(set(repeated_call_ids)):
            raise ValueError(
                "kernel calls cannot belong to multiple block repeats")
        object.__setattr__(self, "calls", call_tuple)
        object.__setattr__(self, "block_repeats", block_repeat_tuple)
        object.__setattr__(
            self, "_topological_calls", self._compute_topological_order())

    @property
    def logical_call_count(self) -> int:
        return len(self.calls) + sum(
            (repeat.repeat_count - 1) * len(repeat.call_ids)
            for repeat in self.block_repeats
        )

    def __iter__(self) -> Iterator[KernelCall]:
        return iter(self.calls)

    def topological_order(self) -> Tuple[KernelCall, ...]:
        return self._topological_calls

    def _compute_topological_order(self) -> Tuple[KernelCall, ...]:
        position = {
            call.call_id: index for index, call in enumerate(self.calls)
        }
        by_id = {call.call_id: call for call in self.calls}
        indegree = {
            call.call_id: len(set(call.dependencies)) for call in self.calls
        }
        successors = {call.call_id: [] for call in self.calls}
        for call in self.calls:
            for dependency in set(call.dependencies):
                successors[dependency].append(call.call_id)
        ready = [
            call.call_id for call in self.calls if indegree[call.call_id] == 0
        ]
        ordered = []
        while ready:
            next_ready = []
            for call_id in ready:
                ordered.append(by_id[call_id])
                for successor in successors[call_id]:
                    indegree[successor] -= 1
                    if indegree[successor] == 0:
                        next_ready.append(successor)
            ready = sorted(next_ready, key=position.__getitem__)
        if len(ordered) != len(self.calls):
            raise ValueError("KernelPlan dependencies must be acyclic")
        return tuple(ordered)


def hardware_for_call(call: KernelCall, hardware_by_id):
    """Return the HWSpec one planned call resolves to, or fail loud.

    A plan whose call names a hardware the caller supplied no spec for cannot
    be priced; the failure names the missing hardware rather than defaulting.
    """

    return require_hardware(call.kernel_spec.hardware_id, hardware_by_id)


def require_hardware(hardware_id: str, hardware_by_id):
    """Return the HWSpec for one hardware id, or fail loud when absent."""

    if hardware_id not in hardware_by_id:
        raise ValueError(f"no HWSpec supplied for {hardware_id}")
    return hardware_by_id[hardware_id]
