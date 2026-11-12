"""Causal native M-L execution with external collective/P2P completion.

No upstream source is patched. Compact occurrences and eager launch gaps are
expanded by the pinned M-L helpers. Network operations replace (never add to)
native COMMUNICATION duration. Ring is a fluid traffic abstraction; it
does not emulate NCCL protocol or a separate GPU reduction kernel.
"""

from collections import defaultdict

from src.kernel.kernel_spec import KernelKind
from src.serving.kernel_lowering.execution_timeline import (
    _DeviceLaunchGapTracker,
    _logical_schedule,
    _plan_schedule_inputs,
    call_host_lanes,
    call_reserved_resource_ids,
)

from network_sim import TransferRequest
from orbitserve.runtime.engine.cell_events import DISPATCH, STEP_COMPLETE
from orbitserve.runtime.timeline import nanoseconds
from orbitserve.runtime.clock import Time, ticks


class CommunicationFailed(RuntimeError):
    def __init__(self, event):
        self.event = event.model_dump()
        super().__init__(f"network transfer failed: {event.id}: {event.reason}")


def ring_transfer_phases(operation, participants, size_bytes):
    """Traffic for equal-input ring collectives, not a NCCL kernel emulator.

    AllGather forwards one rank's input per hop. AllReduce has N-1 reduce-
    scatter hops and N-1 gather hops, each forwarding one Nth-sized chunk.
    Unequal/ragged inputs need their own explicit contract.
    """
    ranks = tuple(participants)
    if len(ranks) < 2 or len(set(ranks)) != len(ranks):
        raise ValueError("ring requires at least two distinct participants")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise ValueError("per-rank collective bytes must be a positive integer")
    if operation == "all_gather":
        chunk, hops = size_bytes, len(ranks) - 1
    elif operation == "all_reduce":
        if size_bytes % len(ranks):
            raise NotImplementedError(
                "equal-chunk ring all-reduce requires divisible payload bytes"
            )
        chunk, hops = size_bytes // len(ranks), 2 * (len(ranks) - 1)
    else:
        raise NotImplementedError(f"ring collective not connected: {operation}")
    sends = tuple(
        (rank, ranks[(index + 1) % len(ranks)], chunk)
        for index, rank in enumerate(ranks)
    )
    return tuple(sends for _ in range(hops))


def variable_collective_phases(operation, participants, bytes_by_owner, *, algorithm):
    """Explicit direct broadcast/reduce traffic for ragged AgRs semantics.

    vLLM's PyNCCL ragged branch groups Broadcast/Reduce calls per owner. Direct
    fanout/fanin here is a declared fluid algorithm, not NCCL's selected tree
    or ring, protocol, reduction kernel, scratch allocation or launch overhead.
    """
    ranks, sizes = tuple(participants), tuple(bytes_by_owner)
    if len(ranks) < 2 or len(set(ranks)) != len(ranks) or len(sizes) != len(ranks):
        raise ValueError(
            "variable collective requires one size per distinct participant"
        )
    if any(type(size) is not int or size < 0 for size in sizes):
        raise ValueError("owner payloads must be nonnegative integer bytes")
    if algorithm != "direct":
        raise NotImplementedError(
            "ragged collective currently requires explicit direct traffic"
        )
    if operation not in ("all_gatherv", "reduce_scatterv"):
        raise NotImplementedError(f"variable collective not connected: {operation}")
    sends = []
    for owner, size in zip(ranks, sizes):
        if size == 0:
            continue
        for other in ranks:
            if other != owner:
                sends.append(
                    (owner, other, size)
                    if operation == "all_gatherv"
                    else (other, owner, size)
                )
    return (tuple(sends),) if sends else ()


class NetworkBatchExecutor:
    """One event loop/runtime; exclusive rank placements, depth-one per Cell.

    start_batch returns immediately. Its callback runs only after every kernel
    and transfer finishes. Future compute times are not pre-booked from an
    isolated route estimate. All transfers share the same physical runtime.
    """

    def __init__(self, loop, network, placements):
        self.loop, self.network = loop, network
        self.placements = {}
        device_ids = set()
        for placement in placements:
            identity = (placement.cell_id, placement.pool_name, placement.rank)
            if identity in self.placements or placement.gpu_device_id in device_ids:
                raise ValueError("duplicate rank placement or shared physical GPU")
            device = network.topology.devices[placement.gpu_device_id]
            if device.kind != "gpu" or device.satellite != placement.satellite_id:
                raise ValueError("rank placement does not match physical GPU ownership")
            if (
                network.topology.satellite_to_cell[device.satellite]
                != placement.cell_id
            ):
                raise ValueError(
                    "rank placement does not match physical Cell ownership"
                )
            self.placements[identity] = placement
            device_ids.add(placement.gpu_device_id)
        self._active = set()
        self._generation = 0
        self._callbacks = {}
        self._transfer_counter = 0
        self.records, self.transfer_events = [], []

    def _device(self, cell_id, pool, rank):
        return self.placements[(cell_id, pool, rank)].gpu_device_id

    def _make_request(self, **fields):
        return TransferRequest(**fields)

    def _arm_network(self):
        self._generation += 1
        generation = self._generation
        next_ns = self.network.next_event_time_ns()
        if next_ns is not None:
            at_s = Time.from_ns(next_ns)
            if at_s < self.loop.now_s:
                raise RuntimeError("network clock missed an event")
            self.loop.schedule(
                at_s,
                DISPATCH,
                "network:advance",
                lambda: self._pump(generation, next_ns),
            )

    def _pump(self, generation, next_ns):
        if generation != self._generation:
            return
        for event in self.network.advance_to(next_ns):
            self.transfer_events.append(event.model_dump())
            callback = self._callbacks.pop(event.id)
            if event.kind == "failed":
                raise CommunicationFailed(event)
            if event.completion_time_ns != next_ns:
                raise RuntimeError(
                    "network completion was not delivered at its event time"
                )
            callback()
        self._arm_network()

    def _phase(
        self, cell_id, pool, operation, correlation_id, sends, on_complete, record
    ):
        pending = set()
        phase = []
        now_ns = nanoseconds(self.loop.now_s)

        def finished(identifier):
            pending.remove(identifier)
            if not pending:
                on_complete()

        for source_rank, target_rank, size in sends:
            self._transfer_counter += 1
            identifier = f"compute-transfer-{self._transfer_counter}"
            pending.add(identifier)
            request = self._make_request(
                id=identifier,
                source=self._device(cell_id, pool, source_rank),
                target=self._device(cell_id, pool, target_rank),
                size_bytes=size,
                ready_time_ns=now_ns,
                origin="multi_gpu",
                operation=operation,
                correlation_id=correlation_id,
            )
            self.network.submit(request)
            self._callbacks[identifier] = lambda identifier=identifier: finished(
                identifier
            )
            phase.append(request.model_dump())
        record["phases"].append(phase)
        self._arm_network()

    def _communication(self, cell_id, call, occurrence, record, on_complete):
        transfer = call.kernel_spec.transfer
        if transfer is None:
            raise ValueError("communication kernel has no physical transfer contract")
        participants = transfer.participants
        correlation = f"{cell_id}/{call.call_id}/{occurrence}"
        size = transfer.bytes
        try_collective = getattr(self.network, "try_collective", None)
        scoped_participants = (
            (transfer.source_rank, transfer.destination_rank)
            if transfer.operation == "send_recv"
            else participants
        )
        if try_collective and try_collective(
            cell_id,
            transfer,
            tuple(
                self._device(cell_id, call.pool_name, rank)
                for rank in scoped_participants
            ),
            record,
            on_complete,
        ):
            return
        if transfer.operation == "send_recv":
            if transfer.bytes_semantics != "payload" or len(participants) != 2:
                raise NotImplementedError("PP requires a two-rank payload transfer")
            self._phase(
                cell_id,
                call.pool_name,
                "pp_send_recv",
                correlation,
                [(transfer.source_rank, transfer.destination_rank, size)],
                on_complete,
                record,
            )
        elif transfer.operation in ("all_reduce", "all_gather"):
            if (
                transfer.bytes_semantics != "per_rank_input"
                or transfer.algorithm != "ring"
            ):
                raise NotImplementedError(
                    "external collectives require equal per-rank inputs and ring"
                )
            if transfer.operation == "all_reduce" and transfer.reduction != "sum":
                raise NotImplementedError("external ring only supports sum reduction")
            phases = ring_transfer_phases(transfer.operation, participants, size)

            def advance(index):
                if index == len(phases):
                    on_complete()
                    return
                if transfer.operation == "all_gather":
                    operation = "tp_all_gather"
                else:
                    operation = (
                        "tp_all_reduce_scatter"
                        if index < len(participants) - 1
                        else "tp_all_reduce_gather"
                    )
                self._phase(
                    cell_id,
                    call.pool_name,
                    operation,
                    correlation,
                    phases[index],
                    lambda: advance(index + 1),
                    record,
                )

            advance(0)
        elif transfer.operation in ("all_gatherv", "reduce_scatterv"):
            phases = variable_collective_phases(
                transfer.operation,
                participants,
                transfer.owner_sizes,
                algorithm=transfer.algorithm,
            )

            def advance_variable(index):
                if index == len(phases):
                    on_complete()
                else:
                    self._phase(
                        cell_id,
                        call.pool_name,
                        transfer.operation,
                        correlation,
                        phases[index],
                        lambda: advance_variable(index + 1),
                        record,
                    )

            advance_variable(0)
        else:
            raise NotImplementedError(
                f"network operation not connected: {transfer.operation}"
            )

    def submit_kv_transfer(self, transfer_id, source, target, size_bytes, on_complete):
        """PD/migration payload sharing the same runtime as GPU collectives."""
        if (
            not transfer_id
            or transfer_id in self._callbacks
            or type(size_bytes) is not int
            or size_bytes <= 0
            or source == target
            or not callable(on_complete)
        ):
            raise ValueError(
                "invalid KV transfer identity, endpoints, bytes or callback"
            )
        self.network.submit(
            self._make_request(
                id=transfer_id,
                source=self._device(*source),
                target=self._device(*target),
                size_bytes=size_bytes,
                ready_time_ns=nanoseconds(self.loop.now_s),
                origin="midlayer",
                operation="pd_kv",
                correlation_id=transfer_id,
            )
        )
        self._callbacks[transfer_id] = on_complete
        self._arm_network()

    def execute_variable_collective(
        self,
        cell_id,
        pool_name,
        operation,
        participants,
        bytes_by_owner,
        correlation_id,
        on_complete,
        *,
        algorithm,
    ):
        """Nonblocking collective event API for the DP graph integration.

        Caller releases compute only in on_complete. No native duration is
        added. This API alone does not execute any MoE or GPU reduction.
        """
        participants, bytes_by_owner = tuple(participants), tuple(bytes_by_owner)
        phases = variable_collective_phases(
            operation, participants, bytes_by_owner, algorithm=algorithm
        )
        for rank in participants:
            self._device(cell_id, pool_name, rank)
        record = {
            "operation": operation,
            "participants": tuple(participants),
            "bytes_by_owner": tuple(bytes_by_owner),
            "algorithm": algorithm,
            "start_time_ns": nanoseconds(self.loop.now_s),
            "end_time_ns": None,
            "phases": [],
        }

        def advance(index):
            if index == len(phases):
                record["end_time_ns"] = nanoseconds(self.loop.now_s)
                on_complete()
            else:
                self._phase(
                    cell_id,
                    pool_name,
                    operation,
                    correlation_id,
                    phases[index],
                    lambda: advance(index + 1),
                    record,
                )

        advance(0)
        return record

    def start_batch(
        self,
        cell_id,
        step_id,
        components,
        execution,
        on_complete,
        *,
        coordinated_dp=False,
        engine_terminals=None,
        on_engine_complete=None,
    ):
        if len(execution.deployment.worker_pools) != 1:
            raise NotImplementedError("submit each execution pool separately")
        pool_name, pool = next(iter(execution.deployment.worker_pools.items()))
        identity = (cell_id, pool_name)
        if identity in self._active:
            raise NotImplementedError("network pool supports one in-flight batch")
        if type(coordinated_dp) is not bool:
            raise ValueError("coordinated_dp must be boolean")
        if execution.host_runtime_prepare_profiles is not None:
            raise NotImplementedError(
                "Host runtime preparation is outside this network slice"
            )
        if pool.parallel_plan.config.data_parallel_size != 1 and not coordinated_dp:
            raise NotImplementedError(
                "DP batches require coordinated multi-engine execution"
            )
        plan = components.kernel_plan
        terminal_engines = {}
        engine_pending = {}
        if coordinated_dp:
            dp = pool.parallel_plan.config.data_parallel_size
            if (
                dp < 2
                or plan.block_repeats
                or engine_terminals is None
                or set(engine_terminals) != set(range(dp))
                or not callable(on_engine_complete)
            ):
                raise ValueError(
                    "coordinated DP requires explicit output terminals per engine"
                )
            known = {call.call_id for call in plan.calls}
            normalized = {}
            for engine, value in engine_terminals.items():
                terminals = (value,) if isinstance(value, str) else value
                if (
                    not isinstance(terminals, tuple)
                    or not terminals
                    or any(not isinstance(item, str) for item in terminals)
                    or len(set(terminals)) != len(terminals)
                ):
                    raise ValueError(
                        "DP engine output terminals must be distinct call IDs"
                    )
                normalized[engine] = terminals
            flat = tuple(call_id for ids in normalized.values() for call_id in ids)
            if len(set(flat)) != len(flat) or not set(flat) <= known:
                raise ValueError(
                    "DP engine terminals must be distinct calls in the joint plan"
                )
            producers = {
                dependency for call in plan.calls for dependency in call.dependencies
            }
            if set(flat) & producers:
                raise ValueError("DP completion cannot use an intermediate kernel")
            terminal_engines = {
                (call_id, -1): engine
                for engine, ids in normalized.items()
                for call_id in ids
            }
            engine_pending = {engine: set(ids) for engine, ids in normalized.items()}
        completed_engines = set()
        inputs = _plan_schedule_inputs(plan)
        logical = tuple(_logical_schedule(plan, *inputs))
        calls = inputs[1]
        host_end = {}
        for interval in components.execution_timeline.host_intervals:
            if (
                interval.event_kind != "eager_submit"
                or interval.runtime_graph_mode != "NONE"
            ):
                raise NotImplementedError(
                    "network path currently supports native eager Host only"
                )
            key = (interval.call_id, interval.call_occurrence_index)
            host_end[key] = max(
                host_end.get(key, 0),
                nanoseconds(interval.end_time_s),
            )
        gaps = (
            _DeviceLaunchGapTracker(execution.host_launch_profiles)
            if execution.host_launch_profiles
            else None
        )
        start_ns = nanoseconds(self.loop.now_s)
        start_ticks = ticks(self.loop.now_s)
        batch = {
            "cell_id": cell_id,
            "step_id": step_id,
            "start_time_ns": start_ns,
            "start_time_ticks": str(start_ticks),
            "end_time_ns": None,
            "calls": [],
            "native_batch_duration_s": components.execution_timeline.duration_s,
            "native_communication_duration_charged_s": 0.0,
            "model": "native M-L compute and eager Host; hosted collective and PP communication",
        }
        self.records.append(batch)
        tasks, successors, resource_last = {}, defaultdict(list), {}
        for key, dependencies, position, _ in logical:
            call = calls[key[0]]
            communication = call.kernel_spec.kind is KernelKind.COMMUNICATION
            if communication:
                resources = tuple(
                    f"network-endpoint:{self._device(cell_id, call.pool_name, r)}"
                    for r in call.kernel_spec.transfer.participants
                )
            else:
                # Native stream/resource ordering remains authoritative.
                resources = call_reserved_resource_ids(call)
                self._device(cell_id, call.pool_name, call.rank)
            deps = set(dependencies)
            for resource in resources:
                if resource in resource_last:
                    deps.add(resource_last[resource])
                resource_last[resource] = key
            gap_ns = (
                nanoseconds(
                    gaps.observe(
                        call.kernel_spec.hardware_id,
                        tuple((lane, None) for lane in call_host_lanes(call)),
                    )
                )
                if gaps
                else 0
            )
            if execution.host_launch_profiles is not None and key not in host_end:
                raise ValueError(
                    "native eager Host record missing a logical occurrence"
                )
            record = {
                "call_id": call.call_id,
                "occurrence": key[1],
                "rank": call.rank,
                "kind": call.kernel_spec.kind.value,
                "communication": communication,
                "dependencies": [list(d) for d in sorted(deps)],
                "position": position,
                "start_time_ns": None,
                "end_time_ns": None,
                "phases": [],
                "native_prediction_s": components.predictions[call.call_id].t_total_s,
                "host_ready_time_ns": start_ns + (host_end.get(key, 0)),
                "host_ready_time_ticks": str(start_ticks + host_end.get(key, 0) * 10**9),
                "launch_gap_ns": gap_ns,
            }
            batch["calls"].append(record)
            tasks[key] = {
                "remaining": len(deps),
                "dependency_end_ns": start_ns,
                "dependency_end_ticks": start_ticks,
                "call": call,
                "record": record,
                "done": False,
            }
            for dep in deps:
                successors[dep].append(key)
        if len(tasks) != plan.logical_call_count:
            raise RuntimeError("compact M-L plan was not fully expanded")
        self._active.add(identity)
        unfinished = len(tasks)

        def finish(key):
            nonlocal unfinished
            task = tasks[key]
            if task["done"]:
                raise RuntimeError("duplicate kernel completion")
            task["done"] = True
            end_ns = nanoseconds(self.loop.now_s)
            task["record"]["end_time_ns"] = end_ns
            task['record']['end_time_ticks'] = str(ticks(self.loop.now_s))
            unfinished -= 1
            if key in terminal_engines:
                engine = terminal_engines[key]
                engine_pending[engine].remove(key[0])
                if not engine_pending[engine]:
                    completed_engines.add(engine)
                    on_engine_complete(engine, self.loop.now_s)
            for next_key in successors[key]:
                successor = tasks[next_key]
                successor["remaining"] -= 1
                successor["dependency_end_ns"] = max(
                    successor["dependency_end_ns"], end_ns
                )
                successor['dependency_end_ticks'] = max(successor['dependency_end_ticks'], ticks(self.loop.now_s))
                if successor["remaining"] == 0:
                    ready(next_key)
            if unfinished == 0:
                if terminal_engines and completed_engines != set(
                    terminal_engines.values()
                ):
                    raise RuntimeError(
                        "joint DP graph drained without every engine completion"
                    )
                batch["end_time_ns"] = end_ns
                batch['end_time_ticks'] = str(ticks(self.loop.now_s))
                self._active.remove(identity)
                on_complete(self.loop.now_s)

        def launch(key):
            task = tasks[key]
            call, record = task["call"], task["record"]
            record["start_time_ns"] = nanoseconds(self.loop.now_s)
            record['start_time_ticks'] = str(ticks(self.loop.now_s))
            if record["communication"]:
                self._communication(cell_id, call, key[1], record, lambda: finish(key))
            else:
                end_time = Time(self.loop.now_s) + Time.from_ns(nanoseconds(record['native_prediction_s']))
                self.loop.schedule(
                    end_time, STEP_COMPLETE, "kernel:complete", lambda: finish(key)
                )

        def ready(key):
            task, record = tasks[key], tasks[key]["record"]
            at_ticks = max(
                task['dependency_end_ticks'] + record['launch_gap_ns'] * 10**9,
                int(record['host_ready_time_ticks']),
            )
            self.loop.schedule(
                Time.from_ticks(at_ticks), DISPATCH, "kernel:launch", lambda: launch(key)
            )

        for key in sorted(tasks, key=lambda key: tasks[key]["record"]["position"]):
            if tasks[key]["remaining"] == 0:
                ready(key)
