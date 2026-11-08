"""Causal native M-L execution with external collective/P2P completion.

No upstream source is patched. Compact occurrences and eager launch gaps are
expanded by the pinned M-L helpers. Network operations replace (never add to)
native COMMUNICATION duration. Ring is a fluid traffic abstraction; it
does not emulate NCCL protocol or a separate GPU reduction kernel.
"""
from collections import defaultdict

from network_sim.comm.model import TransferRequest
from orbitserve.runtime.engine.cell_events import DISPATCH, STEP_COMPLETE
from orbitserve.runtime.engine.network_events import nanoseconds
from src.kernel.kernel_spec import KernelKind
from src.serving.kernel_lowering.execution_timeline import (
    _plan_schedule_inputs, _logical_schedule,
    call_host_lanes, call_reserved_resource_ids,
)


def ring_transfer_phases(operation, participants, size_bytes):
    """Traffic for equal-input ring collectives, not a NCCL kernel emulator.

    AllGather forwards one rank's input per hop. AllReduce has N-1 reduce-
    scatter hops and N-1 gather hops, each forwarding one Nth-sized chunk.
    Unequal/ragged inputs need their own explicit contract.
    """
    ranks = tuple(participants)
    if len(ranks) < 2 or len(set(ranks)) != len(ranks):
        raise ValueError("ring requires at least two distinct participants")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError("per-rank collective bytes must be a positive integer")
    if operation == "all_gather":
        chunk, hops = size_bytes, len(ranks) - 1
    elif operation == "all_reduce":
        if size_bytes % len(ranks):
            raise NotImplementedError("equal-chunk ring all-reduce requires divisible payload bytes")
        chunk, hops = size_bytes // len(ranks), 2 * (len(ranks) - 1)
    else:
        raise NotImplementedError(f"ring collective not connected: {operation}")
    sends = tuple((rank, ranks[(index + 1) % len(ranks)], chunk)
                  for index, rank in enumerate(ranks))
    return tuple(sends for _ in range(hops))


def variable_collective_phases(operation, participants, bytes_by_owner, *, algorithm):
    """Explicit direct broadcast/reduce traffic for ragged AgRs semantics.

    vLLM's PyNCCL ragged branch groups Broadcast/Reduce calls per owner. Direct
    fanout/fanin here is a declared fluid algorithm, not NCCL's selected tree
    or ring, protocol, reduction kernel, scratch allocation or launch overhead.
    """
    ranks, sizes = tuple(participants), tuple(bytes_by_owner)
    if len(ranks) < 2 or len(set(ranks)) != len(ranks) or len(sizes) != len(ranks):
        raise ValueError("variable collective requires one size per distinct participant")
    if any(type(size) is not int or size < 0 for size in sizes):
        raise ValueError("owner payloads must be nonnegative integer bytes")
    if algorithm != "direct":
        raise NotImplementedError("ragged collective currently requires explicit direct traffic")
    if operation not in ("all_gatherv", "reduce_scatterv"):
        raise NotImplementedError(f"variable collective not connected: {operation}")
    sends = []
    for owner, size in zip(ranks, sizes):
        if size == 0:
            continue
        for other in ranks:
            if other != owner:
                sends.append((owner, other, size) if operation == "all_gatherv"
                             else (other, owner, size))
    return (tuple(sends),) if sends else ()


class NetworkBatchExecutor:
    """Execute one Cell's batches on the shared clock and network.

    Each pool can have one batch in flight. The batch callback runs after
    all local kernels and network transfers finish.
    """

    def __init__(self, cell_id, transport, placements, *, on_batch_boundary=None):
        self.cell_id = cell_id
        self.transport = transport
        self.loop = transport.loop
        self._on_batch_boundary = on_batch_boundary
        network = transport.network
        self.placements = {}
        device_ids = set()
        for placement in placements:
            if placement.cell_id != cell_id:
                raise ValueError("rank placement belongs to another Cell")
            identity = (cell_id, placement.pool_name, placement.rank)
            if identity in self.placements or placement.gpu_device_id in device_ids:
                raise ValueError("duplicate rank placement or shared physical GPU")
            device = network.topology.devices[placement.gpu_device_id]
            if device.kind != "gpu" or device.satellite != placement.satellite_id:
                raise ValueError("rank placement does not match physical GPU ownership")
            if network.topology.satellite_to_cell[device.satellite] != placement.cell_id:
                raise ValueError("rank placement does not match physical Cell ownership")
            self.placements[identity] = placement
            device_ids.add(placement.gpu_device_id)
        self._active = set()
        self._transfer_counter = 0
        self.records = []
        self.input_distributions = []

    def _device(self, cell_id, pool, rank):
        return self.placements[(cell_id, pool, rank)].gpu_device_id

    def _phase(self, cell_id, pool, operation, correlation_id, sends, on_complete, record):
        phase = []
        now_ns = nanoseconds(self.loop.now_s)
        for source_rank, target_rank, size in sends:
            self._transfer_counter += 1
            phase.append(TransferRequest(
                id=f"compute-transfer-{self.cell_id}-{self._transfer_counter}",
                source=self._device(cell_id, pool, source_rank),
                target=self._device(cell_id, pool, target_rank),
                size_bytes=size,
                ready_time_ns=now_ns,
                origin="multi_gpu",
                operation=operation,
                correlation_id=correlation_id,
            ))
        pending = {request.id for request in phase}

        def finished(identifier):
            pending.remove(identifier)
            if not pending:
                on_complete()

        self.transport.submit_all(phase, finished)
        record["phases"].append([request.model_dump() for request in phase])

    def _communication(self, cell_id, call, occurrence, record, on_complete):
        transfer = call.kernel_spec.transfer
        if transfer is None:
            raise ValueError("communication kernel has no physical transfer contract")
        participants = transfer.participants
        correlation = f"{cell_id}/{call.call_id}/{occurrence}"
        size = transfer.bytes
        if transfer.operation == "send_recv":
            if transfer.bytes_semantics != "payload" or len(participants) != 2:
                raise NotImplementedError("PP requires a two-rank payload transfer")
            self._phase(cell_id, call.pool_name, "pp_send_recv", correlation,
                        [(transfer.source_rank, transfer.destination_rank, size)], on_complete, record)
        elif transfer.operation in ("all_reduce", "all_gather"):
            if transfer.bytes_semantics != "per_rank_input" or transfer.algorithm != "ring":
                raise NotImplementedError("external collectives require equal per-rank inputs and ring")
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
                    operation = ("tp_all_reduce_scatter" if index < len(participants) - 1
                                 else "tp_all_reduce_gather")
                self._phase(cell_id, call.pool_name, operation, correlation, phases[index],
                            lambda: advance(index + 1), record)
            advance(0)
        elif transfer.operation in ("all_gatherv", "reduce_scatterv"):
            phases = variable_collective_phases(transfer.operation, participants,
                                                 transfer.owner_sizes, algorithm=transfer.algorithm)
            def advance_variable(index):
                if index == len(phases):
                    on_complete()
                else:
                    self._phase(cell_id, call.pool_name, transfer.operation, correlation,
                                phases[index], lambda: advance_variable(index + 1), record)
            advance_variable(0)
        else:
            raise NotImplementedError(f"network operation not connected: {transfer.operation}")

    def start_batch(self, cell_id, step_id, components, execution, on_complete, *, token_count, pool_name="main"):
        if cell_id != self.cell_id:
            raise ValueError("batch belongs to another Cell")
        pool = execution.deployment.worker_pools[pool_name]
        if pool_name in self._active:
            raise NotImplementedError("network pool supports one in-flight batch")
        if execution.host_runtime_prepare_profiles is not None:
            raise NotImplementedError("Host runtime preparation is outside this network slice")
        if pool.parallel_plan.config.data_parallel_size != 1:
            raise NotImplementedError("each Cell owns one replica")
        if type(token_count) is not int or token_count <= 0:
            raise ValueError("batch input distribution requires the scheduled token count")
        self._active.add(pool_name)
        participants = pool.physical_ranks_for_pipeline_stage(0)

        def begin_plan():
            self._start_plan(pool_name, step_id, components, execution, on_complete)

        if len(participants) == 1:
            begin_plan()
            return
        record = dict(cell_id=cell_id, step_id=step_id, ready_ns=nanoseconds(self.loop.now_s), phases=[])
        self.input_distributions.append(record)

        def distributed():
            record["completed_ns"] = nanoseconds(self.loop.now_s)
            begin_plan()

        self._phase(cell_id, pool_name, "input_token_ids", f"{cell_id}/{step_id}/input",
                    [(participants[0], rank, token_count * 8) for rank in participants[1:]],
                    distributed, record)

    def _start_plan(self, pool_name, step_id, components, execution, on_complete):
        cell_id = self.cell_id
        plan = components.kernel_plan
        inputs = _plan_schedule_inputs(plan)
        logical = tuple(_logical_schedule(plan, *inputs))
        calls = inputs[1]
        host_end = {}
        for interval in components.execution_timeline.host_intervals:
            if interval.event_kind != "eager_submit" or interval.runtime_graph_mode != "NONE":
                raise NotImplementedError("network path currently supports native eager Host only")
            key = (interval.call_id, interval.call_occurrence_index)
            host_end[key] = max(host_end[key] if key in host_end else 0, nanoseconds(interval.end_time_s))
        gaps = None
        if execution.host_launch_profiles:
            from src.serving.kernel_lowering.execution_timeline import _DeviceLaunchGapTracker
            gaps = _DeviceLaunchGapTracker(execution.host_launch_profiles)
        start_ns = nanoseconds(self.loop.now_s)
        batch = {"cell_id": cell_id, "step_id": step_id, "start_time_ns": start_ns,
                 "end_time_ns": None, "calls": [], "native_batch_duration_s": components.execution_timeline.duration_s,
                 "native_communication_duration_charged_s": 0.0,
                 "model": "native M-L compute and eager Host; FlowSim ring and PP payload"}
        self.records.append(batch)
        tasks, successors, resource_last = {}, defaultdict(list), {}
        for key, dependencies, position, _ in logical:
            call = calls[key[0]]
            communication = call.kernel_spec.kind is KernelKind.COMMUNICATION
            if communication:
                resources = tuple(f"network-endpoint:{self._device(cell_id, call.pool_name, r)}"
                                  for r in call.kernel_spec.transfer.participants)
            else:
                # Native stream/resource ordering remains authoritative.
                resources = call_reserved_resource_ids(call)
                self._device(cell_id, call.pool_name, call.rank)
            deps = set(dependencies)
            for resource in resources:
                if resource in resource_last:
                    deps.add(resource_last[resource])
                resource_last[resource] = key
            gap_ns = (nanoseconds(gaps.observe(call.kernel_spec.hardware_id,
                      tuple((lane, None) for lane in call_host_lanes(call)))) if gaps else 0)
            if execution.host_launch_profiles is not None and key not in host_end:
                raise ValueError("native eager Host record missing a logical occurrence")
            record = {"call_id": call.call_id, "occurrence": key[1], "rank": call.rank,
                      "kind": call.kernel_spec.kind.value, "communication": communication,
                      "dependencies": [list(d) for d in sorted(deps)], "position": position,
                      "start_time_ns": None, "end_time_ns": None, "phases": [],
                      "native_prediction_s": components.predictions[call.call_id].t_total_s,
                      "host_ready_time_ns": start_ns + (host_end[key] if key in host_end else 0),
                      "launch_gap_ns": gap_ns}
            batch["calls"].append(record)
            tasks[key] = {"remaining": len(deps), "dependency_end_ns": start_ns,
                          "call": call, "record": record, "done": False}
            for dep in deps:
                successors[dep].append(key)
        if len(tasks) != plan.logical_call_count:
            raise RuntimeError("compact M-L plan was not fully expanded")
        unfinished = len(tasks)

        def finish_batch():
            self._active.remove(pool_name)
            on_complete(self.loop.now_s)
            if self._on_batch_boundary is not None:
                self._on_batch_boundary()

        def finish(key):
            nonlocal unfinished, ready
            task = tasks[key]
            if task["done"]:
                raise RuntimeError("duplicate kernel completion")
            task["done"] = True
            end_ns = nanoseconds(self.loop.now_s)
            task["record"]["end_time_ns"] = end_ns
            unfinished -= 1
            for next_key in successors[key]:
                successor = tasks[next_key]
                successor["remaining"] -= 1
                successor["dependency_end_ns"] = max(successor["dependency_end_ns"], end_ns)
                if successor["remaining"] == 0:
                    ready(next_key)
            if unfinished == 0:
                # Break the ready/launch/finish callback cycle after its last task.
                ready = None
                batch["end_time_ns"] = end_ns
                self.loop.schedule(self.loop.now_s, STEP_COMPLETE, f"{cell_id}:step_done",
                                   finish_batch, owner_id=cell_id)

        def launch(key):
            task = tasks[key]
            call, record = task["call"], task["record"]
            record["start_time_ns"] = nanoseconds(self.loop.now_s)
            if record["communication"]:
                self._communication(cell_id, call, key[1], record, lambda: finish(key))
            else:
                end_ns = record["start_time_ns"] + nanoseconds(record["native_prediction_s"])
                self.loop.schedule(end_ns / 1e9, STEP_COMPLETE, "kernel:complete", lambda: finish(key))

        def ready(key):
            task, record = tasks[key], tasks[key]["record"]
            at_ns = max(task["dependency_end_ns"] + record["launch_gap_ns"], record["host_ready_time_ns"])
            self.loop.schedule(at_ns / 1e9, DISPATCH, "kernel:launch", lambda: launch(key))

        for key in sorted(tasks, key=lambda key: tasks[key]["record"]["position"]):
            if tasks[key]["remaining"] == 0:
                ready(key)
