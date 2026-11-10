"""Evaluate a Cell's batch locally and publish its completion on the shared clock."""
from dataclasses import asdict

from network_sim.comm.operator import CommunicationOperator
from orbitserve.runtime.engine.cell_events import STEP_COMPLETE
from orbitserve.runtime.timeline import nanoseconds
from src.kernel.kernel_spec import KernelKind
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.kernel_lowering.execution_timeline import (
    _plan_schedule_inputs, _logical_schedule, call_host_lanes, call_reserved_resource_ids,
)


class NetworkBatchExecutor:
    """Time the dependency graph locally, with one in-flight batch per pool."""

    def __init__(self, cell_id, loop, comm, placements, *, on_batch_boundary=None):
        self.cell_id = cell_id
        self.loop = loop
        self.comm = comm
        self._on_batch_boundary = on_batch_boundary
        self.placements = {}
        device_ids = set()
        for placement in placements:
            if placement.cell_id != cell_id:
                raise ValueError("rank placement belongs to another Cell")
            identity = (cell_id, placement.pool_name, placement.rank)
            if identity in self.placements or placement.gpu_device_id in device_ids:
                raise ValueError("duplicate rank placement or shared physical GPU")
            self.placements[identity] = placement
            device_ids.add(placement.gpu_device_id)
        self._active = set()
        self.records = []
        self.input_distributions = []

    def _device(self, pool, rank):
        return self.placements[(self.cell_id, pool, rank)].gpu_device_id

    def _operator(self, call):
        transfer = call.kernel_spec.transfer
        if transfer is None:
            raise ValueError("communication kernel has no transfer contract")
        ranks = transfer.participants
        if transfer.operation == "send_recv":
            ranks = (transfer.source_rank, transfer.destination_rank)
        elif transfer.operation == "broadcast" and transfer.source_rank is not None:
            ranks = (transfer.source_rank, *(rank for rank in ranks if rank != transfer.source_rank))
        return CommunicationOperator(
            operation=transfer.operation, algorithm=transfer.algorithm,
            participants=tuple(self._device(call.pool_name, rank) for rank in ranks),
            bytes=transfer.bytes, bytes_semantics=transfer.bytes_semantics,
            ring_order=tuple(self._device(call.pool_name, rank) for rank in transfer.ring_order),
            reduction=transfer.reduction,
            partition_bytes=getattr(transfer, "partition_bytes", getattr(transfer, "owner_sizes", ())),
            traffic_matrix_bytes=getattr(transfer, "traffic_matrix_bytes", ()),
        )

    def start_batch(self, cell_id, step_id, components, execution, on_complete, *,
                    token_count, pool_name="main", on_next_batch=None):
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
        if isinstance(components, KernelPlan):
            plan = components
            predictions = execution.predict_local_plan(plan)
            native_timeline = None
        else:
            # Older explicitly selected dev roots return an execution bundle.
            plan, predictions = components.kernel_plan, components.predictions
            native_timeline = components.execution_timeline

        ready_ns = nanoseconds(self.loop.now_s)
        start_ns = ready_ns
        distribution = None
        ranks = pool.physical_ranks_for_pipeline_stage(0)
        if len(ranks) > 1:
            request = CommunicationOperator("broadcast", "direct",
                tuple(self._device(pool_name, rank) for rank in ranks), token_count * 8, "payload")
            estimate = self.comm.estimate(request)
            start_ns += estimate.duration_ns
            distribution = dict(cell_id=cell_id, step_id=step_id, ready_ns=ready_ns,
                                completed_ns=start_ns, communication_estimate=asdict(estimate))

        batch = self._time_plan(step_id, plan, predictions, native_timeline, execution, start_ns)
        end_s = batch["end_time_ns"] / 1e9
        self._active.add(pool_name)

        def finish():
            self._active.remove(pool_name)
            if distribution is not None:
                self.input_distributions.append(distribution)
            self.records.append(batch)
            on_complete(end_s)
            if self._on_batch_boundary is not None:
                self._on_batch_boundary()
            if on_next_batch is not None:
                on_next_batch()

        self.loop.schedule(end_s, STEP_COMPLETE, "batch:complete", finish,
                           owner_id=cell_id, metadata={"pool_name": pool_name, "step_id": step_id})

    def _time_plan(self, step_id, plan, predictions, native_timeline, execution, start_ns):
        inputs = _plan_schedule_inputs(plan)
        calls = inputs[1]
        logical = _logical_schedule(plan, *inputs)
        host_end = {}
        gaps = None
        if execution.host_launch_profiles:
            from src.serving.kernel_lowering.execution_timeline import _DeviceLaunchGapTracker
            gaps = _DeviceLaunchGapTracker(execution.host_launch_profiles)
            if native_timeline is None:
                from src.serving.kernel_lowering.execution_timeline import eager_host_ready_times
                logical = tuple(logical)
                keys = tuple(key for key, _, _, _ in logical)
                host_end = {key: nanoseconds(end_s) for key, end_s in
                            eager_host_ready_times(inputs[0], keys, execution.host_launch_profiles).items()}
        if native_timeline is not None:
            for interval in native_timeline.host_intervals:
                if interval.event_kind != "eager_submit" or interval.runtime_graph_mode != "NONE":
                    raise NotImplementedError("network path currently supports native eager Host only")
                key = (interval.call_id, interval.call_occurrence_index)
                host_end[key] = max(host_end.get(key, 0), nanoseconds(interval.end_time_s))

        records, end_by_key, resource_last, estimates = [], {}, {}, {}
        for key, dependencies, position, _ in logical:
            call = calls[key[0]]
            communication = call.kernel_spec.kind is KernelKind.COMMUNICATION
            if communication:
                if call.call_id not in estimates:
                    operator = self._operator(call)
                    estimates[call.call_id] = operator, self.comm.estimate(operator)
                operator, estimate = estimates[call.call_id]
                resources = tuple(f"network-endpoint:{gpu}" for gpu in operator.participants)
                prediction = None
                duration_ns = estimate.duration_ns
            else:
                self._device(call.pool_name, call.rank)
                resources = call_reserved_resource_ids(call)
                prediction = predictions[call.call_id].t_total_s
                duration_ns = nanoseconds(prediction)
            deps = set(dependencies)
            for resource in resources:
                if resource in resource_last:
                    deps.add(resource_last[resource])
                resource_last[resource] = key
            gap_ns = (nanoseconds(gaps.observe(call.kernel_spec.hardware_id,
                      tuple((lane, None) for lane in call_host_lanes(call)))) if gaps else 0)
            if execution.host_launch_profiles is not None and key not in host_end:
                raise ValueError("native eager Host record missing a logical occurrence")
            host_ready_ns = start_ns + host_end.get(key, 0)
            dependency_end_ns = max((end_by_key[dep] for dep in deps), default=start_ns)
            begin_ns = max(dependency_end_ns + gap_ns, host_ready_ns)
            end_ns = begin_ns + duration_ns
            end_by_key[key] = end_ns
            record = dict(call_id=call.call_id, occurrence=key[1], rank=call.rank,
                          kind=call.kernel_spec.kind.value, communication=communication,
                          dependencies=sorted(deps), position=position, start_time_ns=begin_ns,
                          end_time_ns=end_ns, native_prediction_s=prediction,
                          host_ready_time_ns=host_ready_ns, launch_gap_ns=gap_ns)
            if communication:
                record.update(communication_operation=operator.operation, communication_estimate=asdict(estimate))
            records.append(record)
        if len(records) != plan.logical_call_count or not records:
            raise RuntimeError("compact M-L plan was not fully expanded")
        return dict(cell_id=self.cell_id, step_id=step_id, start_time_ns=start_ns,
                    end_time_ns=max(end_by_key.values()), calls=records,
                    native_batch_duration_s=None if native_timeline is None else native_timeline.duration_s,
                    native_communication_duration_charged_s=0.0,
                    model="native M-L compute and isolated operator communication")
