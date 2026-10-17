"""Run-local Serving workload construction, lowering, and prediction."""

from dataclasses import dataclass, replace
from collections import OrderedDict
import time
from typing import Callable, Mapping

from src.serving.deployment.deployment_plan import DeploymentPlan
from src.serving.kernel_lowering.execution_timeline import (
    ExecutionResourceCalendar,
    _common_id_prefix,
    _normalize_id,
    _plan_structure_key,
    build_execution_timeline,
    build_plan_structure_runtime_graph_segments,
    build_runtime_graph_segments,
)
from src.serving.kernel_lowering.kernel_predict import (
    KernelPredictionMemo,
    predict_kernel_plan,
)
from src.serving.kernel_lowering.lower_workload import (
    KernelLoweringMemo,
    lower_workload,
)
from src.serving.kv_cache.kv_cache_events import KVCacheEvent
from src.serving.schedule.schedule_step import ScheduleStep
from src.serving.workload.build_workload import (
    BlockConstructionMemo,
    build_workload,
)
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.runtime_inputs import ExpertRoutingMemo, RuntimeInputs
from src.serving.workload.workload_graph import WorkloadGraph
from src.kernel.predictors.common import graph_replayed_launch
from src.serving.host_timing import (
    HostLaunchProfile,
    HostRuntimePrepareProfile,
    RuntimePrepareEvent,
    RuntimeGraphSegmentDescriptor,
)


@dataclass(frozen=True)
class ServingExecutionComponents:
    workload: object
    kernel_plan: object
    predictions: Mapping[str, object]
    execution_timeline: object


def _runtime_graph_mode(
    schedule: ScheduleStep,
    graph_execution,
) -> str:
    uniform_decode = all(
        request.scheduled_tokens
        == graph_execution.uniform_decode_query_len
        for request in schedule.requests
    )
    return graph_execution.resolve(
        schedule.token_count,
        uniform_decode=uniform_decode,
    ).runtime_mode


_RUNTIME_PREPARE_EVENTS_CACHE_LIMIT = 8192


def _cached_runtime_prepare_events(
    cache: OrderedDict,
    builder,
    kernel_plan,
    materialize_timeline_details: bool,
):
    """Return prepare events for one plan, cached per normalized structure.

    The builder output is a pure function of the prefix-normalized plan
    structure (``_plan_structure_key`` covers every input the builder
    reads: identifiers, ranks, lanes, kinds, variants, transfer
    operations, and repeat layout), so it is cached per structure.  The
    summary timeline consumes the normalized form directly; the detailed
    timeline gets keys remapped to the current step's identifier prefix.
    Returns ``(events, normalized_events, events_token)`` with exactly one
    of the two event forms populated."""

    prefix = _common_id_prefix(kernel_plan)
    structure_key = _plan_structure_key(
        kernel_plan.topological_order(),
        kernel_plan.block_repeats,
        prefix,
    )
    if structure_key in cache:
        cached = cache[structure_key]
        cache.move_to_end(structure_key)
    else:
        raw_events = builder(kernel_plan)
        cached = (
            prefix,
            raw_events,
            {
                (
                    pool_name,
                    rank,
                    _normalize_id(call_id, prefix),
                    occurrence_index,
                ): mechanisms
                for (pool_name, rank, call_id, occurrence_index),
                mechanisms in raw_events.items()
            },
        )
        cache[structure_key] = cached
        while len(cache) > _RUNTIME_PREPARE_EVENTS_CACHE_LIMIT:
            cache.popitem(last=False)
    building_prefix, cached_events, normalized_events = cached
    if not materialize_timeline_details:
        return None, normalized_events, cached
    if building_prefix == prefix:
        return cached_events, None, None
    return (
        {
            (
                pool_name,
                rank,
                (
                    prefix + call_id[len(building_prefix):]
                    if call_id.startswith(building_prefix)
                    else call_id
                ),
                occurrence_index,
            ): mechanisms
            for (pool_name, rank, call_id, occurrence_index),
            mechanisms in cached_events.items()
        },
        None,
        None,
    )


class ServingExecutionContext:
    """Own all exact run-local reuse state for one Serving execution."""

    def __init__(
        self,
        model: ModelConfig,
        deployment: DeploymentPlan,
        hardware_by_id: Mapping[str, object],
        launch_policy: str,
        prediction_method: str,
        runtime_inputs: RuntimeInputs,
        *,
        execution_mode: str = "accelerated",
        host_launch_profiles: Mapping[str, HostLaunchProfile] | None = None,
        host_runtime_prepare_profiles: Mapping[
            str, HostRuntimePrepareProfile] | None = None,
        runtime_prepare_event_builder: Callable[
            [object], Mapping[
                tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]
            ]
        ] | None = None,
        materialize_timeline_details: bool = True,
    ) -> None:
        if not isinstance(model, ModelConfig):
            raise TypeError("Serving execution model must be ModelConfig")
        if not isinstance(deployment, DeploymentPlan):
            raise TypeError(
                "Serving execution deployment must be DeploymentPlan")
        if not isinstance(hardware_by_id, Mapping) or not hardware_by_id:
            raise TypeError(
                "Serving execution hardware_by_id must be a non-empty mapping")
        if not isinstance(runtime_inputs, RuntimeInputs):
            raise TypeError(
                "Serving execution runtime_inputs must be RuntimeInputs")
        if execution_mode not in ("accelerated", "reference"):
            raise ValueError(
                f"unknown Serving execution mode: {execution_mode!r}")
        if (execution_mode == "accelerated"
                and runtime_inputs.expert_routing.source_kind
                == "replay_observation"):
            raise NotImplementedError(
                "per-layer expert routing replay requires reference execution")
        self.model = model
        self.deployment = deployment
        self.hardware_by_id = dict(hardware_by_id)
        self.launch_policy = launch_policy
        self.prediction_method = prediction_method
        self.runtime_inputs = runtime_inputs
        self.execution_mode = execution_mode
        if not isinstance(materialize_timeline_details, bool):
            raise TypeError(
                "timeline detail materialization flag must be boolean")
        self.materialize_timeline_details = materialize_timeline_details
        self._executed_workloads = 0
        self._phase_wall_time_s = {
            "workload_construction": 0.0,
            "kernel_lowering": 0.0,
            "kernel_prediction": 0.0,
            "timeline_construction": 0.0,
        }
        if host_launch_profiles is None:
            self.host_launch_profiles = None
        else:
            self.host_launch_profiles = dict(host_launch_profiles)
        if host_runtime_prepare_profiles is None:
            self.host_runtime_prepare_profiles = None
        else:
            self.host_runtime_prepare_profiles = dict(
                host_runtime_prepare_profiles)
        if ((self.host_runtime_prepare_profiles is None)
                != (runtime_prepare_event_builder is None)):
            raise ValueError(
                "runtime prepare profiles and event builder must be supplied together")
        self.runtime_prepare_event_builder = runtime_prepare_event_builder
        self._runtime_prepare_events_cache: OrderedDict = OrderedDict()
        accelerated = execution_mode == "accelerated"
        if accelerated:
            self._block_memo = BlockConstructionMemo()
            self._routing_memo = ExpertRoutingMemo()
            self._lowering_memo = KernelLoweringMemo()
            self._prediction_memo = KernelPredictionMemo()
        else:
            self._block_memo = None
            self._routing_memo = None
            self._lowering_memo = None
            self._prediction_memo = None

    def _add_side_stream_fork_node_boundary(self, kernel_plan, predictions):
        """Charge SGLang's per-layer side-stream fork on every graph node.

        Under graph capture SGLang runs ``kv_a_layernorm`` on ``alt_stream``
        forked from and joined back to the main stream once per MLA layer
        (v0.5.19 ``forward_mla.py:316-335``; ``alt_stream`` exists on CUDA,
        ``deepseek_v2.py:2666-2674``).  A replayed graph with that topology
        spaces every dependent node by the hardware's measured side-stream
        fork boundary on top of the pure-chain node the kernel prices assume.
        """

        if self.runtime_inputs.serving_framework != "sglang":
            return predictions
        replayed = [
            call for call in kernel_plan.calls
            if graph_replayed_launch(call.kernel_spec)]
        if not replayed:
            return predictions
        if self.model.model_type != "deepseek_v3" or not self.model.q_lora_rank:
            raise NotImplementedError(
                "SGLang graph node boundary is source-backed only for the "
                "DeepSeek-V3 MLA alt-stream fork")
        adjusted = dict(predictions)
        for call in replayed:
            increment_s = self.hardware_by_id[
                call.kernel_spec.hardware_id
            ].side_stream_fork_graph_node_increment_s()
            prediction = adjusted[call.call_id]
            adjusted[call.call_id] = replace(
                prediction,
                t_total_s=prediction.t_total_s + increment_s,
                t_launch_s=prediction.t_launch_s + increment_s,
            )
        return adjusted

    def execute_step(
        self,
        schedule: ScheduleStep,
        *,
        kv_cache_events: tuple[KVCacheEvent, ...] = (),
        origin_time_s: float = 0.0,
        shared_resource_calendar: ExecutionResourceCalendar | None = None,
        runtime_prepare_events: Mapping[
            tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]] | None = None,
    ) -> ServingExecutionComponents:
        started = time.perf_counter()
        workload = build_workload(
            schedule,
            self.model,
            self.deployment,
            launch_policy=self.launch_policy,
            kv_cache_events=kv_cache_events,
            runtime_inputs=self.runtime_inputs,
            routing_memo=self._routing_memo,
            block_memo=self._block_memo,
        )
        self._phase_wall_time_s["workload_construction"] += (
            time.perf_counter() - started)
        return self.execute_workload(
            workload,
            origin_time_s=origin_time_s,
            shared_resource_calendar=shared_resource_calendar,
            runtime_graph_mode=_runtime_graph_mode(
                schedule, self.runtime_inputs.graph_execution),
            runtime_prepare_events=runtime_prepare_events,
        )

    def execute_workload(
        self,
        workload: WorkloadGraph,
        *,
        origin_time_s: float = 0.0,
        shared_resource_calendar: ExecutionResourceCalendar | None = None,
        runtime_graph_mode: str | None = None,
        runtime_graph_segments: tuple[
            RuntimeGraphSegmentDescriptor, ...
        ] | None = None,
        runtime_prepare_events: Mapping[
            tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]] | None = None,
    ) -> ServingExecutionComponents:
        if not isinstance(workload, WorkloadGraph):
            raise TypeError(
                "Serving execution workload must be WorkloadGraph")
        started = time.perf_counter()
        kernel_plan = lower_workload(
            workload,
            self.deployment,
            self.hardware_by_id,
            self._lowering_memo,
        )
        self._phase_wall_time_s["kernel_lowering"] += (
            time.perf_counter() - started)
        started = time.perf_counter()
        predictions = predict_kernel_plan(
            kernel_plan,
            self.hardware_by_id,
            self.prediction_method,
            self._prediction_memo,
        )
        predictions = self._add_side_stream_fork_node_boundary(
            kernel_plan, predictions)
        self._phase_wall_time_s["kernel_prediction"] += (
            time.perf_counter() - started)
        normalized_runtime_prepare_events = None
        runtime_prepare_events_token = None
        if self.runtime_prepare_event_builder is not None:
            if runtime_prepare_events is not None:
                raise ValueError(
                    "explicit runtime prepare events conflict with context builder")
            if runtime_graph_mode is None:
                raise ValueError(
                    "runtime prepare builder requires an explicit runtime Graph mode")
            # Source occurrence rules are authored against the eager program;
            # the execution timeline discards events on occurrences captured
            # by a runtime Graph segment and charges the eager remainder.
            (
                runtime_prepare_events,
                normalized_runtime_prepare_events,
                runtime_prepare_events_token,
            ) = _cached_runtime_prepare_events(
                self._runtime_prepare_events_cache,
                self.runtime_prepare_event_builder,
                kernel_plan,
                self.materialize_timeline_details,
            )
        started = time.perf_counter()
        plan_structure_runtime_graph_segments = None
        if self.host_launch_profiles is not None:
            if runtime_graph_mode is None:
                raise ValueError(
                    "Host launch timing requires an explicit runtime Graph mode")
            if runtime_graph_segments is None:
                piecewise_eager_kinds = (
                    self.runtime_inputs.graph_execution
                    .piecewise_eager_workload_kinds
                )
                plan_hardware_ids = {
                    call.kernel_spec.hardware_id
                    for call in kernel_plan.calls
                }
                if len(plan_hardware_ids) != 1:
                    raise ValueError(
                        "runtime Graph segment production requires one hardware")
                host_profile = self.host_launch_profiles[
                    next(iter(plan_hardware_ids))]
                sync_before_replay = (
                    host_profile.runtime_name == "CANN"
                    and host_profile.framework_name == "torch_npu")
                if self.materialize_timeline_details:
                    runtime_graph_segments = build_runtime_graph_segments(
                        workload,
                        kernel_plan,
                        runtime_graph_mode,
                        piecewise_eager_kinds,
                        sync_before_replay=sync_before_replay,
                    )
                else:
                    # The summary builder works in the plan-structure
                    # identifier space, so take the segments there directly
                    # instead of translating them into plan space and back.
                    segments, prefix = (
                        build_plan_structure_runtime_graph_segments(
                            workload,
                            kernel_plan,
                            runtime_graph_mode,
                            piecewise_eager_kinds,
                            sync_before_replay=sync_before_replay,
                        ))
                    if prefix is None:
                        runtime_graph_segments = segments
                    else:
                        plan_structure_runtime_graph_segments = (
                            segments, prefix)
        execution_timeline = build_execution_timeline(
            kernel_plan,
            predictions,
            origin_time_s=origin_time_s,
            shared_resource_calendar=shared_resource_calendar,
            host_launch_profiles=self.host_launch_profiles,
            host_runtime_prepare_profiles=(
                self.host_runtime_prepare_profiles),
            runtime_prepare_events=runtime_prepare_events,
            runtime_graph_mode=(
                runtime_graph_mode
                if self.host_launch_profiles is not None else None),
            runtime_graph_segments=runtime_graph_segments,
            materialize_timeline_details=self.materialize_timeline_details,
            normalized_runtime_prepare_events=(
                normalized_runtime_prepare_events),
            runtime_prepare_events_token=runtime_prepare_events_token,
            plan_structure_runtime_graph_segments=(
                plan_structure_runtime_graph_segments),
        )
        self._phase_wall_time_s["timeline_construction"] += (
            time.perf_counter() - started)
        self._executed_workloads += 1
        return ServingExecutionComponents(
            workload,
            kernel_plan,
            predictions,
            execution_timeline,
        )

    def cache_stats(self) -> dict[str, object]:
        if self.execution_mode == "reference":
            return {
                "execution_mode": "reference",
                "block_construction": None,
                "expert_routing": None,
                "kernel_lowering": None,
                "kernel_prediction": None,
            }
        return {
            "execution_mode": "accelerated",
            "block_construction": self._memo_stats(self._block_memo),
            "expert_routing": self._memo_stats(self._routing_memo),
            "kernel_lowering": self._memo_stats(self._lowering_memo),
            "kernel_prediction": self._memo_stats(self._prediction_memo),
        }

    def performance_stats(self) -> dict[str, object]:
        return {
            "executed_workloads": self._executed_workloads,
            "phase_wall_time_s": dict(self._phase_wall_time_s),
            "parallel_worker_count": 0,
            "parallel_step_count": 0,
            "parallel_execution_wall_time_s": 0.0,
            "parallel_worker_aggregate_phase_s": {
                name: 0.0 for name in self._phase_wall_time_s
            },
        }

    @staticmethod
    def _memo_stats(memo: object) -> dict[str, int]:
        if memo is None:
            raise RuntimeError("accelerated Serving execution lost a memo")
        return {
            "entries": memo.entry_count,
            "hits": memo.hit_count,
            "misses": memo.miss_count,
        }
