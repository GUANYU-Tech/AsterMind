"""Run-local Serving workload construction, lowering, and prediction."""

from dataclasses import dataclass, replace
from collections import OrderedDict
import time
import gc
from contextlib import contextmanager
from threading import RLock
from typing import Callable, Mapping
from types import MappingProxyType

from src.serving.deployment.deployment_plan import DeploymentPlan
from src.serving.kernel_lowering.execution_timeline import (
    ExecutionResourceCalendar,
    _common_id_prefix,
    _normalize_id,
    _plan_structure_key,
    build_execution_timeline,
    build_plan_structure_runtime_graph_segments,
    build_runtime_graph_segments,
    call_host_lanes,
    iter_call_occurrences,
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
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.graph_padding import VLLMDenseGraphInputs
from src.serving.workload.runtime_inputs import RuntimeInputs, V32PaddedMTPInputs
from src.serving.workload.workload_graph import WorkloadGraph
from src.kernel.predictors.common import graph_replayed_launch
from src.serving.host_timing import (
    HostLaunchProfile,
    HostRuntimePrepareProfile,
    RuntimePrepareEvent,
    RuntimeGraphSegmentDescriptor,
)


_METADATA_COLLECTION_LOCK = RLock()
_METADATA_COLLECTION_USERS = 0
_METADATA_COLLECTION_WAS_ENABLED = False


@contextmanager
def _metadata_collection_scope():
    """Defer cyclic scans while a transient metadata step is being built.

    CPython reference counting still releases tensor handles at their exact
    source lifetime boundaries. Automatic cycle collection resumes on scope
    exit, including exceptions; nested/threaded steps share the outer scope.
    """
    global _METADATA_COLLECTION_USERS, _METADATA_COLLECTION_WAS_ENABLED
    with _METADATA_COLLECTION_LOCK:
        if _METADATA_COLLECTION_USERS == 0:
            _METADATA_COLLECTION_WAS_ENABLED = gc.isenabled()
            if _METADATA_COLLECTION_WAS_ENABLED:
                gc.disable()
        _METADATA_COLLECTION_USERS += 1
    try:
        yield
    finally:
        with _METADATA_COLLECTION_LOCK:
            _METADATA_COLLECTION_USERS -= 1
            if _METADATA_COLLECTION_USERS == 0 and _METADATA_COLLECTION_WAS_ENABLED:
                gc.enable()


@dataclass(frozen=True)
class ServingExecutionComponents:
    workload: object
    kernel_plan: object
    predictions: Mapping[str, object]
    execution_timeline: object
    source_activation_memory: Mapping[str, object] | None = None


def bind_model_source_input_frames(runtime_inputs, frames):
    """Freeze explicit step scenarios at the common Serving boundary."""
    if frames is None:
        return None
    if (not isinstance(frames, Mapping) or not frames
            or any(not isinstance(step_id, str) or not step_id for step_id in frames)):
        raise ValueError("per-step source binding requires nonempty named explicit frames")
    if runtime_inputs.mtp_draft_sampling_source in ("vllm_0_30_legacy_full_logits_greedy",
            "vllm_0_27_1_eager_full_logits_greedy"):
        valid = all(isinstance(state, V32PaddedMTPInputs) for state in frames.values())
    elif (runtime_inputs.symbolic_forward_source == "cuda_eager_inplace"
            and runtime_inputs.model_execution_version == "vllm_0_30_0" and runtime_inputs.graph_execution.enabled):
        # None explicitly declares an eager step; it does not reuse a tail
        # from an earlier captured step or skip a missing frame key.
        valid = (any(isinstance(state, VLLMDenseGraphInputs) for state in frames.values())
            and all(state is None or isinstance(state, VLLMDenseGraphInputs) for state in frames.values()))
    else:
        valid = False
    if not valid:
        raise ValueError("per-step source binding requires matching padded MTP or captured dense source and typed explicit frames")
    return MappingProxyType(dict(frames))


def execute_heterogeneous_flash_request(model, backend, *, input_ids, output_tokens, fpga_cards=8, root_seed=0):
    """Drive a live virtual PD request with the existing real CPU Scheduler.

    The supplied colocated 0.30 Scheduler is a length/control-plane driver,
    not the physical TP8/DP8 PD allocator. Its declared dummy feedback is
    retained. Event progression uses a fixed clock because no durations are
    supplied; callers must not interpret it as zero latency or measured QPS.
    This function owns and closes the supplied backend, including on failure.
    """
    from contextlib import closing
    from src.serving.request_gen.request_stream import RequestSpec
    from src.serving.workload.traced_heterogeneous_request import FlashHeterogeneousRequest

    with closing(backend):
        runtime = backend.runtime_info
        if (runtime["version"] != "0.30.0" or runtime["model_executed"] is not False
                or runtime["weights_loaded"] is not False or runtime["runtime_acceptance"] is not False
                or backend.config.role != "colocated" or backend.config.max_requests != 1
                or backend.config.scheduler_pipeline_depth != 1):
            raise ValueError("heterogeneous functional request requires the explicit CPU 0.30 single-request logical driver")
        program = FlashHeterogeneousRequest(model, input_ids=input_ids, output_tokens=output_tokens,
            fpga_cards=fpga_cards, root_seed=root_seed)
        backend.add_request(RequestSpec(0.0, "one", len(input_ids), output_tokens))
        steps, emitted, finished = [], 0, []
        while backend.has_work():
            if len(steps) >= output_tokens:
                raise RuntimeError("Scheduler exceeded the non-MTP single-request step bound")
            step = backend.plan_step(0.0)
            if step is None or len(step.schedule.requests) != 1:
                raise RuntimeError("real Scheduler did not produce the single admitted request")
            request, = step.schedule.requests
            if (request.request_id != "one" or step.embedding_token_source != "vllm_length_only_simulation"
                    or request.mtp_proposer_mode != "disabled" or step.generated_tokens["one"] != 1):
                raise ValueError("functional heterogeneous scope requires explicit length-only non-MTP feedback")
            if not steps:
                if (request.phase != "prefill" or request.computed_tokens != 0
                        or request.scheduled_tokens != len(input_ids) or step.embedding_token_ids != input_ids):
                    raise ValueError("Scheduler prompt IDs or query extent disagree with the virtual request")
                program.prefill()
            else:
                if (request.phase != "decode" or request.scheduled_tokens != 1
                        or request.computed_tokens != len(input_ids) + emitted - 1):
                    raise ValueError("Scheduler decode is not the consecutive live cache/token step")
                if program.decode_state is None:
                    program.handoff()
                program.decode()
            if program.emitted_tokens != emitted + 1:
                raise RuntimeError("virtual model output and Scheduler emission counts disagree")
            backend.complete_step(step, 0.0)
            emitted += step.generated_tokens["one"]
            finished.extend(step.finished_request_ids)
            steps.append(step)
        if emitted != output_tokens or finished != ["one"]:
            raise RuntimeError("real Scheduler did not finish the complete requested output length")
        return program, tuple(steps), runtime


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


def _source_capture_host_segments(plan, regions, *, sync_before_replay):
    """Map executed source boundaries to existing Host replay descriptors."""
    # Repeats retain actual source node IDs for every layer. Map these to
    # template-call occurrence keys, which are the timeline's execution ABI;
    # never infer later capture boundaries from only the first layer.
    calls = {call.call_id: call for call in plan.calls}
    repeats = {name: repeat for repeat in plan.block_repeats for name in repeat.call_ids}
    source_maps = {}
    for repeat in plan.block_repeats:
        if not repeat.trace_node_ids:
            raise NotImplementedError("source capture Host timing requires every repeated source occurrence")
        source_maps[repeat.block_id] = dict(repeat.trace_node_ids)

    def occurrences():
        for call_id, index in iter_call_occurrences(plan):
            call = calls[call_id]
            if call_id not in repeats:
                yield call, index, call.source_node_ids
            else:
                sources = source_maps[repeats[call_id].block_id]
                yield call, index, tuple(sources[node][index] for node in call.source_node_ids)

    owners, members, lanes = {}, {}, {}
    for pool, rank, identifier, nodes in regions:
        key = (pool, rank, identifier)
        if key in members or not nodes:
            raise ValueError("source capture Host regions require unique nonempty source segments")
        members[key] = []
        lanes[key] = (pool, rank)
        for node in nodes:
            lane_node = (pool, rank, node)
            if lane_node in owners:
                raise ValueError("source capture Host regions overlap on a source node")
            owners[lane_node] = key
    covered, total = 0, 0
    for call, occurrence, source_nodes in occurrences():
        for pool, rank in call_host_lanes(call):
            total += 1
            labels = {owners[(pool, rank, node)] if (pool, rank, node) in owners else None
                for node in source_nodes}
            if len(labels) != 1:
                raise NotImplementedError("lowered fusion crosses a source capture/eager boundary")
            label, = labels
            if label is not None:
                members[label].append((call.call_id, occurrence))
                covered += 1
    if not covered or any(not calls for calls in members.values()):
        raise ValueError("source capture Host segment lost every lowered call occurrence")
    descriptors = tuple(RuntimeGraphSegmentDescriptor(key[2], *lanes[key], tuple(calls), sync_before_replay)
        for key, calls in members.items())
    # FULL/PIECEWISE here describes whole-step Host coverage; the source
    # capture's FULL mode and user GraphExecutionConfig remain unchanged.
    return descriptors, "FULL" if covered == total else "PIECEWISE"


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
        model_source_inputs_by_step: Mapping[str, V32PaddedMTPInputs | VLLMDenseGraphInputs | None] | None = None,
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
        from src.serving.workload.traced_models import bind_ordinary_forward_source
        runtime_inputs = bind_ordinary_forward_source(model, deployment, runtime_inputs)
        if runtime_inputs.symbolic_forward_source is None:
            raise ValueError("Serving requires a bound virtual model source and its declared runtime inputs")
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
        self.model_source_inputs_by_step = bind_model_source_input_frames(runtime_inputs, model_source_inputs_by_step)
        self._source_workspace_capacity_by_pool = {}
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
        self._source_capture_ports = {}
        from src.serving.workload.traced_forward import SourceProgramCache
        self._source_program_cache = SourceProgramCache()
        self._recorded_source_program_bound = False
        accelerated = execution_mode == "accelerated"
        if accelerated:
            self._lowering_memo = KernelLoweringMemo()
            self._prediction_memo = KernelPredictionMemo()
        else:
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

    @_metadata_collection_scope()
    def execute_step(
        self,
        schedule: ScheduleStep,
        *,
        kv_cache_events: tuple[KVCacheEvent, ...] = (),
        origin_time_s: float = 0.0,
        shared_resource_calendar: ExecutionResourceCalendar | None = None,
        runtime_prepare_events: Mapping[
            tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]] | None = None,
        embedding_token_ids: tuple[int, ...] | None = None,
        model_source_inputs: object | None = None,
        source_prompt_lengths: tuple[int, ...] | None = None,
    ) -> ServingExecutionComponents:
        # A receipt belongs only to a successfully completed current step.
        # Reset before validation so failures never expose a preceding bind.
        self._recorded_source_program_bound = False
        # Token ownership is per scheduled step, unlike the run's immutable
        # execution/source settings. Never retain a preceding step's IDs.
        if self.model_source_inputs_by_step is not None:
            if model_source_inputs is not None:
                raise ValueError("model-source frames require one owner, not both backend transport and execution mapping")
            if schedule.step_id not in self.model_source_inputs_by_step:
                raise ValueError("scheduled step is absent from explicitly bound model-source inputs")
            model_source_inputs = self.model_source_inputs_by_step[schedule.step_id]
        framed_mtp = (self.runtime_inputs.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy"
            or (self.runtime_inputs.mtp_draft_sampling_source == "vllm_0_27_1_eager_full_logits_greedy"
                and isinstance(self.runtime_inputs.model_source_inputs, V32PaddedMTPInputs)))
        if framed_mtp:
            if not isinstance(model_source_inputs, V32PaddedMTPInputs) or embedding_token_ids is None:
                raise ValueError("padded MTP execution requires explicit per-step source state and current target token IDs")
            if (model_source_inputs.request_ids != tuple(request.request_id for request in schedule.requests)
                    or len(model_source_inputs.draft_input_token_ids[0]) != schedule.token_count
                    or any(len(ids) != len(schedule.requests) for ids in model_source_inputs.draft_input_token_ids[1:])
                    or {request.mtp_proposer_tokens for request in schedule.requests
                        if request.mtp_proposer_mode == "padded"} != {len(model_source_inputs.draft_input_token_ids)}):
                raise ValueError("padded MTP per-step source state disagrees with scheduled requests, padded rows or draft depth")
        step_inputs = {}
        if source_prompt_lengths is not None:
            if (len(source_prompt_lengths) != len(schedule.requests)
                    or any(type(length) is not int or length < 1 for length in source_prompt_lengths)):
                raise ValueError("prompt lengths must cover the scheduled requests in order")
            if self.model.model_type == "deepseek_v4":
                step_inputs["v4_source_prompt_lengths"] = source_prompt_lengths
            elif self.model.model_type in ("kimi_k2", "glm_moe_dsa"):
                step_inputs["symbolic_source_prompt_lengths"] = source_prompt_lengths
        if embedding_token_ids is not None:
            step_inputs["embedding_token_ids"] = embedding_token_ids
        if model_source_inputs is not None or self.runtime_inputs.model_source_inputs is not None:
            step_inputs["model_source_inputs"] = model_source_inputs
        runtime = (replace(self.runtime_inputs, **step_inputs) if step_inputs else self.runtime_inputs)
        if runtime.symbolic_forward_source == "cuda_eager_inplace" and runtime.target_sampling_source is None:
            raise NotImplementedError("static model-only source cannot execute complete Serving step; an explicit sampler is required")
        if schedule.pool_name in self._source_workspace_capacity_by_pool:
            runtime = replace(runtime, symbolic_workspace_initial_capacity_bytes=
                self._source_workspace_capacity_by_pool[schedule.pool_name])
        source_storage, source_graph, source_capture_active = None, None, False
        source_memory_context = None
        source_capture_mode = "NONE"
        source_capture_regions = []
        recorded_source_program_bound = False

        def observe_source(source, graph):
            nonlocal source_storage, source_graph, source_capture_active, source_capture_mode
            nonlocal recorded_source_program_bound, source_memory_context
            from src.serving.kv_cache.activation_memory import symbolic_forward_storage_report
            from src.serving.workload.traced_forward import ForwardTrace, ForwardCollectiveGroup
            if source_storage is not None:
                raise RuntimeError("symbolic step executed its forward more than once")
            # Only bind() installs a recorded graph on these concrete sources.
            # A cache lookup can still return a program whose bind is rejected;
            # wrappers and freshly executed unbound sources are not receipts.
            recorded_source_program_bound = (
                isinstance(source, (ForwardTrace, ForwardCollectiveGroup))
                and source._recorded_graph is not None and source.graph is graph)
            source_storage = symbolic_forward_storage_report(source, graph,
                source_profile=runtime.symbolic_forward_source)
            source_graph = graph
            from src.serving.kernel_lowering.memory_bindings import snapshot_trace_memory
            source_memory_context = snapshot_trace_memory(source, graph)
            if runtime.graph_execution.enabled:
                from src.serving.kv_cache.activation_memory import source_capture_activation_peak
                traces = (source,) if isinstance(source, ForwardTrace) else source.traces
                if self.host_launch_profiles is not None:
                    source_orders = source.operator_order_by_rank
                for trace in traces:
                    captured_regions = []
                    for capture in trace.source_captures:
                        source_capture_active = True
                        if source_capture_mode != "NONE" and source_capture_mode != capture["runtime_mode"]:
                            raise RuntimeError("source ranks disagree on capture runtime mode")
                        source_capture_mode = capture["runtime_mode"]
                        if capture["end_event"] is None:
                            raise RuntimeError("source capture lost its actual forward boundary")
                        key = (schedule.pool_name, trace.rank, capture["key"])
                        if capture["source_stage"] is None:
                            named_ports = {**capture["inputs"], "hidden_states": capture["output"]}
                        else:
                            named_ports = {**{"input:" + role: port for role, port in capture["inputs"].items()},
                                **{"output:" + role: port for role, port in capture["outputs"].items()}}
                        ports = {role: {name: value for name, value in port.items() if name != "allocation_id"}
                            for role, port in named_ports.items()}
                        replay = key in self._source_capture_ports
                        if replay and self._source_capture_ports[key] != ports:
                            raise RuntimeError("source capture changed static input/output identity or typed extent")
                        self._source_capture_ports[key] = ports
                        backing_bytes = {}
                        for port in named_ports.values():
                            identity = port["allocation_id"]
                            if identity in backing_bytes and backing_bytes[identity] != port["bytes"]:
                                raise RuntimeError("captured aliases disagree on their backing capacity")
                            backing_bytes[identity] = port["bytes"]
                        if self.host_launch_profiles is not None:
                            start = capture["node_start"]
                            local_nodes = capture["model_node_ids"]
                            canonical_nodes = source_orders[trace.rank][start:start + len(local_nodes)]
                            if len(canonical_nodes) != len(local_nodes):
                                raise RuntimeError("source capture lost its canonical operator-order boundary")
                            canonical = dict(zip(local_nodes, canonical_nodes, strict=True))
                            for index, nodes in enumerate(capture["source_segments"]):
                                segment_nodes = tuple(canonical[node] for node in nodes)
                                segment_pools = {graph.node(node).pool_name for node in segment_nodes}
                                if len(segment_pools) != 1:
                                    raise ValueError("a local source capture segment must belong to exactly one worker pool")
                                source_capture_regions.append((next(iter(segment_pools)), trace.rank,
                                    f"{capture['key']}:source-segment-{index}", segment_nodes))
                        peak, live = source_capture_activation_peak(trace, capture)
                        captured_regions.append({
                            **capture, "logical_replay": replay, "static_ports": ports,
                            "configured_capture_sizes": runtime.graph_execution.capture_sizes,
                            "configured_capture_sizes_unit": "model_input_token_rows",
                            "static_port_bytes": sum(backing_bytes.values()),
                            "known_capture_activation_peak_bytes": peak,
                            "capture_peak_live_allocation_ids": live,
                            "capture_peak_scope": "source_region_including_eager_regions_not_graph_allocator_pool",
                            "scope": "same_source_forward_not_compiler_pool_reservation_or_GPU_replay"})
                    if len(captured_regions) == 1:
                        source_storage["ranks"][str(trace.rank)]["source_capture"] = captured_regions[0]
                    elif captured_regions:
                        source_storage["ranks"][str(trace.rank)]["source_captures"] = tuple(captured_regions)

        started = time.perf_counter()
        from src.serving.workload.traced_models import build_symbolic_workload
        workload = build_symbolic_workload(schedule, self.model, self.deployment,
            launch_policy=self.launch_policy, kv_cache_events=kv_cache_events,
            runtime_inputs=runtime, hardware_by_id=self.hardware_by_id, trace_observer=observe_source,
            source_program_cache=self._source_program_cache)
        if source_graph is not workload or source_storage is None:
            raise RuntimeError("symbolic step lost the storage facts of its actual forward graph")
        self._phase_wall_time_s["workload_construction"] += (
            time.perf_counter() - started)
        components = self.execute_workload(
            workload,
            origin_time_s=origin_time_s,
            shared_resource_calendar=shared_resource_calendar,
            runtime_graph_mode=(source_capture_mode
                if runtime.graph_execution.enabled
                else _runtime_graph_mode(schedule, self.runtime_inputs.graph_execution)),
            runtime_prepare_events=runtime_prepare_events,
            recorded_source_bound=recorded_source_program_bound,
            source_memory_context=source_memory_context,
            source_capture_regions=(tuple(source_capture_regions)
                if source_capture_active and self.host_launch_profiles is not None else None),
        )
        if runtime.symbolic_workspace_initial_capacity_bytes is not None:
            # The synchronous source has no surviving external arena aliases
            # at a forward return. Rebind its manager's final capacity on the
            # next trace; keep only scalar state, not an old graph or tensor.
            pool = self.deployment.worker_pools[schedule.pool_name]
            capacities = tuple(source_storage["ranks"][str(rank)]["source_workspace_capacity_bytes"]["worker.workspace"]
                for rank in pool.ranks)
            if any(final < initial for initial, final in zip(
                    runtime.symbolic_workspace_initial_capacity_bytes, capacities, strict=True)):
                raise RuntimeError("source workspace capacity cannot shrink across forward calls")
            self._source_workspace_capacity_by_pool[schedule.pool_name] = capacities
        result = (components if source_storage is None else
            replace(components, source_activation_memory={**source_storage,
                **({"host_capture_timing": {"source_runtime_mode": source_capture_mode,
                    "step_host_coverage_mode": components.execution_timeline.runtime_graph_mode,
                    "scope": "source capture regions replay; every uncovered call remains eager; not compiler/provider acceptance"}}
                    if source_capture_active and self.host_launch_profiles is not None else {})}))
        self._recorded_source_program_bound = recorded_source_program_bound
        return result

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
        source_capture_regions: tuple | None = None,
        require_timeline_details: bool = False,
        recorded_source_bound: bool = False,
        source_memory_context=None,
    ) -> ServingExecutionComponents:
        if type(recorded_source_bound) is not bool:
            raise TypeError("recorded source binding flag must be boolean")
        if not isinstance(require_timeline_details, bool):
            raise TypeError("require_timeline_details must be boolean")
        materialize_details = self.materialize_timeline_details or require_timeline_details
        if not isinstance(workload, WorkloadGraph):
            raise TypeError(
                "Serving execution workload must be WorkloadGraph")
        if source_capture_regions is not None and (self.host_launch_profiles is None or runtime_graph_segments is not None
                or runtime_graph_mode not in ("FULL", "PIECEWISE")):
            raise ValueError("source capture Host regions require their own explicit captured-mode Host segment binding")
        started = time.perf_counter()
        # This complete raw-source MoE domain need not have a recorded-prefix
        # cache hit. lower_workload still validates every EP port and routing
        # record, then proves equal physical contracts and the exact recurrence
        # DAG before folding; nonuniform layers retain their actual calls.
        kernel_plan = lower_workload(
            workload,
            self.deployment,
            self.hardware_by_id,
            self._lowering_memo,
            compact_source_layers=(self._lowering_memo is not None and not materialize_details
                and (self.model.num_experts == 0 or recorded_source_bound
                    or (self.model.model_type == "qwen3_moe" and self.model.num_experts == 128
                        and self.model.compute_dtype == self.model.weight_dtype == "bf16"
                        and self.model.num_hidden_layers == 48 and self.model.hidden_size == 2048
                        and self.model.moe_intermediate_size == 768
                        and self.runtime_inputs.model_execution_version == "vllm_0_11_0"
                        and self.runtime_inputs.expert_routing.source_kind == "profile_observation"
                        and self.runtime_inputs.expert_routing.distribution == "model_profile_layer_mean_v1"
                        and len(self.deployment.worker_pools) == 1
                        and all(pool.hardware_id == "nvidia_h200" and len(pool.ranks) == 2
                            and pool.parallel_plan.config.tensor_parallel_size == 2
                            and pool.parallel_plan.config.enable_expert_parallel
                            and pool.parallel_plan.config.expert_parallel_size == 2
                            and pool.parallel_plan.config.is_rank_local
                            and all(value == 1 for value in (pool.parallel_plan.config.pipeline_parallel_size,
                                pool.parallel_plan.config.sequence_parallel_size, pool.parallel_plan.config.context_parallel_size,
                                pool.parallel_plan.config.data_parallel_size))
                            for pool in self.deployment.worker_pools.values())))
                and self.deployment.mode == "colocated"),
        )
        if source_memory_context is not None:
            from src.serving.kernel_lowering.memory_bindings import bind_trace_memory
            kernel_plan = bind_trace_memory(kernel_plan, source_memory_context)
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
                materialize_details,
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
                if source_capture_regions is not None:
                    runtime_graph_segments, runtime_graph_mode = _source_capture_host_segments(
                        kernel_plan, source_capture_regions, sync_before_replay=sync_before_replay)
                elif materialize_details:
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
            materialize_timeline_details=materialize_details,
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
                "kernel_lowering": None,
                "kernel_prediction": None,
            }
        return {
            "execution_mode": "accelerated",
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
