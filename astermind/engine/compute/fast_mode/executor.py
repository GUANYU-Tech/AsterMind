"""Adapt Scheduler output to the batch predictor, on the existing Cell clock."""

import json
from pathlib import Path
from itertools import count
from time import perf_counter

from astermind.engine.clock import Time
from astermind.engine.comm.frame_surrogate import NoRouteError
from astermind.engine.comm.native_intra import native_enabled
from astermind.engine.compute.batch_executor import RuntimeBatchExecutor
from astermind.common.hardware_capabilities import FAST_MODE_HARDWARE, PORTABLE_BATCH_HARDWARE
from astermind.engine.timeline import nanoseconds
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.workload.model_config import GLM_MOE_DSA_CONFIG_SHA256

from .admission import supports_portable_fp8_reference
from .communication import RankProgram
from .comm_direct import DirectGroups
from .comm_reuse import BoundCommunicationPlan, UnsupportedFormation
from .comm_pricing import RankCommunicationPricing
from .frame_forecast import forecast_routes
from .hardware import Hardware
from .predictor import Predictor, Row
from .quote import BatchCommunication, mixed_runtime_waves
from .static_cell import static_rank_path_window
from .v4_small_gemm import SmallFp8Gemm


class FastModeExecutor(RuntimeBatchExecutor):
    """Price one scheduled batch, then release it on the existing timeline."""

    compute_mode = "fast_mode"
    compute_timing_reference = None
    portable_hardware = False
    link_contention = False
    runtime_transport_options = {}
    static_rank_paths = None

    def configure(self, execution, config):

        pool = execution.deployment.worker_pools["main"]
        parallel = pool.parallel_plan.config
        if (pool.hardware_id not in FAST_MODE_HARDWARE or config.sequence_parallel_size != 1
                or config.deployment != "colocated" or config.host_launch_profile_file):
            raise ValueError("fast_mode requires a supported colocated hardware profile, SP1, and device-only timing")
        if native_enabled(self.comm.policy, self.cell_id):
            raise ValueError("fast_mode requires surrogate communication")
        model_path = config.root / config.model_file
        model_name = Path(config.model_file).stem
        model = json.loads(model_path.read_text(encoding="utf-8-sig"))
        ranges = pipeline_layer_ranges_for_model(execution.model, parallel.pipeline_parallel_size)
        widened_fp8 = supports_portable_fp8_reference(config, execution.model, parallel)
        hardware_spec = execution.hardware_by_id[pool.hardware_id]
        hardware = Hardware.from_hardware_spec(
            hardware_spec, compute_dtype=execution.model.compute_dtype,
            widen_fp8_to_bf16=widened_fp8,
        ).with_roofline(pool.hardware_id, config.fast_roofline)
        self.link_contention = config.fast_link_contention
        self.compute_timing_reference = "fp8_widened_bf16_reference_roofline" if widened_fp8 else None
        v4_small_gemm = (
            model_name in ("deepseek-v4-pro", "deepseek-v4-flash") and pool.hardware_id == "nvidia_h200"
            and parallel.data_parallel_size == parallel.expert_parallel_size == 1
            and execution.runtime_inputs.symbolic_forward_source == "vllm_0_30_v4_sync_fp8_ds_mla"
            and hardware_spec.block_fp8_gemm_tma_ring_service is not None
            and not execution.runtime_inputs.graph_execution.enabled
        )
        if v4_small_gemm:
            self.compute_timing_reference = "h200_v4_fp8_single_wave"
        grouped_glm_ffn = (
            model_name == "glm-5.2"
            and execution.model.config_sha256 == GLM_MOE_DSA_CONFIG_SHA256
            and pool.hardware_id == "nvidia_h200"
            and parallel.tensor_parallel_size > 1 and parallel.pipeline_parallel_size > 1
            and parallel.data_parallel_size == parallel.expert_parallel_size == 1
            and not parallel.enable_expert_parallel
        )
        if grouped_glm_ffn:
            self.compute_timing_reference = "glm52_grouped_shared_routed_ffn"
        predictor = Predictor(model_name, model, parallel.tensor_parallel_size, parallel.data_parallel_size,
                              parallel.expert_parallel_size, ranges, hardware=hardware,
                              small_fp8_gemm=SmallFp8Gemm(hardware_spec, hardware) if v4_small_gemm else None,
                              glm_dsa_bf16_timing=config.fast_glm_dsa_score == "bf16_timing_proxy",
                              grouped_glm_ffn=grouped_glm_ffn)
        devices = tuple(self._device("main", rank) for rank in range(len(pool.ranks)))
        program = RankProgram(predictor, ranges, len(devices))
        if pool.hardware_id in PORTABLE_BATCH_HARDWARE:
            self._validate_portable_placement(devices)
        self.predictor = predictor
        self.pricing = RankCommunicationPricing(self.comm, program, devices, self.cell_id)
        self.portable_hardware = pool.hardware_id in PORTABLE_BATCH_HARDWARE
        self.fixed = BoundCommunicationPlan(
            self.pricing, direct_factory=DirectGroups if pool.hardware_id == "nvidia_h200" else None,
        )
        self.prepared_frame = None
        self.bound_available = False
        from astermind.engine.comm.event_runtime import RuntimeCommunication
        from astermind.engine.comm.target_network import TargetNetworkFrames
        self.runtime_transport_options = {}
        if type(self.comm) is RuntimeCommunication and type(self.comm.network) is TargetNetworkFrames:
            from astermind.engine.comm.intra_hierarchy import ScaleUpPairIndex
            topology = self.comm.network.network.topology
            index = getattr(self.comm, '_fast_scale_up_index', None)
            if index is None:
                index = self.comm._fast_scale_up_index = ScaleUpPairIndex(topology)
            self.runtime_transport_options = {'fast_scale_up_index': index}
            if not (self.link_contention or self.comm.shared_bandwidth):
                self.static_rank_paths = static_rank_path_window(self.comm.network, devices, program)

    def _has_static_rank_paths(self, start, end):
        if self.static_rank_paths is None:
            return False
        ready, horizon, revision = self.static_rank_paths
        net = self.comm.network.host.net
        return (ready <= start <= end <= horizon and revision == net.revision
                and not (net.disabled_device_links or net.failed_device_links or net.active_faults))

    def _validate_portable_placement(self, devices):
        frame = self.comm.network.frame_at(self.loop.now_s)
        nodes = frame.node_states
        participants = {}
        for gpu in devices:
            owner = nodes[gpu].satellite_id
            participants.setdefault(owner, set()).add(gpu)
        for ranks in participants.values():
            for gpu in ranks:
                hierarchy = nodes[gpu].attributes.get("intra_hierarchy")
                if hierarchy and not any(ranks <= set(group) for group in hierarchy["su_groups"]):
                    raise ValueError("non-H200 fast_mode requires one scale-up domain per satellite")

    def prepare_batch(self, step, source_prompt_lengths):
        rows = tuple(Row(row.phase, row.scheduled_tokens, row.computed_tokens, row.data_parallel_rank,
                         int(row.phase == "decode" or row.computed_tokens + row.scheduled_tokens >= length))
                     for row, length in zip(step.schedule.requests, source_prompt_lengths, strict=True))
        return self.predictor.estimate(rows)

    def _price(self, components):
        routes = self.comm._refresh()
        if routes is not self.prepared_frame and not (self.comm.active or self.comm.tasks):
            self.prepared_frame = routes
            try:
                self.fixed.prepare()
            except UnsupportedFormation:
                self.bound_available = False
            else:
                self.bound_available = True
        if self.bound_available:
            before = self.fixed.stats["ordinary_fallback"]
            try:
                seconds, deadline = self.fixed.price(components)
            except UnsupportedFormation:
                pass
            else:
                used_original = (before != self.fixed.stats["ordinary_fallback"]
                                 or self.fixed.direct is None and any(self.pricing.program.experts))
                return seconds, deadline, "partial_reuse" if used_original else "reused"
        before = self.pricing.frame_quotes.hits
        seconds, deadline = self.pricing.price(components)
        return seconds, deadline, "reused" if self.pricing.frame_quotes.hits != before else "isolated"

    def _start_batch(self, cell_id, step_id, components, execution, on_complete, *,
                     token_count, pool_name, on_next_batch, replica_tokens, state):
        if (pool_name != "main" or components.replica_tokens != replica_tokens
                or sum(components.replica_tokens) != token_count):
            raise ValueError("fast_mode requires the Scheduler's actual per-DP token counts")
        compute_s = sum(components.stage_compute_s)
        start = self.loop.now_s
        crossing = None
        fallback = None
        quote_started = perf_counter()
        if self.link_contention:
            communication_s, end, pricing = 0., None, "runtime"
        else:
            try:
                communication_s, frame_end, pricing = self._price(components)
            except NoRouteError:
                communication_s, end, pricing = 0., None, "runtime"
                fallback = "initial_route_unavailable"
            else:
                end = start + Time(compute_s + communication_s)
                if frame_end is not None and end >= frame_end:
                    if self._has_static_rank_paths(start, end):
                        crossing = "static_cell"
                    else:
                        fallback = self._crossing_reason(components, frame_end, end)
                        if fallback is None:
                            crossing = "certified"
        if fallback is not None:
            pricing, crossing = "runtime", "runtime"
        quote_wall_s = perf_counter() - quote_started
        self._active.add(pool_name)
        record = dict(cell_id=cell_id, pool_name=pool_name, step_id=step_id,
                      mode="fast_mode", communication_pricing=pricing,
                      compute_s=compute_s, communication_s=communication_s,
                      communication_quote_wall_s=quote_wall_s,
                      start_time_ns=nanoseconds(start), start_time_ticks=str(start.ticks),
                      native_communication_duration_charged_s=0., calls=[])
        if self.compute_timing_reference is not None:
            record["compute_timing_reference"] = self.compute_timing_reference
        state["record"] = record
        if self.link_contention:
            record["runtime_reason"] = "link_bandwidth_contention"
        if crossing is not None:
            record["cross_frame"] = crossing
        if fallback is not None:
            record["runtime_reason"] = fallback

        def complete():
            if not state["alive"]:
                return
            self._active.remove(pool_name)
            state["alive"] = False
            now = self.loop.now_s
            record.update(status="completed", end_time_ns=nanoseconds(now), end_time_ticks=str(now.ticks))
            self._publish_completed_batch(record, on_complete)
            if self._on_batch_boundary:
                self._on_batch_boundary()
            if on_next_batch:
                on_next_batch()

        if pricing != "runtime":
            self.loop.schedule(end, 1, "cell:fast-mode-complete", complete, owner_id=cell_id)
        else:
            self._run_runtime_phases(components, state, complete)

    def _crossing_reason(self, components, boundary, end):
        """Certify all actual operator paths against read-only projected Frames."""
        initial = BatchCommunication(self.comm, collect_paths=True)
        try:
            self.pricing.quote_batch(components, initial)
        except NoRouteError:
            return "initial_route_unavailable"
        if not initial.complete_paths:
            return "incomplete_path_evidence"
        while boundary is not None and boundary <= end:
            routes = forecast_routes(self.comm.network, boundary)
            if routes is None:
                return "frame_not_forecastable"
            next_boundary = routes.frame.valid_until_s
            if routes.frame.start_time_s > boundary or (next_boundary is not None and next_boundary <= boundary):
                raise ValueError("forecast does not cover its requested frame boundary")
            quote = BatchCommunication(self.comm, routes=routes, time=boundary, collect_paths=True)
            try:
                self.pricing.quote_batch(components, quote)
            except NoRouteError:
                return "path_unavailable_in_future_frame"
            if not quote.complete_paths or initial.signature != quote.signature:
                return "used_path_or_service_contract_changed"
            boundary = next_boundary
        return None

    def _run_runtime_phases(self, components, state, complete):
        """Run the original rank operations at their compact-model readiness times."""
        phases = iter(self.pricing.runtime_phases(components))
        identities = count()
        compute_ticks = 0
        record = state["record"]

        def advance():
            nonlocal compute_ticks
            if not state["alive"]:
                return
            for phase in phases:
                if isinstance(phase, Time):
                    if phase:
                        compute_ticks += phase.ticks
                        self.loop.schedule(self.loop.now_s + phase, 1, "cell:roofline-stage", guarded_advance,
                                           owner_id=self.cell_id)
                        return
                elif launch(phase):
                    return
            communication_ticks = self.loop.now_s.ticks - state["origin"].ticks - compute_ticks
            record.update(compute_elapsed_ticks=str(compute_ticks), communication_ticks=str(communication_ticks),
                          communication_s=float(Time.from_ticks(communication_ticks)))
            complete()

        def guarded_advance():
            try:
                advance()
            except Exception as error:
                if not state["alive"]:
                    raise
                self._abort_batch(state, error)

        def launch(branches):
            live = [(branch, first) for branch in map(iter, branches)
                    if (first := next(branch, None)) is not None]
            pending = len(live)

            def submit(branch, operator):
                if not state["alive"]:
                    return
                identity = f"{self.cell_id}/{state['pool']}/{state['step']}/roofline-{next(identities)}"

                def done(_):
                    nonlocal pending
                    state["tasks"].discard(identity)
                    if not state["alive"]:
                        return
                    try:
                        other = next(branch, None)
                    except Exception as error:
                        self._abort_batch(state, error)
                        return
                    if other is not None:
                        submit(branch, other)
                    else:
                        pending -= 1
                        if pending == 0:
                            guarded_advance()

                self._submit_runtime_operator(state, operator, done, identity)

            for branch, first in live:
                submit(branch, first)
            return bool(live)

        guarded_advance()

    def _submit_runtime_operator(self, state, operator, done, identity):
        waves = mixed_runtime_waves(self.comm, operator) if self.portable_hardware else None
        if waves is None:
            self._submit_batch_task(state, operator, done, identity, **self.runtime_transport_options)
            return

        pending_waves = iter(enumerate(wave for wave in waves if wave))

        def next_wave():
            if not state["alive"]:
                return
            current = next(pending_waves, None)
            if current is None:
                done(None)
                return
            index, messages = current
            pending = len(messages)

            def arrived(task_id):
                nonlocal pending
                state["tasks"].discard(task_id)
                pending -= 1
                if pending == 0:
                    self.loop.schedule(self.loop.now_s, 1, "cell:fast-mode-comm-wave",
                                       next_wave, owner_id=self.cell_id)

            for peer, message in enumerate(messages):
                if not state["alive"]:
                    return
                task_id = f"{identity}/wave-{index}/message-{peer}"
                source, target = message.participants
                routes = self.comm._refresh()
                nodes = routes.frame.node_states
                if nodes[source].satellite_id == nodes[target].satellite_id:
                    try:
                        routes.route(source, target, message.bytes)
                    except NoRouteError:
                        # A later wave can lose its local path after planning.
                        # Let the unchanged V1 transfer wait or retry.
                        record = state.get("record")
                        if record is not None:
                            record["local_route_waits"] = record.get("local_route_waits", 0) + 1
                        state["tasks"].add(task_id)
                        self.comm.submit_transfer(
                            task_id, source, target, message.bytes, lambda key=task_id: arrived(key),
                            lambda error: self._abort_batch(state, error),
                            **self.runtime_transport_options,
                        )
                        continue
                self._submit_batch_task(state, message, lambda _, key=task_id: arrived(key), task_id,
                                        **self.runtime_transport_options)

        next_wave()
