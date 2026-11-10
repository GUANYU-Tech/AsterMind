"""Shared physical network around the existing native Scheduler/Cell seam."""
from dataclasses import asdict
import time

from .bootstrap import activate


class Simulation:
    def __init__(self, config, *, midlayer_root=None, cell_factory=None, network_factory=None,
                 operator_comm_factory=None):
        startup = time.perf_counter()
        self.config = config
        self.root = activate(midlayer_root)
        from orbitserve.runtime.engine.cell_events import CellEventLoop
        self.loop = CellEventLoop()
        self.topology = config.topology()
        if network_factory is None:
            if config.backend == "flowsim":
                from network_sim.comm.runtime import CommunicationRuntime
                network_factory = CommunicationRuntime
            else:
                from network_sim.comm.ns3_runtime import NS3Runtime
                network_factory = lambda topology: NS3Runtime(topology, packet_bytes=config.packet_bytes)
        self.network = network_factory(self.topology)
        from orbitserve.runtime.engine.network_events import NetworkEventDriver
        self.transport = NetworkEventDriver(self.loop, self.network)
        self.executors = {}
        self.cells, self.requests = {}, {r.id: r for r in config.requests}
        self.state = {r.id: {"status": "pending", "arrival_ns": r.arrival_ns,
                            "received_tokens": 0, "first_token_received_ns": None,
                            "last_token_received_ns": None} for r in config.requests}
        self.completions, self.outputs, self.decisions = [], [], []
        self.request_state_events = []
        self._transport_serial = 0
        self._received_chunks = {r.id: set() for r in config.requests}
        self._cell_configs = {c.id: c for c in config.cells}
        from astermind.cell_storage import SessionHistoryStore
        from .migration import NetworkKVMigration
        self.history = SessionHistoryStore()
        self.migrations = NetworkKVMigration(self.transport,
            {c.id: {p.rank: p.device for p in c.placements} for c in config.cells})
        self._session_cells = {}
        try:
            from network_sim.comm.operator import OperatorCommunication
            from astermind.ArchPrism.execution.network_batch_executor import NetworkBatchExecutor
            from astermind.ArchPrism.execution.placement import RankPlacement
            self.operator_comm = (operator_comm_factory or OperatorCommunication)(self.topology)
            for c in config.cells:
                placements = tuple(RankPlacement(c.id, p.pool, p.rank,
                    self.topology.devices[p.device].satellite, p.device) for p in c.placements)
                executor = NetworkBatchExecutor(c.id, self.loop, self.operator_comm, placements,
                    on_batch_boundary=self.migrations.on_batch_boundary)
                self.executors[c.id] = executor
                if cell_factory is None:
                    cell = self._native_cell(c, executor)
                else:
                    cell = cell_factory(c, self.loop, self._completed, self._output, executor)
                self.cells[c.id] = cell
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        self.startup_wall_s = time.perf_counter() - startup

    def _native_cell(self, cell, executor):
        from astermind.ArchPrism.midlayer.runtime import MidlayerCellConfig
        from astermind.ArchPrism.scheduler.serving import ServingScheduler
        p = cell.parallel
        config = MidlayerCellConfig(self.root, model_file=cell.model, hardware_file=cell.hardware,
            hardware_id=cell.hardware_id, max_model_len=cell.max_model_len,
            max_batched_tokens=cell.max_batched_tokens, max_requests=cell.max_requests,
            num_kv_blocks=cell.num_kv_blocks, kv_block_size_tokens=cell.kv_block_size_tokens,
            scheduler_adapter=cell.scheduler_adapter, enable_prefix_caching=cell.prefix_caching,
            enable_expert_parallel=cell.expert_parallel_enabled, expert_routing_seed=cell.expert_routing_seed,
            host_launch_profile_file=cell.host_launch_profile, memory_policy_file=cell.memory_policy_file,
            watermark=cell.watermark, deployment=cell.deployment,
            tensor_parallel_size=p.tp, pipeline_parallel_size=p.pp)
        if cell.deployment == "pd_disagg":
            from astermind.pd_cell import PrefillDecodeCell
            return PrefillDecodeCell(cell.id, self.loop, config, executor,
                                     self._completed, self._output, self._cell_failure,
                                     submit_handoff=self.transport.submit_transport)
        return ServingScheduler.create(cell.id, self.loop, config, self._completed,
            on_output=self._output, on_failure=self._cell_failure,
            batch_executor=executor, history_store=self.history)

    def _cell_failure(self, failure):
        self._update_request(failure["request_id"], "cell_failed", status="failed", failure=failure)

    def _submit(self, source, target, size, operation, request_id, callback):
        self._transport_serial += 1
        self.transport.submit_transport(f"serving-{self._transport_serial}", source, target, size,
            operation, request_id, callback,
            lambda event: self._update_request(request_id, "transfer_failed",
                                               status="failed", failure=event.model_dump()))

    def _update_request(self, request_id, reason, **changes):
        state = self.state[request_id]
        previous = state["status"]
        state.update(changes)
        event = self.loop.events[-1]
        self.request_state_events.append(dict(request_id=request_id, reason=reason,
            time_ns=round(self.loop.now_s * 1e9), event_sequence=event["sequence"],
            previous=previous, status=state["status"], cell_id=state.get("cell"),
            received_tokens=state["received_tokens"]))

    def _select(self, request):
        candidates = []
        for c in self.config.cells:
            if request.cell is not None and request.cell != c.id:
                continue
            if request.cell is None and request.session_id in self._session_cells and self._session_cells[request.session_id] != c.id:
                continue
            if request.input_tokens + request.output_tokens > c.max_model_len:
                continue
            if c.deployment == "pd_disagg" and request.session_id is not None:
                continue
            try:
                path = self.network.route(request.user_device, c.ingress_device, request.input_bytes)
            except ValueError:
                if self.topology.scenario.link_policy is None:
                    continue
                try:
                    self.network.candidate_path(request.user_device, c.ingress_device)
                except ValueError:
                    continue
                # Acquisition is handled by the communication runtime. A
                # structural candidate is eligible to wait, never ready early.
                path = {"isolated_estimate_ns": float("inf")}
            assigned = sum(s.get("cell") == c.id and s["status"] in ("input", "computing") for s in self.state.values())
            # Explicit deterministic first policy. Reservation/load stays at
            # whole requests; native vLLM decides actual admission and KV.
            candidates.append((assigned, path["isolated_estimate_ns"], c.id))
        if not candidates:
            raise RuntimeError(f"No reachable Cell for {request.id}")
        return self._cell_configs[min(candidates)[2]]

    def _arrive(self, request):
        from orbitserve.runtime.engine.cell_events import CellRequest
        cell_config = self._select(request)
        self._update_request(request.id, "cell_selected", status="input", cell=cell_config.id)
        if request.session_id is not None:
            self._session_cells[request.session_id] = cell_config.id
        self.decisions.append(dict(request=request.id, cell=cell_config.id, time_ns=request.arrival_ns,
                                   policy="least_assigned_then_isolated_ingress_then_id"))
        def ready():
            self._update_request(request.id, "input_delivered", status="computing",
                                 cell_ready_ns=round(self.loop.now_s * 1e9))
            cell_request = CellRequest(request.id, self.loop.now_s, request.input_tokens, request.output_tokens,
                prefix_hit_tokens=request.prefix_hit_tokens, session_id=request.session_id,
                context_id=request.context_id, parent_context_id=request.parent_context_id)
            self.cells[cell_config.id].submit_ready_request(cell_request)
        self._submit(request.user_device, cell_config.ingress_device, request.input_bytes,
                     "request_input", request.id, ready)

    def _output(self, output):
        self.outputs.append(asdict(output))
        request = self.requests[output.request_id]
        c = self._cell_configs[output.cell_id]
        def received():
            state = self.state[request.id]
            now = round(self.loop.now_s * 1e9)
            if output.token_start in self._received_chunks[request.id]:
                raise RuntimeError("duplicate user delivery")
            self._received_chunks[request.id].add(output.token_start)
            changes = dict(received_tokens=state["received_tokens"] + output.token_count,
                           last_token_received_ns=now)
            if output.token_start == 0:
                changes["first_token_received_ns"] = now
            self._update_request(request.id, "token_delivered", **changes)
            self._finish(request.id)
        source = c.ingress_device if c.deployment == "pd_disagg" and output.token_start == 0 else c.output_device
        self._submit(source, request.user_device,
            output.token_count * request.output_bytes_per_token, "request_output", request.id, received)

    def _completed(self, completion):
        self.completions.append(asdict(completion))
        self._update_request(completion.request_id, "cell_completed",
                             compute_complete_ns=round(completion.completion_time_s * 1e9))
        self._finish(completion.request_id)
        for migration in self.config.migrations:
            if migration.after_request == completion.request_id:
                self.migrations.start(migration, self.cells[migration.source_cell],
                    self.cells[migration.target_cell], self.history.history(migration.session_id, migration.context_id))

    def _finish(self, request_id):
        state = self.state[request_id]
        if state["status"] == "failed":
            return
        if "compute_complete_ns" in state and state["received_tokens"] == self.requests[request_id].output_tokens:
            self._update_request(request_id, "user_completed", status="completed",
                completed_ns=max(state["compute_complete_ns"], state["last_token_received_ns"] or 0))

    def run(self):
        from orbitserve.runtime.engine.cell_events import ARRIVAL
        started = time.perf_counter()
        for request in self.config.requests:
            self.loop.schedule(request.arrival_ns / 1e9, ARRIVAL, f"arrival:{request.id}",
                               lambda request=request: self._arrive(request))
        self.positions = {}
        for at, positions in self.topology.orbit_grid or ():
            if at <= self.config.horizon_ns:
                self.loop.schedule(at / 1e9, ARRIVAL, f"orbit:{at}",
                    lambda positions=positions: self.positions.update(positions))
        while (any(s["status"] not in ("completed", "failed") for s in self.state.values())
               or any(m["state"] in ("waiting_for_batch_boundary", "transferring") for m in self.migrations.records.values())):
            at = self.loop.next_event_time_s
            if at is None:
                raise RuntimeError("Simulation stalled before all requests reached a terminal state")
            if at > self.config.horizon_ns / 1e9:
                raise RuntimeError("Simulation horizon exceeded; no partial success is reported")
            self.loop.run(until_s=at)
        return dict(schema_version=1, status="completed" if all(s["status"] == "completed" for s in self.state.values()) else "completed_with_failures",
            configuration=self.config.model_dump(), requests=self.state, decisions=self.decisions,
            compute_completions=self.completions, produced_outputs=self.outputs,
            cells={key: c.report() for key, c in self.cells.items()},
            transfers=self.transport.transfer_events, batches=self.batches,
            input_distributions=self.input_distributions,
            migrations=self.migrations.records,
            network=self.network.snapshot(), event_log=self.loop.events,
            request_state_events=self.request_state_events,
            environment=self.topology.environment_report, final_positions_itrs_km=self.positions,
            measurement=dict(runtime_startup_wall_s=self.startup_wall_s,
                             event_loop_wall_s=time.perf_counter() - started,
                             simulated_time_ns=round(self.loop.now_s * 1e9)),
            provenance=dict(midlayer_root=str(self.root), communication=self.config.backend,
                batch_communication="isolated_operator_estimate",
                graph_backends={c.id: c.graph_backend for c in self.config.cells},
                accuracy="integration tests; no new hardware accuracy claim"))

    @property
    def batches(self):
        return sorted((batch for executor in self.executors.values() for batch in executor.records),
                      key=lambda batch: batch["start_time_ns"])

    @property
    def input_distributions(self):
        return sorted((record for executor in self.executors.values()
                       for record in executor.input_distributions), key=lambda record: record["ready_ns"])

    def close(self):
        errors = []
        for resource in (*self.cells.values(), self.network):
            try:
                resource.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("simulation resources failed to close", errors)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(str(cleanup_error))
