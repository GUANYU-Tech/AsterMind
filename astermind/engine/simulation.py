"""Shared physical network around the existing native Scheduler/Cell seam."""

import time
from dataclasses import asdict

from astermind.engine.cell_bridge import CellTransferExecutor
from astermind.engine.cell_migration import NetworkKVMigration
from astermind.engine.timeline import nanoseconds

from astermind.common.bootstrap import activate
from astermind.engine.network_host import DeviceCommunicationRuntime, DeviceRuntimeHost
from astermind.builder.physical.loading import load_physics


class Simulation:
    def __init__(
        self,
        config,
        *,
        midlayer_root=None,
        cell_factory=None,
        network_factory=None,
        physical_output_dir=None,
        startup_progress=None,
        cell_init_workers=None,
    ):
        self.cells = {}
        self.network = self.host = self._physics_lifetime = None
        try:
            self._initialize(
                config,
                midlayer_root=midlayer_root,
                cell_factory=cell_factory,
                network_factory=network_factory,
                physical_output_dir=physical_output_dir,
                startup_progress=startup_progress,
                cell_init_workers=cell_init_workers,
            )
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        config,
        *,
        midlayer_root=None,
        cell_factory=None,
        network_factory=None,
        physical_output_dir=None,
        startup_progress=None,
        cell_init_workers=None,
    ):
        from astermind.engine.startup_resources import startup_policy

        self.cell_startup_policy = startup_policy(len(config.cells), cell_init_workers)
        startup = time.perf_counter()
        self.startup_stages = []
        stage_started, previous = startup, None

        def report(phase, **details):
            nonlocal stage_started, previous
            now = time.perf_counter()
            if previous is not None:
                self.startup_stages.append({**previous, "wall_s": now - stage_started})
            stage_started, previous = now, dict(phase=phase, **details)
            if startup_progress is not None:
                startup_progress({**previous, "elapsed_s": now - startup})

        report("bootstrap")
        self.config = config
        self.root = activate(midlayer_root)
        from astermind.engine.compute.provenance import inputs
        self.native_inputs = inputs(self.root,config.cells)
        self.cells = {}
        self.network = self.host = self._physics_lifetime = None
        self.topology = config.topology()
        report("physics")
        self.physical, self._physics_lifetime = load_physics(
            config, physical_output_dir
        )
        report("network")
        self.host = DeviceRuntimeHost(self.topology, config, self.physical)
        self.loop = self.host.timeline
        self.network = (network_factory or DeviceCommunicationRuntime)(
            self.host, self.topology
        )
        from astermind.engine.compute.placement import RankPlacement

        placements = tuple(
            RankPlacement(
                c.id,
                p.pool,
                p.rank,
                self.topology.devices[p.device].satellite,
                p.device,
            )
            for c in config.cells
            for p in c.placements
        )
        from astermind.engine.compute.batch_executor import RuntimeBatchExecutor
        if config.compute_mode == "fast_mode":
            from astermind.engine.compute.fast_mode import FastModeExecutor
            executor_type = FastModeExecutor
        else:
            executor_type = RuntimeBatchExecutor
        self.executor = CellTransferExecutor(self.loop, self.network)
        self.executors = {
            cell.id: executor_type(cell.id, self.loop, self.network.communication_policy,
                tuple(p for p in placements if p.cell_id == cell.id),
                on_batch_boundary=self.executor.notify_batch_boundary)
            for cell in config.cells
        }
        from astermind.engine.record_store import CompressedRecords
        self.host.log = CompressedRecords()
        self.loop.events = CompressedRecords()
        self.cells, self.requests = {}, {r.id: r for r in config.requests}
        self.state = {
            r.id: {
                "status": "pending",
                "arrival_ns": r.arrival_ns,
                "received_tokens": 0,
                "first_token_received_ns": None,
                "last_token_received_ns": None,
            }
            for r in config.requests
        }
        self.completions, self.outputs, self.decisions = [], [], []
        self.request_state_events = CompressedRecords()
        self._kv_waiters = {}
        self._parent_waiters = {}
        self._request_transfers = {}
        self._failed_cells = set()
        self._transport_serial = 0
        self._received_chunks = {r.id: set() for r in config.requests}
        self._cell_configs = {c.id: c for c in config.cells}
        from astermind.engine.compute.cell_storage import SessionHistoryStore

        self.history = SessionHistoryStore()
        self.migrations = NetworkKVMigration(
            self.loop,
            self.executor,
            {c.id: {p.rank: p.device for p in c.placements} for c in config.cells},
        )
        self._session_cells = {}
        self.cell_init_workers = self.cell_startup_policy["workers"]
        if cell_factory is None:
            # Resolve the native module graph on the owning thread first.
            from astermind.engine.compute.serving import ServingScheduler  # noqa: F401
            from astermind.engine.startup import initialize_cells

            self.cells = initialize_cells(
                config.cells,
                self._native_cell,
                workers=self.cell_init_workers,
                progress=lambda **details: report("cells", **details),
            )
        else:
            # Injected factories may mutate shared state during construction.
            self.cell_init_workers = 1
            self.cell_startup_policy.update(workers=1, source="custom_factory")
            for index, c in enumerate(config.cells, 1):
                report("cell", cell=c.id, index=index, total=len(config.cells))
                self.cells[c.id] = cell_factory(
                    c, self.loop, self._completed, self._output, self.executor
                )
        report("ready")
        self.startup_wall_s = time.perf_counter() - startup

    def _native_cell(self, cell):
        from astermind.engine.compute.runtime import MidlayerCellConfig
        from astermind.engine.compute.serving import ServingScheduler
        from astermind.engine.compute.pd_cell import PrefillDecodeCell
        from astermind.engine.compute.deployment import validate_cell_deployment

        p = cell.parallel
        config = MidlayerCellConfig(
            self.root, model_file=cell.model, hardware_file=cell.hardware,
            hardware_id=cell.hardware_id, max_model_len=cell.max_model_len,
            max_batched_tokens=cell.max_batched_tokens, max_requests=cell.max_requests,
            num_kv_blocks=cell.num_kv_blocks, kv_block_size_tokens=cell.kv_block_size_tokens,
            scheduler_adapter=cell.scheduler_adapter, enable_prefix_caching=cell.prefix_caching,
            enable_chunked_prefill=cell.enable_chunked_prefill,
            long_prefill_token_threshold=cell.long_prefill_token_threshold,
            enable_expert_parallel=cell.expert_parallel_enabled,
            expert_routing_seed=cell.expert_routing_seed, host_launch_profile_file=cell.host_launch_profile,
            memory_policy_file=cell.memory_policy_file, watermark=cell.watermark,
            tensor_parallel_size=p.tp, pipeline_parallel_size=p.pp,
            data_parallel_size=p.dp, sequence_parallel_size=p.sp,
            deployment=cell.deployment, execution_mode=cell.execution_mode,
            compute_mode=self.config.executor_compute_mode, fast_glm_dsa_score=cell.fast_glm_dsa_score)
        executor = self.executors[cell.id]
        if cell.deployment == "pd_disagg":
            created = PrefillDecodeCell(cell.id, self.loop, config, executor,
                self._completed, self._output, self._cell_failure,
                submit_handoff=self.executor.submit_transport)
        else:
            created = ServingScheduler.create(cell.id, self.loop, config, self._completed,
                on_output=self._output, on_failure=self._cell_failure,
                batch_executor=executor, history_store=self.history)
        try:
            validate_cell_deployment(cell, created.execution, executor.placements.values())
            if self.config.compute_mode == "fast_mode":
                executor.configure(created.execution, config)
        except BaseException:
            created.close()
            raise
        return created

    # Prerequisite lifecycle ported from Communication 9c08c9c; the existing
    # selector, native executor, tick clock and migration transaction remain.
    def _cell_failure(self, failure):
        if failure.get('failure_scope')=='pool_fail_stop':
            self._failed_cells.add(failure['cell_id'])
            # Uploads not yet admitted to the Scheduler are invisible to its
            # failure list. Invalidate them here before their ready callbacks.
            for pending_id,pending in tuple(self.state.items()):
                if pending.get('cell')==failure['cell_id'] and pending['status']=='input':
                    self._request_failed(pending_id,pending.get('attempt',0),'cell_failed_during_upload',
                        dict(failure,request_id=pending_id))
        request_id, attempt = failure['request_id'], 0
        self._request_failed(request_id,attempt,'cell_failed',failure)

    def _attempt_current(self,request_id,attempt):
        state=self.state[request_id]
        return state.get('attempt',0)==attempt and state['status'] not in ('failed','completed')

    def _request_failed(self,request_id,attempt,reason,failure):
        if not self._attempt_current(request_id,attempt):return
        self._update_request(request_id,reason,status='failed',failure=failure)
        self._cancel_request_transfers(request_id,attempt)
        for callback in self._parent_waiters.pop(request_id, ()):
            callback(failure)
        # A failed output delivery does not invalidate an already computed
        # parent context. Only failed computation prevents its KV prerequisite.
        if "compute_complete_ns" not in self.state[request_id]:
            for migration in self.config.migrations:
                if migration.after_request == request_id and migration.before_request is not None:
                    self._request_failed(migration.before_request, 0, 'kv_source_failed', failure)

    def _cancel_request_transfers(self,request_id,attempt):
        pending=getattr(self,'_request_transfers',{}).pop((request_id,attempt),set())
        if pending:
            self.network.cancel(tuple(pending), 'request_failed')

    def _submit(self, source, target, size, operation, request_id, callback):
        self._transport_serial += 1
        attempt=self.state[request_id].get('attempt',0)
        identity=f'serving-{self._transport_serial}'
        pending=self._request_transfers.setdefault((request_id,attempt),set())
        pending.add(identity)
        def complete():
            pending.discard(identity)
            if self._attempt_current(request_id,attempt):callback()
        def failed(event):
            pending.discard(identity)
            self._request_failed(request_id,attempt,'transfer_failed',event.model_dump())
        try:
            self.executor.submit_transport(identity, source, target, size,
                operation, request_id, complete, failed)
        except (ValueError, RuntimeError) as error:
            pending.discard(identity)
            self._request_failed(request_id, attempt, 'transfer_submission_failed',
                                 dict(message=str(error), operation=operation))

    def _update_request(self, request_id, reason, **changes):
        # Communication 9c08c9c state trace, retaining the authoritative tick.
        state = self.state[request_id]
        previous = state["status"]
        state.update(changes)
        event = self.loop.current_event
        self.request_state_events.append(dict(
            request_id=request_id, reason=reason,
            time_ns=nanoseconds(self.loop.now_s), time_ticks=str(self.loop.now_s.ticks),
            event_sequence=event.sequence if event else None,
            previous=previous, status=state["status"], cell_id=state.get("cell"),
            received_tokens=state["received_tokens"], attempt=state.get("attempt", 0),
        ))

    def _select(self, request):
        from astermind.engine.policies.scheduling import select_cell

        return select_cell(self, request)

    def _arrive(self, request):
        from astermind.engine.policies.admission import admit_request

        return admit_request(self, request)

    def _output(self, output):
        if not self._attempt_current(output.request_id, 0):
            return
        self.outputs.append(asdict(output))
        request = self.requests[output.request_id]
        c = self._cell_configs[output.cell_id]

        def received():
            state = self.state[request.id]
            from astermind.engine.timeline import nanoseconds
            now = nanoseconds(self.loop.now_s)
            if output.token_start in self._received_chunks[request.id]:
                raise RuntimeError("duplicate user delivery")
            self._received_chunks[request.id].add(output.token_start)
            state["received_tokens"] += output.token_count
            if output.token_start == 0:
                state["first_token_received_ns"] = now
                state['first_token_received_ticks'] = str(self.loop.now_s.ticks)
                self._record_request("first_delivered", request.id)
            state["last_token_received_ns"] = now
            state['last_token_received_ticks'] = str(self.loop.now_s.ticks)
            self._update_request(request.id, 'token_delivered')
            if state["received_tokens"] == request.output_tokens:
                self._record_request("output_delivered", request.id)
            self._finish(request.id)

        self._submit(
            c.output_device,
            request.user_device,
            output.token_count * request.output_bytes_per_token,
            "request_output",
            request.id,
            received,
        )

    def _completed(self, completion):
        if not self._attempt_current(completion.request_id, 0):
            return
        self.completions.append(asdict(completion))
        from astermind.engine.timeline import nanoseconds
        self._update_request(completion.request_id, 'cell_completed',
            compute_complete_ns=nanoseconds(completion.completion_time_s),
            compute_complete_ticks=str(completion.completion_time_s.ticks))
        self._record_request("compute_complete", completion.request_id)
        self._finish(completion.request_id)
        for callback in self._parent_waiters.pop(completion.request_id, ()):
            callback()
        for migration in self.config.migrations:
            if migration.after_request == completion.request_id:
                try:
                    self.migrations.start(
                        migration,
                        self.cells[migration.source_cell],
                        self.cells[migration.target_cell],
                        self.history.history(migration.session_id, migration.context_id),
                        on_complete=lambda record, migration=migration: (
                            self._session_cells.update({migration.session_id: migration.target_cell}),
                            self._kv_terminal(migration.id),
                        ),
                        on_failure=lambda error, mid=migration.id: self._kv_terminal(mid, error),
                    )
                except (ValueError, RuntimeError) as error:
                    # Keep diagnostics and rollback state from the existing transaction.
                    record = self.migrations.records.setdefault(migration.id, {})
                    record.update(state="failed", message=str(error))
                    self._kv_terminal(migration.id, record)

    def _kv_terminal(self, migration_id, error=None):
        for callback in self._kv_waiters.pop(migration_id, ()):
            callback(migration_id,error)

    def _record_request(self, kind, request_id):
        self.host.log.append(
            dict(
                kind=kind,
                request_id=request_id,
                time_s=self.loop.now_s,
                cell_id=self.state[request_id].get("cell"),
            )
        )

    def _finish(self, request_id):
        state = self.state[request_id]
        if state["status"] in {"failed", "completed"}:
            return
        if (
            "compute_complete_ns" in state
            and state["received_tokens"] == self.requests[request_id].output_tokens
        ):
            state["completed_ns"] = max(
                state["compute_complete_ns"], state["last_token_received_ns"] or 0
            )
            state['completed_ticks'] = str(self.loop.now_s.ticks)
            self._update_request(request_id, 'user_completed', status='completed')
            self.host.log.append(
                dict(
                    kind="request_complete",
                    request_id=request_id,
                    time_s=self.loop.now_s,
                    cell_id=state["cell"],
                )
            )

    def run(self):
        from astermind.engine.cell_events import ARRIVAL

        started = time.perf_counter()
        for request in self.config.requests:
            self.loop.schedule(
                self.host.now.from_ns(request.arrival_ns),
                ARRIVAL,
                f"arrival:{request.id}",
                lambda request=request: self._arrive(request),
            )
        self.positions = {}
        for at, positions in self.topology.orbit_grid or ():
            if at <= self.config.horizon_ns:
                self.loop.schedule(
                    self.host.now.from_ns(at),
                    ARRIVAL,
                    f"orbit:{at}",
                    lambda positions=positions: self.positions.update(positions),
                )
        self.host.run()
        self.migrations.close()
        if any(s["status"] not in {"completed", "failed"} for s in self.state.values()):
            raise RuntimeError(
                "Simulation horizon exceeded before all native requests completed"
            )
        # Communication 9c08c9c user-observed metrics; preserve undefined values.
        for request_id, state in self.state.items():
            first, last = state["first_token_received_ns"], state["last_token_received_ns"]
            count = self.requests[request_id].output_tokens
            state["metrics"] = dict(
                ttft_ns=None if first is None else first - state["arrival_ns"],
                tpot_ns=None if first is None or last is None or count <= 1 or state["status"] != "completed"
                else (last - first) / (count - 1),
                e2e_ns=None if state["status"] != "completed" else state["completed_ns"] - state["arrival_ns"],
            )
        return {
            "schema_version": 2,
            "status": "completed"
            if all(s["status"] == "completed" for s in self.state.values())
            and all(m["state"] == "completed" for m in self.migrations.records.values())
            else "completed_with_failures",
            "configuration": self.config.model_dump(),
            "requests": self.state,
            "decisions": self.decisions,
            "request_state_events": self.request_state_events,
            "compute_completions": self.completions,
            "produced_outputs": self.outputs,
            "cells": {key: cell.report() for key, cell in self.cells.items()},
            "transfers": self.executor.transfer_events,
            "batches": [row for executor in self.executors.values() for row in executor.records],
            "input_distributions": [row for executor in self.executors.values() for row in executor.input_distributions],
            "communication_events": self.network.communication_policy.trace,
            "communication_runtime_stats": self.network.communication_policy.stats,
            "migrations": self.migrations.records,
            "network": self.network.snapshot(),
            "event_log": self.loop.events,
            "environment": self.topology.environment_report,
            "final_positions_itrs_km": self.positions,
            "measurement": {
                "runtime_startup_wall_s": self.startup_wall_s,
                "startup_stages": self.startup_stages,
                "cell_init_workers": self.cell_init_workers,
                "cell_startup_policy": self.cell_startup_policy,
                "event_loop_wall_s": time.perf_counter() - started,
                "simulated_time_ns": round(self.loop.now_s * 1e9),
            },
            "provenance": {
                "midlayer_root": str(self.root),
                "communication": "runtime_comm",
                "compute_mode": self.config.compute_mode,
                "compute_modes": {c.id: self.config.executor_compute_mode for c in self.config.cells},
                "native_inputs": self.native_inputs,
                "source_commit": "9c08c9cad81df5a4fee5e606114753df8dab6c2d",
                "timeline_mode": "orbitserve",
                "timeline_authority": "integer_ticks",
                "graph_backends": {c.id: c.graph_backend for c in self.config.cells},
                "network_model": "orbitserve_dynamic_device_graph",
                "accuracy": "integration tests; no new hardware accuracy claim",
            },
        }

    def close(self):
        from contextlib import ExitStack

        with ExitStack() as cleanup:
            if self._physics_lifetime is not None:
                cleanup.callback(self._physics_lifetime.cleanup)
            if self.host is not None:
                cleanup.callback(self.host.close)
            for cell in self.cells.values():
                cleanup.callback(cell.close)
            if hasattr(self, "migrations"):
                cleanup.callback(self.migrations.close)
            if self.network is not None:
                self.network.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
