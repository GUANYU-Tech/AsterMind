"""Shared physical network around the existing native Scheduler/Cell seam."""

import time
from dataclasses import asdict

from orbitserve.runtime.engine.cell_bridge import bind_batch_executor
from orbitserve.runtime.engine.cell_migration import NetworkKVMigration

from .bootstrap import activate
from .network import DeviceCommunicationRuntime, DeviceRuntimeHost
from .physics import load_physics


class Simulation:
    def __init__(
        self,
        config,
        *,
        midlayer_root=None,
        cell_factory=None,
        network_factory=None,
        physical_output_dir=None,
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
    ):
        startup = time.perf_counter()
        self.config = config
        self.root = activate(midlayer_root)
        self.cells = {}
        self.network = self.host = self._physics_lifetime = None
        self.topology = config.topology()
        self.physical, self._physics_lifetime = load_physics(
            config, physical_output_dir
        )
        self.host = DeviceRuntimeHost(self.topology, config, self.physical)
        self.loop = self.host.timeline
        self.network = (network_factory or DeviceCommunicationRuntime)(
            self.host, self.topology
        )
        from cell_communication import RankPlacement

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
        from network_batch_executor import NetworkBatchExecutor

        self.executor = bind_batch_executor(
            NetworkBatchExecutor, self.loop, self.network, placements
        )
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
        self._transport_serial = 0
        self._received_chunks = {r.id: set() for r in config.requests}
        self._cell_configs = {c.id: c for c in config.cells}
        from cell_storage import SessionHistoryStore

        self.history = SessionHistoryStore()
        self.migrations = NetworkKVMigration(
            self.loop,
            self.executor,
            {c.id: {p.rank: p.device for p in c.placements} for c in config.cells},
        )
        self._session_cells = {}
        try:
            for c in config.cells:
                if cell_factory is None:
                    cell = self._native_cell(c)
                else:
                    cell = cell_factory(
                        c, self.loop, self._completed, self._output, self.executor
                    )
                self.cells[c.id] = cell
        except BaseException:
            self.close()
            raise
        self.startup_wall_s = time.perf_counter() - startup

    def _native_cell(self, cell):
        from midlayer_cell import MidlayerCellConfig

        from .native_cell import NetworkMidlayerCell

        p = cell.parallel
        config = MidlayerCellConfig(
            self.root,
            model_file=cell.model,
            hardware_file=cell.hardware,
            hardware_id=cell.hardware_id,
            max_model_len=cell.max_model_len,
            max_batched_tokens=cell.max_batched_tokens,
            max_requests=cell.max_requests,
            num_kv_blocks=cell.num_kv_blocks,
            kv_block_size_tokens=cell.kv_block_size_tokens,
            scheduler_adapter="vllm_0_31",
            enable_prefix_caching=cell.prefix_caching,
            enable_expert_parallel=cell.native_expert_parallel,
            expert_routing_seed=cell.expert_routing_seed,
            host_launch_profile_file=cell.host_launch_profile,
            memory_policy_file=cell.memory_policy_file,
            watermark=cell.watermark,
            tensor_parallel_size=p.tp,
            pipeline_parallel_size=p.pp,
        )
        created = NetworkMidlayerCell.create(
            cell.id,
            self.loop,
            config,
            self._completed,
            on_output=self._output,
            batch_executor=self.executor,
            history_store=self.history,
        )
        from .native_execution import NativeStepExecution

        created.execution = NativeStepExecution(created.execution)
        return created

    def _submit(self, source, target, size, operation, request_id, callback):
        self._transport_serial += 1
        self.executor.submit_transport(
            f"serving-{self._transport_serial}",
            source,
            target,
            size,
            operation,
            request_id,
            callback,
            lambda event: self.state[request_id].update(
                status="failed", failure=event.model_dump()
            ),
        )

    def _select(self, request):
        candidates = []
        for c in self.config.cells:
            if request.cell is not None and request.cell != c.id:
                continue
            if (
                request.cell is None
                and request.session_id in self._session_cells
                and self._session_cells[request.session_id] != c.id
            ):
                continue
            if request.input_tokens + request.output_tokens > c.max_model_len:
                continue
            try:
                path = self.network.route(
                    request.user_device, c.ingress_device, request.input_bytes
                )
            except ValueError:
                if (
                    self.topology.scenario.link_policy is None
                    and not self.topology.scenario.dynamic_external
                ):
                    continue
                try:
                    self.network.candidate_path(request.user_device, c.ingress_device)
                except ValueError:
                    continue
                # Acquisition is handled by the communication runtime. A
                # structural candidate is eligible to wait, never ready early.
                path = {"isolated_estimate_ns": float("inf")}
            assigned = sum(
                s.get("cell") == c.id and s["status"] in ("input", "computing")
                for s in self.state.values()
            )
            # Explicit deterministic first policy. Reservation/load stays at
            # whole requests; native vLLM decides actual admission and KV.
            candidates.append((assigned, path["isolated_estimate_ns"], c.id))
        if not candidates:
            raise RuntimeError(f"No reachable Cell for {request.id}")
        return self._cell_configs[min(candidates)[2]]

    def _arrive(self, request):
        from orbitserve.runtime.engine.cell_events import CellRequest

        cell_config = self._select(request)
        state = self.state[request.id]
        state.update(status="input", cell=cell_config.id)
        if request.session_id is not None:
            self._session_cells[request.session_id] = cell_config.id
        self.decisions.append(
            {
                "request": request.id,
                "cell": cell_config.id,
                "time_ns": request.arrival_ns,
                "policy": "least_assigned_then_isolated_ingress_then_id",
            }
        )

        def ready():
            state.update(status="computing", cell_ready_ns=round(self.loop.now_s * 1e9))
            cell_request = CellRequest(
                request.id,
                self.loop.now_s,
                request.input_tokens,
                request.output_tokens,
                prefix_hit_tokens=request.prefix_hit_tokens,
                session_id=request.session_id,
                context_id=request.context_id,
                parent_context_id=request.parent_context_id,
            )
            self.cells[cell_config.id].submit_ready_request(cell_request)

        self._submit(
            request.user_device,
            cell_config.ingress_device,
            request.input_bytes,
            "request_input",
            request.id,
            ready,
        )

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
            state["received_tokens"] += output.token_count
            if output.token_start == 0:
                state["first_token_received_ns"] = now
            state["last_token_received_ns"] = now
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
        self.completions.append(asdict(completion))
        self.state[completion.request_id]["compute_complete_ns"] = round(
            completion.completion_time_s * 1e9
        )
        self._finish(completion.request_id)
        for migration in self.config.migrations:
            if migration.after_request == completion.request_id:
                self.migrations.start(
                    migration,
                    self.cells[migration.source_cell],
                    self.cells[migration.target_cell],
                    self.history.history(migration.session_id, migration.context_id),
                    on_complete=lambda record, migration=migration: (
                        self._session_cells.update(
                            {migration.session_id: migration.target_cell}
                        )
                    ),
                )

    def _finish(self, request_id):
        state = self.state[request_id]
        if state["status"] == "failed":
            return
        if (
            "compute_complete_ns" in state
            and state["received_tokens"] == self.requests[request_id].output_tokens
        ):
            state["status"] = "completed"
            state["completed_ns"] = max(
                state["compute_complete_ns"], state["last_token_received_ns"] or 0
            )

    def run(self):
        from orbitserve.runtime.engine.cell_events import ARRIVAL

        started = time.perf_counter()
        for request in self.config.requests:
            self.loop.schedule(
                request.arrival_ns / 1e9,
                ARRIVAL,
                f"arrival:{request.id}",
                lambda request=request: self._arrive(request),
            )
        self.positions = {}
        for at, positions in self.topology.orbit_grid or ():
            if at <= self.config.horizon_ns:
                self.loop.schedule(
                    at / 1e9,
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
        return {
            "schema_version": 1,
            "status": "completed"
            if all(s["status"] == "completed" for s in self.state.values())
            and all(m["state"] == "completed" for m in self.migrations.records.values())
            else "completed_with_failures",
            "configuration": self.config.model_dump(),
            "requests": self.state,
            "decisions": self.decisions,
            "compute_completions": self.completions,
            "produced_outputs": self.outputs,
            "cells": {
                key: {
                    "steps": c.steps,
                    "final_load": c.snapshot(),
                    "execution_performance": c.execution.performance_stats(),
                    "scheduler_runtime": asdict(c.backend.runtime_info),
                }
                for key, c in self.cells.items()
            },
            "transfers": self.executor.transfer_events,
            "batches": self.executor.records,
            "input_distributions": self.executor.input_distributions,
            "migrations": self.migrations.records,
            "network": self.network.snapshot(),
            "event_log": self.loop.events,
            "environment": self.topology.environment_report,
            "final_positions_itrs_km": self.positions,
            "measurement": {
                "runtime_startup_wall_s": self.startup_wall_s,
                "event_loop_wall_s": time.perf_counter() - started,
                "simulated_time_ns": round(self.loop.now_s * 1e9),
            },
            "provenance": {
                "midlayer_root": str(self.root),
                "communication": self.config.backend,
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
