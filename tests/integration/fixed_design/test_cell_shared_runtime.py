"""Native Cell seams exercised on real Runtime events and physical transport."""

import importlib.util
import unittest
from dataclasses import replace
from types import SimpleNamespace

from orbitserve.runtime import (
    CommunicationExecutor,
    LinkProfile,
    PhysicalTransfer,
    RuntimeStrategySpec,
    run_simulation,
)
from orbitserve.runtime.engine.cell_bridge import (
    CellDeployment,
    CellRuntime,
    CellTransfer,
    CellTransferExecutor,
    RuntimeCellNetwork,
    bind_batch_executor,
)
from orbitserve.runtime.engine.cell_events import CellCompletion, CellTokenOutput
from orbitserve.runtime.engine.cell_migration import NetworkKVMigration
from orbitserve.runtime.network import RuntimeNetworkFrames
from orbitserve.runtime.timeline import GlobalTimeline
from tests.unit.runtime.test_runtime_engine import _request, _scenario, _trace


def scenario(*, requests=(), transport_model="fluid", **network_options):
    trace = _trace()
    trace = replace(
        trace,
        ephemeris_samples=tuple(
            {
                **row,
                "position_itrs_km": [
                    7000.0 if row["satellite_id"] == "sat-a" else 7050.0,
                    0.0,
                    0.0,
                ],
            }
            for row in trace.ephemeris_samples
        ),
    )
    spec = _scenario(
        RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
        ),
        trace=trace,
    )
    network = replace(
        spec.network,
        transport_model=transport_model,
        link_profiles={
            **spec.network.link_profiles,
            "communication_isl": LinkProfile(8e6, 0.0, capacity_schedule=((0.5, 0.5),)),
        },
        **network_options,
    )
    return replace(spec, network=network, requests=tuple(requests))


class ProbeRuntime:
    """Own no compute clock; register external work before Runtime starts."""

    def __init__(self, setup):
        self.setup, self.jobs = setup, {}

    def initialize(self, engine, requests):
        self.loop = GlobalTimeline(host=engine)
        self.network = RuntimeCellNetwork(engine)
        self.executor = CellTransferExecutor(self.loop, self.network)
        self.setup(self, engine)

    def close(self):
        self.network.close()
        if hasattr(self, "migration"):
            self.migration.close()


class StorageCell:
    storage_signature = ("same-model", "same-shards")
    _inflight = None

    def __init__(self, cell_id):
        self.cell_id, self.paused, self.calls = cell_id, False, []

    def pause_for_migration(self, identifier):
        self.paused = True

    def resume_after_migration(self, identifier):
        self.paused = False

    def migration_memory_time(self, size, stage):
        return 0.1

    def storage_rpc(self, operation, *args):
        self.calls.append(operation)
        return {"accepted": True, "bytes": 1_000_000}


class CellSharedRuntimeTests(unittest.TestCase):
    def test_native_cell_tokens_share_dynamic_transport_and_host_clock(self):
        self._assert_shared_clock("fluid")

    @unittest.skipUnless(
        importlib.util.find_spec("_flowsim"), "requires shared-runtime FlowSim"
    )
    def test_cell_protocol_with_real_flowsim(self):
        self._assert_shared_clock("flowsim")

    @unittest.skipUnless(importlib.util.find_spec("ns"), "requires native ns-3")
    def test_cell_protocol_with_real_ns3(self):
        self._assert_shared_clock("ns3")

    def _assert_shared_clock(self, backend):
        requests = tuple(
            {
                **_request(f"r{i}", i * 0.1, session_id=f"s{i}"),
                "input_bytes": 1.0,
                "prompt_tokens": 4,
            }
            for i in range(2)
        )
        outputs = []

        def factory(cell_id, loop, network, on_output, on_complete):
            executor = CellTransferExecutor(loop, network)

            class Cell:
                def __init__(self):
                    self.cell_id, self.pending = cell_id, 0

                def snapshot(self):
                    return {"pending_request_count": self.pending}

                def close(self):
                    pass

                def submit_ready_request(self, request):
                    self.pending += 1

                    def completed():
                        self.pending -= 1
                        outputs.append(loop.now_s)
                        on_output(
                            CellTokenOutput(
                                cell_id,
                                request.request_id,
                                0,
                                request.output_tokens,
                                loop.now_s,
                                True,
                            )
                        )
                        on_complete(
                            CellCompletion(
                                cell_id,
                                request.request_id,
                                request.cell_ready_time_s,
                                request.cell_ready_time_s,
                                loop.now_s,
                                loop.now_s,
                                loop.now_s,
                                request.output_tokens,
                                loop.now_s - request.cell_ready_time_s,
                                0.0,
                            )
                        )

                    executor.submit_transport(
                        request.request_id,
                        "sat-a",
                        "sat-b",
                        1_000_000,
                        "native_collective",
                        None,
                        completed,
                    )

            return Cell()

        runtime = CellRuntime({"sat-a": CellDeployment(factory, {0: "sat-a"})})
        result = run_simulation(
            scenario(requests=requests, transport_model=backend),
            compute_runtime=runtime,
        )
        self.assertEqual(result.request_summary["completed_request_count"], 2)
        self.assertTrue(all(t > 3 for t in outputs))
        events = result.state_summary["event_log"]
        self.assertEqual(
            [e["time_s"] for e in events], sorted(e["time_s"] for e in events)
        )
        transfers = [
            e
            for e in events
            if e["kind"] == "bandwidth_update" and e.get("stage") == "native_collective"
        ]
        self.assertTrue(
            any(
                e["interval_start_s"] == 0.5 and e["link_capacity_bps"] == 4e6
                for e in transfers
            )
        )
        self.assertAlmostEqual(
            sum(e["bytes_sent"] for e in transfers), 2_000_000.0, places=3
        )

    def test_per_rank_migration_waits_for_all_shards_and_target_write(self):
        source, target = StorageCell("A"), StorageCell("B")

        def setup(runtime, engine):
            runtime.migration = NetworkKVMigration(
                runtime.loop,
                runtime.executor,
                {"A": {0: "sat-a", 1: "sat-a"}, "B": {0: "sat-b", 1: "sat-b"}},
            )
            runtime.record = runtime.migration.start(
                SimpleNamespace(id="kv", mode="move"), source, target, {}
            )

        runtime = ProbeRuntime(setup)
        run_simulation(scenario(), compute_runtime=runtime)
        self.assertEqual(runtime.record["state"], "completed")
        self.assertEqual(runtime.record["bytes"], 2_000_000)
        self.assertEqual(len(runtime.record["partitions"]), 2)
        self.assertAlmostEqual(
            runtime.record["completed_time_s"]
            - runtime.record["network_complete_time_s"],
            0.1,
        )
        self.assertIn("commit_import", target.calls)
        self.assertFalse(source.paused or target.paused)

    def test_fault_rolls_back_native_reservations_and_releases_both_cells(self):
        source, target = StorageCell("A"), StorageCell("B")

        def setup(runtime, engine):
            runtime.migration = NetworkKVMigration(
                runtime.loop, runtime.executor, {"A": "sat-a", "B": "sat-b"}
            )
            runtime.record = runtime.migration.start(
                SimpleNamespace(id="kv", mode="copy"), source, target, {}
            )

        fault = {
            "target_type": "link",
            "target_id": "communication_isl:sat-a:sat-b",
            "start_s": 0.3,
            "end_s": 3.0,
        }
        runtime = ProbeRuntime(setup)
        spec = scenario(faults=(fault,), failure_policy="fail")
        # Use the generated identifier rather than assume a link-name convention.
        from orbitserve.runtime.engine.topology import candidate_links

        link = next(
            e for e in candidate_links(spec.network) if e["kind"] == "communication_isl"
        )
        spec = replace(
            spec,
            network=replace(
                spec.network, faults=({**fault, "target_id": link["link_id"]},)
            ),
        )
        run_simulation(spec, compute_runtime=runtime)
        self.assertEqual(runtime.record["state"], "failed")
        self.assertIn("rollback_import", target.calls)
        self.assertNotIn("commit_import", target.calls)
        self.assertFalse(source.paused or target.paused)

    def test_horizon_retires_external_work_without_publishing_target_kv(self):
        source, target = StorageCell("A"), StorageCell("B")

        def setup(runtime, engine):
            runtime.migration = NetworkKVMigration(
                runtime.loop, runtime.executor, {"A": "sat-a", "B": "sat-b"}
            )
            runtime.record = runtime.migration.start(
                SimpleNamespace(id="kv", mode="copy"), source, target, {}
            )

        runtime = ProbeRuntime(setup)
        run_simulation(
            replace(scenario(), time_horizon_s=0.25), compute_runtime=runtime
        )
        self.assertEqual(runtime.record["state"], "failed")
        self.assertNotIn("commit_import", target.calls)
        self.assertFalse(source.paused or target.paused)

    def test_current_network_frame_reads_exact_host_state(self):
        def setup(runtime, engine):
            frames = RuntimeNetworkFrames(engine)
            self.assertEqual(len(frames.frame_at(0).node_states), 4)
            with self.assertRaisesRegex(ValueError, "current time"):
                frames.frame_at(1)

        run_simulation(scenario(), compute_runtime=ProbeRuntime(setup))

    def test_public_communication_executor_binds_to_runtime_resources(self):
        completed = []

        def setup(runtime, engine):
            executor = CommunicationExecutor.for_runtime(
                engine, network=runtime.network
            )
            for identifier in ("first", "second"):
                executor.submit(
                    PhysicalTransfer(identifier, "sat-a", "sat-b", 1_000_000),
                    completed.append,
                )

        run_simulation(scenario(), compute_runtime=ProbeRuntime(setup))
        self.assertEqual(len(completed), 2)
        self.assertTrue(all(row.completed_at_s > 3 for row in completed))
        self.assertEqual({row.backend for row in completed}, {"fluid:flow"})

    def test_native_executor_submits_input_distribution_before_collective(self):
        completed, started = [], []

        class NativeExecutor:
            def __init__(self, loop, network, placements):
                self.loop, self.network = loop, network
                self._callbacks = {}
                self.transfer_events = []

            def _phase(
                self, cell, pool, operation, correlation, sends, callback, record
            ):
                identifier = operation
                self.network.submit(
                    CellTransfer(
                        identifier,
                        "sat-a",
                        "sat-b",
                        1_000_000,
                        self.network.now_ns,
                        operation=operation,
                    )
                )
                self._callbacks[identifier] = callback
                self._arm_network()

            def start_batch(
                self, cell, step, components, execution, callback, **kwargs
            ):
                started.append(self.loop.now_s)
                self._phase(
                    cell,
                    "main",
                    "collective",
                    None,
                    None,
                    lambda: callback(self.loop.now_s),
                    {},
                )

        def setup(runtime, engine):
            executor = bind_batch_executor(
                NativeExecutor, runtime.loop, runtime.network, ()
            )
            executor.boundary_listeners.append(
                lambda: completed.append(runtime.loop.now_s)
            )
            pool = SimpleNamespace(
                physical_ranks_for_pipeline_stage=lambda stage: (0, 1)
            )
            execution = SimpleNamespace(
                deployment=SimpleNamespace(worker_pools={"main": pool}),
                input_distribution_in_graph=False,
                last_step_token_count=4,
            )
            executor.start_batch("A", 0, None, execution, lambda at_s: None)

        run_simulation(scenario(), compute_runtime=ProbeRuntime(setup))
        self.assertEqual(len(started), 1)
        self.assertGreater(started[0], 1)
        self.assertGreater(completed[0], started[0] + 1)


if __name__ == "__main__":
    unittest.main()
