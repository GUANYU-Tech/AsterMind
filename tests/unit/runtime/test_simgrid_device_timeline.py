"""All device/GPU callback and surrogate deadlines use the native global clock."""

import importlib.util
from types import SimpleNamespace
import unittest

from simulator.bootstrap import activate
from simulator.network import DeviceCommunicationRuntime, DeviceRuntimeHost
from orbitserve.runtime.engine.cell_bridge import CellTransfer
from tests.unit.runtime.test_scoped_protobridge import graph


@unittest.skipUnless(importlib.util.find_spec("simgrid"), "requires native SimGrid")
class SimGridDeviceTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        activate()

    def test_three_communication_scopes_complete_on_native_global_events(self):
        for mode in ("flowsim", "protobridge_satellite", "protobridge_cell"):
            with self.subTest(mode=mode):
                config = SimpleNamespace(
                    horizon_ns=1_000_000_000,
                    backend="simgrid",
                    timeline_mode="simgrid",
                    runtime_network={},
                    physical_link_bindings={},
                    communication_mode=mode,
                    protobridge_startup_us=0,
                )
                host = DeviceRuntimeHost(graph(), config)
                network = DeviceCommunicationRuntime(host, graph())
                self.addCleanup(host.close)
                self.addCleanup(network.close)
                completed = []
                network.subscribe(lambda event: completed.append(event))
                request = CellTransfer(
                    id="gpu-transfer",
                    source="a0",
                    target="a1",
                    size_bytes=32,
                    ready_time_ns=0,
                    origin="multi_gpu",
                    operation="all_reduce",
                    correlation_id="request",
                )
                host.timeline.schedule(
                    0, 0, "gpu_ready", lambda: network.submit(request)
                )
                host.run()
                self.assertEqual(len(completed), 1)
                self.assertEqual(completed[0].kind, "completed")
                self.assertEqual(host.now, 1)
                self.assertEqual(
                    host.transport_provenance["timeline_authority"], "simgrid_engine"
                )
                self.assertGreater(host.transport.scheduler_stats["dispatched"], 0)
                self.assertEqual(network.snapshot()["timeline_mode"], "simgrid")
                if mode != "flowsim":
                    self.assertTrue(network.communication_policy.records)
                self.assertTrue(
                    all(
                        a["time_s"] <= b["time_s"]
                        for a, b in zip(host.log, host.log[1:])
                    )
                )

    def test_same_time_fault_priority_and_recursive_gpu_dependency(self):
        config = SimpleNamespace(
            horizon_ns=1_000_000,
            backend="simgrid",
            timeline_mode="simgrid",
            runtime_network={},
            physical_link_bindings={},
            communication_mode="flowsim",
        )
        host = DeviceRuntimeHost(graph(), config)
        self.addCleanup(host.close)
        order = []
        host.push(
            0.0005, 10, "external_callback", payload=lambda: order.append("compute")
        )
        host.push(
            0.0005, -2, "external_callback", payload=lambda: order.append("fault")
        )
        host.timeline.schedule(
            0.0005,
            0,
            "launch",
            lambda: host.timeline.schedule(
                host.now, 0, "dependent", lambda: order.append("dependent")
            ),
        )
        host.run()
        self.assertEqual(order, ["fault", "compute", "dependent"])

    def test_programmatic_host_cannot_bypass_mode_backend_validation(self):
        for timeline, backend in [
            ("simgrid", "flowsim"),
            ("orbitserve", "simgrid"),
            ("invalid", "flowsim"),
        ]:
            with (
                self.subTest(timeline=timeline, backend=backend),
                self.assertRaises(ValueError),
            ):
                DeviceRuntimeHost(
                    graph(), SimpleNamespace(timeline_mode=timeline, backend=backend)
                )

    def test_hosted_same_time_callbacks_preserve_native_double_and_pending_count(self):
        from orbitserve.runtime.engine.simgrid_events import simgrid_time

        config = SimpleNamespace(
            horizon_ns=1_000_000_000,
            backend="simgrid",
            timeline_mode="simgrid",
            runtime_network={},
            physical_link_bindings={},
            communication_mode="flowsim",
        )
        host = DeviceRuntimeHost(graph(), config)
        self.addCleanup(host.close)
        instants = []

        def current_callback():
            instants.append(float(host.now))
            self.assertEqual(host.timeline.report()["pending_events"], 0)
            scheduled = host.timeline.schedule(
                host.now, 0, "same_time", lambda: instants.append(float(host.now))
            )
            self.assertEqual(float(scheduled.time_s), float(host.transport.now))
            self.assertEqual(host.timeline.report()["pending_events"], 1)

        at = simgrid_time(0.006621751990538874)
        host.timeline.schedule(at, 0, "initial", current_callback)
        self.assertEqual(host.timeline.report()["pending_events"], 1)
        host.run()
        self.assertEqual(instants[0], instants[1])
        self.assertEqual(host.timeline.report()["pending_events"], 0)

    def test_manual_device_fault_and_restore_on_all_three_scopes(self):
        from simulator.device_controls import apply_device_command

        for mode in ("flowsim", "protobridge_satellite", "protobridge_cell"):
            with self.subTest(mode=mode):
                config = SimpleNamespace(
                    horizon_ns=1_000_000_000,
                    backend="simgrid",
                    timeline_mode="simgrid",
                    runtime_network={
                        "failure_policy": "retry_next_sample",
                        "max_transfer_retries": 20,
                    },
                    physical_link_bindings={},
                    communication_mode=mode,
                    protobridge_startup_us=0,
                )
                host = DeviceRuntimeHost(graph(), config)
                network = DeviceCommunicationRuntime(host, host.net.topology)
                done = []
                network.subscribe(done.append)
                network.submit(
                    CellTransfer(
                        "inflight",
                        "a0",
                        "a1",
                        1000,
                        0,
                        origin="multi_gpu",
                        operation="tp",
                    )
                )
                host.timeline.schedule(
                    0.0001,
                    -3,
                    "manual-fault",
                    lambda: apply_device_command(
                        host, dict(action="device_link_fail", target_id="a0")
                    ),
                )
                host.timeline.schedule(
                    0.0002,
                    -3,
                    "manual-restore",
                    lambda: apply_device_command(
                        host, dict(action="device_link_restore", target_id="a0")
                    ),
                )
                try:
                    host.run()
                    self.assertEqual(len(done), 1)
                    self.assertEqual(done[0].kind, "completed")
                    self.assertFalse(host.net.failed_device_links)
                    self.assertEqual(
                        sum(e["kind"] == "topology_command" for e in host.log), 2
                    )
                finally:
                    network.close()
                    host.close()
