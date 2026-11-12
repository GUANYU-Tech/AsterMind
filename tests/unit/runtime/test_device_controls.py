from types import SimpleNamespace
import unittest
from tests.unit.runtime.test_scoped_protobridge import graph
from simulator.bootstrap import activate
from simulator.network import DeviceRuntimeHost, DeviceCommunicationRuntime
from simulator.device_controls import apply_device_command
from orbitserve.runtime.engine.cell_bridge import CellTransfer


class DeviceControlsTests(unittest.TestCase):
    def test_gpu_link_fault_recovery_reaches_both_communication_solvers(self):
        activate()
        for mode in ("flowsim", "protobridge_cell"):
            with self.subTest(mode=mode):
                config = SimpleNamespace(
                    horizon_ns=1_000_000_000,
                    backend="flowsim",
                    communication_mode=mode,
                    physical_link_bindings={},
                    runtime_network={
                        "failure_policy": "retry_next_sample",
                        "max_transfer_retries": 20,
                    },
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

                def control(action):
                    return apply_device_command(
                        host, dict(action=action, target_id="a0")
                    )

                host.timeline.schedule(
                    0.0001, -3, "test-fault", lambda: control("device_link_fail")
                )
                host.timeline.schedule(
                    0.0002, -3, "test-restore", lambda: control("device_link_restore")
                )
                try:
                    host.run()
                    self.assertEqual(len(done), 1)
                    self.assertEqual(done[0].kind, "completed")
                    self.assertEqual(
                        len([e for e in host.log if e["kind"] == "topology_command"]), 2
                    )
                    self.assertFalse(host.net.failed_device_links)
                finally:
                    network.close()
                    host.close()
