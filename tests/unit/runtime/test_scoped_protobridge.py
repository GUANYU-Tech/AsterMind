"""Communication boundaries, shared-clock completion and fabric reservations."""

from types import SimpleNamespace
import unittest

from astermind.builder.topology import PhysicalTopology
from astermind.builder.models import Scenario
from astermind.engine.cell_bridge import CellTransfer, CellTransferExecutor
from astermind.common.bootstrap import activate
from astermind.engine.network_host import DeviceCommunicationRuntime, DeviceRuntimeHost


def graph(schedule=()):
    satellites = ["a", "b", "c"]
    devices, links = [], []
    for owner in satellites:
        devices += [
            dict(id=owner + str(i), satellite=owner, kind="gpu") for i in range(2)
        ]
        devices += [
            dict(id=owner + "s", satellite=owner, kind="switch", capacity_gbps=0.001),
            dict(
                id=owner + "t",
                satellite=owner,
                kind="terminal",
                external_access=True,
                max_external_links=2,
            ),
        ]
        links += [
            dict(
                id=owner + str(i),
                source=owner + str(i),
                target=owner + "s",
                bandwidth_gbps=0.008,
            )
            for i in range(2)
        ]
        links += [
            dict(
                id=owner + "e",
                source=owner + "s",
                target=owner + "t",
                bandwidth_gbps=0.008,
            )
        ]
    links += [
        dict(
            id="ab",
            source="at",
            target="bt",
            bandwidth_gbps=0.0008,
            latency_ns=1000,
            capacity_schedule_ns=schedule,
        ),
        dict(
            id="bc", source="bt", target="ct", bandwidth_gbps=0.0004, latency_ns=10000
        ),
    ]
    return PhysicalTopology(
        Scenario.model_validate(
            dict(
                cells=[
                    dict(id="A", satellites=["a", "b"]),
                    dict(id="B", satellites=["c"]),
                ],
                satellites=[dict(id=s) for s in satellites],
                devices=devices,
                links=links,
            )
        )
    )


class ScopedProtoBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        activate()

    def test_native_host_registers_control_boundaries_on_shared_clock(self):
        from astermind.engine import runtime_control

        controller = SimpleNamespace()
        config = SimpleNamespace(
            horizon_ns=1_000_000_000,
            backend="flowsim",
            runtime_network={},
            physical_link_bindings={},
            communication_mode="protobridge_cell",
        )
        with runtime_control(controller):
            host = DeviceRuntimeHost(graph(), config)
        try:
            self.assertIs(host.controller, controller)
            self.assertEqual(
                [event[0] for event in host.heap if event[3] == "control_boundary"],
                [t for t in host.net.environment_times if t < host.end],
            )
            self.assertTrue(any(e[3] == "control_boundary" for e in host.heap))
        finally:
            host.close()

    def run_mode(self, mode):
        topology = graph()
        config = SimpleNamespace(
            horizon_ns=1_000_000_000,
            backend="flowsim",
            runtime_network={},
            physical_link_bindings={},
            communication_mode=mode,
            protobridge_startup_us=0,
        )
        host = DeviceRuntimeHost(topology, config)
        network = DeviceCommunicationRuntime(host, topology)
        done = []
        network.subscribe(lambda event: done.append(event))
        for identity, source, target in [
            ("local", "a0", "a1"),
            ("cell", "a0", "b0"),
            ("external", "a0", "c0"),
        ]:
            network.submit(
                CellTransfer(
                    identity,
                    source,
                    target,
                    1000,
                    0,
                    origin="multi_gpu",
                    operation="tp",
                )
            )
        try:
            host.run()
            return network.snapshot(), done
        finally:
            network.close()
            host.close()

    def test_three_scopes_always_keep_cross_cell_on_flowsim(self):
        for mode, expected in [
            ("flowsim", set()),
            ("protobridge_satellite", {"local"}),
            ("protobridge_cell", {"local", "cell"}),
        ]:
            with self.subTest(mode=mode):
                report, done = self.run_mode(mode)
                self.assertEqual({e.id for e in done}, {"local", "cell", "external"})
                self.assertTrue(all(e.kind == "completed" for e in done))
                self.assertEqual(
                    {r["correlation_id"] for r in report["surrogate_operations"]},
                    expected,
                )
                flows = {
                    e.get("cell_transfer_id")
                    for e in report["event_log"]
                    if e["kind"] == "transmit_start"
                }
                self.assertNotIn("local", flows if "local" in expected else set())
                self.assertTrue(
                    any(
                        e["id"] == "external" and e.get("backend") != "protobridge"
                        for e in report["flows"]
                    )
                )

    def test_collective_waits_and_external_transfer_respects_reserved_fabric(self):
        topology = graph()
        config = SimpleNamespace(
            horizon_ns=1_000_000_000,
            backend="flowsim",
            runtime_network={},
            physical_link_bindings={},
            communication_mode="protobridge_cell",
            protobridge_startup_us=0,
        )
        host = DeviceRuntimeHost(topology, config)
        network = DeviceCommunicationRuntime(host, topology)
        record, completions = {"call_id": "allreduce"}, []
        transfer = SimpleNamespace(
            operation="all_reduce",
            bytes=1000,
            algorithm="ring",
            bytes_semantics="per_rank_input",
            reduction="sum",
        )
        try:
            self.assertTrue(
                network.try_collective(
                    "A",
                    transfer,
                    ("a0", "b0"),
                    record,
                    lambda: completions.append(host.now),
                )
            )
            executor = CellTransferExecutor(host.timeline, network)
            executor.submit_transport(
                "kv",
                "a0",
                "c0",
                1000,
                "pd_kv",
                None,
                lambda: completions.append(host.now),
            )
            host.run()
            row = record["surrogate"]
            self.assertGreater(row["end_s"], row["start_s"])
            self.assertEqual(completions[0], row["end_s"])
            starts = [e["time_s"] for e in host.log if e["kind"] == "transmit_start"]
            self.assertTrue(starts)
            self.assertGreaterEqual(min(starts), row["end_s"])
            self.assertEqual(len(network.communication_policy.records), 1)
            self.assertGreater(row["attempts"][0]["parameters"]["link_latency_us"], 0)
        finally:
            network.close()
            host.close()

    def test_same_cell_name_cannot_hide_cross_cell_participant(self):
        topology = graph()
        config = SimpleNamespace(
            horizon_ns=1_000_000_000,
            backend="flowsim",
            runtime_network={},
            physical_link_bindings={},
            communication_mode="protobridge_cell",
            protobridge_startup_us=0,
        )
        host = DeviceRuntimeHost(topology, config)
        network = DeviceCommunicationRuntime(host, topology)
        try:
            self.assertFalse(
                network.try_collective("A", None, ("a0", "c0"), {}, lambda: None)
            )
        finally:
            network.close()
            host.close()

    def test_bandwidth_change_restarts_inflight_surrogate_without_early_completion(
        self,
    ):
        topology = graph(((5_000_000, 0.1),))
        config = SimpleNamespace(
            horizon_ns=1_000_000_000,
            backend="flowsim",
            runtime_network={},
            physical_link_bindings={},
            communication_mode="protobridge_cell",
            protobridge_startup_us=0,
        )
        host = DeviceRuntimeHost(topology, config)
        network = DeviceCommunicationRuntime(host, topology)
        completed = []
        network.subscribe(completed.append)
        try:
            network.submit(
                CellTransfer(
                    "changing", "a0", "b0", 1000, 0, origin="multi_gpu", operation="pp"
                )
            )
            host.run()
            row = network.communication_policy.records[0]
            self.assertEqual(len(completed), 1)
            self.assertEqual(len(row["attempts"]), 2)
            first, last = row["attempts"]
            self.assertEqual(first["invalidated_s"], 0.005)
            self.assertLess(
                last["parameters"]["packet_bw_bytes_per_us"],
                first["parameters"]["packet_bw_bytes_per_us"],
            )
            self.assertAlmostEqual(row["end_s"], last["start_s"] + last["duration_s"])
            self.assertGreater(row["end_s"], first["start_s"] + first["duration_s"])
        finally:
            network.close()
            host.close()

    def test_future_ready_and_queued_cancellation_keep_exactly_one_terminal_event(self):
        topology = graph()
        config = SimpleNamespace(
            horizon_ns=1_000_000_000,
            backend="flowsim",
            runtime_network={},
            physical_link_bindings={},
            communication_mode="protobridge_cell",
            protobridge_startup_us=0,
        )
        host = DeviceRuntimeHost(topology, config)
        network = DeviceCommunicationRuntime(host, topology)
        completed = []
        network.subscribe(completed.append)
        try:
            network.submit(
                CellTransfer(
                    "future",
                    "a0",
                    "a1",
                    1000,
                    100_000_000,
                    origin="multi_gpu",
                    operation="tp",
                )
            )
            network.submit(
                CellTransfer(
                    "cancel",
                    "a0",
                    "c0",
                    1000,
                    200_000_000,
                    origin="orbitserve",
                    operation="pd_kv",
                )
            )
            network.cancel(["cancel"])
            host.run()
            self.assertEqual(len(completed), 2)
            self.assertEqual(
                next(e for e in completed if e.id == "cancel").kind, "failed"
            )
            self.assertGreaterEqual(
                network.communication_policy.records[0]["start_s"], 0.1
            )
            self.assertFalse(any(e["kind"] == "transmit_start" for e in host.log))
        finally:
            network.close()
            host.close()
