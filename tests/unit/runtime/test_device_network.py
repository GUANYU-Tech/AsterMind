"""Device-level resources must remain real shared hops, not satellite aliases."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from astermind.builder.topology import PhysicalTopology
from astermind.builder.models import Scenario
from astermind.engine.cell_bridge import CellTransferExecutor
from astermind.engine.comm.frame_surrogate import CommPolicy
from astermind.engine.network_host import DeviceCommunicationRuntime, DeviceRuntimeHost


def topology():
    devices = [{"id": f"gpu{i}", "satellite": "sat", "kind": "gpu"} for i in range(4)]
    devices += [
        {
            "id": "switch",
            "satellite": "sat",
            "kind": "switch",
            "capacity_gbps": 0.000008,
        }
    ]
    return PhysicalTopology(
        Scenario.model_validate(
            {
                "cells": [{"id": "A", "satellites": ["sat"]}],
                "satellites": [{"id": "sat"}],
                "devices": devices,
                "links": [
                    {
                        "id": f"p{i}",
                        "source": f"gpu{i}",
                        "target": "switch",
                        "bandwidth_gbps": 8.0,
                    }
                    for i in range(4)
                ],
            }
        )
    )


def run_transfers(backend):
    graph = topology()
    config = SimpleNamespace(
        horizon_ns=4_000_000_000,
        backend=backend,
        runtime_network={},
        communication=CommPolicy(runtime_protocol="payload_only",inter_parameters=dict(startup_us=0)),
        physical_link_bindings={},
    )
    host = DeviceRuntimeHost(graph, config)
    network = DeviceCommunicationRuntime(host, graph)
    executor = CellTransferExecutor(host.timeline, network)
    completed = []
    try:
        for identifier, source, target in [
            ("a", "gpu0", "gpu1"),
            ("b", "gpu2", "gpu3"),
        ]:
            executor.submit_transport(
                identifier,
                source,
                target,
                1000,
                "collective",
                None,
                lambda identifier=identifier: completed.append((identifier, host.now)),
            )
        host.run()
        return completed, network.communication_policy.trace, {}, network.events
    finally:
        network.close()
        host.close()


class DeviceNetworkTests(unittest.TestCase):
    def test_device_projection_preserves_directional_rf_pool_across_ground_links(self):
        from astermind.engine import LinkProfile, NetworkNodeSpec, NetworkSpec, NetworkState

        physical_nodes = (
            NetworkNodeSpec(
                "sat",
                "communication",
                sgl_terminals=2,
                radio_capacity_bps={
                    "user_uplink_bps": 8000,
                    "user_downlink_bps": 16000,
                },
                capacity_assumptions=(
                    "Synthetic rates for device projection conservation checks.",
                ),
            ),
            NetworkNodeSpec("u", "ground", sgl_terminals=1, ground_kind="user"),
            NetworkNodeSpec("v", "ground", sgl_terminals=1, ground_kind="user"),
        )
        trace = SimpleNamespace(
            time={"start_s": 0.0, "end_s": 4.0},
            satellite_catalog=({"satellite_id": "sat"},),
            ephemeris_samples=tuple(
                {
                    "satellite_id": "sat",
                    "time_s": float(t),
                    "position_itrs_km": (7000.0, 0.0, 0.0),
                }
                for t in range(5)
            ),
            isl_windows=(),
            sgl_windows=tuple(
                {"ingress_id": g, "satellite_id": "sat", "start_s": 0.0, "end_s": 4.0}
                for g in ("u", "v")
            ),
        )
        spec = NetworkSpec(
            physical_nodes,
            {
                "sgl": LinkProfile(
                    32000, 0, uplink_profile="up", downlink_profile="down"
                ),
                "up": LinkProfile(8000, 0),
                "down": LinkProfile(16000, 0),
            },
            communication_topology="explicit",
        )
        physical = NetworkState(
            trace,
            spec,
            tuple(
                {"ingress_id": g, "lat_deg": 0.0, "lon_deg": 0.0} for g in ("u", "v")
            ),
        )
        raw = topology().scenario.model_dump(mode="json")
        raw["devices"][-1]["capacity_gbps"] = 8.0
        raw["cells"] += [{"id": g, "satellites": [g]} for g in ("u", "v")]
        raw["satellites"] += [{"id": g} for g in ("u", "v")]
        raw["devices"] += [
            {
                "id": "radio",
                "satellite": "sat",
                "kind": "terminal",
                "external_access": True,
                "max_external_links": 2,
            }
        ]
        raw["links"] += [
            {
                "id": "access",
                "source": "switch",
                "target": "radio",
                "bandwidth_gbps": 8.0,
            }
        ]
        for g in ("u", "v"):
            raw["devices"].append(
                {"id": g, "satellite": g, "kind": "terminal", "external_access": True}
            )
            raw["links"].append(
                {"id": g, "source": g, "target": "radio", "bandwidth_gbps": 8.0}
            )
        graph = PhysicalTopology(Scenario.model_validate(raw))
        config = SimpleNamespace(
            horizon_ns=4_000_000_000,
            backend="fluid",
            runtime_network={},
        communication=CommPolicy(runtime_protocol="payload_only",inter_parameters=dict(startup_us=0)),
            ground_owners=("u", "v"),
            physical_link_bindings={},
        )
        host = DeviceRuntimeHost(graph, config, physical)
        network = DeviceCommunicationRuntime(host, graph)
        executor = CellTransferExecutor(host.timeline, network)
        completed = {}
        try:
            for name, source, target in [
                ("up-u", "u", "gpu0"),
                ("up-v", "v", "gpu1"),
                ("down", "gpu0", "u"),
            ]:
                executor.submit_transport(
                    name,
                    source,
                    target,
                    1000,
                    "test",
                    None,
                    lambda name=name: completed.update({name: host.now}),
                )
            host.run()
            self.assertTrue(2.0 <= completed["up-u"] < 2.01)
            self.assertTrue(2.0 <= completed["up-v"] < 2.01)
            self.assertTrue(0.5 <= completed["down"] < 0.51)
            used = {
                p
                for pools in host.net.transport_options()["link_pools"].values()
                for p in pools
            }
            self.assertEqual(
                used, {"radio:sat:user_uplink_bps", "radio:sat:user_downlink_bps"}
            )
        finally:
            network.close()
            host.close()

    def test_ground_user_cannot_relay_between_satellites(self):
        from astermind.builder.config import SimulationConfig

        path = (
            Path(__file__).resolve().parents[3]
            / "examples/native/session_migration_tp2.json"
        )
        value = json.loads(path.read_text())
        value["physical"]["links"] = [
            link for link in value["physical"]["links"] if link["id"] != "isl"
        ]
        config = SimulationConfig.model_validate(value)
        graph = config.topology()
        host = DeviceRuntimeHost(
            graph, SimpleNamespace(**{**vars(config), "backend": "fluid"})
        )
        try:
            network = DeviceCommunicationRuntime(host, graph)
            with self.assertRaises(ValueError):
                network.route("gpu0", "gpu1", 1000)
            with self.assertRaises(ValueError):
                network.candidate_path("gpu0", "gpu1")
            self.assertEqual(network.route("user", "gpu0", 1000)["path"][0], "user")
        finally:
            host.close()

    def test_legacy_allocator_is_rejected_instead_of_ignoring_acquisition(self):
        from astermind.builder.config import SimulationConfig

        path = (
            Path(__file__).resolve().parents[3]
            / "examples/native/session_migration_tp2.json"
        )
        value = json.loads(path.read_text())
        value["physical"]["link_policy"] = {
            "mode": "simple_keep",
            "check_interval_ns": 100,
            "initial_acquisition_ns": 10,
            "reacquisition_ns": 10,
        }
        with self.assertRaisesRegex(ValueError, "canonical physical_network allocator"):
            SimulationConfig.model_validate(value)

    def test_intra_p2p_keeps_distinct_ports_and_explicit_noncompetition(self):
        completed, events, _, _ = run_transfers('runtime_comm')
        self.assertEqual(len(completed), 2)
        self.assertEqual(completed[0][1],completed[1][1])
        services=[event for event in events if event['kind']=='intra_service']
        self.assertEqual(len(services),2)
        self.assertTrue(all(not event['intra_competition'] for event in services))
        self.assertTrue(all(event['bandwidth_gbps']==0.000008 for event in services))
        self.assertEqual({tuple(event['path']) for event in services},
                         {('gpu0','switch','gpu1'),('gpu2','switch','gpu3')})

    def test_gpu_cannot_forward_between_two_switches(self):
        graph = topology()
        raw = graph.scenario.model_dump(mode="json")
        raw["devices"] += [
            {"id": "second-switch", "satellite": "sat", "kind": "switch"}
        ]
        raw["links"] = [
            link
            for link in raw["links"]
            if link["target"] != "switch" or link["source"] == "gpu0"
        ]
        raw["links"] += [
            {
                "id": "bridge",
                "source": "gpu0",
                "target": "second-switch",
                "bandwidth_gbps": 8.0,
            },
            {
                "id": "target",
                "source": "gpu1",
                "target": "second-switch",
                "bandwidth_gbps": 8.0,
            },
        ]
        split = PhysicalTopology(Scenario.model_validate(raw))
        config = SimpleNamespace(
            horizon_ns=1_000_000_000,
            backend="fluid",
            runtime_network={},
        communication=CommPolicy(runtime_protocol="payload_only",inter_parameters=dict(startup_us=0)),
            physical_link_bindings={},
        )
        host = DeviceRuntimeHost(split, config)
        try:
            network = DeviceCommunicationRuntime(host, split)
            with self.assertRaises(ValueError):
                network.candidate_path("switch", "gpu1")
        finally:
            host.close()


if __name__ == "__main__":
    unittest.main()
