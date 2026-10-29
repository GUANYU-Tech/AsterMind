"""Mentor acceptance: one frozen constellation and canonical workload, 3x4 matrix.

Uses real SGP4 propagation from synthetic mean elements, with one frozen trace
reused across every backend. Small RF capacities isolate networking invariants;
this is a correctness case, not a vendor-throughput or orbit-accuracy benchmark.
"""

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from orbitserve.orbit import generate_orbit_trace, load_orbit_trace
from orbitserve.runtime import (
    LinkProfile,
    NetworkNodeSpec,
    NetworkSpec,
    NetworkState,
    run_simulation,
)
from orbitserve.workload import load_workload_bundle
from tests.unit.orbit.test_sgp4_elements import _bundle, _satellite
from tests.unit.runtime.test_runtime_engine import _scenario
from tests.unit.runtime.test_runtime_event_order import strategy

POLICIES = ("earliest_arrival", "min_hops", "min_propagation")


@lru_cache(maxsize=1)
def orbital_trace(ingress_id, lat_deg, lon_deg):
    elements = _bundle()
    elements["satellites"] = [
        _satellite(n, i) for i, n in enumerate(("sat-a", "sat-b", "sat-c"), 10001)
    ]
    # Keep sat-a unambiguously closer at initial association. With 0.01 degrees
    # the rounded gateway position made sat-b closer by only 2 cm, so retain_valid
    # correctly kept sat-b and the later fault on sat-a never hit a transfer.
    elements["satellites"][1]["mean_anomaly_deg"] = 0.1
    elements["satellites"][2]["mean_motion_rev_per_day"] = 14.3
    with tempfile.TemporaryDirectory() as folder:
        generate_orbit_trace(
            folder,
            elements=elements,
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            sample_times_s=tuple(float(t) for t in range(121)),
            ground_ingresses=(
                {
                    "ingress_id": ingress_id,
                    "lat_deg": lat_deg,
                    "lon_deg": lon_deg,
                    "min_elevation_deg": 5.0,
                },
            ),
            isl_pairs=(("sat-a", "sat-b"), ("sat-a", "sat-c"), ("sat-b", "sat-c")),
            workers=1,
        )
        return load_orbit_trace(Path(folder) / "trace_manifest.json")


def acceptance_spec():
    workload = load_workload_bundle(
        Path(__file__).resolve().parents[2] / "fixtures/workloads/coupled_physical"
    ).to_dict()
    base = _scenario(strategy(), requests=workload["session_requests"])
    gateway = {
        **workload["gateways"][0],
        "ingress_id": workload["gateways"][0]["gateway_id"],
    }
    ids = ("sat-a", "sat-b", "sat-c")
    trace = orbital_trace(gateway["ingress_id"], gateway["lat_deg"], gateway["lon_deg"])
    fault = {
        "target_type": "link",
        "target_id": f"sgl:{gateway['ingress_id']}:sat-a",
        "start_s": 10.0,
        "end_s": 60.0,
    }
    nodes = tuple(
        NetworkNodeSpec(
            n,
            "communication",
            4,
            1,
            radio_capacity_bps={
                "feeder_uplink_bps": 256.0,
                "feeder_downlink_bps": 1024.0,
            },
            capacity_assumptions=(
                "Scaled RF aggregate for acceptance, not vendor parameters.",
            ),
        )
        for n in ids[:2]
    )
    nodes += (
        NetworkNodeSpec("sat-c", "compute", 4),
        NetworkNodeSpec(gateway["ingress_id"], "ground", sgl_terminals=1),
    )
    network = NetworkSpec(
        nodes,
        {
            "communication_isl": LinkProfile(8000.0, 0.5),
            "cross_layer": LinkProfile(8000.0, 0.5),
            "sgl": LinkProfile(
                256.0,
                0.5,
                capacity_schedule=((12.2, 0.5), (18.0, 1.0)),
                downlink_profile="downlink",
            ),
            "downlink": LinkProfile(1024.0, 0.0),
        },
        communication_topology="explicit",
        explicit_pairs=(("sat-a", "sat-b"),),
        faults=(fault,),
        failure_policy="retry_next_sample",
        max_transfer_retries=12,
    )
    return replace(
        base,
        network=network,
        environment_state=trace,
        gateways=tuple(workload["gateways"]),
        workload_bundle=workload,
        time_horizon_s=120.0,
    )


def run_acceptance(mode, policy):
    base = acceptance_spec()
    return run_simulation(
        replace(
            base,
            network=replace(base.network, transport_model=mode),
            strategy=replace(base.strategy, routing_policy=policy),
        )
    )


class NetworkAcceptanceTests(unittest.TestCase):
    def test_fault_target_is_the_initially_selected_contact(self):
        base = acceptance_spec()
        state = NetworkState(base.environment_state, base.network, base.gateways)
        target = base.network.faults[0]["target_id"]
        backup = "sgl:ground-00:sat-b"
        for time_s in (0.5, 7.0, 9.0):
            with self.subTest(time_s=time_s):
                self.assertEqual(state.link_at(target, time_s)["state"], "UP")
                self.assertEqual(state.link_at(backup, time_s)["state"], "DOWN")
        self.assertLess(
            state.distance(state.edge_by_id[target], 0.0),
            state.distance(state.edge_by_id[backup], 0.0) - 0.01,
        )

    def verify_matrix_row(self, mode, policy):
        result = run_acceptance(mode, policy)
        self.assertEqual(
            result.request_summary["completed_request_count"], 3, result.request_records
        )
        events = result.state_summary["event_log"]
        self.assertEqual(
            [e["time_s"] for e in events], sorted(e["time_s"] for e in events)
        )
        losses = [
            e
            for e in events
            if e["kind"] == "transfer_cancelled" and e["time_s"] == 10.0
        ]
        self.assertTrue(losses)
        self.assertTrue(
            any(e["kind"] == "route_recovery" and e["time_s"] == 10.0 for e in events)
        )
        paths = [
            leg for r in result.request_records for leg in r["routing"]["selected_path"]
        ]
        backup = [
            leg
            for leg in paths
            if leg["source_node_id"] == "ground-00" and leg["target_node_id"] == "sat-b"
        ]
        self.assertTrue(backup)
        self.assertAlmostEqual(min(leg["start_s"] for leg in backup), 10.5)
        self.assertTrue(
            any(
                leg["contact_ready_s"] == 10.5 and leg["acquisition_wait_s"] == 0.5
                for leg in backup
            )
        )
        self.assertTrue(any(leg["queue_wait_s"] >= 0.5 - 1e-9 for leg in backup))
        for leg in paths:
            observed = sum(s["bytes"] for s in leg["segments"])
            expected = (
                leg["bytes_discarded"]
                if "cancelled_at_s" in leg
                else leg["payload_bytes"]
            )
            self.assertAlmostEqual(observed, expected, places=5, msg=str(leg))
            if "cancelled_at_s" not in leg:
                self.assertGreaterEqual(leg["arrival_s"], leg["finish_s"])
        # Every transmitted byte is either delivered to the next hop or lost on
        # the interrupted hop; delivered prefixes must never be subtracted.
        self.assertAlmostEqual(
            sum(s["bytes"] for link in paths for s in link["segments"]),
            sum(link["payload_bytes"] for link in paths if "cancelled_at_s" not in link)
            + sum(
                e["bytes_discarded"]
                for e in events
                if e["kind"] == "transfer_cancelled"
            ),
            places=5,
        )
        if mode != "queue":
            self.assertTrue(
                any(
                    s["concurrent_flows"] > 1
                    for link in paths
                    for s in link["segments"]
                )
            )
            capacities = [
                s["link_capacity_bps"] for link in backup for s in link["segments"]
            ]
            self.assertIn(128.0, capacities)
            self.assertIn(256.0, capacities)
        else:
            segments = [s for link in backup for s in link["segments"]]
            self.assertTrue(any(s["bandwidth_bps"] == 128.0 for s in segments))
        base = acceptance_spec()
        state = NetworkState(base.environment_state, base.network, base.gateways)
        edge = state.edges[0]
        self.assertNotEqual(state.distance(edge, 8.0), state.distance(edge, 9.0))
        return result

    def test_queue_three_routes(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                self.verify_matrix_row("queue", policy)

    def test_fluid_three_routes(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                self.verify_matrix_row("fluid", policy)

    @unittest.skipUnless(importlib.util.find_spec("simgrid"), "requires native SimGrid")
    def test_simgrid_three_routes(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                self.verify_matrix_row("simgrid", policy)

    @unittest.skipUnless(importlib.util.find_spec("ns"), "requires native ns-3")
    def test_ns3_three_routes(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                self.verify_matrix_row("ns3", policy)
