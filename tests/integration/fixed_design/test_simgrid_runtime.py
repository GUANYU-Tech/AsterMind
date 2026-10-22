"""Run actual shared transmissions through the chronological business engine."""

import importlib.util
import unittest
from dataclasses import replace

from orbitserve.runtime import LinkProfile, run_simulation
from tests.unit.runtime.test_runtime_engine import _scenario, _request
from tests.unit.runtime.test_runtime_event_order import strategy


def scenario(mode="simgrid", arrivals=(0.0, 0.0), sizes=None):
    requests = tuple(
        {
            **_request(f"request-{i}", time, session_id=f"session-{i}"),
            "input_bytes": sizes[i] if sizes else 1000.0,
        }
        for i, time in enumerate(arrivals)
    )
    spec = _scenario(strategy(), requests=requests)
    profiles = {**spec.network.link_profiles, "sgl": LinkProfile(8000, 0)}
    return replace(
        spec,
        network=replace(spec.network, transport_model=mode, link_profiles=profiles),
    )


def inputs(result):
    return [r["routing"]["selected_path"][0] for r in result.request_records]


@unittest.skipUnless(
    importlib.util.find_spec("simgrid"),
    "Install OrbitServe[simgrid] to run native backend integration",
)
class SimGridRuntimeTests(unittest.TestCase):
    def test_queue_and_native_sharing_have_different_causal_events(self):
        queue = run_simulation(scenario("queue"))
        shared = run_simulation(scenario())
        self.assertEqual(shared.request_summary["completed_request_count"], 2)
        self.assertEqual([link["finish_s"] for link in inputs(queue)], [1.0, 2.0])
        for hop in inputs(shared):
            self.assertEqual(hop["start_s"], 0.0)
            self.assertAlmostEqual(hop["finish_s"], 2.0, places=8)
            self.assertAlmostEqual(
                sum(s["bytes"] for s in hop["segments"]), 1000.0, places=6
            )
        starts = [
            e
            for e in shared.state_summary["event_log"]
            if e["kind"] == "transmit_start"
        ]
        self.assertTrue(all(e["finish_s"] is None for e in starts))
        self.assertEqual(shared.provenance["network_transport"]["backend"], "simgrid")
        self.assertEqual(shared.state_summary["network"]["directed_reservations"], [])

    def test_midway_join_recomputes_completion_and_records_shared_rates(self):
        result = run_simulation(scenario(arrivals=(0.0, 0.5)))
        a, b = inputs(result)
        self.assertAlmostEqual(a["finish_s"], 1.5, places=8)
        self.assertAlmostEqual(b["finish_s"], 2.0, places=8)
        self.assertTrue(
            any(abs(s["bandwidth_bps"] - 4000) < 1e-5 for s in a["segments"])
        )
        times = [e["time_s"] for e in result.state_summary["event_log"]]
        self.assertEqual(times, sorted(times))
        self.assertTrue(
            any(
                e["kind"] == "bandwidth_update"
                for e in result.state_summary["event_log"]
            )
        )

    def test_capacity_changes_and_zero_capacity_keep_transmitted_progress(self):
        spec = scenario(arrivals=(0.0,))
        spec = replace(
            spec,
            network=replace(
                spec.network,
                link_profiles={
                    **spec.network.link_profiles,
                    "sgl": LinkProfile(8000, 0, capacity_schedule=((0.5, 0), (1.5, 1))),
                },
            ),
        )
        result = run_simulation(spec)
        hop = inputs(result)[0]
        self.assertAlmostEqual(hop["finish_s"], 2.0, places=8)
        self.assertAlmostEqual(
            sum(s["bytes"] for s in hop["segments"]), 1000.0, places=6
        )
        self.assertTrue(any(s["bytes"] == 0 for s in hop["segments"]))

    def test_unexpected_contact_loss_cancels_without_inventing_packet_loss(self):
        spec = scenario(arrivals=(0.0, 0.25), sizes=(1000, 250))
        trace = spec.environment_state
        windows = tuple(
            {**w, "end_s": 1.1} if w["ingress_id"] == "gateway-0" else w
            for w in trace.sgl_windows
        ) + tuple(
            {**w, "start_s": 3.0}
            for w in trace.sgl_windows
            if w["ingress_id"] == "gateway-0"
        )
        result = run_simulation(
            replace(spec, environment_state=replace(trace, sgl_windows=windows))
        )
        self.assertEqual(
            result.request_records[0]["outcome"]["uncompleted_reason"],
            "network_contact_lost_during_transmission",
        )
        events = result.state_summary["event_log"]
        self.assertTrue(any(e["kind"] == "transfer_cancelled" for e in events))
        self.assertFalse(any(e["kind"] == "transfer_dropped" for e in events))

    def test_horizon_cancels_active_work_and_never_reports_false_completion(self):
        result = run_simulation(replace(scenario(arrivals=(0.0,)), time_horizon_s=0.5))
        self.assertEqual(result.request_summary["completed_request_count"], 0)
        self.assertEqual(
            result.request_records[0]["outcome"]["uncompleted_reason"],
            "simulation_horizon_exceeded",
        )

    def test_overlap_cannot_observe_future_kv_in_native_mode(self):
        spec = _scenario(
            strategy(),
            requests=(
                _request("a", 0),
                _request("b", 0.1, context_tokens_before_turn=10),
            ),
        )
        result = run_simulation(
            replace(spec, network=replace(spec.network, transport_model="simgrid"))
        )
        self.assertEqual(result.request_summary["completed_request_count"], 2)
        self.assertEqual(result.request_records[1]["kv"]["kv_hit_tokens"], 0)
