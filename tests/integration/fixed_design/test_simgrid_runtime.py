"""Run actual shared transmissions through the chronological business engine."""

import importlib.util
import unittest
from dataclasses import replace

from astermind.engine import LinkProfile
from tests.network_case import NetworkCase, traffic, run_traffic


def scenario(mode="simgrid", arrivals=(0.0, 0.0), sizes=None):
    requests = tuple(
        {
            **traffic(f"request-{i}", time),
            "input_bytes": sizes[i] if sizes else 1000.0,
        }
        for i, time in enumerate(arrivals)
    )
    spec = NetworkCase(traffic=requests)
    profiles = {**spec.network.link_profiles, "sgl": LinkProfile(8000, 0)}
    return replace(
        spec,
        network=replace(spec.network, transport_model=mode, link_profiles=profiles),
    )


def inputs(result):
    return [r["path"][0] for r in result.transfer_jobs]


@unittest.skipUnless(
    importlib.util.find_spec("simgrid"),
    "Install OrbitServe[simgrid] to run native backend integration",
)
class SimGridRuntimeTests(unittest.TestCase):
    def test_queue_and_native_sharing_have_different_causal_events(self):
        queue = run_traffic(scenario("queue"))
        shared = run_traffic(scenario())
        self.assertEqual(shared.completed, 2)
        self.assertEqual([link["finish_s"] for link in inputs(queue)], [1.0, 2.0])
        for hop in inputs(shared):
            self.assertEqual(hop["start_s"], 0.0)
            self.assertAlmostEqual(hop["finish_s"], 2.0, places=8)
            self.assertAlmostEqual(
                sum(s["bytes"] for s in hop["segments"]), 1000.0, places=6
            )
        starts = [
            e
            for e in shared.log
            if e["kind"] == "transmit_start"
        ]
        self.assertTrue(all(e["finish_s"] is None for e in starts))
        self.assertEqual(shared.transport_provenance["backend"], "simgrid")
        self.assertEqual(shared.net.calendars, {})

    def test_midway_join_recomputes_completion_and_records_shared_rates(self):
        result = run_traffic(scenario(arrivals=(0.0, 0.5)))
        a, b = inputs(result)
        self.assertAlmostEqual(a["finish_s"], 1.5, places=8)
        self.assertAlmostEqual(b["finish_s"], 2.0, places=8)
        self.assertTrue(
            any(abs(s["bandwidth_bps"] - 4000) < 1e-5 for s in a["segments"])
        )
        times = [e["time_s"] for e in result.log]
        self.assertEqual(times, sorted(times))
        self.assertTrue(
            any(
                e["kind"] == "bandwidth_update"
                for e in result.log
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
        result = run_traffic(spec)
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
        result = run_traffic(
            replace(spec, environment_state=replace(trace, sgl_windows=windows))
        )
        self.assertEqual(
            result.transfer_jobs[0]["failure"],
            "network_contact_lost_during_transmission",
        )
        events = result.log
        self.assertTrue(any(e["kind"] == "transfer_cancelled" for e in events))
        self.assertFalse(any(e["kind"] == "transfer_dropped" for e in events))

    def test_horizon_cancels_active_work_and_never_reports_false_completion(self):
        result = run_traffic(replace(scenario(arrivals=(0.0,)), time_horizon_s=0.5))
        self.assertEqual(result.completed, 0)
        self.assertEqual(
            result.transfer_jobs[0]["failure"],
            "simulation_horizon_exceeded",
        )
