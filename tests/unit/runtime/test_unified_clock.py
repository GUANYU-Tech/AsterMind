"""Sub-nanosecond causality, origin independence and strict capacity evidence."""

import unittest
from orbitserve.runtime.clock import CLOCK, Time
from orbitserve.runtime.timeline import GlobalTimeline
from orbitserve.runtime.engine.flowsim_transport import FlowSimTransport
from orbitserve.runtime.setup.network_spec import NetworkNodeSpec
from orbitserve.runtime.service_precision import bandwidth_check
from orbitserve.runtime.clock_audit import clock_checks


class UnifiedClockTests(unittest.TestCase):
    def test_thousand_simultaneous_flows_share_capacity_and_complete_together(self):
        engine, leg = self.transport()
        for index in range(1000):
            engine.start(str(index), leg, 4)
        response = engine.advance_until(1)
        self.assertEqual(len(response['completed']), 1000)
        self.assertEqual(set(response['bytes_sent'].values()), {4.})
        self.assertEqual(engine.now.ticks, 160000000000)
        for resource in response['resources']:
            self.assertLessEqual(resource['allocated_bandwidth_bps'], resource['capacity_bps'])

    def transport(self):
        nodes = {name: NetworkNodeSpec(name, "compute") for name in ("a", "b")}
        engine = FlowSimTransport(
            nodes, [dict(link_id="ab", endpoint_a="a", endpoint_b="b")], {}
        )
        self.addCleanup(engine.close)
        leg = dict(link_id="ab", source_node_id="a", target_node_id="b")
        engine.configure({engine.resource(leg): 200e9})
        return engine, leg

    def test_large_origin_has_exact_same_service_and_does_not_merge_events(self):
        durations = []
        for origin in (0, 8.659443882295493, 1000000000):
            engine, leg = self.transport()
            engine.advance_until(origin)
            first = engine.now
            engine.start("four-bytes", leg, 4)
            response = engine.advance_until(first + 1)
            durations.append(engine.now.ticks - first.ticks)
            self.assertEqual(response["bytes_sent"]["four-bytes"], 4)
            self.assertEqual(response["allocated_rates_bps"]["four-bytes"], 200e9)
            loop = GlobalTimeline(start_s=first)
            observed = []
            loop.schedule(
                engine.now, 0, "first", lambda: observed.append(loop.now_s.ticks)
            )
            loop.schedule(
                engine.now + 1e-18,
                0,
                "second",
                lambda: observed.append(loop.now_s.ticks),
            )
            loop.run()
            self.assertEqual(observed[1] - observed[0], 1)
        self.assertEqual(durations, [160000000] * 3)

    def test_shared_capacity_and_conservation_under_staggered_arrivals(self):
        engine, leg = self.transport()
        total = {}
        engine.start("a", leg, 4)
        response = engine.advance_until(Time(".00000000008"))
        total["a"] = response["bytes_sent"]["a"]
        engine.start("b", leg, 4)
        while engine.active:
            response = engine.advance_until(1)
            for row in response["resources"]:
                self.assertLessEqual(
                    row["allocated_bandwidth_bps"], row["capacity_bps"]
                )
            for key, value in response["bytes_sent"].items():
                total[key] = total.get(key, 0) + value
        self.assertEqual(total, {"a": 4, "b": 4})
        self.assertEqual(engine.now.ticks, 320000000)

    def test_same_time_priority_and_sequence_and_past_rejection(self):
        loop = GlobalTimeline(start_s=8)
        now = loop.now_s
        seen = []
        for name, priority in [("compute", 10), ("fault", -2), ("a", 10), ("b", 10)]:
            loop.schedule(
                now + 1e-18, priority, name, lambda name=name: seen.append(name)
            )
        loop.run()
        self.assertEqual(seen, ["fault", "compute", "a", "b"])
        with self.assertRaises(ValueError):
            loop.schedule(now, 0, "past", lambda: None)

    def test_no_precision_warning_can_hide_real_capacity_violation(self):
        row = dict(
            interval_start_s=8.0,
            interval_end_s=8.0,
            interval_start_ticks=str(8 * 10**18),
            interval_end_ticks=str(8 * 10**18 + 160000000),
            allocated_bandwidth_bps=200e9,
            link_capacity_bps=200e9,
            bytes_sent=4.0,
        )
        row["interval_end_s"] = float(Time.from_ticks(int(row["interval_end_ticks"])))
        self.assertEqual(bandwidth_check(row)["state"], "pass")
        row["bytes_sent"] = 4.01
        self.assertEqual(bandwidth_check(row)["state"], "fail")
        result = dict(
            network=dict(
                clock=CLOCK,
                event_log=[
                    dict(
                        kind="resource_service",
                        time_ticks="0",
                        resources=[
                            dict(
                                resource_id="pool",
                                allocated_bandwidth_bps=201e9,
                                capacity_bps=200e9,
                            )
                        ],
                    )
                ],
            )
        )
        self.assertEqual(
            next(c for c in clock_checks(result) if c["id"] == "shared_capacity")[
                "state"
            ],
            "fail",
        )
