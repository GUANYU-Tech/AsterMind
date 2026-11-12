"""Sub-nanosecond causality, origin independence and strict capacity evidence."""

import unittest
from astermind.engine.clock import CLOCK, Time
from astermind.engine.timeline import GlobalTimeline
from astermind.engine.service_precision import bandwidth_check
from astermind.engine.clock_audit import clock_checks


class UnifiedClockTests(unittest.TestCase):




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
