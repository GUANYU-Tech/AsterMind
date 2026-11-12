import unittest

from astermind.engine.service_precision import bandwidth_check


class ServicePrecisionTests(unittest.TestCase):
    def test_recorded_four_byte_subnanosecond_interval_is_a_precision_warning(self):
        event = dict(
            interval_start_s=8.659443882295493,
            interval_end_s=8.659443882455491,
            allocated_bandwidth_bps=200002203922.26114,
            link_capacity_bps=200e9,
            bytes_sent=4.0,
            payload_bytes=4.0,
        )
        result = bandwidth_check(event, native_ns_clock=True)
        self.assertEqual(result["state"], "precision")
        self.assertLessEqual(result["excess_bytes"], result["precision_bound_bytes"])

    def test_real_overallocation_cannot_be_excused_by_clock_rounding(self):
        event = dict(
            interval_start_s=10.0,
            interval_end_s=11.0,
            allocated_bandwidth_bps=8800.0,
            link_capacity_bps=8000.0,
            bytes_sent=1100.0,
            payload_bytes=1100.0,
        )
        self.assertEqual(bandwidth_check(event, native_ns_clock=True)["state"], "fail")

    def test_invalid_bytes_or_inconsistent_recorded_rate_still_fail(self):
        event = dict(
            interval_start_s=1.0,
            interval_end_s=2.0,
            allocated_bandwidth_bps=8000.0,
            link_capacity_bps=8000.0,
            bytes_sent=1000.0,
            payload_bytes=1000.0,
        )
        self.assertEqual(bandwidth_check(event)["state"], "pass")
        for replacement in (
            dict(bytes_sent=-1),
            dict(bytes_sent=1200),
            dict(allocated_bandwidth_bps=float("nan")),
            dict(interval_end_s=0.5),
        ):
            self.assertEqual(
                bandwidth_check(dict(event, **replacement), native_ns_clock=True)[
                    "state"
                ],
                "fail",
            )

    def test_native_clock_duration_is_checked_against_global_time(self):
        event = dict(
            interval_start_s=1.0,
            interval_end_s=2.0,
            allocated_bandwidth_bps=8000.0,
            link_capacity_bps=8000.0,
            bytes_sent=1000.0,
            payload_bytes=1000.0,
            native_interval_start_ns=1e9,
            native_interval_end_ns=2e9,
        )
        self.assertEqual(bandwidth_check(event, native_ns_clock=True)["state"], "pass")
        event["native_interval_end_ns"] = 3e9
        self.assertEqual(bandwidth_check(event, native_ns_clock=True)["state"], "fail")
