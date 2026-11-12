"""SimGrid evidence uses native double intervals, never an exact-as claim."""

import copy
import unittest

from astermind.engine.simgrid_events import native_time, simgrid_time
from astermind.engine.service_precision import bandwidth_check


class SimGridPrecisionContractTests(unittest.TestCase):
    def record(self):
        start = 0.006621239990538873
        finish = 0.006621751990538874
        return dict(
            interval_start_s=simgrid_time(start),
            interval_end_s=simgrid_time(finish),
            allocated_bandwidth_bps=512 * 8 / (finish - start),
            link_capacity_bps=8e9,
            bytes_sent=512,
            service_clock="simgrid_float64_seconds",
            native_interval_start_s=start,
            native_interval_end_s=finish,
            native_service_error_bytes=1e-6,
        )

    def test_native_clock_roundtrip_cannot_move_a_current_callback(self):
        for value in (0.006621751990538874, 0.1, 1 / 3, 8.659443882295493):
            self.assertEqual(native_time(simgrid_time(value)), value)
            self.assertEqual(float(simgrid_time(value)), value)

    def test_migration_barrier_uses_exact_stamps_and_legacy_ns_projection(self):
        from astermind.engine.clock_audit import migration_barrier_valid

        migration = dict(
            network_complete_time_s=1.458470516301513,
            completed_time_s=1.4584718283328155,
        )
        event = dict(
            kind="completed",
            completion_time_ns=1458470517,
            time_ticks="1458470516301513000",
        )
        self.assertTrue(migration_barrier_valid(migration, [event]))
        migration.update(
            network_complete_time_ticks=event["time_ticks"],
            completed_time_ticks="1458471828332815500",
        )
        self.assertTrue(migration_barrier_valid(migration, [event]))
        migration["network_complete_time_ticks"] = str(int(event["time_ticks"]) - 1)
        self.assertFalse(migration_barrier_valid(migration, [event]))
        migration["network_complete_time_ticks"] = event["time_ticks"]
        migration["completed_time_ticks"] = str(int(event["time_ticks"]) - 1)
        self.assertFalse(migration_barrier_valid(migration, [event]))

    def test_native_service_passes_without_reinterpreting_projected_ticks(self):
        row = self.record()
        row.update(
            interval_start_ticks=str(row["interval_start_s"].ticks),
            interval_end_ticks=str(row["interval_end_s"].ticks),
        )
        self.assertEqual(bandwidth_check(row)["state"], "pass")

    def test_real_capacity_violation_or_changed_bytes_are_rejected(self):
        for key, value in [
            ("link_capacity_bps", 4e9),
            ("bytes_sent", 600),
            ("allocated_bandwidth_bps", 12e9),
            ("native_service_error_bytes", -1),
            ("native_interval_start_s", 0),
        ]:
            with self.subTest(key=key):
                row = copy.copy(self.record())
                row[key] = value
                self.assertEqual(bandwidth_check(row)["state"], "fail")

    def test_legacy_ns_callback_only_aligns_the_rounded_current_instant(self):
        from types import SimpleNamespace
        from astermind.engine.clock import Time
        from astermind.engine.cell_bridge import RuntimeCellNetwork, CellTransfer

        now = Time.from_ticks(1500166782047599077)
        network = RuntimeCellNetwork(
            SimpleNamespace(now=now, net=SimpleNamespace(nodes={}))
        )
        request = CellTransfer("legacy", "a", "b", 1, network.now_ns)
        self.assertEqual(network.ready_time(request), now)
        with self.assertRaisesRegex(ValueError, "past"):
            network.ready_time(
                CellTransfer("past", "a", "b", 1, network.now_ns - 0.001)
            )
        with self.assertRaisesRegex(ValueError, "past"):
            network.ready_time(
                CellTransfer(
                    "exact-past",
                    "a",
                    "b",
                    1,
                    network.now_ns,
                    ready_time_ticks=str(now.ticks - 1),
                )
            )
