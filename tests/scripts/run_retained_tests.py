"""Run retained tests with the user's exact FlowSim exclusions; safe for spawn."""

from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    excluded = {
        "tests.unit.runtime.test_scoped_protobridge.ScopedProtoBridgeTests.test_native_host_registers_control_boundaries_on_shared_clock",
        "tests.unit.runtime.test_scoped_protobridge.ScopedProtoBridgeTests.test_bandwidth_change_restarts_inflight_surrogate_without_early_completion",
        "tests.unit.runtime.test_scoped_protobridge.ScopedProtoBridgeTests.test_future_ready_and_queued_cancellation_keep_exactly_one_terminal_event",
        "tests.unit.runtime.test_unified_clock.UnifiedClockTests.test_shared_capacity_and_conservation_under_staggered_arrivals",
        "tests.unit.runtime.test_flowsim_transport.FlowSimTransportTests.test_shared_radio_pool_across_distinct_links",
        "tests.unit.runtime.test_scoped_protobridge.ScopedProtoBridgeTests.test_same_cell_name_cannot_hide_cross_cell_participant",
        "tests.unit.runtime.test_flowsim_transport.FlowSimTransportTests.test_advance_reaches_exact_host_time_after_unit_conversion",
        "tests.unit.runtime.test_flowsim_transport.FlowSimTransportTests.test_capacity_change_zero_pause_and_cancel",
        "tests.unit.runtime.test_scoped_protobridge.ScopedProtoBridgeTests.test_three_scopes_always_keep_cross_cell_on_flowsim",
        "tests.unit.runtime.test_flowsim_transport.FlowSimTransportTests.test_tiny_service_records_its_native_clock_interval",
        "tests.unit.runtime.test_flowsim_statistics.FlowSimStatisticsTests.test_capacity_integral_includes_idle_links_and_reports_are_read_only",
        "tests.unit.runtime.test_flowsim_transport.FlowSimTransportTests.test_flowsim_is_default_transport",
        "tests.unit.runtime.test_flowsim_transport.FlowSimTransportTests.test_sub_ulp_native_completion_drains_fractional_tail",
        "tests.unit.runtime.test_flowsim_statistics.FlowSimStatisticsTests.test_cancel_before_first_advance_does_not_create_busy_interval",
        "tests.unit.runtime.test_device_network.DeviceNetworkTests.test_real_flowsim_shares_same_satellite_switch",
        "tests.unit.runtime.test_unified_clock.UnifiedClockTests.test_thousand_simultaneous_flows_share_capacity_and_complete_together",
        "tests.unit.runtime.test_device_controls.DeviceControlsTests.test_gpu_link_fault_recovery_reaches_both_communication_solvers",
        "tests.integration.fixed_design.test_cell_shared_runtime.CellSharedRuntimeTests.test_cell_protocol_with_real_flowsim",
        "tests.unit.runtime.test_scoped_protobridge.ScopedProtoBridgeTests.test_collective_waits_and_external_transfer_respects_reserved_fabric",
        "tests.unit.runtime.test_unified_clock.UnifiedClockTests.test_large_origin_has_exact_same_service_and_does_not_merge_events",
    }
    assert len(excluded) == 20, excluded
    loader = unittest.TestLoader()
    suite = loader.discover("tests", top_level_dir=".")
    assert not loader.errors, loader.errors

    def flatten(items):
        for item in items:
            if isinstance(item, unittest.TestSuite):
                yield from flatten(item)
            else:
                yield item

    tests = list(flatten(suite))
    found = {test.id() for test in tests}
    assert excluded <= found, excluded - found
    selected = [test for test in tests if test.id() not in excluded]
    print("USER-EXCLUDED FLOWSIM TESTS", len(excluded), flush=True)
    for name in sorted(excluded):
        print(name, flush=True)
    print("SELECTED", len(selected), "IMPORT ERRORS", len(loader.errors), flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
    if result.skipped:
        print("UNEXPECTED SKIPS", result.skipped, flush=True)
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
