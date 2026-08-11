import math
import unittest
from types import SimpleNamespace

from orbitserve.runtime.engine.routing import (
    C_KM_PER_S,
    COMMUNICATION_SHELL_ALTITUDE_KM,
    EARTH_RADIUS_KM,
    build_route_plan,
    communication_access_candidate,
    compute_access_candidates,
)
from orbitserve.runtime.setup import CommunicationIngressProfile


def _environment():
    return SimpleNamespace(
        ephemeris_samples=(
            {
                "time_s": 0.0,
                "satellite_id": "sat-equator",
                "position_itrs_km": [EARTH_RADIUS_KM + 600.0, 0.0, 0.0],
            },
            {
                "time_s": 0.0,
                "satellite_id": "sat-quarter-orbit",
                "position_itrs_km": [0.0, EARTH_RADIUS_KM + 600.0, 0.0],
            },
        )
    )


def _access(altitude_km=550.0):
    return communication_access_candidate(
        CommunicationIngressProfile(
            altitude_km=altitude_km,
            uplink_bandwidth_bps=1e9,
            downlink_bandwidth_bps=2e9,
        ),
        {
            "arrival_time_s": 0.0,
            "ingress_id": "gateway-0",
            "lat_deg": 0.0,
            "lon_deg": 0.0,
        },
        horizon_s=100.0,
    )


class FixedShellArcRoutingTests(unittest.TestCase):
    def test_route_uses_fixed_450_km_shell_and_compute_ephemeris(self):
        route = build_route_plan(
            _environment(),
            access=_access(),
            compute_satellite_id="sat-quarter-orbit",
            time_s=0.0,
            input_bytes=100.0,
            output_bytes=40.0,
            isl_bandwidth_bps=0.0,
        )

        expected_s = (
            COMMUNICATION_SHELL_ALTITUDE_KM
            + (EARTH_RADIUS_KM + COMMUNICATION_SHELL_ALTITUDE_KM) * (math.pi / 2.0)
            + abs(600.0 - COMMUNICATION_SHELL_ALTITUDE_KM)
        ) / C_KM_PER_S
        self.assertIsNotNone(route)
        assert route is not None
        self.assertAlmostEqual(route.inbound_legs[0].propagation_delay_s, expected_s)
        self.assertAlmostEqual(route.outbound_legs[0].propagation_delay_s, expected_s)
        self.assertEqual(route.provenance["communication_model"], "fixed_shell_arc")
        self.assertEqual(route.provenance["communication_shell_altitude_km"], 450.0)
        self.assertEqual(route.provenance["reachability_assumption"], "always_reachable")

    def test_fixed_shell_height_does_not_change_the_existing_ingress_contract(self):
        first = _access(altitude_km=450.0)
        legacy_height = _access(altitude_km=550.0)
        self.assertEqual(first.access_type, legacy_height.access_type)
        self.assertEqual(first.ingress_id, legacy_height.ingress_id)
        self.assertEqual(first.satellite_id, legacy_height.satellite_id)
        self.assertEqual(first.start_s, legacy_height.start_s)
        self.assertEqual(first.end_s, legacy_height.end_s)
        self.assertEqual(first.anchor_unit_vector, legacy_height.anchor_unit_vector)
        self.assertEqual(first.uplink_bandwidth_bps, legacy_height.uplink_bandwidth_bps)
        self.assertEqual(first.downlink_bandwidth_bps, legacy_height.downlink_bandwidth_bps)
        self.assertAlmostEqual(
            first.propagation_delay_s,
            COMMUNICATION_SHELL_ALTITUDE_KM / C_KM_PER_S,
        )
        self.assertEqual(first.propagation_delay_s, legacy_height.propagation_delay_s)

    def test_communication_route_is_available_without_sgl_or_isl_windows(self):
        route = build_route_plan(
            _environment(),
            access=_access(),
            compute_satellite_id="sat-equator",
            time_s=0.0,
            input_bytes=100.0,
            output_bytes=40.0,
            isl_bandwidth_bps=0.0,
        )
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(len(route.inbound_legs), 1)
        self.assertEqual(route.inbound_legs[0].kind, "communication_access")
        self.assertIsNone(route.inbound_legs[0].valid_until_s)

    def test_non_arc_ingress_selection_keeps_the_trace_access_input(self):
        environment = SimpleNamespace(
            ephemeris_samples=_environment().ephemeris_samples,
            sgl_windows=(
                {
                    "ingress_id": "gateway-0",
                    "satellite_id": "sat-equator",
                    "start_s": 0.0,
                    "end_s": 100.0,
                    "propagation_delay_s": 0.0125,
                },
            ),
            isl_windows=(),
        )
        access = compute_access_candidates(
            environment,
            ingress_id="gateway-0",
            time_s=10.0,
            uplink_bandwidth_bps=1e9,
            downlink_bandwidth_bps=2e9,
        )
        self.assertEqual(len(access), 1)
        self.assertEqual(access[0].access_type, "compute_satellite")
        self.assertEqual(access[0].satellite_id, "sat-equator")
        self.assertEqual(access[0].propagation_delay_s, 0.0125)


if __name__ == "__main__":
    unittest.main()
