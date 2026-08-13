import math
from types import SimpleNamespace
import unittest

from orbitserve.orbit import ORBIT_TRACE_SCHEMA, OrbitTraceBundle, OrbitTraceManifest
from orbitserve.runtime.engine.routing import (
    C_KM_PER_S,
    COMMUNICATION_SHELL_ALTITUDE_KM,
    EARTH_RADIUS_KM,
    AccessCandidate,
    build_route_plan,
    communication_access_candidate,
)


def _trace():
    manifest = OrbitTraceManifest(
        schema_version=ORBIT_TRACE_SCHEMA,
        trace_model="unit_geometry",
        evaluated=True,
        trace_id="fixed-shell-arc-unit-trace",
        time={"epoch": "2026-01-01T00:00:00Z", "start_s": 0.0, "end_s": 100.0, "duration_s": 100.0, "step_s": 1.0},
        files={}, sha256={}, valid_for_paper_main_results=False,
        scope="orbit_geometry_only", warnings=(), provenance={},
    )
    return OrbitTraceBundle(
        trace_manifest=manifest,
        satellite_catalog=({"satellite_id": "sat-a", "compute_capable": True}, {"satellite_id": "sat-b", "compute_capable": True}),
        ephemeris_samples=(
            {"time_s": 0.0, "satellite_id": "sat-a", "position_itrs_km": [7_000.0, 0.0, 0.0]},
            {"time_s": 0.0, "satellite_id": "sat-b", "position_itrs_km": [0.0, 7_000.0, 0.0]},
        ),
        sgl_windows=({"ingress_id": "gateway-0", "satellite_id": "sat-a", "start_s": 0.0, "end_s": 100.0, "propagation_delay_s": 0.01},),
        isl_windows=({"endpoint_a": "sat-a", "endpoint_b": "sat-b", "start_s": 0.0, "end_s": 100.0, "propagation_delay_s": 0.03, "distance_km": 9_000.0},),
        sunlight_windows=(),
    )


def _request():
    return {"request_id": "request-0", "arrival_time_s": 0.0, "lat_deg": 0.0, "lon_deg": 0.0, "ingress_id": "gateway-0"}


class FixedShellArcRoutingTests(unittest.TestCase):
    def test_fixed_shell_arc_delay_is_independent_of_configured_profile_altitude(self):
        trace = _trace()
        profile_550 = SimpleNamespace(configured=True, altitude_km=550.0, uplink_bandwidth_bps=1e9, downlink_bandwidth_bps=2e9)
        profile_900 = SimpleNamespace(configured=True, altitude_km=900.0, uplink_bandwidth_bps=1e9, downlink_bandwidth_bps=2e9)
        access_550 = communication_access_candidate(profile_550, _request(), 100.0)
        access_900 = communication_access_candidate(profile_900, _request(), 100.0)
        plan_550 = build_route_plan(trace, access=access_550, compute_satellite_id="sat-b", time_s=0.0, input_bytes=100.0, output_bytes=20.0, isl_bandwidth_bps=1e9)
        plan_900 = build_route_plan(trace, access=access_900, compute_satellite_id="sat-b", time_s=0.0, input_bytes=100.0, output_bytes=20.0, isl_bandwidth_bps=1e9)

        compute_altitude_km = 7_000.0 - EARTH_RADIUS_KM
        expected = (
            COMMUNICATION_SHELL_ALTITUDE_KM
            + (EARTH_RADIUS_KM + COMMUNICATION_SHELL_ALTITUDE_KM) * (math.pi / 2.0)
            + abs(compute_altitude_km - COMMUNICATION_SHELL_ALTITUDE_KM)
        ) / C_KM_PER_S
        self.assertAlmostEqual(plan_550.inbound_legs[0].propagation_delay_s, expected)
        self.assertAlmostEqual(plan_550.outbound_legs[0].propagation_delay_s, expected)
        self.assertEqual(plan_550.to_dict(), plan_900.to_dict())
        self.assertEqual(plan_550.provenance["reachability_assumption"], "always_reachable")

    def test_same_endpoint_uses_fixed_shell_radial_legs(self):
        trace = _trace()
        profile = SimpleNamespace(configured=True, altitude_km=550.0, uplink_bandwidth_bps=1e9, downlink_bandwidth_bps=1e9)
        access = communication_access_candidate(profile, _request(), 100.0)
        plan = build_route_plan(trace, access=access, compute_satellite_id="sat-a", time_s=0.0, input_bytes=100.0, output_bytes=20.0, isl_bandwidth_bps=1e9)

        expected = (COMMUNICATION_SHELL_ALTITUDE_KM + abs((7_000.0 - EARTH_RADIUS_KM) - COMMUNICATION_SHELL_ALTITUDE_KM)) / C_KM_PER_S
        self.assertAlmostEqual(plan.arc_distance_rad, 0.0)
        self.assertAlmostEqual(plan.inbound_legs[0].propagation_delay_s, expected)
        self.assertAlmostEqual(plan.outbound_legs[0].propagation_delay_s, expected)

    def test_compute_satellite_routing_is_unchanged(self):
        access = AccessCandidate(
            access_type="compute_satellite", ingress_id="gateway-0", satellite_id="sat-a",
            start_s=0.0, end_s=100.0, uplink_bandwidth_bps=1e9, downlink_bandwidth_bps=2e9,
            propagation_delay_s=0.01, anchor_unit_vector=(1.0, 0.0, 0.0),
        )
        plan = build_route_plan(_trace(), access=access, compute_satellite_id="sat-b", time_s=0.0, input_bytes=100.0, output_bytes=20.0, isl_bandwidth_bps=1e9)

        self.assertEqual(plan.access_type, "compute_satellite")
        self.assertEqual(plan.access_satellite_id, "sat-a")
        self.assertEqual(plan.route_satellite_ids, ("sat-a", "sat-b"))
        self.assertEqual(tuple(leg.kind for leg in plan.inbound_legs), ("sgl", "isl"))
        self.assertEqual(tuple(leg.propagation_delay_s for leg in plan.inbound_legs), (0.01, 0.03))
        self.assertEqual(tuple(leg.propagation_delay_s for leg in plan.outbound_legs), (0.03, 0.01))


if __name__ == "__main__":
    unittest.main()
