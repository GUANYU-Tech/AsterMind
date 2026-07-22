import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


from orbitserve.runtime.routing import (
    RouteLeg,
    find_access_candidates,
    query_link_sample_at,
    sample_route_leg,
)
from orbitserve.runtime.scenario import build_runtime_scenario
from orbitserve.orbit import (
    OrbitTraceBundle as PublicOrbitTraceBundle,
    OrbitTraceManifest as PublicOrbitTraceManifest,
    RuntimeTraceBundle as PublicRuntimeTraceBundle,
    RuntimeTraceManifest as PublicRuntimeTraceManifest,
)
from orbitserve.orbit.simulation_traces import (
    CIRCULAR_GEOMETRY_TRACE,
    OrbitTraceBundle,
    OrbitTraceManifest,
    RuntimeTraceBundle,
    RuntimeTraceManifest,
    SATELLITE_MOTION_TRACE_SCHEMA,
    SURROGATE_ORBIT_TRACE,
    load_surrogate_trace_bundle,
    trace_file_sha256,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class OrbitTraceManifestAndBundleTests(unittest.TestCase):
    def test_canonical_types_and_compatibility_aliases_are_object_identical(self):
        self.assertIs(RuntimeTraceManifest, OrbitTraceManifest)
        self.assertIs(RuntimeTraceBundle, OrbitTraceBundle)
        self.assertIs(PublicOrbitTraceManifest, OrbitTraceManifest)
        self.assertIs(PublicOrbitTraceBundle, OrbitTraceBundle)
        self.assertIs(PublicRuntimeTraceManifest, OrbitTraceManifest)
        self.assertIs(PublicRuntimeTraceBundle, OrbitTraceBundle)

        manifest = OrbitTraceManifest(trace_id="canonical-type-test")
        bundle = OrbitTraceBundle(trace_id="canonical-type-test", trace_manifest=manifest)
        self.assertIsInstance(manifest, RuntimeTraceManifest)
        self.assertIsInstance(bundle, RuntimeTraceBundle)

    def test_nested_manifest_roundtrip_preserves_current_surface(self):
        nested = {
            "schema_version": "orbitserve.satellite_motion_trace.v1",
            "trace_model": CIRCULAR_GEOMETRY_TRACE,
            "evaluated": True,
            "trace_id": "nested-trace",
            "files": {
                "satellite_catalog_path": "satellite_catalog.json",
                "ephemeris_trace_path": "ephemeris_trace.jsonl",
                "access_trace_path": "nested_access.json",
                "link_trace_path": "nested_link.json",
                "energy_trace_path": "nested_energy.json",
                "calibration_report_path": "calibration.json",
            },
            "sha256": {
                "satellite_catalog_sha256": "sha-catalog",
                "ephemeris_trace_sha256": "sha-ephemeris",
                "access_trace_sha256": "sha-access-nested",
                "link_trace_sha256": "sha-link-nested",
                "energy_trace_sha256": "sha-energy-nested",
                "calibration_report_sha256": "sha-calibration",
            },
            "time": {"duration_s": 20.0, "step_s": 5.0},
            "calibration": {"stk_calibrated": False, "calibration_report_evaluated": False},
            "scope": "geometry_trace_surrogate",
            "valid_for_paper_main_results": False,
        }
        nested_manifest = RuntimeTraceManifest.from_value(nested)
        nested_payload = nested_manifest.to_dict()

        self.assertEqual(nested_manifest.satellite_catalog_path, "satellite_catalog.json")
        self.assertEqual(nested_manifest.ephemeris_trace_path, "ephemeris_trace.jsonl")
        self.assertEqual(nested_manifest.access_trace_path, "nested_access.json")
        self.assertEqual(nested_manifest.access_trace_sha256, "sha-access-nested")
        self.assertEqual(nested_payload["files"], nested["files"])
        self.assertEqual(nested_payload["sha256"], nested["sha256"])
        self.assertEqual(nested_payload["time"], nested["time"])
        self.assertEqual(nested_payload["calibration"], nested["calibration"])
        self.assertEqual(nested_payload["ephemeris_trace_path"], "ephemeris_trace.jsonl")

    def test_build_runtime_scenario_preserves_full_trace_manifest_surface(self):
        nested_manifest = {
            "schema_version": "orbitserve.satellite_motion_trace.v1",
            "trace_model": CIRCULAR_GEOMETRY_TRACE,
            "evaluated": True,
            "trace_id": "scenario-trace",
            "files": {
                "satellite_catalog_path": "satellite_catalog.json",
                "ephemeris_trace_path": "ephemeris_trace.jsonl",
                "access_trace_path": "access_trace.json",
                "link_trace_path": "link_trace.json",
                "energy_trace_path": "energy_trace.json",
                "calibration_report_path": None,
            },
            "sha256": {
                "satellite_catalog_sha256": "sha-catalog",
                "ephemeris_trace_sha256": "sha-ephemeris",
                "access_trace_sha256": "sha-access",
                "link_trace_sha256": "sha-link",
                "energy_trace_sha256": "sha-energy",
                "calibration_report_sha256": None,
            },
            "time": {"duration_s": 95.0, "step_s": 30.0, "final_boundary_s": 95.0},
            "calibration": {"stk_calibrated": False, "valid_for_paper_main_results": False},
            "scope": "geometry_trace_surrogate",
            "valid_for_paper_main_results": False,
        }

        scenario = build_runtime_scenario(
            scenario_id="trace-scenario",
            simulation_result={"trace_manifest": nested_manifest, "evaluated": True},
            simulation_config={"duration_s": 60.0},
        ).to_dict()
        trace_ref = scenario["simulation_trace_ref"]

        self.assertEqual(trace_ref["trace_manifest"]["files"], nested_manifest["files"])
        self.assertEqual(trace_ref["trace_manifest"]["sha256"], nested_manifest["sha256"])
        self.assertEqual(trace_ref["files"], nested_manifest["files"])
        self.assertEqual(trace_ref["sha256"], nested_manifest["sha256"])
        self.assertEqual(trace_ref["time"], nested_manifest["time"])
        self.assertEqual(trace_ref["calibration"], nested_manifest["calibration"])
        self.assertEqual(trace_ref["scope"], "geometry_trace_surrogate")
        self.assertFalse(trace_ref["valid_for_paper_main_results"])

    def test_motion_trace_without_dynamic_link_samples_fails_closed(self):
        link = {
            "link_id": "sgl:gw-0:sat-0",
            "kind": "sgl",
            "endpoint_a": "gw-0",
            "endpoint_b": "sat-0",
            "bandwidth_bps": 1000.0,
            "propagation_delay_s": 0.01,
            "edge_available_external": True,
        }
        motion_bundle = RuntimeTraceBundle(
            trace_id="malformed-motion",
            trace_model=CIRCULAR_GEOMETRY_TRACE,
            trace_manifest=RuntimeTraceManifest(
                schema_version=SATELLITE_MOTION_TRACE_SCHEMA,
                trace_model=CIRCULAR_GEOMETRY_TRACE,
                evaluated=True,
            ),
            horizon_s=10.0,
            step_s=1.0,
            links=(link,),
        )

        active = motion_bundle.active_links_at(0.0)
        sample = motion_bundle.link_sample_at("sgl:gw-0:sat-0", 0.0)

        self.assertFalse(active.available)
        self.assertEqual(active.reason, "active_graph_unavailable")
        self.assertFalse(sample.available)
        self.assertEqual(sample.reason, "link_sample_missing")

    def test_sgl_availability_is_derived_from_access_window_between_samples(self):
        link_id = "sgl:gw-0:sat-0"
        link = {
            "link_id": link_id,
            "kind": "sgl",
            "endpoint_a": "gw-0",
            "endpoint_b": "sat-0",
            "bandwidth_bps": 1000.0,
            "propagation_delay_s": 0.9,
            "edge_available_external": False,
            "unavailable_reason": "below_min_elevation",
        }
        window = {
            "ingress_id": "gw-0",
            "satellite_id": "sat-0",
            "window_id": "access-0",
            "start_s": 30.0,
            "end_s": 90.0,
            "sgl_link_id": link_id,
            "bandwidth_bps": 2000.0,
            "propagation_delay_s": 0.012,
            "tau_setup_s": 0.003,
        }
        bundle = RuntimeTraceBundle(
            trace_id="sgl-window-query",
            trace_model=CIRCULAR_GEOMETRY_TRACE,
            trace_manifest=RuntimeTraceManifest(
                schema_version=SATELLITE_MOTION_TRACE_SCHEMA,
                trace_model=CIRCULAR_GEOMETRY_TRACE,
                evaluated=True,
            ),
            horizon_s=120.0,
            step_s=60.0,
            access_windows=(window,),
            links=(link,),
            link_samples=(
                {
                    "time_s": 0.0,
                    "link_id": link_id,
                    "edge_available_external": False,
                    "unavailable_reason": "below_min_elevation",
                    "B_raw_bps": 10.0,
                    "tau_prop_s": 1.0,
                },
            ),
        )

        sample = bundle.link_sample_at(link_id, 45.0)
        self.assertTrue(sample.available)
        self.assertEqual(sample.sample_time_s, 45.0)
        self.assertEqual(sample.value["availability_source"], "access_window")
        self.assertTrue(sample.value["edge_available_external"])
        self.assertIsNone(sample.value["unavailable_reason"])
        self.assertEqual(sample.value["B_raw_bps"], 2000.0)
        self.assertEqual(sample.value["tau_prop_s"], 0.012)

        candidates = find_access_candidates(bundle, "gw-0", 45.0)
        self.assertEqual(len(candidates), 1)
        leg = sample_route_leg(
            bundle,
            RouteLeg(
                link_id=link_id,
                kind="sgl",
                endpoint_a="gw-0",
                endpoint_b="sat-0",
                payload_bytes=128.0,
            ),
            45.0,
        )
        self.assertEqual(leg.bandwidth_bps, 2000.0)
        self.assertEqual(leg.propagation_delay_s, 0.012)
        self.assertEqual(leg.setup_delay_s, 0.003)

        outside = bundle.link_sample_at(link_id, 95.0)
        self.assertFalse(outside.available)
        self.assertEqual(outside.reason, "no_sgl_access_window")

    def test_generic_routing_query_uses_sgl_access_window(self):
        link_id = "sgl:gw-0:sat-0"
        trace = SimpleNamespace(
            access_windows=(
                {
                    "ingress_id": "gw-0",
                    "satellite_id": "sat-0",
                    "window_id": "access-0",
                    "start_s": 30.0,
                    "end_s": 90.0,
                    "sgl_link_id": link_id,
                    "bandwidth_bps": 2000.0,
                    "propagation_delay_s": 0.012,
                },
            ),
            links=(
                {
                    "link_id": link_id,
                    "kind": "sgl",
                    "endpoint_a": "gw-0",
                    "endpoint_b": "sat-0",
                    "bandwidth_bps": 1000.0,
                    "propagation_delay_s": 0.9,
                },
            ),
            link_samples=(
                {
                    "time_s": 0.0,
                    "link_id": link_id,
                    "edge_available_external": False,
                    "unavailable_reason": "below_min_elevation",
                },
            ),
        )

        sample = query_link_sample_at(trace, link_id, 45.0)
        self.assertTrue(sample.available)
        self.assertEqual(sample.sample_time_s, 45.0)
        self.assertEqual(sample.value["availability_source"], "access_window")
        self.assertTrue(sample.value["edge_available_external"])

    def test_nested_bundle_loader_access_half_open_horizon_and_stale_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "satellite_catalog.json"
            ephemeris_path = root / "ephemeris_trace.jsonl"
            access_path = root / "access_trace.json"
            link_path = root / "link_trace.json"
            energy_path = root / "energy_trace.json"

            _write_json(catalog_path, [{"satellite_id": "sat-0"}])
            ephemeris_path.write_text(
                json.dumps({"time_s": 0.0, "satellite_id": "sat-0", "r_eci_km": [1.0, 0.0, 0.0]}) + "\n",
                encoding="utf-8",
            )
            _write_json(
                access_path,
                {
                    "access_windows": [
                        {
                            "ingress_id": "gw-0",
                            "satellite_id": "sat-0",
                            "window_id": "access-0",
                            "start_s": 5.0,
                            "end_s": 10.0,
                            "sgl_link_id": "sgl:gw-0:sat-0",
                            "propagation_delay_s": 0.01,
                            "bandwidth_bps": 1000.0,
                        }
                    ]
                },
            )
            _write_json(
                link_path,
                {
                    "links": [
                        {
                            "link_id": "sgl:gw-0:sat-0",
                            "kind": "sgl",
                            "endpoint_a": "gw-0",
                            "endpoint_b": "sat-0",
                            "bandwidth_bps": 1000.0,
                            "propagation_delay_s": 0.01,
                        }
                    ],
                    "link_samples": [
                        {
                            "time_s": 0.0,
                            "link_id": "sgl:gw-0:sat-0",
                            "edge_available_external": True,
                            "tau_prop_s": 0.01,
                            "B_raw_bps": 1000.0,
                        }
                    ],
                    "active_graph_snapshots": [
                        {"time_s": 0.0, "active_link_ids": ["sgl:gw-0:sat-0"]}
                    ],
                },
            )
            _write_json(
                energy_path,
                {
                    "satellite_initial_soc": {"sat-0": 0.8},
                    "energy_windows": [
                        {
                            "satellite_id": "sat-0",
                            "start_s": 0.0,
                            "end_s": 20.0,
                            "sunlit": True,
                            "dvfs_multiplier": 1.0,
                        }
                    ],
                },
            )
            manifest = RuntimeTraceManifest.from_value(
                {
                    "trace_model": CIRCULAR_GEOMETRY_TRACE,
                    "evaluated": True,
                    "trace_id": "bundle-trace",
                    "files": {
                        "satellite_catalog_path": catalog_path.name,
                        "ephemeris_trace_path": ephemeris_path.name,
                        "access_trace_path": access_path.name,
                        "link_trace_path": link_path.name,
                        "energy_trace_path": energy_path.name,
                    },
                    "sha256": {
                        "satellite_catalog_sha256": trace_file_sha256(catalog_path),
                        "ephemeris_trace_sha256": trace_file_sha256(ephemeris_path),
                        "access_trace_sha256": trace_file_sha256(access_path),
                        "link_trace_sha256": trace_file_sha256(link_path),
                        "energy_trace_sha256": trace_file_sha256(energy_path),
                    },
                    "time": {"duration_s": 20.0, "step_s": 5.0},
                    "calibration": {"stk_calibrated": False},
                    "scope": "geometry_trace_surrogate",
                    "provenance": {"trace_dir": tmpdir},
                }
            )

            bundle = load_surrogate_trace_bundle(manifest)

        self.assertIsInstance(bundle, RuntimeTraceBundle)
        self.assertEqual(len(bundle.satellite_catalog), 1)
        self.assertEqual(len(bundle.ephemeris_samples), 1)
        self.assertTrue(bundle.access_at("gw-0", 5.0).available)
        self.assertFalse(bundle.access_at("gw-0", 10.0).available)
        self.assertEqual(bundle.access_at("gw-0", 10.0).reason, "no_access_window")
        self.assertFalse(bundle.access_at("gw-0", 20.0).available)
        self.assertEqual(bundle.access_at("gw-0", 20.0).reason, "request_time_outside_trace_horizon")
        self.assertTrue(bundle.active_links_at(4.0).available)
        no_window = bundle.link_sample_at("sgl:gw-0:sat-0", 12.0)
        self.assertFalse(no_window.available)
        self.assertFalse(no_window.stale)
        self.assertEqual(no_window.reason, "no_sgl_access_window")


if __name__ == "__main__":
    unittest.main()
