import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.experiments import load_experiment_spec
from orbitserve.runtime.routing import find_access_candidates
from orbitserve.orbit.simulation_traces import RuntimeTraceBundle, RuntimeTraceManifest, load_surrogate_trace_bundle
from orbitserve.workload import GatewaySpec, RegionSpec, WorkloadGenerationResult, materialize_workload_bundle


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/fixtures/manifests/smoke_circular_geometry_trace.json"
CIRCULAR_TRACE_MODEL = "circular_geometry_trace"


def _optional_builder():
    try:
        module = importlib.import_module("orbitserve.orbit.orbit_traces")
    except ModuleNotFoundError as exc:
        if exc.name == "orbitserve.orbit.orbit_traces":
            return None
        raise
    return getattr(module, "build_satellite_motion_trace", None)


def _as_dict(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value if isinstance(value, dict) else {}


def _trace_manifest(value):
    payload = _as_dict(value)
    manifest_payload = payload.get("trace_manifest") or payload
    return RuntimeTraceManifest.from_value(manifest_payload), manifest_payload, payload


def _trace_dir(manifest_payload, result_payload, tmpdir):
    provenance = manifest_payload.get("provenance") or result_payload.get("provenance") or {}
    return Path(provenance.get("trace_dir") or result_payload.get("trace_dir") or tmpdir)


def _read_trace_file(trace_dir, relative_path):
    path = Path(relative_path)
    if not path.is_absolute():
        path = trace_dir / path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_workload():
    spec = load_experiment_spec(MANIFEST)
    return materialize_workload_bundle(spec.workload, base_dir=Path(spec.base_dir))


def _gateway_and_direct_workload():
    return WorkloadGenerationResult(
        bundle_dir=None,
        manifest={
            "schema_version": "orbitserve.workload",
            "workload_id": "unit_gateway_and_direct",
            "provenance": {"config_sha256": "unit"},
        },
        regions=(
            RegionSpec(
                region_id="us_east",
                region_name="US East",
                lat_deg=39.0,
                lon_deg=-77.0,
                demand_weight=1.0,
            ),
        ),
        gateways=(
            GatewaySpec(
                gateway_id="gw_us_east",
                lat_deg=38.95,
                lon_deg=-77.45,
                served_region_ids=("us_east",),
                gateway_bandwidth_bps=1_000_000_000,
            ),
        ),
        requests=(
            {
                "request_id": "req-direct-0",
                "arrival_time_s": 10.0,
                "region_id": "us_east",
                "access_mode": "direct_user_sgl",
                "ingress_id": "user_direct_a",
                "ingress_lat_deg": 39.0,
                "ingress_lon_deg": -77.0,
                "prompt_tokens": 16,
                "output_tokens": 8,
                "input_bytes": 64,
                "output_bytes": 32,
                "ttft_deadline_s": 5.0,
                "e2e_deadline_s": 30.0,
            },
        ),
        sessions=(),
        timebin_demand=(),
        summary={"request_count": 1},
        status="loaded",
    )


@unittest.skipIf(
    _optional_builder() is None,
    "orbitserve.orbit.orbit_traces.build_satellite_motion_trace is not available yet",
)
class CircularGeometryTraceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_trace = _optional_builder()

    def _build(
        self,
        *,
        workload=None,
        constellation=None,
        simulation_overrides=None,
        trace_options_overrides=None,
    ):
        spec = load_experiment_spec(MANIFEST)
        workload = workload or _base_workload()
        constellation = constellation or spec.constellation
        options = dict(spec.simulation.trace_options)
        if trace_options_overrides:
            options.update(trace_options_overrides)
        simulation = replace(
            spec.simulation,
            trace_output_dir=None,
            trace_options=options,
            **(simulation_overrides or {}),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            simulation = replace(simulation, trace_output_dir=tmpdir)
            result = self.build_trace(
                config=simulation,
                constellation=constellation,
                workload_generation=workload,
                base_dir=ROOT,
                experiment_id="unit_circular_geometry_trace",
            )
            manifest, manifest_payload, result_payload = _trace_manifest(result)
            trace_dir = _trace_dir(manifest_payload, result_payload, tmpdir)
            materialized = {
                "manifest": manifest,
                "manifest_payload": manifest_payload,
                "result_payload": result_payload,
                "trace_dir": trace_dir,
                "access": _read_trace_file(trace_dir, manifest.access_trace_path),
                "link": _read_trace_file(trace_dir, manifest.link_trace_path),
                "energy": _read_trace_file(trace_dir, manifest.energy_trace_path),
                "runtime_bundle": load_surrogate_trace_bundle(manifest),
                "access_sha256_actual": _sha256_file(trace_dir / manifest.access_trace_path),
                "link_sha256_actual": _sha256_file(trace_dir / manifest.link_trace_path),
                "energy_sha256_actual": _sha256_file(trace_dir / manifest.energy_trace_path),
            }
            files = manifest_payload.get("files") or {}
            satellite_catalog_path = files.get("satellite_catalog_path")
            if satellite_catalog_path:
                materialized["satellite_catalog"] = _read_trace_file(trace_dir, satellite_catalog_path)
            return materialized

    def test_access_membership_excludes_window_end_boundary(self):
        from orbitserve.orbit.orbit_traces import _access_membership

        link_id = "sgl:gw-0:sat-0"
        access_trace = {
            "visible_count_samples": (
                {"time_s": 10.0},
                {"time_s": 20.0},
                {"time_s": 30.0},
            ),
            "access_windows": (
                {
                    "ingress_id": "gw-0",
                    "satellite_id": "sat-0",
                    "sgl_link_id": link_id,
                    "start_s": 10.0,
                    "end_s": 20.0,
                },
            ),
        }

        membership = _access_membership(access_trace)
        bundle = RuntimeTraceBundle(
            trace_id="access-membership-half-open",
            trace_model=CIRCULAR_TRACE_MODEL,
            horizon_s=40.0,
            access_windows=tuple(access_trace["access_windows"]),
            links=({"link_id": link_id, "kind": "sgl"},),
        )

        self.assertTrue(membership.get((link_id, 10.0)))
        self.assertNotIn((link_id, 20.0), membership)
        self.assertFalse(membership.get((link_id, 30.0), False))
        self.assertTrue(bundle.access_at("gw-0", 10.0).available)
        self.assertFalse(bundle.access_at("gw-0", 20.0).available)
        self.assertEqual(bundle.link_sample_at(link_id, 20.0).reason, "no_sgl_access_window")

    def test_time_grid_uses_regular_sample_count_and_final_boundary(self):
        built = self._build(
            workload=_gateway_and_direct_workload(),
            simulation_overrides={"duration_s": 95.0, "step_s": 30.0},
        )
        time_grid = built["manifest_payload"]["time"]

        self.assertEqual(time_grid["regular_sample_count"], 4)
        self.assertEqual(time_grid["sample_count"], 4)
        self.assertEqual(time_grid["final_boundary_s"], 95.0)
        self.assertEqual(time_grid["tail_duration_s"], 5.0)
        self.assertFalse(time_grid["include_final_boundary_sample"])
        sample_times = {sample["time_s"] for sample in built["link"].get("link_samples", ())}
        self.assertTrue({0.0, 30.0, 60.0, 90.0}.issubset(sample_times))

    def test_time_grid_extends_to_late_workload_request_arrivals(self):
        workload = _gateway_and_direct_workload()
        arrival_request = dict(workload.requests[0])
        arrival_request.update({"request_id": "req-late-arrival", "arrival_time_s": 125.0})
        time_only_request = dict(workload.requests[0])
        time_only_request.pop("arrival_time_s", None)
        time_only_request.update({"request_id": "req-late-time-only", "time_s": 185.0})
        workload = replace(
            workload,
            requests=(arrival_request, time_only_request),
            summary={"request_count": 2},
        )

        built = self._build(
            workload=workload,
            simulation_overrides={"duration_s": 90.0, "step_s": 30.0},
            trace_options_overrides={"trace_horizon_margin_s": 45.0},
        )
        manifest_payload = built["manifest_payload"]
        time_grid = manifest_payload["time"]
        provenance = manifest_payload["provenance"]
        warnings = " ".join(manifest_payload["warnings"])

        self.assertEqual(time_grid["configured_duration_s"], 90.0)
        self.assertEqual(time_grid["effective_duration_s"], 230.0)
        self.assertEqual(time_grid["duration_s"], 230.0)
        self.assertEqual(time_grid["workload_max_request_time_s"], 185.0)
        self.assertEqual(time_grid["trace_horizon_margin_s"], 45.0)
        self.assertEqual(time_grid["workload_request_time_field_counts"]["arrival_time_s"], 1)
        self.assertEqual(time_grid["workload_request_time_field_counts"]["time_s"], 1)
        self.assertEqual(provenance["configured_duration_s"], 90.0)
        self.assertEqual(provenance["effective_duration_s"], 230.0)
        self.assertIn(":230s", manifest_payload["trace_id"])
        self.assertIn("horizon extended", warnings)

        access = built["runtime_bundle"].access_at("gw_us_east", 185.0)
        self.assertNotEqual(access.reason, "request_time_outside_trace_horizon")

    def test_non_positive_trace_margin_still_covers_half_open_horizon(self):
        workload = _gateway_and_direct_workload()
        request = dict(workload.requests[0])
        request.update({"request_id": "req-at-boundary", "arrival_time_s": 120.0})
        workload = replace(workload, requests=(request,), summary={"request_count": 1})

        built = self._build(
            workload=workload,
            simulation_overrides={"duration_s": 120.0, "step_s": 30.0},
            trace_options_overrides={"trace_horizon_margin_s": 0.0},
        )
        time_grid = built["manifest_payload"]["time"]
        warnings = " ".join(built["manifest_payload"]["warnings"])

        self.assertGreater(time_grid["duration_s"], 120.0)
        self.assertEqual(time_grid["trace_horizon_margin_s"], 30.0)
        self.assertIn("non-positive trace_horizon_margin_s", warnings)

        access = built["runtime_bundle"].access_at("gw_us_east", 120.0)
        self.assertNotEqual(access.reason, "request_time_outside_trace_horizon")

    def test_satellite_catalog_expands_orbital_set_clusters_deterministically(self):
        built = self._build()
        catalog = built.get("satellite_catalog") or built["access"].get("satellite_catalog")

        self.assertEqual(len(catalog), 8)
        self.assertEqual([row["satellite_id"] for row in catalog], [f"sat-{idx:05d}" for idx in range(8)])
        self.assertEqual(catalog[0]["orbital_set_name"], "walker_550_53_motion_smoke")
        self.assertEqual(catalog[0]["plane_index"], 0)
        self.assertEqual(catalog[0]["cell_index"], 0)
        self.assertEqual(catalog[0]["cluster_member_index"], 0)
        self.assertEqual(catalog[1]["cluster_member_index"], 1)
        self.assertEqual(catalog[0]["compute_cell_id"], "cell:walker_550_53_motion_smoke:0:0")
        self.assertIn("semantic_satellite_id", catalog[0])
        self.assertIn("mean_motion_rad_s", catalog[0]["orbit"])

    def test_gateway_and_direct_ingress_feed_access_trace(self):
        built = self._build(
            workload=_gateway_and_direct_workload(),
            trace_options_overrides={"ground_ingress_source": "workload_bundle_gateways_and_direct_requests"},
        )
        ingress_catalog = built["access"].get("ingress_catalog", ())
        ingress_by_id = {row["ingress_id"]: row for row in ingress_catalog}

        self.assertIn("gw_us_east", ingress_by_id)
        self.assertIn("user_direct_a", ingress_by_id)
        self.assertEqual(ingress_by_id["gw_us_east"]["ingress_type"], "gateway")
        self.assertEqual(ingress_by_id["user_direct_a"]["ingress_type"], "direct_user")
        window_ingress_ids = {row["ingress_id"] for row in built["access"].get("access_windows", ())}
        self.assertTrue({"gw_us_east", "user_direct_a"}.issubset(window_ingress_ids))

    def test_generated_sgl_capacity_fails_closed_and_positive_capacity_stays_available(self):
        spec = load_experiment_spec(MANIFEST)
        invalid_profiles = {
            "missing": {},
            "zero_bandwidth": {
                "bandwidth_bps": 0.0,
                "terminal_count_per_satellite": 1,
            },
            "zero_terminal_count": {
                "bandwidth_bps": 1_000_000_000.0,
                "terminal_count_per_satellite": 0,
            },
        }

        for label, gateway_profile in invalid_profiles.items():
            with self.subTest(label=label):
                built = self._build(
                    constellation=replace(spec.constellation, gateway_profile=gateway_profile)
                )
                bundle = built["runtime_bundle"]
                self.assertTrue(bundle.access_windows)
                window = bundle.access_windows[0]
                query_time_s = (float(window["start_s"]) + float(window["end_s"])) / 2.0
                sample = bundle.link_sample_at(str(window["sgl_link_id"]), query_time_s)
                self.assertFalse(sample.available)
                self.assertIn(sample.reason, {"invalid_sgl_bandwidth", "invalid_sgl_terminal_count"})
                self.assertEqual(
                    find_access_candidates(bundle, str(window["ingress_id"]), query_time_s),
                    (),
                )

        for field, value in (
            ("bandwidth_bps", float("inf")),
            ("terminal_count_per_satellite", float("inf")),
        ):
            with self.subTest(nonfinite_field=field), self.assertRaisesRegex(ValueError, "finite"):
                replace(
                    spec.constellation,
                    gateway_profile={
                        "bandwidth_bps": 1_000_000_000.0,
                        "terminal_count_per_satellite": 1,
                        field: value,
                    },
                )

        built = self._build(
            constellation=replace(
                spec.constellation,
                gateway_profile={
                    "bandwidth_bps": 1_000_000_000.0,
                    "terminal_count_per_satellite": 1,
                },
            )
        )
        bundle = built["runtime_bundle"]
        window = bundle.access_windows[0]
        query_time_s = (float(window["start_s"]) + float(window["end_s"])) / 2.0
        self.assertTrue(bundle.link_sample_at(str(window["sgl_link_id"]), query_time_s).available)
        self.assertTrue(find_access_candidates(bundle, str(window["ingress_id"]), query_time_s))





    def test_shadow_trace_and_manifest_sha256_provenance_are_auditable(self):
        built = self._build()
        manifest = built["manifest"]
        trace_dir = built["trace_dir"]

        self.assertEqual(manifest.trace_model, CIRCULAR_TRACE_MODEL)
        self.assertTrue(manifest.evaluated)
        self.assertEqual(manifest.scope, "geometry_trace_surrogate")
        self.assertFalse(manifest.valid_for_paper_main_results)
        self.assertEqual(built["access_sha256_actual"], manifest.access_trace_sha256)
        self.assertEqual(built["link_sha256_actual"], manifest.link_trace_sha256)
        self.assertEqual(built["energy_sha256_actual"], manifest.energy_trace_sha256)
        self.assertTrue(built["energy"].get("shadow_windows") or built["energy"].get("energy_windows"))
        sunlight = built["energy"]["sunlight_contract_summary"]
        self.assertTrue(sunlight["evaluated"])
        self.assertGreater(sunlight["actual_sunlit_fraction"], 0.0)
        self.assertLessEqual(sunlight["actual_sunlit_fraction"], 1.0)
        self.assertEqual(sunlight["sample_count_per_satellite"], 360)
        self.assertGreater(sunlight["maximum_shadow_window_s"], 0.0)
        self.assertIn("docs/research/parameters/satellite_motion_trace_model.md", json.dumps(manifest.provenance))
        self.assertIn("sha256", json.dumps(built["manifest_payload"]))


if __name__ == "__main__":
    unittest.main()
