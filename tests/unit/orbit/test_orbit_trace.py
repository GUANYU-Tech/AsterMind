import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import orbitserve.orbit as orbit
from orbitserve.orbit import (
    ORBIT_TRACE_SCHEMA,
    ORBIT_TRACE_SCOPE,
    load_orbit_trace,
    write_orbit_trace,
)


def _inputs(reverse=False):
    data = {
        "time": {
            "epoch": "2026-07-20T00:00:00Z",
            "start_s": -0.0,
            "end_s": 120.0,
            "duration_s": 120.0,
            "step_s": 60.0,
        },
        "satellite_catalog": [
            {"satellite_id": "sat-b", "plane_index": 1, "cluster_member_index": 0},
            {"satellite_id": "sat-a", "plane_index": 0, "cluster_member_index": 0},
        ],
        "ephemeris_samples": [
            {
                "satellite_id": sat,
                "time_s": time_s,
                "source_reference_frame": "TEME",
                "earth_fixed_reference_frame": "ITRS",
                "position_teme_km": [7000.0, time_s, 0.0],
                "velocity_teme_km_s": [0.0, 7.5, 0.0],
                "position_itrs_km": [7000.0, time_s, 0.0],
                "velocity_itrs_km_s": [0.0, 7.5, 0.0],
            }
            for time_s in (120.0, 60.0, 0.0)
            for sat in ("sat-b", "sat-a")
        ],
        "sgl_windows": [
            {
                "ingress_id": "gateway-b",
                "satellite_id": "sat-b",
                "start_s": 60.0,
                "end_s": 120.0,
                "propagation_delay_s": 0.003,
            },
            {
                "ingress_id": "gateway-a",
                "satellite_id": "sat-a",
                "start_s": 0.0,
                "end_s": 60.0,
                "propagation_delay_s": 0.002,
            },
        ],
        "isl_windows": [
            {
                "endpoint_a": "sat-b",
                "endpoint_b": "sat-a",
                "start_s": 0.0,
                "end_s": 120.0,
                "propagation_delay_s": 0.005,
            }
        ],
        "sunlight_windows": [
            {"satellite_id": "sat-b", "start_s": 60.0, "end_s": 120.0, "sunlit": False},
            {"satellite_id": "sat-a", "start_s": 0.0, "end_s": 120.0, "sunlit": True},
        ],
        "trace_model": "unit_geometry",
        "provenance": {"model": "unit", "parameters": {"beta": 2, "alpha": 1}},
        "warnings": ("warning-b", "warning-a", "warning-a"),
    }
    if not reverse:
        return data

    def reverse_mapping(value):
        if isinstance(value, dict):
            return {key: reverse_mapping(value[key]) for key in reversed(tuple(value))}
        if isinstance(value, list):
            return [reverse_mapping(item) for item in value]
        return value

    reversed_data = reverse_mapping(data)
    for collection in (
        "satellite_catalog",
        "ephemeris_samples",
        "sgl_windows",
        "isl_windows",
        "sunlight_windows",
    ):
        reversed_data[collection].reverse()
        reversed_data[collection].append(dict(reversed_data[collection][0]))
    return reversed_data


def _write(output_dir, data):
    return write_orbit_trace(output_dir, **data)


def _file_bytes(directory):
    return {
        path.name: path.read_bytes()
        for path in sorted(Path(directory).iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


class OrbitTraceTests(unittest.TestCase):
    def test_legacy_circular_and_runtime_interfaces_are_not_public(self):
        legacy_names = (
            "CIRCULAR_GEOMETRY_TRACE",
            "ConstellationDesign",
            "FormationCluster",
            "IDEAL_FULL_VISIBILITY",
            "OrbitalSet",
            "RuntimeTraceBundle",
            "RuntimeTraceManifest",
            "SURROGATE_ORBIT_TRACE",
            "SimulationConfig",
            "SimulationResult",
            "TRACE_BACKED_RUNTIME_SURROGATE_SCOPE",
            "active_sgl_access_window",
            "has_sgl_access_windows",
            "load_surrogate_trace_bundle",
            "normalize_trace_scope",
            "resolve_trace_path",
            "run_simulation",
            "sgl_window_link_sample",
        )
        self.assertTrue(all(not hasattr(orbit, name) for name in legacy_names))

    def test_roundtrip_is_geometry_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            manifest = _write(trace_dir, _inputs())
            bundle = load_orbit_trace(trace_dir / "trace_manifest.json")
            bundle_from_manifest = load_orbit_trace(manifest, base_dir=trace_dir)

            self.assertEqual(manifest.schema_version, ORBIT_TRACE_SCHEMA)
            self.assertEqual(bundle.scope, ORBIT_TRACE_SCOPE)
            self.assertEqual(bundle.trace_id, manifest.trace_id)
            self.assertEqual(bundle_from_manifest.trace_id, manifest.trace_id)
            self.assertEqual([row["satellite_id"] for row in bundle.satellite_catalog], ["sat-a", "sat-b"])
            self.assertEqual(bundle.isl_windows[0]["endpoint_a"], "sat-a")
            self.assertEqual(bundle.isl_windows[0]["endpoint_b"], "sat-b")
            self.assertFalse(hasattr(bundle, "satellite_initial_soc"))
            self.assertFalse(hasattr(bundle, "terminal_catalog"))
            self.assertFalse(hasattr(bundle, "active_graph_snapshots"))

            forbidden = {"satellite_initial_soc", "terminal_catalog", "active_graph_snapshots"}
            for content in _file_bytes(trace_dir).values():
                self.assertTrue(content.endswith(b"\n"))
                self.assertFalse(content.endswith(b"\n\n"))
                self.assertNotIn(b"\r\n", content)
                self.assertTrue(forbidden.isdisjoint(_all_keys(json.loads(content))))

    def test_different_directories_and_input_order_produce_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first"
            second = Path(tmpdir) / "second"
            first_manifest = _write(first, _inputs())
            second_manifest = _write(second, _inputs(reverse=True))

            self.assertEqual(first_manifest.trace_id, second_manifest.trace_id)
            self.assertEqual(_file_bytes(first), _file_bytes(second))
            parsed = json.loads((first / "trace_manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn(str(first), json.dumps(parsed))
            self.assertTrue(all("\\" not in path for path in parsed["files"].values()))

    def test_independent_processes_ignore_python_hash_seed(self):
        script = """
import json, sys
from pathlib import Path
from orbitserve.orbit import write_orbit_trace
data = json.loads(sys.argv[2])
write_orbit_trace(Path(sys.argv[1]), **data)
"""
        payload = _inputs()
        payload["warnings"] = list(payload["warnings"])
        encoded = json.dumps(payload, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = []
            for seed in ("1", "987654"):
                output = Path(tmpdir) / f"seed-{seed}"
                env = dict(os.environ)
                env["PYTHONHASHSEED"] = seed
                subprocess.run(
                    [sys.executable, "-c", script, str(output), encoded],
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                outputs.append(_file_bytes(output))
            self.assertEqual(outputs[0], outputs[1])

    def test_writer_rejects_runtime_owned_fields_and_non_finite_values(self):
        forbidden_cases = (
            ("satellite_catalog", "terminal_catalog"),
            ("sgl_windows", "active_graph_snapshots"),
            ("sunlight_windows", "satellite_initial_soc"),
        )
        for collection, field in forbidden_cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                data = _inputs()
                data[collection][0][field] = {}
                with self.assertRaisesRegex(ValueError, "not allowed"):
                    _write(Path(tmpdir) / "trace", data)

        with tempfile.TemporaryDirectory() as tmpdir:
            data = _inputs()
            data["ephemeris_samples"][0]["position_teme_km"][0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                _write(Path(tmpdir) / "trace", data)

    def test_loader_fails_closed_on_checksum_or_manifest_content_tampering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            _write(trace_dir, _inputs())
            catalog_path = trace_dir / "satellite_catalog.json"
            catalog_path.write_bytes(catalog_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_orbit_trace(trace_dir / "trace_manifest.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            _write(trace_dir, _inputs())
            manifest_path = trace_dir / "trace_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["warnings"].append("tampered")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest content SHA-256 mismatch"):
                load_orbit_trace(manifest_path)

    def test_manifest_hash_is_defined_without_self_hash_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            _write(trace_dir, _inputs())
            manifest = json.loads((trace_dir / "trace_manifest.json").read_text(encoding="utf-8"))
            expected = manifest["sha256"].pop("trace_manifest_content_sha256")
            canonical = json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8") + b"\n"
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
