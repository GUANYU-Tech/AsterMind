"""Core data is independent of optional views; parallel views preserve physics."""

import gzip
import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from astermind.builder.physical.network import run_network_experiment
from astermind.builder.physical.network_exports import export_network_artifacts
from astermind.builder.physical.physical_dataset import load_dataset, reuse_dataset
from astermind.builder.physical.snapshot_store import SnapshotSource

from astermind.builder.orbit import load_orbit_trace, write_orbit_trace
from astermind.engine import LinkProfile, NetworkSpec, NetworkState
from astermind.engine.state.attribute_codec import MAGIC
from tests.integration.fixed_design.test_dynamic_network_experiment import manifest


def canonical(value):
    return json.loads(json.dumps(value))


class ParallelPhysicalExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.input = manifest()
        cls.input["time"]["step_s"] = 10.0
        cls.input['network']['fault_timing'] = 'continuous'
        cls.input["network"]["faults"] = [
            {
                "target_type": "link",
                "target_id": "cross_layer:sat-00001:sat-00002",
                "start_s": 0.5,
                "end_s": 30.5,
            }
        ]
        cls.core = cls.root / "core"
        cls.result = run_network_experiment(
            cls.input, cls.core, _workers=1, export_mode="core", chunk_samples=4
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy_core(self, name):
        target = self.root / name
        reuse_dataset(self.core, target, self.input, include_exports=False)
        return target

    def test_core_is_complete_and_queryable_without_optional_views(self):
        _, index, result = load_dataset(self.core)
        self.assertEqual(index["schema_version"], "orbitserve.physical_dataset.v2")
        self.assertFalse((self.core / "frames").exists())
        self.assertFalse((self.core / "physical_exports.json").exists())
        self.assertEqual(result["artifacts"]["frames"], [])
        source = SnapshotSource(self.core)
        trace = load_orbit_trace(self.core / "orbit_trace/trace_manifest.json")
        network = NetworkState(
            trace,
            NetworkSpec.from_value(self.input["network"]),
            self.input["ground_ingresses"],
            physical_state=self.core / "physical_state.sqlite",
        ).audit_timeline()
        with patch.object(
            LinkProfile,
            "attributes",
            side_effect=AssertionError("physical recomputation"),
        ):
            for t in (0.0, 0.5, 2.0, 10.0, 30.5, 40.0, 119.9, 120.0):
                self.assertEqual(
                    canonical(source.snapshot(t)), canonical(network.snapshot(t))
                )

    def test_snapshot_static_metadata_is_isolated_for_callers(self):
        source = SnapshotSource(self.core)
        first = source.snapshot(0.0)
        expected = source.snapshot(0.0)["nodes"][0]["radio_parameters"]["generation"]
        first["nodes"][0]["radio_parameters"]["generation"] = "changed by caller"
        self.assertEqual(
            source.snapshot(0.0)["nodes"][0]["radio_parameters"]["generation"], expected
        )

    def test_disk_backed_session_reuses_one_connection_and_releases_it(self):
        import astermind.engine.state.physical_dataset as attributes

        times = (0.0, 0.5, 2.0, 10.0, 30.5, 40.0, 119.9, 120.0)
        expected = [canonical(SnapshotSource(self.core).snapshot(t)) for t in times]
        with (
            patch.object(attributes, 'COMPRESSED_READ_CACHE_BYTES', 0),
            patch.object(attributes, 'DECODED_READ_CACHE_BYTES', 0),
        ):
            source = SnapshotSource(self.core)
            self.assertIsNone(source.physical_attributes._packed)
            original_connect = sqlite3.connect
            with patch.object(attributes.sqlite3, 'connect', wraps=original_connect) as opened:
                with source:
                    connection = source.physical_attributes._read_connection
                    for t, frame in zip(times, expected):
                        self.assertEqual(canonical(source.snapshot(t)), frame)
                    self.assertEqual(opened.call_count, 1)
                self.assertIsNone(source.physical_attributes._read_connection)
                self.assertFalse(source._positions)
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute('SELECT 1')

    def test_serial_export_releases_read_session_after_failure(self):
        import astermind.builder.physical.network_exports as exports

        from astermind.engine.state.physical_dataset import PhysicalAttributes

        folder = self.copy_core('read-session-failure')
        original_close = PhysicalAttributes.close_read_session
        with (
            patch.object(SnapshotSource, 'snapshot', side_effect=ValueError('injected export failure')),
            patch.object(PhysicalAttributes, 'close_read_session', autospec=True,
                         side_effect=original_close) as closed,
        ):
            with self.assertRaisesRegex(ValueError, 'injected export failure'):
                export_network_artifacts(folder, workers=1)
            self.assertEqual(closed.call_count, 1)
            self.assertIsNone(exports._source)
        self.assertFalse((folder / 'physical_exports.json').exists())
        load_dataset(folder, verify_exports=False)

    def test_numeric_trace_preserves_legacy_identity_and_every_value(self):
        trace = load_orbit_trace(self.core / "orbit_trace/trace_manifest.json")
        payload = json.loads(
            (self.core / "orbit_trace/ephemeris_trace.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["storage"], "orbitserve.ephemeris_columns.v1")
        legacy = self.root / "legacy-trace"
        write_orbit_trace(
            legacy,
            time=trace.time,
            satellite_catalog=trace.satellite_catalog,
            ephemeris_samples=trace.ephemeris_samples,
            sgl_windows=trace.sgl_windows,
            isl_windows=trace.isl_windows,
            sunlight_windows=trace.sunlight_windows,
            trace_model=trace.trace_model,
            provenance={
                k: v
                for k, v in trace.provenance.items()
                if k not in {"writer", "algorithm_version"}
            },
            warnings=trace.warnings,
        )
        restored = load_orbit_trace(legacy / "trace_manifest.json")
        self.assertEqual(restored.trace_id, trace.trace_id)
        self.assertEqual(restored.ephemeris_samples, trace.ephemeris_samples)

    def test_parallel_views_match_serial_including_fractional_fault_boundaries(self):
        first, second = self.copy_core("serial"), self.copy_core("parallel")
        one = export_network_artifacts(
            first, workers=1, compression_level=1, chunk_samples=4
        )
        many = export_network_artifacts(
            second, workers=2, compression_level=1, chunk_samples=4
        )
        self.assertEqual(many["execution"]["workers"], 2)
        self.assertEqual(one["artifacts"], many["artifacts"])
        for item in one["artifacts"]["frames"]:
            self.assertEqual(
                gzip.decompress((first / item["artifact"]).read_bytes()),
                gzip.decompress((second / item["artifact"]).read_bytes()),
            )
        for name in (
            "astermind_network.jsonl.gz",
            "node_positions.jsonl.gz",
            "link_attributes.jsonl.gz",
            "link_events.jsonl.gz",
        ):
            self.assertEqual(
                gzip.decompress((first / name).read_bytes()),
                gzip.decompress((second / name).read_bytes()),
            )
        rows = gzip.decompress(
            (second / "astermind_network.jsonl.gz").read_bytes()
        ).splitlines()
        self.assertTrue(any(json.loads(row)["time_s"] == 0.5 for row in rows))
        self.assertTrue(any(json.loads(row)["time_s"] == 30.5 for row in rows))
        load_dataset(second)

    def test_optional_corruption_does_not_prevent_core_loading(self):
        target = self.copy_core("optional-corrupt")
        result = export_network_artifacts(target, workers=1, chunk_samples=4)
        frame = target / result["artifacts"]["frames"][0]["artifact"]
        frame.unlink()
        load_dataset(target, verify_exports=False)
        with self.assertRaisesRegex(ValueError, "missing or corrupt"):
            load_dataset(target)

    def test_failed_export_keeps_complete_core_and_cleans_temporary_files(self):
        target = self.copy_core("failed")
        with (
            patch(
                'astermind.builder.physical.network_exports.export_cell_network',
                side_effect=RuntimeError("export failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "export failed"),
        ):
            export_network_artifacts(target, workers=1, chunk_samples=4)
        load_dataset(target)
        self.assertFalse((target / "physical_exports.json").exists())
        self.assertFalse(list(target.rglob("*.tmp")))

    def test_core_integrity_checks_cover_numeric_shards(self):
        target = self.copy_core("shard-corrupt")
        path = next((target / "state/positions").glob("*.npy"))
        data = path.read_bytes()
        path.unlink()  # Do not mutate a shared hard link.
        path.write_bytes(data[:-1])
        with self.assertRaisesRegex(ValueError, "missing or corrupt"):
            load_dataset(target)

    def test_attribute_storage_uses_numeric_columns_and_legacy_blocks_still_load(self):
        from astermind.engine.state.physical_dataset import PhysicalAttributes

        target = self.copy_core("legacy-block")
        dbpath = target / "physical_state.sqlite"
        data = dbpath.read_bytes()
        dbpath.unlink()
        dbpath.write_bytes(data)
        reader = PhysicalAttributes(dbpath)
        with sqlite3.connect(dbpath) as db:
            link, start, value = db.execute(
                "SELECT link,time,value FROM attributes LIMIT 1"
            ).fetchone()
            self.assertTrue(zlib.decompress(value).startswith(MAGIC))
            block = reader._block(link, start)
            rows = [(t, block.at(i)) for i, t in enumerate(block.times)]
            old = zlib.compress(json.dumps(rows).encode())
            db.execute(
                "UPDATE attributes SET value=? WHERE link=? AND time=?",
                (old, link, start),
            )
        legacy = PhysicalAttributes(dbpath)
        self.assertEqual(legacy.at(link, start), reader.at(link, start))

    def test_compression_levels_preserve_values_and_are_validated(self):
        for level in (0, 3, 9):
            target = self.copy_core(f"level-{level}")
            result = export_network_artifacts(
                target, workers=1, compression_level=level, chunk_samples=4
            )
            item = result["artifacts"]["frames"][0]
            with gzip.open(target / item["artifact"], "rt", encoding="utf-8") as stream:
                frame = json.load(stream)
            expected = canonical(SnapshotSource(self.core).snapshot(item["time_s"]))
            frame.pop("events_until_next_boundary")
            self.assertEqual(frame, expected)
        with self.assertRaises(ValueError):
            export_network_artifacts(
                self.copy_core("invalid-level"), compression_level=10
            )
