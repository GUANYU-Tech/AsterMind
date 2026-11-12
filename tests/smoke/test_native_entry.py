"""The experiment CLI has one SimulationConfig execution path."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from astermind.builder.config import SimulationConfig
from astermind.experiments.cli import main

ROOT = Path(__file__).resolve().parents[2]


class NativeEntryTests(unittest.TestCase):
    def scene(self):
        return json.loads((ROOT / "examples/native/two_cells.json").read_text())

    def test_schema_defaults_remain_distinct(self):
        for version, expected in (
            (1, "flowsim"),
            ("orbitserve.native_cells.v1", "protobridge_cell"),
        ):
            with self.subTest(version=version):
                value = self.scene()
                value["schema_version"] = version
                value.pop("communication_mode", None)
                self.assertEqual(
                    SimulationConfig.model_validate(value).communication_mode, expected
                )

    def test_removed_fields_are_rejected_even_when_empty(self):
        for name, value in (("orbit", None), ("orbit", {}), ("link_parameters", {})):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValidationError) as error:
                    SimulationConfig.model_validate({**self.scene(), name: value})
                self.assertIn("extra_forbidden", str(error.exception))
        fields = SimulationConfig.model_json_schema()["properties"]
        self.assertNotIn("orbit", fields)
        self.assertNotIn("link_parameters", fields)

    def test_cli_delegates_all_schemas_and_preserves_options(self):
        with patch(
            'astermind.experiments.cli.run_native_cell_manifest', return_value=7
        ) as execute:
            status = main(
                [
                    "scene.json",
                    "--midlayer-root",
                    "/ml",
                    "--out",
                    "out.json",
                    "--full",
                    "--artifact-dir",
                    "audit",
                ]
            )
        self.assertEqual(status, 7)
        args = execute.call_args.args[0]
        self.assertEqual(args.manifest, Path("scene.json"))
        self.assertEqual(args.midlayer_root, Path("/ml"))
        self.assertTrue(args.full)
        self.assertEqual(args.artifact_dir, Path("audit"))

    def test_load_resolves_physical_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.json"
            path.write_text(
                json.dumps({**self.scene(), "physical_network": "physics.json"})
            )
            self.assertEqual(
                SimulationConfig.load(path).physical_network,
                str(Path(folder) / "physics.json"),
            )
