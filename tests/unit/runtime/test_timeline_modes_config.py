"""Canonical backend/mode combinations and legacy configuration compatibility."""

import json
from pathlib import Path
import unittest

from pydantic import ValidationError
from astermind.builder.config import SimulationConfig
from astermind.builder.scene import SceneDeployment


class TimelineModeConfigTests(unittest.TestCase):
    def configuration(self, **updates):
        value = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "examples/native/two_cells.json"
            ).read_text()
        )
        value.update(updates)
        return SimulationConfig.model_validate(value)

    def test_existing_documents_keep_orbitserve_timeline(self):
        value = self.configuration()
        self.assertEqual(value.timeline_mode, "orbitserve")
        self.assertEqual(value.backend, "flowsim")
        self.assertEqual(SceneDeployment().timeline_mode, "orbitserve")
        self.assertEqual(SceneDeployment().communication_mode, "protobridge_cell")

    def test_both_backends_accept_all_three_communication_scopes_and_roundtrip(self):
        for timeline, backend in [("orbitserve", "flowsim"), ("simgrid", "simgrid")]:
            for scope in ("flowsim", "protobridge_cell", "protobridge_satellite"):
                value = self.configuration(
                    timeline_mode=timeline, backend=backend, communication_mode=scope
                )
                restored = SimulationConfig.model_validate_json(value.model_dump_json())
                self.assertEqual(restored, value)

    def test_invalid_manual_json_cannot_mix_clock_and_backend(self):
        for value in [
            dict(timeline_mode="simgrid", backend="flowsim"),
            dict(timeline_mode="orbitserve", backend="simgrid"),
            dict(timeline_mode="unknown"),
            dict(timeline_mode="simgrid", backend="ns3"),
        ]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.configuration(**value)

    def test_ns3_legacy_mode_remains_available_and_rejects_surrogate(self):
        self.assertEqual(self.configuration(backend="ns3").timeline_mode, "orbitserve")
        with self.assertRaises(ValidationError):
            self.configuration(backend="ns3", communication_mode="protobridge_cell")
