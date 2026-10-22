"""A saved physical world is reusable without running the physical engines."""

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.experiments.models import ExperimentSpec
from orbitserve.experiments.network import run_network_experiment
from orbitserve.experiments.network_inputs import materialize_network_input
from orbitserve.experiments.physical_dataset import load_dataset, reuse_dataset

from orbitserve.orbit import load_orbit_trace
from orbitserve.runtime import LinkProfile, NetworkSpec, NetworkState

ROOT = Path(__file__).resolve().parents[3]


class PhysicalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.spec = load_experiment_spec(
            ROOT / "tests/fixtures/manifests/coupled_physical.json"
        )
        cls.concrete = materialize_network_input(cls.spec.physical_network)
        cls.dataset = cls.root / "physical"
        run_network_experiment(cls.spec.physical_network, cls.dataset, _workers=1)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_runtime_reuses_dataset_with_physical_engines_disabled(self):
        reference = replace(self.spec, physical_dataset=str(self.dataset))
        failure = AssertionError("runtime attempted physical recomputation")
        with (
            patch(
                "orbitserve.experiments.design_evaluation.run_network_experiment",
                side_effect=failure,
            ),
            patch(
                "orbitserve.experiments.network.generate_orbit_trace",
                side_effect=failure,
            ),
            patch.object(LinkProfile, "attributes", side_effect=failure),
            patch.object(NetworkState, "_compile", side_effect=failure),
            patch.object(NetworkState, "sun_separation", side_effect=failure),
        ):
            first = run_experiment(
                reference, output_dir=self.root / "run-queue", _workers=1
            )
            physical = deepcopy(self.spec.physical_network)
            physical["network"]["transport_model"] = "fluid"
            second = run_experiment(
                replace(reference, physical_network=physical),
                output_dir=self.root / "run-fluid",
                _workers=1,
            )
        self.assertEqual(first.runtime.request_summary["completed_request_count"], 3)
        self.assertEqual(second.runtime.request_summary["completed_request_count"], 3)
        self.assertTrue(first.network["physical_dataset"]["reused"])
        self.assertEqual(first.network["trace_id"], second.network["trace_id"])

    def test_dataset_only_manifest_loads_without_original_sources_or_initialization(
        self,
    ):
        payload = json.loads(
            (ROOT / "tests/fixtures/manifests/coupled_physical.json").read_text()
        )
        payload.pop("physical_network")
        payload["physical_dataset"] = "physical/physical_dataset.json"
        with patch(
            "orbitserve.experiments.network_inputs.materialize_network_input",
            side_effect=AssertionError("constellation initialized again"),
        ):
            spec = ExperimentSpec.from_dict(payload, base_dir=str(self.root))
        self.assertEqual(spec.physical_network, json.loads(json.dumps(self.concrete)))

    def test_loaded_timeline_matches_live_at_non_sample_times_and_forks_are_isolated(
        self,
    ):
        trace = load_orbit_trace(self.dataset / "orbit_trace/trace_manifest.json")
        spec = NetworkSpec.from_value(self.concrete["network"])
        live = NetworkState(trace, spec, self.concrete["ground_ingresses"])
        saved = NetworkState(
            trace,
            spec,
            self.concrete["ground_ingresses"],
            physical_state=self.dataset / "physical_state.sqlite",
        )
        for t in (0.0, 0.5, 1.0, 2.0, 7.25, 59.9, 120.0):
            self.assertEqual(
                json.loads(json.dumps(saved.snapshot(t))),
                json.loads(json.dumps(live.snapshot(t))),
            )
        fork = saved.fork()
        fork.calendars[("test", "a", "b")] = [(1.0, 2.0)]
        self.assertFalse(saved.calendars)

    def test_fault_forecasts_remain_causal_and_use_saved_physics(self):
        from tests.integration.fixed_design.test_dynamic_network_experiment import (
            manifest,
        )

        physical = manifest()
        physical["network"]["faults"] = [
            {
                "target_type": "link",
                "target_id": "cross_layer:sat-00001:sat-00002",
                "start_s": 0.5,
                "end_s": 30.5,
            }
        ]
        folder = self.root / "faults"
        run_network_experiment(physical, folder, _workers=1)
        trace = load_orbit_trace(folder / "orbit_trace/trace_manifest.json")
        spec = NetworkSpec.from_value(physical["network"])
        live = NetworkState(trace, spec, physical["ground_ingresses"])
        saved = NetworkState(
            trace,
            spec,
            physical["ground_ingresses"],
            physical_state=folder / "physical_state.sqlite",
        )
        with patch.object(
            LinkProfile, "attributes", side_effect=AssertionError("budget recomputed")
        ):
            arrival, _ = saved.transfer("sat-00001", "sat-00002", 0.0, 1000.0)
            self.assertGreater(arrival, 0.5)  # Future outage has not been observed.
        for t, index, restored, fault in saved.fault_observations():
            live.observe_fault(index, fault, t, restored=restored)
            expected = json.loads(json.dumps(live.snapshot(t)))
            with patch.object(
                LinkProfile,
                "attributes",
                side_effect=AssertionError("budget recomputed"),
            ):
                saved.observe_fault(index, fault, t, restored=restored)
                self.assertEqual(json.loads(json.dumps(saved.snapshot(t))), expected)

    def test_combined_run_output_can_be_reused_by_the_original_source_config(self):
        from orbitserve.experiments.physical_dataset import validate_dataset_input

        first = run_experiment(self.spec, output_dir=self.root / "combined", _workers=1)
        _, _, index = validate_dataset_input(
            str(self.root / "combined/network"), self.spec.physical_network
        )
        self.assertEqual(first.network["trace_id"], index["trace_id"])

    def test_missing_corrupt_or_changed_input_is_rejected(self):
        changed = deepcopy(self.concrete)
        changed["time"]["step_s"] *= 2
        with self.assertRaisesRegex(ValueError, "不匹配"):
            load_dataset(self.dataset, changed)
        for i, name in enumerate(
            ("frames/frame-000001.json.gz", "physical_state.sqlite")
        ):
            copy = self.root / f"corrupt-{i}"
            reuse_dataset(str(self.dataset), copy, self.concrete)
            artifact = copy / name
            # Replace rather than mutate shared immutable hard links.
            artifact.unlink()
            if i:
                artifact.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "missing or corrupt"):
                load_dataset(copy, self.concrete)
        self.assertTrue(
            json.loads((self.dataset / "physical_dataset.json").read_text())["files"]
        )
