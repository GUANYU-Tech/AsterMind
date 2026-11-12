"""A saved physical world is reusable without running the physical engines."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from simulator.config import SimulationConfig
from simulator.physics import load_physics
from tests.system_case import physical_case
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
        cls.physical_input = physical_case()["physical_network"]
        cls.concrete = materialize_network_input(cls.physical_input)
        cls.dataset = cls.root / "physical"
        run_network_experiment(cls.physical_input, cls.dataset, _workers=1)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self, **changes):
        raw = json.loads((ROOT / "simulator/examples/two_cells_ns3.json").read_text())
        raw.update(physical_dataset=str(self.dataset), **changes)
        return SimulationConfig.model_validate(raw)

    def test_runtime_reuses_dataset_with_physical_engines_disabled(self):
        from types import SimpleNamespace
        from tests.network_case import NetworkHost
        from orbitserve.runtime.engine.cell_bridge import RuntimeCellNetwork, CellTransfer
        from orbitserve.runtime.timeline import GlobalTimeline

        def execute(physical):
            host = NetworkHost(SimpleNamespace(time_horizon_s=physical.end_s, routing_policy="earliest_arrival"), SimpleNamespace(network=physical, time=physical.trace.time))
            host.timeline = GlobalTimeline(host=host)
            network = RuntimeCellNetwork(host)
            try:
                for row in physical.frames:
                    host.push(row["time_s"], 0, "network_update")
                for event in physical.events:
                    host.push(event["time_s"], 0, "link_state", payload=event)
                for at in physical.environment_times:
                    host.push(at, 0, "topology_sample")
                for at, index, restored, fault in physical.fault_observations():
                    host.push(at, -2, "fault_batch", payload=((index, fault, restored),))
                source = next(n.node_id for n in physical.nodes.values() if n.role == "ground")
                target = next(n.node_id for n in physical.nodes.values() if n.role == "compute")
                network.submit(CellTransfer("saved-physics", source, target, 128, 0))
                host.run()
                self.assertEqual(len(network.events), 1)
                self.assertEqual(network.events[0]["kind"], "completed")
                self.assertGreater(sum(e.get("bytes_sent", 0) for e in host.log if e["kind"] == "bandwidth_update"), 0)
            finally:
                network.close()
                host.close()

        failure = AssertionError("runtime attempted physical recomputation")
        with patch("orbitserve.experiments.network.run_network_experiment", side_effect=failure), patch("orbitserve.experiments.network.generate_orbit_trace", side_effect=failure), patch.object(LinkProfile, "attributes", side_effect=failure), patch.object(NetworkState, "_compile", side_effect=failure), patch.object(NetworkState, "sun_separation", side_effect=failure):
            first, lifetime = load_physics(self.config())
            second, _ = load_physics(self.config(backend="simgrid", timeline_mode="simgrid"))
            self.assertIsNone(lifetime)
            self.assertEqual(first.dataset_identity, second.dataset_identity)
            self.assertEqual(first.dataset_root, str(self.dataset))
            self.assertEqual(first.snapshot(1), second.snapshot(1))
            execute(first)
            execute(second)

    def test_dataset_only_manifest_loads_without_original_sources_or_initialization(self):
        path = self.root / "scene.json"
        payload = self.config().model_dump(mode="json")
        payload["physical_dataset"] = "physical/physical_dataset.json"
        path.write_text(json.dumps(payload))
        with patch("orbitserve.experiments.network_inputs.materialize_network_input", side_effect=AssertionError("constellation initialized again")):
            config = SimulationConfig.load(path)
            physical, _ = load_physics(config)
        self.assertEqual(physical.dataset_root, str(self.dataset))

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
        physical['network']['fault_timing'] = 'continuous'
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

        first = run_network_experiment(self.physical_input, self.root / "combined", _workers=1)
        _, _, index = validate_dataset_input(str(self.root / "combined"), self.physical_input)
        self.assertEqual(first["trace_id"], index["trace_id"])

    def run_faults(self, runtime_network):
        from types import SimpleNamespace
        from tests.network_case import NetworkHost
        physical, _ = load_physics(self.config(runtime_network=runtime_network))
        host = NetworkHost(SimpleNamespace(time_horizon_s=physical.end_s, routing_policy="earliest_arrival"), SimpleNamespace(network=physical, time=physical.trace.time))
        try:
            for at, index, restored, fault in physical.fault_observations():
                host.push(at, -2, "fault_batch", payload=((index, fault, restored),))
            host.run()
            return physical, host.log
        finally:
            host.close()

    def test_runtime_random_fault_choices_reuse_identical_physical_dataset(self):
        physical=deepcopy(self.physical_input)
        physical['network'].update(fault_timing='boundary', random_faults=dict(
            enabled=True, model='fixed_ratio', target_type='link', ratio=.5,
            seed=51, start_step=2, end_step=3))
        with patch.object(LinkProfile, 'attributes', side_effect=AssertionError('physics recomputed')):
            state, events = self.run_faults({k: physical["network"][k] for k in ("fault_timing", "random_faults")})
        self.assertEqual(state.dataset_root, str(self.dataset))
        self.assertTrue(any(e["kind"] == "fault_injected" and e["source"] == "random" for e in events))

    def test_grouped_models_reuse_physics_and_do_not_invent_censored_repairs(self):
        physical=deepcopy(self.physical_input)
        last=round(physical['time']['duration_s']/physical['time']['step_s'])
        common=dict(enabled=True,seed=42,start_step=0,end_step=2)
        physical['network'].update(fault_timing='boundary',random_faults=dict(
            nodes=dict(common,model='weibull',up_scale_steps=.01,up_shape=2,
                       duration=dict(distribution='fixed',steps=1)),
            links=dict(common,model='per_step',failure_probability=1.,
                       duration=dict(distribution='exponential',mean_steps=3))))
        with patch.object(LinkProfile,'attributes',side_effect=AssertionError('physics recomputed')), \
             patch('orbitserve.runtime.setup.faults.random.Random.expovariate',return_value=last+2.25):
            state, events = self.run_faults({k: physical["network"][k] for k in ("fault_timing", "random_faults")})
        self.assertEqual(state.dataset_root, str(self.dataset))
        onsets=[e for e in events if e['kind']=='fault_injected' and e['source']=='random']
        self.assertEqual({e['target_type'] for e in onsets},{'node','link'})
        self.assertTrue(all(e['right_censored'] for e in onsets if e['target_type']=='link'))
        self.assertFalse(any(e['kind']=='fault_restored' and e['source']=='random' and e['target_type']=='link'
                             for e in events))

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
