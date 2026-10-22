import json
import unittest
from copy import deepcopy
from pathlib import Path

from orbitserve.experiments.network_inputs import materialize_network_input
from orbitserve.runtime import NetworkSpec, candidate_links

ROOT = Path(__file__).resolve().parents[3]


class NetworkInputTests(unittest.TestCase):
    def test_radio_generation_and_ground_service_survive_materialization(self):
        value = json.loads((ROOT / 'tests/fixtures/network/multi_shell.json').read_text())
        value['constellations']['communication_shells'][0]['radio_profile'] = 'starlink_v3_design_160_20261021'
        value['ground_ingresses'][0]['ground_kind'] = 'user'
        result = materialize_network_input(value)
        spec = NetworkSpec.from_value(result['network'])
        comm = [n for n in spec.nodes if n.shell_id == 'comm-low']
        self.assertTrue(all(n.radio_profile == 'starlink_v3_design_160_20261021' for n in comm))
        self.assertEqual(next(n.ground_kind for n in spec.nodes if n.role == 'ground'), 'user')
        self.assertNotIn('ground_kind', result['ground_ingresses'][0])
        self.assertEqual(value['constellations']['communication_shells'][0]['source']['altitude_km'], 550.)

    def test_multiple_communication_shells_and_compute_cells_are_separate(self):
        value = json.loads(
            (ROOT / "tests/fixtures/network/multi_shell.json").read_text()
        )
        result = materialize_network_input(value)
        self.assertEqual(
            value["constellations"]["communication_shells"][0]["source"]["altitude_km"],
            550.0,
        )
        self.assertNotIn("nodes", value["network"])
        spec = NetworkSpec.from_value(result["network"])
        comm = [n for n in spec.nodes if n.role == "communication"]
        compute = [n for n in spec.nodes if n.role == "compute"]
        self.assertEqual(len(comm), 48)
        self.assertEqual(len(compute), 8)
        self.assertEqual(len({n.cell_id for n in compute}), 2)
        self.assertEqual(sum(n.compute_target for n in compute), 2)
        nodes = {n.node_id: n for n in spec.nodes}
        degrees = {n.node_id: 0 for n in comm}
        for edge in candidate_links(spec):
            a, b = nodes[edge["endpoint_a"]], nodes[edge["endpoint_b"]]
            if a.role == b.role == "communication":
                self.assertEqual(a.shell_id, b.shell_id)
            if edge["kind"] == "communication_isl":
                a, b = nodes[edge["endpoint_a"]], nodes[edge["endpoint_b"]]
                self.assertEqual(a.shell_id, b.shell_id)
                degrees[a.node_id] += 1
                degrees[b.node_id] += 1
        self.assertEqual(
            {degrees[n.node_id] for n in comm if n.shell_id == "comm-low"}, {4}
        )
        self.assertEqual(
            {degrees[n.node_id] for n in comm if n.shell_id == "comm-high"}, {3}
        )

    def test_cross_shell_configuration_and_explicit_bypass_are_rejected(self):
        value = json.loads(
            (ROOT / "tests/fixtures/network/multi_shell.json").read_text()
        )
        concrete = materialize_network_input(value)
        spec = NetworkSpec.from_value(concrete["network"])
        unsupported = {**spec.to_dict(), "inter_shell_links": True}
        with self.assertRaisesRegex(TypeError, "inter_shell_links"):
            NetworkSpec.from_value(unsupported)

        from dataclasses import replace

        first = next(n.node_id for n in spec.nodes if n.shell_id == "comm-low")
        second = next(n.node_id for n in spec.nodes if n.shell_id == "comm-high")
        with self.assertRaisesRegex(ValueError, "cross-shell"):
            candidate_links(replace(spec, explicit_pairs=((first, second),)))

    def test_formation_shape_is_checked_and_physical_members_have_distinct_elements(
        self,
    ):
        value = json.loads(
            (ROOT / "tests/fixtures/network/multi_shell.json").read_text()
        )
        result = materialize_network_input(value)
        physical = [
            r
            for r in result["elements"]["satellites"]
            if r["satellite_id"].startswith("compute")
        ]
        self.assertEqual(len({r["mean_anomaly_deg"] for r in physical}), 8)
        bad = deepcopy(value)
        bad["constellations"]["compute_formations"][0]["cluster"]["grid_shape"] = "3x2"
        with self.assertRaisesRegex(ValueError, "grid_shape"):
            materialize_network_input(bad)

    def test_user_worker_configuration_is_rejected(self):
        from orbitserve.experiments.network import run_network_experiment

        value = json.loads(
            (ROOT / "tests/fixtures/network/multi_shell.json").read_text()
        )
        value["execution"] = {"workers": 99}
        import tempfile

        with (
            tempfile.TemporaryDirectory() as out,
            self.assertRaisesRegex(ValueError, "fields"),
        ):
            run_network_experiment(value, out)
