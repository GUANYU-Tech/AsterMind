import json
import math
import unittest
from copy import deepcopy
from pathlib import Path

from astermind.builder.physical.network_inputs import materialize_network_input
from astermind.engine import NetworkSpec, candidate_links

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
        from astermind.builder.physical.network import run_network_experiment

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

    def test_physical_override_only_changes_the_selected_cell(self):
        value = json.loads((ROOT / 'tests/fixtures/network/multi_shell.json').read_text())
        original = materialize_network_input(value)
        cells = sorted({row['cell_id'] for row in original['network']['nodes'] if row['role'] == 'compute'})
        formation = value['constellations']['compute_formations'][0]
        formation['cell_overrides'] = {cells[0]: {
            'satellites_per_cluster': 3, 'grid_shape': '3x1',
            'formation_pitch_m': 1000, 'altitude_km': 650, 'topology': 'fully_connected',
        }}
        rebuilt = materialize_network_input(value)
        members = [row for row in rebuilt['network']['nodes'] if row.get('cell_id') == cells[0]]
        self.assertEqual(len(members), 3)
        previous_sibling = [row for row in original['elements']['satellites'] if row['satellite_id'].startswith(cells[1] + '-')]
        rebuilt_sibling = [row for row in rebuilt['elements']['satellites'] if row['satellite_id'].startswith(cells[1] + '-')]
        # Synthetic catalog numbers are assigned across the rebuilt dataset.
        def without_catalog_number(rows):
            return [{key: value for key, value in row.items() if key != 'satnum'} for row in rows]
        self.assertEqual(without_catalog_number(rebuilt_sibling), without_catalog_number(previous_sibling))
        self.assertEqual(rebuilt['network']['cell_topologies'][cells[0]], 'fully_connected')

    def test_static_ring_and_chain_geometry_survive_orbit_propagation(self):
        from astermind.builder.orbit import load_sgp4_element_bundle, propagate_sgp4_teme

        manifest = json.loads((ROOT / 'tests/fixtures/network/multi_shell.json').read_text())
        formation = manifest['constellations']['compute_formations'][0]
        formation['cluster'].update(motion_model='static_formation', satellites_per_cluster=10,
                                    formation_pitch_m=500., grid_shape='5x2')
        formation['topology'] = 'ring'
        name = formation['name']
        first, second = [f'{name}-p000-s{i:04d}' for i in range(2)]
        formation['source'].update(planes=1, clusters_per_plane=2, walker_phase=0)
        formation['cell_overrides'] = {second:{'topology':'chain','satellites_per_cluster':4}}
        raw = materialize_network_input(manifest)
        bundle = load_sgp4_element_bundle(raw['elements'])
        samples = propagate_sgp4_teme(bundle, simulation_epoch_utc=manifest['time']['epoch_utc'],
                                      sample_times_s=(0., 600.)).samples
        for time in (0., 600.):
            frame = {s.satellite_id:s.position_teme_km for s in samples if s.time_s == time}
            ring = [frame[f'{first}-m{i:03d}'] for i in range(10)]
            center = tuple(sum(axis)/10 for axis in zip(*ring))
            for i, point in enumerate(ring):
                self.assertAlmostEqual(math.dist(point, center), .5/(2*math.sin(math.pi/10)), places=7)
                self.assertAlmostEqual(math.dist(point, ring[(i+1)%10]), .5, places=7)
            chain = [frame[f'{second}-m{i:03d}'] for i in range(4)]
            for a,b in zip(chain,chain[1:]):
                self.assertAlmostEqual(math.dist(a,b), .5, places=7)
            self.assertGreater(math.dist(chain[0],chain[-1]), 1.49)

    def test_static_cell_override_preserves_other_free_propagating_members(self):
        manifest = json.loads((ROOT / 'tests/fixtures/network/multi_shell.json').read_text())
        f = manifest['constellations']['compute_formations'][0]
        target = f['name']+'-p000-s0000'
        original = materialize_network_input(manifest)
        f['cell_overrides'] = {target:{'motion_model':'static_formation','topology':'fully_connected'}}
        updated = materialize_network_input(manifest)
        for before, after in zip(original['elements']['satellites'],updated['elements']['satellites'],strict=True):
            if before['satellite_id'].startswith(target+'-'):
                self.assertIn('fixed_lvlh_offset_km',after)
            else:
                self.assertEqual(before,after)
