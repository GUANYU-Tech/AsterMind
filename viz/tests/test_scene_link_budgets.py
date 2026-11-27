"""Check which physical budget each full-scene link actually consumes."""
import json
from pathlib import Path
import unittest

from astermind.builder.physical.network_inputs import materialize_network_input
from astermind.engine import NetworkSpec, candidate_links
from astermind.engine.setup.islsim import OPTICAL_LINKS, RF_DIRECTIONS
from astermind.engine.setup.radio_capacity import directional_profile


class SceneLinkBudgetsTest(unittest.TestCase):
    def setUp(self):
        template = Path(__file__).resolve().parents[1] / 'templates' / 'starlink_550.json'
        self.scene = json.loads(template.read_text(encoding='utf-8'))
        shell = self.scene['constellations']['communication_shells'][0]
        shell['source'].update(planes=1, satellites_per_plane=3, walker_phase=0)
        shell['isl_profile'] = 'obsolete'
        shell['hardware_link_profiles'] = {'communication_isl': {'bandwidth_bps': 1}}
        formation = self.scene['constellations']['compute_formations'][0]
        formation['cluster'].update(satellites_per_cluster=2, grid_shape='2x1')
        formation['sgl_terminals'] = 1
        source = self.scene['ground_ingresses'][0]
        self.scene['ground_ingresses'] = [dict(source, ingress_id=k, ground_kind=k)
                                          for k in ('user', 'gateway')]
        self.net = self.scene['network']
        self.net.update(optical_link_model='fixed', inter_cell_links=True,
                        ground_roles=['communication', 'compute'])
        self.profiles = self.net['link_profiles'] = {
            kind: {'capacity_model': 'fixed', 'bandwidth_bps': (51 + i) * 1e9,
                   'setup_time_s': 2 if kind in OPTICAL_LINKS else .1}
            for i, kind in enumerate((*OPTICAL_LINKS, *RF_DIRECTIONS, 'sgl'))}

    def spec(self):
        return NetworkSpec.from_value(materialize_network_input(self.scene)['network'])

    def test_direction_selection_and_shell_overrides(self):
        spec = self.spec()
        nodes = {node.node_id: node for node in spec.nodes}
        covered = set()
        for edge in candidate_links(spec):
            kind = edge['kind']
            self.assertEqual(edge['profile_id'], kind)
            if kind != 'sgl':
                result = spec.link_profiles[kind].attributes(50, 0)
                self.assertEqual(result['budget_bandwidth_bps'], self.profiles[kind]['bandwidth_bps'])
                covered.add(kind)
                continue
            a, b = (nodes[edge[k]] for k in ('endpoint_a', 'endpoint_b'))
            ground, satellite = (a, b) if a.role == 'ground' else (b, a)
            service = 'user' if ground.ground_kind == 'user' else 'feeder'
            for start, direction in [(ground, 'uplink'), (satellite, 'downlink')]:
                key = service + '_' + direction
                self.assertIs(directional_profile(nodes, spec.link_profiles, edge, start.node_id),
                              spec.link_profiles[key])
                covered.add((satellite.role, key))
        self.assertEqual(covered, set(OPTICAL_LINKS) | {(role, key) for key in RF_DIRECTIONS
                                                       for role in ('compute', 'communication')})

    def test_one_model_switch_and_recording_compatibility(self):
        self.net['optical_link_model'] = 'islsim'
        spec = self.spec()
        for key in OPTICAL_LINKS:
            self.assertEqual(spec.link_profiles[key].capacity_model, 'islsim')
            self.assertIsNone(spec.link_profiles[key].islsim_config['rx']['terminal_cap_gbps'])
        recording = spec.to_dict()
        del recording['optical_link_model']
        profile = recording['link_profiles']['communication_isl']
        profile['capacity_model'] = 'optical_photon_budget'
        with self.assertRaises(ValueError):
            NetworkSpec.from_value(recording)
        archived = NetworkSpec.from_recording(recording)
        self.assertEqual(archived.link_profiles['communication_isl'].capacity_model, 'optical_photon_budget')
        with self.assertRaises(ValueError):
            archived.link_profiles['communication_isl'].attributes(50, 0)
