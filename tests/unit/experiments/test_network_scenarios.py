import json
from pathlib import Path
import unittest

from astermind.builder.physical.network_inputs import materialize_network_input
from astermind.builder.orbit import walker_delta_elements
from astermind.builder.orbit.sgp4.propagation import _initialize_satellite, _load_sgp4_api
from astermind.engine import NetworkSpec, candidate_links

ROOT = Path(__file__).resolve().parents[3]


class NetworkScenarioTests(unittest.TestCase):
    def fixture(self):
        return json.loads((ROOT/'tests/fixtures/network/multi_shell.json').read_text())

    def test_all_three_scenarios_and_mismatched_selection(self):
        for scenario in ('compute_only','communication_only','mixed'):
            value = self.fixture()
            value['scenario_type'] = scenario
            if scenario == 'compute_only':
                value['constellations']['communication_shells'] = []
                value['constellations']['compute_formations'][0]['sgl_terminals'] = 2
            if scenario == 'communication_only':
                value['constellations']['compute_formations'] = []
            result = materialize_network_input(value)
            spec = NetworkSpec.from_value(result['network'])
            edges = candidate_links(spec)
            self.assertTrue(edges)
            if scenario == 'compute_only':
                self.assertEqual(spec.ground_roles, ('compute',))
                self.assertTrue(any(e['kind']=='sgl' for e in edges))
            self.assertEqual('communication' in {n.role for n in spec.nodes},scenario!='compute_only')
            self.assertEqual('compute' in {n.role for n in spec.nodes},scenario!='communication_only')
        value['scenario_type'] = 'compute_only'
        with self.assertRaisesRegex(ValueError,'scenario_type'):
            materialize_network_input(value)

    def test_egress_modes_and_quota_materialization(self):
        for mode, members in (('single',[0]),('multi',[0,2]),('free',[])):
            value = self.fixture()
            formation = value['constellations']['compute_formations'][0]
            formation.update(egress_policy=mode,ingress_members=members,max_user_links=5,max_gateway_links=2)
            spec = NetworkSpec.from_value(materialize_network_input(value)['network'])
            cells = {n.cell_id for n in spec.nodes if n.role=='compute'}
            for cell in cells:
                actual = [n for n in spec.nodes if n.cell_id==cell and spec.external_member(n)]
                self.assertEqual(len(actual), formation['cluster']['satellites_per_cluster'] if mode=='free' else len(members))
                self.assertTrue(all(n.max_user_links==5 and n.max_gateway_links==2 for n in actual))

    def test_dawn_dusk_works_for_communication_and_compute(self):
        value = self.fixture()
        shell = value['constellations']['communication_shells'][0]
        shell['source'].update(type='dawn_dusk',planes=1,ltan_hours=18)
        shell['topology']='grid'
        formation = value['constellations']['compute_formations'][0]
        formation['source'].update(type='dawn_dusk',planes=1)
        spec = NetworkSpec.from_value(materialize_network_input(value)['network'])
        self.assertTrue(candidate_links(spec))

    def tle_source(self):
        from sgp4.exporter import export_tle
        source = walker_delta_elements(name='observed',altitude_km=550,inclination_deg=53,
                                      planes=4,satellites_per_plane=6,epoch_utc='2026-07-20T00:00:00Z')
        present = [s for i,s in enumerate(source.satellites) if i not in (1,2,6,7,8,9,10,11)]
        text = '\n'.join('\n'.join(export_tle(_initialize_satellite(_load_sgp4_api(),s))) for s in present)
        return dict(type='tle',tle_text=text,planes=4,satellites_per_plane=6,raan0_deg=0,u0_deg=0),present

    def test_irregular_tle_preserves_holes_and_observed_elements(self):
        value = self.fixture()
        source,present = self.tle_source()
        shell = value['constellations']['communication_shells'][0]
        shell.update(source=source,topology='honeycomb',topology_repair='none')
        value['constellations']['communication_shells']=[shell]
        result = materialize_network_input(value)
        nodes = [n for n in result['network']['nodes'] if n['role']=='communication']
        self.assertEqual(len(nodes),len(present))
        self.assertFalse(any(n['plane']==1 for n in nodes))
        self.assertFalse(any(n['plane']==0 and n['slot'] in (1,2) for n in nodes))
        self.assertEqual(result['initialization_provenance'][0]['missing_grid_positions'],8)
        spec = NetworkSpec.from_value(result['network'])
        baseline = candidate_links(spec)
        shell['topology_repair']='ronchetti_intra_orbit'
        repaired = materialize_network_input(value)
        self.assertEqual(result['elements'],repaired['elements'])
        self.assertGreater(len(candidate_links(NetworkSpec.from_value(repaired['network']))),len(baseline))

    def test_tle_explicit_catalog_map_and_ambiguous_grid_are_checked(self):
        source,present = self.tle_source()
        value=self.fixture()
        shell=value['constellations']['communication_shells'][0]
        shell.update(source=source,topology='grid')
        source['grid_positions']={str(present[0].satnum):[2,4]}
        concrete=materialize_network_input(value)
        nodes=[n for n in concrete['network']['nodes'] if n.get('shell_id')==shell['shell_id']]
        self.assertEqual(sum(n['plane'] is not None for n in nodes),1)
        source['grid_positions'][str(present[1].satnum)]=[2,4]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            materialize_network_input(value)
        del source['grid_positions']
        del source['satellites_per_plane']
        with self.assertRaisesRegex(ValueError,'nominal'):
            materialize_network_input(value)

    def test_compute_tle_requires_explicit_partition_and_preserves_observations(self):
        source,present = self.tle_source()
        value=self.fixture()
        value['constellations']['communication_shells']=[]
        formation=value['constellations']['compute_formations'][0]
        formation['source']=source
        count=formation['cluster']['satellites_per_cluster']
        source['cell_members']={f'cell-{i//count}':[s.satnum for s in present[i:i+count]] for i in range(0,len(present),count)}
        result=materialize_network_input(value)
        spec=NetworkSpec.from_value(result['network'])
        self.assertEqual(len([n for n in spec.nodes if n.role=='compute']),len(present))
        self.assertTrue(result['elements']['provenance']['observationally_fitted'])
        self.assertEqual({r['satnum'] for r in result['elements']['satellites']},{s.satnum for s in present})
        source['cell_members']['cell-0'][0]=source['cell_members']['cell-1'][0]
        with self.assertRaisesRegex(ValueError,'partition'):
            materialize_network_input(value)
