import json
import tempfile
from pathlib import Path
import unittest
from copy import deepcopy

from orbitserve.experiments.network import run_network_experiment
from tests.system_case import physical_case
from tests.unit.experiments import test_network_scenarios as scenario_fixtures


class NetworkScenarioIntegrationTests(unittest.TestCase):
    def test_irregular_tle_baseline_and_method1_export_actual_sparse_topologies(self):
        value = scenario_fixtures.NetworkScenarioTests().fixture()
        source, present = scenario_fixtures.NetworkScenarioTests().tle_source()
        shell = value['constellations']['communication_shells'][0]
        shell.update(source=source, topology='honeycomb')
        value['constellations']['communication_shells']=[shell]
        value['constellations']['compute_formations']=[]
        value['scenario_type']='communication_only'
        value['time'].update(duration_s=2.,step_s=1.)
        value['route_probes']=[]
        candidates=[]
        for mode in ('none','ronchetti_intra_orbit'):
            shell['topology_repair']=mode
            with tempfile.TemporaryDirectory() as folder:
                result=run_network_experiment(value,folder,_workers=1)
                self.assertTrue(result['checks']['passed'])
                self.assertEqual(result['checks']['physical_satellites'],len(present))
                self.assertEqual(result['provenance']['initialization'][0]['missing_grid_positions'],8)
                candidates.append(result['candidate_links'])
                self.assertTrue((Path(folder)/'network_result.json').is_file())
        self.assertGreater(len(candidates[1]),len(candidates[0]))
        self.assertTrue(any(e.get('topology_patch') for e in candidates[1]))

    def test_compute_only_exports_direct_ground_access(self):
        spec=physical_case()
        physical=deepcopy(spec["physical_network"])
        physical['scenario_type']='compute_only'
        physical['constellations']['communication_shells']=[]
        physical['network']['faults']=[]
        physical['network']['ground_matching']='deferred_acceptance'
        for formation in physical['constellations']['compute_formations']:
            formation.update(sgl_terminals=2,max_user_links=1,max_gateway_links=1,
                             egress_policy='free')
        with tempfile.TemporaryDirectory() as folder:
            result = run_network_experiment(physical, folder, _workers=1)
            self.assertTrue(result["checks"]["passed"])
            materialized = json.loads((Path(folder) / "materialized_input.json").read_text())
            self.assertEqual(materialized["network"]["ground_roles"], ["compute"])
            self.assertFalse(any(n["role"] == "communication" for n in materialized["network"]["nodes"]))
