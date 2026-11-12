"""Real ISLSim budgets feed every transport backend, without formula patches."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astermind.engine import NetworkState
from tests.network_case import run_traffic
from astermind.engine.setup import islsim
from astermind.builder.physical.network import run_network_experiment
from astermind.builder.physical.network_inputs import snapshot_network_input
from astermind.builder.physical.physical_dataset import input_fingerprint, load_dataset
from tests.integration.fixed_design.test_network_acceptance import acceptance_spec
from tests.integration.fixed_design.test_dynamic_network_experiment import manifest


class ISLSimRuntimeTests(unittest.TestCase):
    def verify_backend(self, mode):
        base = acceptance_spec()
        profiles = {name: replace(p, capacity_model='islsim') if name != 'sgl' and p.capacity_model != 'free_space_shannon'
                    and name != 'downlink' else p for name, p in base.network.link_profiles.items()}
        spec = replace(base, network=replace(base.network, link_profiles=profiles, transport_model=mode))
        _, _, (_, model, _) = islsim._api()
        with patch.object(model, 'compute_link', wraps=model.compute_link) as native:
            result = run_traffic(spec)
        self.assertGreater(native.call_count, 0)
        self.assertEqual(result.request_summary['completed_request_count'], 3)
        state = NetworkState(spec.environment_state, spec.network, spec.gateways)
        links = [link for link in state.snapshot(1.)['links'] if link['kind'] != 'sgl']
        self.assertTrue(links)
        self.assertTrue(all(link['capacity_model'] == 'islsim' for link in links))
        for request in result.transfer_jobs:
            for leg in request['routing']['selected_path']:
                if 'cancelled_at_s' not in leg:
                    self.assertAlmostEqual(leg['propagation_delay_s'], leg['distance_km'] / 299792.458)

    def test_queue_uses_real_islsim(self): self.verify_backend('queue')
    def test_fluid_uses_real_islsim(self): self.verify_backend('fluid')

    @unittest.skipUnless(importlib.util.find_spec('simgrid'), 'requires native SimGrid')
    def test_simgrid_uses_real_islsim(self): self.verify_backend('simgrid')

    @unittest.skipUnless(importlib.util.find_spec('ns'), 'requires native ns-3')
    def test_ns3_uses_real_islsim(self): self.verify_backend('ns3')

    def test_default_experiment_freezes_native_input_and_rejects_other_model_dataset(self):
        original = manifest()
        original['network']['link_profiles']['cross_layer'].pop('capacity_model')
        frozen = snapshot_network_input(original)
        p = frozen['network']['link_profiles']['cross_layer']
        self.assertEqual(p['capacity_model'], 'islsim')
        self.assertEqual(set(p['islsim_config']), {'tx', 'rx', 'channel'})
        self.assertEqual(p['islsim_source_sha256'], {})
        self.assertNotIn('capacity_model', original['network']['link_profiles']['cross_layer'])
        with tempfile.TemporaryDirectory() as folder:
            result = run_network_experiment(original, folder, _workers=1, export_mode='core')
            self.assertTrue(result['checks']['passed'])
            saved = json.loads((Path(folder) / 'materialized_input.json').read_text())
            self.assertEqual(saved['network']['link_profiles']['cross_layer']['islsim_config'], p['islsim_config'])
            self.assertEqual(set(saved['network']['link_profiles']['cross_layer']['islsim_source_sha256']),
                             {'config.py', 'link_model.py', 'validator.py'})
            load_dataset(folder, saved)
            changed = snapshot_network_input(original, bind_islsim=True)
            changed['network']['link_profiles']['cross_layer']['capacity_model'] = 'fixed'
            self.assertNotEqual(input_fingerprint(changed), input_fingerprint(frozen))
            with self.assertRaises(ValueError):
                load_dataset(folder, changed)
