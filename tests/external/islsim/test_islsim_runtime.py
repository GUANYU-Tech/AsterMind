"""Real ISLSim budgets feed the shared Comm, without formula patches."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astermind.engine import NetworkState
from tests.network_case import device_runtime
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
        completed = []
        submitted = []
        delivery_paths = {}
        expected_ids = {
            f"{row['request_id']}:{direction}"
            for row in spec.traffic for direction in ('input', 'output')
        }
        self.assertEqual(len(expected_ids), 2 * len(spec.traffic))
        with patch.object(model, 'compute_link', wraps=model.compute_link) as native:
            state = NetworkState(spec.environment_state, spec.network, spec.gateways)
            host, runtime = device_runtime(state)
            try:
                policy = runtime.communication_policy

                def failed(error):
                    self.fail(f"real ISLSim transfer failed: {error}")

                def submit(row, direction):
                    identity = f"{row['request_id']}:{direction}"
                    source, target = row['ingress_id'], spec.target
                    if direction == 'output':
                        source, target = target, source
                    size = row['input_bytes'] if direction == 'input' else row.get('output_bytes', 0)
                    self.assertEqual(size, int(size))
                    self.assertNotIn(identity, submitted)
                    submitted.append(identity)
                    started = host.now

                    def done():
                        self.assertNotIn(identity, completed)
                        self.assertGreaterEqual(host.now, started)
                        completed.append(identity)
                        if direction == 'input':
                            submit(row, 'output')

                    # Use the production Comm object owned by DeviceCommunicationRuntime.
                    policy.submit_transfer(identity, source + ':hub', target + ':hub',
                                           int(size), done, failed)

                for row in spec.traffic:
                    host.timeline.schedule(row['arrival_time_s'], 30,
                                           'islsim:' + row['request_id'],
                                           lambda row=row: submit(row, 'input'))
                host.run()
                self.assertEqual(set(completed), expected_ids)
                self.assertEqual(len(completed), len(expected_ids))
                self.assertEqual(set(submitted), expected_ids)
                self.assertFalse(policy.tasks, 'unfinished communication tasks')
                delivery_paths = dict(policy.delivery_paths)
                for identity in expected_ids:
                    self.assertIn(identity, delivery_paths)
                    self.assertGreaterEqual(len(delivery_paths[identity]), 2)
            finally:
                runtime.close()
                host.close()
        self.assertGreater(native.call_count, 0)
        links = [link for link in state.snapshot(1.)['links'] if link['kind'] != 'sgl']
        self.assertTrue(links)
        self.assertTrue(all(link['capacity_model'] == 'islsim' for link in links))
        for row in spec.traffic:
            _, path = state.transfer(row['ingress_id'], spec.target,
                                     row['arrival_time_s'], row['input_bytes'])
            self.assertTrue(path)
            for leg in path:
                self.assertAlmostEqual(leg['propagation_delay_s'],
                                       leg['distance_km'] / 299792.458)

    def test_runtime_comm_uses_real_islsim(self): self.verify_backend('runtime_comm')

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
