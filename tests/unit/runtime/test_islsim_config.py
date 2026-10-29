"""The repository exports ISLSim inputs without a local model checkout."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from orbitserve.experiments.network_inputs import snapshot_network_input
from orbitserve.experiments.physical_dataset import input_fingerprint
from orbitserve.runtime import LinkProfile, NetworkSpec, prepare_islsim_config
from orbitserve.runtime.setup import islsim
from tests.integration.fixed_design.test_dynamic_network_experiment import manifest


class ISLSimConfigTests(unittest.TestCase):
    def test_network_configuration_and_identity_do_not_load_external_model(self):
        value = manifest()
        value['network']['link_profiles']['cross_layer'].pop('capacity_model')
        before = deepcopy(value)
        with patch.object(islsim, '_api', side_effect=AssertionError('must not load external model')):
            snapshot = snapshot_network_input(value)
            profile = NetworkSpec.from_value(snapshot['network']).link_profiles['cross_layer']
            self.assertEqual(profile.capacity_model, 'islsim')
            self.assertEqual(set(profile.islsim_config), {'tx', 'rx', 'channel'})
            self.assertEqual(profile.islsim_source_sha256, {})
            self.assertEqual(input_fingerprint(value), input_fingerprint(snapshot))
        self.assertEqual(value, before)

    def test_partial_parameters_are_complete_and_independent(self):
        parameters = {'tx': {'tx_power_w': .05, 'modulation_pool': [[4, 'DP-QPSK']]},
                      'rx': {'terminal_cap_gbps': 100.}}
        before = deepcopy(parameters)
        result = prepare_islsim_config(parameters)
        self.assertEqual(result['rx']['terminal_cap_gbps'], 100.)
        self.assertEqual(result['tx']['modulation_pool'], [[4, 'DP-QPSK']])
        self.assertEqual(result['tx']['wavelength_m'], 1.55e-6)
        result['tx']['modulation_pool'].clear()
        self.assertTrue(prepare_islsim_config()['tx']['modulation_pool'])
        self.assertEqual(parameters, before)

    def test_invalid_inputs_are_rejected_before_export(self):
        invalid = [None, [], {'unknown': {}}, {'tx': []}, {'tx': {'unknown': 1}},
                   {'tx': {'wdm_channels': True}}, {'tx': {'dual_polarization': 'false'}},
                   {'tx': {'wdm_power_mode': 'invalid'}}, {'tx': {'modulation_pool': []}},
                   {'tx': {'modulation_pool': [[3, 'invalid']]}}, {'tx': {'modulation_pool': [[4, '']]}},
                   {'rx': {'ber_threshold': 1}}, {'channel': {'optical_efficiency': 0}}]
        invalid.extend({'tx': {'tx_power_w': value}} for value in (0, -1, True, '5', float('nan'), float('inf')))
        for parameters in invalid[1:]:  # None means use OrbitServe's explicit defaults.
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                prepare_islsim_config(parameters)

    def test_cli_writes_native_json_with_missing_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = (Path(directory) / name for name in ('profile.json', 'islsim.json'))
            source.write_text(json.dumps({'islsim_config': {'rx': {'terminal_cap_gbps': 200}}}))
            environment = {**os.environ, 'ORBITSERVE_ISLSIM_ROOT': '/nonexistent/islsim-config-test'}
            process = subprocess.run([sys.executable, '-m', 'orbitserve.experiments.islsim_cli',
                                      '--input', str(source), '--out', str(output)],
                                     capture_output=True, text=True, env=environment)
            self.assertEqual(process.returncode, 0, process.stderr)
            value = json.loads(output.read_text())
            self.assertEqual(set(value), {'tx', 'rx', 'channel'})
            self.assertEqual(value['rx']['terminal_cap_gbps'], 200)
            self.assertNotIn(b'\r', output.read_bytes())

    def test_real_calculation_requires_local_provider_without_fallback(self):
        with patch.object(islsim, '_api', side_effect=islsim.ISLSimUnavailable('missing local ISLSim')):
            profile = LinkProfile(1e12, 0, capacity_model='islsim')
            with self.assertRaisesRegex(islsim.ISLSimUnavailable, 'missing local ISLSim'):
                profile.attributes(1000, 0)
