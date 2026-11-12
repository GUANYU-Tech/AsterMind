"""Local checks that exported generation inputs drive the real ISLSim model."""
from copy import deepcopy
import json
import unittest

from astermind.builder.physical.network_inputs import materialize_network_input
from astermind.engine import NetworkSpec, STARLINK_RADIO_PROFILES, prepare_islsim_config
from astermind.engine.setup import islsim
from tests.unit.runtime.test_starlink_presets import generated_profiles, ROOT


class ISLSimPresetTests(unittest.TestCase):
    def test_generation_caps_reach_real_model(self):
        for identity in STARLINK_RADIO_PROFILES:
            with self.subTest(identity=identity):
                spec, _ = generated_profiles(identity)
                shell = next(s for s in spec.communication_shells if s['shell_id'] == 'comm-low')
                profile = spec.link_profiles[shell['isl_profile']]
                result = profile.attributes(1000, 0)
                self.assertEqual(result['islsim_total_bw_gbps'], profile.bandwidth_bps / 1e9)
                self.assertTrue(result['islsim_source_sha256'])

    def test_scenario_parameters_change_real_feasibility(self):
        original = json.loads((ROOT / 'tests/fixtures/network/multi_shell.json').read_text())
        shell = original['constellations']['communication_shells'][0]
        shell.update(radio_profile='starlink_v2_mini_2024', isl_terminals=3)
        baseline = materialize_network_input(original)
        shell['hardware_link_profiles'] = {'communication_isl': {
            'islsim_config': {'tx': {'tx_power_w': .05, 'modulation_pool': [[4, 'DP-QPSK']]},
                              'rx': {'terminal_cap_gbps': 80}}}}
        changed = materialize_network_input(original)
        name = 'comm-low__hardware__communication_isl'
        a, b = (NetworkSpec.from_value(value['network']).link_profiles[name] for value in (baseline, changed))
        self.assertTrue(a.attributes(1000, 0)['link_feasible'])
        self.assertFalse(b.attributes(1000, 0)['link_feasible'])

    def test_exported_defaults_are_accepted_by_native_configuration(self):
        exported = prepare_islsim_config()
        _, _, (config, _, validator) = islsim._api()
        value = deepcopy(exported)
        native = config.FSOConfig(tx=config.TXConfig(**value['tx']), rx=config.RXConfig(**value['rx']),
                                  channel=config.ChannelConfig(**value['channel']))
        validator.validate_config(native)
        self.assertEqual(config.config_to_dict(native), exported)
