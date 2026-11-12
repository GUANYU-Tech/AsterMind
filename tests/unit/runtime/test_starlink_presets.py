"""Public Starlink aggregates, assumptions and actual directed RF execution."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from astermind.builder.physical.network_inputs import materialize_network_input
from astermind.engine import NetworkNodeSpec, NetworkSpec, NetworkState, STARLINK_RADIO_PROFILES
from astermind.engine.setup.radio_capacity import directional_profile, max_min_rates, hop_pools

ROOT = Path(__file__).resolve().parents[3]


def generated_profiles(identity):
    manifest = json.loads((ROOT/'tests/fixtures/network/multi_shell.json').read_text())
    shell = manifest['constellations']['communication_shells'][0]
    shell.update(radio_profile=identity, isl_terminals=3 if 'v2_' in identity else 6)
    result = materialize_network_input(manifest)
    spec = NetworkSpec.from_value(result['network'])
    return spec, manifest


def directed_state(identity):
    generated, _ = generated_profiles(identity)
    satellite = replace(next(n for n in generated.nodes if n.shell_id == 'comm-low'), node_id='a', sgl_terminals=2)
    grounds = (NetworkNodeSpec('u', 'ground', sgl_terminals=1, ground_kind='user'),
               NetworkNodeSpec('g', 'ground', sgl_terminals=1, ground_kind='gateway'))
    trace = SimpleNamespace(time={'start_s':0., 'end_s':20., 'step_s':10.},
        satellite_catalog=({'satellite_id':'a'},),
        ephemeris_samples=tuple(dict(satellite_id='a', time_s=t, position_itrs_km=(7000.+t*10,0.,0.)) for t in (0.,10.,20.)),
        isl_windows=(), sgl_windows=tuple(dict(ingress_id=g.node_id,satellite_id='a',start_s=0.,end_s=20.) for g in grounds))
    spec = NetworkSpec((satellite,*grounds),generated.link_profiles,communication_topology='explicit')
    return NetworkState(trace,spec,tuple(dict(ingress_id=g.node_id,lat_deg=0.,lon_deg=0.) for g in grounds),
                        sun_positions=tuple(dict(time_s=t,position_itrs_km=(0.,1.5e8,0.)) for t in (0.,10.,20.)))


class StarlinkPresetTests(unittest.TestCase):
    def test_materialization_prefills_both_generations_without_changing_input(self):
        for identity in STARLINK_RADIO_PROFILES:
            with self.subTest(identity=identity):
                spec, original = generated_profiles(identity)
                self.assertNotIn('hardware_link_profiles',original['constellations']['communication_shells'][0])
                node = next(n for n in spec.nodes if n.shell_id=='comm-low')
                self.assertEqual(set(node.sgl_profiles),{'user_uplink','user_downlink','feeder_uplink','feeder_downlink'})
                self.assertTrue(node.capacity_assumptions)
                self.assertTrue(all(name in spec.link_profiles for name in node.sgl_profiles.values()))
                self.assertEqual(spec.link_profiles[node.sgl_profiles['user_uplink']].frequency_hz,14.25e9)
                self.assertEqual(spec.link_profiles[node.sgl_profiles['user_downlink']].frequency_hz,12e9)
                shell = next(s for s in spec.communication_shells if s['shell_id']=='comm-low')
                optical = spec.link_profiles[shell['isl_profile']]
                self.assertEqual(optical.capacity_model,'islsim')
                self.assertEqual(set(optical.islsim_config), {'tx', 'rx', 'channel'})
                self.assertEqual(optical.islsim_source_sha256, {})
                self.assertEqual(optical.bandwidth_bps,node.optical_terminal_bandwidth_bps)
                self.assertEqual(optical.islsim_config['rx']['terminal_cap_gbps'], optical.bandwidth_bps / 1e9)

    def test_generation_native_overrides_change_export_and_input_identity(self):
        from astermind.builder.physical.network_inputs import snapshot_network_input
        from astermind.builder.physical.physical_dataset import input_fingerprint
        original = json.loads((ROOT/'tests/fixtures/network/multi_shell.json').read_text())
        shell = original['constellations']['communication_shells'][0]
        shell.update(radio_profile='starlink_v2_mini_2024', isl_terminals=3)
        original_snapshot = snapshot_network_input(original)
        baseline = materialize_network_input(original)
        shell['hardware_link_profiles'] = {'communication_isl': {
            'islsim_config': {'tx': {'tx_power_w': .05, 'modulation_pool': [[4, 'DP-QPSK']]},
                              'rx': {'terminal_cap_gbps': 80}}}}
        changed_snapshot = snapshot_network_input(original)
        changed = materialize_network_input(original)
        name = 'comm-low__hardware__communication_isl'
        a, b = (NetworkSpec.from_value(value['network']).link_profiles[name] for value in (baseline, changed))
        self.assertEqual(b.islsim_config['tx']['tx_power_w'], .05)
        self.assertEqual(b.islsim_config['rx']['terminal_cap_gbps'], 80)
        self.assertEqual(b.bandwidth_bps, 100e9)
        self.assertNotIn('rx_aperture_m', shell['hardware_link_profiles']['communication_isl']['islsim_config']['rx'])
        self.assertNotEqual(a.islsim_config, b.islsim_config)
        self.assertNotEqual(input_fingerprint(original_snapshot), input_fingerprint(changed_snapshot))

    def test_direction_depends_on_both_ground_service_and_physical_transmitter(self):
        for identity in STARLINK_RADIO_PROFILES:
            state = directed_state(identity)
            directions = {}
            for source,target,frequency in [('u','a',14.25e9),('a','u',12e9),('g','a',83.5e9),('a','g',73.5e9)]:
                edge = next(e for e in state.edges if {e['endpoint_a'],e['endpoint_b']}=={source,target})
                profile = directional_profile(state.nodes,state.spec.link_profiles,edge,source)
                self.assertEqual(profile.frequency_hz,frequency)
                rate = state.directional_attributes(edge,source,target,2.)['bandwidth_bps']
                # Verify the transfer actually consumes the selected direction,
                # rather than merely exporting different JSON metadata.
                _,path = state.transfer(source,target,2.,1e6)
                self.assertAlmostEqual(path[0]['transmission_s'],8e6/rate,places=10)
                self.assertNotEqual(rate,state.directional_attributes(edge,source,target,12.)['bandwidth_bps'])
                directions[source,target]=rate
            self.assertGreater(directions['a','u'],3*directions['u','a'])
            self.assertGreater(directions['g','a'],directions['a','g'])

    def test_published_values_and_estimates_are_separate_and_satellite_pools_are_shared(self):
        v2=STARLINK_RADIO_PROFILES['starlink_v2_mini_2024']
        self.assertNotIn('user_uplink_bps',v2['capacities_bps'])
        self.assertAlmostEqual(v2['simulation_defaults']['radio_capacity_bps']['user_uplink_bps'],160e9/24)
        newer=STARLINK_RADIO_PROFILES['starlink_v3_progress_2025']
        self.assertNotIn('user_uplink_bps',newer['capacities_bps'])
        self.assertEqual(newer['simulation_defaults']['radio_capacity_bps']['user_uplink_bps'],200e9)
        state=directed_state('starlink_v3_design_160_20261021')
        up=hop_pools(state.nodes,'u','a')[0]
        down=hop_pools(state.nodes,'a','u')[0]
        rates=max_min_rates({'u1':(up,), 'u2':(up,), 'd':(down,)},state.pool_capacities)
        self.assertAlmostEqual(rates['u1']+rates['u2'],160e9)
        self.assertAlmostEqual(rates['d'],1e12)

    def test_explicit_budget_override_and_invalid_reference(self):
        value=json.loads((ROOT/'tests/fixtures/network/multi_shell.json').read_text())
        shell=value['constellations']['communication_shells'][0]
        shell.update(radio_profile='starlink_v3_design_160_20261021',hardware_link_profiles={
            'user_uplink':dict(capacity_model='fixed',bandwidth_bps=123456.,setup_time_s=0.)})
        result=materialize_network_input(value)
        spec=NetworkSpec.from_value(result['network'])
        satellite=next(n for n in spec.nodes if n.shell_id=='comm-low')
        self.assertEqual(spec.link_profiles[satellite.sgl_profiles['user_uplink']].bandwidth_bps,123456.)
        with self.assertRaisesRegex(ValueError,'nonrecursive'):
            replace(spec,nodes=(replace(satellite,sgl_profiles={'user_uplink':'unknown'}),))
