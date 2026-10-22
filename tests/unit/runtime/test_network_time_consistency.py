"""Independent per-sample oracles; never use the production budget as expected data."""
import math
import unittest
from dataclasses import replace

from orbitserve.runtime import LinkProfile, NetworkState
from tests.integration.fixed_design.test_network_acceptance import acceptance_spec

C = 299792458.
K = 1.380649e-23
H = 6.62607015e-34


def expected_rate(profile, distance_km, time_s):
    if profile.capacity_model == 'free_space_shannon':
        gains = 10 ** ((profile.transmit_gain_db + profile.receive_gain_db) / 10)
        pr = profile.transmit_power_w * gains * (C / profile.frequency_hz / (4 * math.pi * distance_km * 1000)) ** 2
        pr *= 10 ** (-(profile.system_loss_db + profile.atmospheric_loss_db + profile.pointing_loss_db) / 10)
        snr = pr / (K * profile.noise_temperature_k * profile.channel_bandwidth_hz)
        raw = profile.implementation_efficiency * profile.channel_bandwidth_hz * math.log2(1 + snr)
    elif profile.capacity_model == 'optical_photon_budget':
        waist = profile.transmit_aperture_m / 2
        theta = profile.beam_divergence_rad or profile.wavelength_m / (math.pi * waist)
        radius2 = waist ** 2 + (theta * distance_km * 1000) ** 2
        fraction = -math.expm1(-2 * (profile.receive_aperture_m / 2) ** 2 / radius2)
        pr = profile.transmit_power_w * profile.transmit_aperture_efficiency * profile.receive_aperture_efficiency * fraction
        pr *= 10 ** (-(profile.system_loss_db + profile.pointing_loss_db + profile.atmospheric_loss_db) / 10)
        raw = pr * profile.wavelength_m / (H * C * profile.required_photons_per_bit) * profile.implementation_efficiency
    else:
        raw = profile.bandwidth_bps
    multiplier = 1.
    for t, value in profile.capacity_schedule:
        if t <= time_s:
            multiplier = value
    return min(raw, profile.bandwidth_bps) * multiplier


def temporal_state():
    spec = acceptance_spec()
    rf = LinkProfile(1e15, 0, capacity_model='free_space_shannon', frequency_hz=20e9,
        channel_bandwidth_hz=20e6, transmit_power_w=50., transmit_gain_db=30., receive_gain_db=30.,
        implementation_efficiency=1e-4, system_loss_db=2., capacity_schedule=((2.25, .5), (5.75, 1.)))
    optical = LinkProfile(1e15, 0, capacity_model='optical_photon_budget', wavelength_m=1550e-9,
        transmit_aperture_m=.05, receive_aperture_m=.05, transmit_power_w=.2, required_photons_per_bit=20.,
        pointing_loss_db=1., capacity_schedule=((2.25, .75),))
    profiles = {'sgl': replace(rf, downlink_profile='dl'), 'dl': replace(rf, transmit_power_w=5.),
                'communication_isl': optical, 'cross_layer': optical}
    nodes = tuple(replace(n, radio_capacity_bps={}, capacity_assumptions=(), sgl_terminals=4) for n in spec.network.nodes)
    state = NetworkState(spec.environment_state, replace(spec.network, nodes=nodes, faults=(), link_profiles=profiles), spec.gateways)
    return state


def assert_physical_snapshot(case, state, time_s):
    snapshot = state.snapshot(time_s)
    sample = math.floor(time_s)
    positions = {r['satellite_id']: r['position_itrs_km'] for r in state.trace.ephemeris_samples if r['time_s'] == sample}
    positions.update(state.ground_positions)
    for node in snapshot['nodes']:
        case.assertEqual(tuple(node['position_itrs_km']), tuple(positions[node['node_id']]))
    for link in snapshot['links']:
        a, b = positions[link['endpoint_a']], positions[link['endpoint_b']]
        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        case.assertAlmostEqual(link['distance_km'], distance, places=9)
        case.assertAlmostEqual(link['propagation_delay_s'], distance * 1000 / C, places=12)
        case.assertEqual(link['attributes_time_s'], sample)
        for direction in link['directions']:
            profile = state.spec.link_profiles[link.get('profile_id', link['kind'])]
            name = profile.uplink_profile if state.nodes[direction['source_node_id']].role == 'ground' else profile.downlink_profile
            profile = state.spec.link_profiles[name] if link['kind'] == 'sgl' and name else profile
            case.assertAlmostEqual(direction['bandwidth_bps'] / expected_rate(profile, distance, time_s), 1., places=10)
    return snapshot


class NetworkTimeConsistencyTests(unittest.TestCase):
    def test_every_sgp4_sample_and_mid_sample_uses_time_matched_geometry_and_budget(self):
        state = temporal_state()
        rates = []
        for time_s in [*(float(t) for t in range(120)), .75, 1.75, 2.25, 5.75, 79.9]:
            snapshot = assert_physical_snapshot(self, state, time_s)
            rates.append(snapshot['links'][0]['directions'][0]['bandwidth_bps'])
        self.assertGreater(max(rates) - min(rates), 1.)  # Fixture must expose stale-rate bugs.

    def test_transfer_integrates_each_sample_and_exact_capacity_boundary(self):
        state = temporal_state()
        _, path = state.transfer('ground-00', 'sat-a', 1.25, 16384, kinds={'sgl'})
        leg = path[0]
        self.assertGreater(leg['finish_s'], 5.75)
        self.assertTrue({2.25, 5.75} <= {s['start_s'] for s in leg['segments']})
        total = 0.
        profile = state.spec.link_profiles['sgl']
        for segment in leg['segments']:
            sample = math.floor(segment['start_s'])
            pos = next(r['position_itrs_km'] for r in state.trace.ephemeris_samples if r['satellite_id'] == 'sat-a' and r['time_s'] == sample)
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, state.ground_positions['ground-00'])))
            expected = expected_rate(profile, distance, segment['start_s'])
            self.assertAlmostEqual(segment['bandwidth_bps'] / expected, 1., places=10)
            self.assertAlmostEqual(segment['distance_km'], distance, places=9)
            self.assertEqual(segment['attributes_time_s'], sample)
            total += expected * (segment['finish_s'] - segment['start_s']) / 8
        self.assertAlmostEqual(total, 16384., places=7)
        self.assertAlmostEqual(leg['distance_km'] / leg['propagation_delay_s'], C / 1000, places=7)
        self.assertNotEqual(leg['distance_km'], leg['start_distance_km'])

    def test_fixed_capacity_can_stay_constant_while_geometry_changes(self):
        state = temporal_state()
        profiles = {k: LinkProfile(1234., 0.) for k in state.spec.link_profiles}
        fixed = NetworkState(state.trace, replace(state.spec, link_profiles=profiles), acceptance_spec().gateways)
        for time_s in (0., 12., 73.):
            self.assertTrue(all(d['bandwidth_bps'] == 1234. for e in fixed.snapshot(time_s)['links'] for d in e['directions']))
