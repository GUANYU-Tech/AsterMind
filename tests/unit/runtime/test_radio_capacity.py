import unittest
from dataclasses import replace
from types import SimpleNamespace

from astermind.engine import NetworkNodeSpec, NetworkSpec, LinkProfile, NetworkState, NetworkUnavailable
from astermind.engine.setup.radio_capacity import radio_parameters, hop_pools, max_min_rates
from tests.unit.runtime.test_dynamic_network import network


def access_network(**kwargs):
    trace = SimpleNamespace(
        time={'start_s': 0., 'end_s': 10.},
        satellite_catalog=({'satellite_id': 'a'}, {'satellite_id': 'b'}),
        ephemeris_samples=tuple(dict(satellite_id=n, time_s=float(t), position_itrs_km=(x, 0., 0.))
                                for t in range(11) for n, x in (('a', 7000.), ('b', 7100.))),
        isl_windows=(), sgl_windows=tuple(dict(ingress_id='g', satellite_id=n, start_s=0., end_s=10.) for n in ('a', 'b')))
    spec = NetworkSpec((NetworkNodeSpec('a', 'communication', sgl_terminals=1),
                        NetworkNodeSpec('b', 'communication', sgl_terminals=1),
                        NetworkNodeSpec('g', 'ground', sgl_terminals=1)),
                       {'sgl': LinkProfile(8000, .4)}, communication_topology='explicit', fault_timing='continuous', **kwargs)
    return NetworkState(trace, spec, ({'ingress_id': 'g', 'lat_deg': 0., 'lon_deg': 0.},))


class FaultAllocationTests(unittest.TestCase):
    def test_longest_available_reallocation_never_rewrites_prefault_choices(self):
        base = access_network(ground_access_policy='longest_available')
        fault = dict(target_type='link', target_id='sgl:a:g', start_s=1.25, end_s=5.4)
        state = NetworkState(base.trace, replace(base.spec, faults=(fault,)),
                             ({'ingress_id': 'g', 'lat_deg': 0., 'lon_deg': 0.},))
        state.observe_fault(0, fault, 1.25)
        self.assertEqual(state.link_at('sgl:a:g', 1.)['state'], 'UP')
        self.assertEqual(state.link_at('sgl:b:g', 1.)['state'], 'DOWN')
        state.observe_fault(0, fault, 5.4, restored=True)
        self.assertEqual(state.link_at('sgl:a:g', 1.)['state'], 'UP')
        audited = NetworkState(base.trace, replace(base.spec, faults=(fault,)),
                             ({'ingress_id': 'g', 'lat_deg': 0., 'lon_deg': 0.},)).audit_timeline()
        self.assertEqual(audited.link_at('sgl:a:g', 1.)['state'], 'UP')
        self.assertEqual(audited.link_at('sgl:b:g', 3.)['state'], 'UP')

    def test_single_terminal_is_released_and_backup_reacquires_when_observed(self):
        state = access_network(handover_policy='quality_first')
        fault = dict(target_type='link', target_id='sgl:a:g', start_s=1.25, end_s=5.4)
        state.observe_fault(0, fault, 1.25)
        self.assertEqual(state.link_at('sgl:a:g', 1.25)['state'], 'DOWN')
        setup = state.link_at('sgl:b:g', 1.25)
        self.assertEqual(setup['state'], 'SETTING_UP')
        self.assertEqual(setup['setup_start_s'], 1.25)
        self.assertEqual(setup['ready_s'], 1.65)
        arrival, path = state.transfer('g', 'b', 2., 1000)
        self.assertAlmostEqual(path[0]['start_s'], 2.)
        self.assertGreater(arrival, 3.)
        # Repair has not been observed; its configured end must not help forecasts.
        with self.assertRaises(NetworkUnavailable):
            state.transfer('g', 'a', 5.5, 10)
        state.observe_fault(0, fault, 5.4, restored=True)
        renewed = state.link_at('sgl:a:g', 6.)
        self.assertEqual(renewed['setup_start_s'], 6.)
        self.assertEqual(renewed['ready_s'], 6.4)

    def test_repair_keeps_backup_until_policy_allows_a_new_selection(self):
        for policy in ('retain_valid','quality_first'):
            state=access_network(handover_policy=policy)
            fault=dict(target_type='link',target_id='sgl:a:g',start_s=1.25,end_s=5.4)
            state.observe_fault(0,fault,1.25)
            state.observe_fault(0,fault,5.4,restored=True)
            self.assertEqual(state.link_at('sgl:b:g',5.5)['state'],'UP')
            selected=state.link_at('sgl:b:g' if policy=='retain_valid' else 'sgl:a:g',6.)
            self.assertEqual(selected['setup_start_s'],1.25 if policy=='retain_valid' else 6.)

    def test_overlap_node_failure_and_prediction_fork_do_not_release_live_state(self):
        state = access_network()
        first = dict(target_type='link', target_id='sgl:a:g', start_s=1.25, end_s=5.4)
        second = dict(target_type='node', target_id='g', start_s=2.25, end_s=4.25)
        state.observe_fault(0, first, 1.25)
        prediction = state.fork()
        prediction.observe_fault(1, second, 2.25)
        self.assertEqual(state.link_at('sgl:b:g', 3.)['state'], 'UP')
        self.assertEqual(prediction.link_at('sgl:b:g', 3.)['state'], 'DOWN')
        prediction.observe_fault(1, second, 4.25, restored=True)
        self.assertEqual(prediction.link_at('sgl:b:g', 5.)['setup_start_s'], 4.25)


class RadioCapacityTests(unittest.TestCase):
    def test_generation_limits_count_optical_ports_not_rf_beams(self):
        for version, ports, antennae in (('starlink_v2_mini_2024', 3, 3), ('starlink_v3_design_160_20261021', 6, 4)):
            node = NetworkNodeSpec('s', 'communication', isl_terminals=ports, radio_profile=version)
            public = radio_parameters(node)['terminals']
            self.assertEqual(public['optical_count'], ports)
            self.assertEqual(public['rf_feeder_antennas'], antennae)
            with self.assertRaisesRegex(ValueError, 'optical terminal count'):
                replace(node, isl_terminals=ports + 1)
            base = network(edges=tuple(('s', n, 0., 20.) for n in 'abcdefg'), terminals=7, bandwidth=1e12)
            nodes = tuple(node if n.node_id == 's' else replace(n, role='communication') for n in base.spec.nodes)
            state = NetworkState(base.trace, replace(base.spec, nodes=nodes))
            up = [e for e in state.snapshot(0)['links'] if e['state'] == 'UP']
            self.assertEqual(len(up), ports)

    def test_endpoint_optical_rate_limit_applies_to_links_in_both_directions(self):
        base = network(bandwidth=1e12)
        nodes = (NetworkNodeSpec('a', 'communication', 3, radio_profile='starlink_v2_mini_2024',
                                optical_terminal_bandwidth_bps=100e9, terminal_assumptions=('Advertised duplex-total scenario.',)),
                 NetworkNodeSpec('b', 'communication', 6, radio_profile='starlink_v3_design_160_20261021',
                                optical_terminal_bandwidth_bps=200e9, terminal_assumptions=('Advertised duplex-total scenario.',)))
        state = NetworkState(base.trace, replace(base.spec, nodes=nodes))
        for a, b in (('a', 'b'), ('b', 'a')):
            _, path = state.transfer(a, b, 0., 1e6)
            self.assertAlmostEqual(path[0]['transmission_s'], 8e6 / 100e9)
        self.assertEqual(state.snapshot(0)['links'][0]['bandwidth_bps'], 100e9)
        self.assertTrue(all(d['bandwidth_bps'] == 100e9 for d in state.snapshot(0)['links'][0]['directions']))

    def test_shared_forecast_can_choose_a_less_congested_route(self):
        base = network(edges=(('a', 'd', 0., 20.), ('a', 'b', 0., 20.), ('b', 'd', 0., 20.)), bandwidth=8000)
        for mode in ('fluid', 'simgrid', 'ns3'):
            state = NetworkState(base.trace, replace(base.spec, transport_model=mode))
            state.shared_load['communication_isl:a:d', 'a', 'd'] = 3
            _, path = state.transfer('a', 'd', 0., 1000)
            self.assertEqual([leg['target_node_id'] for leg in path], ['b', 'd'])

    def test_forecasts_include_contenders_for_every_shared_backend(self):
        for mode in ('queue', 'fluid', 'simgrid', 'ns3'):
            base = network(bandwidth=8000)
            state = NetworkState(base.trace, replace(base.spec, transport_model=mode))
            edge = state.edges[0]
            state.shared_load[edge['link_id'], 'a', 'b'] = 1
            _, path = state.transfer('a', 'b', 0., 1000)
            expected = 1 if mode == 'queue' else 2.004 if mode == 'ns3' else 2
            self.assertAlmostEqual(path[0]['transmission_s'], expected)

    def test_versioned_profiles_do_not_confuse_geometry_link_and_satellite_capacity(self):
        node = NetworkNodeSpec('s', 'communication', radio_profile='starlink_v3_design_160_20261021')
        nodes = {'s': node, 'u': NetworkNodeSpec('u', 'ground', ground_kind='user'),
                 'g': NetworkNodeSpec('g', 'ground')}
        caps = radio_parameters(node)['capacities_bps']
        self.assertEqual(caps['user_uplink_bps'], 160e9)
        self.assertEqual(caps['user_downlink_bps'], 1e12)
        self.assertEqual(hop_pools(nodes, 'u', 's'), ('radio:s:user_uplink_bps',))
        self.assertEqual(hop_pools(nodes, 's', 'u'), ('radio:s:user_downlink_bps',))
        self.assertEqual(hop_pools(nodes, 'g', 's'), ('radio:s:feeder_combined_bps',))
        v2 = replace(node, radio_profile='starlink_v2_mini_2024')
        self.assertEqual(radio_parameters(v2)['capacities_bps'], {'user_downlink_bps': 96e9})
        with self.assertRaisesRegex(ValueError, 'unpublished'):
            hop_pools({**nodes, 's': v2}, 'u', 's')
        with self.assertRaisesRegex(ValueError, 'capacity_assumptions'):
            replace(v2, radio_capacity_bps={'user_uplink_bps': 7e9})
        with self.assertRaisesRegex(ValueError, 'versioned'):
            replace(node, radio_profile='V3')

    def test_progressive_filling_redistributes_unused_share(self):
        rates = max_min_rates({'a': ('slow', 'pool'), 'b': ('fast', 'pool')},
                              {'slow': 2., 'fast': 100., 'pool': 10.})
        self.assertEqual(rates, {'a': 2., 'b': 8.})

    def test_directional_link_profiles_and_aggregate_reservation_calendars(self):
        state = access_network()
        sat = replace(state.spec.nodes[0], sgl_terminals=2,
                      radio_capacity_bps={'feeder_uplink_bps': 4000, 'feeder_downlink_bps': 16000},
                      capacity_assumptions=('Small synthetic acceptance capacities.',))
        ground = replace(state.spec.nodes[-1], node_id='g2')
        trace = SimpleNamespace(**vars(state.trace))
        trace.sgl_windows = (*trace.sgl_windows, dict(ingress_id='g2', satellite_id='a', start_s=0., end_s=10.))
        spec = replace(state.spec, nodes=(sat, state.spec.nodes[1], state.spec.nodes[-1], ground),
                       link_profiles={'sgl': LinkProfile(32000, .4, uplink_profile='ul', downlink_profile='dl'),
                                      'ul': LinkProfile(8000, 0), 'dl': LinkProfile(32000, 0)})
        state = NetworkState(trace, spec, tuple(dict(ingress_id=n, lat_deg=0., lon_deg=0.) for n in ('g', 'g2')))
        _, first = state.transfer('g', 'a', .4, 1000, commit=True)
        _, second = state.transfer('g2', 'a', .4, 1000)
        _, reverse = state.transfer('a', 'g', .4, 1000)
        self.assertAlmostEqual(first[0]['transmission_s'], 2.)
        self.assertGreaterEqual(second[0]['start_s'], first[0]['finish_s'])
        self.assertAlmostEqual(reverse[0]['transmission_s'], .5)
        self.assertAlmostEqual(reverse[0]['start_s'], .4)
        changed = NetworkState(trace, replace(spec, link_profiles={**spec.link_profiles,
            'ul': LinkProfile(8000, 0, capacity_schedule=((5.25, .25),))}),
            tuple(dict(ingress_id=n, lat_deg=0., lon_deg=0.) for n in ('g', 'g2')))
        self.assertIn(5.25, changed.frame_times)
        _, dynamic = changed.transfer('g', 'a', 4.4, 1000)
        self.assertAlmostEqual(dynamic[0]['finish_s'], 7.55)

