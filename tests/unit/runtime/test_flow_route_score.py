"""Analytic oracles for path bottleneck routing, separate from execution."""
import unittest
from dataclasses import replace
from tests.unit.runtime.test_dynamic_network import network
from astermind.engine.setup.network_spec import LinkProfile


class FlowRouteScoreTests(unittest.TestCase):
    def test_bottleneck_prefers_two_hops_over_serially_faster_direct_link(self):
        net = network((('a','b',0.,20.), ('b','d',0.,20.), ('a','d',0.,20.)), bandwidth=8000.)
        net.spec = replace(net.spec, link_profiles={**net.spec.link_profiles,
            'slow': LinkProfile(6000., 0.)})
        for edge in net.edges:
            if {edge['endpoint_a'],edge['endpoint_b']} == {'a','d'}:
                edge['profile_id']='slow'
        arrival, path = net.transfer('a','d',0.,1000.,policy='earliest_arrival')
        self.assertEqual([leg['target_node_id'] for leg in path], ['b','d'])
        self.assertAlmostEqual(arrival, 1.)
        # Its retained per-hop feasibility forecast is 2 s, not the score.
        self.assertAlmostEqual(path[-1]['arrival_s'], 2.)
        cached, _ = net.transfer('a','d',1.,1000.,policy='earliest_arrival')
        self.assertAlmostEqual(cached, 2.)
        self.assertEqual(net.route_cache_info()['hits'], 1)

    def test_propagation_is_summed_but_serialization_is_charged_once(self):
        positions={name:((7000.,i*299792.458,0.),)*2 for i,name in enumerate(('a','b','c'))}
        net=network((('a','b',0.,20.),('b','c',0.,20.)), bandwidth=8000.,positions=positions)
        arrival,path=net.transfer('a','c',0.,1000.,policy='earliest_arrival')
        self.assertAlmostEqual(arrival,3.)
        self.assertAlmostEqual(path[-1]['arrival_s'],4.)

    def test_shared_pool_counts_existing_flows_and_new_flow_once(self):
        net=network((('a','b',0.,20.),('b','c',0.,20.)),bandwidth=8000.)
        net.pool_capacities['exit']=4000.
        net.shared_pool_load['exit']=1
        net.pools_for_hop=lambda a,b: ('exit',)
        arrival,_=net.transfer('a','c',0.,1000.,policy='earliest_arrival')
        self.assertAlmostEqual(arrival,4.)
        self.assertFalse(net.calendars)

    def test_prefix_score_alone_does_not_prune_the_best_complete_path(self):
        # The high-rate detour wins at m, but a common slow final edge makes
        # the lower-propagation prefix best for the complete path.
        xy={'a':(0,0),'b':(.5,0),'c':(.5,1),'m':(1,0),'z':(1.1,0)}
        positions={n:((7000.,x*29979.2458,y*29979.2458),)*2 for n,(x,y) in xy.items()}
        net=network(tuple((a,b,0.,20.) for a,b in
            (('a','b'),('b','m'),('a','c'),('c','m'),('m','z'))),
            bandwidth=8000.,positions=positions)
        net.spec=replace(net.spec,link_profiles={**net.spec.link_profiles,
            'low':LinkProfile(800.,0.),'tail':LinkProfile(400.,0.)})
        for edge in net.edges:
            pair={edge['endpoint_a'],edge['endpoint_b']}
            if 'b' in pair: edge['profile_id']='low'
            if 'z' in pair: edge['profile_id']='tail'
        # A smaller payload keeps the old feasibility check inside the horizon.
        arrival,path=net.transfer('a','z',0.,100.,policy='earliest_arrival')
        self.assertEqual([leg['target_node_id'] for leg in path],['b','m','z'])
        self.assertAlmostEqual(arrival,2.11)
