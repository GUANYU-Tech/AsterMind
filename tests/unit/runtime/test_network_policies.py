import unittest
from dataclasses import replace
from types import SimpleNamespace

from orbitserve.runtime import NetworkNodeSpec, NetworkSpec, LinkProfile, NetworkState, candidate_links
from orbitserve.runtime.engine.association import select_ground_links


class SparseTopologyTests(unittest.TestCase):
    def grid(self, topology='honeycomb', repair='none', wrap=True, zero=False):
        nodes = tuple(NetworkNodeSpec(f'c{p}{s}', 'communication', 0 if zero and s == 1 else 4,
                                      plane=p, slot=s)
                      for p in range(4) for s in range(6) if zero or s not in (1, 2))
        return NetworkSpec(nodes, {'communication_isl': LinkProfile(1e9, 0)},
                           communication_topology=topology, communication_grid_shape=(4, 6),
                           topology_repair=repair, wrap_slots=wrap)

    def pairs(self, spec):
        return {tuple(sorted((e['endpoint_a'], e['endpoint_b']))): e for e in candidate_links(spec)}

    def test_baseline_leaves_runs_empty_and_method1_bridges_only_within_orbit(self):
        for topology in ('grid', 'honeycomb'):
            baseline = self.pairs(self.grid(topology))
            repaired = self.pairs(self.grid(topology, 'ronchetti_intra_orbit'))
            extra = set(repaired) - set(baseline)
            self.assertEqual(extra, {(f'c{p}0', f'c{p}3') for p in range(4)})
            self.assertTrue(all(repaired[p]['topology_patch'] == 'ronchetti_intra_orbit' for p in extra))

    def test_zero_terminal_satellite_counts_as_hole(self):
        repaired = self.pairs(self.grid(repair='ronchetti_intra_orbit', zero=True))
        self.assertIn(('c00', 'c02'), repaired)
        self.assertEqual(repaired['c00', 'c02']['topology_patch'], 'ronchetti_intra_orbit')

    def test_missing_entire_plane_and_unplaced_satellites_do_not_shift_indices(self):
        spec = self.grid()
        nodes = tuple(n for n in spec.nodes if n.plane != 1) + (NetworkNodeSpec('misaligned', 'communication', 4),)
        edges = candidate_links(replace(spec, nodes=nodes))
        self.assertFalse(any('misaligned' in (e['endpoint_a'], e['endpoint_b']) for e in edges))
        self.assertFalse(any({e['endpoint_a'][1], e['endpoint_b'][1]} == {'0', '2'} for e in edges))

    def test_open_chain_never_patches_across_seam(self):
        spec = self.grid(repair='ronchetti_intra_orbit', wrap=False)
        spec = replace(spec, nodes=tuple(n for n in spec.nodes if n.slot != 0))
        self.assertFalse(any(e.get('topology_patch') for e in candidate_links(spec)))

    def test_patch_is_not_established_through_100km_atmosphere(self):
        spec = self.grid(repair='ronchetti_intra_orbit')
        edge = next(e for e in candidate_links(spec) if e.get('topology_patch'))
        nodes = tuple(n for n in spec.nodes if n.node_id in (edge['endpoint_a'], edge['endpoint_b']))
        spec = replace(spec, nodes=nodes)
        trace = SimpleNamespace(time={'start_s': 0., 'end_s': 10.},
            satellite_catalog=[{'satellite_id': n.node_id} for n in nodes], sgl_windows=[],
            ephemeris_samples=[dict(satellite_id=n.node_id, time_s=t, position_itrs_km=pos)
                for t in (0.,10.) for n,pos in zip(nodes, ((7000.,0.,0.),(-7000.,0.,0.)))],
            isl_windows=[dict(endpoint_a=edge['endpoint_a'],endpoint_b=edge['endpoint_b'],start_s=0.,end_s=10.)])
        state = NetworkState(trace, spec)
        self.assertTrue(all(e['state'] == 'DOWN' for e in state.snapshot(0)['links']))
        self.assertIn('patch_atmosphere_blocked', {e['reason'] for e in state.snapshot(0)['links']})


class AssociationTests(unittest.TestCase):
    def test_both_handover_modes_apply_to_ground_and_cross_layer_contacts(self):
        for kind in ('sgl','cross_layer'):
            for handover in ('retain_valid','quality_first'):
                for matching in ('greedy','deferred_acceptance'):
                    endpoint='g' if kind=='sgl' else 'm'
                    nodes=(NetworkNodeSpec('a','communication',4,1),NetworkNodeSpec('b','communication',4,1),
                           NetworkNodeSpec(endpoint,'ground',sgl_terminals=1) if kind=='sgl'
                           else NetworkNodeSpec(endpoint,'compute',1))
                    spec=NetworkSpec(nodes,{kind:LinkProfile(100,2)},communication_topology='explicit',
                                     ground_matching=matching,handover_policy=handover)
                    satellites='ab' if kind=='sgl' else 'abm'
                    trace=SimpleNamespace(time={'start_s':0.,'end_s':20.},
                        satellite_catalog=[{'satellite_id':n} for n in satellites],
                        ephemeris_samples=[dict(satellite_id=n,time_s=t,position_itrs_km=(d,0,0))
                            for t in (0.,10.,20.) for n,d in
                            ((('a',7000 if t==0 else 7500),('b',7300 if t==0 else 7000))
                             if kind=='sgl' else (('a',7010 if t==0 else 7100),('b',7040 if t==0 else 7010),('m',7000)))],
                        sgl_windows=[dict(ingress_id='g',satellite_id=n,start_s=0.,end_s=20.) for n in 'ab'] if kind=='sgl' else [],
                        isl_windows=[dict(endpoint_a=n,endpoint_b='m',start_s=0.,end_s=20.) for n in 'ab'] if kind=='cross_layer' else [])
                    state=NetworkState(trace,spec,[dict(ingress_id='g',lat_deg=0,lon_deg=0)] if kind=='sgl' else [])
                    first=[e for e in state.snapshot(5)['links'] if e['state']=='UP']
                    self.assertEqual(first[0]['endpoint_a'],'a')
                    at_change=[e for e in state.snapshot(10)['links'] if e['state']!='DOWN']
                    self.assertEqual(len(at_change),1)
                    self.assertEqual(at_change[0]['endpoint_a'],'a' if handover=='retain_valid' else 'b')
                    self.assertEqual(at_change[0]['state'],'UP' if handover=='retain_valid' else 'SETTING_UP')
                    self.assertEqual(at_change[0]['setup_start_s'],0 if handover=='retain_valid' else 10)
                    self.assertEqual(len([e for e in state.snapshot(12)['links'] if e['state']=='UP']),1)
                    self.assertEqual(NetworkSpec.from_value(spec.to_dict()),spec)

    def test_unknown_handover_and_cross_layer_ranking_are_rejected(self):
        spec=self.state().spec
        with self.assertRaisesRegex(ValueError,'handover_policy'):
            replace(spec,handover_policy='unknown')
        with self.assertRaisesRegex(ValueError,'cross_layer_access_policy'):
            replace(spec,cross_layer_access_policy='highest_elevation')

    def test_retained_ground_contact_is_not_replaced_by_a_better_ranked_peer(self):
        for matching in ('greedy','deferred_acceptance'):
            for outage in (False,True):
                nodes=(NetworkNodeSpec('a','communication',sgl_terminals=1),
                       NetworkNodeSpec('b','communication',sgl_terminals=1),
                       NetworkNodeSpec('g','ground',sgl_terminals=1))
                spec=NetworkSpec(nodes,{'sgl':LinkProfile(100,2)},communication_topology='explicit',
                                 ground_matching=matching,ground_access_policy='nearest_distance')
                trace=SimpleNamespace(time={'start_s':0.,'end_s':20.},
                    satellite_catalog=[{'satellite_id':n} for n in 'ab'],isl_windows=[],
                    sgl_windows=[dict(ingress_id='g',satellite_id='a',start_s=0.,end_s=10. if outage else 20.),
                                 dict(ingress_id='g',satellite_id='b',start_s=0.,end_s=20.)],
                    ephemeris_samples=[dict(satellite_id=n,time_s=t,position_itrs_km=(distance,0,0))
                        for t in (0.,10.,20.) for n,distance in
                        (('a',7000 if t==0 else 7500),('b',7300 if t==0 else 7000))])
                state=NetworkState(trace,spec,[dict(ingress_id='g',lat_deg=0,lon_deg=0)])
                before=[e for e in state.snapshot(5)['links'] if e['state']=='UP']
                self.assertEqual([e['link_id'] for e in before],['sgl:a:g'])
                after=state.snapshot(10)['links']
                if outage:
                    replacement=next(e for e in after if e['link_id']=='sgl:b:g')
                    self.assertEqual(replacement['state'],'SETTING_UP')
                    self.assertEqual(replacement['ready_s'],12)
                    self.assertEqual([e['link_id'] for e in state.snapshot(12)['links'] if e['state']=='UP'],['sgl:b:g'])
                else:
                    retained=[e for e in after if e['state']=='UP']
                    self.assertEqual([e['link_id'] for e in retained],['sgl:a:g'])
                    self.assertEqual(retained[0]['setup_start_s'],0)

    def test_retained_contacts_consume_total_and_service_quotas(self):
        for matching in ('greedy','deferred_acceptance'):
            state=self.state(matching)
            edges=self.edges()
            retained=[next(e for e in edges if e['link_id']=='u1-s1')]
            candidates=[e for e in edges if e not in retained]
            result=select_ground_links(state,candidates,0,retained=retained)
            self.assertNotIn('u2-s1',result)
            self.assertNotIn('u1-s2',result)
            self.assertIn('g-s1',result)

    def state(self, matching='deferred_acceptance', user_limit=1, gateway_limit=1):
        nodes = [NetworkNodeSpec('s1', 'communication', sgl_terminals=2,
                                max_user_links=user_limit, max_gateway_links=gateway_limit),
                 NetworkNodeSpec('s2', 'communication', sgl_terminals=2,
                                max_user_links=user_limit, max_gateway_links=gateway_limit)]
        nodes += [NetworkNodeSpec(n, 'ground', sgl_terminals=capacity, ground_kind=kind)
                  for n, capacity, kind in (('u1',1,'user'), ('u2',1,'user'), ('g',2,'gateway'))]
        spec = NetworkSpec(tuple(nodes), {'sgl': LinkProfile(1e9,0)}, communication_topology='explicit',
                           ground_matching=matching, ground_access_policy='longest_available',
                           satellite_access_policy='nearest_distance')
        return SimpleNamespace(nodes={n.node_id:n for n in nodes}, spec=spec)

    def edges(self):
        return [dict(endpoint_a=g,endpoint_b=s,link_id=g+'-'+s,distance_km=d,available_until_s=end)
                for g,s,d,end in (('u1','s1',4,20),('u1','s2',5,10),('u2','s1',1,20),('u2','s2',9,10),
                                  ('g','s1',3,20),('g','s2',2,20))]

    def test_two_sided_rejection_retries_next_choice_and_gateway_can_use_two_satellites(self):
        state = self.state()
        result = select_ground_links(state, self.edges(), 0)
        self.assertEqual(result, {'u2-s1','u1-s2','g-s1','g-s2'})
        self.assertEqual(result, select_ground_links(state, list(reversed(self.edges())), 0))

    def test_zero_user_quota_does_not_consume_gateway_quota_in_either_matching(self):
        for matching in ('greedy', 'deferred_acceptance'):
            result = select_ground_links(self.state(matching, user_limit=0), self.edges(), 0)
            self.assertEqual(result, {'g-s1','g-s2'})

    def test_shared_total_limit_is_enforced_with_service_quotas(self):
        state = self.state(user_limit=8, gateway_limit=8)
        state.nodes = {k:replace(n, sgl_terminals=1) if n.role!='ground' else n for k,n in state.nodes.items()}
        result = select_ground_links(state, self.edges(), 0)
        self.assertEqual(len(result), 2)
        self.assertEqual(len({s.split('-')[-1] for s in result}), 2)

    def test_explicit_pairs_cannot_bypass_single_cell_egress(self):
        nodes = (NetworkNodeSpec('m0','compute',4,cell_id='cell',cell_ingress=True),
                 NetworkNodeSpec('m1','compute',4,cell_id='cell',compute_target=False),
                 NetworkNodeSpec('r','communication',4))
        spec = NetworkSpec(nodes, {k:LinkProfile(1e9,0) for k in ('cell_isl','cross_layer','communication_isl')},
                           communication_topology='explicit',cell_egress_policies={'cell':'single'})
        self.assertEqual(sum(e['kind']=='cross_layer' for e in candidate_links(spec)), 1)
        with self.assertRaisesRegex(ValueError,'bypasses'):
            candidate_links(replace(spec,explicit_pairs=(('m1','r'),)))
        free = replace(spec,cell_egress_policies={'cell':'free'})
        self.assertEqual(sum(e['kind']=='cross_layer' for e in candidate_links(free)), 2)
        self.assertTrue(all(n.cell_ingress for n in free.nodes if n.cell_id))
