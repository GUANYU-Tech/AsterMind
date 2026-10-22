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
