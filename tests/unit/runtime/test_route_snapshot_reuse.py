"""Shared graph reads and endpoint restrictions without whole-graph queries."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from astermind.builder.config import SimulationConfig
from astermind.engine.network_host import DeviceRuntimeHost, DeviceCommunicationRuntime
from astermind.engine.policies.routing import register_routing_policy
from astermind.engine.state.network import NetworkSpec, NetworkState, NetworkUnavailable
from tests.unit.runtime.test_comm_port import scenario
from tests.unit.runtime.test_dynamic_network import network


class RouteSnapshotReuseTests(unittest.TestCase):
    def test_short_search_does_not_filter_unrelated_nodes(self):
        edges = (('a', 'b', 0., 20.),) + tuple(
            (f'x{i}', f'y{i}', 0., 20.) for i in range(100))
        for policy in ('min_hops', 'min_propagation'):
            state = network(edges)
            with patch.object(state, 'route_node_allowed', wraps=state.route_node_allowed) as allowed:
                self.assertEqual(state.select_route('a', 'b', 0, policy).nodes, ('a', 'b'))
            self.assertTrue(allowed.call_args_list)
            self.assertTrue(all(call.args[0] in {'a', 'b'} for call in allowed.call_args_list))

    def test_custom_policy_sees_filtered_read_only_mapping(self):
        state = network((('a', 'b', 0., 20.), ('b', 'd', 0., 20.),
                         ('a', 'c', 0., 20.), ('c', 'd', 0., 20.),
                         ('a', 'g', 0., 20.), ('g', 'd', 0., 20.)))
        state.nodes['g'] = replace(state.nodes['g'], role='ground')

        def select(graph, source, target):
            self.assertNotIn('b', graph.adjacency)
            self.assertNotIn('g', graph.adjacency)
            self.assertEqual(tuple(e.target for e in graph.adjacency['a']), ('c',))
            with self.assertRaises(TypeError):
                graph.adjacency['a'] = ()
            return ('a', 'c', 'd')

        with patch.dict('astermind.engine.policies.routing._ROUTING_POLICIES'):
            register_routing_policy('unit-shared-view', select)
            self.assertEqual(state.select_route('a', 'd', 0, 'unit-shared-view',
                                               avoid_nodes={'b'}).nodes, ('a', 'c', 'd'))
        with self.assertRaises(NetworkUnavailable):
            state.select_route('a', 'd', 0, kinds={'not-a-link-kind'})
        for policy in ('min_hops', 'min_propagation'):
            with self.assertRaises(NetworkUnavailable):
                state.select_route('a', 'd', 0, policy, avoid_nodes={'b', 'c'})

    def test_comm_and_route_queries_share_one_link_read_per_snapshot(self):
        config = SimulationConfig.model_validate(scenario(capacity_schedule_ns=((500, .5),)))
        topology = config.topology()
        host = DeviceRuntimeHost(topology, config)
        outer = DeviceCommunicationRuntime(host, topology)
        self.addCleanup(host.close)
        self.addCleanup(outer.close)
        comm = outer.communication_policy
        with patch.object(host.net, 'link_at', wraps=host.net.link_at) as read:
            first = comm._refresh()
            outer.route('ag', 'bg', 1000)
            outer.route('bg', 'ag', 1000)
            self.assertEqual(read.call_count, len(host.net.edges))
            snapshot = host.net.path_route_snapshot(host.now)
            old_states = snapshot.link_states
            host.now = host.now.from_ns(500)
            comm.network_changed()
            self.assertEqual(read.call_count, 2 * len(host.net.edges))
            self.assertIsNot(first, comm.routes)
            self.assertIsNot(old_states, snapshot.link_states)
        self.assertGreater(comm.stats['network_frame_wall_s'], 0)
        self.assertGreater(host.net.path_route_cache_info()['snapshot_wall_s'], 0)

    def test_offline_distance_read_never_expands_full_attribute_record(self):
        original = network(bandwidth=8e9)
        original.trace.trace_id = 'unit-shared-snapshot'
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'physical.sqlite'
            original.save_physical_state(path)
            loaded = NetworkState(original.trace, original.spec, physical_state=path)
            attrs = loaded.physical_attributes
            for edge in loaded.edges:
                expected = attrs.at(edge['link_id'], 0)['geometry']['distance_km']
                with patch.object(attrs, '_cached_at', side_effect=AssertionError('full record expansion')):
                    self.assertEqual(loaded.distance(edge, 0), expected)

    def test_old_offline_defaults_are_equivalent_but_a_changed_budget_is_not(self):
        from astermind.engine.state.physical_dataset import PhysicalAttributes

        original = network(bandwidth=8e9)
        original.trace.trace_id = 'unit-offline-default'
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'physical.sqlite'
            original.save_physical_state(path)
            initialize = PhysicalAttributes.__init__

            def read_old_metadata(reader, path):
                initialize(reader, path)
                for node in reader.metadata['network']['nodes']:
                    node.pop('optical_terminal_bandwidth_bps')

            with patch.object(PhysicalAttributes, '__init__', read_old_metadata):
                restored = NetworkState(original.trace, original.spec, physical_state=path)
                self.assertEqual(restored.snapshot(0), original.snapshot(0))
                changed = original.spec.to_dict()
                profile = next(iter(changed['link_profiles'].values()))
                profile['bandwidth_bps'] *= 2
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    NetworkState(original.trace, NetworkSpec.from_value(changed), physical_state=path)


if __name__ == '__main__':
    unittest.main()
