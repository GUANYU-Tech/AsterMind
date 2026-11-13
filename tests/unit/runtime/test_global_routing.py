"""Global routing selection must reach real model and system transfers."""
import unittest

from astermind.builder.config.communication import CellCommPolicy, CommPolicy
from astermind.engine.comm.operator import CommunicationOperator
from tests.unit.runtime import test_comm_port as fixtures


def alternatives(policy):
    raw = fixtures.scenario(horizon_ns=1_000_000)
    raw['communication']['routing_policy'] = policy
    physical = raw['physical']
    for device in physical['devices']:
        if device['kind'] == 'terminal':
            device['max_external_links'] = 3
    physical['links'][-1].update(bandwidth_gbps=1, latency_ns=100)
    for owner, latency, bandwidth in (('c', 1, 1), ('d', 10, 100)):
        physical['cells'].append(dict(id=owner, satellites=[owner]))
        physical['satellites'].append(dict(id=owner))
        physical['devices'].append(dict(id=owner+'t', satellite=owner,
            kind='terminal', external_access=True, capacity_gbps=100, max_external_links=2))
        for endpoint in ('at', 'bt'):
            physical['links'].append(dict(id=endpoint+owner, source=endpoint,
                target=owner+'t', bandwidth_gbps=bandwidth, latency_ns=latency))
    return raw


class GlobalRoutingTests(unittest.TestCase):
    def runtime(self, raw):
        fixture = fixtures.CommPortTests()
        self.addCleanup(fixture.doCleanups)
        return fixture.runtime(raw)

    def test_default_and_scope_are_explicit(self):
        self.assertEqual(CommPolicy().routing_policy, 'min_propagation')
        with self.assertRaises(ValueError):
            CommPolicy(routing_policy='unknown')
        with self.assertRaises(ValueError):
            CellCommPolicy(routing_policy='min_hops')

    def test_min_hops_ignores_internal_links_and_gates(self):
        raw = alternatives('min_hops')
        physical = raw['physical']
        # A one-external-hop path has more device edges than either relay path.
        physical['links'] = [link for link in physical['links'] if link['id'] != 'isl']
        previous = 'at'
        for i in range(5):
            device = f'as{i}'
            physical['devices'].append(dict(id=device, satellite='a', kind='switch', capacity_gbps=100))
            physical['links'].append(dict(id=f'local{i}', source=previous,
                target=device, bandwidth_gbps=100, latency_ns=100))
            previous = device
        physical['devices'].append(dict(id='ax', satellite='a', kind='terminal',
            external_access=True, capacity_gbps=100, max_external_links=1))
        raw['cells'][0]['external_terminals'].append('ax')
        physical['links'].append(dict(id='to_ax', source=previous, target='ax', bandwidth_gbps=100))
        physical['links'].append(dict(id='direct', source='ax', target='bt', bandwidth_gbps=1))
        host, network, comm = self.runtime(raw)
        path = comm.route('ag', 'bg', 1000)['path']
        self.assertEqual(path, ['ag', 'at', *[f'as{i}' for i in range(5)], 'ax', 'bt', 'bg'])

    def test_each_policy_selects_its_distinct_physical_path(self):
        expected = {'min_hops': ['at','bt'], 'min_propagation': ['at','ct','bt'],
                    'earliest_arrival': ['at','dt','bt']}
        for policy, middle in expected.items():
            with self.subTest(policy=policy):
                host, network, comm = self.runtime(alternatives(policy))
                route = comm.route('ag', 'bg', 1000)
                self.assertEqual(route['path'], ['ag', *middle, 'bg'])
                self.assertEqual(route['routing_policy'], policy)
                scheduler_route = network.route('ag', 'bg', 1000)
                self.assertEqual(list(scheduler_route['path']), route['path'])
                # A query must not reserve a second transport calendar.
                self.assertFalse(host.net.calendars)

    def test_omitted_policy_routes_by_propagation_for_scheduler_and_comm(self):
        raw = alternatives('earliest_arrival')
        del raw['communication']['routing_policy']
        host, network, comm = self.runtime(raw)
        expected = ['ag', 'at', 'ct', 'bt', 'bg']
        self.assertEqual(comm.route('ag', 'bg', 1000)['path'], expected)
        self.assertEqual(list(network.route('ag', 'bg', 1000)['path']), expected)

    def test_model_and_system_transfers_use_the_same_global_policy(self):
        for policy, hop in (('min_hops','bt'), ('min_propagation','ct'), ('earliest_arrival','dt')):
            with self.subTest(policy=policy):
                host, network, comm = self.runtime(alternatives(policy))
                done = []
                comm.submit_transfer('system', 'ag', 'bg', 1000, lambda: done.append('system'), self.fail)
                op = CommunicationOperator('send_recv', 'send_recv', ('ag','bg'), 1000, 'payload',
                    cell_id='a', task_id='model')
                comm.submit(op, lambda event: done.append('model'), self.fail, task_id='model')
                host.run()
                self.assertCountEqual(done, ['system','model'])
                admissions = [row for row in comm.trace if row['kind'] == 'admit']
                self.assertEqual(len(admissions), 2)
                self.assertTrue(all(row['path'][2] == hop for row in admissions))
                self.assertFalse(comm.tasks or comm.flows or comm.active)

    def test_default_terminal_detour_falls_back_from_whole_hierarchical_model(self):
        from examples.native.compute_comm_matrix import scene
        from astermind.engine.comm.planner import plan_communication
        host, network, comm = self.runtime(scene('DS05'))
        task = CommunicationOperator('all_reduce', 'ring',
            tuple(f'satA-gpu-{i}' for i in range(8)), 4096,
            'per_rank_input', reduction='sum', cell_id='cellA')
        plan = plan_communication(comm.network, comm.policy, task, host.now)
        self.assertTrue(plan.waves)
        paths = [route['path'] for wave in plan.waves for route, size in wave]
        self.assertTrue(any('satA-terminal' in path for path in paths))


if __name__ == '__main__':
    unittest.main()
