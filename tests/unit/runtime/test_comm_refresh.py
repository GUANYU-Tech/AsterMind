"""Repeated notifications must not reprice an unchanged physical snapshot."""
import unittest
from unittest.mock import patch

from astermind.engine.clock import Time
from astermind.engine.comm.event_runtime import RuntimeCommunication
from astermind.engine.comm.frame_surrogate import CommPolicy, NoRouteError
from astermind.engine.device_controls import apply_device_command
from astermind.engine.timeline import GlobalTimeline
from tests.comm_fixture import Frames
from tests.network_case import device_runtime
from tests.unit.runtime import test_comm_port as fixtures
from tests.unit.runtime.test_dynamic_network import network as physical_network


class CommunicationRefreshTests(unittest.TestCase):
    def runtime(self, **kwargs):
        fixture = fixtures.CommPortTests()
        self.addCleanup(fixture.doCleanups)
        return fixture.runtime(**kwargs)

    def fast_runtime(self, **options):
        raw = fixtures.scenario(**options)
        for cell in raw['cells']:
            cell.update(hardware_id='nvidia_h200', hardware='configs/hardware/nvidia_h200.yaml')
        return self.runtime(raw={**raw, 'compute_mode': 'fast_mode'})

    def test_fast_builds_the_first_idle_frame_only_when_a_request_needs_it(self):
        host, _, comm = self.fast_runtime(capacity_schedule_ns=((500, 0.5),))
        completed = []
        when = Time.from_ns(750)
        host.timeline.schedule(when, 1, 'submit', lambda: comm.submit_transfer(
            'request', 'ag', 'bg', 1000, lambda: completed.append(host.now), self.fail))
        with patch.object(comm.network, 'frame_at', wraps=comm.network.frame_at) as frames:
            host.run()
        self.assertEqual([call.args[0] for call in frames.call_args_list], [when])
        self.assertEqual(completed, [Time.from_ns(2760)])

    def test_fast_waiting_transfer_still_reacts_to_recovered_link(self):
        host, _, comm = self.fast_runtime()
        completed = []
        apply_device_command(host, {'action': 'device_link_fail', 'target_id': 'alocal'})
        comm.submit_transfer('waiting', 'ag', 'bg', 1000,
                             lambda: completed.append(host.now), self.fail)
        host.timeline.schedule(Time.from_ns(500), 0, 'restore',
            lambda: apply_device_command(host, {'action': 'device_link_restore', 'target_id': 'alocal'}))
        host.run()
        self.assertEqual(completed, [Time.from_ns(1510)])

    def test_ten_thousand_initial_notifications_build_once_and_deliver_once(self):
        host, network, comm = self.runtime()
        for index in range(10_000):
            host.push(Time(), 0, 'link_state', payload={'revision': host.net.revision})
        completed = []
        host.timeline.schedule(Time(), 1, 'submit', lambda: comm.submit_transfer(
            'request', 'ag', 'bg', 1000, lambda: completed.append(host.now), self.fail))
        with patch.object(comm.network, 'frame_at', wraps=comm.network.frame_at) as frames:
            host.run()
        self.assertEqual(completed, [Time.from_ns(1010)])
        self.assertEqual(sum(c.args[0] == Time() for c in frames.call_args_list), 1)
        self.assertEqual(comm.stats['coalesced_network_notifications'], 10_000)
        self.assertEqual(comm.stats['network_frame_builds'], frames.call_count)
        self.assertEqual(sum(r['kind'] == 'link_state' for r in host.log), 10_000)
        self.assertAlmostEqual(sum(r['application_bytes'] for r in comm.trace if r['kind'] == 'service'), 1000)

    def test_capacity_boundary_duplicates_do_not_double_debit_or_shift_completion(self):
        host, network, comm = self.runtime(capacity_schedule_ns=((500, 0.5),))
        completed = []
        comm.submit_transfer('request', 'ag', 'bg', 1000,
                             lambda: completed.append(host.now), self.fail)
        for index in range(1000):
            host.push(Time.from_ns(500), 0, 'link_state', payload={'revision': host.net.revision})
        with patch.object(comm.network, 'frame_at', wraps=comm.network.frame_at) as frames:
            host.run()
        self.assertEqual(completed, [Time.from_ns(1510)])
        self.assertEqual(sum(c.args[0] == Time.from_ns(500) for c in frames.call_args_list), 1)
        self.assertAlmostEqual(sum(r['application_bytes'] for r in comm.trace if r['kind'] == 'service'), 1000)
        self.assertFalse(comm.tasks or comm.flows or comm.active)

    def test_same_tick_manual_fault_and_restore_are_distinct_network_versions(self):
        host, network, comm = self.runtime()
        original = comm.route('ag', 'bg')['path']
        comm.network_changed()
        self.assertEqual(comm.stats['network_frame_builds'], 1)
        apply_device_command(host, {'action': 'device_link_fail', 'target_id': 'alocal'})
        with self.assertRaises(NoRouteError):
            comm.route('ag', 'bg')
        apply_device_command(host, {'action': 'device_link_restore', 'target_id': 'alocal'})
        self.assertEqual(comm.route('ag', 'bg')['path'], original)
        self.assertEqual(comm.stats['network_frame_builds'], 3)

    def test_route_query_after_revision_change_cannot_reuse_the_old_frame(self):
        host, network, comm = self.runtime()
        comm.route('ag', 'bg')
        host.net.failed_device_links.add('isl')
        host.net.revision += 1
        with self.assertRaises(NoRouteError):
            comm.route('ag', 'bg')
        self.assertEqual(comm.stats['network_frame_builds'], 2)

    def test_waiting_flow_wakes_once_when_a_physical_link_appears(self):
        physical = physical_network(edges=(('a', 'b', 0., 20.), ('b', 'c', 6., 20.)), bandwidth=8e9)
        host, network = device_runtime(physical)
        self.addCleanup(host.close)
        self.addCleanup(network.close)
        comm, completed = network.communication_policy, []
        comm.submit_transfer('waiting', 'a:hub', 'c:hub', 1000,
                             lambda: completed.append(host.now), self.fail)
        for index in range(1000):
            host.push(Time(6), 0, 'link_state', payload={'revision': host.net.revision})
        with patch.object(comm.network, 'frame_at', wraps=comm.network.frame_at) as frames:
            host.run()
        self.assertEqual(len(completed), 1)
        self.assertGreater(completed[0], Time(6))
        self.assertEqual(sum(c.args[0] == Time(6) for c in frames.call_args_list), 1)
        self.assertAlmostEqual(sum(r['application_bytes'] for r in comm.trace if r['kind'] == 'service'), 1000)

    def test_unversioned_provider_still_honors_explicit_same_tick_changes(self):
        frames, loop = Frames(), GlobalTimeline()
        comm = RuntimeCommunication(frames, loop, CommPolicy())
        self.addCleanup(comm.close)
        comm.network_changed()
        self.assertTrue(comm.route('g0', 'g1')['path'])
        frames.change, frames.down = Time(), True
        comm.network_changed()
        with self.assertRaises(NoRouteError):
            comm.route('g0', 'g1')
        self.assertEqual(comm.stats['network_frame_builds'], 2)
        self.assertEqual(comm.stats['coalesced_network_notifications'], 0)
