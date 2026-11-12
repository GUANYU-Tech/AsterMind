"""Real SimGrid owns deadlines as well as shared communication activities."""

import importlib.util
import unittest
from types import SimpleNamespace

from astermind.engine.clock import Time
from astermind.engine.simgrid_events import SimGridEventQueue, native_time
from astermind.engine.simgrid_transport import SimGridTransport


@unittest.skipUnless(importlib.util.find_spec("simgrid"), "requires native SimGrid")
class SimGridGlobalTests(unittest.TestCase):
    def setUp(self):
        self.engine = SimGridTransport(
            ("a", "b", "c"),
            (dict(link_id="ab", endpoint_a="a", endpoint_b="b", kind="isl"),),
            {"isl": SimpleNamespace(bandwidth_bps=8000)},
            global_scheduler=True,
        )
        self.addCleanup(self.engine.close)
        self.events = SimGridEventQueue(self.engine)
        self.leg = dict(link_id="ab", source_node_id="a", target_node_id="b")
        self.engine.configure({self.engine.resource(self.leg): 8000})

    def schedule(self, identity, at, priority=10):
        return self.events.append(
            (Time(at), priority, identity, "callback", None, identity)
        )

    def test_native_timers_order_priority_sequence_and_dynamic_same_time(self):
        self.schedule(3, 2, 10)
        self.schedule(2, 1, 10)
        self.schedule(1, 1, -2)
        response = self.engine.advance_next(3)
        self.assertEqual(response["time_s"], 1)
        self.assertEqual(self.events.pop_due()[2], 1)
        self.schedule(4, 1, 0)
        self.assertEqual(self.events.pop_due()[2], 4)
        self.assertEqual(self.events.pop_due()[2], 2)
        self.assertIsNone(self.events.pop_due())
        self.assertEqual(self.engine.advance_next(3)["time_s"], 2)
        self.assertEqual(self.events.pop_due()[2], 3)
        self.assertEqual(self.engine.scheduler_stats["dispatched"], 4)

    def test_network_completion_preempts_later_compute_timer(self):
        self.schedule(1, 3)
        self.engine.start("payload", self.leg, 1000)
        response = self.engine.advance_next(4)
        self.assertEqual(response["completed"], ["payload"])
        self.assertAlmostEqual(response["time_s"], 1)
        self.assertEqual(response["bytes_sent"]["payload"], 1000)
        self.assertIsNone(self.events.pop_due())
        self.assertEqual(self.engine.advance_next(4)["time_s"], 3)
        self.assertEqual(self.events.pop_due()[2], 1)

    def test_request_arrival_changes_native_sharing_before_old_completion(self):
        self.schedule(1, 0.25)
        self.engine.start("first", self.leg, 1000)
        response = self.engine.advance_next(4)
        self.assertEqual(response["time_s"], 0.25)
        self.assertAlmostEqual(response["bytes_sent"]["first"], 250)
        self.events.pop_due()
        self.engine.start("second", self.leg, 1000)
        response = self.engine.advance_next(4)
        self.assertEqual(response["completed"], ["first"])
        self.assertAlmostEqual(response["time_s"], 1.75)
        response = self.engine.advance_next(4)
        self.assertEqual(response["completed"], ["second"])
        self.assertAlmostEqual(response["time_s"], 2)

    def test_cancelled_deadline_does_not_advance_or_dispatch(self):
        self.schedule(1, 1)
        self.schedule(2, 2)
        self.events.retain(lambda event: event[2] != 1)
        self.assertEqual(self.engine.advance_next(3)["time_s"], 2)
        self.assertEqual(self.events.pop_due()[2], 2)
        self.assertEqual(self.engine.scheduler_stats["cancelled"], 1)

    def test_zero_capacity_and_horizon_are_native_events(self):
        self.engine.start("paused", self.leg, 1000)
        self.engine.configure({self.engine.resource(self.leg): 0})
        self.schedule(1, 1)
        response = self.engine.advance_next(2)
        self.assertEqual(response["bytes_sent"]["paused"], 0)
        self.events.pop_due()
        self.engine.configure({self.engine.resource(self.leg): 8000})
        response = self.engine.advance_next(2)
        self.assertEqual(response["completed"], ["paused"])
        self.assertTrue(response["horizon_reached"])

    def test_new_edge_rebuild_preserves_global_deadline_and_partial_flow(self):
        self.schedule(1, 0.5)
        self.schedule(2, 2)
        self.engine.start("payload", self.leg, 1000)
        self.engine.advance_next(3)
        self.events.pop_due()
        self.engine.register_edge(
            dict(link_id="bc", endpoint_a="b", endpoint_b="c", kind="isl")
        )
        response = self.engine.advance_next(3)
        self.assertEqual(response["completed"], ["payload"])
        self.assertAlmostEqual(response["time_s"], 1)
        self.assertEqual(self.engine.advance_next(3)["time_s"], 2)
        self.assertEqual(self.events.pop_due()[2], 2)
        self.assertEqual(self.engine.scheduler_stats["scheduled"], 2)
        self.assertEqual(self.engine.scheduler_stats["dispatched"], 2)

    def test_platform_rebuild_preserves_already_due_callbacks(self):
        self.schedule(1, 0, 10)
        self.schedule(2, 0, -2)
        self.engine.register_edge(
            dict(link_id="bc", endpoint_a="b", endpoint_b="c", kind="isl")
        )
        self.assertEqual(self.events.pop_due()[2], 2)
        self.assertEqual(self.events.pop_due()[2], 1)

    def test_deadlines_never_round_into_the_past(self):
        at = Time("1.000000000000000001")
        aligned = self.schedule(1, at)
        self.assertGreaterEqual(aligned, at)
        self.assertEqual(aligned, Time(native_time(at)))
        self.engine.advance_next(2)
        self.events.pop_due()
        with self.assertRaisesRegex(RuntimeError, "past"):
            self.schedule(2, 0)

    def test_recursive_fractional_deadlines_use_actual_engine_clock(self):
        for identity in range(1, 101):
            at = self.engine.now + Time("0.000000013333333333")
            self.schedule(identity, at)
            self.engine.advance_next(1)
            event = self.events.pop_due()
            self.assertGreaterEqual(event[0], at)
            self.assertEqual(event[0], self.engine.now)
            self.schedule(identity + 1000, self.engine.now)
            self.assertEqual(self.events.pop_due()[0], self.engine.now)

    def test_thousands_of_future_deadlines_do_not_allocate_actor_threads(self):
        from pathlib import Path

        tasks = Path(f'/proc/{self.engine.process.pid}/task')
        before = len(list(tasks.iterdir()))
        for identity in range(3000):
            self.schedule(identity, 1 + identity / 10000)
        self.assertLessEqual(len(list(tasks.iterdir())), before + 2)
        self.events.retain(lambda event: event[2] in {0, 1500, 2999})
        for identity in (0, 1500, 2999):
            self.engine.advance_next(3)
            self.assertEqual(self.events.pop_due()[2], identity)
        self.assertEqual(self.engine.scheduler_stats['cancelled'], 2997)

    def test_many_shared_communications_use_one_native_wait(self):
        from pathlib import Path

        tasks = Path(f'/proc/{self.engine.process.pid}/task')
        before = len(list(tasks.iterdir()))
        for identity in range(1000):
            self.engine.start(str(identity), self.leg, 1)
        self.assertLessEqual(len(list(tasks.iterdir())), before + 2)
        response = self.engine.advance_next(3)
        self.assertEqual(len(response['completed']), 1000)
        self.assertAlmostEqual(response['time_s'], 1)
        self.assertAlmostEqual(sum(response['bytes_sent'].values()), 1000)
