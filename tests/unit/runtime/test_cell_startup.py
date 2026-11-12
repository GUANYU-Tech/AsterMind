"""Startup overlaps independent work without exposing partial Cells."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulator.startup import initialize_cells
from simulator.startup_resources import GIB, startup_policy


class CellStartupTests(unittest.TestCase):
    def test_bounded_parallelism_and_configuration_order(self):
        barrier, lock = Barrier(2, timeout=5), Lock()
        active = maximum = 0
        configs = [SimpleNamespace(id=str(i)) for i in range(6)]
        progress = []

        def create(config):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait()
            with lock:
                active -= 1
            return SimpleNamespace(id=config.id)

        result = initialize_cells(
            configs, create, workers=2, progress=lambda **row: progress.append(row)
        )
        self.assertEqual(maximum, 2)
        self.assertEqual(list(result), [c.id for c in configs])
        self.assertEqual(progress[-1]["completed"], 6)
        self.assertEqual(progress[-1]["active"], [])
        self.assertTrue(all(len(row["active"]) <= 2 for row in progress))

    def test_cancellation_cleans_late_successes_without_starting_more_cells(self):
        started = Barrier(3, timeout=5)
        release, cancelling = Event(), Event()
        calls, closed = [], []
        configs = [SimpleNamespace(id=str(i)) for i in range(4)]

        def create(config):
            calls.append(config.id)
            started.wait()
            if not release.wait(5):
                raise TimeoutError("test did not release startup")
            return SimpleNamespace(close=lambda: closed.append(config.id))

        def cancel(**row):
            started.wait()
            cancelling.set()
            raise RuntimeError("cancelled by caller")

        with ThreadPoolExecutor(max_workers=1) as owner:
            future = owner.submit(
                initialize_cells, configs, create, workers=2, progress=cancel
            )
            self.assertTrue(cancelling.wait(5))
            self.assertFalse(future.done())
            release.set()
            with self.assertRaisesRegex(RuntimeError, "cancelled by caller"):
                future.result(timeout=5)
        self.assertCountEqual(calls, ["0", "1"])
        self.assertCountEqual(closed, calls)

    def test_factory_failure_closes_other_successful_cells(self):
        closed = []
        barrier = Barrier(2, timeout=5)

        def create(config):
            barrier.wait()
            if config.id == "bad":
                raise ValueError("scheduler startup failed")
            return SimpleNamespace(close=lambda: closed.append(config.id))

        with self.assertRaisesRegex(ValueError, "scheduler startup failed"):
            initialize_cells(
                [SimpleNamespace(id="good"), SimpleNamespace(id="bad")],
                create,
                workers=2,
            )
        self.assertEqual(closed, ["good"])

    def test_explicit_serial_fallback_and_invalid_worker_limits(self):
        calls = []
        configs = [SimpleNamespace(id=str(i)) for i in range(3)]
        result = initialize_cells(configs, lambda c: calls.append(c.id) or c, workers=1)
        self.assertEqual(calls, list(result))
        for workers in (0, -1, True, 1.5, "2"):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                initialize_cells(
                    configs,
                    lambda c: self.fail("invalid limit started work"),
                    workers=workers,
                )

    @patch("simulator.startup_resources.available_cpu_count", return_value=16)
    @patch("simulator.startup_resources.available_memory_bytes", return_value=64 * GIB)
    def test_automatic_policy_uses_all_available_cpus_when_memory_allows(self, *_):
        self.assertEqual(startup_policy(32)["workers"], 16)
        self.assertEqual(startup_policy(2)["workers"], 2)
        self.assertEqual(startup_policy(8, requested=1)["workers"], 1)

    @patch("simulator.startup_resources.available_cpu_count", return_value=16)
    @patch("simulator.startup_resources.available_memory_bytes", return_value=8 * GIB)
    def test_automatic_policy_reserves_memory_for_the_rest_of_the_server(self, *_):
        policy = startup_policy(32)
        self.assertEqual(policy["workers"], 6)
        self.assertEqual(policy["memory_slots"], 6)
        self.assertEqual(policy["source"], "automatic")
