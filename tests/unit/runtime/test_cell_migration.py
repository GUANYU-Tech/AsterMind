"""Transactions must retain native references until every partition terminates."""

import unittest
from types import SimpleNamespace

from astermind.engine.simulation import Simulation
from astermind.engine.cell_events import CellCompletion
from astermind.engine.cell_migration import NetworkKVMigration
from astermind.engine.timeline import GlobalTimeline


class Cell:
    storage_signature = ("compatible",)
    _inflight = None

    def __init__(self, cell_id, *, accepted=True):
        self.cell_id, self.accepted = cell_id, accepted
        self.paused, self.calls = False, []

    def pause_for_migration(self, identifier):
        self.paused = True

    def resume_after_migration(self, identifier):
        self.paused = False

    def migration_memory_time(self, size, operation):
        return 1.0

    def storage_rpc(self, operation, *args):
        self.calls.append((operation, args))
        return {"accepted": self.accepted, "bytes": 100, "reason": "capacity"}


class CellMigrationTests(unittest.TestCase):
    def setup_migration(self, target=None):
        loop = GlobalTimeline()
        sends = []
        executor = SimpleNamespace(
            boundary_listeners=[],
            network=SimpleNamespace(candidate_path=lambda *args: None),
            submit_transport=lambda *args: sends.append(args),
        )
        source, target = Cell("A"), target or Cell("B")
        migration = NetworkKVMigration(
            loop, executor, {"A": {0: "a0", 1: "a1"}, "B": {0: "b0", 1: "b1"}}
        )
        return loop, sends, source, target, executor, migration

    def test_failure_waits_for_other_inflight_partition(self):
        loop, sends, source, target, _, migration = self.setup_migration()
        record = migration.start(
            SimpleNamespace(id="kv", mode="move"), source, target, {}
        )
        loop.run(until_s=1)
        sends[0][-1](SimpleNamespace(model_dump=lambda: {"reason": "fault"}))
        self.assertTrue(source.paused and target.paused)
        self.assertNotIn("rollback_import", [op for op, _ in target.calls])
        sends[1][-2]()
        self.assertEqual(record["state"], "failed")
        self.assertFalse(source.paused or target.paused)
        self.assertIn(("release_export", ("kv", False)), source.calls)

    def test_capacity_denial_releases_source_without_network_transfers(self):
        loop, sends, source, target, _, migration = self.setup_migration(
            Cell("B", accepted=False)
        )
        record = migration.start(
            SimpleNamespace(id="kv", mode="copy"), source, target, {}
        )
        loop.run()
        self.assertEqual(record["state"], "capacity_blocked")
        self.assertFalse(sends)
        self.assertFalse(source.paused or target.paused)

    def test_close_while_waiting_for_batch_releases_pauses_and_listener(self):
        _, _, source, target, executor, migration = self.setup_migration()
        source._inflight = object()
        record = migration.start(
            SimpleNamespace(id="kv", mode="copy"), source, target, {}
        )
        migration.close()
        self.assertEqual(record["state"], "failed")
        self.assertFalse(source.paused or target.paused)
        self.assertEqual(executor.boundary_listeners, [])

    def test_close_during_target_write_does_not_commit_later(self):
        loop, sends, source, target, _, migration = self.setup_migration()
        record = migration.start(
            SimpleNamespace(id="kv", mode="copy"), source, target, {}
        )
        loop.run(until_s=1)
        for send in sends:
            send[-2]()
        migration.close()
        loop.run()
        self.assertEqual(record["state"], "failed")
        self.assertNotIn("commit_import", [op for op, _ in target.calls])
        self.assertFalse(source.paused or target.paused)

    def test_invalid_mode_is_rejected_before_taking_native_references(self):
        _, _, source, target, _, migration = self.setup_migration()
        with self.assertRaises(ValueError):
            migration.start(
                SimpleNamespace(id="kv", mode="invalid"), source, target, {}
            )
        self.assertFalse(source.paused or target.paused or source.calls or target.calls)

    def test_session_placement_changes_only_after_target_commit(self):
        loop, sends, source, target, _, migration = self.setup_migration()
        runtime = Simulation.__new__(Simulation)
        runtime.migrations = migration
        runtime.cells = {"A": source, "B": target}
        runtime._session_cells = {"session": "A"}
        runtime.completions = []
        runtime.state = {"request": {}}
        runtime.history = SimpleNamespace(history=lambda *args: {})
        runtime.config = SimpleNamespace(migrations=(SimpleNamespace(id="kv", mode="move", source_cell="A", target_cell="B", session_id="session", context_id="ctx", after_request="request"),))
        runtime._record_request = lambda *args: None
        runtime._finish = lambda *args: None
        from astermind.engine.clock import Time
        runtime._completed(CellCompletion("A", "request", Time(), Time(), Time(), None, None, 1, 0., 0.))
        loop.run(until_s=1)
        for send in sends:
            send[-2]()
        self.assertEqual(runtime._session_cells["session"], "A")
        loop.run()
        self.assertEqual(runtime._session_cells["session"], "B")

    def test_all_cells_close_even_when_one_native_cleanup_fails(self):
        runtime = Simulation.__new__(Simulation)
        runtime._physics_lifetime = runtime.host = runtime.network = None
        closed = []

        def failed_close():
            closed.append("B")
            raise RuntimeError("native close failed")

        runtime.cells = {
            "A": SimpleNamespace(close=lambda: closed.append("A")),
            "B": SimpleNamespace(close=failed_close),
        }
        with self.assertRaisesRegex(RuntimeError, "native close failed"):
            runtime.close()
        self.assertCountEqual(closed, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
