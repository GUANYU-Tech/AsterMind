import unittest

from simulator.config import SimulationConfig, Migration
from simulator.examples.make_examples import example
from simulator.migration import NetworkKVMigration
from network_sim import CommunicationRuntime, PhysicalTopology, Scenario


class NativeBlockFixture:
    """RPC protocol fixture; tests delivery ordering, not allocator accuracy."""
    storage_signature = ("same-model", 16, 100)
    _inflight = None

    def __init__(self, cell_id):
        self.cell_id, self.paused = cell_id, None
        self.reserved = self.committed = self.rolled_back = False

    def pause_for_migration(self, identity):
        assert self.paused is None
        self.paused = identity

    def resume_after_migration(self, identity):
        assert self.paused == identity
        self.paused = None

    def migration_memory_time(self, size, kind):
        return 0.

    def storage_rpc(self, action, *args):
        if action == "export_prefix":
            return {"payload": "fixture"}
        if action == "reserve_import":
            self.reserved = True
            return dict(accepted=True, bytes=1000000)
        if action == "commit_import":
            self.committed = True
        if action == "rollback_import":
            self.rolled_back = True
        if action == "release_export":
            return dict(removed_blocks=0, retained_blocks=1)


class MigrationTests(unittest.TestCase):
    def run_case(self, failure, *, wait_for_boundaries=False):
        from orbitserve.runtime.engine.network_events import NetworkEventDriver
        from orbitserve.runtime.engine.cell_events import CellEventLoop
        config = example()
        if failure:
            config["physical"]["link_policy"] = dict(mode="simple_keep", check_interval_ns=1000,
                initial_acquisition_ns=0, reacquisition_ns=0)
            next(l for l in config["physical"]["links"] if l["id"] == "isl")["availability_windows"] = [dict(start_ns=0,end_ns=1000)]
        topology = SimulationConfig.model_validate(config).topology()
        loop = CellEventLoop()
        source, target = NativeBlockFixture("cellA"), NativeBlockFixture("cellB")
        with CommunicationRuntime(topology) as network:
            transport = NetworkEventDriver(loop, network)
            migration = NetworkKVMigration(transport, dict(cellA="gpu0", cellB="gpu1"))
            spec = Migration(id="copy", after_request="r0", source_cell="cellA", target_cell="cellB",
                             session_id="s", context_id="c")
            if wait_for_boundaries:
                source._inflight = target._inflight = object()
            record = migration.start(spec, source, target, {})
            if wait_for_boundaries:
                self.assertEqual(record["state"], "waiting_for_batch_boundary")
                source._inflight = None
                migration.on_batch_boundary()
                self.assertFalse(target.reserved)
                target._inflight = None
                migration.on_batch_boundary()
            loop.run(until_s=0)
            self.assertTrue(target.reserved)
            self.assertFalse(target.committed)
            while record["state"] in ("transferring", "waiting_for_batch_boundary"):
                at = loop.next_event_time_s
                self.assertIsNotNone(at)
                loop.run(until_s=at)
            self.assertIsNone(source.paused)
            self.assertIsNone(target.paused)
            self.assertEqual(target.committed, not failure)
            self.assertEqual(target.rolled_back, failure)
            self.assertEqual(record["state"], "failed" if failure else "completed")

    def test_commit_only_after_shared_network_completion(self):
        self.run_case(False)

    def test_link_failure_rolls_back_native_reservation(self):
        self.run_case(True)

    def test_migration_waits_for_both_cell_boundaries(self):
        self.run_case(False, wait_for_boundaries=True)

    def multiple_rank_case(self, failure):
        from orbitserve.runtime.engine.network_events import NetworkEventDriver
        from orbitserve.runtime.engine.cell_events import CellEventLoop
        raw = dict(cells=[dict(id="cellA", satellites=["a"]), dict(id="cellB", satellites=["b"])],
                   satellites=[dict(id=i) for i in "ab"], devices=[], links=[])
        for owner in "ab":
            for rank in range(2):
                raw["devices"].extend([dict(id=f"g{owner}{rank}", satellite=owner, kind="gpu"),
                    dict(id=f"t{owner}{rank}", satellite=owner, kind="terminal", external_access=True)])
                raw["links"].append(dict(id=f"local{owner}{rank}", source=f"g{owner}{rank}",
                    target=f"t{owner}{rank}", bandwidth_gbps=8))
        for rank in range(2):
            raw["links"].append(dict(id=f"isl{rank}", source=f"ta{rank}", target=f"tb{rank}",
                bandwidth_gbps=8 if rank == 0 else 1))
        if failure:
            raw["link_policy"] = dict(mode="simple_keep", check_interval_ns=1_000_000_000,
                initial_acquisition_ns=0, reacquisition_ns=0)
            raw["links"][-2]["availability_windows"] = [dict(start_ns=0, end_ns=1000)]
        loop = CellEventLoop()
        source, target = NativeBlockFixture("cellA"), NativeBlockFixture("cellB")
        with CommunicationRuntime(PhysicalTopology(Scenario.model_validate(raw))) as network:
            transport = NetworkEventDriver(loop, network)
            migration = NetworkKVMigration(transport,
                dict(cellA={0: "ga0", 1: "ga1"}, cellB={0: "gb0", 1: "gb1"}))
            record = migration.start(Migration(id="multi", after_request="r", source_cell="cellA",
                target_cell="cellB", session_id="s", context_id="c"), source, target, {})
            loop.run(until_s=0.000002 if failure else 0.002)
            self.assertFalse(target.committed)
            self.assertFalse(target.rolled_back)
            self.assertIsNotNone(source.paused)
            self.assertIsNotNone(target.paused)
            self.assertEqual(record["state"], "transferring")
            loop.run(until_s=0.02)
            self.assertEqual(record["bytes"], 2_000_000)
            self.assertEqual({p["source"] for p in record["partitions"]}, {"ga0", "ga1"})
            self.assertEqual(target.committed, not failure)
            self.assertEqual(target.rolled_back, failure)
            self.assertEqual(record["state"], "failed" if failure else "completed")
            self.assertIsNone(source.paused)

    def test_all_ranks_must_arrive_before_kv_commit(self):
        self.multiple_rank_case(False)

    def test_failed_rank_keeps_source_protected_until_other_rank_is_terminal(self):
        self.multiple_rank_case(True)


if __name__ == "__main__":
    unittest.main()
