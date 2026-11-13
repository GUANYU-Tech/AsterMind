"""Explicit dependency contracts; no compute or transport replacement."""

from pathlib import Path
import unittest
from types import SimpleNamespace

from astermind.engine.simulation import Simulation
from astermind.engine.timeline import GlobalTimeline
from astermind.engine.cell_migration import NetworkKVMigration
from astermind.engine.cell_events import CellCompletion
from tests.unit.runtime.test_cell_migration import Cell

from astermind.builder.config import SimulationConfig

ROOT = Path(__file__).resolve().parents[3]


class PrerequisiteConfigTests(unittest.TestCase):
    def scenario(self):
        raw = SimulationConfig.load(ROOT / "examples/native/dynamic_migration_tp2.json").model_dump()
        raw["migrations"][0]["before_request"] = "context1"
        return raw

    def test_explicit_receiver_and_legacy_unbound_migration(self):
        raw = self.scenario()
        config = SimulationConfig.model_validate(raw)
        self.assertEqual(config.predecessor(config.requests[1]), "context0")
        raw["migrations"][0].pop("before_request")
        self.assertIsNone(SimulationConfig.model_validate(raw).migrations[0].before_request)

    def test_receiver_requires_explicit_matching_cell(self):
        for cell in (None, "cellA"):
            with self.subTest(cell=cell):
                raw = self.scenario()
                raw["requests"][1]["cell"] = cell
                with self.assertRaisesRegex(ValueError, "explicit receiving request Cell"):
                    SimulationConfig.model_validate(raw)

    def test_missing_parent_is_rejected(self):
        raw = self.scenario()
        raw["requests"][1]["parent_context_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "explicit parent request"):
            SimulationConfig.model_validate(raw)

    def test_parent_request_id_remains_a_valid_default_context(self):
        raw = self.scenario()
        raw["requests"][0]["context_id"] = None
        raw["requests"][1]["parent_context_id"] = "context0"
        raw["migrations"][0]["context_id"] = "context0"
        config = SimulationConfig.model_validate(raw)
        self.assertEqual(config.predecessor(config.requests[1]), "context0")

    def test_duplicate_context_is_rejected(self):
        raw = self.scenario()
        raw["requests"][1]["context_id"] = "c0"
        with self.assertRaisesRegex(ValueError, "duplicate session context"):
            SimulationConfig.model_validate(raw)

    def test_cyclic_parent_dependencies_are_rejected(self):
        raw = self.scenario()
        raw["requests"][0]["parent_context_id"] = "c1"
        with self.assertRaisesRegex(ValueError, "cyclic KV"):
            SimulationConfig.model_validate(raw)

    def test_unknown_receiver_is_rejected(self):
        raw = self.scenario()
        raw["migrations"][0]["before_request"] = "missing"
        with self.assertRaisesRegex(ValueError, "known receiving request"):
            SimulationConfig.model_validate(raw)

    def test_migration_history_must_match_receiver(self):
        for field in ("session_id", "context_id"):
            with self.subTest(field=field):
                raw = self.scenario()
                raw["migrations"][0][field] = "different"
                with self.assertRaisesRegex(ValueError, "KV history"):
                    SimulationConfig.model_validate(raw)


class PrerequisiteLifecycleTests(unittest.TestCase):
    def setup_runtime(self, *, input_delay=1, parent_time=2, accepted=True, bound=True):
        raw = PrerequisiteConfigTests().scenario()
        raw["requests"][1]["arrival_ns"] = 0
        if not bound:
            raw["migrations"] = ()
        config = SimulationConfig.model_validate(raw)
        loop = GlobalTimeline()
        runtime = Simulation.__new__(Simulation)
        runtime.config, runtime.loop = config, loop
        runtime.requests = {r.id: r for r in config.requests}
        runtime.state = {r.id: dict(status="pending", arrival_ns=0, received_tokens=0,
                                   first_token_received_ns=None,last_token_received_ns=None)
                         for r in config.requests}
        runtime.state["context0"].update(status="computing", cell="cellA")
        runtime._cell_configs = {c.id:c for c in config.cells}
        runtime._session_cells, runtime._failed_cells = {}, set()
        runtime._kv_waiters, runtime._parent_waiters, runtime._request_transfers = {}, {}, {}
        runtime._transport_serial = 0
        runtime.completions, runtime.outputs, runtime.decisions = [], [], []
        runtime.request_state_events = []
        runtime.host = SimpleNamespace(log=[])
        runtime.history = SimpleNamespace(history=lambda *args: ())
        runtime._received_chunks = {r.id:set() for r in config.requests}
        pending, cancelled = {}, []
        def cancel(identities, reason):
            for identity in identities:
                callback = pending.pop(identity, None)
                if callback:
                    cancelled.append(identity)
                    callback(SimpleNamespace(model_dump=lambda:dict(reason=reason)))
        network = SimpleNamespace(route=lambda *args:dict(isolated_estimate_ns=0),
                                  candidate_path=lambda *args:None, cancel=cancel)
        runtime.network = network
        def submit(identity, source, target, size, operation, correlation, done, fail):
            pending[identity] = fail
            delay = input_delay if operation == "request_input" else 1
            def delivered():
                if pending.pop(identity, None) is not None:
                    done()
            loop.schedule(loop.now_s+delay, 1, "test:delivery", delivered)
        runtime.executor = SimpleNamespace(network=network, boundary_listeners=[], submit_transport=submit)
        source, target = Cell("cellA"), Cell("cellB", accepted=accepted)
        target.admissions = []
        target.submit_ready_request = lambda request: target.admissions.append(loop.now_s)
        runtime.cells = {"cellA":source,"cellB":target}
        runtime.migrations = NetworkKVMigration(loop,runtime.executor,
                                               {"cellA":{0:"a"},"cellB":{0:"b"}})
        child = config.requests[1]
        runtime._arrive(child)
        def parent_complete():
            runtime._completed(CellCompletion("cellA","context0",loop.now_s,loop.now_s,
                                             loop.now_s,None,None,2,0.,0.))
        loop.schedule(parent_time,1,"test:parent",parent_complete)
        return runtime, target, cancelled

    def test_input_and_kv_order_both_gate_exactly_once(self):
        for input_delay in (1, 5, 9):
            with self.subTest(input_delay=input_delay):
                runtime,target,_=self.setup_runtime(input_delay=input_delay)
                runtime.loop.run()
                child=runtime.state["context1"]
                self.assertEqual(target.admissions,[max(input_delay,5)])
                self.assertEqual(child["cell_ready_ns"],max(child["input_ready_ns"],child["kv_ready_ns"]))
                runtime._kv_terminal("copy0")
                self.assertEqual(len(target.admissions),1,"duplicate notification must not admit twice")
                self.assertEqual(runtime.migrations.records["copy0"]["state"],"completed")

    def test_parent_without_migration_still_gates_admission(self):
        runtime,target,_=self.setup_runtime(parent_time=7,bound=False)
        runtime.loop.run()
        self.assertEqual(target.admissions,[7])
        self.assertNotIn("completed_ns",runtime.state["context0"],"parent outputs need not be delivered")

    def test_capacity_denial_fails_receiver_and_releases_cells(self):
        runtime,target,_=self.setup_runtime(accepted=False)
        runtime.loop.run()
        self.assertEqual(runtime.state["context1"]["status"],"failed")
        self.assertEqual(runtime.migrations.records["copy0"]["state"],"capacity_blocked")
        self.assertFalse(target.admissions or target.paused or runtime.cells["cellA"].paused)

    def test_parent_failure_cancels_input_and_ignores_late_completion(self):
        runtime,target,cancelled=self.setup_runtime(input_delay=20,parent_time=7)
        runtime.loop.schedule(2,1,"test:fail",lambda:runtime._request_failed("context0",0,"test_failure",dict(reason="failed")))
        runtime.loop.run()
        self.assertEqual(runtime.state["context1"]["status"],"failed")
        self.assertTrue(cancelled)
        self.assertFalse(target.admissions or runtime.migrations.records)
        self.assertNotIn("compute_complete_ns",runtime.state["context0"])

    def test_parent_output_failure_after_compute_does_not_invalidate_kv(self):
        runtime,target,_=self.setup_runtime()
        runtime.loop.schedule(3,1,"test:output_failure",lambda:runtime._request_failed("context0",0,"transfer_failed",dict(reason="output")))
        runtime.loop.run()
        self.assertEqual(target.admissions,[5])
        self.assertEqual(runtime.state["context0"]["status"],"failed")
        self.assertEqual(runtime.migrations.records["copy0"]["state"],"completed")

    def test_admission_error_becomes_terminal_failure(self):
        runtime,target,_=self.setup_runtime()
        def rejected(request):
            raise ValueError("native admission rejected")
        target.submit_ready_request=rejected
        runtime.loop.run()
        self.assertEqual(runtime.state["context1"]["status"],"failed")
        self.assertEqual(runtime.state["context1"]["failure"]["reason"],"native admission rejected")

    def test_incompatible_storage_fails_receiver_before_taking_references(self):
        runtime, target, _ = self.setup_runtime()
        target.storage_signature = ("different model, KV format or parallel layout",)
        runtime.loop.run()
        self.assertEqual(runtime.state["context1"]["status"], "failed")
        self.assertIn("matching model, KV format and TP/PP partitions",
                      runtime.state["context1"]["failure"]["message"])
        self.assertFalse(target.admissions or target.calls or runtime.cells["cellA"].calls)
        self.assertFalse(target.paused or runtime.cells["cellA"].paused)


class TraceClockTests(unittest.TestCase):
    def test_current_event_exists_only_inside_its_callback(self):
        for hosted in (False, True):
            with self.subTest(hosted=hosted):
                from astermind.engine.clock import Time
                pending = []
                host = SimpleNamespace(now=Time(), push=lambda *args, **kw: pending.append(kw["payload"]))
                loop = GlobalTimeline(host=host if hosted else None)
                seen = []
                def callback():
                    seen.append(loop.current_event)
                    raise ValueError("trace cleanup")
                event = loop.schedule(Time(), 1, "trace:test", callback)
                self.assertIsNone(loop.current_event)
                with self.assertRaisesRegex(ValueError, "trace cleanup"):
                    pending[0]() if hosted else loop.run()
                self.assertEqual(seen, [event])
                self.assertIsNone(loop.current_event)


if __name__ == "__main__":
    unittest.main()
