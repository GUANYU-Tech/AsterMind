"""Exercise Fast Mode through the V1 communication runtime and timeline."""

from types import SimpleNamespace
from collections import Counter
import json
import unittest

from astermind.common.bootstrap import activate
activate()
from astermind.builder.config.communication import CommPolicy
from astermind.engine.clock import Time
from astermind.engine.comm.event_runtime import RuntimeCommunication
from astermind.engine.comm.operator import CommunicationOperator
from astermind.engine.comm.planner import plan_communication
from astermind.engine.comm.registered_profiles import p2p_duration
from astermind.engine.compute.fast_mode.communication import RankProgram, Transfer
from astermind.engine.compute.fast_mode.executor import FastModeExecutor
from astermind.engine.compute.fast_mode.comm_reuse import BoundCommunicationPlan
from astermind.engine.compute.fast_mode.comm_pricing import RankCommunicationPricing
from astermind.engine.compute.fast_mode.predictor import Predictor, Row
from astermind.engine.compute.fast_mode.quote import BatchCommunication
from astermind.engine.timeline import GlobalTimeline
from tests.comm_fixture import Frames


class TwoFrames(Frames):
    def __init__(self, changed):
        super().__init__(change=Time("0.00001"))
        self.changed = changed

    def frame_at(self, time):
        frame = super().frame_at(time)
        if time < self.change and self.changed == "restored":
            frame.established_topology_graph = [
                link for link in frame.established_topology_graph if link.link_id != "ab"]
        if time >= self.change and self.changed == "reroute":
            frame.established_topology_graph = [
                link for link in frame.established_topology_graph if link.link_id != "ab"]
            frame.established_topology_graph.append(SimpleNamespace(link_id="alternate-isl",
                source_node_id="g2", target_node_id="g3", bandwidth_bps=200e9,
                propagation_delay_s=0., attributes={"bidirectional": False}, shared_capacity_pools=()))
        if time >= self.change and (not self.changed or self.changed == "restored"):
            for link in frame.established_topology_graph:
                if link.link_id == "ab":
                    link.bandwidth_bps = 400e9
                elif link.link_id == "grounda":
                    link.bandwidth_bps = 10e9
        return frame

    def forecast_routes(self, time):
        return self.routes(self.frame_at(time))


class BatchFrameTests(unittest.TestCase):
    def test_mixed_star_a800_direct_quotes_and_executes_local_flow_with_a800_profile(self):
        timeline = GlobalTimeline()
        runtime = RuntimeCommunication(Frames(), timeline, CommPolicy(
            cell_policies={"cell": {"hardware_id": "nvidia_a800"}}, trace_service=True,
            inter_parameters=dict(startup_us=0, hop_processing_us=0, routing_processing_us=0)))
        size = 524288
        task = CommunicationOperator("all_to_all", None, ("g0", "g2", "g1", "g3"),
                                     size + 512, "aggregate_input", cell_id="cell",
                                     traffic_matrix_bytes=((0, size, 0, 0), (0, 0, 512, 0),
                                                           (0, 0, 0, 0), (0, 0, 0, 0)))
        quote = BatchCommunication(runtime)
        plan = plan_communication(runtime.network, runtime.policy, task, quote.time, quote.routes)
        self.assertFalse(plan.local_runtime)
        path = quote.routes.route("g0", "g2", size)
        expected, _ = p2p_duration(quote.routes, path, size, "single_su",
                                   ("nvidia_a800", "nvidia_a800"), plan.policy, include_startup=True)
        self.assertEqual(quote._p2p(plan, path, size, task.operation), Time.from_ns(expected))
        done = []
        executor = FastModeExecutor("cell", timeline, runtime, ())
        executor.portable_hardware = True
        state = {"alive": True, "tasks": set()}
        executor._submit_runtime_operator(state, task, done.append, "mixed")
        timeline.run()
        self.assertEqual(len(done), 1)
        events = [record for record in runtime.trace if record["kind"] == "intra_service"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["duration_ns"], expected)
        self.assertEqual({row["profile_hardware"] for row in events[0]["endpoint_profiles"]}, {"nvidia_a800"})
        self.assertTrue(any(record["kind"] == "admit" for record in runtime.trace))
        self.assertFalse(state["tasks"])

    def test_mixed_operator_waits_for_local_path_lost_before_a_later_wave(self):
        down, up = Time("0.00001"), Time("0.001")

        class BrokenLocalLink(Frames):
            def frame_at(self, at):
                frame = super().frame_at(at)
                frame.valid_until_s = down if at < down else up if at < up else None
                if down <= at < up:
                    frame.established_topology_graph = [
                        link for link in frame.established_topology_graph if link.link_id != "g2a"]
                return frame

        timeline = GlobalTimeline()
        runtime = RuntimeCommunication(BrokenLocalLink(), timeline, CommPolicy(
            cell_policies={"cell": {"hardware_id": "nvidia_a800"}},
            inter_parameters=dict(startup_us=0, hop_processing_us=0, routing_processing_us=0)))
        remote, local = 10 * 1024**2, 512 * 1024
        task = CommunicationOperator("all_to_all", None, ("g0", "g2", "g1", "g3"),
                                     remote + local, "aggregate_input", cell_id="cell",
                                     traffic_matrix_bytes=((0, 0, remote, 0), (local, 0, 0, 0),
                                                           (0, 0, 0, 0), (0, 0, 0, 0)))
        executor = FastModeExecutor("cell", timeline, runtime, ())
        executor.portable_hardware = True
        state = {"alive": True, "tasks": set(), "record": {}}
        completed = []
        executor._submit_runtime_operator(state, task, lambda _: completed.append(timeline.now_s), "mixed")
        timeline.schedule(down, 0, "link-down", runtime.network_changed)
        timeline.schedule(up, 0, "link-up", runtime.network_changed)
        timeline.run()
        self.assertEqual(len(completed), 1)
        self.assertGreater(completed[0], up)
        self.assertEqual(state["record"]["local_route_waits"], 1)
        self.assertTrue(any(row["kind"] == "route_wait" for row in runtime.trace))
        self.assertFalse(state["tasks"])

    def run_case(self, changed):
        frames = TwoFrames(changed)
        loop = GlobalTimeline()
        comm = RuntimeCommunication(frames, loop, CommPolicy(inter_parameters=dict(
            startup_us=0, hop_processing_us=0, routing_processing_us=0)))
        transfer = Transfer("send_recv", "hidden_states", (0, 1), 1_000_000, 0)
        program = SimpleNamespace(cards=2, tp=1, dp=1, stages=((), ()), input_transfers=(),
                                  boundaries=(((transfer,),), ()), experts=((), ()), ranges=((0, 0), (0, 0)),
                                  p=SimpleNamespace(estimate=lambda *_, **kw: ((), 0.)))
        executor = FastModeExecutor("cell", loop, comm, ())
        executor.pricing = RankCommunicationPricing(comm, program, ("g0", "g1"), "cell")
        executor.fixed = BoundCommunicationPlan(executor.pricing)
        executor.prepared_frame = None
        executor.bound_available = False
        batch = SimpleNamespace(replica_tokens=(1,), stage_compute_s=(0., 0.), rows=())
        quoted = None if changed == "restored" else Time(executor._price(batch)[0])
        complete, failed = [], []
        executor.start_batch("cell", "step", batch, None, complete.append, token_count=1,
                             replica_tokens=(1,), on_failure=failed.append)
        loop.run()
        self.assertFalse(failed)
        self.assertEqual(len(complete), 1)
        return executor.records[0], complete[0], quoted, comm.trace

    def test_unrelated_link_change_keeps_exact_quote(self):
        record, completed, quoted, trace = self.run_case(False)
        self.assertEqual(completed, quoted)
        self.assertEqual(record["cross_frame"], "certified")
        self.assertEqual(record["communication_pricing"], "isolated")
        self.assertFalse(any(event["kind"] == "comm_plan" for event in trace))

    def test_used_link_capacity_change_runs_original_operator(self):
        record, completed, quoted, trace = self.run_case(True)
        self.assertGreater(completed, quoted)
        self.assertEqual(record["communication_pricing"], "runtime")
        self.assertEqual(sum(event["kind"] == "comm_plan" for event in trace), 1)

    def test_initial_missing_route_waits_for_real_restore(self):
        record, completed, _, trace = self.run_case("restored")
        self.assertEqual(record["runtime_reason"], "initial_route_unavailable")
        self.assertGreater(completed, Time("0.00001"))
        self.assertEqual(sum(event["kind"] == "route_wait" for event in trace), 1)

    def test_lost_used_hop_retries_over_the_new_directed_path(self):
        record, completed, quoted, trace = self.run_case("reroute")
        self.assertEqual(record["communication_pricing"], "runtime")
        self.assertGreater(completed, quoted)
        self.assertEqual(sum(event["kind"] == "retry" for event in trace), 1)
        paths = [event["path"] for event in trace if event["kind"] == "admit"]
        self.assertEqual(paths, [["g0", "a", "b", "g1"], ["g0", "a", "g2", "g3", "b", "g1"]])

    def test_wide_ep_dispatch_precedes_expert_compute_and_combine(self):
        root = activate()
        model = json.loads((root / "configs/model/glm-5.2.json").read_text(encoding="utf-8"))
        ranges = ((0, 38), (38, 78))
        predictor = Predictor("glm-5.2", model, 2, 2, 4, ranges)
        batch = predictor.estimate((Row("decode", 8, 31, 0, 8), Row("decode", 8, 31, 1, 8)))
        program = RankProgram(predictor, ranges, 8)
        runtime = SimpleNamespace(policy=CommPolicy(ep_strategy="direct"))
        pricing = RankCommunicationPricing(runtime, program, tuple(f"g{i}" for i in range(8)), "cell")
        events = []
        compute_ticks = 0
        for phase in pricing.runtime_phases(batch):
            if isinstance(phase, Time):
                compute_ticks += phase.ticks
                if phase:
                    events.append("compute")
            else:
                operators = [op for branch in phase for op in branch]
                if operators and operators[0].operation == "all_to_all":
                    events.append("dispatch" if len(operators) > 1 else "combine")
        first = events.index("dispatch")
        self.assertEqual(events[first - 1:first + 3], ["compute", "dispatch", "compute", "combine"])
        self.assertEqual(events.count("dispatch"), 75)
        self.assertEqual(events.count("combine"), 75)
        self.assertEqual(compute_ticks, sum(Time(duration).ticks for duration in batch.stage_compute_s))

    def test_runtime_preserves_wide_ep_rank_operations(self):
        root = activate()
        for model_name, tp, dp, stages in (
                ("glm-5.2", 2, 2, ((0, 38), (38, 78))),
                ("deepseek-v4-pro", 8, 4, ((0, 61),))):
            with self.subTest(model=model_name, stages=stages):
                model = json.loads((root / f"configs/model/{model_name}.json").read_text(encoding="utf-8"))
                layer_count = model.get("text_config", model)["num_hidden_layers"]
                if len(stages) == 1:
                    stages = ((0, layer_count),)
                predictor = Predictor(model_name, model, tp, dp, tp * dp, stages)
                batch = predictor.estimate(tuple(Row("decode", 8, 31, d, 8) for d in range(dp)))
                cards = tp * dp * len(stages)
                program = RankProgram(predictor, stages, cards)
                runtime = SimpleNamespace(policy=CommPolicy(ep_strategy="direct"))
                pricing = RankCommunicationPricing(runtime, program, tuple(f"g{i}" for i in range(cards)), "cell")
                expected = Counter()
                ordinary = [*program.input_transfers,
                            *(op for stage in program.stages for op in stage),
                            *(op for boundary in program.boundaries for phase in boundary for op in phase)]
                for transfer in ordinary:
                    size = transfer.payload(batch, tp)
                    if size:
                        expected[pricing._operator(transfer.operation, transfer.ranks, size)] += 1
                for stage in program.experts:
                    for group in stage:
                        for op, repetitions in pricing._expert_operations(group, batch):
                            expected[op] += group.repetitions * repetitions
                actual = Counter(op for phase in pricing.runtime_phases(batch) if not isinstance(phase, Time)
                                 for branch in phase for op in branch)
                self.assertEqual(actual, expected)
