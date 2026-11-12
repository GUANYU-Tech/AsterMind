import importlib.util
import unittest
from dataclasses import replace
from types import SimpleNamespace

from astermind.engine.events import _EventEngine
from astermind.engine.state.network import NetworkState
from tests.unit.runtime.test_dynamic_network import network


def execute(mode, target_type):
    base = network(
        edges=(
            ("a", "b", 0.0, 20.0),
            ("b", "c", 0.0, 20.0),
            ("a", "d", 0.0, 20.0),
            ("d", "c", 0.0, 20.0),
        ),
        bandwidth=8000,
    )
    fault = dict(
        target_type=target_type,
        target_id="b" if target_type == "node" else "communication_isl:a:b",
        start_s=0.25,
        end_s=4.0,
    )
    trace = base.trace
    trace.ephemeris_samples = tuple(
        dict(row, time_s=float(t))
        for t in range(21)
        for row in trace.ephemeris_samples
        if row["time_s"] == 0
    )
    net = NetworkState(
        trace,
        replace(
            base.spec,
            transport_model=mode,
            failure_policy="retry_next_sample",
            faults=(fault,),
        ),
    )
    spec = SimpleNamespace(
        serving_profile=SimpleNamespace(internal_comm_accounting="embedded"),
        time_horizon_s=20,
        routing_policy="earliest_arrival",
    )
    engine = _EventEngine(
        spec, SimpleNamespace(network=net, time=trace.time)
    )
    engine.debit = lambda *args, **kwargs: None
    engine.on_delivered = lambda job, unused: job.update(done=True)
    job = dict(request=dict(request_id="r"), path=[], ledger=[], done=False)
    engine.jobs["r"] = job
    engine.send(job, "a", "c", 1000, None, "input", "delivered")
    engine.push(0.25, -2, "fault_start", None, (0, fault))
    engine.push(4.0, -2, "fault_end", None, (0, fault))
    engine.run()
    return job, engine.log


class RecoveryTests(unittest.TestCase):
    def verify_middle_hop_retains_delivered_prefix(self, mode):
        base = network(
            edges=(
                ("s", "a", 0.0, 20.0),
                ("a", "b", 0.0, 20.0),
                ("b", "t", 0.0, 20.0),
                ("a", "d", 0.0, 20.0),
                ("d", "t", 0.0, 20.0),
            ),
            bandwidth=8000,
        )
        fault = dict(
            target_type="link",
            target_id="communication_isl:a:b",
            start_s=1.25,
            end_s=10.0,
        )
        trace = base.trace
        trace.ephemeris_samples = tuple(
            dict(row, time_s=float(t))
            for t in range(21)
            for row in trace.ephemeris_samples
            if row["time_s"] == 0
        )
        net = NetworkState(
            trace,
            replace(
                base.spec,
                transport_model=mode,
                failure_policy="retry_next_sample",
                faults=(fault,),
            ),
        )
        spec = SimpleNamespace(
            serving_profile=SimpleNamespace(internal_comm_accounting="embedded"),
            time_horizon_s=20,
            routing_policy="earliest_arrival",
        )
        engine = _EventEngine(
            spec, SimpleNamespace(network=net, time=trace.time)
        )
        engine.debit = lambda *args, **kwargs: None
        engine.on_delivered = lambda job, unused: job.update(done=True)
        job = dict(request=dict(request_id="r"), path=[], ledger=[], done=False)
        engine.jobs["r"] = job
        engine.send(job, "s", "t", 1000, None, "input", "delivered")
        engine.push(1.25, -2, "fault_start", None, (0, fault))
        engine.push(10.0, -2, "fault_end", None, (0, fault))
        engine.run()
        delivered = [p for p in job["path"] if not p.get("cancelled_at_s")]
        self.assertEqual(
            [(p["source_node_id"], p["target_node_id"]) for p in delivered],
            [("s", "a"), ("a", "d"), ("d", "t")],
            engine.log[-12:],
        )
        recovery = next(e for e in engine.log if e["kind"] == "route_recovery")
        self.assertEqual(recovery["source_node_id"], "a")
        self.assertEqual(recovery["original_source_node_id"], "s")
        self.assertEqual(recovery["time_s"], 1.25)  # Continuous faults retry when observed.
        loss = next(e for e in engine.log if e["kind"] == "transfer_cancelled")
        if mode == "ns3":
            self.assertGreater(loss["bytes_discarded"], 0)
            self.assertLess(
                loss["bytes_discarded"], 250
            )  # Includes native PPP overhead.
        else:
            self.assertAlmostEqual(loss["bytes_discarded"], 250, places=5)
        self.assertEqual(loss["retained_payload_bytes"], 1000)
        self.assertTrue(
            any(
                e["kind"] == "route_plan" and e.get("reason") == "recovery"
                for e in engine.log
            )
        )

    def test_queue_middle_hop_restarts_at_buffered_node(self):
        self.verify_middle_hop_retains_delivered_prefix("queue")

    def test_fluid_middle_hop_restarts_at_buffered_node(self):
        self.verify_middle_hop_retains_delivered_prefix("fluid")

    @unittest.skipUnless(importlib.util.find_spec("ns"), "requires native ns-3")
    def test_ns3_middle_hop_restarts_at_buffered_node(self):
        self.verify_middle_hop_retains_delivered_prefix("ns3")

    @unittest.skipUnless(importlib.util.find_spec("simgrid"), "requires native SimGrid")
    def test_native_middle_hop_restarts_at_buffered_node(self):
        self.verify_middle_hop_retains_delivered_prefix("simgrid")

    def verify_long_access_recovery(self, mode):
        from astermind.engine import LinkProfile
        from tests.network_case import NetworkCase, traffic, run_traffic

        request = {**traffic(arrival=7.25), "input_bytes": 7750}
        spec = NetworkCase(traffic=(request,))
        trace = spec.environment_state
        samples = tuple(
            dict(row, time_s=float(t))
            for t in range(101)
            for row in trace.ephemeris_samples
            if row["time_s"] == 0
        )
        trace = replace(
            trace,
            ephemeris_samples=samples,
            sgl_windows=(
                *trace.sgl_windows,
                dict(
                    ingress_id="gateway-0",
                    satellite_id="sat-b",
                    start_s=0.0,
                    end_s=100.0,
                ),
            ),
        )
        nodes = tuple(
            replace(n, sgl_terminals=2) if n.node_id in {"gateway-0", "sat-b"} else n
            for n in spec.network.nodes
        )
        net = replace(
            spec.network,
            nodes=nodes,
            transport_model=mode,
            explicit_pairs=(*spec.network.explicit_pairs, ("gateway-0", "sat-b")),
            link_profiles={**spec.network.link_profiles, "sgl": LinkProfile(8000, 0)},
        )
        ref = run_traffic(replace(spec, network=net, environment_state=trace))
        initial = ref.transfer_jobs[0]["path"][0]
        self.assertAlmostEqual(initial["finish_s"], 15.0, places=6)
        fault = dict(
            target_type="link", target_id=initial["link_id"], start_s=10.0, end_s=25.0
        )
        result = run_traffic(
            replace(
                spec,
                network=replace(
                    net, failure_policy="retry_next_sample", faults=(fault,)
                ),
                environment_state=trace,
            )
        )
        self.assertEqual(result.completed, 1)
        events = result.log
        loss = next(e for e in events if e["kind"] == "transfer_cancelled")
        self.assertEqual(loss["time_s"], 10.0)
        self.assertAlmostEqual(loss["bytes_discarded"], 2750, places=5)
        retry = next(e for e in events if e["kind"] == "route_recovery")
        self.assertEqual(retry["time_s"], 10.0)
        path = result.transfer_jobs[0]["path"]
        self.assertNotEqual(path[1]["link_id"], initial["link_id"])
        self.assertEqual(path[1]["start_s"], 10.0)
        self.assertEqual(path[1]["payload_bytes"], 7750)

    def test_queue_long_transmission_uses_backup_ground_access(self):
        self.verify_long_access_recovery("queue")

    @unittest.skipUnless(importlib.util.find_spec("simgrid"), "requires native SimGrid")
    def test_native_long_transmission_uses_backup_ground_access(self):
        self.verify_long_access_recovery("simgrid")

    def verify_recovery(self, mode, target):
        job, events = execute(mode, target)
        losses = [e for e in events if e["kind"] == "transfer_cancelled"]
        self.assertEqual(len(losses), 1)
        self.assertAlmostEqual(losses[0]["bytes_discarded"], 250, places=5)
        retry = next(e for e in events if e["kind"] == "route_recovery")
        self.assertEqual(retry["time_s"], 0.25)
        self.assertEqual(retry["retransmit_bytes"], 1000)
        successful = [p for p in job["path"] if not p.get("cancelled_at_s")]
        self.assertEqual([p["target_node_id"] for p in successful], ["d", "c"])
        self.assertTrue(all(p["payload_bytes"] == 1000 for p in successful))
        self.assertEqual(len([e for e in events if e["kind"] == "delivered"]), 1)
        self.assertEqual(
            [e["time_s"] for e in events], sorted(e["time_s"] for e in events)
        )

    def test_queue_link_and_relay_node_failures(self):
        for target in ("link", "node"):
            with self.subTest(target=target):
                self.verify_recovery("queue", target)

    @unittest.skipUnless(importlib.util.find_spec("simgrid"), "requires native SimGrid")
    def test_native_link_and_relay_node_failures(self):
        for target in ("link", "node"):
            with self.subTest(target=target):
                self.verify_recovery("simgrid", target)
