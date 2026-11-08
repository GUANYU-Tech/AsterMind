"""Check acquisition reservations, policy differences and real flow interruption."""
import unittest

from network_sim import CommunicationRuntime, PhysicalTopology, Scenario, TransferRequest
from network_sim.physical.link_policy import LinkController
from network_sim.physical._vendor.orbitserve_link_selection import _degree_limited_edges, _edge_sort_key
from test_network import fixture, request


def dynamic(raw=None, initial=20, reacquisition=50, interval=100, mode="simple_keep"):
    raw = fixture() if raw is None else raw
    raw["link_policy"] = dict(mode=mode, check_interval_ns=interval,
                              initial_acquisition_ns=initial, reacquisition_ns=reacquisition)
    return raw


def partner_fixture(mode="simple_keep"):
    raw = dict(cells=[dict(id="c", satellites=["a", "b", "c"])],
               satellites=[dict(id=i) for i in "abc"],
               devices=[dict(id="t"+i, satellite=i, kind="terminal") for i in "abc"],
               links=[dict(id="ab", source="ta", target="tb", bandwidth_gbps=8,
                           distance_km=10, availability_windows=[dict(start_ns=0, end_ns=250)]),
                      dict(id="ac", source="ta", target="tc", bandwidth_gbps=8,
                           distance_km=1, availability_windows=[dict(start_ns=100, end_ns=1000)])])
    return dynamic(raw, mode=mode)


class LinkPolicyTests(unittest.TestCase):
    def runtime(self, raw):
        runtime = CommunicationRuntime(PhysicalTopology(Scenario.model_validate(raw)))
        self.addCleanup(runtime.close)
        return runtime

    def controller(self, raw):
        return LinkController(PhysicalTopology(Scenario.model_validate(raw)))

    def test_keep_old_link_then_fill_free_terminal_at_next_check(self):
        controller = self.controller(partner_fixture())
        self.assertEqual(controller.states["ab"].ready_ns, 20)
        controller.advance_to(200)
        self.assertEqual(controller.states["ab"].state, "up")
        self.assertEqual(controller.states["ac"].state, "down")
        controller.advance_to(250)
        self.assertEqual(controller.states["ab"].state, "down")
        self.assertEqual(controller.next_event_time_ns(), 300)
        controller.advance_to(300)
        self.assertEqual(controller.states["ac"].ready_ns, 350)
        controller.advance_to(350)
        self.assertEqual(controller.states["ac"].state, "up")
        self.assertEqual([h["reason"] for h in controller.history],
                         ["initial", "acquisition_complete", "visibility_loss", "reacquisition", "acquisition_complete"])

    def test_capture_occupies_slot_and_check_does_not_reset_timer(self):
        raw = partner_fixture()
        raw["link_policy"]["initial_acquisition_ns"] = 220
        controller = self.controller(raw)
        controller.advance_to(200)
        self.assertEqual(controller.states["ab"].ready_ns, 220)
        self.assertEqual(controller.states["ac"].state, "down")
        controller.advance_to(220)
        self.assertEqual(controller.states["ab"].state, "up")

    def test_canceled_initial_capture_is_still_initial_next_time(self):
        raw = dynamic(initial=80, interval=100)
        raw["links"][0]["availability_windows"] = [dict(start_ns=0, end_ns=40), dict(start_ns=100, end_ns=500)]
        controller = self.controller(raw)
        controller.advance_to(100)
        self.assertEqual(controller.states["isl0"].ready_ns, 180)
        controller.advance_to(180)
        self.assertEqual(controller.states["isl0"].state, "up")

    def test_os_reference_reranks_instead_of_holding_old_link(self):
        raw = partner_fixture("os_reference")
        raw["link_policy"]["os_terminal_count"] = 1
        controller = self.controller(raw)
        controller.advance_to(100)
        self.assertEqual(controller.states["ab"].state, "down")
        self.assertEqual(controller.states["ac"].ready_ns, 150)
        candidates = (("a", "b", {"link_id": "ab"}, 10), ("a", "c", {"link_id": "ac"}, 1))
        selected = _degree_limited_edges(tuple(sorted(candidates, key=_edge_sort_key)), 1)
        self.assertEqual(selected[0][2]["link_id"], "ac")

    def test_terminal_limits_are_active_reservations_not_candidate_count(self):
        raw = partner_fixture()
        raw["devices"][0]["max_external_links"] = 0
        controller = self.controller(raw)
        controller.advance_to(200)
        self.assertTrue(all(s.state == "down" for s in controller.states.values()))

    def test_os_walker_reference_preserves_rings_and_matches_each_boundary(self):
        raw = dict(cells=[dict(id="cell", satellites=list("abcd"))],
                   satellites=[dict(id=i) for i in "abcd"],
                   devices=[dict(id="t"+i, satellite=i, kind="terminal", max_external_links=2) for i in "abcd"],
                   links=[dict(id=a+b, source="t"+a, target="t"+b, bandwidth_gbps=8, distance_km=d)
                          for a,b,d in (("a","b",20),("c","d",20),("a","c",3),("a","d",1),("b","c",2),("b","d",4))])
        raw = dynamic(raw, initial=0, mode="os_reference")
        raw["link_policy"].update(os_terminal_count=2, os_layout=dict(
            plane_by_node=dict(a=0,b=0,c=1,d=1), same_plane_neighbor_pairs=[["a","b"],["c","d"]],
            adjacent_plane_pairs=[[0,1]]))
        controller = self.controller(raw)
        self.assertEqual({i for i,s in controller.states.items() if s.state == "up"}, {"ab","cd","ad","bc"})

    def test_wait_for_capture_then_transmit_without_repeated_cost(self):
        runtime = self.runtime(dynamic(initial=200, interval=100))
        runtime.submit(request(size=100))
        self.assertEqual(runtime.snapshot()["statuses"][0]["state"], "waiting_link")
        with self.assertRaisesRegex(ValueError, "no route"):
            runtime.route("gpu0", "gpu2")
        self.assertIn("termA0", runtime.candidate_path("gpu0", "gpu2"))
        runtime.advance_to(199)
        self.assertEqual(runtime.snapshot()["statuses"][0]["remaining_bytes"], 100)
        events = runtime.advance_to(300)
        self.assertEqual(events[0].completion_time_ns, 300)
        runtime.submit(request("second", ready=300, size=100))
        self.assertEqual(runtime.advance_to(400)[0].completion_time_ns, 400)
        self.assertEqual(runtime.advance_to(450), ())

    def test_loss_fails_inflight_flow_and_releases_shared_switch_capacity(self):
        raw = dynamic(initial=0, interval=1000)
        raw["links"][0]["availability_windows"] = [dict(start_ns=0, end_ns=500)]
        runtime = self.runtime(raw)
        runtime.submit(request())
        runtime.submit(request("survivor", source="gpu1", target="gpu3"))
        events = runtime.advance_to(1250)
        self.assertEqual([(e.id, e.kind) for e in events], [("first", "failed"), ("survivor", "completed")])
        self.assertEqual(events[0].failure_time_ns, 500)
        self.assertEqual(events[0].remaining_bytes, 750)
        self.assertEqual(events[1].completion_time_ns, 1250)
        self.assertEqual(runtime.advance_to(1400), ())
        self.assertEqual(runtime.snapshot()["statuses"][0]["state"], "failed")

    def test_reacquired_link_does_not_silently_resume_failed_flow(self):
        raw = dynamic(initial=0, reacquisition=50, interval=100)
        raw["links"][0]["availability_windows"] = [dict(start_ns=0, end_ns=250), dict(start_ns=300, end_ns=3000)]
        runtime = self.runtime(raw)
        runtime.submit(request())
        failed = runtime.advance_to(300)
        self.assertEqual(failed[0].failure_time_ns, 250)
        runtime.submit(request("retry", ready=300, size=100))
        done = runtime.advance_to(500)
        self.assertEqual([(e.id, e.completion_time_ns) for e in done], [("retry", 450)])
        self.assertEqual(runtime.snapshot()["statuses"][0]["state"], "failed")

    def test_future_ready_task_uses_route_at_start_not_submission(self):
        raw = dynamic(initial=0, interval=100)
        raw["links"][0]["availability_windows"] = [dict(start_ns=0, end_ns=50), dict(start_ns=200, end_ns=1000)]
        runtime = self.runtime(raw)
        runtime.submit(request(ready=100, size=100))
        runtime.advance_to(100)
        self.assertEqual(runtime.snapshot()["statuses"][0]["state"], "waiting_link")
        self.assertEqual(runtime.advance_to(350)[0].completion_time_ns, 350)

    def test_completion_at_loss_boundary_wins_and_propagation_loss_fails(self):
        raw = dynamic(initial=0)
        raw["links"][0]["availability_windows"] = [dict(start_ns=0, end_ns=100)]
        runtime = self.runtime(raw)
        runtime.submit(request(size=100))
        self.assertEqual(runtime.advance_to(100)[0].kind, "completed")
        raw = dynamic(fixture(latency=10), initial=0)
        raw["links"][0]["availability_windows"] = [dict(start_ns=0, end_ns=110)]
        runtime = self.runtime(raw)
        runtime.submit(request(size=100))
        event = runtime.advance_to(200)[0]
        self.assertEqual(event.kind, "failed")
        self.assertEqual(event.failure_time_ns, 110)
        self.assertEqual(event.remaining_bytes, 0)
        self.assertEqual(runtime.advance_to(300), ())

    def test_custom_selector_and_disabled_candidate(self):
        raw = dynamic(initial=0)
        raw["links"][0]["enabled"] = False
        runtime = self.runtime(raw)
        self.assertNotIn("isl0", runtime.controller.states)
        with self.assertRaisesRegex(ValueError, "no route"):
            runtime.submit(request())
        topology = PhysicalTopology(Scenario.model_validate(dynamic(initial=0)))
        with CommunicationRuntime(topology, link_selector=lambda candidates, old, now: ["isl1"]) as runtime:
            self.assertEqual(runtime.controller.states["isl0"].state, "down")
            self.assertEqual(runtime.controller.states["isl1"].state, "up")

    def test_dynamic_inter_cell_multi_terminal_sharing(self):
        raw = dynamic(initial=200)
        raw["cells"] = [dict(id="c0", satellites=["satA"]), dict(id="c1", satellites=["satB"])]
        for device in raw["devices"]:
            if device["kind"] == "terminal":
                device["external_access"] = True
        runtime = self.runtime(raw)
        runtime.submit(request())
        runtime.submit(request("kv", source="gpu1", target="gpu3", origin="orbitserve"))
        events = runtime.advance_to(2200)
        self.assertEqual({e.completion_time_ns for e in events}, {2200})
        gate = next(r for r in runtime.snapshot()["resources"] if r["id"] == "device:switchA")
        self.assertEqual(gate["peak_active_flows"], 2)

    def test_dynamic_requires_horizon_and_checks_input_windows(self):
        runtime = self.runtime(dynamic())
        with self.assertRaisesRegex(ValueError, "finite|until_ns"):
            runtime.run()
        for bad in (0, -1, 1.5, True):
            with self.subTest(interval=bad):
                with self.assertRaises(ValueError):
                    PhysicalTopology(Scenario.model_validate(dynamic(interval=bad)))
        raw = fixture()
        raw["links"][0]["availability_windows"] = [dict(start_ns=0, end_ns=5)]
        with self.assertRaisesRegex(ValueError, "dynamic"):
            PhysicalTopology(Scenario.model_validate(raw))
        raw = dynamic()
        raw["links"][0]["availability_windows"] = [dict(start_ns=10, end_ns=20), dict(start_ns=15, end_ns=30)]
        with self.assertRaises(ValueError):
            Scenario.model_validate(raw)


if __name__ == "__main__":
    unittest.main()
