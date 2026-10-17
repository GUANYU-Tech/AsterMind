"""Check physical paths, resource competition and the external clock contract."""

import copy
from pathlib import Path
import unittest

from network_sim import CommunicationRuntime, PhysicalTopology, Scenario, TransferRequest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/public_basic.json"


def fixture(capacity=8, latency=0):
    raw = Scenario.load(EXAMPLE).model_dump(mode="json")
    raw["devices"][2]["capacity_gbps"] = capacity
    for link in raw["links"]:
        link["latency_ns"] = latency
    raw["templates"][0]["latency_ns"] = latency
    return raw


def request(identifier="first", source="gpu0", target="gpu2", size=1000, ready=0, origin="test"):
    return TransferRequest(id=identifier, source=source, target=target, size_bytes=size,
                           ready_time_ns=ready, origin=origin)


class NetworkTests(unittest.TestCase):
    def runtime(self, raw=None):
        runtime = CommunicationRuntime(PhysicalTopology(Scenario.model_validate(raw or fixture())))
        self.addCleanup(runtime.close)
        return runtime

    def test_physical_path_and_isolated_formula(self):
        runtime = self.runtime(fixture(latency=10))
        route = runtime.route("gpu0", "gpu2", 1000)
        self.assertEqual(route["path"], ["gpu0", "switchA", "termA0", "termB0", "gpu2"])
        self.assertEqual(route["latency_ns"], 40)
        self.assertEqual(route["bandwidth_gbps"], 8)
        runtime.submit(request())
        event = runtime.run()[0]
        self.assertEqual(event.completion_time_ns, 1040)
        self.assertEqual(event.completion_time_ns, route["isolated_estimate_ns"])

    def test_switch_shares_across_distinct_terminal_paths(self):
        runtime = self.runtime()
        first = runtime.route("gpu0", "gpu2")
        second = runtime.route("gpu1", "gpu3")
        self.assertEqual(set(first["resources"]) & set(second["resources"]), {"device:switchA"})
        runtime.submit(request())
        runtime.submit(request("second", "gpu1", "gpu3"))
        runtime.advance_to(0)
        self.assertEqual([s["rate_gbps"] for s in runtime.snapshot()["statuses"]], [4, 4])
        self.assertEqual([e.completion_time_ns for e in runtime.run()], [2000, 2000])
        gate = next(r for r in runtime.snapshot()["resources"] if r["id"] == "device:switchA")
        self.assertEqual(gate["peak_active_flows"], 2)
        self.assertEqual(gate["bytes_transferred"], 2000)
        self.assertEqual(gate["utilization"], 1)

    def test_no_switch_cap_means_independent_ports(self):
        runtime = self.runtime(fixture(capacity=None))
        runtime.submit(request())
        runtime.submit(request("second", "gpu1", "gpu3"))
        self.assertEqual([e.completion_time_ns for e in runtime.run()], [1000, 1000])

    def test_link_sharing_without_switch_cap(self):
        runtime = self.runtime(fixture(capacity=None))
        runtime.submit(request())
        runtime.submit(request("second"))
        self.assertEqual([e.completion_time_ns for e in runtime.run()], [2000, 2000])

    def test_later_submission_keeps_prior_progress(self):
        runtime = self.runtime()
        runtime.submit(request())
        runtime.advance_to(500)
        self.assertEqual(runtime.snapshot()["statuses"][0]["remaining_bytes"], 500)
        runtime.submit(request("second", "gpu1", "gpu3", ready=500))
        runtime.advance_to(500)
        self.assertEqual([s["rate_gbps"] for s in runtime.snapshot()["statuses"]], [4, 4])
        events = runtime.run()
        self.assertEqual([(e.id, e.completion_time_ns) for e in events], [("first", 1500), ("second", 2000)])

    def test_unused_fair_share_is_reallocated(self):
        raw = fixture()
        raw["links"][0]["bandwidth_gbps"] = 2
        runtime = self.runtime(raw)
        runtime.submit(request())
        runtime.submit(request("second", "gpu1", "gpu3"))
        runtime.advance_to(0)
        self.assertEqual([s["rate_gbps"] for s in runtime.snapshot()["statuses"]], [2, 6])
        self.assertEqual([(e.id, e.completion_time_ns) for e in runtime.run()], [("second", 1334), ("first", 4000)])

    def test_device_capacity_is_shared_by_opposite_directions(self):
        runtime = self.runtime()
        runtime.submit(request())
        runtime.submit(request("reverse", "gpu2", "gpu0"))
        self.assertEqual([e.completion_time_ns for e in runtime.run()], [2000, 2000])

    def test_link_directions_have_independent_capacity(self):
        raw = fixture(capacity=None)
        runtime = self.runtime(raw)
        runtime.submit(request())
        runtime.submit(request("reverse", "gpu2", "gpu0"))
        self.assertEqual([e.completion_time_ns for e in runtime.run()], [1000, 1000])

    def test_propagation_does_not_hold_bandwidth(self):
        runtime = self.runtime(fixture(latency=10))
        runtime.submit(request(size=100))
        self.assertEqual(runtime.advance_to(100), ())
        self.assertEqual(runtime.snapshot()["statuses"][0]["state"], "propagating")
        runtime.submit(request("second", ready=100, size=100))
        events = runtime.run()
        self.assertEqual([(e.id, e.completion_time_ns) for e in events], [("first", 140), ("second", 240)])

    def test_future_ready_time_and_exactly_once_events(self):
        runtime = self.runtime()
        runtime.submit(request(ready=100))
        self.assertEqual(runtime.next_event_time_ns(), 100)
        runtime.advance_to(99)
        self.assertEqual(runtime.snapshot()["statuses"][0]["state"], "scheduled")
        runtime.advance_to(100)
        self.assertEqual(runtime.next_event_time_ns(), 1100)
        self.assertEqual(len(runtime.advance_to(1100)), 1)
        self.assertEqual(runtime.advance_to(1200), ())

    def test_origins_compete_in_one_runtime(self):
        runtime = self.runtime()
        for i, origin in enumerate(("multi_gpu", "orbitserve", "midlayer")):
            runtime.submit(request(str(i), origin=origin))
        events = runtime.run()
        self.assertEqual({e.origin for e in events}, {"multi_gpu", "orbitserve", "midlayer"})
        self.assertEqual({e.completion_time_ns for e in events}, {3000})

    def test_gpu_cannot_be_used_as_implicit_router(self):
        raw = {"cells": [{"id": "c", "satellites": ["s"]}], "satellites": [{"id": "s"}],
               "devices": [{"id": g, "satellite": "s", "kind": "gpu"} for g in ("a", "b", "c")],
               "links": [{"id": "ab", "source": "a", "target": "b", "bandwidth_gbps": 8},
                         {"id": "bc", "source": "b", "target": "c", "bandwidth_gbps": 8}]}
        runtime = self.runtime(raw)
        runtime.route("a", "b")
        with self.assertRaisesRegex(ValueError, "no route"):
            runtime.route("a", "c")

    def test_disabled_or_directional_link(self):
        raw = fixture()
        raw["links"][0]["bidirectional"] = False
        runtime = self.runtime(raw)
        runtime.route("gpu0", "gpu2")
        with self.assertRaisesRegex(ValueError, "no route"):
            runtime.route("gpu2", "gpu0")
        raw["links"][0]["enabled"] = False
        runtime = self.runtime(raw)
        with self.assertRaisesRegex(ValueError, "no route"):
            runtime.submit(request())

    def test_rank_placement_is_external(self):
        runtime = self.runtime(fixture(latency=10))
        placement = {0: "gpu0", 1: "gpu1"}
        local = runtime.route(placement[0], placement[1])["path"]
        placement[1] = "gpu2"
        remote = runtime.route(placement[0], placement[1])["path"]
        self.assertEqual(local, ["gpu0", "switchA", "gpu1"])
        self.assertIn("termA0", remote)

    def test_config_rejects_bad_references_and_limits(self):
        for mutation in ("unknown", "duplicate", "terminal_limit", "unknown_field", "negative_bw", "self"):
            with self.subTest(mutation=mutation):
                raw = fixture()
                if mutation == "unknown": raw["links"][0]["target"] = "missing"
                if mutation == "duplicate": raw["devices"].append(copy.deepcopy(raw["devices"][0]))
                if mutation == "terminal_limit": raw["devices"][3]["max_external_links"] = 0
                if mutation == "unknown_field": raw["secret_typo"] = 1
                if mutation == "negative_bw": raw["links"][0]["bandwidth_gbps"] = -1
                if mutation == "self": raw["links"][0]["target"] = raw["links"][0]["source"]
                with self.assertRaises(ValueError):
                    PhysicalTopology(Scenario.model_validate(raw))

    def test_inter_cell_exit_capability(self):
        raw = fixture()
        raw["cells"] = [{"id": "c0", "satellites": ["satA"]}, {"id": "c1", "satellites": ["satB"]}]
        with self.assertRaisesRegex(ValueError, "external_access"):
            PhysicalTopology(Scenario.model_validate(raw))
        for device in raw["devices"]:
            if device["kind"] == "terminal": device["external_access"] = True
        runtime = self.runtime(raw)
        runtime.submit(request())
        self.assertEqual(len(runtime.run()), 1)
        self.assertEqual(len(runtime.topology.view("cells")["links"]), 2)
        self.assertEqual(len(runtime.topology.view("satellites", "c0")["external_links"]), 2)

    def test_zero_bytes_and_duplicate_ids(self):
        runtime = self.runtime(fixture(latency=10))
        runtime.submit(request(size=0))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runtime.submit(request())
        self.assertEqual(runtime.run()[0].completion_time_ns, 0)

    def test_templates_and_pure_communication_satellite(self):
        raw = fixture()
        raw["cells"][0]["satellites"].append("relay")
        raw["satellites"].append({"id": "relay"})
        raw["devices"].extend([{"id": "relay0", "satellite": "relay", "kind": "terminal"},
                              {"id": "relay1", "satellite": "relay", "kind": "terminal"},
                              {"id": "relaySwitch", "satellite": "relay", "kind": "switch", "capacity_gbps": 4}])
        raw["links"][0]["target"] = "relay0"
        raw["links"].append({"id": "relayOut", "source": "relay1", "target": "termB0", "bandwidth_gbps": 8})
        raw["templates"].append({"id": "relayTemplate", "kind": "switch", "switch": "relaySwitch",
                                 "devices": ["relay0", "relay1"], "bandwidth_gbps": 8})
        runtime = self.runtime(raw)
        self.assertIn("relaySwitch", runtime.route("gpu0", "gpu2")["path"])
        runtime.submit(request())
        self.assertEqual(runtime.run()[0].completion_time_ns, 2000)

    def test_terminal_capacity_limits_endpoint_flow(self):
        raw = fixture(capacity=None)
        raw["devices"][3]["capacity_gbps"] = 4
        runtime = self.runtime(raw)
        runtime.submit(request(target="termA0"))
        self.assertEqual(runtime.run()[0].completion_time_ns, 2000)

    def test_three_views_share_identities(self):
        topology = PhysicalTopology(Scenario.load(EXAMPLE))
        self.assertEqual(len(topology.view("cells")["cells"]), 1)
        self.assertEqual(len(topology.view("satellites", "cell0")["links"]), 2)
        self.assertEqual(len(topology.view("devices", "satA")["devices"]), 5)
        self.assertEqual(topology.canonical.metadata["geometry_unspecified"], ["satA", "satB"])

    def test_fc_and_ring_are_physical_generators(self):
        for kind, count in (("fc", 6), ("ring", 4)):
            with self.subTest(kind=kind):
                raw = {"cells": [{"id": "c", "satellites": ["s"]}], "satellites": [{"id": "s"}],
                       "devices": [{"id": g, "satellite": "s", "kind": "gpu"} for g in ("a", "b", "c", "d")],
                       "templates": [{"id": "layout", "kind": kind, "devices": ["a", "b", "c", "d"],
                                      "bandwidth_gbps": 8}]}
                runtime = self.runtime(raw)
                self.assertEqual(len(runtime.topology.links), count)
                self.assertEqual(runtime.route("a", "b")["path"], ["a", "b"])
                if kind == "fc":
                    self.assertEqual(runtime.route("a", "c")["path"], ["a", "c"])
                else:
                    with self.assertRaisesRegex(ValueError, "no route"):
                        runtime.route("a", "c")

    def test_multiple_switches_and_zero_latency_cycle(self):
        raw = fixture()
        raw["devices"].append({"id": "secondSwitch", "satellite": "satA", "kind": "switch", "capacity_gbps": 4})
        raw["templates"][0]["devices"].remove("termA1")
        raw["links"].extend([
            {"id": "switches", "source": "switchA", "target": "secondSwitch", "bandwidth_gbps": 16},
            {"id": "outSecond", "source": "secondSwitch", "target": "termA1", "bandwidth_gbps": 16},
            {"id": "cycle", "source": "secondSwitch", "target": "termA0", "bandwidth_gbps": 16}])
        runtime = self.runtime(raw)
        route = runtime.route("gpu1", "gpu3")
        self.assertIn("secondSwitch", route["path"])
        self.assertEqual(route["bandwidth_gbps"], 4)
        runtime.submit(request(target="gpu3"))
        self.assertEqual(runtime.run()[0].completion_time_ns, 2000)

    def test_clock_and_request_validation(self):
        runtime = self.runtime()
        runtime.advance_to(100)
        with self.assertRaises(ValueError): runtime.advance_to(99)
        with self.assertRaises(ValueError): runtime.submit(request())
        with self.assertRaises(ValueError): runtime.route("gpu0", "gpu0")
        with self.assertRaises(ValueError): request(size=-1)
        with self.assertRaises(ValueError): request(size=1.5)


if __name__ == "__main__":
    unittest.main()
