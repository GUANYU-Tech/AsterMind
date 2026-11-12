"""Persistent routes must avoid searches without freezing transport timing."""

import unittest
from dataclasses import replace

from astermind.engine import NetworkState, NetworkUnavailable
from tests.unit.runtime.test_dynamic_network import network


def diamond(*, contact_end=20.0, extra=False):
    edges = [
        ("a", "b", 0.0, 20.0), ("b", "d", 0.0, contact_end),
        ("a", "c", 0.0, 20.0), ("c", "d", 0.0, 20.0),
    ]
    if extra:
        edges.append(("e", "f", 0.0, 5.0))
    return network(tuple(edges), bandwidth=8000.0)


def nodes(path):
    return [leg["target_node_id"] for leg in path]


class PersistentRoutingTests(unittest.TestCase):
    def test_repeated_flows_and_forwarding_suffixes_search_only_once(self):
        state = diamond()
        state.transfer("a", "d", 0.0, 100.0)
        for index in range(100):
            at = index / 100.0
            _, path = state.transfer("a", "d", at, 100.0)
            self.assertEqual(nodes(path), ["b", "d"])
            self.assertAlmostEqual(path[0]["start_s"], at)
            _, suffix = state.transfer("b", "d", at, 100.0, avoid_nodes={"a"})
            self.assertEqual(nodes(suffix), ["d"])
        self.assertEqual(state.route_cache_info()["searches"], 1)
        self.assertEqual(state.route_cache_info()["hits"], 200)

    def test_contention_changes_timing_but_not_the_cached_path(self):
        state = diamond()
        _, first = state.transfer("a", "d", 0.0, 100.0)
        link_id = first[0]["link_id"]
        state.shared_load[(link_id, "a", "b")] = 3
        # Use shared-bandwidth forecasting, as the serving backends do.
        state.spec = replace(state.spec, transport_model="runtime_comm")
        arrival, reused = state.transfer("a", "d", 1.0, 100.0)
        self.assertEqual(nodes(reused), ["b", "d"])
        self.assertAlmostEqual(arrival, 1.5)
        self.assertEqual(state.route_cache_info()["searches"], 1)
        _, fresh = state.transfer("a", "d", 1.0, 100.0, recompute=True)
        self.assertEqual(nodes(fresh), ["c", "d"])
        self.assertEqual(nodes(state.transfer("a", "d", 2.0, 100.0)[1]), ["b", "d"])

    def test_unrelated_contact_change_keeps_the_route(self):
        state = diamond(extra=True)
        state.transfer("a", "d", 0.0, 100.0)
        self.assertEqual(nodes(state.transfer("a", "d", 5.0, 100.0)[1]), ["b", "d"])
        self.assertEqual(state.route_cache_info()["searches"], 1)
        self.assertEqual(state.route_cache_info()["invalidations"], 0)

    def test_new_shortcut_does_not_preempt_an_unchanged_route(self):
        state = network(
            (("a", "b", 0.0, 20.0), ("b", "d", 0.0, 20.0),
             ("a", "d", 5.0, 20.0)), bandwidth=8000.0,
        )
        self.assertEqual(nodes(state.transfer("a", "d", 0.0, 100.0)[1]), ["b", "d"])
        self.assertEqual(nodes(state.transfer("a", "d", 5.0, 100.0)[1]), ["b", "d"])
        self.assertEqual(state.route_cache_info()["searches"], 1)
        _, fresh = state.transfer("a", "d", 5.0, 100.0, recompute=True)
        self.assertEqual(nodes(fresh), ["d"])

    def test_wait_and_link_kind_restrictions_are_not_shared_cache_entries(self):
        state = network((("a", "b", 5.0, 20.0),), bandwidth=8000.0)
        state.transfer("a", "b", 0.0, 100.0)
        with self.assertRaises(NetworkUnavailable):
            state.transfer("a", "b", 0.0, 100.0, allow_wait=False)
        with self.assertRaises(NetworkUnavailable):
            state.transfer("a", "b", 6.0, 100.0, kinds=set())

    def test_changed_contact_replans_only_the_affected_route(self):
        state = diamond(contact_end=5.0, extra=True)
        state.transfer("a", "d", 0.0, 100.0)
        state.transfer("a", "c", 0.0, 100.0)
        _, route = state.transfer("a", "d", 5.0, 100.0)
        self.assertEqual(nodes(route), ["c", "d"])
        state.transfer("a", "c", 5.0, 100.0)
        self.assertEqual(state.route_cache_info()["searches"], 3)
        self.assertEqual(state.route_cache_info()["invalidations"], 1)

    def test_changed_geometry_replans_at_the_sample_boundary(self):
        original = diamond()
        original.trace.ephemeris_samples = tuple(
            {"satellite_id": node, "time_s": at,
             "position_itrs_km": (7010.0 if node == "b" and at >= 5 else 7000.0, 0, 0)}
            for at in (0.0, 5.0, 20.0) for node in "abcd"
        )
        state = NetworkState(original.trace, original.spec)
        self.assertEqual(nodes(state.transfer("a", "d", 0.0, 100.0)[1]), ["b", "d"])
        self.assertEqual(nodes(state.transfer("a", "d", 4.0, 100.0)[1]), ["b", "d"])
        self.assertEqual(nodes(state.transfer("a", "d", 5.0, 100.0)[1]), ["c", "d"])
        self.assertEqual(state.route_cache_info()["searches"], 2)

    def test_capacity_change_refreshes_timing_and_invalidates_path(self):
        state = network(bandwidth=8000.0, schedule=((5.0, 0.5),))
        self.assertAlmostEqual(state.transfer("a", "b", 0.0, 100.0)[0], 0.1)
        self.assertAlmostEqual(state.transfer("a", "b", 5.0, 100.0)[0], 5.2)
        self.assertEqual(state.route_cache_info()["searches"], 2)
        self.assertEqual(state.route_cache_info()["invalidations"], 1)

    def test_fault_and_fork_do_not_poison_the_parent_route(self):
        state = diamond()
        state.transfer("a", "d", 0.0, 100.0)
        state.transfer("a", "c", 0.0, 100.0)
        child = state.fork()
        fault = {"target_type": "node", "target_id": "b", "start_s": 1.0, "end_s": 10.0}
        child.observe_fault(0, fault, 1.0)
        self.assertEqual(nodes(child.transfer("a", "d", 1.0, 100.0)[1]), ["c", "d"])
        child.transfer("a", "c", 1.0, 100.0)
        self.assertEqual(child.route_cache_info()["searches"], 1)
        self.assertEqual(nodes(state.transfer("a", "d", 1.0, 100.0)[1]), ["b", "d"])
        self.assertEqual(state.route_cache_info()["invalidations"], 0)

    def test_avoid_nodes_cannot_be_bypassed_by_a_cache_hit(self):
        state = diamond()
        state.transfer("a", "d", 0.0, 100.0)
        _, route = state.transfer("a", "d", 1.0, 100.0, avoid_nodes={"b"})
        self.assertEqual(nodes(route), ["c", "d"])
        with self.assertRaises(NetworkUnavailable):
            state.transfer("a", "d", 1.0, 100.0, avoid_nodes={"b", "c"})

    def test_long_payload_can_fall_back_when_cached_contact_will_not_fit(self):
        state = diamond(contact_end=5.0)
        state.transfer("a", "d", 0.0, 100.0)
        arrival, route = state.transfer("a", "d", 4.0, 1000.0)
        self.assertEqual(nodes(route), ["c", "d"])
        self.assertAlmostEqual(arrival, 6.0)
        self.assertEqual(state.route_cache_info()["replay_failures"], 1)

    def test_reservations_are_fresh_and_preview_calendars_are_isolated(self):
        state = network(bandwidth=8000.0)
        state.transfer("a", "b", 0.0, 100.0)
        preview = state.fork()
        preview.transfer("a", "b", 0.0, 100.0, commit=True)
        self.assertFalse(state.calendars)
        self.assertAlmostEqual(state.transfer("a", "b", 0.0, 100.0, commit=True)[0], 0.1)
        self.assertAlmostEqual(state.transfer("a", "b", 0.0, 100.0, commit=True)[0], 0.2)
        self.assertAlmostEqual(state.transfer("b", "a", 0.0, 100.0, commit=True)[0], 0.1)
        self.assertEqual(state.route_cache_info()["searches"], 2)
