import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from orbitserve.runtime import (
    LinkProfile,
    NetworkNodeSpec,
    NetworkSpec,
    NetworkState,
    NetworkUnavailable,
    RuntimeStrategySpec,
    candidate_links,
)


def layout(topology="grid", planes=4, slots=6, **kwargs):
    nodes = tuple(
        NetworkNodeSpec(f"c-{p}-{s}", "communication", 4, plane=p, slot=s)
        for p in range(planes)
        for s in range(slots)
    )
    return NetworkSpec(
        nodes,
        {"communication_isl": LinkProfile(1e9, 2)},
        communication_topology=topology,
        **kwargs,
    )


def network(
    edges=(("a", "b", 0.0, 20.0),),
    *,
    setup=0.0,
    bandwidth=8.0,
    schedule=(),
    terminals=3,
    positions=None,
    limit=None,
):
    ids = sorted({n for a, b, _, _ in edges for n in (a, b)})
    spec = NetworkSpec(
        tuple(NetworkNodeSpec(n, "compute", terminals) for n in ids),
        {"communication_isl": LinkProfile(bandwidth, setup, limit, schedule)},
        communication_topology="explicit",
        explicit_pairs=tuple(sorted({tuple(sorted((a, b))) for a, b, _, _ in edges})),
    )
    rows = []
    for t in (0.0, 20.0):
        for n in ids:
            pos = positions[n][int(t > 0)] if positions else (7000.0, 0.0, 0.0)
            rows.append({"satellite_id": n, "time_s": t, "position_itrs_km": pos})
    trace = SimpleNamespace(
        time={"start_s": 0.0, "end_s": 20.0, "step_s": 20.0},
        satellite_catalog=tuple({"satellite_id": n} for n in ids),
        ephemeris_samples=rows,
        sgl_windows=(),
        isl_windows=tuple(
            dict(endpoint_a=a, endpoint_b=b, start_s=start, end_s=end)
            for a, b, start, end in edges
        ),
    )
    return NetworkState(trace, spec)


class DynamicNetworkTests(unittest.TestCase):
    def test_hop_replanning_does_not_return_to_an_already_visited_node(self):
        state = network(
            edges=(
                ("a", "b", 0.0, 20.0),
                ("a", "d", 0.0, 20.0),
                ("b", "c", 0.0, 20.0),
                ("c", "d", 0.0, 20.0),
            )
        )
        _, path = state.transfer("b", "d", 0.0, 1.0, avoid_nodes={"a"})
        self.assertEqual([leg["target_node_id"] for leg in path], ["c", "d"])
        with self.assertRaises(NetworkUnavailable):
            state.transfer("b", "d", 0.0, 1.0, avoid_nodes={"a", "c"})

    def test_unrelated_events_do_not_duplicate_blocked_link_history(self):
        state = network(
            edges=(("a", "b", 0.0, 20.0), ("c", "d", 3.0, 8.0)), terminals=0
        )
        link = next(edge for edge in state.edges if edge["endpoint_a"] == "a")
        self.assertEqual(len(state.link_frames[link["link_id"]]), 1)
        self.assertEqual(
            state.link_frames[link["link_id"]][0]["reason"], "terminal_unavailable"
        )

    def test_complete_snapshots_include_blocked_links_and_final_boundary(self):
        state = network(edges=(("a", "b", 5.0, 12.0),))
        before = state.snapshot(0)["links"][0]
        self.assertEqual(before["state"], "DOWN")
        self.assertEqual(before["reason"], "geometry_blocked")
        self.assertEqual(before["attributes_time_s"], 0)
        end = state.snapshot(20)
        self.assertEqual(len(end["nodes"]), 2)
        self.assertEqual(end["links"][0]["reason"], "horizon_end")
        with self.assertRaises(NetworkUnavailable):
            state.snapshot(20.01)

    def test_radio_link_budget_distance_and_hardware_cap(self):
        profile = LinkProfile(
            1e12,
            0.0,
            capacity_model="free_space_shannon",
            channel_bandwidth_hz=1e6,
            frequency_hz=2e10,
            transmit_power_w=1.0,
            transmit_gain_db=10.0,
            receive_gain_db=10.0,
        )
        near, far = profile.attributes(1000.0, 0.0), profile.attributes(2000.0, 0.0)
        self.assertAlmostEqual(
            far["path_loss_db"] - near["path_loss_db"], 20 * math.log10(2)
        )
        self.assertLess(far["bandwidth_bps"], near["bandwidth_bps"])
        self.assertLessEqual(
            replace(profile, bandwidth_bps=10.0).attributes(1.0, 0.0)["bandwidth_bps"],
            10.0,
        )
        with self.assertRaisesRegex(ValueError, "requires positive"):
            LinkProfile(1e9, 0.0, capacity_model="free_space_shannon")

    def test_sun_exclusion_checks_direction_not_satellite_eclipse(self):
        original = network(
            positions={"a": ((7000.0, 0, 0),) * 2, "b": ((7001.0, 0, 0),) * 2}
        )
        spec = replace(
            original.spec,
            link_profiles={
                "communication_isl": LinkProfile(8.0, 0.0, sun_exclusion_angle_deg=5.0)
            },
        )
        state = NetworkState(
            original.trace,
            spec,
            sun_positions=(
                {"time_s": 0.0, "position_itrs_km": (1.5e8, 0, 0)},
                {"time_s": 20.0, "position_itrs_km": (1.5e8, 0, 0)},
            ),
        )
        self.assertEqual(state.snapshot(0)["links"][0]["reason"], "sun_exclusion")
        with self.assertRaises(NetworkUnavailable):
            state.transfer("a", "b", 0, 1)

    def test_ground_access_nearest_and_longest_available_choose_different_links(self):
        trace = SimpleNamespace(
            time={"start_s": 0.0, "end_s": 20.0, "step_s": 20.0},
            satellite_catalog=({"satellite_id": "a"}, {"satellite_id": "b"}),
            ephemeris_samples=tuple(
                {"satellite_id": n, "time_s": t, "position_itrs_km": (x, 0.0, 0.0)}
                for t in (0.0, 20.0)
                for n, x in (("a", 7000.0), ("b", 7100.0))
            ),
            isl_windows=(),
            sgl_windows=tuple(
                dict(ingress_id="g", satellite_id=n, start_s=0.0, end_s=end)
                for n, end in (("a", 5.0), ("b", 20.0))
            ),
        )
        spec = NetworkSpec(
            (
                NetworkNodeSpec("a", "compute", sgl_terminals=1),
                NetworkNodeSpec("b", "compute", sgl_terminals=1),
                NetworkNodeSpec("g", "ground", sgl_terminals=1),
            ),
            {"sgl": LinkProfile(1e9, 1.0)},
            communication_topology="explicit",
            ground_roles=("compute",),
        )
        for policy, target in (("nearest_distance", "a"), ("longest_available", "b")):
            state = NetworkState(
                trace,
                replace(spec, ground_access_policy=policy),
                ({"ingress_id": "g", "lat_deg": 0.0, "lon_deg": 0.0},),
            )
            up = [e for e in state.snapshot(1)["links"] if e["state"] == "UP"]
            self.assertEqual(len(up), 1)
            self.assertIn(target, (up[0]["endpoint_a"], up[0]["endpoint_b"]))

    def test_grid_and_honeycomb_degree_and_edge_count(self):
        for topology, degree in (("grid", 4), ("honeycomb", 3)):
            spec = layout(topology)
            edges = candidate_links(spec)
            counts = {n.node_id: 0 for n in spec.nodes}
            for e in edges:
                counts[e["endpoint_a"]] += 1
                counts[e["endpoint_b"]] += 1
            self.assertEqual(set(counts.values()), {degree})
            self.assertEqual(len(edges), len(spec.nodes) * degree // 2)
            self.assertEqual(
                candidate_links(replace(spec, nodes=tuple(reversed(spec.nodes)))), edges
            )

    def test_honeycomb_is_bipartite_without_triangular_neighbors(self):
        spec = layout("honeycomb")
        nodes = {n.node_id: n for n in spec.nodes}
        for e in candidate_links(spec):
            a, b = nodes[e["endpoint_a"]], nodes[e["endpoint_b"]]
            self.assertNotEqual((a.plane + a.slot) % 2, (b.plane + b.slot) % 2)

    def test_invalid_honeycomb_wrap_rejected_and_missing_slot_left_empty(self):
        with self.assertRaisesRegex(ValueError, "even"):
            candidate_links(layout("honeycomb", planes=3))
        spec = replace(layout(), nodes=layout().nodes[:-1])
        links = candidate_links(spec)
        self.assertEqual(len(links), 44)
        self.assertFalse(any('c-3-5' in (e['endpoint_a'], e['endpoint_b']) for e in links))

    def test_setup_is_not_rounded_to_geometry_step_or_repeated_per_request(self):
        state = network(setup=2.0)
        self.assertEqual(state.snapshot(0)["links"][0]["state"], "SETTING_UP")
        self.assertEqual(state.snapshot(2)["links"][0]["state"], "UP")
        self.assertEqual(state.transfer("a", "b", 0, 1)[0], 3.0)
        self.assertEqual(state.transfer("a", "b", 3, 1)[0], 4.0)
        self.assertEqual(
            len([e for e in state.events if e["state"] == "SETTING_UP"]), 1
        )

    def test_visibility_shorter_than_setup_and_reconnect(self):
        state = network((("a", "b", 0, 1), ("a", "b", 5, 20)), setup=2.0)
        self.assertEqual(state.snapshot(1)["links"][0]["state"], "DOWN")
        self.assertEqual(state.transfer("a", "b", 0, 1)[0], 8.0)
        self.assertEqual(
            [e["time_s"] for e in state.events if e["state"] == "UP"], [7.0]
        )

    def test_terminal_limit_and_release_at_exact_contact_boundary(self):
        state = network((("a", "b", 0, 4), ("a", "c", 0, 20)), setup=1.0, terminals=1)
        before = state.snapshot(2)["links"]
        self.assertEqual(sum(e["state"] == "UP" for e in before), 1)
        self.assertEqual(state.transfer("a", "c", 4, 1)[0], 6.0)

    def test_piecewise_capacity_is_integrated(self):
        state = network(bandwidth=16.0, schedule=((0.0, 0.5), (1.0, 1.0)))
        done, path = state.transfer("a", "b", 0, 3)
        self.assertEqual(done, 2.0)
        self.assertEqual([s["bytes"] for s in path[0]["segments"]], [1.0, 2.0])

    def test_zero_capacity_pauses_and_resumes(self):
        state = network(schedule=((1.0, 0.0), (3.0, 1.0)))
        self.assertEqual(state.transfer("a", "b", 0, 2)[0], 4.0)

    def test_full_duplex_contention_preview_isolation_and_conservation(self):
        state = network()
        preview = state.fork()
        preview.transfer("a", "b", 0, 2, commit=True)
        self.assertFalse(state.calendars)
        state.transfer("a", "b", 0, 2, commit=True)
        self.assertEqual(state.transfer("a", "b", 0, 2, commit=True)[0], 4.0)
        self.assertEqual(state.transfer("b", "a", 0, 2, commit=True)[0], 2.0)
        self.assertEqual(len(state.calendars), 2)

    def test_short_contact_does_not_hide_viable_alternative(self):
        state = network(
            (("a", "b", 0, 1), ("b", "d", 0, 20), ("a", "c", 0, 20), ("c", "d", 0, 20))
        )
        done, path = state.transfer("a", "d", 0, 2)
        self.assertEqual(done, 4.0)
        self.assertEqual([leg["target_node_id"] for leg in path], ["c", "d"])

    def test_range_changes_only_when_the_sampled_position_changes(self):
        state = network(
            positions={
                "a": ((7000.0, 0, 0), (7000.0, 0, 0)),
                "b": ((7000.0, 0, 0), (7020.0, 0, 0)),
            },
            limit=5.0,
        )
        for t in (0.0, 4.0, 5.0, 19.999):
            self.assertEqual(state.snapshot(t)["links"][0]["state"], "UP")
            self.assertEqual(state.position("b", t), (7000.0, 0.0, 0.0))
            self.assertEqual(state.snapshot(t)["links"][0]["attributes_time_s"], 0.0)
        self.assertEqual(state.position("b", 20.0), (7020.0, 0.0, 0.0))
        self.assertFalse(any(0.0 < e["time_s"] < 20.0 for e in state.events))

    def test_one_second_snapshots_hold_positions_and_rf_attributes(self):
        original = network()
        trace = original.trace
        trace.ephemeris_samples = tuple(
            {
                "satellite_id": n,
                "time_s": t,
                "position_itrs_km": (7000.0 + (t if n == "b" else 0.0), 0.0, 0.0),
            }
            for t in range(21)
            for n in ("a", "b")
        )
        state = NetworkState(trace, original.spec)
        for t in (7.0, 7.25, 7.999999):
            self.assertEqual(state.position("b", t), (7007.0, 0.0, 0.0))
            self.assertEqual(state.snapshot(t)["links"][0]["distance_km"], 7.0)
        self.assertEqual(state.position("b", 8.0), (7008.0, 0.0, 0.0))

    def test_transfer_across_one_second_boundary_preserves_bytes_and_fractional_finish(
        self,
    ):
        original = network(
            bandwidth=800.0,
            schedule=(
                (0.0, 0.5),
                (1.0, 1.0),
            ),
        )
        original.trace.ephemeris_samples = tuple(
            {"satellite_id": n, "time_s": t, "position_itrs_km": (7000.0, 0.0, 0.0)}
            for t in range(21)
            for n in ("a", "b")
        )
        state = NetworkState(original.trace, original.spec)
        done, path = state.transfer("a", "b", 0.5, 100.0)
        self.assertAlmostEqual(done, 1.75)
        self.assertAlmostEqual(sum(p["bytes"] for p in path[0]["segments"]), 100.0)
        self.assertEqual(
            [(p["start_s"], p["finish_s"]) for p in path[0]["segments"]],
            [(0.5, 1.0), (1.0, 1.75)],
        )

    def test_unchanged_links_do_not_retain_one_attribute_dictionary_per_sample(self):
        original = network()
        original.trace.ephemeris_samples = tuple(
            {"satellite_id": n, "time_s": t, "position_itrs_km": (7000.0 + t, 0.0, 0.0)}
            for t in range(21)
            for n in ("a", "b")
        )
        state = NetworkState(original.trace, original.spec)
        self.assertEqual(sum(len(rows) for rows in state.link_frames.values()), 1)
        self.assertEqual(len(state.frames), 20)

    def test_horizon_nan_and_removed_mode_fail_closed(self):
        state = network()
        for t in (-1.0, 20.0, float("nan")):
            with self.assertRaises(NetworkUnavailable):
                state.transfer("a", "b", t, 1)
        with self.assertRaises(ValueError):
            state.transfer("a", "b", 0, float("nan"))
        with self.assertRaises(ValueError):
            RuntimeStrategySpec(ingress_selection="communication_satellite")

    def test_cell_members_and_cross_layer_ingress_are_explicit(self):
        members = tuple(
            NetworkNodeSpec(
                f"m-{i}",
                "compute",
                5,
                cell_id="cell",
                cell_ingress=i == 0,
                compute_target=i == 0,
            )
            for i in range(4)
        )
        comm = NetworkNodeSpec("relay", "communication", 5)
        profiles = {k: LinkProfile(1e9, 1) for k in ("cell_isl", "cross_layer")}
        ring = NetworkSpec(
            (*members, comm), profiles, communication_topology="explicit"
        )
        edges = candidate_links(ring)
        self.assertEqual(sum(e["kind"] == "cell_isl" for e in edges), 4)
        self.assertEqual(sum(e["kind"] == "cross_layer" for e in edges), 1)
        self.assertEqual(
            sum(
                e["kind"] == "cell_isl"
                for e in candidate_links(replace(ring, cell_topology="fully_connected"))
            ),
            6,
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            replace(
                ring, nodes=tuple(replace(n, compute_target=True) for n in ring.nodes)
            )
