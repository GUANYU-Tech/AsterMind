"""Causality regressions for the request-driven event scheduler."""

import unittest
from dataclasses import replace

from orbitserve.runtime import (
    RuntimeStrategySpec,
    NetworkSpec,
    NetworkNodeSpec,
    LinkProfile,
    run_simulation,
)
from tests.unit.runtime.test_runtime_engine import _scenario, _request, _trace


def strategy():
    return RuntimeStrategySpec(
        latency_weight=1.0,
        kv_reuse_weight=0.0,
        energy_weight=0.0,
        compute_load_weight=0.0,
    )


class RuntimeEventOrderTests(unittest.TestCase):
    def test_overlapping_request_cannot_read_future_kv(self):
        spec = _scenario(
            strategy(),
            requests=(
                _request("a", 0),
                _request("b", 0.1, context_tokens_before_turn=10),
            ),
        )
        result = run_simulation(spec)
        self.assertTrue(all(r["outcome"]["completed"] for r in result.request_records))
        self.assertEqual(result.request_records[1]["kv"]["kv_hit_tokens"], 0)
        log = result.state_summary["event_log"]
        publish = next(
            e["time_s"]
            for e in log
            if e["kind"] == "kv_publish" and e["request_id"] == "a"
        )
        self.assertGreater(publish, 0.1)
        self.assertEqual([e["time_s"] for e in log], sorted(e["time_s"] for e in log))

    def test_completed_kv_is_visible_to_later_request(self):
        result = run_simulation(
            _scenario(
                strategy(),
                requests=(
                    _request("a", 0),
                    _request("b", 2, context_tokens_before_turn=10),
                ),
            )
        )
        self.assertEqual(result.request_records[1]["kv"]["kv_hit_tokens"], 10)

    def test_ready_order_not_request_arrival_order_controls_compute(self):
        first = _request("slow-input", 0)
        first["input_bytes"] = 8000
        later = _request("fast-input", 0.5, lon_deg=90, session_id="other")
        later.update(input_bytes=1)
        spec = _scenario(strategy(), requests=(first, later))
        network = NetworkSpec(
            (
                NetworkNodeSpec("sat-a", "communication", 2, 1),
                NetworkNodeSpec("sat-b", "compute", 2, 1),
                NetworkNodeSpec("gateway-0", "ground", sgl_terminals=1),
                NetworkNodeSpec("gateway-1", "ground", sgl_terminals=1),
            ),
            {
                "communication_isl": LinkProfile(1e9, 0),
                "cross_layer": LinkProfile(1e9, 0),
                "sgl": LinkProfile(8000, 0),
            },
            communication_topology="explicit",
            ground_roles=(),
            explicit_pairs=(
                ("sat-a", "sat-b"),
                ("gateway-0", "sat-a"),
                ("gateway-1", "sat-b"),
            ),
        )
        spec = replace(
            spec,
            network=network,
            serving_profile=replace(
                spec.serving_profile,
                max_concurrency=1,
                prefill_rate_table=tuple(
                    {**r, "prefill_tokens_per_s": 1.0}
                    for r in spec.serving_profile.prefill_rate_table
                ),
                decode_rate_table=tuple(
                    {**r, "decode_tokens_per_s": 1.0}
                    for r in spec.serving_profile.decode_rate_table
                ),
            ),
        )
        result = run_simulation(spec)
        rows = {r["request"]["request_id"]: r for r in result.request_records}
        self.assertTrue(all(r["outcome"]["completed"] for r in rows.values()))
        self.assertLess(rows["fast-input"]["provenance"]["compute_start_s"], 1.0)
        self.assertGreaterEqual(
            rows["slow-input"]["provenance"]["compute_start_s"],
            rows["fast-input"]["provenance"]["compute_complete_s"],
        )

    def test_request_time_is_not_rounded_to_topology_sample(self):
        trace = _trace()
        trace = replace(
            trace,
            trace_manifest=replace(
                trace.trace_manifest, time={**trace.time, "step_s": 60.0}
            ),
        )
        result = run_simulation(
            _scenario(strategy(), requests=(_request("fractional", 7.25),), trace=trace)
        )
        arrivals = [
            e
            for e in result.state_summary["event_log"]
            if e["kind"] == "request_arrival"
        ]
        self.assertEqual(arrivals[0]["time_s"], 7.25)
        self.assertLess(result.request_records[0]["provenance"]["compute_start_s"], 8.0)
        self.assertEqual(
            result.provenance["runtime_engine"], "chronological_discrete_event_runtime"
        )

    def test_causal_turn_releases_after_actual_response_completion(self):
        second = _request("second", 90, context_tokens_before_turn=10)
        second.update(
            arrival_time_mode="after_dependency_completion",
            depends_on_request_id="first",
            inter_turn_gap_s=0.25,
        )
        result = run_simulation(
            _scenario(strategy(), requests=(_request("first", 7.25), second))
        )
        first, later = result.request_records
        expected = 7.25 + first["timing"]["e2e_s"] + 0.25
        self.assertAlmostEqual(later["request"]["arrival_time_s"], expected)
        self.assertEqual(later["kv"]["kv_hit_tokens"], 10)
        self.assertLess(later["request"]["arrival_time_s"], 90)

    def test_dependency_cycle_is_rejected(self):
        a, b = _request("a", 0), _request("b", 0)
        a.update(
            arrival_time_mode="after_dependency_completion",
            depends_on_request_id="b",
            inter_turn_gap_s=0,
        )
        b.update(
            arrival_time_mode="after_dependency_completion",
            depends_on_request_id="a",
            inter_turn_gap_s=0,
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            run_simulation(_scenario(strategy(), requests=(a, b)))

    def cell_scenario(self, outage=False):
        spec = _scenario(strategy(), requests=(_request("cell-request", 0.25),))
        network = NetworkSpec(
            (
                NetworkNodeSpec(
                    "sat-a", "compute", 2, 1, cell_id="cell", cell_ingress=True
                ),
                NetworkNodeSpec(
                    "sat-b", "compute", 2, 1, cell_id="cell", compute_target=False
                ),
                NetworkNodeSpec("gateway-0", "ground", sgl_terminals=1),
                NetworkNodeSpec("gateway-1", "ground", sgl_terminals=1),
            ),
            {
                "cell_isl": LinkProfile(
                    1e9, 3.0, capacity_schedule=((3.5, 0.0),) if outage else ()
                ),
                "sgl": LinkProfile(1e9, 0.0),
            },
            communication_topology="explicit",
            ground_roles=("compute",),
        )
        return replace(spec, network=network)

    def test_cell_waits_for_physical_links_and_debits_both_members(self):
        result = run_simulation(self.cell_scenario())
        row = result.request_records[0]
        self.assertTrue(row["outcome"]["completed"])
        self.assertEqual(row["provenance"]["compute_start_s"], 3.0)
        self.assertEqual(
            set(row["placement"]["compute_owner_satellite_ids"]), {"sat-a", "sat-b"}
        )
        self.assertAlmostEqual(row["energy"]["compute_energy_j"], 12.0)
        members = result.state_summary["compute_nodes"]
        self.assertTrue(
            any(e["compute_j"] > 0 for e in members["sat-b"]["energy_events"])
        )
        self.assertTrue(
            any(
                e["kind"] == "compute_wait_for_cell"
                for e in result.state_summary["event_log"]
            )
        )

    def test_internal_link_outage_cancels_compute_without_publishing_kv(self):
        result = run_simulation(self.cell_scenario(outage=True))
        self.assertFalse(result.request_records[0]["outcome"]["completed"])
        self.assertEqual(
            result.request_records[0]["outcome"]["uncompleted_reason"],
            "compute_cell_disconnected_during_service",
        )
        self.assertFalse(
            any(e["kind"] == "kv_publish" for e in result.state_summary["event_log"])
        )

    def test_duplicate_request_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            run_simulation(
                _scenario(
                    strategy(), requests=(_request("same", 0), _request("same", 1))
                )
            )

    def test_failure_propagates_to_causal_dependents(self):
        first = _request("first", 0)
        first["input_bytes"] = 1e20
        later = _request("dependent", 1)
        later.update(
            arrival_time_mode="after_dependency_completion",
            depends_on_request_id="first",
            inter_turn_gap_s=0,
        )
        spec = _scenario(strategy(), requests=(first, later))
        result = run_simulation(spec)
        self.assertFalse(any(r["outcome"]["completed"] for r in result.request_records))
        self.assertEqual(
            result.request_records[1]["outcome"]["uncompleted_reason"],
            "dependency_uncompleted",
        )

    def test_runtime_cannot_silently_select_sampled_time_mode(self):
        with self.assertRaisesRegex(ValueError, "event_driven"):
            replace(_scenario(strategy(), requests=()), time_mode="time_sliced")

    def test_cell_capacity_drop_at_exact_compute_finish_does_not_cancel_completed_work(
        self,
    ):
        spec = self.cell_scenario()
        first = run_simulation(spec).request_records[0]
        finish = first["provenance"]["compute_complete_s"]
        profiles = dict(spec.network.link_profiles)
        profiles["cell_isl"] = replace(
            profiles["cell_isl"], capacity_schedule=((finish, 0.0),)
        )
        result = run_simulation(
            replace(spec, network=replace(spec.network, link_profiles=profiles))
        )
        self.assertTrue(
            any(e["kind"] == "kv_publish" for e in result.state_summary["event_log"])
        )

    def test_transfer_lifecycle_has_causal_identity_and_real_link_wait(self):
        first, second = _request("first", 7.25), _request("second", 7.25)
        first["input_bytes"] = second["input_bytes"] = 100000
        spec = _scenario(strategy(), requests=(first, second))
        profiles = dict(spec.network.link_profiles)
        profiles["sgl"] = replace(profiles["sgl"], bandwidth_bps=800000)
        result = run_simulation(
            replace(spec, network=replace(spec.network, link_profiles=profiles))
        )
        self.assertEqual(result.request_summary["completed_request_count"], 2)
        lifecycle = {}
        for event in result.state_summary["event_log"]:
            if event["kind"] in {
                "transfer_queued",
                "transmit_start",
                "transmit_complete",
                "hop_arrival",
            }:
                lifecycle.setdefault(event["transfer_id"], []).append(event)
        self.assertTrue(lifecycle)
        waits = []
        for events in lifecycle.values():
            self.assertEqual(
                [e["kind"] for e in events],
                [
                    "transfer_queued",
                    "transmit_start",
                    "transmit_complete",
                    "hop_arrival",
                ],
            )
            queued, start, finish, arrival = events
            self.assertEqual(queued["time_s"], queued["ready_s"])
            self.assertAlmostEqual(
                start["time_s"] - queued["time_s"], queued["queue_wait_s"]
            )
            self.assertAlmostEqual(
                finish["time_s"] - start["time_s"], queued["transmission_s"]
            )
            self.assertAlmostEqual(
                arrival["time_s"] - finish["time_s"], queued["propagation_delay_s"]
            )
            self.assertLessEqual(queued["time_s"], start["time_s"])
            self.assertLessEqual(finish["time_s"], arrival["time_s"])
            self.assertEqual(len({e["link_id"] for e in events}), 1)
            waits.append(queued["queue_wait_s"])
        self.assertGreater(max(waits), 0)
        self.assertFalse(
            any(
                e["kind"] == "transfer_dropped"
                for e in result.state_summary["event_log"]
            )
        )
