"""Physical-role and energy gates must hold before any request is dispatched."""

from dataclasses import replace
import unittest

from orbitserve.runtime import LinkProfile, run_simulation
from orbitserve.runtime.engine.simulator import (
    _build_energy_trace,
    _preview_candidate,
    _select_access,
)
from orbitserve.runtime.state.runtime_state import RuntimeState
from tests.unit.runtime.test_runtime_engine import _request, _scenario, _trace
from tests.unit.runtime import test_runtime_event_order as event_fixtures
from tests.unit.runtime.test_runtime_event_order import strategy


class CandidateSafetyTests(unittest.TestCase):
    def test_network_role_cannot_override_explicit_non_compute_hardware(self):
        trace = _trace()
        trace = replace(
            trace,
            satellite_catalog=tuple(
                {**row, "compute_capable": False} for row in trace.satellite_catalog
            ),
        )
        result = run_simulation(_scenario(strategy(), trace=trace))
        self.assertEqual(result.request_summary["completed_request_count"], 0)
        self.assertEqual(
            result.request_records[0]["outcome"]["uncompleted_reason"],
            "eligible_compute_set_empty",
        )

    def test_ineligible_compute_node_can_still_relay_to_healthy_target(self):
        trace = _trace()
        trace = replace(
            trace,
            satellite_catalog=tuple(
                {**row, "compute_capable": row["satellite_id"] == "sat-b"}
                for row in trace.satellite_catalog
            ),
        )
        result = run_simulation(_scenario(strategy(), trace=trace))
        self.assertEqual(result.request_summary["completed_request_count"], 1)
        row = result.request_records[0]
        self.assertEqual(row["decision"]["serving_compute_node_id"], "sat-b")
        self.assertEqual(row["routing"]["selected_path"][0]["target_node_id"], "sat-a")

    def test_catalog_fault_flags_still_exclude_compute_targets(self):
        trace = _trace()
        trace = replace(
            trace,
            satellite_catalog=tuple(
                {**row, "failed": True} for row in trace.satellite_catalog
            ),
        )
        result = run_simulation(_scenario(strategy(), trace=trace))
        self.assertEqual(result.request_summary["completed_request_count"], 0)

    def relay_spec(self, **energy_overrides):
        spec = _scenario(strategy())
        nodes = tuple(
            replace(n, role="communication") if n.node_id == "sat-a" else n
            for n in spec.network.nodes
        )
        profiles = {**spec.network.link_profiles, "cross_layer": LinkProfile(1e9, 0)}
        profile = {**spec.energy_profile, "charge_rate_w": 0.0, **energy_overrides}
        return replace(
            spec,
            network=replace(spec.network, nodes=nodes, link_profiles=profiles),
            energy_profile={**spec.energy_profile, "node_profiles": {"sat-a": profile}},
        )

    def assert_rejected_before_dispatch(self, result):
        self.assertEqual(result.request_summary["completed_request_count"], 0)
        self.assertEqual(
            result.request_records[0]["outcome"]["uncompleted_reason"],
            "no_feasible_compute_candidate",
        )
        self.assertFalse(
            any(
                e["kind"] in {"transmit_start", "compute_start", "kv_publish"}
                for e in result.state_summary["event_log"]
            )
        )
        self.assertTrue(
            all(
                not node["energy_events"]
                for node in result.state_summary["compute_nodes"].values()
            )
        )

    def test_low_soc_relay_is_rejected_before_dispatch(self):
        self.assert_rejected_before_dispatch(
            run_simulation(self.relay_spec(initial_soc=0.05))
        )

    def test_relay_that_cannot_afford_predicted_link_bytes_is_rejected(self):
        self.assert_rejected_before_dispatch(
            run_simulation(
                self.relay_spec(
                    initial_soc=0.8,
                    battery_capacity_j=100.0,
                    link_energy_j_per_byte=1.0,
                )
            )
        )

    def test_non_owner_cell_member_energy_is_part_of_candidate_gate(self):
        spec = event_fixtures.RuntimeEventOrderTests().cell_scenario()
        profile = {**spec.energy_profile, "initial_soc": 0.05, "charge_rate_w": 0.0}
        result = run_simulation(
            replace(
                spec,
                energy_profile={
                    **spec.energy_profile,
                    "node_profiles": {"sat-b": profile},
                },
            )
        )
        self.assert_rejected_before_dispatch(result)

    def test_healthy_relay_still_completes_with_actual_link_energy(self):
        result = run_simulation(self.relay_spec(initial_soc=0.8))
        self.assertEqual(result.request_summary["completed_request_count"], 1)
        self.assertTrue(
            any(
                e["link_bytes_accounted"] > 0
                for e in result.state_summary["compute_nodes"]["sat-a"]["energy_events"]
            )
        )

    def test_preview_does_not_mutate_live_energy_or_queue_state(self):
        spec = self.relay_spec(initial_soc=0.8)
        from types import SimpleNamespace
        from orbitserve.runtime.state.network import NetworkState

        trace = spec.environment_state
        environment = SimpleNamespace(
            **{
                key: getattr(trace, key)
                for key in (
                    "satellite_catalog",
                    "time",
                    "ephemeris_samples",
                    "sgl_windows",
                    "isl_windows",
                )
            },
            network=NetworkState(trace, spec.network, spec.gateways),
        )
        energy, initial = _build_energy_trace(trace, spec.energy_profile)
        state = RuntimeState()
        for row in trace.satellite_catalog:
            state.compute_node_for(
                row["satellite_id"], initial_soc=initial, max_concurrency=2
            )
        before = state.to_dict()
        access = _select_access(
            state=state,
            request=_request(),
            simulation_spec=spec,
            environment_state=environment,
            energy_trace=energy,
            eligible_satellite_ids=("sat-b",),
            sgl_bandwidth_bps=0,
            horizon_s=100,
        )
        forecast = _preview_candidate(
            state=state,
            request=_request(),
            strategy=spec.strategy,
            environment_state=environment,
            energy_trace=energy,
            serving=spec.serving_profile,
            access=access,
            compute_satellite_id="sat-b",
            isl_bandwidth_bps=0,
        )
        self.assertTrue(forecast["candidate"].feasible)
        self.assertEqual(state.to_dict(), before)

    def test_runtime_lru_eviction_forces_context_recomputation(self):
        spec = _scenario(
            strategy(),
            requests=(
                _request("a", 0, session_id="a"),
                _request("b", 3, session_id="b"),
                _request("a-again", 6, session_id="a", context_tokens_before_turn=10),
            ),
        )
        result = run_simulation(
            replace(
                spec,
                serving_profile=replace(
                    spec.serving_profile, kv_cache_capacity_tokens=12
                ),
            )
        )
        self.assertEqual(result.request_summary["completed_request_count"], 3)
        self.assertEqual(result.request_records[2]["kv"]["kv_hit_tokens"], 0)
        self.assertEqual(
            result.request_records[2]["kv"]["context_recompute_tokens"], 10
        )
        cache = result.state_summary["kv_caches"]["sat-a"]
        self.assertLessEqual(cache["used_tokens"], cache["capacity_tokens"])
