import sys
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.runtime.routing import TraceQueryResult
from orbitserve.runtime.contracts import RuntimeStrategySpec
from orbitserve.runtime.deployment_contracts import RuntimeDeployment, RuntimeDeploymentCatalog
from orbitserve.runtime.engines.trace_backed import run_trace_backed_runtime
from orbitserve.runtime.runtime_metrics import aggregate_runtime_records


def _link(link_id, kind, endpoint_a, endpoint_b, propagation_delay_s, bandwidth_bps=1_000_000.0):
    return {
        "link_id": link_id,
        "kind": kind,
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "bandwidth_bps": bandwidth_bps,
        "propagation_delay_s": propagation_delay_s,
        "setup_delay_s": 0.0,
        "bidirectional": True,
        "edge_available_external": True,
    }


def _request(request_id="req-0", arrival_time_s=0.0, **overrides):
    row = {
        "request_id": request_id,
        "arrival_time_s": arrival_time_s,
        "ingress_id": "ingress",
        "session_id": "session-0",
        "prompt_tokens": 10,
        "output_tokens": 4,
        "input_bytes": 1_000.0,
        "output_bytes": 400.0,
        "ttft_deadline_s": 100.0,
        "e2e_deadline_s": 200.0,
    }
    row.update(overrides)
    return row


class FakePDTraceBundle:
    def __init__(self):
        self.trace_model = "surrogate_orbit_trace"
        self.time = {"duration_s": 1_000.0, "step_s": 10.0}
        self.warnings = ()
        self.satellite_catalog = (
            {"satellite_id": "sat-a"},
            {"satellite_id": "sat-b"},
            {"satellite_id": "sat-c"},
        )
        self.access_windows = (
            {
                "ingress_id": "ingress",
                "satellite_id": "sat-a",
                "window_id": "access-a",
                "start_s": 0.0,
                "end_s": 1_000.0,
                "sgl_link_id": "sgl:ingress:sat-a",
                "propagation_delay_s": 0.001,
                "bandwidth_bps": 1_000_000.0,
            },
        )
        self.links = (
            _link("sgl:ingress:sat-a", "sgl", "ingress", "sat-a", 0.001),
            _link("isl:sat-a:sat-b", "isl", "sat-a", "sat-b", 0.010),
            _link("isl:sat-b:sat-c", "isl", "sat-b", "sat-c", 0.020),
        )
        self.physical_links = self.links
        self._links_by_id = {row["link_id"]: row for row in self.links}
        self._active_link_ids = tuple(row["link_id"] for row in self.links)
        self.satellite_initial_soc = {"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8}
        self.energy_windows = tuple(
            {
                "satellite_id": sat["satellite_id"],
                "start_s": 0.0,
                "end_s": 1_000.0,
                "sunlit": True,
                "dvfs_multiplier": 1.0,
                "charge_rate_w": 0.0,
                "idle_power_w": 0.0,
                "compute_energy_j_per_token": 0.0,
                "link_energy_j_per_byte": 0.0,
                "battery_capacity_j": 10_000_000.0,
                "soc_min": 0.2,
                "energy_accounting_mode": "event_incremental_v1",
            }
            for sat in self.satellite_catalog
        )

    def access_at(self, ingress_id, time_s):
        windows = tuple(
            window
            for window in self.access_windows
            if window["ingress_id"] == ingress_id and window["start_s"] <= time_s < window["end_s"]
        )
        return TraceQueryResult(
            available=bool(windows),
            reason=None if windows else "no_access_window",
            sample_time_s=time_s,
            value=windows,
        )

    def active_links_at(self, time_s):
        return TraceQueryResult(
            available=True,
            sample_time_s=time_s,
            value={
                "active_link_ids": self._active_link_ids,
                "links": tuple(self._links_by_id[link_id] for link_id in self._active_link_ids),
            },
        )

    def link_sample_at(self, link_id, time_s):
        link = self._links_by_id[link_id]
        return TraceQueryResult(
            available=True,
            sample_time_s=time_s,
            value={
                "time_s": time_s,
                "link_id": link_id,
                "B_raw_bps": link["bandwidth_bps"],
                "tau_prop_s": link["propagation_delay_s"],
                "tau_setup_s": link["setup_delay_s"],
                "edge_available_external": True,
            },
        )


class FlakyKVTraceBundle(FakePDTraceBundle):
    def active_links_at(self, time_s):
        if 1.04 < time_s < 1.20:
            active_link_ids = tuple(
                link_id for link_id in self._active_link_ids
                if link_id != "isl:sat-b:sat-c"
            )
        else:
            active_link_ids = self._active_link_ids
        return TraceQueryResult(
            available=True,
            sample_time_s=time_s,
            value={
                "active_link_ids": active_link_ids,
                "links": tuple(self._links_by_id[link_id] for link_id in active_link_ids),
            },
        )


class FlakyRemainingResponseTraceBundle(FakePDTraceBundle):
    def __init__(self):
        super().__init__()
        self._zero_time_sgl_samples = 0
        self._execution_started = False

    def link_sample_at(self, link_id, time_s):
        if link_id == "sgl:ingress:sat-a" and abs(time_s) < 1e-9:
            self._zero_time_sgl_samples += 1
            if self._zero_time_sgl_samples >= 4:
                self._execution_started = True
        if self._execution_started and link_id == "isl:sat-b:sat-c" and time_s > 2.0:
            return TraceQueryResult(
                available=False,
                reason="link_unavailable:remaining_response",
                sample_time_s=time_s,
                warnings=("link_unavailable:remaining_response",),
            )
        return super().link_sample_at(link_id, time_s)


def _pd_catalog(
    kv_cache_bytes_per_token=256.0,
    decode_rate=4.0,
    kv_cache_capacity_tokens=1_000_000,
    placement_scope="same_plane_cross_cluster",
    pair_scope="same_plane_cross_cluster",
    pair_rule="same_plane_cross_cluster",
):
    internal_comm_accounting = "externalized_pd_kv"
    return RuntimeDeploymentCatalog(
        deployments=(
            RuntimeDeployment(
                deployment_id="deployment_0",
                role="colocated",
                member_satellites=("sat-b", "sat-c"),
                entry_satellite_ids=("sat-b",),
                prefill_tokens_per_s=10.0,
                decode_tokens_per_s=decode_rate,
                pd_mode="disaggregated",
                placement_model="aggregate_logical_pool_compat",
            ),
            RuntimeDeployment(
                deployment_id="deployment_0:prefill:0",
                role="prefill",
                member_satellites=("sat-b",),
                entry_satellite_ids=("sat-b",),
                placement_scope=placement_scope,
                prefill_tokens_per_s=10.0,
                per_group_prefill_tokens_per_s=10.0,
                kv_cache_bytes_per_token=kv_cache_bytes_per_token,
                pd_mode="disaggregated",
                placement_model="explicit_pd_deployment_group",
                internal_comm_accounting=internal_comm_accounting,
                plane_index=0,
                cell_index=0,
                paired_deployment_ids=("deployment_0:decode:0",),
                paired_deployment_scopes={"deployment_0:decode:0": pair_scope},
                paired_deployment_rules={"deployment_0:decode:0": pair_rule},
            ),
            RuntimeDeployment(
                deployment_id="deployment_0:decode:0",
                role="decode",
                member_satellites=("sat-c",),
                entry_satellite_ids=("sat-c",),
                placement_scope=placement_scope,
                decode_tokens_per_s=decode_rate,
                per_group_decode_tokens_per_s=decode_rate,
                kv_cache_bytes_per_token=kv_cache_bytes_per_token,
                kv_cache_capacity_tokens=kv_cache_capacity_tokens,
                pd_mode="disaggregated",
                placement_model="explicit_pd_deployment_group",
                internal_comm_accounting=internal_comm_accounting,
                plane_index=0,
                cell_index=1,
                paired_deployment_ids=("deployment_0:prefill:0",),
                paired_deployment_scopes={"deployment_0:prefill:0": pair_scope},
                paired_deployment_rules={"deployment_0:prefill:0": pair_rule},
            ),
        ),
        placement_model="explicit_pd_deployment_groups",
        selection_policy="fixed_default_deployment_with_explicit_pd_groups",
    )


def _pd_catalog_two_decodes(
    kv_cache_bytes_per_token=256.0,
    kv_cache_capacity_tokens=1_000_000,
    far_kv_cache_capacity_tokens=None,
):
    catalog = _pd_catalog(
        kv_cache_bytes_per_token=kv_cache_bytes_per_token,
        kv_cache_capacity_tokens=kv_cache_capacity_tokens,
        placement_scope="hybrid_explicit_pairs",
        pair_scope="same_plane_cross_cluster",
        pair_rule="user_explicit",
    )
    prefill = catalog.deployments[1]
    decode_near = RuntimeDeployment(
        **{
            **catalog.deployments[2].to_dict(),
            "deployment_id": "deployment_0:decode:near",
            "member_satellites": ("sat-b",),
            "entry_satellite_ids": ("sat-b",),
            "cell_index": 0,
            "paired_deployment_ids": ("deployment_0:prefill:0",),
            "paired_deployment_scopes": {"deployment_0:prefill:0": "same_plane_cross_cluster"},
            "paired_deployment_rules": {"deployment_0:prefill:0": "user_explicit"},
        }
    )
    decode_far = RuntimeDeployment(
        **{
            **catalog.deployments[2].to_dict(),
            "deployment_id": "deployment_0:decode:far",
            "member_satellites": ("sat-c",),
            "entry_satellite_ids": ("sat-c",),
            "cell_index": 1,
            "kv_cache_capacity_tokens": (
                kv_cache_capacity_tokens
                if far_kv_cache_capacity_tokens is None
                else far_kv_cache_capacity_tokens
            ),
            "paired_deployment_ids": ("deployment_0:prefill:0",),
            "paired_deployment_scopes": {"deployment_0:prefill:0": "same_plane_cross_cluster"},
            "paired_deployment_rules": {"deployment_0:prefill:0": "user_explicit"},
        }
    )
    prefill = RuntimeDeployment(
        **{
            **prefill.to_dict(),
            "paired_deployment_ids": ("deployment_0:decode:near", "deployment_0:decode:far"),
            "paired_deployment_scopes": {
                "deployment_0:decode:near": "same_plane_cross_cluster",
                "deployment_0:decode:far": "same_plane_cross_cluster",
            },
            "paired_deployment_rules": {
                "deployment_0:decode:near": "user_explicit",
                "deployment_0:decode:far": "user_explicit",
            },
        }
    )
    return RuntimeDeploymentCatalog(
        deployments=(catalog.deployments[0], prefill, decode_near, decode_far),
        placement_model="explicit_pd_deployment_groups",
        selection_policy="fixed_default_deployment_with_explicit_pd_groups",
    )


class PDTraceBackedRuntimeTests(unittest.TestCase):
    def test_pd_dvfs_disabled_ignores_shadow_trace_multiplier(self):
        bundle = FakePDTraceBundle()
        bundle.energy_windows = tuple(
            {
                **window,
                "sunlit": False,
                "dvfs_multiplier": 2.0,
                "compute_energy_j_per_token": 1.0,
                "battery_capacity_j": 1_000.0,
            }
            for window in bundle.energy_windows
        )

        record = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "shortest_path",
                    "max_hop_count": 2,
                    "dvfs_modes_enabled": [],
                    "enabled_actions": ["local_compute", "cross_satellite_forward"],
                }
            ),
            trace_bundle=bundle,
        ).records[0]

        self.assertAlmostEqual(record.prefill_service_s, 1.0)
        self.assertAlmostEqual(record.decode_service_s, 1.0)
        self.assertEqual(record.prefill_action_type, "shadow_local_compute")
        self.assertEqual(record.decode_action_type, "shadow_local_compute")
        self.assertNotIn("shadow_dvfs", record.shadow_strategies)
        self.assertEqual(record.provenance["prefill_raw_dvfs_multiplier"], 2.0)
        self.assertEqual(record.provenance["prefill_effective_dvfs_multiplier"], 1.0)

    def test_pd_dvfs_enabled_controls_latency_energy_and_metrics(self):
        bundle = FakePDTraceBundle()
        bundle.energy_windows = tuple(
            {
                **window,
                "sunlit": False,
                "dvfs_multiplier": 2.0,
                "compute_energy_j_per_token": 1.0,
                "battery_capacity_j": 1_000.0,
            }
            for window in bundle.energy_windows
        )
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "shortest_path",
                    "max_hop_count": 2,
                    "dvfs_modes_enabled": ["shadow_dvfs"],
                    "hard_gate_overrides": {"shadow_dvfs_compute_energy_scale": 0.5},
                }
            ),
            trace_bundle=bundle,
        )
        record = result.records[0]
        metrics = aggregate_runtime_records(
            result.records,
            time_horizon_s=1_000.0,
            trace_model=result.trace_model,
            scope=result.scope,
        ).runtime_metrics

        self.assertAlmostEqual(record.prefill_service_s, 2.0)
        self.assertAlmostEqual(record.decode_service_s, 2.0)
        self.assertEqual(record.prefill_action_type, "shadow_dvfs")
        self.assertEqual(record.decode_action_type, "shadow_dvfs")
        self.assertIn("shadow_dvfs", record.shadow_strategies)
        self.assertEqual(metrics["shadow_dvfs_request_count"], 1)

    def test_pd_same_satellite_accumulates_both_compute_phases_and_zero_handoff_energy(self):
        bundle = FakePDTraceBundle()
        bundle.energy_windows = tuple(
            {
                **window,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
                "battery_capacity_j": 100.0,
            }
            for window in bundle.energy_windows
        )
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog_two_decodes(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=bundle,
        )
        record = result.records[0]

        self.assertEqual(record.prefill_entry_satellite_id, "sat-b")
        self.assertEqual(record.decode_entry_satellite_id, "sat-b")
        self.assertAlmostEqual(record.soc_after, 0.66)
        self.assertAlmostEqual(record.provenance["predicted_soc_after"], 0.66)
        self.assertFalse(
            any(row.get("energy_stage") == "pd_handoff" for row in record.energy_ledger)
        )
        self.assertEqual(record.provenance["hard_gate_results"]["energy_soc"]["prefill"]["satellite_id"], "sat-b")
        self.assertEqual(record.provenance["hard_gate_results"]["energy_soc"]["decode"]["satellite_id"], "sat-b")

    def test_pd_energy_gate_keeps_prefill_and_decode_thresholds_independent(self):
        bundle = FakePDTraceBundle()
        bundle.satellite_initial_soc = {"sat-a": 0.8, "sat-b": 0.30, "sat-c": 0.80}
        bundle.energy_windows = tuple(
            {
                **window,
                "compute_energy_j_per_token": 0.0,
                "link_energy_j_per_byte": 0.0,
                "soc_min": 0.20 if window["satellite_id"] == "sat-b" else 0.70,
            }
            for window in bundle.energy_windows
        )

        record = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=bundle,
        ).records[0]

        self.assertTrue(record.accepted)
        gate = record.provenance["hard_gate_results"]["energy_soc"]
        self.assertTrue(gate["passed"])
        self.assertAlmostEqual(gate["prefill"]["soc_after"], 0.30)
        self.assertAlmostEqual(gate["decode"]["soc_after"], 0.80)

    def test_pd_multimember_compute_energy_uses_explicit_weights_and_gates_non_entry(self):
        base = _pd_catalog_two_decodes()
        prefill = replace(
            base.prefill_deployments[0],
            member_satellites=("sat-b", "sat-c"),
            entry_satellite_ids=("sat-b",),
            member_energy_weights={"sat-b": 0.25, "sat-c": 0.75},
            paired_deployment_ids=("deployment_0:decode:near",),
        )
        decode = replace(
            next(item for item in base.decode_deployments if item.deployment_id.endswith(":near")),
            member_energy_weights={"sat-b": 1.0},
        )
        catalog = RuntimeDeploymentCatalog(
            deployments=(base.deployments[0], prefill, decode),
            placement_model="explicit_pd_deployment_groups",
        )
        bundle = FakePDTraceBundle()
        bundle.satellite_initial_soc = {"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8}
        bundle.energy_windows = tuple(
            {
                **window,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
                "battery_capacity_j": 100.0,
            }
            for window in bundle.energy_windows
        )

        record = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=bundle,
        ).records[0]
        prefill_rows = [row for row in record.energy_ledger if row.get("energy_stage") == "prefill"]
        prefill_j = {row["satellite_id"]: row["compute_bus_j"] for row in prefill_rows}
        self.assertAlmostEqual(prefill_j["sat-b"], 2.5)
        self.assertAlmostEqual(prefill_j["sat-c"], 7.5)
        self.assertAlmostEqual(sum(prefill_j.values()), 10.0)
        self.assertEqual(record.provenance["prefill_member_energy_weights"], {"sat-b": 0.25, "sat-c": 0.75})

        low_bundle = FakePDTraceBundle()
        low_bundle.satellite_initial_soc = {"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.21}
        low_bundle.energy_windows = tuple(
            {
                **window,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
                "battery_capacity_j": 100.0,
                "soc_min": 0.2,
            }
            for window in low_bundle.energy_windows
        )
        rejected = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=low_bundle,
        ).records[0]
        self.assertTrue(rejected.rejected)
        self.assertEqual(rejected.reject_reason, "no_feasible_scheduler_candidate")

    def test_pd_multimember_non_entry_dvfs_sets_critical_path_and_metrics(self):
        base = _pd_catalog_two_decodes()
        prefill = replace(
            base.prefill_deployments[0],
            member_satellites=("sat-b", "sat-c"),
            entry_satellite_ids=("sat-b",),
            member_energy_weights={"sat-b": 0.5, "sat-c": 0.5},
            paired_deployment_ids=("deployment_0:decode:near",),
        )
        decode = next(item for item in base.decode_deployments if item.deployment_id.endswith(":near"))
        catalog = RuntimeDeploymentCatalog(
            deployments=(base.deployments[0], prefill, decode),
            placement_model="explicit_pd_deployment_groups",
        )
        bundle = FakePDTraceBundle()
        bundle.energy_windows = tuple(
            {
                **window,
                "sunlit": window["satellite_id"] != "sat-c",
                "dvfs_multiplier": 2.0 if window["satellite_id"] == "sat-c" else 1.0,
                "compute_energy_j_per_token": 1.0,
                "battery_capacity_j": 1_000.0,
            }
            for window in bundle.energy_windows
        )
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 2,
                    "dvfs_modes_enabled": ["shadow_dvfs"],
                    "hard_gate_overrides": {"shadow_dvfs_compute_energy_scale": 0.5},
                }
            ),
            trace_bundle=bundle,
        )
        record = result.records[0]
        metrics = aggregate_runtime_records(
            result.records,
            time_horizon_s=1_000.0,
            trace_model=result.trace_model,
            scope=result.scope,
        ).runtime_metrics

        self.assertAlmostEqual(record.prefill_service_s, 2.0)
        self.assertEqual(record.prefill_action_type, "shadow_dvfs")
        self.assertEqual(
            record.provenance["prefill_member_actions"],
            {"sat-b": "local_compute", "sat-c": "shadow_dvfs"},
        )
        self.assertIn("prefill:sat-c:shadow_dvfs", record.action_components)
        self.assertIn("shadow_dvfs", record.shadow_strategies)
        self.assertEqual(metrics["shadow_dvfs_request_count"], 1)
        self.assertEqual(record.provenance["selection_score_terms"]["shadow_dvfs_indicator"], 1.0)

    def test_pd_prediction_and_commit_include_idle_gap(self):
        bundle = FakePDTraceBundle()
        bundle.satellite_initial_soc = {"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8}
        bundle.energy_windows = tuple(
            {
                **window,
                "battery_capacity_j": 100.0,
                "idle_power_w": 1.0 if window["satellite_id"] in {"sat-b", "sat-c"} else 0.0,
                "compute_energy_j_per_token": 0.0,
                "link_energy_j_per_byte": 0.0,
                "soc_min": 0.1,
            }
            for window in bundle.energy_windows
        )
        record = run_trace_backed_runtime(
            requests=(_request(arrival_time_s=50.0),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=bundle,
        ).records[0]

        self.assertLess(record.soc_after, 0.31)
        self.assertAlmostEqual(record.provenance["predicted_soc_after"], record.soc_after, places=5)

    def test_pd_shadow_offload_is_visible_to_metrics(self):
        bundle = FakePDTraceBundle()
        bundle.energy_windows = tuple(
            {
                **window,
                "sunlit": window["satellite_id"] != "sat-a",
                "dvfs_multiplier": 2.0 if window["satellite_id"] == "sat-a" else 1.0,
            }
            for window in bundle.energy_windows
        )
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=bundle,
        )
        record = result.records[0]
        metrics = aggregate_runtime_records(
            result.records,
            time_horizon_s=1_000.0,
            trace_model=result.trace_model,
            scope=result.scope,
        ).runtime_metrics

        self.assertEqual(record.access_action_type, "shadow_offload")
        self.assertIn("shadow_offload", record.shadow_strategies)
        self.assertEqual(metrics["shadow_offload_request_count"], 1)

    def test_pd_forwarding_respects_enabled_actions(self):
        record = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "shortest_path",
                    "max_hop_count": 2,
                    "enabled_actions": ["local_compute", "reject_drop"],
                }
            ),
            trace_bundle=FakePDTraceBundle(),
        ).records[0]

        self.assertTrue(record.rejected)
        self.assertEqual(record.reject_reason, "candidate_prediction_failed")

    def test_pd_runtime_reserves_prefill_kv_decode_and_first_token_return(self):
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertTrue(record.accepted)
        self.assertEqual(record.action_type, "pd_disaggregated")
        self.assertEqual(record.prefill_deployment_id, "deployment_0:prefill:0")
        self.assertEqual(record.decode_deployment_id, "deployment_0:decode:0")
        self.assertEqual(record.prefill_entry_satellite_id, "sat-b")
        self.assertEqual(record.decode_entry_satellite_id, "sat-c")
        self.assertEqual(record.assigned_satellite_id, "sat-c")
        self.assertEqual(record.pd_placement_scope, "same_plane_cross_cluster")
        self.assertEqual(record.kv_handoff_bytes, 10 * 256.0)
        self.assertGreater(record.kv_handoff_transmission_s or 0.0, 0.0)
        self.assertGreater(record.kv_handoff_propagation_s or 0.0, 0.0)
        self.assertAlmostEqual(
            record.kv_handoff_visible_s or 0.0,
            (record.kv_handoff_queue_wait_s or 0.0)
            + (record.kv_handoff_setup_s or 0.0)
            + (record.kv_handoff_transmission_s or 0.0)
            + (record.kv_handoff_propagation_s or 0.0),
        )
        self.assertGreater(record.first_token_return_s or 0.0, 0.0)
        self.assertGreater(record.ttft_s or 0.0, record.prefill_service_s or 0.0)
        self.assertGreater(record.e2e_s or 0.0, record.ttft_s or 0.0)
        self.assertEqual(record.response_streaming_model, "first_token_plus_bulk_remaining")
        self.assertEqual(record.kv_capacity_admission, "pass")
        self.assertFalse(record.valid_for_paper_main_results)
        self.assertEqual(result.state_summary["module_status"]["queue_model"], "deployment_fifo")
        self.assertEqual(
            result.state_summary["module_status"]["kv_state"],
            "pd_externalized_kv_handoff_session_actions",
        )
        self.assertEqual(result.state_summary["module_status"]["kv_transfer_model"], "page_chunk_store_and_forward")
        self.assertIn("deployment_0:prefill:0", result.state_summary["compute_queues"])
        self.assertIn("deployment_0:decode:0", result.state_summary["compute_queues"])
        self.assertEqual(record.provenance["internal_comm_accounting"], "externalized_pd_kv")
        self.assertEqual(
            record.provenance["capacity_allocation_assumption"],
            "explicit_pd_deployment_group_per_group_throughput",
        )
        self.assertTrue(result.decision_traces[0]["scheduler_decision"]["evaluated"])
        self.assertEqual(result.decision_traces[0]["selected_candidate"]["action_type"], "pd_disaggregated")
        self.assertAlmostEqual(record.ttft_s or 0.0, sum(
            float(record.ttft_breakdown_s[key])
            for key in (
                "access_wait_s", "uplink_s", "prefill_queue_s", "prefill_compute_s",
                "kv_transfer_s", "decode_queue_s", "first_token_decode_s", "return_path_s",
            )
        ))

    def test_pd_remaining_response_failure_preserves_first_token_ttft_and_owners(self):
        record = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FlakyRemainingResponseTraceBundle(),
        ).records[0]

        self.assertTrue(record.accepted)
        self.assertFalse(record.completed)
        self.assertTrue(record.timeout)
        self.assertEqual(record.timeout_reason, "route_became_unavailable_after_compute")
        self.assertGreater(record.ttft_s or 0.0, 0.0)
        self.assertEqual(record.compute_owner_satellite_id, "sat-c")
        self.assertEqual(record.compute_owner_satellite_ids, ("sat-b", "sat-c"))
        self.assertIn("sat-b", record.route_node_ids)
        self.assertIn("sat-c", record.route_node_ids)
        self.assertEqual(record.provenance["partial_response_stage"], "remaining_output_return")

    def test_zero_output_requests_fail_closed_without_ttft_in_both_pd_modes(self):
        strategy = RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2})
        colocated = RuntimeDeploymentCatalog(
            deployments=(
                RuntimeDeployment(
                    deployment_id="deployment_0",
                    role="colocated",
                    member_satellites=("sat-b",),
                    prefill_tokens_per_s=10.0,
                    decode_tokens_per_s=4.0,
                ),
            )
        )
        records = (
            run_trace_backed_runtime(
                requests=(_request(output_tokens=0, output_bytes=0.0),),
                deployment_catalog=colocated,
                runtime_strategy=strategy,
                trace_bundle=FakePDTraceBundle(),
            ).records[0],
            run_trace_backed_runtime(
                requests=(_request(output_tokens=0, output_bytes=0.0),),
                deployment_catalog=_pd_catalog(),
                runtime_strategy=strategy,
                trace_bundle=FakePDTraceBundle(),
            ).records[0],
        )
        for record in records:
            self.assertTrue(record.rejected)
            self.assertEqual(record.reject_reason, "output_tokens_must_be_positive_for_ttft")
            self.assertIsNone(record.ttft_s)
            self.assertIsNone(record.first_token_decode_s)

    def test_colocated_and_pd_use_the_same_ttft_contract(self):
        base = _pd_catalog(decode_rate=4.0)
        prefill = replace(
            base.prefill_deployments[0],
            member_satellites=("sat-b",),
            entry_satellite_ids=("sat-b",),
            placement_scope="intra_cluster",
            paired_deployment_scopes={"deployment_0:decode:0": "intra_cluster"},
        )
        decode = replace(
            base.decode_deployments[0],
            member_satellites=("sat-b",),
            entry_satellite_ids=("sat-b",),
            placement_scope="intra_cluster",
            paired_deployment_scopes={"deployment_0:prefill:0": "intra_cluster"},
        )
        pd_catalog = RuntimeDeploymentCatalog(
            deployments=(base.default_deployment, prefill, decode),
            placement_model="explicit_pd_deployment_groups",
        )
        colocated_catalog = RuntimeDeploymentCatalog(
            deployments=(
                RuntimeDeployment(
                    deployment_id="deployment_0",
                    role="colocated",
                    member_satellites=("sat-b",),
                    entry_satellite_ids=("sat-b",),
                    prefill_tokens_per_s=10.0,
                    decode_tokens_per_s=4.0,
                ),
            )
        )
        strategy = RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2})
        colocated = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=colocated_catalog,
            runtime_strategy=strategy,
            trace_bundle=FakePDTraceBundle(),
        ).records[0]
        disaggregated = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=pd_catalog,
            runtime_strategy=strategy,
            trace_bundle=FakePDTraceBundle(),
        ).records[0]

        self.assertAlmostEqual(colocated.ttft_s or 0.0, disaggregated.ttft_s or 0.0)
        self.assertEqual(colocated.provenance["ttft_definition"], disaggregated.provenance["ttft_definition"])
        self.assertAlmostEqual(colocated.ttft_s or 0.0, sum(
            float(colocated.ttft_breakdown_s[key])
            for key in (
                "access_wait_s", "uplink_s", "prefill_queue_s", "prefill_compute_s",
                "kv_transfer_s", "decode_queue_s", "first_token_decode_s", "return_path_s",
            )
        ))

    def test_pd_runtime_record_splits_candidate_scope_from_pairing_rule(self):
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(
                placement_scope="hybrid_template_mix",
                pair_scope="same_plane_cross_cluster",
                pair_rule="hybrid_template:same_plane_plus_cross_plane_overflow",
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertTrue(record.accepted)
        self.assertEqual(record.pd_placement_scope, "same_plane_cross_cluster")
        self.assertEqual(record.resolved_pd_pair_scope, "same_plane_cross_cluster")
        self.assertEqual(record.pd_pairing_rule, "hybrid_template:same_plane_plus_cross_plane_overflow")
        self.assertEqual(record.provenance["pd_placement_candidate_scope"], "hybrid_template_mix")

    def test_pd_decode_queue_is_reserved_once_for_full_decode_interval(self):
        decode_rate = 1.0
        result = run_trace_backed_runtime(
            requests=(
                _request("req-0", 0.0),
                _request("req-1", 0.0),
            ),
            deployment_catalog=_pd_catalog(decode_rate=decode_rate),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakePDTraceBundle(),
        )

        first, second = result.records
        self.assertEqual(first.decode_deployment_id, second.decode_deployment_id)
        self.assertEqual(first.decode_service_s, 4.0)
        self.assertEqual(second.decode_service_s, 4.0)
        self.assertGreater(second.ttft_s or 0.0, first.ttft_s or 0.0)
        first_token_ready_s = second.arrival_time_s + (second.ttft_s or 0.0) - (second.first_token_return_s or 0.0)
        decode_start_s = first_token_ready_s - (1.0 / decode_rate)
        expected_decode_done_s = decode_start_s + (second.decode_service_s or 0.0)
        self.assertAlmostEqual(
            result.state_summary["compute_queues"][second.decode_deployment_id]["decode_available_s"],
            expected_decode_done_s,
        )
        self.assertEqual(second.kv_serve_action, "recompute_context")

    def test_pd_incremental_turn_reuses_sticky_session_kv(self):
        result = run_trace_backed_runtime(
            requests=(
                _request("req-0", 0.0, prompt_tokens=10, output_tokens=4),
                _request(
                    "req-1",
                    20.0,
                    prompt_tokens=None,
                    new_user_tokens=2,
                    output_tokens=4,
                    input_bytes=200.0,
                ),
            ),
            deployment_catalog=_pd_catalog(kv_cache_bytes_per_token=256.0),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakePDTraceBundle(),
        )

        second = result.records[1]
        self.assertEqual(second.session_action, "sticky_kv_owner")
        self.assertEqual(second.prompt_tokens, 2)
        self.assertEqual(second.kv_serve_action, "sticky_kv_owner")
        self.assertEqual(second.provenance["token_context_contract"], "incremental_turn_sticky_kv_reuse")
        self.assertEqual(second.provenance["context_tokens_before_turn"], 14)
        self.assertEqual(second.provenance["prefill_tokens_effective"], 2)
        self.assertEqual(second.provenance["context_recompute_tokens"], 0)
        self.assertEqual(second.kv_hit_tokens, 14)
        self.assertEqual(second.kv_miss_tokens, 2)
        self.assertEqual(second.kv_handoff_bytes, 2 * 256.0)
        self.assertTrue(second.provenance["sticky_prefill_kv_hit"])
        session = result.state_summary["sessions"]["session-0"]
        self.assertEqual(session["owner_deployment_id"], "deployment_0:decode:0")
        self.assertEqual(session["context_tokens"], 20)
        self.assertEqual(session["kv_cache_tokens"], 20)

    def test_pd_embedded_internal_comm_accounting_rejects_external_kv_route(self):
        catalog = _pd_catalog()
        deployments = tuple(
            RuntimeDeployment(
                **{
                    **deployment.to_dict(),
                    "internal_comm_accounting": "embedded"
                    if deployment.role in {"prefill", "decode"}
                    else deployment.internal_comm_accounting,
                }
            )
            for deployment in catalog.deployments
        )
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=RuntimeDeploymentCatalog(
                deployments=deployments,
                placement_model="explicit_pd_deployment_groups",
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertEqual(record.reject_reason, "not_implemented:no_external_kv_route")
        self.assertEqual(record.provenance["internal_comm_accounting"], ("embedded", "embedded"))

    def test_pd_kv_capacity_gate_counts_existing_sessions_on_decode_deployment(self):
        result = run_trace_backed_runtime(
            requests=(
                _request("req-0", 0.0, session_id="session-0"),
                _request("req-1", 20.0, session_id="session-1"),
            ),
            deployment_catalog=_pd_catalog(kv_cache_capacity_tokens=20),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakePDTraceBundle(),
        )

        first, second = result.records
        self.assertFalse(first.rejected)
        self.assertTrue(second.rejected)
        self.assertEqual(second.reject_reason, "kv_capacity")
        self.assertEqual(second.kv_capacity_admission, "fail")
        aggregate = aggregate_runtime_records(
            result.records,
            time_horizon_s=100.0,
            trace_model=result.trace_model,
            scope=result.scope,
        ).to_dict()
        self.assertEqual(aggregate["runtime_metrics"]["kv_capacity_reject_count"], 1)

    def test_pd_runtime_migrates_existing_session_kv_when_enabled(self):
        bundle = FakePDTraceBundle()
        bundle.energy_windows = tuple(
            {
                **window,
                "battery_capacity_j": 100.0,
                "idle_power_w": 0.1,
                "compute_energy_j_per_token": 0.0,
                "link_energy_j_per_byte": 0.0,
                "soc_min": 0.1,
            }
            for window in bundle.energy_windows
        )
        result = run_trace_backed_runtime(
            requests=(
                _request("req-0", 0.0, session_id="session-0", prompt_tokens=10, output_tokens=4),
                _request(
                    "req-1",
                    20.0,
                    session_id="session-0",
                    prompt_tokens=None,
                    new_user_tokens=2,
                    output_tokens=4,
                    input_bytes=200.0,
                ),
            ),
            deployment_catalog=_pd_catalog_two_decodes(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 2,
                    "kv_policy": {
                        "kv_serve_actions_enabled": ["migrate_kv"],
                        "kv_post_turn_actions_enabled": ["none"],
                        "kv_transfer_chunk_bytes": 1024,
                    },
                }
            ),
            trace_bundle=bundle,
        )

        first, second = result.records
        self.assertEqual(first.decode_deployment_id, "deployment_0:decode:near")
        self.assertEqual(second.kv_serve_action, "migrate_kv")
        self.assertEqual(second.decode_deployment_id, "deployment_0:decode:far")
        self.assertEqual(second.kv_hit_tokens, 14)
        self.assertEqual(second.context_recompute_tokens, 0)
        self.assertEqual(second.kv_migration_bytes, 14 * 256.0)
        self.assertEqual(second.kv_handoff_bytes, 2 * 256.0)
        self.assertGreater(second.session_kv_migration_bytes or 0.0, 0.0)
        self.assertGreater(second.kv_transfer_attempted_bytes or 0.0, second.kv_handoff_bytes or 0.0)
        energy_gate = second.provenance["hard_gate_results"]["energy_soc"]
        self.assertAlmostEqual(energy_gate["predicted_soc_after"], second.soc_after, places=6)
        self.assertAlmostEqual(
            energy_gate["prefill_start_s"],
            second.provenance["actual_prefill_start_s"],
            places=6,
        )
        self.assertAlmostEqual(
            energy_gate["decode_start_s"],
            second.provenance["actual_decode_start_s"],
            places=6,
        )

    def test_pd_prediction_rebuilds_dependency_chain_after_shared_link_delay(self):
        base = _pd_catalog_two_decodes()
        prefill = base.prefill_deployments[0]
        near = replace(
            next(item for item in base.decode_deployments if item.deployment_id.endswith(":near")),
            member_satellites=("sat-a",),
            entry_satellite_ids=("sat-a",),
            member_energy_weights={"sat-a": 1.0},
        )
        far = next(item for item in base.decode_deployments if item.deployment_id.endswith(":far"))
        catalog = RuntimeDeploymentCatalog(
            deployments=(base.deployments[0], prefill, near, far),
            placement_model="explicit_pd_deployment_groups",
        )
        bundle = FakePDTraceBundle()
        bundle.links = tuple({**row, "bandwidth_bps": 1_000.0} for row in bundle.links)
        bundle.physical_links = bundle.links
        bundle._links_by_id = {row["link_id"]: row for row in bundle.links}
        bundle.energy_windows = tuple(
            {
                **window,
                "start_s": start_s,
                "end_s": end_s,
                "sunlit": sunlit,
                "dvfs_multiplier": multiplier,
                "compute_energy_j_per_token": 0.1,
                "link_energy_j_per_byte": 0.001,
                "battery_capacity_j": 10_000.0,
            }
            for window in bundle.energy_windows
            for start_s, end_s, sunlit, multiplier in (
                (0.0, 78.0, True, 1.0),
                (78.0, 1_000.0, False, 2.0),
            )
        )
        result = run_trace_backed_runtime(
            requests=(
                _request("req-0", 0.0, session_id="session-shared"),
                _request(
                    "req-1",
                    50.0,
                    session_id="session-shared",
                    prompt_tokens=None,
                    new_user_tokens=2,
                    output_tokens=4,
                ),
            ),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 2,
                    "dvfs_modes_enabled": ["shadow_dvfs"],
                    "kv_policy": {
                        "kv_serve_actions_enabled": ["migrate_kv"],
                        "kv_post_turn_actions_enabled": ["none"],
                        "kv_transfer_chunk_bytes": 10_000,
                    },
                }
            ),
            trace_bundle=bundle,
        )
        first, second = result.records
        self.assertEqual(first.decode_entry_satellite_id, "sat-a")
        self.assertEqual(second.decode_entry_satellite_id, "sat-c")
        self.assertEqual(second.kv_serve_action, "migrate_kv")
        self.assertGreater(second.provenance["actual_prefill_start_s"], 78.0)
        self.assertEqual(second.prefill_action_type, "shadow_dvfs")
        self.assertAlmostEqual(second.prefill_service_s, 0.4)
        self.assertAlmostEqual(
            second.provenance["hard_gate_results"]["energy_soc"]["prefill_start_s"],
            second.provenance["actual_prefill_start_s"],
            places=6,
        )
        self.assertAlmostEqual(
            second.provenance["hard_gate_results"]["energy_soc"]["decode_start_s"],
            second.provenance["actual_decode_start_s"],
            places=6,
        )
        predicted_gate = second.provenance["hard_gate_results"]["energy_soc"]
        self.assertEqual(
            predicted_gate["prefill_profile"]["member_actions"],
            second.provenance["prefill_member_actions"],
        )
        self.assertEqual(
            predicted_gate["decode_profile"]["member_actions"],
            second.provenance["decode_member_actions"],
        )
        self.assertAlmostEqual(
            predicted_gate["prefill_profile"]["dvfs_multiplier"],
            second.prefill_service_s / 0.2,
        )
        self.assertAlmostEqual(second.provenance["predicted_ttft_s"], second.ttft_s, places=6)
        self.assertAlmostEqual(second.provenance["predicted_e2e_s"], second.e2e_s, places=6)
        self.assertAlmostEqual(second.provenance["predicted_soc_after"], second.soc_after, places=6)

    def test_pd_runtime_post_turn_replication_occupies_replica_state(self):
        result = run_trace_backed_runtime(
            requests=(_request("req-0", 0.0, session_id="session-0"),),
            deployment_catalog=_pd_catalog_two_decodes(kv_cache_capacity_tokens=100),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 2,
                    "kv_policy": {
                        "kv_serve_actions_enabled": ["recompute_context"],
                        "kv_post_turn_actions_enabled": ["replicate_kv"],
                        "replica_count_limit": 1,
                    },
                }
            ),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertEqual(record.kv_post_turn_action, "replicate_kv")
        self.assertEqual(record.kv_replication_bytes, 14 * 256.0)
        self.assertEqual(record.kv_pages_replicated, 1)
        session = result.state_summary["sessions"]["session-0"]
        self.assertEqual(session["replica_deployment_ids"], ["deployment_0:decode:far"])
        self.assertIn("deployment_0:decode:far", result.state_summary["kv_cache_deployments"])

    def test_pd_runtime_expired_replication_does_not_create_replica_hit_state(self):
        bundle = FakePDTraceBundle()
        bundle.energy_windows = tuple(
            {**window, "link_energy_j_per_byte": 1.0}
            for window in bundle.energy_windows
        )
        result = run_trace_backed_runtime(
            requests=(_request("req-0", 0.0, session_id="session-0"),),
            deployment_catalog=_pd_catalog_two_decodes(kv_cache_capacity_tokens=100),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 2,
                    "kv_policy": {
                        "kv_serve_actions_enabled": ["recompute_context"],
                        "kv_post_turn_actions_enabled": ["replicate_kv"],
                        "replica_count_limit": 1,
                        "session_ttl_s": 0,
                        "kv_transfer_chunk_bytes": 512,
                    },
                }
            ),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertEqual(record.kv_post_turn_action, "replicate_kv")
        self.assertTrue(record.kv_transfer_expired)
        self.assertEqual(record.kv_pages_replicated, 0)
        replication_link_j = sum(
            float(row.get("link_bus_j", 0.0))
            for row in record.energy_ledger
            if row.get("energy_stage") == "replication"
        )
        self.assertGreater(record.kv_transfer_delivered_bytes or 0.0, 0.0)
        self.assertLess(record.kv_transfer_delivered_bytes or 0.0, record.kv_replication_bytes or 0.0)
        self.assertAlmostEqual(replication_link_j, record.kv_transfer_delivered_bytes or 0.0)
        session = result.state_summary["sessions"]["session-0"]
        self.assertEqual(session["replica_deployment_ids"], [])

    def test_pd_replication_capacity_checks_target_deployment(self):
        result = run_trace_backed_runtime(
            requests=(_request("req-0", 0.0, session_id="session-0"),),
            deployment_catalog=_pd_catalog_two_decodes(
                kv_cache_capacity_tokens=100,
                far_kv_cache_capacity_tokens=10,
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 2,
                    "kv_policy": {
                        "kv_serve_actions_enabled": ["recompute_context"],
                        "kv_post_turn_actions_enabled": ["none", "replicate_kv"],
                        "replica_count_limit": 1,
                    },
                }
            ),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertTrue(record.accepted)
        self.assertEqual(record.kv_post_turn_action, "none")
        rejected = record.provenance["rejected_candidates"]
        self.assertTrue(
            any(
                item["provenance"].get("kv_post_turn_action") == "replicate_kv"
                and item["hard_gate_results"]["kv_capacity"]["replica_capacity_passed"] is False
                for item in rejected
            )
        )

    def test_pd_replication_capacity_checks_target_when_primary_capacity_unknown(self):
        result = run_trace_backed_runtime(
            requests=(_request("req-0", 0.0, session_id="session-0"),),
            deployment_catalog=_pd_catalog_two_decodes(
                kv_cache_capacity_tokens=None,
                far_kv_cache_capacity_tokens=10,
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 2,
                    "kv_policy": {
                        "kv_serve_actions_enabled": ["recompute_context"],
                        "kv_post_turn_actions_enabled": ["none", "replicate_kv"],
                        "replica_count_limit": 1,
                    },
                }
            ),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertTrue(record.accepted)
        self.assertEqual(record.kv_capacity_admission, "unknown")
        self.assertEqual(record.kv_post_turn_action, "none")
        rejected = record.provenance["rejected_candidates"]
        self.assertTrue(
            any(
                item["provenance"].get("kv_post_turn_action") == "replicate_kv"
                and item["hard_gate_results"]["kv_capacity"]["capacity_tokens"] is None
                and item["hard_gate_results"]["kv_capacity"]["replica_capacity_passed"] is False
                for item in rejected
            )
        )

    def test_pd_kv_chunk_transfer_interruption_rejects_candidate(self):
        bundle = FlakyKVTraceBundle()
        bundle.energy_windows = tuple(
            {
                **window,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 1.0,
            }
            for window in bundle.energy_windows
        )
        result = run_trace_backed_runtime(
            requests=(_request("req-0", 0.0, session_id="session-0"),),
            deployment_catalog=_pd_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "shortest_path",
                    "max_hop_count": 2,
                    "kv_policy": {"kv_transfer_chunk_bytes": 512},
                }
            ),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertTrue(record.kv_transfer_interrupted)
        self.assertIn("pd_handoff:", record.kv_transfer_interruption_reason or "")
        self.assertGreater(record.kv_transfer_attempted_bytes or 0.0, 0.0)
        self.assertGreater(record.kv_transfer_delivered_bytes or 0.0, 0.0)
        self.assertEqual(record.kv_transfer_committed_page_bytes, 0.0)
        self.assertEqual(record.kv_transfer_useful_bytes, 0.0)
        self.assertFalse(record.served)
        stages = {row.get("energy_stage") for row in record.energy_ledger}
        self.assertIn("input_route", stages)
        self.assertIn("prefill", stages)
        self.assertIn("pd_handoff", stages)
        self.assertNotIn("decode", stages)

    def test_pd_hbm_bytes_capacity_can_reject_even_when_token_capacity_is_large(self):
        result = run_trace_backed_runtime(
            requests=(_request("req-0", 0.0, session_id="session-0"),),
            deployment_catalog=_pd_catalog(kv_cache_capacity_tokens=1_000_000),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "shortest_path",
                    "max_hop_count": 2,
                    "kv_policy": {
                        "cards_per_satellite": 1,
                        "hbm_capacity_gb_per_card": 0.000001,
                        "hbm_usable_fraction": 1.0,
                        "kv_page_size_tokens": 16,
                        "kv_page_metadata_bytes": 0,
                    },
                }
            ),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertEqual(record.reject_reason, "kv_capacity")
        decision = record.provenance["scheduler_decision"]
        kv_capacity = decision["score_terms"]["hard_gate_results"]["kv_capacity"]
        self.assertEqual(kv_capacity["hbm_model"], "aggregate_hbm_bytes")
        self.assertFalse(kv_capacity["card_level_hbm_admission"])

    def test_pd_hbm_existing_usage_uses_page_rounded_session_bytes(self):
        capacity_gb = (2 * 16 * 256) / (1024.0 ** 3)
        result = run_trace_backed_runtime(
            requests=(
                _request("req-0", 0.0, session_id="session-0", prompt_tokens=1, output_tokens=1),
                _request("req-1", 20.0, session_id="session-1", prompt_tokens=1, output_tokens=1),
                _request("req-2", 40.0, session_id="session-2", prompt_tokens=1, output_tokens=1),
            ),
            deployment_catalog=_pd_catalog(kv_cache_capacity_tokens=1_000_000),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "shortest_path",
                    "max_hop_count": 2,
                    "kv_policy": {
                        "cards_per_satellite": 1,
                        "hbm_capacity_gb_per_card": capacity_gb,
                        "hbm_usable_fraction": 1.0,
                        "kv_page_size_tokens": 16,
                        "kv_page_metadata_bytes": 0,
                    },
                }
            ),
            trace_bundle=FakePDTraceBundle(),
        )

        first, second, third = result.records
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertTrue(third.rejected)
        self.assertEqual(third.reject_reason, "kv_capacity")
        session0 = result.state_summary["sessions"]["session-0"]
        self.assertEqual(session0["kv_cache_bytes"], 16 * 256)

    def test_pd_missing_kv_cache_bytes_per_token_rejects_externalized_runtime(self):
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=_pd_catalog(kv_cache_bytes_per_token=None),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertEqual(record.reject_reason, "pd_kv_cache_bytes_per_token_missing")

    def test_pd_runtime_accepts_explicit_groups_without_aggregate_deployment(self):
        catalog = _pd_catalog()
        result = run_trace_backed_runtime(
            requests=(_request(),),
            deployment_catalog=RuntimeDeploymentCatalog(
                deployments=catalog.deployments[1:],
                placement_model="explicit_pd_deployment_groups",
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakePDTraceBundle(),
        )

        record = result.records[0]
        self.assertTrue(record.accepted)
        self.assertFalse(record.rejected)
        self.assertEqual(record.deployment_id, "deployment_0")


if __name__ == "__main__":
    unittest.main()
