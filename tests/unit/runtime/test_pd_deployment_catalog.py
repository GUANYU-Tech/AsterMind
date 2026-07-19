import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


from orbitserve.runtime.scenario import build_runtime_scenario
from orbitserve.runtime.runtime_deployments import build_runtime_deployment_catalog
from orbitserve.orbit.simulation_traces import OrbitTraceManifest, SURROGATE_ORBIT_TRACE


def _scenario(
    throughput,
    *,
    satellite_count=12,
    satellite_catalog=None,
    deployment_id="deployment_0",
    **deployment_overrides,
):
    throughput = dict(throughput)
    prefill_fraction = float(throughput.pop("_test_prefill_fraction", 0.6))
    serving_spec = {
        "prefill_satellite_fraction": prefill_fraction,
        "decode_satellite_fraction": 1.0 - prefill_fraction,
    }
    deployment = {
        "deployment_id": deployment_id,
        "throughput_estimate": throughput,
        "serving_spec": serving_spec,
        **deployment_overrides,
    }
    constellation = {"satellite_count": satellite_count}
    if satellite_catalog is not None:
        constellation["satellite_catalog"] = satellite_catalog
    return SimpleNamespace(
        constellation=constellation,
        serving_deployments=(deployment,),
    )


def _pd_throughput(*, prefill_fraction=0.6, **overrides):
    payload = {
        "evaluated": True,
        "feasible": True,
        "pd_mode": "disaggregated",
        "prefill_tokens_per_s": 1200.0,
        "decode_tokens_per_s": 600.0,
        "request_throughput_per_s": 8.0,
        "_test_prefill_fraction": prefill_fraction,
        "satellites_per_deployment": 2,
        "stage_satellite_count": 1,
        "kv_cache_capacity_tokens": 1_000_000,
        "kv_cache_bytes_per_token": 16.0,
    }
    payload.update(overrides)
    return payload


class PDDeploymentCatalogTests(unittest.TestCase):
    def test_colocated_catalog_keeps_default_aggregate_deployment(self):
        scenario = _scenario(
            {
                "evaluated": True,
                "feasible": True,
                "pd_mode": "colocated",
                "prefill_tokens_per_s": 100.0,
                "decode_tokens_per_s": 50.0,
                "request_throughput_per_s": 2.0,
                "satellites_per_deployment": 4,
                "stage_satellite_count": 2,
            },
            satellite_count=4,
        )

        catalog = build_runtime_deployment_catalog(scenario)

        self.assertIsNotNone(catalog.default_deployment)
        self.assertEqual(catalog.default_deployment.deployment_id, "deployment_0")
        self.assertEqual(catalog.default_deployment.role, "colocated")
        self.assertEqual(catalog.default_deployment.member_satellites, tuple(f"sat-{i:05d}" for i in range(4)))
        self.assertEqual(catalog.default_deployment.entry_satellite_ids, ("sat-00000",))
        self.assertEqual(catalog.default_deployment.prefill_tokens_per_s, 100.0)
        self.assertEqual(catalog.default_deployment.decode_tokens_per_s, 50.0)
        self.assertEqual(catalog.prefill_deployments, ())
        self.assertEqual(catalog.decode_deployments, ())
        self.assertEqual(catalog.to_dict()["default_deployment"]["deployment_id"], "deployment_0")

    def test_colocated_catalog_uses_serving_satellite_budget_when_present(self):
        scenario = _scenario(
            {
                "evaluated": True,
                "feasible": True,
                "pd_mode": "colocated",
                "prefill_tokens_per_s": 100.0,
                "decode_tokens_per_s": 50.0,
                "request_throughput_per_s": 2.0,
                "satellites_per_deployment": 1,
                "stage_satellite_count": 1,
                "serving_satellite_budget_count": 4,
                "satellite_budget_mode": "fixed_total",
                "comparison_group_id": "pd-fair-four",
            },
            satellite_count=8,
        )

        catalog = build_runtime_deployment_catalog(scenario)

        self.assertEqual(catalog.default_deployment.member_satellites, tuple(f"sat-{i:05d}" for i in range(4)))
        self.assertEqual(catalog.provenance["serving_satellite_budget_count"], 4)
        self.assertEqual(catalog.provenance["satellite_budget_mode"], "fixed_total")
        self.assertEqual(catalog.default_deployment.provenance["comparison_group_id"], "pd-fair-four")

    def test_disaggregated_catalog_splits_groups_and_per_group_throughput(self):
        catalog = build_runtime_deployment_catalog(_scenario(_pd_throughput(), satellite_count=10))

        self.assertEqual(catalog.placement_model, "explicit_pd_deployment_groups")
        self.assertEqual(catalog.default_deployment.deployment_id, "deployment_0")
        self.assertEqual(catalog.default_deployment.prefill_tokens_per_s, 3600.0)
        self.assertEqual(catalog.default_deployment.decode_tokens_per_s, 1200.0)
        self.assertEqual(catalog.default_deployment.internal_comm_accounting, "externalized_pd_kv")
        self.assertEqual(len(catalog.prefill_deployments), 3)
        self.assertEqual(len(catalog.decode_deployments), 2)

        first_prefill = catalog.prefill_deployments[0]
        second_prefill = catalog.prefill_deployments[1]
        first_decode = catalog.decode_deployments[0]
        self.assertEqual(first_prefill.deployment_id, "deployment_0:prefill:0")
        self.assertEqual(first_prefill.member_satellites, ("sat-00000", "sat-00001"))
        self.assertEqual(second_prefill.member_satellites, ("sat-00002", "sat-00003"))
        self.assertEqual(first_decode.member_satellites, ("sat-00006", "sat-00007"))
        self.assertEqual(first_prefill.entry_satellite_ids, ("sat-00000",))
        self.assertEqual(first_decode.entry_satellite_ids, ("sat-00006",))
        self.assertEqual(first_prefill.per_group_prefill_tokens_per_s, 1200.0)
        self.assertIsNone(first_prefill.per_group_decode_tokens_per_s)
        self.assertEqual(first_decode.per_group_decode_tokens_per_s, 600.0)
        self.assertIsNone(first_decode.per_group_prefill_tokens_per_s)
        self.assertEqual(first_prefill.kv_cache_capacity_tokens, 1_000_000)
        self.assertEqual(first_prefill.kv_cache_bytes_per_token, 16.0)
        self.assertEqual(first_prefill.satellites_per_deployment, 2)
        self.assertEqual(first_prefill.stage_satellite_count, 1)
        self.assertEqual(first_prefill.paired_deployment_ids, ("deployment_0:decode:0", "deployment_0:decode:1"))
        self.assertEqual(first_decode.paired_deployment_ids, tuple(item.deployment_id for item in catalog.prefill_deployments))

    def test_disaggregated_catalog_can_stride_member_groups_across_catalog(self):
        satellite_catalog = [
            {
                "satellite_id": f"sat-{index:05d}",
                "orbital_set_name": "probe_shell",
                "plane_index": index // 4,
                "cell_index": (index // 2) % 2,
            }
            for index in range(8)
        ]
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(prefill_fraction=0.5),
                satellite_count=8,
                satellite_catalog=satellite_catalog,
                serving_spec={
                    "pd_mode": "disaggregated",
                    "pd_placement": {
                        "placement_scope": "cross_plane",
                        "member_selection": "strided_catalog_groups",
                    },
                },
            )
        )

        self.assertEqual(catalog.prefill_deployments[0].member_satellites, ("sat-00000", "sat-00001"))
        self.assertEqual(catalog.decode_deployments[0].member_satellites, ("sat-00004", "sat-00005"))
        self.assertEqual(catalog.prefill_deployments[0].plane_index, 0)
        self.assertEqual(catalog.decode_deployments[0].plane_index, 1)

    def test_runtime_scenario_carries_trace_satellite_catalog_for_pd_topology(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            catalog_path = tmp / "satellite_catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "satellite_id": "sat-00000",
                            "orbital_set_name": "probe_shell",
                            "plane_index": 0,
                            "cell_index": 0,
                        },
                        {
                            "satellite_id": "sat-00001",
                            "orbital_set_name": "probe_shell",
                            "plane_index": 0,
                            "cell_index": 0,
                        },
                        {
                            "satellite_id": "sat-00002",
                            "orbital_set_name": "probe_shell",
                            "plane_index": 1,
                            "cell_index": 0,
                        },
                        {
                            "satellite_id": "sat-00003",
                            "orbital_set_name": "probe_shell",
                            "plane_index": 1,
                            "cell_index": 0,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            trace_manifest = OrbitTraceManifest(
                trace_model=SURROGATE_ORBIT_TRACE,
                evaluated=True,
                satellite_catalog_path=catalog_path.name,
                provenance={"trace_dir": str(tmp)},
            )
            scenario = build_runtime_scenario(
                scenario_id="catalog_probe",
                constellation={"name": "probe", "satellite_count": 4},
                serving_spec={
                    "pd_mode": "disaggregated",
                    "pd_placement": {"placement_scope": "cross_plane"},
                },
                serving_throughput=_pd_throughput(
                    prefill_fraction=0.5,
                    satellites_per_deployment=2,
                ),
                simulation_result={
                    "trace_evaluated": True,
                    "trace_model": SURROGATE_ORBIT_TRACE,
                    "trace_manifest": trace_manifest.to_dict(),
                },
            )

        self.assertEqual(len(scenario.constellation["satellite_catalog"]), 4)
        catalog = build_runtime_deployment_catalog(scenario)
        self.assertEqual(catalog.prefill_deployments[0].plane_index, 0)
        self.assertEqual(catalog.decode_deployments[0].plane_index, 1)
        self.assertEqual(
            catalog.prefill_deployments[0].paired_deployment_scopes[catalog.decode_deployments[0].deployment_id],
            "cross_plane",
        )

    def test_runtime_scenario_reads_embedded_access_trace_satellite_catalog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            access_path = tmp / "access_trace.json"
            satellite_catalog = [
                {
                    "satellite_id": "sat-00000",
                    "orbital_set_name": "probe_shell",
                    "plane_index": 0,
                    "cell_index": 0,
                }
            ]
            access_path.write_text(
                json.dumps({"satellite_catalog": satellite_catalog, "access_windows": []}),
                encoding="utf-8",
            )
            trace_manifest = OrbitTraceManifest(
                trace_model=SURROGATE_ORBIT_TRACE,
                evaluated=True,
                access_trace_path=access_path.name,
                provenance={"trace_dir": str(tmp)},
            )
            scenario = build_runtime_scenario(
                scenario_id="embedded_catalog_probe",
                constellation={"name": "probe", "satellite_count": 1},
                simulation_result={
                    "trace_evaluated": True,
                    "trace_model": SURROGATE_ORBIT_TRACE,
                    "trace_manifest": trace_manifest.to_dict(),
                },
            )

        self.assertEqual(scenario.constellation["satellite_catalog"], satellite_catalog)

    def test_runtime_derives_pd_group_counts_without_serving_group_counts(self):
        missing_count = build_runtime_deployment_catalog(
            _scenario(_pd_throughput(prefill_fraction=0.5), satellite_count=10)
        )
        zero_count = build_runtime_deployment_catalog(
            _scenario(_pd_throughput(prefill_fraction=0.8), satellite_count=10)
        )

        self.assertEqual(len(missing_count.prefill_deployments) + len(missing_count.decode_deployments), 5)
        self.assertEqual(len(zero_count.prefill_deployments) + len(zero_count.decode_deployments), 5)

    def test_runtime_leaves_incomplete_pd_group_unassigned(self):
        catalog = build_runtime_deployment_catalog(_scenario(_pd_throughput(), satellite_count=9))

        self.assertTrue(catalog.deployments)
        self.assertEqual(catalog.provenance["unused_satellite_count"], 1)
        assigned = {
            satellite
            for deployment in (*catalog.prefill_deployments, *catalog.decode_deployments)
            for satellite in deployment.member_satellites
        }
        self.assertEqual(len(assigned), 8)

    def test_missing_topology_fails_closed_by_default_for_scoped_candidate(self):
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(pd_placement={"placement_scope": "intra_cluster"}),
                satellite_count=10,
            )
        )

        self.assertEqual(catalog.deployments, ())
        self.assertTrue(catalog.provenance["fail_closed"])
        self.assertEqual(catalog.provenance["reason"], "topology_scope_unavailable:missing_cluster_or_cell_identity")

    def test_missing_topology_can_explicitly_downgrade_to_unscoped_audit(self):
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(
                    pd_placement={
                        "placement_scope": "intra_cluster",
                        "fail_closed_on_unsupported_scope": False,
                    }
                ),
                satellite_count=10,
            )
        )

        self.assertTrue(catalog.prefill_deployments)
        self.assertEqual(catalog.provenance["placement_scope"], "unscoped_disaggregated")
        self.assertTrue(any("topology_scope_unavailable:missing_cluster_or_cell_identity" in warning for warning in catalog.warnings))

    def test_payload_internal_comm_accounting_overrides_pd_default(self):
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(),
                satellite_count=10,
                internal_comm_accounting="embedded",
            )
        )

        self.assertEqual(catalog.default_deployment.internal_comm_accounting, "embedded")
        self.assertEqual(catalog.prefill_deployments[0].internal_comm_accounting, "embedded")

    def test_same_plane_scope_filters_pd_pairings_by_plane_and_cell(self):
        satellite_catalog = (
            {"satellite_id": "sat-p0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 0},
            {"satellite_id": "sat-p1", "orbital_set_name": "walker", "plane_index": 1, "cell_index": 0},
            {"satellite_id": "sat-d0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 1},
            {"satellite_id": "sat-d1", "orbital_set_name": "walker", "plane_index": 1, "cell_index": 1},
        )
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(
                    prefill_fraction=0.5,
                    satellites_per_deployment=1,
                    pd_placement={"placement_scope": "same_plane_cross_cluster"},
                ),
                satellite_count=4,
                satellite_catalog=satellite_catalog,
            )
        )

        self.assertEqual(catalog.provenance["placement_scope"], "same_plane_cross_cluster")
        self.assertEqual(catalog.prefill_deployments[0].paired_deployment_ids, ("deployment_0:decode:0",))
        self.assertEqual(catalog.prefill_deployments[1].paired_deployment_ids, ("deployment_0:decode:1",))
        self.assertEqual(catalog.decode_deployments[0].paired_deployment_ids, ("deployment_0:prefill:0",))
        self.assertEqual(catalog.decode_deployments[1].paired_deployment_ids, ("deployment_0:prefill:1",))

    def test_hybrid_explicit_pairs_only_uses_user_supplied_pairs(self):
        satellite_catalog = (
            {"satellite_id": "sat-p0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 0},
            {"satellite_id": "sat-p1", "orbital_set_name": "walker", "plane_index": 1, "cell_index": 0},
            {"satellite_id": "sat-d0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 1},
            {"satellite_id": "sat-d1", "orbital_set_name": "walker", "plane_index": 1, "cell_index": 1},
        )
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(
                    prefill_fraction=0.5,
                    satellites_per_deployment=1,
                    pd_placement={
                        "placement_scope": "hybrid_explicit_pairs",
                        "explicit_pairs": [
                            {
                                "prefill_group_index": 0,
                                "decode_group_index": 1,
                                "scope": "cross_plane",
                            }
                        ],
                    },
                ),
                satellite_count=4,
                satellite_catalog=satellite_catalog,
            )
        )

        self.assertEqual(catalog.provenance["placement_scope"], "hybrid_explicit_pairs")
        self.assertEqual(catalog.prefill_deployments[0].paired_deployment_ids, ("deployment_0:decode:1",))
        self.assertEqual(catalog.prefill_deployments[0].paired_deployment_scopes, {"deployment_0:decode:1": "cross_plane"})
        self.assertEqual(catalog.prefill_deployments[0].paired_deployment_rules, {"deployment_0:decode:1": "user_explicit"})
        self.assertEqual(catalog.prefill_deployments[1].paired_deployment_ids, ())
        self.assertEqual(catalog.decode_deployments[1].paired_deployment_ids, ("deployment_0:prefill:0",))
        self.assertEqual(catalog.decode_deployments[1].paired_deployment_scopes, {"deployment_0:prefill:0": "cross_plane"})
        self.assertEqual(catalog.decode_deployments[1].paired_deployment_rules, {"deployment_0:prefill:0": "user_explicit"})

    def test_hybrid_explicit_pair_scope_mismatch_fails_closed_with_warning(self):
        satellite_catalog = (
            {"satellite_id": "sat-p0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 0},
            {"satellite_id": "sat-d0", "orbital_set_name": "walker", "plane_index": 1, "cell_index": 0},
        )
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(
                    prefill_fraction=0.5,
                    satellites_per_deployment=1,
                    pd_placement={
                        "placement_scope": "hybrid_explicit_pairs",
                        "explicit_pairs": [
                            {
                                "prefill_group_index": 0,
                                "decode_group_index": 0,
                                "declared_scope": "same_plane_cross_cluster",
                            }
                        ],
                    },
                ),
                satellite_count=2,
                satellite_catalog=satellite_catalog,
            )
        )

        self.assertEqual(catalog.deployments, ())
        self.assertIn("deployment_unavailable:no_valid_pd_pair_for_scope", catalog.warnings)
        self.assertTrue(any("pd_pair_scope_mismatch" in warning for warning in catalog.warnings))

    def test_hybrid_template_mix_is_bounded_and_deterministic(self):
        satellite_catalog = (
            {"satellite_id": "sat-p0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 0},
            {"satellite_id": "sat-d0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 1},
            {"satellite_id": "sat-d1", "orbital_set_name": "walker", "plane_index": 1, "cell_index": 0},
            {"satellite_id": "sat-d2", "orbital_set_name": "walker", "plane_index": 2, "cell_index": 0},
        )
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(
                    prefill_fraction=0.25,
                    satellites_per_deployment=1,
                    pd_placement={
                        "placement_scope": "hybrid_template_mix",
                        "hybrid_template": {
                            "template_id": "same_plane_plus_cross_plane_overflow",
                            "max_pairs_per_prefill_group": 2,
                            "max_cross_plane_pairs_per_prefill_group": 1,
                        },
                    },
                ),
                satellite_count=4,
                satellite_catalog=satellite_catalog,
            )
        )

        self.assertEqual(catalog.provenance["placement_scope"], "hybrid_template_mix")
        prefill = catalog.prefill_deployments[0]
        self.assertEqual(prefill.paired_deployment_ids, ("deployment_0:decode:0", "deployment_0:decode:1"))
        self.assertEqual(
            prefill.paired_deployment_scopes,
            {
                "deployment_0:decode:0": "same_plane_cross_cluster",
                "deployment_0:decode:1": "cross_plane",
            },
        )
        self.assertEqual(
            prefill.paired_deployment_rules,
            {
                "deployment_0:decode:0": "hybrid_template:same_plane_plus_cross_plane_overflow",
                "deployment_0:decode:1": "hybrid_template:same_plane_plus_cross_plane_overflow",
            },
        )
        self.assertEqual(catalog.decode_deployments[2].paired_deployment_ids, ())

    def test_unknown_hybrid_template_fails_closed(self):
        satellite_catalog = (
            {"satellite_id": "sat-p0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 0},
            {"satellite_id": "sat-d0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 1},
        )
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(
                    prefill_fraction=0.5,
                    satellites_per_deployment=1,
                    pd_placement={
                        "placement_scope": "hybrid_template_mix",
                        "hybrid_template": {"template_id": "not_a_template"},
                    },
                ),
                satellite_count=2,
                satellite_catalog=satellite_catalog,
            )
        )

        self.assertEqual(catalog.deployments, ())
        self.assertTrue(any("unsupported_pd_hybrid_template:not_a_template" in warning for warning in catalog.warnings))

    def test_hybrid_template_mix_has_default_pair_cap(self):
        satellite_catalog = (
            {"satellite_id": "sat-p0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 0},
            {"satellite_id": "sat-d0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 1},
            {"satellite_id": "sat-d1", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 2},
            {"satellite_id": "sat-d2", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 3},
            {"satellite_id": "sat-d3", "orbital_set_name": "walker", "plane_index": 1, "cell_index": 0},
            {"satellite_id": "sat-d4", "orbital_set_name": "walker", "plane_index": 2, "cell_index": 0},
        )
        catalog = build_runtime_deployment_catalog(
            _scenario(
                _pd_throughput(
                    prefill_fraction=0.16666666666666666,
                    satellites_per_deployment=1,
                    pd_placement={
                        "placement_scope": "hybrid_template_mix",
                        "hybrid_template": "same_plane_plus_cross_plane_overflow",
                    },
                ),
                satellite_count=6,
                satellite_catalog=satellite_catalog,
            )
        )

        prefill = catalog.prefill_deployments[0]
        self.assertEqual(len(prefill.paired_deployment_ids), 3)
        self.assertEqual(
            prefill.paired_deployment_scopes,
            {
                "deployment_0:decode:0": "same_plane_cross_cluster",
                "deployment_0:decode:1": "same_plane_cross_cluster",
                "deployment_0:decode:3": "cross_plane",
            },
        )


if __name__ == "__main__":
    unittest.main()
