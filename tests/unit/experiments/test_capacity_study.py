import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from orbitserve.experiments.capacity_study import (
    StudySpec, arrivals, directional_bandwidth, main, model_catalog, placements, plan_cases, replica_resources,
)


class CapacityStudyTests(unittest.TestCase):
    def test_duplex_aggregate_is_halved_exactly_once(self):
        self.assertEqual(directional_bandwidth(400, "bidirectional_aggregate"), 200)
        self.assertEqual(directional_bandwidth(1000, "bidirectional_aggregate"), 500)
        self.assertEqual(directional_bandwidth(400, "per_direction"), 400)
        with self.assertRaises(ValueError):
            directional_bandwidth(400, "ambiguous")

    def test_physical_card_limit_and_invalid_parameters_rejected(self):
        for update in ({"cards_per_satellite": [100]}, {"cards_per_satellite": [True]}, {"request_rates": [0]},
                       {"link_bandwidth_gbps": [float("nan")]},
                       {"models": ["all", "qwen3-4b"]}, {"seeds": [1, 1]},
                       {"memory_utilization": 0.1, "non_kv_reserve_fraction": 0.2}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                StudySpec.model_validate(update)

    def test_layouts_use_exactly_four_satellites_without_sharing_gpus(self):
        for layout, n, kind in (("cross_replicas", 8, "qwen3"),
                                ("cross_pipeline", 4, "qwen3"),
                                ("cross_tp", 2, "qwen3"),
                                ("local_replicas", 6, "qwen3"),
                                ("target", 8, "glm_moe_dsa"),
                                ("target", 4, "kimi_k2")):
            with self.subTest(layout=layout):
                tp, pp, _, groups = placements(dict(layout=layout, cards_per_satellite=n), kind)
                devices = [device for group in groups for device in group]
                self.assertEqual(len(devices), 4 * n)
                self.assertEqual(len(set(devices)), 4 * n)
                self.assertTrue(all(len(group) == tp * pp for group in groups))
                self.assertEqual({s for s, _ in devices}, set(range(4)))

    def test_target_layout_does_not_silently_leave_allocated_gpus_idle(self):
        with self.assertRaises(NotImplementedError):
            placements(dict(layout="target", cards_per_satellite=2), "glm_moe_dsa")

    def test_replica_resource_counts_do_not_allocate_the_whole_formation_per_replica(self):
        resources, indices = replica_resources([(s, 7) for s in range(4)])
        self.assertEqual(resources["satellite_count"], 4)
        self.assertEqual(resources["accelerators_per_satellite"], 1)
        self.assertEqual(indices, [0, 0, 0, 0])
        resources, indices = replica_resources([(3, 4), (3, 5)])
        self.assertEqual(resources["satellite_ids"], ["sat3"])
        self.assertEqual(resources["accelerators_per_satellite"], 2)
        self.assertEqual(indices, [0, 1])

    def test_arrivals_are_shared_and_not_truncated_at_request_cap(self):
        spec = StudySpec(measurement_s=10, warmup_s=1)
        workload = spec.workloads[0].model_dump()
        first = arrivals(spec, 4, 17, workload)
        self.assertEqual(first, arrivals(spec, 4, 17, workload))
        self.assertNotEqual(first, arrivals(spec, 4, 29, workload))
        self.assertTrue(all(r["arrival_s"] < 11 for r in first))
        with self.assertRaisesRegex(ValueError, "truncate"):
            arrivals(spec.model_copy(update={"max_requests_per_trial": 2}), 4, 17, workload)

    def test_all_catalog_models_and_precision_variants_remain_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name, quant in (("example", {}), ("example-fp8", {"quant_method": "fp8"})):
                (directory / f"{name}.json").write_text(json.dumps(dict(
                    model_type="example", dtype="bfloat16", quantization_config=quant)))
            catalog = model_catalog(directory)
            self.assertEqual(set(catalog), {"example", "example-fp8"})
            spec = StudySpec(hardware=("nvidia_a800",), cards_per_satellite=(1,),
                             layouts=("cross_replicas",), link_bandwidth_gbps=(400,))
            cases = plan_cases(spec, catalog)
            self.assertEqual(len(cases), 2 * len(spec.workloads))
            self.assertEqual(len({r["case_id"] for r in cases}), len(cases))
            self.assertTrue(all(r["per_direction_gbps"] == 200 for r in cases))

    def test_planning_needs_no_native_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "models").mkdir()
            (directory / "models/a.json").write_text('{"model_type":"example"}')
            (directory / "spec.json").write_text('{"models":["all"]}')
            code = main(["plan", str(directory / "spec.json"), "--out", str(directory / "plan.json"),
                         "--astermind-root", str(directory), "--model-dir", str(directory / "models")])
            self.assertEqual(code, 0)
            plan = json.loads((directory / "plan.json").read_text())
            self.assertTrue(plan["cases"])

    def test_nested_text_quantization_is_not_labeled_as_unquantized(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested.json"
            path.write_text(json.dumps({"text_config": {
                "model_type": "example", "dtype": "bfloat16",
                "quantization_config": {"quant_method": "compressed-tensors"}}}))
            item = model_catalog(folder)["nested"]
            self.assertEqual(item["quantization"], "compressed-tensors")
            self.assertEqual(item["compute_dtype"], "bfloat16")

    def test_topology_has_four_real_links_with_independent_bidirectional_budgets(self):
        from orbitserve.experiments.capacity_native import build_topology
        spec = StudySpec()
        case = dict(cards_per_satellite=2, per_direction_gbps=200)
        topology = build_topology(case, spec)
        links = [link for link in topology.links if link.id.startswith("isl")]
        self.assertEqual(len(links), 4)
        self.assertTrue(all(link.bandwidth_gbps == 200 and link.bidirectional for link in links))
        self.assertEqual(sum(d.kind == "gpu" for d in topology.devices.values()), 8)

    def test_cleanup_reaps_all_resources_and_preserves_original_failure(self):
        from orbitserve.experiments.capacity_native import close_resources
        closed = []

        class Resource:
            def __init__(self, name, failure=None):
                self.name, self.failure = name, failure

            def close(self):
                closed.append(self.name)
                if self.failure:
                    raise RuntimeError(self.failure)

        resources = [Resource("scheduler", "secondary shutdown error"), Resource("network")]
        errors = []
        with self.assertRaisesRegex(ValueError, "original"):
            try:
                raise ValueError("original simulation failure")
            finally:
                close_resources(resources, errors)
        self.assertEqual(closed, ["scheduler", "network"])
        self.assertIn("secondary shutdown", errors[0])
