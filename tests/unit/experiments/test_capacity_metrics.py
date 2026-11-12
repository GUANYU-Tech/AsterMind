from pathlib import Path
import tempfile
import unittest

from orbitserve.experiments.capacity_metrics import analyze, trial_metrics
from orbitserve.experiments.capacity_study import StudySpec, Workload, plan_cases, write_json


def record(arrival, completed=True):
    value = dict(arrival_s=arrival, input_tokens=32, output_tokens=4)
    if completed:
        value.update(first_received_s=arrival + 0.01, completed_s=arrival + 0.04)
    return value


class CapacityMetricTests(unittest.TestCase):
    def setUp(self):
        self.spec = StudySpec(warmup_s=1, measurement_s=4, min_measured_requests=4)

    def test_missing_requests_remain_in_attainment_denominator(self):
        records = [record(1 + i / 5, i < 5) for i in range(10)]
        metrics = trial_metrics(records, self.spec, 5)
        self.assertEqual(metrics["incomplete_cohort"], 5)
        self.assertEqual(metrics["targets"]["interactive"]["attainment"], 0.5)
        self.assertFalse(metrics["targets"]["interactive"]["sustainable"])
        self.assertAlmostEqual(metrics["ttft_p95_s"], 0.01)

    def test_drain_completions_do_not_inflate_measured_throughput(self):
        records = [record(1 + i / 5) for i in range(10)]
        records.append(dict(arrival_s=4.9, output_tokens=4, first_received_s=5.1, completed_s=5.2))
        metrics = trial_metrics(records, self.spec, 5)
        self.assertEqual(metrics["measured_arrivals"], 11)
        self.assertEqual(metrics["completed_requests_per_s"], 2.5)
        self.assertEqual(metrics["output_tokens_per_s"], 10)

    def test_unfinished_requests_detect_queue_growth(self):
        records = [record(1 + i / 10, i < 20) for i in range(40)]
        metrics = trial_metrics(records, self.spec, 10)
        self.assertGreater(metrics["backlog_end"], metrics["backlog_midpoint"])
        self.assertFalse(metrics["queue_stable"])

    def test_small_population_cannot_claim_capacity(self):
        metrics = trial_metrics([record(2)], self.spec, 1)
        self.assertFalse(metrics["sufficient_population"])
        self.assertFalse(metrics["targets"]["interactive"]["sustainable"])

    def test_search_ceiling_is_lower_bound_and_missing_seeds_are_not_ignored(self):
        spec = StudySpec(models=("qwen3-4b",), hardware=("nvidia_a800",),
                         cards_per_satellite=(1,), layouts=("cross_replicas",),
                         link_bandwidth_gbps=(400,), workloads=(Workload(name="tiny", input_tokens=32, output_tokens=4),),
                         request_rates=(1, 2), seeds=(17, 29), min_measured_requests=1)
        case = plan_cases(spec, {"qwen3-4b": {}})[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "plan.json", dict(spec=spec.model_dump(mode="json"), cases=[case], assumptions=[]))
            for seed in spec.seeds:
                trial = root / "trials" / str(seed)
                write_json(trial / "input.json", dict(case=case, rate=1, seed=seed))
                write_json(trial / "result.json", dict(status="evaluated", metrics={"sufficient_population": True, "targets": {
                    t.name: dict(sustainable=True, attainment=1) for t in spec.targets}}))
            result = analyze(root, root / "analysis", plots=False)
            self.assertFalse(result["complete"])
            row = result["curves"][0]
            self.assertEqual(row["max_tested_sustainable_rps"], 1)
            self.assertTrue(row["capacity_is_lower_bound"])
            self.assertEqual(row["missing_rates"], [2])
            self.assertFalse(result["recommendations"][0]["final_hardware_recommendation"])
            self.assertTrue((root / "analysis/trials.csv").is_file())
            # A failed seed at a higher rate brackets the result only when every seed ran.
            for seed in spec.seeds:
                trial = root / "trials" / f"r2-{seed}"
                write_json(trial / "input.json", dict(case=case, rate=2, seed=seed))
                write_json(trial / "result.json", dict(status="evaluated", metrics={"sufficient_population": True, "targets": {
                    t.name: dict(sustainable=seed == 17, attainment=1 if seed == 17 else 0.5)
                    for t in spec.targets}}))
            result = analyze(root, root / "analysis", plots=False)
            self.assertTrue(result["complete"])
            row = result["curves"][0]
            self.assertFalse(row["capacity_is_lower_bound"])
            self.assertEqual(row["next_tested_failing_rps"], 2)
            for seed in spec.seeds:
                trial = root / "trials" / f"r2-{seed}"
                write_json(trial / "result.json", dict(status="evaluated", metrics={
                    "sufficient_population": False, "targets": {
                        t.name: dict(sustainable=False, attainment=1) for t in spec.targets}}))
            result = analyze(root, root / "analysis", plots=False)
            row = result["curves"][0]
            self.assertIsNone(row["next_tested_failing_rps"])
            self.assertEqual(row["inconclusive_rates"], [2])
