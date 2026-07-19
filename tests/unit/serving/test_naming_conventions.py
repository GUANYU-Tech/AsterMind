import json
import sys
import unittest
from pathlib import Path


from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.architecture.hardware import hardware_from_single_satellite_case
from orbitserve.architecture.hardware import CommunicationSpec
from orbitserve.serving.capacity import (
    StageServingProfile,
    ServingSpec,
    estimate_serving_throughput,
)
from orbitserve.architecture.single_satellite_cost.models import CommunicationConfig
from orbitserve.workload import TokenDistribution


ROOT = Path(__file__).resolve().parents[3]


def h200_hardware():
    return hardware_from_single_satellite_case(ROOT / "examples/single_satellite/h200_gpu_derived.json")


def chat_workload():
    return TokenDistribution(
            prompt_tokens_p50=512,
            prompt_tokens_p95=2048,
            output_tokens_p50=128,
            output_tokens_p95=512,
    )


class ServingNamingConventionTests(unittest.TestCase):


    def test_json_and_dse_outputs_prefer_canonical_fields(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_dse.json")
        payload = run_experiment(spec).to_dict()
        self.assertEqual(payload["module_status"]["serving_throughput"], "evaluated")
        self.assertNotIn("serving_capacity", payload["module_status"])
        self.assertIn("request_throughput_per_s", payload["performance"])
        self.assertIn("prefill_tokens_per_s", payload["performance"])
        self.assertIn("decode_tokens_per_s", payload["performance"])
        self.assertNotIn("throughput_req_s", payload["performance"])
        self.assertNotIn("prefill_throughput_tok_s", payload["performance"])

        first_row = payload["dse"]["rows"][0]
        self.assertIn("request_throughput_per_s", first_row["performance"])
        self.assertNotIn("throughput_req_s", first_row["performance"])
        json.dumps(payload, ensure_ascii=False, allow_nan=False)

    def test_serving_outputs_use_capacity_only_for_storage_like_fields(self):
        result = estimate_serving_throughput(
            ServingSpec.from_dict({"model": {"preset": "llama3-8b"}}),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )
        payload = result.to_dict()
        capacity_keys = sorted(key for key in _walk_keys(payload) if "capacity" in key)
        self.assertTrue(capacity_keys)
        for key in capacity_keys:
            self.assertTrue(_is_storage_like_capacity_key(key), key)

        performance = result.to_performance_metrics().to_dict()
        capacity_metric_keys = sorted(key for key in _walk_keys(performance) if "capacity" in key)
        self.assertTrue(capacity_metric_keys)
        for key in capacity_metric_keys:
            self.assertTrue(_is_storage_like_capacity_key(key), key)

    def test_bit_byte_units_are_explicit_and_consistent(self):
        hardware = h200_hardware()
        result = estimate_serving_throughput(
            ServingSpec.from_dict(
                {
                    "model": {"preset": "deepseek-moe-671b"},
                    "parallelism": {"strategy": "expert_parallel", "ep": 4, "max_satellites": 4},
                }
            ),
            hardware=hardware,
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        stage = result.provenance["prefill_stage"]
        self.assertIn("ep_communication_bytes", stage)
        self.assertIn("ep_communication_time_s", stage)
        self.assertGreaterEqual(stage["ep_communication_bytes"], 0)
        self.assertGreaterEqual(stage["ep_communication_time_s"], 0)
        self.assertIn("effective_isl_bandwidth_gbps", stage)



def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _is_storage_like_capacity_key(key: str) -> bool:
    return any(
        marker in key
        for marker in (
            "memory_capacity",
            "fast_mem_capacity",
            "storage_capacity",
            "kv_cache_capacity",
        )
    )


def _tracked_text_files():
    roots = ("src", "tests", "docs/research/parameters", "examples", "README.md")
    for root in roots:
        path = ROOT / root
        if path.is_file():
            yield path
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix in {".py", ".md", ".json"}:
                yield child


if __name__ == "__main__":
    unittest.main()
