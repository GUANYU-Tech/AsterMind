"""Offer only supported native combinations and validate generated choices."""

import json
from pathlib import Path
import tempfile
import unittest

from simulator.catalog import (
    cell_member_counts,
    parallel_choices,
    scene_catalog,
    validate_deployment_choice,
)
from simulator.scene import SceneDeployment


class SceneCatalogTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "configs/model").mkdir(parents=True)
        (self.root / "configs/hardware").mkdir()
        for hardware in ("nvidia_a800", "nvidia_h200", "ascend_910b", "metax_c500"):
            (self.root / f"configs/hardware/{hardware}.yaml").touch()
        self.dense = dict(
            model_type="qwen3",
            torch_dtype="bfloat16",
            num_hidden_layers=32,
            tie_word_embeddings=False,
            max_position_embeddings=40960,
        )
        (self.root / "configs/model/qwen3-8b.json").write_text(json.dumps(self.dense))

    def test_inventory_reads_source_and_has_correct_hardware_files(self):
        (self.root / "configs/hardware/nvidia_h200.yaml").write_text(
            "interconnect:\n  nvlink_unidir_bw_GB_s: 450\n"
        )
        value = scene_catalog(self.root)
        self.assertEqual(len(value["hardware"]), 4)
        self.assertTrue(value["models"][0]["available"])
        self.assertEqual(
            value["hardware"][1]["file"], "configs/hardware/nvidia_h200.yaml"
        )
        self.assertEqual(value["hardware"][1]["gpu_link_bandwidth_gbps"], 3600)
        self.assertEqual(value["hardware"][1]["gpu_link_latency_ns"], 0)

    def test_cell_capacity_counts_only_compute_members(self):
        nodes = [
            dict(node_id="a", role="compute", cell_id="cellA"),
            dict(node_id="b", role="compute", cell_id="cellA"),
            dict(node_id="c", role="compute", cell_id=None),
            dict(node_id="relay", role="communication", cell_id=None),
        ]
        (self.root / "materialized_input.json").write_text(
            json.dumps(dict(network=dict(nodes=nodes)))
        )
        self.assertEqual(cell_member_counts(self.root), {"cellA": 2, "c": 1})

    def test_pipeline_uses_uniform_layers_and_tied_models_keep_pp_one(self):
        choices = parallel_choices(self.dense, "nvidia_a800")
        self.assertIn(dict(tp=1, pp=4, expert_parallel=False), choices)
        self.assertIn(dict(tp=2, pp=2, expert_parallel=False), choices)
        self.assertNotIn(dict(tp=1, pp=3, expert_parallel=False), choices)
        tied = {**self.dense, "tie_word_embeddings": True}
        self.assertTrue(
            all(row["pp"] == 1 for row in parallel_choices(tied, "nvidia_a800"))
        )

    def test_hardware_and_model_limits_follow_native_adapter(self):
        for hardware in ("nvidia_h200", "metax_c500", "ascend_910b"):
            self.assertEqual(
                parallel_choices(self.dense, hardware),
                [dict(tp=1, pp=1, expert_parallel=False)],
            )
        self.assertEqual(
            parallel_choices({**self.dense, "torch_dtype": "float16"}, "nvidia_a800"),
            [],
        )
        self.assertEqual(
            parallel_choices(
                {**self.dense, "model_type": "glm_moe_dsa"}, "nvidia_a800"
            ),
            [],
        )
        self.assertEqual(
            parallel_choices({**self.dense, "model_type": "llama"}, "nvidia_a800"),
            [dict(tp=1, pp=1, expert_parallel=False)],
        )

    def test_moe_reuses_tp_two_and_requires_explicit_routing_seed(self):
        moe = {**self.dense, "model_type": "qwen3_moe", "num_experts": 128}
        (self.root / "configs/model/qwen3-30b-a3b.json").write_text(json.dumps(moe))
        self.assertEqual(
            parallel_choices(moe, "nvidia_a800"),
            [dict(tp=2, pp=1, expert_parallel=True)],
        )
        self.assertEqual(parallel_choices(moe, "nvidia_h200"), [])
        value = SceneDeployment(
            model="configs/model/qwen3-30b-a3b.json", tp=2, pp=1, expert_parallel=True
        )
        with self.assertRaisesRegex(ValueError, "种子"):
            validate_deployment_choice(value, self.root)
        validate_deployment_choice(
            value.model_copy(update=dict(expert_routing_seed=0)), self.root
        )

    def test_manual_mismatch_and_invalid_parallelism_rejected(self):
        validate_deployment_choice(SceneDeployment(tp=1, pp=4), self.root)
        for update in (
            dict(hardware_id="nvidia_h200"),
            dict(tp=4, pp=1),
            dict(tp=1, pp=3),
            dict(max_model_len=65536),
            dict(expert_routing_seed=0),
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_deployment_choice(SceneDeployment(**update), self.root)
