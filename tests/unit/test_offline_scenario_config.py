"""Regression checks for file-based configuration before native initialization."""

from copy import deepcopy
import json
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
import unittest

from astermind.builder.config import SimulationConfig

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "native"
EPOCH = "2026-07-20T00:00:00.000000Z"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def make_scene(root):
    copytree(EXAMPLES / "offline_config", root, dirs_exist_ok=True)
    return root / "scenario.json"


def make_automatic_scene(path, dataset):
    scene = json.loads(path.read_text(encoding="utf-8"))
    scene["physical_network"] = str(dataset)
    terminals = {"user", "isl0", "isl1"}
    scene["physical"]["devices"] = [
        device for device in scene["physical"]["devices"]
        if device["kind"] == "gpu" or device["id"] in terminals
    ]
    scene["physical"].pop("links")
    for cell in scene["cells"]:
        cell["external_terminals"] = [
            identifier for identifier in cell["external_terminals"] if identifier in terminals
        ]
    path = path.with_name("automatic.json")
    write_json(path, scene)
    return path


class OfflineScenarioConfigTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.scene_path = make_scene(self.root)

    def test_network_file_workload_deployment_and_comm_lower_to_native_fields(self):
        config = SimulationConfig.load(self.scene_path)
        topology = config.topology()
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.horizon_ns, 10_000_000_000)
        self.assertEqual(config.ground_owners, ("ground",))
        self.assertEqual(config.physical_network["time"], {
            "epoch_utc": EPOCH, "duration_s": 10, "step_s": 0.25,
        })
        self.assertEqual(topology.satellite_to_cell["satA"], "cellA")
        self.assertEqual(topology.satellite_to_cell["satB"], "cellB")
        self.assertEqual(config.cells[0].hardware, "configs/hardware/nvidia_a800.yaml")
        self.assertEqual(config.cells[0].parallel.tp, 2)
        self.assertEqual(config.cells[0].placements[0].device, "gpu0")
        self.assertEqual(config.cells[0].ingress_device, "gpu0")
        self.assertIn("isl0", config.cells[0].external_terminals)
        self.assertTrue(config.cells[0].prefix_caching)
        policy = config.communication.cell_policies["cellA"]
        self.assertEqual(policy.hardware_id, "nvidia_a800")
        self.assertEqual(policy.algorithms["single_su"]["all_reduce"], "ring")
        self.assertEqual(config.communication.gpu_hardware["gpu0"], "nvidia_a800")
        self.assertEqual(config.communication.runtime_protocol, "payload_only")
        self.assertEqual(config.communication.runtime_capacity_basis, "application_effective")
        self.assertEqual(config.routing.policy, "min_propagation")
        self.assertEqual(config.runtime_network["faults"][0]["start_s"], 0.25)

        first, second = config.requests
        self.assertEqual((first.cell, first.user_device, first.input_bytes), ("cellA", "user", 160))
        self.assertEqual((second.cell, second.arrival_ns), ("cellB", 4_000_000_000))
        self.assertEqual((second.input_tokens, second.prefix_hit_tokens, second.input_bytes), (48, 32, 80))
        self.assertEqual(second.output_bytes_per_token, 7)
        self.assertEqual(config.predecessor(second), "context0")
        self.assertEqual(config.migrations[0].before_request, second.id)
        self.assertEqual(config.migrations[0].context_id, first.context_id)

        raw = json.loads(self.scene_path.read_text(encoding="utf-8"))
        from astermind.builder.config import ServingCell
        from astermind.builder.config.resources import Link

        expected_cells = [ServingCell.model_validate(cell) for cell in raw["cells"]]
        self.assertEqual(list(config.cells), expected_cells)
        self.assertEqual(list(topology.links), [
            Link.model_validate(link) for link in raw["physical"]["links"]
        ])

    def test_physical_dataset_manifest_uses_native_dataset_reference(self):
        prepared = self.root / "prepared"
        prepared.mkdir()
        write_json(prepared / "physical_dataset.json", {"schema_version": "orbitserve.physical_dataset.v2"})
        write_json(prepared / "materialized_input.json", {
            "time": {"epoch_utc": EPOCH, "duration_s": 20},
            "ground_ingresses": [{"ingress_id": "ground"}],
        })
        scene = json.loads(self.scene_path.read_text(encoding="utf-8"))
        scene["physical_network"] = "prepared/physical_dataset.json"
        write_json(self.scene_path, scene)
        config = SimulationConfig.load(self.scene_path)
        self.assertEqual(Path(config.physical_dataset), prepared / "physical_dataset.json")
        self.assertIsNone(config.physical_network)
        scene["time"]["duration_s"] = 21
        write_json(self.scene_path, scene)
        with self.assertRaisesRegex(ValueError, "exceeds the precomputed"):
            SimulationConfig.load(self.scene_path)

    def test_prepared_dataset_can_bind_default_terminal_paths(self):
        from astermind.builder.physical.network import run_network_experiment

        source = SimulationConfig.load(self.scene_path)
        prepared = self.root / "actual-physical"
        result = run_network_experiment(
            source.physical_network, prepared, _workers=1, export_mode="core"
        )
        self.assertTrue(result["checks"]["passed"])
        self.assertEqual(result["checks"]["snapshot_count"], 40)

        auto = make_automatic_scene(self.scene_path, prepared / "physical_dataset.json")
        config = SimulationConfig.load(auto)
        topology = config.topology()
        self.assertEqual(set(config.physical_link_bindings.values()), {
            "sgl:ground:satA", "sgl:ground:satB", "communication_isl:satA:satB",
        })
        links = {link.id: link for link in topology.links}
        self.assertEqual(links["gpu0/port"].target, "satA/fabric")
        self.assertEqual(links["satA/egress"].target, "isl0")
        self.assertEqual(topology.devices["isl0"].max_external_links, 2)
        self.assertEqual(links["network:communication_isl:satA:satB"].source, "isl0")
        self.assertEqual(config.requests[1].input_bytes, 80)

    def test_invalid_region_closed_loop_and_unknown_schedule_are_rejected(self):
        path = self.root / "requests.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for update, message in (
            ({"region_id": "unknown"}, "unknown region"),
            ({"slo_class": "unknown"}, "unknown SLO class"),
            ({"wait_after_response_s": 0.2}, "closed-loop arrivals require runtime support"),
            ({"gateway_id": "missing"}, "missing"),
        ):
            with self.subTest(update=update):
                data = deepcopy(original)
                data[1].update(update)
                path.write_text("".join(json.dumps(row) + "\n" for row in data))
                with self.assertRaisesRegex(ValueError, message):
                    SimulationConfig.load(self.scene_path)
        path.write_text("".join(json.dumps(row) + "\n" for row in original))
        schedule_path = self.root / "schedule.json"
        schedule = json.loads(schedule_path.read_text())
        schedule["requests"]["unknown"] = {"cell": "cellA"}
        write_json(schedule_path, schedule)
        with self.assertRaisesRegex(ValueError, "schedule references unknown requests"):
            SimulationConfig.load(self.scene_path)

    def test_planned_reuse_requires_native_blocks_and_a_cross_cell_dependency(self):
        path = self.root / "schedule.json"
        original = json.loads(path.read_text())
        plan = deepcopy(original)
        plan["requests"]["context1"]["history_upload_tokens"] = 1
        write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "complete blocks"):
            SimulationConfig.load(self.scene_path)

        plan = deepcopy(original)
        plan["migrations"][0].pop("before_request")
        write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "migration dependency"):
            SimulationConfig.load(self.scene_path)


    def test_legacy_path_preserves_existing_comm_defaults(self):
        legacy = SimulationConfig.load(EXAMPLES / "dynamic_tp2_pp2.json")
        self.assertTrue(Path(legacy.physical_network).is_absolute())
        self.assertEqual(legacy.communication.default_hardware_id, "nvidia_h200")
        self.assertEqual(legacy.communication.gpu_hardware, {})
