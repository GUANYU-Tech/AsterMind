"""Exercise the documented offline choices at their consuming configuration boundary."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from astermind.builder.config import SimulationConfig
from astermind.builder.physical.network_inputs import materialize_network_input
from astermind.engine import NetworkSpec, candidate_links
from examples.native.compute_comm_matrix import scene
from tests.unit.test_offline_scenario_config import make_scene, write_json

ROOT = Path(__file__).resolve().parents[2]
EPOCH = "2026-07-20T00:00:00.000000Z"
HARDWARE = ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
SCOPES = ("single_su", "intra_cross_su", "inter_satellite")
INTRA = ("ring", "tree", "fullmesh", "fc2", "mesh", "direct", "pairwise")
LAYERED = ("all_reduce", "all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv")
P2P = {"all_to_all": ("pairwise",), "broadcast": ("direct", "pairwise"),
       "send_recv": ("direct", "send_recv")}
CROSS = {
    "all_reduce": ("ring", "tree", "fullmesh"), "all_gather": ("ring", "fullmesh"),
    "all_gatherv": ("direct", "pairwise"), "reduce_scatterv": ("direct", "pairwise"),
    **P2P,
}


class SceneOptions(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "case.json"

    def load(self, raw):
        raw = deepcopy(raw)
        raw.update(schema_version="astermind.scenario.v1", time={"epoch_utc": EPOCH, "duration_s": 30})
        raw.pop("horizon_ns")
        write_json(self.path, raw)
        return SimulationConfig.load(self.path)

    def test_comm_scope_algorithm_matrix(self):
        for scope in SCOPES:
            operations = CROSS if scope == "inter_satellite" else {**dict.fromkeys(LAYERED, INTRA), **P2P}
            for operation, algorithms in operations.items():
                for algorithm in algorithms:
                    with self.subTest(scope=scope, operation=operation, algorithm=algorithm):
                        raw = scene("DS05")
                        raw["communication"]["cell_policies"] = {
                            "cellA": {"algorithms": {scope: {operation: algorithm}}}
                        }
                        config = self.load(raw)
                        self.assertIn("satA", config.topology().intra_hierarchies)
                        self.assertEqual(config.communication.cell_policies["cellA"].algorithms,
                                         {scope: {operation: algorithm}})

        for operation, algorithm in (("all_reduce", "fc2"), ("all_gather", "mesh"),
                                     ("reduce_scatter", "pairwise")):
            with self.subTest(rejected=(operation, algorithm)):
                raw = scene("DS05")
                raw["communication"]["cell_policies"] = {
                    "cellA": {"algorithms": {"inter_satellite": {operation: algorithm}}}
                }
                with self.assertRaisesRegex(ValueError, "unsupported Comm policy"):
                    self.load(raw)

    def test_comm_protocol_capacity_and_overrides(self):
        for protocol, basis in product(("payload_only", "inter_reference"),
                                       ("raw_line", "net_after_existing_fec", "application_effective")):
            with self.subTest(protocol=protocol, basis=basis):
                raw = scene("DS01")
                raw["communication"].update(
                    runtime_protocol=protocol, runtime_capacity_basis=basis, ground_processing_ns=500,
                    inter_parameters={"hop_processing_us": 17, "startup_us": 2},
                )
                policy = self.load(raw).communication
                self.assertEqual((policy.runtime_protocol, policy.runtime_capacity_basis), (protocol, basis))
                self.assertEqual(policy.ground_processing_ns, 500)
                self.assertEqual(policy.inter_parameters.hop_processing_us, 17)
                self.assertEqual(policy.inter_parameters.startup_us, 2)

        profile = {"name": "bench", "su": {"overrides": {"startup_us": 1.5, "wire_efficiency": 0.89}},
                   "so": {"overrides": {"ring_bw_bytes_per_us": 40_000, "phase_fixed_us": 2}}}
        write_json(self.root / "profile.json", profile)
        for setting in ({"intra_profile": profile}, {"intra_profile_file": "profile.json"}):
            raw = scene("DS05")
            raw["communication"].update(setting)
            configured = self.load(raw).communication.resolved_intra_profile
            self.assertEqual(configured.su.overrides.startup_us, 1.5)
            self.assertEqual(configured.su.overrides.wire_efficiency, 0.89)
            self.assertEqual(configured.so.overrides.ring_bw_bytes_per_us, 40_000)
            self.assertEqual(configured.so.overrides.phase_fixed_us, 2)

    def test_cell_models_and_native_compute_configuration(self):
        from astermind.common.bootstrap import activate
        root = activate()
        from astermind.engine.compute.cell_storage import capacity_inputs
        from astermind.engine.compute.runtime import MidlayerCellConfig, create_midlayer_runtime
        from src.kernel.hardware_spec import HWSpec

        for gpu, compute_mode, execution_mode in product(
            HARDWARE, ("midlayer", "roofline"), ("accelerated", "fast", "reference"),
        ):
            with self.subTest(gpu=gpu, compute_mode=compute_mode, mode=execution_mode):
                raw = scene("DS01", compute_mode=compute_mode)
                for cell in raw["cells"]:
                    cell.pop("hardware")
                    cell.update(hardware_id=gpu, execution_mode=execution_mode)
                cfg = self.load(raw)
                cell = cfg.cells[0]
                self.assertEqual(HWSpec(root / cell.hardware).hardware_id, gpu)
                self.assertEqual(cfg.communication.cell_policies[cell.id].hardware_id, gpu)
                self.assertEqual(cfg.compute_mode, compute_mode)
                self.assertEqual(cell.execution_mode, execution_mode)
                _, execution, _ = create_midlayer_runtime(MidlayerCellConfig(
                    root, model_file=cell.model, hardware_file=cell.hardware, hardware_id=gpu,
                    tensor_parallel_size=cell.parallel.tp, pipeline_parallel_size=cell.parallel.pp,
                    compute_mode=cfg.compute_mode, execution_mode=cell.execution_mode,
                ), with_scheduler=False)
                self.assertEqual((execution.prediction_method, execution.execution_mode),
                                 (compute_mode, execution_mode))

        custom = self.root / "hardware.yaml"
        custom.write_bytes((root / "configs/hardware/nvidia_a800.yaml").read_bytes())
        model = self.root / "model.json"
        model.write_bytes((root / "configs/model/qwen3-4b.json").read_bytes())
        raw = scene("DS01")
        for cell in raw["cells"]:
            cell.update(hardware_id="nvidia_a800", hardware=custom.name, model=model.name)
        cell = self.load(raw).cells[0]
        self.assertEqual((Path(cell.hardware), Path(cell.model)), (custom, model))

        for name, adapter, storage in (("DS01", "auto", True), ("DS01", "vllm_0_11", False),
                                       ("DS01", "vllm_0_31", True), ("DS03", "vllm_0_30", False),
                                       ("DS04", "vllm_0_11", False), ("DS07", "vllm_0_30", False),
                                       ("DS08", "vllm_0_11", False)):
            with self.subTest(deployment=name, adapter=adapter):
                raw = scene(name)
                raw["cells"][0]["scheduler_adapter"] = adapter
                cfg = self.load(raw)
                cfg.topology()
                cell = cfg.cells[0]
                options = MidlayerCellConfig(
                    root, model_file=cell.model, hardware_file=cell.hardware, hardware_id=cell.hardware_id,
                    scheduler_adapter=cell.scheduler_adapter, tensor_parallel_size=cell.parallel.tp,
                    pipeline_parallel_size=cell.parallel.pp, data_parallel_size=cell.parallel.dp,
                    enable_expert_parallel=cell.expert_parallel_enabled,
                    expert_routing_seed=cell.expert_routing_seed, deployment=cell.deployment,
                )
                _, execution, storage_enabled = create_midlayer_runtime(options, with_scheduler=False)
                self.assertEqual(storage_enabled, storage)
                self.assertEqual(set(execution.deployment.worker_pools),
                                 {"main"} if name != "DS08" else {"prefill", "decode"})

        hardware = "nvidia_h200"
        raw = scene("DS01")
        cell = raw["cells"][0]
        cell.update(model="configs/model/qwen3-4b.json", placements=cell["placements"][:1],
                    parallel={"tp": 1}, max_model_len=8192, max_batched_tokens=8192,
                    max_requests=256, num_kv_blocks=17, kv_block_size_tokens=16, prefix_caching=True,
                    watermark=0.15, scheduler_adapter="vllm_0_31",
                    memory_policy_file="configs/runtime_memory_policy/nvidia_h200/qwen3-4b.json",
                    host_launch_profile="configs/host_timing/host_launch_profiles/nvidia_h200_vllm011.json")
        loaded = self.load(raw).cells[0]
        options = MidlayerCellConfig(
            root, model_file=loaded.model, hardware_file=loaded.hardware, hardware_id=hardware,
            max_model_len=loaded.max_model_len, max_batched_tokens=loaded.max_batched_tokens,
            max_requests=loaded.max_requests, num_kv_blocks=loaded.num_kv_blocks,
            kv_block_size_tokens=loaded.kv_block_size_tokens, watermark=loaded.watermark,
            scheduler_adapter=loaded.scheduler_adapter, enable_prefix_caching=loaded.prefix_caching,
            memory_policy_file=loaded.memory_policy_file, host_launch_profile_file=loaded.host_launch_profile,
        )
        _, execution, storage = create_midlayer_runtime(options, with_scheduler=False)
        self.assertTrue(storage and execution.host_launch_profiles)
        self.assertEqual((options.max_model_len, options.max_batched_tokens, options.max_requests,
                          options.watermark), (8192, 8192, 256, 0.15))
        spec = HWSpec(root / loaded.hardware)
        from hashlib import sha256
        from dataclasses import replace

        evidence = {"source_kind": "runtime_configuration",
                    "source_reference": "Synthetic configuration-matrix fixture", "source_sha256": None}
        pool = {
            "pool_name": "main",
            "binding": {
                "hardware_id": hardware,
                "hardware_config_sha256": sha256((root / loaded.hardware).read_bytes()).hexdigest(),
                "model_config_sha256": execution.model.config_sha256,
                "deployment": "colocated", "worker_role": "colocated",
                "tensor_parallel_size": 1, "pipeline_parallel_size": 1, "expert_parallel_size": 1,
                "max_model_len": options.max_model_len, "max_batched_tokens": options.max_batched_tokens,
                "max_requests": options.max_requests, "kv_block_size_tokens": options.kv_block_size_tokens,
            },
            "gpu_memory_utilization": {"numerator": 4, "denominator": 5, "provenance": evidence},
            "non_kv_reserves": [
                {"kind": kind, "bytes": 1024, "provenance": evidence}
                for kind in ("activation", "cuda_graph", "communication", "allocator", "fragmentation", "framework")
            ],
            "reserve_scope": "assumed_uniform_across_pool_ranks_unverified",
            "completeness_assertion": "reserves_inherited_unverified_not_complete_for_bound_runtime",
        }
        custom_policy = self.root / "kv-budget.json"
        write_json(custom_policy, {"schema_version": 2, "scheduler_pools": [pool]})
        cell["memory_policy_file"] = custom_policy.name
        resolved = self.load(raw).cells[0].memory_policy_file
        self.assertEqual(Path(resolved), custom_policy)
        capacity = capacity_inputs(replace(options, memory_policy_file=resolved), execution.model, spec)
        self.assertEqual(capacity["source_mode"], "analytical_current_vllm_allocator")
        self.assertGreater(capacity["available_kv_bytes"], 0)
        explicit = capacity_inputs(replace(options, memory_policy_file=None, kv_block_size_tokens=32),
                                   execution.model, spec)
        self.assertEqual(explicit["num_kv_blocks"], 17)
        self.assertEqual(explicit["kv_block_bytes"], capacity["kv_block_bytes"] * 2)

    def test_chunked_prefill_cell_options(self):
        raw = scene("DS07-pp1")
        cell = self.load(raw).cells[0]
        self.assertIsNone(cell.enable_chunked_prefill)
        self.assertEqual(cell.long_prefill_token_threshold, 0)

        raw["cells"][0].update(enable_chunked_prefill=True, long_prefill_token_threshold=128)
        cell = self.load(raw).cells[0]
        self.assertTrue(cell.enable_chunked_prefill)
        self.assertEqual(cell.long_prefill_token_threshold, 128)

        raw["cells"][0]["enable_chunked_prefill"] = False
        with self.assertRaisesRegex(ValueError, "requires enable_chunked_prefill=true"):
            self.load(raw)
        raw["cells"][0]["long_prefill_token_threshold"] = 0
        self.assertFalse(self.load(raw).cells[0].enable_chunked_prefill)

    def test_egress_topology_styles_and_roles(self):
        for mode in ("single_gateway", "multi_gateway", "distributed"):
            raw = scene("DS01" if mode == "single_gateway" else "DS03")
            for cell in raw["cells"]:
                cell["egress_mode"] = mode
            config = self.load(raw)
            self.assertEqual(config.cells[0].egress_mode, mode)
            self.assertEqual(config.topology().devices["user"].kind, "terminal")

        for kind, count in (("fc", 6), ("ring", 4), ("mesh", 4), ("switch", 4)):
            with self.subTest(template=kind):
                raw = scene("DS05")
                physical = raw["physical"]
                members = [f"satA-gpu-{i}" for i in range(4)]
                if kind == "switch":
                    physical["links"] = [link for link in physical["links"]
                                         if not (link["source"] in members and link["target"] == "satA-switch-0")]
                template = dict(id="p", kind=kind, devices=members, bandwidth_gbps=321, latency_ns=75)
                if kind == "switch":
                    template["switch"] = "satA-switch-0"
                if kind == "mesh":
                    template.update(rows=2, columns=2)
                physical["templates"] = [template]
                topology = self.load(raw).topology()
                links = [link for link in topology.links if link.id.startswith("p.")]
                self.assertEqual(len(links), count)
                self.assertTrue(all((link.bandwidth_gbps, link.latency_ns) == (321, 75) for link in links))
                self.assertEqual(topology.intra_hierarchies["satA"]["su_groups"][0], tuple(members))
                bridge = next(link for link in topology.links if link.id == "satA-su-bridge")
                self.assertEqual((bridge.bandwidth_gbps, bridge.latency_ns), (400, 500))

    def test_cell_selection_and_router_options(self):
        for choice, route in product(("least_loaded", "nearest", "estimated_arrival"),
                                     ("min_hops", "min_propagation")):
            raw = scene("DS01")
            raw.update(cell_selection=choice, routing={"policy": route})
            config = self.load(raw)
            self.assertEqual((config.cell_selection, config.routing.policy), (choice, route))
            from astermind.engine.network_host import DeviceRuntimeHost

            host = DeviceRuntimeHost(config.topology(), config)
            try:
                self.assertEqual(host.spec.routing_policy, route)
            finally:
                host.close()

        raw = scene("DS01")
        raw["routing"] = {"policy": "earliest_arrival"}
        with self.assertRaisesRegex(ValueError, "Unregistered routing policy"):
            self.load(raw)

    def test_workload_time_regions_and_history_choices(self):
        path = make_scene(self.root / "workload")
        root = path.parent
        manifest_path = root / "workload_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["gateway_regions"] = {"ground": ["demo"]}
        write_json(manifest_path, manifest)

        network_path = root / "network.json"
        native_network = json.loads(network_path.read_text(encoding="utf-8"))
        for ground_kind in ("gateway", "user"):
            network = deepcopy(native_network)
            network["ground_ingresses"][0].update(ground_kind=ground_kind, altitude_km=0.25)
            node = next(item for item in network["network"]["nodes"] if item["role"] == "ground")
            network["ground_ingresses"][0]["terminal_count"] = node.pop("sgl_terminals")
            write_json(network_path, network)
            configured = SimulationConfig.load(path)
            spec = NetworkSpec.from_value(configured.physical_network["network"])
            station = next(item for item in spec.nodes if item.role == "ground")
            self.assertEqual((station.ground_kind, station.sgl_terminals), (ground_kind, 2))
            geometry = configured.physical_network["ground_ingresses"][0]
            self.assertEqual(geometry["altitude_km"], 0.25)
            self.assertNotIn("ground_kind", geometry)

        schedule_path = root / "schedule.json"
        original = json.loads(schedule_path.read_text(encoding="utf-8"))
        rows_path = root / "requests.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        rows[0]["output_tokens"] = 1
        rows[1]["history_tokens"] = 32
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        for upload, prefix in ((0, 32), (16, 16), (32, 0)):
            for mode in ("copy", "move"):
                with self.subTest(upload=upload, mode=mode):
                    plan = deepcopy(original)
                    plan["migrations"][0]["mode"] = mode
                    plan["requests"]["context1"]["history_upload_tokens"] = upload
                    write_json(schedule_path, plan)
                    cfg = SimulationConfig.load(path)
                    self.assertEqual((cfg.requests[1].input_bytes, cfg.requests[1].prefix_hit_tokens),
                                     ((14 + upload) * 5, prefix))
                    self.assertEqual(cfg.migrations[0].mode, mode)

        manifest["gateway_regions"] = {"ground": []}
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "does not serve region"):
            SimulationConfig.load(path)


    def test_optional_absent_schedule_and_scene_time_constraints(self):
        path = make_scene(self.root / "inputs")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("schedule")
        write_json(path, raw)
        cfg = SimulationConfig.load(path)
        self.assertIsNone(cfg.requests[0].cell)
        self.assertEqual(cfg.requests[1].input_bytes, 240)
        self.assertFalse(cfg.migrations)

        for value, error in ((0, "positive"), (0.0000000001, "nanosecond"),
                             (3, "arrival exceeds")):
            with self.subTest(duration_s=value):
                raw["time"]["duration_s"] = value
                write_json(path, raw)
                with self.assertRaisesRegex(ValueError, error):
                    SimulationConfig.load(path)


class NetworkOptions(unittest.TestCase):
    def fixture(self):
        return json.loads((ROOT / "tests/fixtures/network/coupled_small.json").read_text())

    def test_network_rule_options_and_acquisition_parameters(self):
        ranked = ("nearest_distance", "highest_elevation", "highest_capacity", "longest_available")
        value = self.fixture()
        for ground, satellite, matching in product(ranked, ranked, ("greedy", "deferred_acceptance")):
            raw = deepcopy(value)
            raw["network"].update(ground_access_policy=ground, satellite_access_policy=satellite,
                                  ground_matching=matching)
            spec = NetworkSpec.from_value(materialize_network_input(raw)["network"])
            self.assertEqual((spec.ground_access_policy, spec.satellite_access_policy,
                              spec.ground_matching), (ground, satellite, matching))
            self.assertTrue(any(edge["kind"] == "sgl" for edge in candidate_links(spec)))

        for formation, direct, cross, handover in product(
            ("ring", "fully_connected"), (False, True),
            ("nearest_distance", "highest_capacity", "longest_available"),
            ("retain_valid", "quality_first"),
        ):
            raw = self.fixture()
            raw["constellations"]["compute_formations"][0]["topology"] = formation
            raw["network"].update(inter_cell_links=direct, handover_policy=handover,
                                  cross_layer_access_policy=cross)
            raw["network"]["link_profiles"]["inter_cell"] = {"bandwidth_bps": 6e9, "setup_time_s": 2}
            spec = NetworkSpec.from_value(materialize_network_input(raw)["network"])
            edges = candidate_links(spec)
            self.assertEqual(set(spec.cell_topologies.values()), {formation})
            self.assertEqual(sum(edge["kind"] == "cell_isl" for edge in edges),
                             8 if formation == "ring" else 12)
            self.assertEqual(any(edge["kind"] == "inter_cell" for edge in edges), direct)
            self.assertEqual((spec.cross_layer_access_policy, spec.handover_policy), (cross, handover))

    def test_communication_grid_wrap_offset_and_explicit_edges(self):
        original = self.fixture()
        sample = NetworkSpec.from_value(materialize_network_input(original)["network"])
        neighbors = [node.node_id for node in sample.nodes if node.role == "communication"]
        baseline = None
        for topology in ("grid", "honeycomb", "explicit"):
            raw = deepcopy(original)
            shell = raw["constellations"]["communication_shells"][0]
            shell["topology"] = topology
            if topology == "explicit":
                shell["explicit_pairs"] = [neighbors[:2]]
            elif topology == "grid":
                shell.update(wrap_planes=False, wrap_slots=False, cross_plane_offset=1)
            spec = NetworkSpec.from_value(materialize_network_input(raw)["network"])
            edges = [edge for edge in candidate_links(spec) if edge["kind"] == "communication_isl"]
            if topology == "grid":
                self.assertEqual(len(edges), 35)
                baseline = set(tuple(edge[k] for k in ("endpoint_a", "endpoint_b")) for edge in edges)
            elif topology == "honeycomb":
                self.assertEqual(len(edges), 36)
                self.assertNotEqual(set(tuple(edge[k] for k in ("endpoint_a", "endpoint_b")) for edge in edges), baseline)
            else:
                self.assertEqual([(edge["endpoint_a"], edge["endpoint_b"]) for edge in edges], [tuple(sorted(neighbors[:2]))])

    def test_fixed_lvlh_offsets_and_formation_clearance_inputs(self):
        value = self.fixture()
        cluster = value["constellations"]["compute_formations"][0]["cluster"]
        cluster.update(
            motion_model="reference_fixed_lvlh",
            member_offsets_m=[[0, n * 500, 0] for n in range(4)],
            d_keepout_m=200, sigma_navigation_m=2, sigma_control_m=3,
        )
        concrete = materialize_network_input(value)
        rows = [row for row in concrete["elements"]["satellites"]
                if "fixed_lvlh_offset_km" in row]
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[1]["fixed_lvlh_offset_km"], [0.0, 0.5, 0.0])
        self.assertGreater(concrete["initialization_provenance"][-1]["keepout_m"], 200)
        cluster["member_offsets_m"][1] = [0, 50, 0]
        with self.assertRaisesRegex(ValueError, "exceed keepout"):
            materialize_network_input(value)

    def test_optical_radio_fixed_and_islsim_capacity_choices(self):
        from astermind.engine import LinkProfile

        base = dict(bandwidth_bps=6e9, setup_time_s=1.25, max_distance_km=8000,
                    capacity_schedule=((0, .5), (5, .75)))
        overrides = {
            "fixed": {},
            "free_space_shannon": dict(frequency_hz=20e9, channel_bandwidth_hz=250e6,
                                       transmit_power_w=10, transmit_gain_db=30, receive_gain_db=30,
                                       noise_temperature_k=280, system_loss_db=1, implementation_efficiency=.8),
            "optical_photon_budget": dict(wavelength_m=1550e-9, transmit_power_w=.1,
                                          transmit_aperture_m=.05, receive_aperture_m=.05,
                                          required_photons_per_bit=20, pointing_loss_db=3),
            "islsim": {},
        }
        for model, values in overrides.items():
            with self.subTest(model=model):
                profile = LinkProfile(capacity_model=model, sun_exclusion_angle_deg=3, **base, **values)
                self.assertEqual((profile.setup_time_s, profile.max_distance_km,
                                  profile.sun_exclusion_angle_deg), (1.25, 8000, 3))
                self.assertEqual(profile.bandwidth_at(5), 4.5e9)
                if model != "islsim":
                    current = profile.attributes(550, 0)["bandwidth_bps"]
                    self.assertGreater(current, 0)
                    self.assertLessEqual(current, 3e9)
                else:
                    self.assertTrue(profile.islsim_config)

        raw = self.fixture()
        profiles = raw["network"]["link_profiles"]
        profiles["up"] = dict(profiles["sgl"], bandwidth_bps=4e9)
        profiles["down"] = dict(profiles["sgl"], bandwidth_bps=2e9)
        profiles["sgl"].update(uplink_profile="up", downlink_profile="down")
        spec = NetworkSpec.from_value(materialize_network_input(raw)["network"])
        self.assertEqual(spec.link_profiles["sgl"].uplink_profile, "up")
        self.assertEqual(spec.link_profiles["sgl"].downlink_profile, "down")
