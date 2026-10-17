"""Test suite 4: HWSpec parsing — each yaml → correct resource capacities."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from src.kernel.hardware_spec import HWSpec
from src.kernel.model_contract import (
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
    TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
    TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR,
)


@pytest.mark.parametrize(
    "config_path",
    sorted(
        os.path.join("configs", "hardware", name)
        for name in os.listdir("configs/hardware")
        if name.endswith(".yaml")
    ),
)
def test_all_hardware_configs_declare_supported_kernel_models(config_path):
    import yaml

    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    hardware = HWSpec(config_path)
    assert hardware.kernel_memory_model in {
        "gpu_mshr", "npu_tile", "pim_bank", "systolic_scratchpad"}


class TestHWSpecA800:
    def setup_method(self):
        self.hw = HWSpec("configs/hardware/nvidia_a800.yaml")

    def test_loads_without_error(self):
        assert self.hw is not None

    def test_num_compute_units(self):
        assert self.hw.num_compute_units == 108

    def test_grouped_gemm_pipeline_binds_role_specific_residency(self):
        pipeline = self.hw.grouped_gemm_tile_pipeline

        assert pipeline["registers_per_thread_by_variant"] == {
            "w1": 40,
            "w2": 48,
        }
        assert pipeline["resident_ctas_per_sm_by_variant"] == {
            "w1": 12,
            "w2": 10,
        }

    def test_grouped_gemm_pipeline_rejects_w1_residency_mismatch(
            self, tmp_path):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            config = yaml.safe_load(stream)
        config["scheduling"]["grouped_gemm_tile_pipeline"][
            "resident_ctas_per_sm_by_variant"]["w1"] = 10
        path = tmp_path / "invalid_grouped_gemm_w1_residency.yaml"
        path.write_text(yaml.safe_dump(config))

        with pytest.raises(ValueError, match="must equal 12"):
            HWSpec(str(path))

    def test_staged_ldgsts_l2_read_service_is_source_bounded(self):
        resource = self.hw.resource_by_role("staged_ldgsts_l2_read")
        reference_path = Path(
            "microbench/nvidia_a800/references/"
            "staged_ldgsts_l2_service_sm80.json")
        with reference_path.open() as stream:
            reference = json.load(stream)

        assert resource.scope == "shared"
        assert resource.effective_capacity(
            self.hw.num_compute_units,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        ) == pytest.approx(
            reference["selected_service"][
                "logical_byte_capacity_per_s"])
        assert not reference["production_or_evaluation_duration_consumed"]

    def test_k32_depth6_ldgsts_l2_read_service_is_source_bounded(self):
        resource = self.hw.resource_by_role(
            "staged_ldgsts_k32_depth6_l2_read")
        reference_path = Path(
            "microbench/nvidia_a800/references/"
            "staged_ldgsts_k32_depth6_l2_service_sm80.json")
        with reference_path.open() as stream:
            reference = json.load(stream)

        assert resource.scope == "shared"
        assert resource.effective_capacity(
            self.hw.num_compute_units,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        ) == pytest.approx(
            reference["selected_service"][
                "logical_byte_capacity_per_s"])
        assert not reference["production_or_evaluation_duration_consumed"]

    def test_finite_async_operand_stream_completion_is_topology_bounded(self):
        profile = self.hw.finite_async_operand_stream_completion

        assert profile["model"] == "finite_async_operand_stream_completion_v1"
        assert profile["grid_ctas"] == 64
        assert profile["threads_per_cta"] == 128
        assert profile["operand_pair_bytes"] == 16384
        assert profile["complete_operand_pair_slots"] == 3
        assert profile["minimum_rounds"] == 64
        assert profile["maximum_rounds"] == 512
        assert profile["fixed_completion_ns"] == pytest.approx(10240.0)
        assert profile["l2_resident_ns_per_round"] == pytest.approx(475.0)
        assert profile["dram_unique_ns_per_round"] == pytest.approx(608.0)
        assert profile["reference_path"] == (
            "microbench/nvidia_a800/references/"
            "finite_async_operand_stream_completion_sm80.json")

    def test_rmsnorm_shared_gamma_residency_is_source_bounded(self):
        model = self.hw.rmsnorm_shared_gamma_l1_residency
        reference_path = Path(
            "microbench/nvidia_a800/references/"
            "rmsnorm_gamma_launch_lifetime_sm80.json")

        assert model["model"] == "source_shared_gamma_residency_tier_v2"
        assert model["byte_unit"] == "bf16_gamma_bytes_per_cta"
        assert model["threads_per_cta"] == 1024
        assert model["registers_per_thread"] == 24
        assert model["static_shared_memory_bytes_per_cta"] == 168
        assert model["resident_ctas_per_sm"] == 2
        assert model["launch_lifetime_maximum_gamma_bytes"] == 4096
        assert model["maximum_evidenced_gamma_bytes"] == 32768
        assert model["reference_sha256"] == hashlib.sha256(
            reference_path.read_bytes()).hexdigest()

    def test_rmsnorm_shared_gamma_rejects_resource_signature_change(
            self, tmp_path):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            config = yaml.safe_load(stream)
        config["scheduling"]["rmsnorm_shared_gamma_l1_residency"][
            "registers_per_thread"] = 23
        path = tmp_path / "invalid_rmsnorm_gamma.yaml"
        path.write_text(yaml.safe_dump(config))

        with pytest.raises(ValueError, match="must equal 24"):
            HWSpec(str(path))

    def test_add_rmsnorm_source_round_pipeline_is_source_closed(self):
        model = self.hw.add_rmsnorm_source_round_pipeline
        reference_path = Path(
            "microbench/nvidia_a800/references/"
            "add_rmsnorm_block1024_pipeline_sm80.json")
        vector = self.hw.resource_by_role("compute_vector")

        assert model["model"] == (
            "sm80_bf16_add_rmsnorm_block1024_one_wave_v8")
        assert model["hidden_size"] == 2048
        assert model["additional_hidden_size"] == 2560
        assert model["second_additional_hidden_size"] == 4096
        assert model["third_additional_hidden_size"] == 5120
        assert model["fourth_additional_hidden_size"] == 8192
        assert model["threads_per_cta"] == 1024
        assert model["active_threads_per_source_round"] == 256
        assert model["additional_active_threads_per_source_round"] == 320
        assert model[
            "second_additional_active_threads_per_source_round"] == 512
        assert model[
            "third_additional_active_threads_per_source_round"] == 640
        assert model[
            "fourth_additional_active_threads_per_source_round"] == 1024
        assert model["active_unit_points"] == (1, 4, 16, 108)
        assert model["additional_active_unit_points"] == (1, 4, 16, 108)
        assert model["second_additional_active_unit_points"] == (
            1, 4, 16, 108)
        assert model["third_additional_active_unit_points"] == (
            1, 4, 8, 16, 108)
        assert model["fourth_additional_active_unit_points"] == (
            1, 4, 8, 16, 108)
        assert model["resident_ctas_per_sm"] == 2
        assert model["maximum_one_wave_grid_blocks"] == 108
        assert model["reference_sha256"] == hashlib.sha256(
            reference_path.read_bytes()).hexdigest()
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_256_block1024"] == pytest.approx(
                self.hw.nominal_sm_freq_hz / 615.0)
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_256_block1024"] == pytest.approx(
                self.hw.nominal_sm_freq_hz / 384.0)
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_320_block1024"] == pytest.approx(
                self.hw.nominal_sm_freq_hz / 521.0)
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_320_block1024"] == pytest.approx(
                self.hw.nominal_sm_freq_hz / 469.0)
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_256_active108_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 688.5))
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_256_active108_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 644.0))
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_320_active108_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 710.0))
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_320_active108_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 1015.0))
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_512_block1024"] == pytest.approx(
                self.hw.nominal_sm_freq_hz / 643.0)
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_512_block1024"] == pytest.approx(
                self.hw.nominal_sm_freq_hz / 566.0)
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_512_active108_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 918.0))
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_512_active108_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 1161.0))
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_640_block1024"] == pytest.approx(
                self.hw.nominal_sm_freq_hz / 731.0)
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_640_active8_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 836.0))
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_640_active108_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 1132.0))
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_640_block1024"] == pytest.approx(
                self.hw.nominal_sm_freq_hz / 678.0)
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_640_active8_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 789.0))
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_640_active108_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 1268.0))
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_1024_active8_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 1188.0))
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_1024_active8_block1024"] == (
                pytest.approx(self.hw.nominal_sm_freq_hz / 1075.0))

    def test_silu_nonrecurrent_segmented_services_are_source_bounded(self):
        reference_path = Path(
            "microbench/nvidia_a800/references/"
            "silu_nonrecurrent_segmented_service_sm80.json")
        with reference_path.open() as stream:
            reference = json.load(stream)
        vector = self.hw.resource_by_role("compute_vector")

        for service, evidence in reference["selected_services"].items():
            assert vector.service_rates_per_unit_s[service] == pytest.approx(
                self.hw.nominal_sm_freq_hz / evidence["cycles_per_cta"])
        assert reference["forward_domain"] == {
            "dtype": "bf16",
            "threads_per_cta": 1024,
            "grid_ctas": 256,
            "resident_ctas_per_sm": 2,
            "maximum_ctas_per_sm": 3,
            "cohort_refills": 2,
        }
        assert reference["production_or_evaluation_duration_consumed"] is False

    def test_add_rmsnorm_block256_pipeline_is_source_closed(self):
        model = self.hw.add_rmsnorm_block256_source_round_pipeline
        reference_path = Path(
            "microbench/nvidia_a800/references/"
            "add_rmsnorm_block256_pipeline_sm80.json")
        service_reference_path = Path(
            "microbench/nvidia_a800/references/"
            "add_rmsnorm_round_service_sm80.json")
        pair_reference_path = Path(
            "microbench/nvidia_a800/references/"
            "add_rmsnorm_block256_full_tail_pair_sm80.json")
        vector = self.hw.resource_by_role("compute_vector")

        assert model["model"] == (
            "sm80_bf16_add_rmsnorm_block256_resident_v1")
        assert model["threads_per_cta"] == 256
        assert model["source_round_elements"] == 2048
        assert model["resident_ctas_per_sm"] == 8
        assert model["maximum_grid_blocks"] == 864
        assert model["full_tail_pair_active_threads"] == (64, 128)
        assert model["service_reference_sha256"] == hashlib.sha256(
            service_reference_path.read_bytes()).hexdigest()
        assert model["reference_sha256"] == hashlib.sha256(
            reference_path.read_bytes()).hexdigest()
        assert model["full_tail_pair_reference_sha256"] == hashlib.sha256(
            pair_reference_path.read_bytes()).hexdigest()
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_first_256_block256"] == pytest.approx(
                1e9 / 246.0)
        assert vector.service_rates_per_unit_s[
            "bf16_add_square_resident_256_block256"] == pytest.approx(
                1e9 / 760.2857142857143)
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_first_256_block256"] == pytest.approx(
                1e9 / 256.0)
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_resident_256_block256"] == pytest.approx(
                1e9 / 515.7142857142857)
        pair_services_ns = {
            "bf16_add_square_first_64_block256": 423.125,
            "bf16_add_square_resident_64_block256": 880.410714,
            "bf16_scale_multiply_first_64_block256": 403.125,
            "bf16_scale_multiply_resident_64_block256": 593.071429,
            "bf16_add_square_first_128_block256": 401.375,
            "bf16_add_square_resident_128_block256": 1202.857143,
            "bf16_scale_multiply_first_128_block256": 406.125,
            "bf16_scale_multiply_resident_128_block256": 799.017857,
        }
        for service, pair_ns in pair_services_ns.items():
            assert vector.service_rates_per_unit_s[service] == pytest.approx(
                2e9 / pair_ns)


    def test_l2_latency_ns_positive(self):
        assert self.hw.l2_latency_ns > 0

    def test_cacheline_bytes_positive(self):
        assert self.hw.cacheline_bytes > 0

    def test_fa2_ldgsts_frontend_rates(self):
        assert (
            self.hw.ldgsts_initial_32b_transactions_single_sm_cycle
            == pytest.approx(1.058))
        assert (
            self.hw.ldgsts_initial_32b_transactions_per_active_sm_cycle
            == pytest.approx(1.221))
        assert (
            self.hw.ldgsts_initial_32b_transactions_full_chip_per_cycle
            == pytest.approx(64.26))
        assert self.hw.ldgsts_finite_resident_panel_bytes == 65536
        assert self.hw.ldgsts_finite_resident_full_chip_ctas == 216
        assert (
            self.hw.ldgsts_finite_resident_single_panel_cycles
            == pytest.approx(10371.03703704576))
        assert (
            self.hw.ldgsts_finite_resident_affine_startup_cycles
            == pytest.approx(12495.166942907046))
        assert (
            self.hw.ldgsts_finite_resident_affine_panel_cycles
            == pytest.approx(4150.234554751573))
        assert self.hw.ldgsts_finite_resident_maximum_panels == 64
        reference_path = Path(
            "microbench/nvidia_a800/references/"
            "fa2_resident_ldgsts_finite_sm80.json")
        assert self.hw.ldgsts_finite_resident_reference_sha256 == (
            hashlib.sha256(reference_path.read_bytes()).hexdigest())
        steady_reference_path = Path(
            "microbench/nvidia_a800/references/"
            "fa2_queued_ldgsts_steady_frontend_sm80.json")
        assert self.hw.ldgsts_queued_steady_minimum_grid_ctas == 2048
        assert self.hw.ldgsts_queued_steady_maximum_panels_per_cta == 4
        assert (
            self.hw.ldgsts_queued_steady_average_panels_per_cta
            == pytest.approx(2.5))
        assert (
            self.hw
            .ldgsts_queued_steady_full_chip_initial_32b_transactions_per_cycle
            == pytest.approx(65.56537328723269))
        assert self.hw.ldgsts_queued_steady_reference_sha256 == (
            hashlib.sha256(steady_reference_path.read_bytes()).hexdigest())

    def test_fa2_chunked_q8192_two_resident_ingress_service(self):
        reference_path = Path(
            "microbench/nvidia_a800/references/"
            "fa2_chunked_q8192_two_resident_ingress_sm80.json")
        with reference_path.open() as stream:
            reference = json.load(stream)
        services = tuple(
            service for service in self.hw.memory_access_class_services
            if service.service == (
                "sm80_fa2_chunked_q8192_s4096_two_resident_ldgsts"))

        assert len(services) == 1
        service, = services
        assert service.access_kind == "read"
        assert service.ingress_resource_role == "l2_cache"
        assert service.scope == "full_chip"
        assert service.signature.access_class == service.service
        assert service.signature.dtype == "bf16"
        assert (service.signature.tile_m, service.signature.tile_k) == (64, 128)
        assert service.signature.logical_row_bytes == 65536
        assert service.logical_byte_capacity_per_s == pytest.approx(
            reference["selected_service"]["logical_byte_capacity_per_s"])
        assert reference["resource_contract"]["dynamic_shared_bytes_per_cta"] == 82944
        assert reference["publication_gates"] == {
            "two_resident_ctas_per_sm": True,
            "static_ldgsts_bypass_128_count_is_32": True,
            "no_stack_or_spill": True,
            "dynamic_instruction_and_byte_counters_close_exactly": True,
            "all_native_cells_relative_span_at_most_5_pct": True,
            "selected_grid_source_profiles_within_5_pct": True,
        }

    def test_gemv_scalar_half_dram_source(self):
        assert self.hw.gemv_scalar_half_issue_group_elements_per_lane == 64
        assert self.hw.gemv_scalar_half_resident_ctas_per_sm == 3
        assert (
            self.hw.gemv_scalar_half_dram_transactions_per_sm_s
            == pytest.approx(352325593.27666295))
        assert self.hw.gemv_scalar_half_source_fill_ns == pytest.approx(
            5911.950466958923)

    def test_metax_grouped_operand_pipeline_registry_is_consistent(self):
        """Every configured mechanism must be declared and loadable."""

        import yaml as _yaml
        from src.kernel.hardware_spec import (
            _METAX_GROUPED_OPERAND_PIPELINE_SPECS)
        from src.kernel.model_contract import (
            METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS)

        assert set(_METAX_GROUPED_OPERAND_PIPELINE_SPECS).issubset(
            set(METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS))
        with open("configs/hardware/metax_c500.yaml") as stream:
            scheduling = _yaml.safe_load(stream)["scheduling"]
        configured = {
            key for key in METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS
            if key in scheduling
        }
        assert configured
        assert configured.issubset(set(_METAX_GROUPED_OPERAND_PIPELINE_SPECS))
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        signatures = set()
        for key in METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS:
            mechanism = getattr(hw, key)
            if mechanism is None:
                continue
            signatures.add((
                mechanism["dtype"], mechanism["tile_m"], mechanism["tile_n"],
                mechanism["tile_k"], mechanism["group_size_m"],
                mechanism["threads_per_cta"], mechanism["num_warps"],
                mechanism["num_stages"]))
        assert len(signatures) == len(configured)

    def test_metax_fused_moe_bm16_operand_pipeline_is_typed(self, tmp_path):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        pipeline = hw.metax_fused_moe_bm16_operand_pipeline
        if pipeline is None:
            pytest.skip("C500 BM16 operand-pipeline mechanism is absent")

        assert pipeline["model"] == (
            "metax_bm16_bf16_grouped_operand_pipeline_v1")
        assert {
            name: pipeline[name]
            for name in (
                "tile_m", "tile_n", "tile_k", "group_size_m",
                "num_warps", "num_stages", "threads_per_cta",
                "dynamic_shared_memory_bytes_per_cta",
                "resident_ctas_per_ap", "pipeline_depth",
            )
        } == {
            "tile_m": 16,
            "tile_n": 64,
            "tile_k": 128,
            "group_size_m": 1,
            "num_warps": 4,
            "num_stages": 4,
            "threads_per_cta": 256,
            "dynamic_shared_memory_bytes_per_cta": 40960,
            "resident_ctas_per_ap": 1,
            "pipeline_depth": 4,
        }
        for name in (
                "fixed_span_ns", "local_k_round_interval_ns",
                "empty_wave_slot_ns"):
            assert pipeline[name] > 0.0
        assert not pipeline["production_or_evaluation_duration_consumed"]

        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["metax_fused_moe_bm16_operand_pipeline"][
            "reference_sha256"] = "0" * 64
        path = tmp_path / "invalid_metax_fused_moe_bm16_pipeline.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="is not canonical"):
            HWSpec(str(path))


    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            (
                "ldgsts_initial_32b_transactions_single_sm_cycle",
                None,
                "requires all three",
            ),
            (
                "ldgsts_initial_32b_transactions_per_active_sm_cycle",
                None,
                "requires all three",
            ),
            (
                "ldgsts_initial_32b_transactions_full_chip_per_cycle",
                0.0,
                "must be positive",
            ),
        ),
    )
    def test_fa2_ldgsts_frontend_rejects_incomplete_or_nonpositive_rates(
            self, tmp_path, field, value, message):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["memory_subsystem"][field] = value
        path = tmp_path / "invalid_ldgsts_frontend.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match=message):
            HWSpec(str(path))

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            (
                "ldgsts_queued_steady_minimum_grid_ctas", None,
                "requires its complete",
            ),
            (
                "ldgsts_queued_steady_full_chip_initial_32b_transactions_per_cycle",
                0.0,
                "must be positive",
            ),
            (
                "ldgsts_queued_steady_reference_sha256", "invalid",
                "lowercase SHA256",
            ),
            (
                "ldgsts_queued_steady_minimum_grid_ctas", 216,
                "must exceed the resident",
            ),
            (
                "ldgsts_queued_steady_average_panels_per_cta", 5.0,
                "must not exceed",
            ),
        ),
    )
    def test_fa2_queued_steady_ldgsts_rejects_invalid_contract(
            self, tmp_path, field, value, message):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["memory_subsystem"][field] = value
        path = tmp_path / "invalid_queued_steady_ldgsts.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match=message):
            HWSpec(str(path))

    def test_resources_not_empty(self):
        assert len(self.hw.resources) > 0

    def test_resource_groups_not_empty(self):
        assert len(self.hw.resource_groups) > 0

    def test_empty_kernel_floor_is_not_duplicated_as_additive_dispatch(self):
        assert self.hw.dispatch_overhead_ns == 0
        assert self.hw.empty_kernel_duration_lut is None
        assert self.hw.launch_latency_model["model"] == (
            "isolated_empty_task_scheduler_v1")
        assert self.hw.empty_kernel_duration_for_launch(
            1, 128) == pytest.approx(1184.23236278)
        assert self.hw.empty_kernel_duration_for_launch(
            1024, 1024) == pytest.approx(3075.1833777)
        assert self.hw.empty_kernel_duration_for_launch(
            2048, 128) == pytest.approx(2610.17922062)
        assert self.hw.empty_kernel_duration_for_launch(
            1048576, 1024) == pytest.approx(1914140.3596336998)

    def test_clock_and_l2_capacity_use_separate_nominal_and_run_clock_evidence(self):
        assert self.hw.nominal_sm_freq_hz == 1.14e9
        l2 = self.hw.resource_by_role("l2_cache")
        assert l2.effective_capacity(
            self.hw.num_compute_units,
            self.hw.nominal_sm_freq_hz,
            self.hw.nominal_dram_freq_hz,
        ) == pytest.approx(2563.037314779177 * 1.14e9)

    def test_memory_dependency_task_model_is_a_bounded_complete_max_floor(
            self):
        model = self.hw.memory_dependency_task_latency_model
        contract = "warm_default_cache_memory_dependency_task_v1"

        assert model["contract"] == contract
        assert model["timer_domain"] == "nsys_gpu_kernel"
        assert model["cache_policy"] == "warm_default_cache"
        assert model["maximum_grid_size"] == 108
        assert model["maximum_block_size"] == 1024
        assert self.hw.has_task_duration_floor_contract(contract)
        assert self.hw.memory_dependency_task_duration_ns(
            contract, 1, 32, (1, 1, 1), (32, 1, 1), 1.41e9,
        ) == pytest.approx(1423.7304609010866)
        assert self.hw.memory_dependency_task_duration_ns(
            contract, 3, 128, (3, 1, 1), (128, 1, 1), 1.14e9,
        ) == pytest.approx(1804.390990772361)
        assert self.hw.task_duration_floor_for_task(
            "generic", 108, 1024, (108, 1, 1), (1024, 1, 1),
            task_duration_floor_contract=contract,
            sm_freq_hz=1.14e9,
        ) == pytest.approx(2006.1325921077844)

    def test_register_reduction_dependency_cycles_are_physical_geometries(
            self):
        model = self.hw.register_reduction_dependency_cycles

        assert model["model"] == (
            "isolated_register_reduction_and_cta_barrier_v1")
        assert model["timer_domain"] == "clock64_same_cta_envelope_cycles"
        assert model["threads_per_cta"] == 128
        assert {
            (entry["lane_local_rows_per_thread"],
             entry["score_values_per_lane_local_row"])
            for entry in model["entries"]
        } == {(2, 32), (2, 64), (4, 16), (4, 32)}
        assert model["entries"][3]["fp32_max_quad4_cycles"] == pytest.approx(
            333.5078125)
        assert model["entries"][3][
            "fp32_scale_exp2_sum_cycles"] == pytest.approx(1210.234375)
        assert model["entries"][3][
            "integer_column_predicate_cycles"] == pytest.approx(334.734375)

    def test_fa2_local_spill_dependency_cycles_match_static_contract(self):
        model = self.hw.fa2_local_spill_dependency_cycles

        assert model["model"] == "isolated_fa2_local_spill_round_v1"
        assert model["timer_domain"] == "clock64_same_cta_envelope_cycles"
        assert model["threads_per_cta"] == 128
        assert model["stack_bytes_per_thread"] == 56
        assert model["scalar_store_sites"] == 13
        assert model["scalar_load_sites"] == 13
        assert model["spill_round_cycles"] == pytest.approx(2195.0)
        assert model["source_sha256"] == hashlib.sha256(Path(
            "microbench/nvidia_a800/memory/l1/"
            "fa2_local_spill_round.cu").read_bytes()).hexdigest()

    def test_fa2_dense_d128_causal_local_spill_matches_static_contract(self):
        model = (
            self.hw.fa2_dense_d128_causal_local_spill_dependency_cycles)

        assert model["model"] == (
            "isolated_fa2_dense_d128_causal_local_spill_round_v1")
        assert model["timer_domain"] == "clock64_same_cta_envelope_cycles"
        assert model["threads_per_cta"] == 128
        assert model["stack_bytes_per_thread"] == 176
        assert model["scalar_store_sites"] == 51
        assert model["scalar_load_sites"] == 53
        assert model["spill_round_cycles"] == pytest.approx(8197.0)
        assert model["source_sha256"] == hashlib.sha256(Path(
            "microbench/nvidia_a800/memory/l1/"
            "fa2_local_spill_round.cu").read_bytes()).hexdigest()

    def test_fa2_combine_dependency_cycles_cover_source_geometries(self):
        model = self.hw.fa2_combine_dependency_cycles

        assert model["model"] == (
            "isolated_fa2_combine_source_dependency_v2")
        assert model["threads_per_cta"] == 128
        assert {
            entry["allreduce_width"]
            for entry in model["reduction_entries"]
        } == {2, 4, 8, 16, 32}
        assert {
            (entry["block_m"], entry["head_dim"],
             entry["output_values_per_lane"])
            for entry in model["conversion_entries"]
        } == {(4, 128, 4), (8, 64, 4)}
        assert {
            (entry["block_m"], entry["head_dim"],
             entry["num_splits"], entry["max_splits"])
            for entry in model["source_fragment_entries"]
        } == {
            (8, 64, 2, 2), (8, 64, 4, 4), (8, 64, 8, 8),
            (4, 128, 2, 2), (4, 128, 3, 4), (4, 128, 4, 4),
            (4, 128, 5, 8), (4, 128, 6, 8), (4, 128, 7, 8),
            (4, 128, 8, 8), (4, 128, 9, 16), (4, 128, 11, 16),
            (4, 128, 16, 16), (4, 128, 22, 32),
        }
        assert model["reduction_entries"][4][
            "fp32_max_allreduce_cycles"] == pytest.approx(148.6953125)
        assert model["conversion_entries"][0][
            "fp32_to_bf16_output_conversion_cycles"] == pytest.approx(
            27.390625)
        assert model["source_fragment_entries"][0][
            "post_sum_log_cycles"] == pytest.approx(114.6953125)

    def test_fa2_source_task_outer_envelope_has_bounded_geometry(self):
        model = self.hw.fa2_source_task_outer_envelope

        assert model["model"] == "nsys_source_task_outer_envelope_v1"
        assert model["timer_domain"] == (
            "nsys_gpu_kernel_and_globaltimer_cta_envelope")
        assert model["threads_per_cta"] == 128
        assert (model["minimum_grid_size"], model["maximum_grid_size"]) == (
            8, 64)
        assert self.hw.fa2_source_task_outer_envelope_ns(8, 128) == (
            pytest.approx(2176.0))
        assert self.hw.fa2_source_task_outer_envelope_ns(64, 128) == (
            pytest.approx(2176.0))
        with pytest.raises(ValueError, match="outside the measured"):
            self.hw.fa2_source_task_outer_envelope_ns(65, 128)
        assert model["source_sha256"] == hashlib.sha256(Path(
            "microbench/nvidia_a800/runtime/launch/"
            "fa_task_gap_geometry.cu").read_bytes()).hexdigest()

    def test_fa2_oaccum_load_use_cycles_cover_both_cache_tiers(self):
        model = self.hw.fa2_oaccum_load_use_cycles

        assert model["model"] == (
            "isolated_ldg128_fp32x4_load_to_operands_ready_v1")
        assert model["threads_per_cta"] == 128
        assert model["load_bytes_per_lane"] == 16
        assert model["values_per_lane"] == 4
        assert model["dependent_fp32_fma_per_lane"] == 4
        assert {entry["cache_tier"] for entry in model["cache_entries"]} == {
            "l2_warm_cg", "dram_cg"}
        dram, = tuple(
            entry for entry in model["cache_entries"]
            if entry["cache_tier"] == "dram_cg")
        assert dram["first_load_ready_cycles"] == pytest.approx(
            1020.2413184772522)
        assert dram["additional_load_interval_cycles"] == pytest.approx(
            711.4374187558032)

    @pytest.mark.parametrize(
        ("contract", "grid", "expected_cycles"),
        (
            (
                TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
                1,
                5333.994386215753 + 2.2671764958645855,
            ),
            (
                TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR,
                4,
                4248.705521150855 + 895.1779278546923
                + 4 * 1.0656575859889441,
            ),
        ),
    )
    def test_tensor_iterator_copy_task_floor_uses_runtime_sm_clock(
            self, contract, grid, expected_cycles):
        assert self.hw.has_task_duration_floor_contract(contract)
        duration_ns = self.hw.tensor_iterator_copy_task_duration_ns(
            contract,
            grid_size=grid,
            block_size=128,
            launch_grid=(grid, 1, 1),
            launch_block=(128, 1, 1),
            sm_freq_hz=1.14e9,
        )
        assert duration_ns == pytest.approx(expected_cycles / 1.14)

    def test_tensor_iterator_contract_is_known_but_absent_on_c500(self):
        c500 = HWSpec("configs/hardware/metax_c500.yaml")
        assert not c500.has_task_duration_floor_contract(
            TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR)
        with pytest.raises(ValueError, match="unknown task-duration"):
            c500.has_task_duration_floor_contract("invented_copy_contract")

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("threads_per_cta", 64, "must equal 128"),
            ("source_sha256", "invalid", "lowercase SHA256"),
            ("entries", [], "non-empty list"),
        ),
    )
    def test_register_reduction_dependency_rejects_invalid_schema(
            self, tmp_path, field, value, message):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["register_reduction_dependency_cycles"][field] = (
            value)
        path = tmp_path / "invalid_register_dependency.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises((TypeError, ValueError), match=message):
            HWSpec(str(path))


    @pytest.mark.parametrize(
        ("section", "field", "value", "message"),
        (
            (None, "threads_per_cta", 64, "must equal 128"),
            (None, "source_sha256", "invalid", "lowercase SHA256"),
            ("reduction_entries", 0, {
                "allreduce_width": 3,
                "fp32_max_allreduce_cycles": 42.0,
                "fp32_sum_allreduce_cycles": 43.0,
            }, "allreduce_width is unsupported"),
            ("conversion_entries", 0, {
                "block_m": 4,
                "head_dim": 128,
                "output_values_per_lane": 4,
                "fp32_to_bf16_output_conversion_cycles": 0.0,
            }, "must be finite and positive"),
            ("source_fragment_entries", 0, {
                "block_m": 8,
                "head_dim": 64,
                "num_splits": 2,
                "max_splits": 2,
                "slse_publish_transpose_cycles": 0.0,
                "post_max_exp_local_sum_cycles": 54.0,
                "post_sum_log_cycles": 114.0,
                "scale_exp_smem_barrier_cycles": 130.0,
            }, "must be finite and positive"),
        ),
    )
    def test_fa2_combine_dependency_rejects_invalid_schema(
            self, tmp_path, section, field, value, message):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        model = cfg["scheduling"]["fa2_combine_dependency_cycles"]
        if section is None:
            model[field] = value
        else:
            model[section][field] = value
        path = tmp_path / "invalid_fa2_combine_dependency.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises((TypeError, ValueError), match=message):
            HWSpec(str(path))

    @pytest.mark.parametrize(
        ("grid", "block", "launch_grid", "launch_block", "clock"),
        (
            (109, 128, (109, 1, 1), (128, 1, 1), 1.14e9),
            (8, 128, (4, 2, 1), (128, 1, 1), 1.14e9),
            (8, 128, (8, 1, 1), (64, 2, 1), 1.14e9),
            (8, 128, (8, 1, 1), (128, 1, 1), 1.410000001e9),
        ),
    )
    def test_memory_dependency_task_model_rejects_extrapolation(
            self, grid, block, launch_grid, launch_block, clock):
        contract = "warm_default_cache_memory_dependency_task_v1"

        assert not self.hw.task_duration_floor_contract_supports_launch(
            contract, grid, block, launch_grid, launch_block, clock)
        with pytest.raises(ValueError, match="outside the measured"):
            self.hw.memory_dependency_task_duration_ns(
                contract, grid, block, launch_grid, launch_block, clock)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("timer_domain", "cuda_event", "must be 'nsys_gpu_kernel'"),
            ("base_ns", 0.0, "finite and positive"),
            ("launch_rank", 2, "must equal one"),
            ("fanout_activation_grid", 109, "outside the grid range"),
        ),
    )
    def test_memory_dependency_task_model_rejects_invalid_schema(
            self, tmp_path, field, value, message):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["memory_dependency_task_latency_model"][field] = value
        path = tmp_path / "invalid_memory_dependency_model.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises((TypeError, ValueError), match=message):
            HWSpec(str(path))

    def test_physical_launch_model_generalizes_grid_but_rejects_block_overflow(
            self):
        assert self.hw.empty_kernel_duration_for_launch(1048577, 128) > 0
        with pytest.raises(ValueError, match="hardware thread limit"):
            self.hw.empty_kernel_duration_for_launch(1, 1025)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("submit_ns", -1.0, "finite and non-negative"),
            ("cta_round_ns", 0.0, "required services"),
            ("timer_domain", "cuda_event", "unsupported"),
        ),
    )
    def test_launch_latency_model_rejects_invalid_parameters(
            self, tmp_path, field, value, message):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["launch_latency_model"][field] = value
        path = tmp_path / "invalid_launch_model.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises((TypeError, ValueError), match=message):
            HWSpec(str(path))

    def test_launch_latency_model_is_mutually_exclusive_with_legacy_floor(
            self, tmp_path):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["empty_kernel_duration_ns"] = 1000
        path = tmp_path / "duplicate_launch_model.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="not more than one"):
            HWSpec(str(path))

    def test_launch_latency_model_rejects_unknown_features(self, tmp_path):
        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["launch_latency_model"]["kernel_kind"] = "VECTOR"
        path = tmp_path / "operator_specific_launch_model.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="schema mismatch"):
            HWSpec(str(path))

    def test_launch_latency_model_consumes_full_dim3_products(self):
        flat = self.hw.empty_kernel_duration_for_launch(128, 256)
        dim3 = self.hw.empty_kernel_duration_for_launch(
            128, 256, (8, 4, 4), (8, 8, 4))
        assert dim3 == pytest.approx(flat)

        with pytest.raises(ValueError, match="disagrees with its flattened"):
            self.hw.empty_kernel_duration_for_launch(
                128, 256, (8, 4, 2), (8, 8, 4))

    def test_memory_model_is_explicit(self):
        assert self.hw.kernel_memory_model == "gpu_mshr"
        assert self.hw.gpu_store_completion_scope == "l2"

    def test_directional_dram_capacities_are_explicit_physical_rates(self):
        assert self.hw.dram_arbitration_mode == (
            "shared_directional_capacity")
        assert self.hw.dram_read_capacity_per_clk == 1124
        assert self.hw.dram_write_capacity_per_clk == pytest.approx(
            1128.814187068424)
        assert self.hw.dram_capacity_for_access(
            "write",
            active_units=self.hw.num_compute_units,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        ) == pytest.approx(
            1128.814187068424 * self.hw.nominal_dram_freq_hz)

    def test_unmeasured_communication_setup_is_absent(self):
        assert self.hw.interconnect["nccl_alpha_ns"] is None
        assert self.hw.interconnect["nvlink_signal_latency_ns"] is None
        assert self.hw.interconnect["pcie_latency_ns"] is None

    def test_grouped_moe_operand_services_are_exact_signatures(self):
        assert self.hw.has_memory_access_class_service(
            "grouped_fused_moe_operand_read")
        assert self.hw.has_memory_access_class_service(
            "grouped_fused_moe_operand_pipeline")
        assert self.hw.has_memory_access_class_service(
            "grouped_fused_moe_weight_overlapped_dram_miss_pipeline")
        assert not self.hw.has_memory_access_class_service(
            "indirect_row_gather")
        source_depth_capacities = {
            service.service: service.logical_byte_capacity_per_s
            for service in self.hw.memory_access_class_services
            if service.service.startswith("nvidia_eight_expert_")}
        assert source_depth_capacities == pytest.approx({
            "nvidia_eight_expert_68_active_72_max_"
            "fused_moe_w1_operand_pipeline": 4036651578275.5825,
            "nvidia_eight_expert_132_active_136_max_"
            "fused_moe_w1_operand_pipeline": 3278334384896.408,
        })
        bm16_h2880_capacities = {
            service.service: service.logical_byte_capacity_per_s
            for service in self.hw.memory_access_class_services
            if service.service.startswith("nvidia_bm16_h2880_")
        }
        assert bm16_h2880_capacities == pytest.approx({
            "nvidia_bm16_h2880_31_active_"
            "fused_moe_operand_pipeline": 1703147743460.753,
            "nvidia_bm16_h2880_53_active_"
            "fused_moe_operand_pipeline": 1727870083298.538,
        })
        shallow_e128_capacities = {
            service.service: service.logical_byte_capacity_per_s
            for service in self.hw.memory_access_class_services
            if service.service.startswith(
                "nvidia_bm16_e128_88_active_128_max_")
        }
        assert shallow_e128_capacities == pytest.approx({
            "nvidia_bm16_e128_88_active_128_max_"
            "fused_moe_w1_operand_pipeline": 1858370329133.5552,
            "nvidia_bm16_e128_88_active_128_max_"
            "fused_moe_w2_operand_pipeline": 1803318265947.254,
        })
        measured_bm16_capacities = {
            service.service: service.logical_byte_capacity_per_s
            for service in self.hw.memory_access_class_services
            if service.service.startswith(
                "nvidia_bm16_e128_177_active_248_max_")
            or service.service.startswith(
                "nvidia_bm16_e8_7_active_8_max_")
        }
        assert measured_bm16_capacities == pytest.approx({
            "nvidia_bm16_e128_177_active_248_max_"
            "fused_moe_w1_operand_pipeline": 2975495908026.122,
            "nvidia_bm16_e128_177_active_248_max_"
            "fused_moe_w2_operand_pipeline": 2803015376984.658,
            "nvidia_bm16_e8_7_active_8_max_"
            "fused_moe_w1_operand_pipeline": 1831031427428.9458,
            "nvidia_bm16_e8_7_active_8_max_"
            "fused_moe_w2_operand_pipeline": 1704528963555.0315,
        })
        sparse_w2_capacities = {
            service.service: service.logical_byte_capacity_per_s
            for service in self.hw.memory_access_class_services
            if service.service.startswith("nvidia_bm16_e")
            and (
                service.service.endswith(
                    "active_241_max_fused_moe_w2_operand_pipeline")
                or service.service.endswith(
                    "active_482_max_fused_moe_w2_operand_pipeline"))
        }
        assert sparse_w2_capacities == pytest.approx({
            "nvidia_bm16_e256_16_active_241_max_"
            "fused_moe_w2_operand_pipeline": 871200022008.3638,
            "nvidia_bm16_e512_20_active_482_max_"
            "fused_moe_w2_operand_pipeline": 757777776500.2657,
        })
        grouped_services = tuple(
            service for service in self.hw.memory_access_class_services
            if service.service.startswith("grouped_fused_moe_"))
        assert len(grouped_services) == 8
        grouped_capacities = {
            (
                service.signature.access_class,
                service.signature.logical_row_bytes,
                service.signature.operand_role,
                service.signature.weight_streams_per_group,
            ): service.logical_byte_capacity_per_s
            for service in grouped_services
        }
        assert grouped_capacities == pytest.approx({
            (
                "grouped_fused_moe_operand_read",
                6144, "combined", None,
            ): 2427.25931e9,
            (
                "grouped_fused_moe_operand_pipeline",
                8192, "combined", None,
            ): 3326075187969.925,
            (
                "grouped_fused_moe_operand_pipeline",
                4096, "weight", None,
            ): 1316767342414.141,
            (
                "grouped_fused_moe_weight_overlapped_dram_miss_pipeline",
                4096, "weight", None,
            ): 1316767342414.141,
            (
                "grouped_fused_moe_operand_pipeline",
                65536, "combined", 1,
            ): 2268398732273.012,
            (
                "grouped_fused_moe_operand_pipeline",
                65536, "combined", 2,
            ): 2230576826756.615,
            (
                "grouped_fused_moe_operand_pipeline",
                65536, "combined", 4,
            ): 2172730800328.8557,
            (
                "grouped_fused_moe_operand_pipeline",
                65536, "combined", 8,
            ): 1877410546550.0728,
        })
        assert all(
            service.access_kind == "read"
            and service.ingress_resource_role == "l2_cache"
            and service.scope == "full_chip"
            and service.signature.dtype == "bf16"
            and service.replay_to_dram_fraction is None
            for service in grouped_services)
        assert {
            service.signature.operand_role
            for service in grouped_services
        } == {"combined", "weight"}
        vector_capacities = {
            service.service: service.logical_byte_capacity_per_s
            for service in self.hw.memory_access_class_services
            if service.service.startswith("bf16_vec8_add_rmsnorm")
        }
        assert vector_capacities == pytest.approx({
            "bf16_vec8_add_rmsnorm_mixed_l2_read": 1371683767200.0,
            "bf16_vec8_add_rmsnorm_mixed_l2_write": 914455844800.0,
            "bf16_vec8_add_rmsnorm_mixed_dram_read": 946972684800.0,
            "bf16_vec8_add_rmsnorm_mixed_dram_write": 631315123200.0,
        })


class TestHWSpecC500:
    def test_block256_add_rmsnorm_queued_services_use_scalar_source_depth(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        vector = hw.resource_by_role("compute_vector")

        assert vector.service_rates_per_unit_s[
            "bf16_add_square_queued_256"] == pytest.approx(
                1e9 / (8 * 107.06248576752841))
        assert vector.service_rates_per_unit_s[
            "bf16_scale_multiply_queued_256"] == pytest.approx(
                1e9 / (8 * 124.87499043345451))

    def test_imcontinuous_vec4_reduction_service_is_source_bound(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        mechanism = hw.imcontinuous_vec4_reduction_source_service
        reference_path = Path(
            "microbench/metax_c500/references/"
            "imcontinuous_vec4_reduction_source_service_maca_3_7_2.json")

        assert mechanism["threads_per_cta"] == 512
        assert mechanism["dynamic_shared_bytes"] == 4104
        assert {
            topk: (
                program["block_x"], program["block_y"],
                program["service_cycles_per_cta"], program["service"],
            )
            for topk, program in mechanism["programs"].items()
        } == {
            2: (256, 2, 1191.75, "imcontinuous_vec4_reduce_topk2"),
            4: (128, 4, 1504.125, "imcontinuous_vec4_reduce_topk4"),
            8: (64, 8, 1864.25, "imcontinuous_vec4_reduce_topk8"),
            10: (64, 8, 2040.25, "imcontinuous_vec4_reduce_topk10"),
            16: (32, 16, 2018.125, "imcontinuous_vec4_reduce_topk16"),
        }
        assert mechanism["report_sha256"] == hashlib.sha256(
            reference_path.read_bytes()).hexdigest()


    def test_gemm_bf16_mma_saturation_is_typed(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        saturation = hw.gemm_bf16_mma_saturation

        assert saturation["model"] == (
            "register_only_bf16_mma_accumulator_saturation_v2")
        assert saturation["active_wave_points"][-1] == hw.max_warps_per_unit
        assert saturation["accumulator_chain_points"] == (4, 8, 16, 32)
        assert len(saturation["flops_per_ap_cycle_by_occupancy"]) == 24
        assert saturation["flops_per_ap_cycle_by_occupancy"][(4, 16)] == (
            pytest.approx(1916.971647))
        assert saturation["flops_per_ap_cycle_by_occupancy"][(8, 32)] == (
            pytest.approx(2015.519516))
        assert saturation["fallback_efficiency"] == pytest.approx(
            1994.060665 / 2048.0)
        assert not saturation[
            "production_or_evaluation_duration_consumed"]


    def test_gemm_bf16_mma_saturation_binds_selected_repeat(self, tmp_path):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["gemm_bf16_mma_saturation"][
            "result_sha256"] = "0" * 64
        path = tmp_path / "invalid_gemm_bf16_mma_saturation.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="repeat provenance drifted"):
            HWSpec(str(path))


    def test_metax_fused_moe_bm32_mma_occupancy_is_typed(self, tmp_path):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        occupancy = hw.metax_fused_moe_bm32_bf16_mma_occupancy

        assert occupancy["model"] == "metax_bm32_bf16_mma_occupancy_v1"
        assert {
            name: occupancy[name]
            for name in (
                "tile_m", "tile_n", "tile_k", "threads_per_cta",
                "dynamic_shared_memory_bytes", "resident_ctas_per_ap",
                "active_waves_per_ap", "accumulator_chains",
            )
        } == {
            "tile_m": 32,
            "tile_n": 64,
            "tile_k": 128,
            "threads_per_cta": 256,
            "dynamic_shared_memory_bytes": 49152,
            "resident_ctas_per_ap": 1,
            "active_waves_per_ap": 4,
            "accumulator_chains": 2,
        }
        assert occupancy["flops_per_ap_cycle"] == pytest.approx(
            1352.660353)
        assert not occupancy["production_or_evaluation_duration_consumed"]

        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["metax_fused_moe_bm32_bf16_mma_occupancy"][
            "result_sha256"] = "0" * 64
        path = tmp_path / "invalid_metax_fused_moe_bm32_mma.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="provenance drifted"):
            HWSpec(str(path))


    def test_metax_fused_moe_bm32_operand_pipeline_is_typed(self, tmp_path):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        pipeline = hw.metax_fused_moe_bm32_operand_pipeline

        assert pipeline["model"] == (
            "metax_bm32_bf16_grouped_operand_pipeline_v1")
        assert {
            name: pipeline[name]
            for name in (
                "tile_m", "tile_n", "tile_k", "group_size_m",
                "num_warps", "num_stages", "threads_per_cta",
                "dynamic_shared_memory_bytes_per_cta",
                "resident_ctas_per_ap", "pipeline_depth",
            )
        } == {
            "tile_m": 32,
            "tile_n": 64,
            "tile_k": 128,
            "group_size_m": 1,
            "num_warps": 4,
            "num_stages": 3,
            "threads_per_cta": 256,
            "dynamic_shared_memory_bytes_per_cta": 49152,
            "resident_ctas_per_ap": 1,
            "pipeline_depth": 3,
        }
        assert pipeline["fixed_span_ns"] == pytest.approx(6214.258274)
        assert pipeline["local_k_round_interval_ns"] == pytest.approx(
            215.332038)
        assert pipeline["empty_wave_slot_ns"] == pytest.approx(405.0)
        assert not pipeline["production_or_evaluation_duration_consumed"]

        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["metax_fused_moe_bm32_operand_pipeline"][
            "reference_sha256"] = "0" * 64
        path = tmp_path / "invalid_metax_fused_moe_bm32_pipeline.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="is not canonical"):
            HWSpec(str(path))


    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("threads_per_cta", 128, "must equal 256"),
            ("resident_ctas_per_unit", 4, "must equal 2"),
            ("barrier_cycles", 0.0, "finite and positive"),
            ("report_sha256", "invalid", "lowercase SHA256"),
        ),
    )
    def test_cta_barrier_service_rejects_invalid_schema(
            self, tmp_path, field, value, message):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["cta_barrier_service_cycles"][field] = value
        path = tmp_path / "invalid_cta_barrier.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises((TypeError, ValueError), match=message):
            HWSpec(str(path))


    def test_metax_fa2_d128_combine_dependency_excludes_oaccum(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        model = hw.metax_fa2_combine_dependency_cycles

        assert model["model"] == (
            "isolated_metax_fa2_d128_combine_dependency_v2")
        assert model["threads_per_cta"] == 256
        assert tuple(
            (entry["block_m"], entry["head_dim"], entry["max_splits"],
             entry["predicate_class"])
            for entry in model["entries"]
        ) == ((8, 128, 2, "full"), (8, 128, 8, "partial"))
        expected_stages = {
            "lse_shared_transpose",
            "max_allreduce",
            "post_max_exp_sum",
            "sum_allreduce",
            "post_sum_log",
            "scale_shared_barrier",
            "output_conversion",
        }
        assert all(
            set(entry["stage_cycles"]) == expected_stages
            and "oaccum_load_fma" not in entry["stage_cycles"]
            for entry in model["entries"])
        assert sum(model["entries"][0]["stage_cycles"].values()) == (
            pytest.approx(755.7265625))
        assert sum(model["entries"][1]["stage_cycles"].values()) == (
            pytest.approx(1040.0703125))

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("threads_per_cta", 128, "must equal 256"),
            ("static_shared_bytes", 32768, "must equal 2560"),
            ("resident_ctas_per_unit", 4, "must equal 2"),
            ("barrier_cycles", 0.0, "finite and positive"),
        ),
    )
    def test_flash_mla_combine_barrier_rejects_invalid_schema(
            self, tmp_path, field, value, message):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["flash_mla_combine_barrier_service_cycles"][
            field] = value
        path = tmp_path / "invalid_flash_mla_combine_barrier.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises((TypeError, ValueError), match=message):
            HWSpec(str(path))

    def test_metaxlink_uses_retained_topology_bottleneck(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")

        assert hw.interconnect["nvlink_bw_bytes_s"] == pytest.approx(50.7e9)
        assert hw.interconnect["nvlink_signal_latency_ns"] is None
        assert hw.interconnect["nccl_alpha_ns"] is None

    def test_fa2_ldgsts_frontend_is_absent(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")

        assert (
            hw.ldgsts_initial_32b_transactions_single_sm_cycle is None)
        assert (
            hw.ldgsts_initial_32b_transactions_per_active_sm_cycle is None)
        assert (
            hw.ldgsts_initial_32b_transactions_full_chip_per_cycle is None)

    def test_indirect_row_gather_services_are_exact_full_chip_signatures(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")

        assert hw.has_memory_access_class_service("indirect_row_gather")
        assert hw.has_memory_access_class_service(
            "metax_grouped_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_grouped_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_global_deep_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_global_large_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_long_lifetime_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_long_lifetime_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_repeated_fused_moe_deep_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_repeated_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_repeated_fused_moe_balanced_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_single_tile_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_single_tile_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_eight_active_twelve_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_single_tile_large_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_single_tile_large_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_small_single_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_small_single_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_active_grid_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e256_sixteen_active_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_sparse_padded_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e512_sparse_padded_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e512_forty_active_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e512_twenty_active_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e512_twenty_active_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e512_ten_active_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e512_ten_active_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e512_143_active_490_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e512_143_active_490_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e256_sparse_colliding_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e256_sparse_colliding_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e256_106_active_248_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e256_106_active_248_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_31_active_32_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_31_active_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_four_active_mid_k_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_four_active_mid_k_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_eight_active_mid_k_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e32_seven_active_mid_k_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e32_seven_active_mid_k_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm64_e32_32_active_40_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm64_e32_32_active_40_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_single_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_single_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_single_tile_large_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_single_tile_large_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_single_tile_large_deep_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_single_tile_large_deep_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e512_single_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e512_single_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_single_tile_mid_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_single_tile_mid_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_single_tile_large_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_single_tile_large_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_repeated_fused_moe_mid_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_repeated_fused_moe_mid_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_steady_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_steady_k192_n512_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_68_active_72_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_68_active_72_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_68_active_72_max_k128_n448_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_68_active_72_max_k448_n64_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e32_32_active_48_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e32_32_active_48_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e32_47_active_64_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e32_47_active_64_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e32_78_active_96_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e32_78_active_96_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e32_143_active_160_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e32_143_active_160_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_long_lifetime_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_long_lifetime_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e512_dense_ten_to_eleven_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e512_dense_ten_to_eleven_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e512_dense_activation_refreshed_large_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_long_lifetime_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_long_lifetime_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_deep_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_eight_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_eight_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_topk16_eight_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_152_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_152_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_64_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_124_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_128_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_128_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_136_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_136_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_184_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_184_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_248_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_bm16_e128_topk16_248_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_eight_tile_wide_n_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_twelve_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_twelve_tile_deep_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_twelve_tile_k128_n448_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_twelve_tile_k192_n512_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_twenty_one_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_twenty_one_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_twenty_active_twenty_four_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_eight_expert_twenty_active_twenty_four_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_sparse_two_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_sparse_two_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_188_active_254_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_188_active_254_max_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_single_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_single_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_upper_two_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_upper_two_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_repeated_source_topk8_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_repeated_source_topk16_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_repeated_k24_n32_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_eight_tile_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_eight_tile_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_eight_tile_mid_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_eight_tile_mid_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_single_tile_mid_deep_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_single_tile_mid_deep_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_four_tile_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_four_tile_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_four_tile_mid_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_four_tile_mid_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_four_tile_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_four_tile_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_four_tile_large_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_four_tile_large_fused_moe_w2_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_upper_two_tile_fused_moe_long_k_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e256_upper_two_tile_fused_moe_wide_n_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_topk8_128_active_190_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_topk16_128_active_190_max_fused_moe_w1_operand_pipeline")
        assert hw.has_memory_access_class_service(
            "metax_e128_128_active_190_max_fused_moe_w2_operand_pipeline")
        gather_services = tuple(
            service for service in hw.memory_access_class_services
            if service.service == "indirect_row_gather")
        assert len(gather_services) == 2
        by_signature = {
            (
                service.signature.tile_m,
                service.signature.tile_k,
                service.signature.logical_row_bytes,
            ): service
            for service in gather_services
        }
        assert set(by_signature) == {(16, 64, 128), (64, 32, 64)}
        assert by_signature[(16, 64, 128)].logical_byte_capacity_per_s == (
            686.45e9)
        assert by_signature[(64, 32, 64)].logical_byte_capacity_per_s == (
            733.75e9)
        assert all(
            service.access_kind == "read"
            and service.ingress_resource_role == "l2_cache"
            and service.scope == "full_chip"
            and service.signature.dtype == "bf16"
            and service.signature.operand_role is None
            and service.replay_to_dram_fraction is None
            for service in by_signature.values())
        grouped = {
            service.service: service
            for service in hw.memory_access_class_services
            if service.signature.operand_role == "combined"
        }
        expected_grouped = {
            "metax_grouped_fused_moe_long_k_operand_pipeline": (
                508126047370.714),
            "metax_grouped_fused_moe_wide_n_operand_pipeline": (
                1993803727360.7832),
            "metax_global_deep_fused_moe_long_k_operand_pipeline": (
                622465137115.988),
            "metax_global_large_fused_moe_wide_n_operand_pipeline": (
                1164381308417.5315),
            "metax_long_lifetime_fused_moe_long_k_operand_pipeline": (
                655124084652.9813),
            "metax_long_lifetime_fused_moe_wide_n_operand_pipeline": (
                2148102840309.0188),
            "metax_repeated_fused_moe_deep_k_operand_pipeline": (
                604950401763.9849),
            "metax_repeated_fused_moe_wide_n_operand_pipeline": (
                691367424846.1565),
            "metax_repeated_fused_moe_balanced_n_operand_pipeline": (
                1008425801855.0852),
            "metax_single_tile_fused_moe_long_k_operand_pipeline": (
                1014482658007.0348),
            "metax_single_tile_fused_moe_wide_n_operand_pipeline": (
                3043404294435.8525),
            "metax_eight_expert_eight_active_twelve_max_fused_moe_w2_operand_pipeline": (
                453555513307.89343),
            "metax_eight_expert_single_tile_large_fused_moe_w1_operand_pipeline": (
                2364856960157.138),
            "metax_eight_expert_single_tile_large_fused_moe_w2_operand_pipeline": (
                2256095038664.965),
            "metax_eight_expert_small_single_tile_fused_moe_w1_operand_pipeline": (
                2520459174586.7446),
            "metax_eight_expert_small_single_tile_fused_moe_w2_operand_pipeline": (
                2221858022107.694),
            "metax_bm16_active_grid_fused_moe_wide_n_operand_pipeline": (
                783788844198.5836),
            "metax_bm16_e256_sixteen_active_fused_moe_w2_operand_pipeline": (
                449908713550.8917),
            "metax_bm16_sparse_padded_fused_moe_w2_operand_pipeline": (
                721893344416.5354),
            "metax_bm16_e512_sparse_padded_fused_moe_w2_operand_pipeline": (
                828216918508.234),
            "metax_bm16_e512_twenty_active_fused_moe_w1_operand_pipeline": (
                1584908267019.4075),
            "metax_bm16_e512_twenty_active_fused_moe_w2_operand_pipeline": (
                547386985280.8124),
            "metax_bm16_e512_ten_active_fused_moe_w1_operand_pipeline": (
                1447856912943.8123),
            "metax_bm16_e512_ten_active_fused_moe_w2_operand_pipeline": (
                1413192877522.3435),
            "metax_bm16_e512_143_active_490_max_fused_moe_w1_operand_pipeline": (
                1554099288802.709),
            "metax_bm16_e512_143_active_490_max_fused_moe_w2_operand_pipeline": (
                1141998571106.0378),
            "metax_bm16_e512_75_active_80_max_fused_moe_w1_operand_pipeline": (
                1675343725255.03),
            "metax_bm16_e512_75_active_485_max_fused_moe_w2_operand_pipeline": (
                1137230511315.115),
            "metax_bm16_e512_mid_sparse_fused_moe_w1_operand_pipeline": (
                1554099288802.709),
            "metax_bm16_e512_mid_sparse_fused_moe_w2_operand_pipeline": (
                1141998571106.0378),
            "metax_bm16_e512_forty_active_fused_moe_w2_operand_pipeline": (
                847722621999.7976),
            "metax_bm16_e256_sparse_colliding_fused_moe_w1_operand_pipeline": (
                1582666648493.977),
            "metax_bm16_e256_sparse_colliding_fused_moe_w2_operand_pipeline": (
                1016441619707.4285),
            "metax_bm16_e256_106_active_248_max_fused_moe_w1_operand_pipeline": (
                1621798650893.6812),
            "metax_bm16_e256_106_active_248_max_fused_moe_w2_operand_pipeline": (
                1308892268146.4368),
            "metax_bm16_31_active_32_max_fused_moe_w1_operand_pipeline": (
                1467381238218.3628),
            "metax_bm16_e128_31_active_fused_moe_w2_operand_pipeline": (
                1202796738293.8816),
            "metax_bm16_four_active_mid_k_fused_moe_w1_operand_pipeline": (
                1465595151651.7842),
            "metax_bm16_four_active_mid_k_fused_moe_w2_operand_pipeline": (
                1292946576951.871),
            "metax_bm16_eight_active_mid_k_fused_moe_w2_operand_pipeline": (
                1040258553917.1323),
            "metax_bm16_e32_seven_active_mid_k_fused_moe_w1_operand_pipeline": (
                1556595588883.4685),
            "metax_bm16_e32_seven_active_mid_k_fused_moe_w2_operand_pipeline": (
                1408400036885.4778),
            "metax_bm16_e128_53_active_64_max_fused_moe_w1_operand_pipeline": (
                1619579010866.7063),
            "metax_bm16_e128_53_active_124_max_fused_moe_w2_operand_pipeline": (
                1412604069151.0583),
            "metax_bm16_e128_eight_active_mid_k_fused_moe_w1_operand_pipeline": (
                1567652114864.0654),
            "metax_bm16_e128_eight_active_mid_k_fused_moe_w2_operand_pipeline": (
                1458171355233.6504),
            "metax_e256_single_tile_fused_moe_w1_operand_pipeline": (
                2566911866344.2544),
            "metax_e256_single_tile_fused_moe_w2_operand_pipeline": (
                2456692585670.414),
            "metax_e256_single_tile_large_fused_moe_w1_operand_pipeline": (
                2441914081428.5366),
            "metax_e256_single_tile_large_fused_moe_w2_operand_pipeline": (
                2610440313654.1196),
            "metax_e256_single_tile_large_deep_fused_moe_w1_operand_pipeline": (
                1561166932969.9375),
            "metax_e256_single_tile_large_deep_fused_moe_w2_operand_pipeline": (
                1514143376972.5815),
            "metax_e512_single_tile_fused_moe_w1_operand_pipeline": (
                1393628793100.2441),
            "metax_e512_single_tile_fused_moe_w2_operand_pipeline": (
                1665962537394.1636),
            "metax_e128_single_tile_mid_fused_moe_w1_operand_pipeline": (
                2131235306913.2432),
            "metax_e128_single_tile_mid_fused_moe_w2_operand_pipeline": (
                1370484530548.881),
            "metax_e128_single_tile_large_fused_moe_w1_operand_pipeline": (
                2615159180369.558),
            "metax_e128_single_tile_large_fused_moe_w2_operand_pipeline": (
                2476333629644.2046),
            "metax_repeated_fused_moe_mid_long_k_operand_pipeline": (
                1070846486054.9147),
            "metax_repeated_fused_moe_mid_wide_n_operand_pipeline": (
                874114351029.0723),
            "metax_eight_expert_steady_fused_moe_wide_n_operand_pipeline": (
                2274302584843.376),
            "metax_eight_expert_steady_k192_n512_fused_moe_w1_operand_pipeline": (
                2408725024059.311),
            "metax_eight_expert_68_active_72_max_fused_moe_w1_operand_pipeline": (
                2658770776305.729),
            "metax_eight_expert_68_active_72_max_fused_moe_w2_operand_pipeline": (
                1313057123385.8208),
            "metax_eight_expert_68_active_72_max_k128_n448_fused_moe_w1_operand_pipeline": (
                2760263325486.602),
            "metax_eight_expert_68_active_72_max_k448_n64_fused_moe_w2_operand_pipeline": (
                1158616598974.0437),
            "metax_e32_32_active_48_max_fused_moe_w1_operand_pipeline": (
                3639649767346.0957),
            "metax_e32_32_active_48_max_fused_moe_w2_operand_pipeline": (
                1745590172592.3005),
            "metax_e32_47_active_64_max_fused_moe_w1_operand_pipeline": (
                3012523245854.0615),
            "metax_e32_47_active_64_max_fused_moe_w2_operand_pipeline": (
                517471589714.3835),
            "metax_e32_78_active_96_max_fused_moe_w1_operand_pipeline": (
                2261726943705.3633),
            "metax_e32_78_active_96_max_fused_moe_w2_operand_pipeline": (
                597925559051.866),
            "metax_e32_143_active_160_max_fused_moe_w1_operand_pipeline": (
                1964611445153.842),
            "metax_e32_143_active_160_max_fused_moe_w2_operand_pipeline": (
                887011464428.7026),
            "metax_e256_long_lifetime_fused_moe_long_k_operand_pipeline": (
                898416143776.8531),
            "metax_e256_long_lifetime_fused_moe_wide_n_operand_pipeline": (
                605373260135.7303),
            "metax_e512_dense_ten_to_eleven_tile_fused_moe_w1_operand_pipeline": (
                1200039183512.872),
            "metax_e512_dense_ten_to_eleven_tile_fused_moe_w2_operand_pipeline": (
                1042966078286.4961),
            "metax_e512_dense_activation_refreshed_large_w2_operand_pipeline": (
                608685060855.3701),
            "metax_e128_long_lifetime_fused_moe_w1_operand_pipeline": (
                1183722906708.2878),
            "metax_e128_long_lifetime_fused_moe_w2_operand_pipeline": (
                635660657583.6455),
            "metax_e128_deep_fused_moe_w2_operand_pipeline": (
                622072184468.4724),
            "metax_e128_eight_tile_fused_moe_w1_operand_pipeline": (
                1174968747077.5234),
            "metax_e128_eight_tile_fused_moe_w2_operand_pipeline": (
                697195713781.3363),
            "metax_e128_topk16_eight_tile_fused_moe_w1_operand_pipeline": (
                2902547283916.5557),
            "metax_bm16_e128_topk16_152_max_fused_moe_w1_operand_pipeline": (
                1761832966670.8213),
            "metax_bm16_e128_topk16_152_max_fused_moe_w2_operand_pipeline": (
                1661172049826.454),
            "metax_bm16_e128_topk16_64_max_fused_moe_w1_operand_pipeline": (
                1602507209923.7031),
            "metax_bm16_e128_topk16_124_max_fused_moe_w2_operand_pipeline": (
                1360622965414.0073),
            "metax_bm16_e128_topk16_128_max_fused_moe_w1_operand_pipeline": (
                1660836902137.097),
            "metax_bm16_e128_topk16_128_max_fused_moe_w2_operand_pipeline": (
                1534314599056.3462),
            "metax_bm16_e128_topk16_136_max_fused_moe_w1_operand_pipeline": (
                1701974968954.6611),
            "metax_bm16_e128_topk16_136_max_fused_moe_w2_operand_pipeline": (
                1612920383318.8184),
            "metax_bm16_e128_topk16_184_max_fused_moe_w1_operand_pipeline": (
                1887405248292.8606),
            "metax_bm16_e128_topk16_184_max_fused_moe_w2_operand_pipeline": (
                1732107112545.6567),
            "metax_bm16_e128_topk16_248_max_fused_moe_w1_operand_pipeline": (
                2330675172794.005),
            "metax_bm16_e128_topk16_248_max_fused_moe_w2_operand_pipeline": (
                2057914182595.3792),
            "metax_e128_eight_tile_wide_n_fused_moe_w2_operand_pipeline": (
                681690158790.7874),
            "metax_e128_single_tile_fused_moe_w1_operand_pipeline": (
                2568786698424.878),
            "metax_e128_single_tile_fused_moe_w2_operand_pipeline": (
                2369132030230.114),
            "metax_eight_expert_twenty_one_tile_fused_moe_w1_operand_pipeline": (
                2402529620775.3833),
            "metax_eight_expert_twenty_one_tile_fused_moe_w2_operand_pipeline": (
                640827817324.4935),
            "metax_eight_expert_twenty_active_twenty_four_max_fused_moe_w1_operand_pipeline": (
                2171491917292.5054),
            "metax_eight_expert_twenty_active_twenty_four_max_fused_moe_w2_operand_pipeline": (
                623804617631.5361),
            "metax_eight_expert_twelve_tile_fused_moe_w2_operand_pipeline": (
                408286512237.02045),
            "metax_eight_expert_twelve_tile_deep_fused_moe_w2_operand_pipeline": (
                541864677430.61395),
            "metax_eight_expert_twelve_tile_k128_n448_fused_moe_w1_operand_pipeline": (
                2059442115031.219),
            "metax_eight_expert_twelve_tile_k192_n512_fused_moe_w1_operand_pipeline": (
                1856470554518.9795),
            "metax_e128_sparse_two_tile_fused_moe_w1_operand_pipeline": (
                1068521762264.2522),
            "metax_e128_sparse_two_tile_fused_moe_w2_operand_pipeline": (
                539536391141.0292),
            "metax_e128_188_active_254_max_fused_moe_w1_operand_pipeline": (
                1591336154581.2666),
            "metax_e128_188_active_254_max_fused_moe_w2_operand_pipeline": (
                537916948909.9443),
            "metax_e128_upper_two_tile_fused_moe_w1_operand_pipeline": (
                1336953737898.0642),
            "metax_e128_upper_two_tile_fused_moe_w2_operand_pipeline": (
                808752430005.3013),
            "metax_e128_topk8_128_active_190_max_fused_moe_w1_operand_pipeline": (
                2559333566462.93),
            "metax_e128_topk16_128_active_190_max_fused_moe_w1_operand_pipeline": (
                2698553810793.005),
            "metax_e128_128_active_190_max_fused_moe_w2_operand_pipeline": (
                2201895980056.4463),
            "metax_e128_repeated_source_topk8_fused_moe_w1_operand_pipeline": (
                1668209111834.5667),
            "metax_e128_repeated_source_topk16_fused_moe_w1_operand_pipeline": (
                2682085658245.4253),
            "metax_e128_repeated_k24_n32_fused_moe_w2_operand_pipeline": (
                732468315351.4436),
            "metax_e256_eight_tile_fused_moe_long_k_operand_pipeline": (
                979450127601.1329),
            "metax_e256_eight_tile_fused_moe_wide_n_operand_pipeline": (
                663648043090.0348),
            "metax_e256_eight_tile_mid_fused_moe_w1_operand_pipeline": (
                1236532659297.142),
            "metax_e256_eight_tile_mid_fused_moe_w2_operand_pipeline": (
                1075395665052.2191),
            "metax_e256_single_tile_mid_deep_fused_moe_w1_operand_pipeline": (
                2691223045093.1665),
            "metax_e256_single_tile_mid_deep_fused_moe_w2_operand_pipeline": (
                2885818204727.008),
            "metax_e256_four_tile_fused_moe_long_k_operand_pipeline": (
                1244471286428.599),
            "metax_e256_four_tile_fused_moe_wide_n_operand_pipeline": (
                983283928173.2489),
            "metax_e256_four_tile_mid_fused_moe_long_k_operand_pipeline": (
                1076271868877.4132),
            "metax_e256_four_tile_mid_fused_moe_wide_n_operand_pipeline": (
                1010401399126.2913),
            "metax_e128_four_tile_fused_moe_w1_operand_pipeline": (
                1042007322320.5751),
            "metax_e128_four_tile_fused_moe_w2_operand_pipeline": (
                960616571004.986),
            "metax_e128_four_tile_large_fused_moe_w1_operand_pipeline": (
                1419461474794.4766),
            "metax_e128_four_tile_large_fused_moe_w2_operand_pipeline": (
                875191968734.7042),
            "metax_e256_upper_two_tile_fused_moe_long_k_operand_pipeline": (
                798218071802.4584),
            "metax_e256_upper_two_tile_fused_moe_wide_n_operand_pipeline": (
                1070485657117.5874),
            "metax_bm64_e32_32_active_36_max_fused_moe_w1_operand_pipeline": (
                3238510330193.884),
            "metax_bm64_e32_32_active_36_max_fused_moe_w2_operand_pipeline": (
                3128545600407.7847),
            "metax_bm64_e32_32_active_40_max_fused_moe_w1_operand_pipeline": (
                3445481983842.3296),
            "metax_bm64_e32_32_active_40_max_fused_moe_w2_operand_pipeline": (
                3224484621459.3301),
            "metax_e512_1010_active_1144_max_fused_moe_w2_operand_pipeline": (
                906987228025.9921),
            "metax_bm16_e512_521_active_800_max_fused_moe_w1_operand_pipeline": (
                1980519986532.635),
            "metax_bm16_e512_521_active_800_max_fused_moe_w2_operand_pipeline": (
                1665184547517.7207),
            "metax_bm64_e512_512_active_664_max_fused_moe_w1_operand_pipeline": (
                2600228822291.929),
            "metax_bm64_e512_512_active_664_max_fused_moe_w2_operand_pipeline": (
                2471695544407.433),
            "metax_e256_long_lifetime_mid_fused_moe_w1_operand_pipeline": (
                1254956266011.0376),
            "metax_e256_long_lifetime_mid_fused_moe_w2_operand_pipeline": (
                982773151357.9355),
        }
        for name, expected_capacity in expected_grouped.items():
            assert grouped[name].logical_byte_capacity_per_s == pytest.approx(
                expected_capacity)
        assert all(
            (
                service.signature.tile_m,
                service.signature.tile_n,
                service.signature.tile_k,
                service.signature.group_size_m,
                service.signature.logical_row_bytes,
            ) in {
                (16, 32, 64, 1, 6144),
                (64, 64, 32, 8, 8192),
            }
            and service.signature.operand_role == "combined"
            for service in grouped.values())


    def test_access_class_schema_rejects_duplicate_exact_signature(
            self, tmp_path):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        services = cfg["memory_subsystem"][
            "memory_access_class_services"]
        services.append(dict(services[0]))
        path = tmp_path / "duplicate_access_class.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="repeat an exact service"):
            HWSpec(str(path))

    def test_access_class_schema_rejects_non_full_chip_scope(self, tmp_path):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["memory_subsystem"]["memory_access_class_services"][0][
            "scope"] = "per_unit"
        path = tmp_path / "per_unit_access_class.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="scope must be 'full_chip'"):
            HWSpec(str(path))

    def test_grouped_operand_service_normalizes_exact_signature_and_replay(
            self, tmp_path):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        service = cfg["memory_subsystem"][
            "memory_access_class_services"][0]
        service["service"] = "grouped_fused_moe_operand_read"
        service["signature"]["access_class"] = service["service"]
        service["signature"]["operand_role"] = "combined"
        service["signature"]["tile_n"] = 32
        service["signature"]["group_size_m"] = 1
        service["replay_to_dram_fraction"] = 0.125
        cfg["memory_subsystem"]["memory_access_class_services"] = [service]
        path = tmp_path / "grouped_operand_access_class.yaml"
        path.write_text(yaml.safe_dump(cfg))

        hardware = HWSpec(str(path))
        capability = hardware.memory_access_class_services[0]

        assert capability.signature.operand_role == "combined"
        assert capability.signature.tile_n == 32
        assert capability.signature.group_size_m == 1
        assert capability.replay_to_dram_fraction == pytest.approx(0.125)
        assert hardware.memory_access_class_service(
            capability.service,
            capability.access_kind,
            capability.ingress_resource_role,
            capability.signature,
        ) == capability

    @pytest.mark.parametrize(
        ("signature_updates", "replay_fraction", "message"),
        (
            (
                {"tile_n": None},
                0.25,
                "grouped operand fields must be all present or all null",
            ),
            (
                {
                    "operand_role": "activation",
                    "tile_n": 32,
                    "group_size_m": 1,
                },
                1.01,
                "replay_to_dram_fraction must be within",
            ),
        ),
    )
    def test_grouped_operand_service_rejects_incomplete_or_invalid_evidence(
            self, tmp_path, signature_updates, replay_fraction, message):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        service = cfg["memory_subsystem"][
            "memory_access_class_services"][0]
        service["service"] = "grouped_fused_moe_operand_read"
        service["signature"]["access_class"] = service["service"]
        service["signature"]["operand_role"] = "combined"
        service["signature"]["tile_n"] = 32
        service["signature"]["group_size_m"] = 1
        service["signature"].update(signature_updates)
        service["replay_to_dram_fraction"] = replay_fraction
        cfg["memory_subsystem"]["memory_access_class_services"] = [service]
        path = tmp_path / "invalid_grouped_operand_access_class.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match=message):
            HWSpec(str(path))

    def test_production_launch_latency_uses_maca_372_scheduler_mechanism(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        assert hw.dispatch_overhead_ns == 0.0
        assert hw.empty_kernel_duration_lut is None
        assert hw.launch_latency_model["model"] == (
            "isolated_nonempty_task_scheduler_v1")
        assert hw.launch_latency_model["timer_domain"] == (
            "mctracer_gpu_kernel_event")
        assert hw.launch_latency_model["software_revision"] == "maca_3.7.2"
        assert hw.empty_kernel_duration_for_launch(
            1, 64) == pytest.approx(2204.2009241440924)
        assert hw.empty_kernel_duration_for_launch(
            104, 64) == pytest.approx(2308.8943270192003)
        assert hw.empty_kernel_duration_for_launch(
            105, 128) == pytest.approx(2538.0728866014615)
        assert hw.empty_kernel_duration_for_launch(
            425984, 1024) == pytest.approx(4362666.654013936)

    def test_nonempty_scheduler_model_consumes_dim3_products(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        flat = hw.empty_kernel_duration_for_launch(128, 256)
        dim3 = hw.empty_kernel_duration_for_launch(
            128, 256, (8, 4, 4), (8, 8, 4))
        assert dim3 == pytest.approx(flat)

        with pytest.raises(ValueError, match="disagrees with its flattened"):
            hw.empty_kernel_duration_for_launch(
                128, 256, (8, 4, 2), (8, 8, 4))

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("cta_descriptor_ns", 0.0, "required services"),
            ("single_wave_round_ns", -1.0, "finite and non-negative"),
            ("timer_domain", "maca_event", "unsupported"),
            ("software_revision", "", "must be non-empty"),
        ),
    )
    def test_nonempty_scheduler_model_rejects_invalid_parameters(
            self, tmp_path, field, value, message):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["launch_latency_model"][field] = value
        path = tmp_path / "invalid_nonempty_launch_model.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises((TypeError, ValueError), match=message):
            HWSpec(str(path))

    def test_exact_empty_kernel_floor_preserves_full_launch_geometry(
            self, tmp_path):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        del cfg["scheduling"]["launch_latency_model"]
        cfg["scheduling"]["empty_kernel_duration_lut"] = {
            "geometry_policy": "exact_grid_and_block_only",
            "duration_unit": "ns_per_launch",
            "entries": [
                {
                    "grid": [1024, 1, 1], "block": [1024, 1, 1],
                    "duration_ns": 11000,
                },
                {
                    "grid": [1, 512, 2], "block": [1024, 1, 1],
                    "duration_ns": 12000,
                },
            ],
        }
        path = tmp_path / "exact_c500_launch_floor.yaml"
        path.write_text(yaml.safe_dump(cfg))
        hw = HWSpec(str(path))

        assert hw.empty_kernel_duration_for_launch(
            1024, 1024, (1024, 1, 1), (1024, 1, 1),
        ) == 11000
        assert hw.empty_kernel_duration_for_launch(
            1024, 1024, (1, 512, 2), (1024, 1, 1),
        ) == 12000
        with pytest.raises(ValueError, match="requires full launch"):
            hw.empty_kernel_duration_for_launch(1024, 1024)
        with pytest.raises(ValueError, match="absent from the exact"):
            hw.empty_kernel_duration_for_launch(
                1024, 1024, (2, 512, 1), (1024, 1, 1),
            )

    def test_exact_empty_kernel_floor_rejects_duplicate_geometry(
            self, tmp_path):
        with open("configs/hardware/metax_c500.yaml") as stream:
            cfg = yaml.safe_load(stream)
        entry = {
            "grid": [1, 1, 1], "block": [64, 1, 1],
            "duration_ns": 11000,
        }
        cfg["scheduling"]["empty_kernel_duration_lut"] = {
            "geometry_policy": "exact_grid_and_block_only",
            "duration_unit": "ns_per_launch",
            "entries": [entry, dict(entry)],
        }
        path = tmp_path / "duplicate_c500_launch_floor.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="duplicate geometry"):
            HWSpec(str(path))

    def test_compute_capacities_match_static_and_mechanism_evidence(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")

        expected_per_peu_per_clock = {
            "compute_tensor": 512.0,
            "compute_fma": 32.0,
            "compute_sfu": 4.0,
            "compute_alu": 16.0,
        }
        for role, expected in expected_per_peu_per_clock.items():
            resource = hw.resource_by_role(role)
            assert resource.total_units == 104 * 4
            per_peu_per_clock = (
                resource.effective_capacity(
                    hw.num_compute_units,
                    hw.nominal_sm_freq_hz,
                    hw.nominal_dram_freq_hz,
                )
                / hw.nominal_sm_freq_hz
                / resource.total_units
            )
            assert per_peu_per_clock == pytest.approx(expected)

    def test_silu_dependency_services_use_per_peu_mechanism_rates(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        resource = hw.resource_by_role("compute_sfu")

        assert resource.total_units == 104 * 4
        assert resource.service_rate_for_one_unit(
            "exp_dependency") == pytest.approx(25835099.9728141 / 4)
        assert resource.service_rate_for_one_unit(
            "reciprocal_one_plus_dependency") == pytest.approx(
                8417452.829123823 / 4)
        assert resource.setup_time_for_service(
            "exp_dependency") == pytest.approx(4.04296770284418e-09)
        assert resource.setup_time_for_service(
            "reciprocal_one_plus_dependency") == pytest.approx(
                1.4246088539948659e-08)

    def test_silu_slot2_service_matches_one_instruction_per_ap_clock(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        resource = hw.resource_by_role("compute_vector")

        assert resource.total_units == 104
        assert resource.service_rate_for_one_unit(
            "vector_alu_issue_slot_instruction") == pytest.approx(
                hw.nominal_sm_freq_hz)

    def test_warp_issue_capacity_matches_five_slot_ap_envelope(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        issue = hw.resource_by_role("warp_issue")

        assert issue.scope == "per_unit"
        assert issue.total_units == 104
        assert issue.effective_capacity(
            hw.num_compute_units,
            hw.nominal_sm_freq_hz,
            hw.nominal_dram_freq_hz,
        ) == pytest.approx(5 * 104 * hw.nominal_sm_freq_hz)
    def test_matrix_vector_execution_is_separate_from_shared_issue(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        groups = {group.name: group for group in hw.resource_groups}

        assert groups["tensor_pipe"].members == ["tensor_core_fp16"]
        assert groups["vector_pipe"].members == [
            "fma_pipe", "sfu_pipe", "alu_pipe"]
        assert groups["warp_issue"].members == ["warp_dispatch"]
        assert "matrix_vector_issue" not in groups

    def test_unmeasured_source_ceiling_is_explicitly_absent(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")

        assert hw.kernel_memory_model == "gpu_mshr"
        assert hw.source_slots_per_unit is None
        assert not hasattr(hw, "dram_saturation_warps")
        assert not hasattr(hw, "dram_saturation_warps_cached")

    def test_c500_directional_dram_capacities_have_physical_evidence(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")

        assert hw.dram_arbitration_mode == "shared_directional_capacity"
        assert hw.dram_read_capacity_per_clk == pytest.approx(918.372)
        assert hw.dram_write_capacity_per_clk == pytest.approx(
            893.2556047625426
        )

class TestHWSpecNPU:
    def test_l2_replacement_capacity_is_evidence_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.l2_size_bytes == 192 * 1024 * 1024
        assert hw.l2_replacement_capacity_bytes == 160 * 1024 * 1024
        assert hw.l2_replacement_capacity_evidence_ref == (
            "microbench/ascend_910b/references/"
            "shared_l2_replacement_capacity_cann_8_5_1.json")
        for path in (
                "configs/hardware/nvidia_a800.yaml",
                "configs/hardware/metax_c500.yaml"):
            other = HWSpec(path)
            assert other.l2_replacement_capacity_bytes == other.l2_size_bytes
            assert other.l2_replacement_capacity_evidence_ref is None

    def test_loads_without_error(self):
        hw = HWSpec("tests/kernel/hardware_specs/ascend_910b_fixture.yaml")
        assert hw is not None


    def test_grouped_operand_service_is_exact_and_duration_independent(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        services = {
            service.service: service
            for service in hw.memory_access_class_services
            if service.service.startswith("cann_grouped_moe_operand_")}
        assert set(services) == {
            "cann_grouped_moe_operand_a_l2_hit",
            "cann_grouped_moe_operand_a_dram_miss",
            "cann_grouped_moe_operand_b_l2_hit",
            "cann_grouped_moe_operand_b_k2880_stride_l2_hit",
            "cann_grouped_moe_operand_b_dram_miss",
            "cann_grouped_moe_operand_b_k2880_strided_dram_miss",
            "cann_grouped_moe_operand_b_k6144_m1_dram_miss",
        }
        expected = {
            "cann_grouped_moe_operand_a_l2_hit": (
                "activation", 16384, 3607356961118.079,
                "l2_resident_pair_service"),
            "cann_grouped_moe_operand_a_dram_miss": (
                "activation", 16384, 1511744941456.1821,
                "dram_miss_operand_services"),
            "cann_grouped_moe_operand_b_l2_hit": (
                "weight", 32768, 4259227643602.7666,
                "l2_resident_pair_service"),
            "cann_grouped_moe_operand_b_k2880_stride_l2_hit": (
                "weight", 32768, 2461377225093.7627,
                "parameters_ns"),
            "cann_grouped_moe_operand_b_dram_miss": (
                "weight", 32768, 1537682661220.9097,
                "dram_miss_operand_services"),
            "cann_grouped_moe_operand_b_k2880_strided_dram_miss": (
                "weight", 32768, 794835731957.9452, "parameters_ns"),
            "cann_grouped_moe_operand_b_k6144_m1_dram_miss": (
                "weight", 32768, 1194094410048.0, "parameters_ns"),
        }
        for service_name, service in services.items():
            role, logical_bytes, capacity, anchor = expected[service_name]
            assert service.access_kind == "read"
            assert service.ingress_resource_role == "l2_cache"
            assert service.scope == "full_chip"
            assert service.logical_byte_capacity_per_s == pytest.approx(
                capacity)
            assert service.replay_to_dram_fraction is None
            assert service.signature.access_class == service.service
            assert service.signature.dtype == "bf16"
            assert service.signature.tile_m == 128
            assert service.signature.tile_n == 256
            assert service.signature.tile_k == 64
            assert service.signature.logical_row_bytes == logical_bytes
            assert service.signature.operand_role == role
            assert service.signature.group_size_m == 1
            reference_names = {
                "cann_grouped_moe_operand_a_l2_hit":
                    "grouped_matmul_operand_pair_cann_8_5_1.json",
                "cann_grouped_moe_operand_a_dram_miss":
                    "grouped_matmul_operand_pair_cann_8_5_1.json",
                "cann_grouped_moe_operand_b_l2_hit":
                    "grouped_matmul_operand_pair_cann_8_5_1.json",
                "cann_grouped_moe_operand_b_k2880_stride_l2_hit":
                    "grouped_matmul_k2880_b_stride_cann_8_5_1.json",
                "cann_grouped_moe_operand_b_dram_miss":
                    "grouped_matmul_operand_pair_cann_8_5_1.json",
                "cann_grouped_moe_operand_b_k2880_strided_dram_miss":
                    "grouped_matmul_k2880_m1_b_dram_cann_8_5_1.json",
                "cann_grouped_moe_operand_b_k6144_m1_dram_miss":
                    "grouped_matmul_k6144_m1_b_dram_cann_8_5_1.json",
            }
            reference_name = reference_names[service_name]
            assert service.evidence_ref == (
                "microbench/ascend_910b/references/"
                f"{reference_name}#{anchor}")
        evidence_path = services[
            "cann_grouped_moe_operand_a_l2_hit"].evidence_ref.split("#", 1)[0]
        evidence = json.loads(Path(evidence_path).read_text())
        assert evidence["measurement_gate_passed"] is True
        assert evidence["mechanism_publishable"] is True

        assert evidence[
            "production_or_evaluation_duration_consumed"] is False
        assert evidence["dram_miss_operand_services"][
            "a_logical_byte_capacity_per_s"] == pytest.approx(
                services["cann_grouped_moe_operand_a_dram_miss"].
                logical_byte_capacity_per_s)
        assert evidence["dram_miss_operand_services"][
            "b_logical_byte_capacity_per_s"] == pytest.approx(
                services["cann_grouped_moe_operand_b_dram_miss"].
                logical_byte_capacity_per_s)
        stride_evidence_path = services[
            "cann_grouped_moe_operand_b_k2880_stride_l2_hit"].evidence_ref
        stride_evidence = json.loads(Path(
            stride_evidence_path.split("#", 1)[0]).read_text())
        assert stride_evidence["publication_scope"] == {
            "source_residency": "l2_resident",
            "dtype": "bf16",
            "rows": 64,
            "columns": 256,
            "source_row_stride_elements": 2880,
            "source_tile_span_bytes": 363392,
            "copied_bytes_per_descriptor": 32768,
            "active_aic_groups": 24,
            "iterations_per_worker": 224,
            "source_allocation_bytes": 67108864,
            "source_tiles_per_worker": 7,
            "aggregate_copied_ring_bytes": 5505024,
            "source_ring_constraint": (
                "the 5.25 MiB aggregate copied ring remains resident in the "
                "192 MiB shared L2"),
            "source_lifecycle": "two_buffer_stream",
        }
        assert stride_evidence["parameters_ns"][
            "two_buffer_stream_full_chip_logical_byte_capacity_per_s"] == (
                pytest.approx(services[
                    "cann_grouped_moe_operand_b_k2880_stride_l2_hit"].
                    logical_byte_capacity_per_s))
        assert stride_evidence[
            "production_or_evaluation_duration_consumed"] is False
        assert stride_evidence["parameter_fit_from_production_duration"] is False
        assert stride_evidence["mechanism_publishable"] is True
        assert stride_evidence["forward_eligible"] is True
        cold_b_evidence_path = services[
            "cann_grouped_moe_operand_b_k6144_m1_dram_miss"].evidence_ref
        cold_b_evidence = json.loads(Path(
            cold_b_evidence_path.split("#", 1)[0]).read_text())
        assert cold_b_evidence["mechanism_publishable"] is True
        assert cold_b_evidence["forward_eligible"] is True
        assert cold_b_evidence[
            "production_or_evaluation_duration_consumed"] is False
        assert cold_b_evidence["parameters_ns"][
            "two_buffer_stream_full_chip_logical_byte_capacity_per_s"] == (
            pytest.approx(services[
                "cann_grouped_moe_operand_b_k6144_m1_dram_miss"].
                logical_byte_capacity_per_s))
        k2880_cold_b_evidence_path = services[
            "cann_grouped_moe_operand_b_k2880_strided_dram_miss"].evidence_ref
        k2880_cold_b_evidence = json.loads(Path(
            k2880_cold_b_evidence_path.split("#", 1)[0]).read_text())
        assert k2880_cold_b_evidence["mechanism_publishable"] is True
        assert k2880_cold_b_evidence["forward_eligible"] is True
        assert k2880_cold_b_evidence[
            "production_or_evaluation_duration_consumed"] is False
        assert [
            (program["base_m"], program["depth_a1"],
             program["depth_b1"], program["step_ka"],
             program["step_kb"])
            for program in k2880_cold_b_evidence["publication_scope"][
                "source_programs"]
        ] == [
            (16, 120, 8, 60, 4),
            (32, 56, 8, 28, 4),
            (64, 24, 8, 12, 4),
            (128, 8, 8, 4, 4),
        ]
        assert k2880_cold_b_evidence["parameters_ns"][
            "two_buffer_stream_full_chip_logical_byte_capacity_per_s"] == (
            pytest.approx(services[
                "cann_grouped_moe_operand_b_k2880_strided_dram_miss"].
                logical_byte_capacity_per_s))
        assert not hasattr(hw, "mshr_concurrency")
        assert not hasattr(hw, "dram_outstanding_per_sm")


    def test_elementwise_single_tile_finite_program_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        resource = hw.resource_by_role("mte3")
        service_name = (
            "aiv_elementwise_single_tile_finite_worker_completion_47_workers")
        service = resource.exact_request_services[service_name]
        assert service["request_size_unit"] == "source_worker_programs"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        source_path = Path(evidence["source_path"])
        for name, expected_sha256 in evidence["source_files_sha256"].items():
            assert hashlib.sha256(
                (source_path / name).read_bytes()).hexdigest() == (
                    expected_sha256)
        assert resource.exact_request_service_time_s(
            service_name, 1) == pytest.approx(1080.0e-9)

    @pytest.mark.parametrize(
        ("service_name", "expected_ns"),
        (
            ("aiv_mul_q4q1_finite_worker_completion_48_workers_11008b_x2",
             3320.0),
            ("aiv_mul_q4q1_finite_worker_completion_48_workers_13824b_x2",
             4400.0),
            ("aiv_mul_q4q1_finite_worker_completion_48_workers_18432b_x3",
             7980.0),
            ("aiv_mul_q4q1_finite_worker_completion_48_workers_17664b_x5",
             12520.0),
        ),
    )
    def test_mul_q4q1_additional_finite_program_is_source_bound(
            self, service_name, expected_ns):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        resource = hw.resource_by_role("mte3")
        service = resource.exact_request_services[service_name]
        assert service["request_size_unit"] == "source_worker_programs"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["service"] == service_name
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        source_path = Path(evidence["source_path"])
        for name, expected_sha256 in evidence["source_files_sha256"].items():
            assert hashlib.sha256(
                (source_path / name).read_bytes()).hexdigest() == (
                    expected_sha256)
        assert resource.exact_request_service_time_s(
            service_name, 1) == pytest.approx(expected_ns * 1e-9)

    def test_model_clock_domain_is_not_hardcoded_to_sm(self):
        hw = HWSpec("tests/kernel/hardware_specs/ascend_910b_fixture.yaml")
        tensor = hw.resource_by_role("compute_tensor")
        half_clock = hw.nominal_sm_freq_hz / 2.0
        assert tensor.effective_capacity(
            hw.num_compute_units,
            half_clock,
            hw.nominal_dram_freq_hz,
        ) == pytest.approx(tensor.capacity / 2.0)

    def test_shared_vector_resource_preserves_physical_unit_count(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        vector = hw.resource_by_role("compute_vector")

        assert vector.scope == "shared"
        assert vector.num_units == 0
        assert vector.total_units == 48


    def test_swiglu_queue_edge_services_use_only_production_valid_cells(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        evidence_path = Path(
            "microbench/ascend_910b/references/"
            "aiv_swiglu_t1024_queue_edge_completion_cann_8_5_1.json")
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
        evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
        assert evidence["source_tile_evidence"]["unloaded_control"] == {
            "payload_bytes": 19456, "block_dim": 1}
        assert evidence["source_tile_evidence"][
            "representative_geometries"] == [
                {"payload_bytes": 19456, "block_dim": 24},
                {"payload_bytes": 19456, "block_dim": 48},
            ]
        assert evidence["source_tile_evidence"][
            "source_valid_payload_control"] == {
                "elements": 13312, "dtype_bytes": 2,
                "payload_bytes": 26624, "block_dim": 48,
            }
        direction_contract = {
            "tque_mte2_to_vector": (
                "mte2", "aiv_swiglu_tque_mte2_to_vector_completion"),
            "tque_vector_to_mte3": (
                "mte3", "aiv_swiglu_tque_vector_to_mte3_completion"),
        }
        consumed_cells = set()
        for cell, result in evidence["published_cells"].items():
            worker_text, direction, payload_text = cell.split(":")
            worker_count = int(worker_text)
            payload_bytes = int(payload_text)
            role, service_prefix = direction_contract[direction]
            services = hw.resource_by_role(role).exact_request_services
            if worker_count == 1:
                assert all(
                    not name.startswith(service_prefix + "_1_")
                    for name in services)
                continue
            service_name = f"{service_prefix}_{worker_count}_workers"
            service = services[service_name]
            assert service["request_size_unit"] == "payload_bytes"
            assert service["service_time_unit"] == "ns_per_request_per_aiv"
            assert service["lookup_policy"] == "exact_only"
            if worker_count == 24:
                assert service["evidence_ref"] == str(evidence_path)
                assert service["evidence_sha256"] == evidence_sha256
            else:
                assert worker_count == 48
            assert service[
                "production_or_evaluation_duration_consumed"] is False
            assert service["service_time_s_by_request_size"][
                payload_bytes] == pytest.approx(
                    result["service_time_ns"] * 1e-9)
            consumed_cells.add(cell)

        assert consumed_cells == {
            cell for cell in evidence["published_cells"]
            if not cell.startswith("1:")
        }


    def test_scalar_get_value_service_is_hierarchy_insensitive(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        vector = hw.resource_by_role("compute_vector")
        service = vector.exact_request_services[
            "scalar_global_get_value_serialized"]

        assert service["request_size_unit"] == "int32_values"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_path = Path(service["evidence_ref"])
        evidence_bytes = evidence_path.read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["mechanism_publishable"] is True
        assert evidence["scope"] == (
            "hierarchy-insensitive serialized scalar global read")
        assert vector.exact_request_service_time_s(
            "scalar_global_get_value_serialized", 1
        ) == pytest.approx(92.0439453125e-9)

    def test_aic_scalar_get_value_service_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        scalar = hw.resource_by_role("compute_fma")
        service = scalar.exact_request_services[
            "aic_scalar_global_get_value_serialized"]

        assert service["request_size_unit"] == "int32_values"
        assert service["service_time_unit"] == "ns_per_request_per_aic"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_path = Path(service["evidence_ref"])
        evidence_bytes = evidence_path.read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_source_semantics"]["qk_count"] == (
            "one GetValue per 128-token GM-to-L1 K page")
        assert scalar.exact_request_service_time_s(
            "aic_scalar_global_get_value_serialized", 1
        ) == pytest.approx(93.31694666544601e-9)


    @pytest.mark.parametrize(
        (
            "resource_role", "service_name", "request_size", "expected_ns",
            "evidence_gate",
        ),
        (
            (
                "mte2", "aiv_mte2_request_completion", 1024, 210.0,
                "unloaded_mechanism_gate_passed",
            ),
            (
                "mte3", "aiv_mte3_request_completion", 1024, 260.0,
                "unloaded_mechanism_gate_passed",
            ),
            (
                "mte2", "aiv_datacopypad_mte2_to_vector_completion",
                4096, 440.0,
                "all_per_run_unloaded_replication_gates_passed",
            ),
            (
                "mte3", "aiv_datacopypad_vector_to_mte3_completion",
                4096, 360.0,
                "all_per_run_unloaded_replication_gates_passed",
            ),
            (
                "mte2",
                "aiv_datacopypad_mte2_to_vector_completion_two_workers",
                4096, 450.0,
                "all_per_run_two_worker_replication_gates_passed",
            ),
            (
                "mte2",
                "aiv_datacopypad_mte2_to_vector_completion_32_workers",
                4096, 430.0,
                "all_per_run_replication_gates_passed",
            ),
            (
                "mte2",
                "aiv_add_rmsnorm_single_n_input_pair_completion_one_worker",
                4096, 280.0, "cross_run_stability_gate_passed",
            ),
            (
                "mte2",
                "aiv_datacopypad_mte2_to_vector_completion_43_workers",
                4096, 780.0,
                "all_per_run_replication_gates_passed",
            ),
            (
                "mte2",
                "aiv_datacopypad_mte2_to_vector_completion_47_workers",
                4096, 890.0,
                "all_per_run_replication_gates_passed",
            ),
            (
                "mte3",
                "aiv_datacopypad_vector_to_mte3_completion_two_workers",
                4096, 340.0,
                "all_per_run_two_worker_replication_gates_passed",
            ),
            (
                "mte3",
                "aiv_rmsnorm_key0_h2048_first_mte3_completion_32_workers",
                4096, 3590.0, "mechanism_gate",
            ),
            (
                "mte3",
                "aiv_rmsnorm_key0_h5120_first_mte3_completion_32_workers",
                10240, 2570.0, "mechanism_gate",
            ),
            (
                "mte3",
                "aiv_add_rmsnorm_key30_terminal_output_sequence_32_workers",
                4096, 1440.0, "cross_run_stability_gate_passed",
            ),
            (
                "mte3",
                "aiv_datacopypad_vector_to_mte3_completion_32_workers",
                5120, 240.0,
                "all_per_run_replication_gates_passed",
            ),
            (
                "mte3",
                "aiv_datacopypad_vector_to_mte3_completion_43_workers",
                4096, 700.0,
                "all_per_run_replication_gates_passed",
            ),
            (
                "mte3",
                "aiv_datacopypad_vector_to_mte3_completion_47_workers",
                4096, 880.0,
                "all_per_run_replication_gates_passed",
            ),
        ),
    )
    def test_aiv_memory_completion_service_is_exact_and_source_bound(
            self, resource_role, service_name, request_size, expected_ns,
            evidence_gate):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        resource = hw.resource_by_role(resource_role)
        service = resource.exact_request_services[service_name]

        assert resource.type == "memory"
        assert service["request_size_unit"] == "payload_bytes"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence[evidence_gate] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        assert resource.exact_request_service_time_s(
            service_name, request_size) == pytest.approx(expected_ns * 1e-9)

    def test_unstable_32_worker_mte3_cell_is_not_published(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mte3 = hw.resource_by_role("mte3")

        with pytest.raises(ValueError, match="has no request size 4096"):
            mte3.exact_request_service_time_s(
                "aiv_datacopypad_vector_to_mte3_completion_32_workers",
                4096,
            )

    @pytest.mark.parametrize(
        "service_name,expected",
        (
            (
                "aiv_reshape_cache_equal_chain_completion",
                {1024: 780.0, 2048: 780.0},
            ),
            (
                "aiv_reshape_cache_equal_chain_completion_48_workers",
                {1024: 1060.0, 2048: 1810.0},
            ),
        ),
    )
    def test_reshape_cache_equal_chain_services_are_exact_and_source_bound(
            self, service_name, expected):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mte3 = hw.resource_by_role("mte3")
        service = mte3.exact_request_services[service_name]

        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["service"] == service_name
        assert evidence["worker_count"] in (1, 48)
        assert evidence["production_or_evaluation_duration_consumed"] is False
        source_bytes = Path(evidence["source_reference"]).read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == (
            evidence["source_reference_sha256"])
        for payload_bytes, expected_ns in expected.items():
            assert mte3.exact_request_service_time_s(
                service_name, payload_bytes) == pytest.approx(
                    expected_ns * 1e-9)
        with pytest.raises(ValueError, match="has no request size 512"):
            mte3.exact_request_service_time_s(service_name, 512)

    @pytest.mark.parametrize(
        ("payload_bytes", "expected"),
        (
            (1024, {1: 3120.0, 2: 3760.0, 6: 7250.0, 22: 19650.0}),
            (2048, {1: 3130.0, 2: 4370.0, 6: 7330.0, 22: 21040.0}),
        ),
    )
    def test_reshape_cache_equal_repeated_chain_service_is_exact(
            self, payload_bytes, expected):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mte3 = hw.resource_by_role("mte3")
        service_name = (
            "aiv_reshape_cache_equal_repeated_chain_completion_48_workers_"
            f"{payload_bytes}b")
        service = mte3.exact_request_services[service_name]

        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["service"] == service_name
        assert evidence["request_contract"]["worker_count"] == 48
        assert evidence["request_contract"]["payload_bytes"] == payload_bytes
        assert evidence["stability_gate"]["passed"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        source = evidence["source_chain"]
        source_bytes = Path(source["reference"]).read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == (
            source["reference_sha256"])
        for repeat_count, expected_ns in expected.items():
            assert mte3.exact_request_service_time_s(
                service_name, repeat_count) == pytest.approx(
                    expected_ns * 1e-9)
        with pytest.raises(ValueError, match="has no request size 3"):
            mte3.exact_request_service_time_s(service_name, 3)

    def test_reshape_cache_unequal_repeated_pair_service_is_exact(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mte3 = hw.resource_by_role("mte3")
        service_name = (
            "aiv_reshape_cache_unequal_repeated_pair_completion_48_workers_"
            "2048b")
        service = mte3.exact_request_services[service_name]

        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["service"] == service_name
        assert evidence["request_contract"]["worker_count"] == 48
        assert evidence["request_contract"]["payload_bytes"] == 2048
        assert evidence["request_contract"]["repeat_counts"] == [22]
        assert evidence["stability_gate"]["passed"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        source = evidence["source_chain"]
        source_bytes = Path(source["reference"]).read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == (
            source["reference_sha256"])
        assert mte3.exact_request_service_time_s(
            service_name, 22) == pytest.approx(7070.0e-9)
        with pytest.raises(ValueError, match="has no request size 21"):
            mte3.exact_request_service_time_s(service_name, 21)

    @pytest.mark.parametrize(
        "worker_count,expected",
        (
            (2, {1024: 380.0, 2048: 420.0}),
            (4, {1024: 400.0, 2048: 440.0}),
            (8, {1024: 420.0, 2048: 440.0}),
            (16, {1024: 440.0, 2048: 460.0}),
        ),
    )
    def test_reshape_cache_equal_warm_chain_services_are_exact(
            self, worker_count, expected):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mte3 = hw.resource_by_role("mte3")
        service_name = (
            "aiv_reshape_cache_equal_warm_chain_completion_"
            f"{worker_count}_workers")
        service = mte3.exact_request_services[service_name]

        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["production_or_evaluation_duration_consumed"] is False
        assert evidence["worker_count"] == worker_count
        source_bytes = Path(evidence["source_reference"]).read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == (
            evidence["source_reference_sha256"])
        source = json.loads(source_bytes)
        assert source["decision"]["forward_eligible"] is True
        assert source["gates"]["mechanism_gate"] == "pass"
        for payload_bytes, expected_ns in expected.items():
            assert mte3.exact_request_service_time_s(
                service_name, payload_bytes) == pytest.approx(
                    expected_ns * 1e-9)


    def test_rope_key24_grid47_steady_mte2_service_is_exact(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service_name = (
            "aiv_rope_key24_mte2_steady_completion_47_workers_4096b")
        service = hw.resource_by_role("mte2").exact_request_services[
            service_name]
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()

        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["stability_gate"] == "pass"
        assert evidence["production_or_evaluation_duration_consumed"] is False
        assert hw.resource_by_role("mte2").exact_request_service_time_s(
            service_name, 20) == pytest.approx(121.0e-9)
        assert hw.resource_by_role("mte2").exact_request_service_time_s(
            service_name, 21) == pytest.approx(
                123.80952380952381e-9)
        with pytest.raises(ValueError, match="has no request size 19"):
            hw.resource_by_role("mte2").exact_request_service_time_s(
                service_name, 19)

    def test_rope_key22_first_q_slice_services_are_exact_and_auditable(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        evidence_path = Path(
            "microbench/ascend_910b/references/"
            "aiv_rope_key22_first_q_slice_mte_edge_completion_cann_8_5_1.json")
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
        evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
        expected = {
            (
                "mte2",
                "aiv_rope_key22_first_q_slice_mte2_to_vector_completion",
            ): ("1:mte2_to_vector:8192", 420.0),
            (
                "mte2",
                "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_16_workers",
            ): ("16:mte2_to_vector:8192", 900.0),
            (
                "mte3",
                "aiv_rope_key22_first_q_slice_vector_to_mte3_completion",
            ): ("1:mte3_to_vector:8192", 360.0),
            (
                "mte3",
                "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_16_workers",
            ): ("16:mte3_to_vector:8192", 880.0),
        }

        assert evidence["source_tile_evidence"]["payload_bytes"] == 8192
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        for (resource_role, service_name), (cell, expected_ns) in (
                expected.items()):
            service = hw.resource_by_role(
                resource_role).exact_request_services[service_name]
            assert service["evidence_ref"] == str(evidence_path)
            assert service["evidence_sha256"] == evidence_sha256
            assert service["service_time_s_by_request_size"] == {
                8192: pytest.approx(expected_ns * 1e-9)}
            assert evidence["published_cells"][cell][
                "service_time_ns"] == expected_ns

    def test_rope_key22_service_entry_must_match_retained_evidence(
            self, tmp_path):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        stack = list(config["components"])
        mte2 = None
        while stack:
            component = stack.pop()
            if component["name"] == "mte2_pipe":
                mte2 = component
                break
            if "children" in component:
                stack.extend(component["children"])
        assert mte2 is not None
        service = mte2["exact_request_services"][
            "aiv_rope_key22_first_q_slice_mte2_to_vector_completion"]
        service["entries"][0]["service_time_ns"] = 421.0
        path = tmp_path / "invalid_key22_rope_service.yaml"
        path.write_text(yaml.safe_dump(config))

        with pytest.raises(ValueError, match="inline entries disagree"):
            HWSpec(str(path))

    @pytest.mark.parametrize(
        (
            "resource_role", "service_name", "evidence_name", "expected_ns",
            "campaign",
        ),
        (
            (
                "mte2",
                "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_4_workers",
                "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_4_workers_cann_8_5_1.json",
                540.0,
                "rope_key22_active4_replication_edges",
            ),
            (
                "mte3",
                "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_4_workers",
                "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_4_workers_cann_8_5_1.json",
                350.0,
                "rope_key22_active4_replication_edges",
            ),
            (
                "mte2",
                "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_8_workers",
                "aiv_rope_key22_first_q_slice_mte2_to_vector_completion_8_workers_cann_8_5_1.json",
                670.0,
                "rope_key22_active8_replication_edges",
            ),
            (
                "mte3",
                "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_8_workers",
                "aiv_rope_key22_first_q_slice_vector_to_mte3_completion_8_workers_cann_8_5_1.json",
                510.0,
                "rope_key22_active8_replication_edges",
            ),
        ),
    )
    def test_rope_key22_replicated_first_q_slice_services_are_exact(
            self, resource_role, service_name, evidence_name, expected_ns,
            campaign):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        evidence_path = Path(
            "microbench/ascend_910b/references") / evidence_name
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
        service = hw.resource_by_role(
            resource_role).exact_request_services[service_name]

        assert service["evidence_ref"] == str(evidence_path)
        assert service["evidence_sha256"] == hashlib.sha256(
            evidence_bytes).hexdigest()
        assert service["service_time_s_by_request_size"] == {
            8192: pytest.approx(expected_ns * 1e-9)}
        assert evidence["campaign"] == campaign
        assert evidence["measurement_gate_passed"] is True
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False

    def test_rmsnorm_local_scalar_service_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        vector = hw.resource_by_role("compute_vector")
        service = vector.exact_request_services[
            "rmsnorm_local_scalar_dependency"]

        assert service["request_size_unit"] == (
            "lowered_scalar_operations_per_chain")
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["measurement_gate_passed"] is True
        assert evidence["provenance_gate_passed"] is True
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        assert vector.exact_request_service_time_s(
            "rmsnorm_local_scalar_dependency", 2
        ) == pytest.approx(50.0e-9)

    @pytest.mark.parametrize(
        ("resource_role", "service_name", "expected_ns"),
        (
            (
                "mte2",
                "aiv_rmsnorm_key0_h2048_mte2_to_vector_completion_4_workers",
                650.0,
            ),
            (
                "mte2",
                "aiv_rmsnorm_key0_h2048_mte2_to_vector_completion_8_workers",
                840.0,
            ),
            (
                "mte3",
                "aiv_rmsnorm_key0_h2048_vector_to_mte3_completion_4_workers",
                560.0,
            ),
            (
                "mte3",
                "aiv_rmsnorm_key0_h2048_vector_to_mte3_completion_8_workers",
                720.0,
            ),
        ),
    )
    def test_rmsnorm_key0_h2048_mte_edge_service_is_source_bound(
            self, resource_role, service_name, expected_ns):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        resource = hw.resource_by_role(resource_role)
        service = resource.exact_request_services[service_name]

        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["mechanism_publishable"] is True
        assert evidence[
            "production_or_evaluation_duration_consumed"] is False
        assert resource.exact_request_service_time_s(
            service_name, 4096) == pytest.approx(expected_ns * 1e-9)

    def test_rmsnorm_key0_gamma_input_sequence_service_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mte2 = hw.resource_by_role("mte2")
        service_name = (
            "aiv_rmsnorm_key0_gamma_input_sequence_completion")
        service = mte2.exact_request_services[service_name]

        assert service["request_size_unit"] == (
            "payload_bytes_per_source_tensor")
        assert service["service_time_unit"] == "ns_per_request_per_aiv"
        assert service["lookup_policy"] == "exact_only"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["independent_runs"] == 3
        assert evidence["samples_per_cell_per_run"] == 120
        assert evidence["cross_run_stability_gate_passed"] is True
        assert evidence["mechanism_publishable"] is True
        assert evidence[
            "production_or_evaluation_duration_consumed"] is False
        assert {
            request_size: mte2.exact_request_service_time_s(
                service_name, request_size) * 1e9
            for request_size in (4096, 5120, 8192, 10240, 16384)
        } == pytest.approx({
            4096: 620.0,
            5120: 730.0,
            8192: 710.0,
            10240: 770.0,
            16384: 880.0,
        })

    def test_rmsnorm_half_interval_reduction_service_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        vector = hw.resource_by_role("compute_vector")

        assert vector.service_rates_per_unit_s[
            "rmsnorm_half_interval_reduce"
        ] == pytest.approx(43849883505.65282)
        assert vector.service_setup_time_s[
            "rmsnorm_half_interval_reduce"
        ] == pytest.approx(8.694661458349991e-08)
        evidence = json.loads(Path(
            "microbench/ascend_910b/references/"
            "rmsnorm_half_interval_reduce_cann_8_5_1.json"
        ).read_text())
        assert evidence["mechanism_publishable"] is True
        assert evidence["maximum_validation_ape_percent"] < 10.0
        assert evidence[
            "production_or_evaluation_duration_consumed"] is False

    def test_add_rmsnorm_reduce_sum_fp32_service_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        vector = hw.resource_by_role("compute_vector")

        assert vector.service_rates_per_unit_s[
            "add_rmsnorm_reduce_sum_fp32"
        ] == pytest.approx(28800251118.261005)
        assert vector.service_setup_time_s[
            "add_rmsnorm_reduce_sum_fp32"
        ] == pytest.approx(4.611165364599998e-08)
        evidence = json.loads(Path(
            "microbench/ascend_910b/references/"
            "add_rmsnorm_reduce_sum_fp32_cann_8_5_1.json"
        ).read_text())
        assert evidence["mechanism_publishable"] is True
        assert evidence["maximum_validation_ape_percent"] < 0.01
        assert evidence[
            "production_or_evaluation_duration_consumed"] is False

    def test_add_rmsnorm_row_group_input_pair_service_is_exact(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mte2 = hw.resource_by_role("mte2")
        service_name = (
            "aiv_add_rmsnorm_row_group_input_pair_completion_32_workers")
        service = mte2.exact_request_services[service_name]

        assert service["request_size_unit"] == "payload_bytes"
        assert service["service_time_unit"] == "ns_per_request_per_aiv"
        assert service["lookup_policy"] == "exact_only"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["source_domain"] == {
            "cann_tiling_key": 30,
            "active_aiv_workers": 32,
            "dtype": "bf16",
            "requests_per_row": 2,
            "exact_hidden_payload_cells": [
                {"hidden_elements": 2048,
                 "payload_bytes_per_equal_input_request": 4096},
                {"hidden_elements": 2560,
                 "payload_bytes_per_equal_input_request": 5120},
            ],
        }
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        assert mte2.exact_request_service_time_s(
            service_name, 4096) == pytest.approx(550.0e-9)
        assert mte2.exact_request_service_time_s(
            service_name, 5120) == pytest.approx(950.0e-9)

    def test_add_rmsnorm_key31_residual_chain_service_is_exact(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mte2 = hw.resource_by_role("mte2")
        service_name = (
            "add_rmsnorm_key31_residual_reload_completion_chain")
        service = mte2.exact_request_services[service_name]

        assert service["request_size_unit"] == (
            "source_split_d_residual_reload_chains")
        assert service["service_time_unit"] == (
            "ns_per_maximum_worker_chain")
        assert service["lookup_policy"] == "exact_only"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["service"] == service_name
        assert evidence["composition"] == (
            "maximum_with_same_source_stage_mte2_service")
        assert evidence["measurement_gate_passed"] is True
        assert evidence["provenance_gate_passed"] is True
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        expected_ns = {2: 330.0, 4: 840.0, 12: 2760.0, 44: 9200.0}
        assert service["service_time_s_by_request_size"] == {
            request_size: pytest.approx(service_time_ns * 1e-9)
            for request_size, service_time_ns in expected_ns.items()
        }
        assert {
            int(request_size): service_time_ns
            for request_size, service_time_ns
            in evidence["service_time_ns_by_request_size"].items()
        } == expected_ns

    def test_add_rmsnorm_key31_scalar_handoff_service_is_exact(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        scalar = hw.resource_by_role("scalar_aiv")
        service_name = "add_rmsnorm_split_d_local_scalar_handoff"
        service = scalar.exact_request_services[service_name]

        assert service["request_size_unit"] == (
            "source_split_d_scalar_handoff_chains")
        assert service["service_time_unit"] == "ns_per_request_per_aiv"
        assert service["lookup_policy"] == "exact_only"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["service"] == service_name
        assert evidence["composition"] == (
            "one source-exact request per CANN key-31 AddRmsNorm row and "
            "8192-element output column tile, after the residual reload "
            "cast and before output scaling")
        assert evidence["measurement_gate_passed"] is True
        assert evidence["provenance_gate_passed"] is True
        assert evidence["mechanism_publishable"] is True
        assert evidence["production_or_evaluation_duration_consumed"] is False
        assert scalar.exact_request_service_time_s(
            service_name, 1) == pytest.approx(17.22249349e-9)

    def test_local_scalar_fill_service_is_exact_and_duration_independent(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        vector = hw.resource_by_role("compute_vector")
        service = vector.exact_request_services["local_scalar_fill_s_v"]

        assert service["request_size_unit"] == "fp32_scalar_writes"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["mechanism_publishable"] is True
        assert vector.exact_request_service_time_s(
            "local_scalar_fill_s_v", 64
        ) == pytest.approx(291.110839844e-9)
        assert vector.exact_request_service_time_s(
            "local_scalar_fill_s_v", 128
        ) == pytest.approx(569.444173177e-9)

    @pytest.mark.parametrize(
        ("service_name", "request_size", "expected_ns"),
        (
            ("local_ub_copy_contiguous", 128, 10.0),
            ("local_ub_copy_contiguous", 512, 10.555826823),
            ("local_ub_copy_stride_64b", 64, 10.0),
            ("local_ub_copy_stride_64b", 4096, 45.0),
            ("local_ub_copy_stride_128b", 1024, 13.889160156),
            ("local_ub_copy_stride_128b", 4096, 27.639160156),
        ),
    )
    def test_local_ub_copy_service_is_exact_and_duration_independent(
            self, service_name, request_size, expected_ns):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        vector = hw.resource_by_role("compute_vector")
        service = vector.exact_request_services[service_name]

        assert service["request_size_unit"] == "payload_bytes"
        assert service["production_or_evaluation_duration_consumed"] is False
        evidence_bytes = Path(service["evidence_ref"]).read_bytes()
        assert hashlib.sha256(evidence_bytes).hexdigest() == (
            service["evidence_sha256"])
        evidence = json.loads(evidence_bytes)
        assert evidence["mechanism_publishable"] is True
        assert vector.exact_request_service_time_s(
            service_name, request_size
        ) == pytest.approx(expected_ns * 1e-9)

    def test_exact_vector_request_service_rejects_duration_contamination(
            self, tmp_path):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        aiv = next(
            component for component in config["components"]
            if component["name"] == "aiv_core")
        subcore = next(
            child for child in aiv["children"]
            if child["name"] == "subcore")
        vector = next(
            child for child in subcore["children"]
            if child["name"] == "vector_pipe")
        vector["exact_request_services"][
            "norm_sqrt_duplicate_div_serialized"][
                "production_or_evaluation_duration_consumed"] = True
        config_path = tmp_path / "invalid_exact_request_service.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)

        with pytest.raises(
                ValueError, match="cannot consume production or evaluation"):
            HWSpec(config_path)

    @pytest.mark.parametrize(
        "field,value,error",
        (
            (
                "evidence_sha256",
                "0" * 64,
                "evidence SHA256 mismatch",
            ),
            (
                "evidence_ref",
                "/tmp/vector_norm_request_cann_8_5_1.json",
                "evidence_ref must be repository-relative",
            ),
        ),
    )
    def test_exact_vector_request_service_rejects_unretained_evidence(
            self, tmp_path, field, value, error):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        aiv = next(
            component for component in config["components"]
            if component["name"] == "aiv_core")
        subcore = next(
            child for child in aiv["children"]
            if child["name"] == "subcore")
        vector = next(
            child for child in subcore["children"]
            if child["name"] == "vector_pipe")
        service = vector["exact_request_services"][
            "norm_sqrt_duplicate_div_serialized"]
        service[field] = value
        config_path = tmp_path / "invalid_exact_request_evidence.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)

        with pytest.raises(ValueError, match=error):
            HWSpec(config_path)

    def test_exact_vector_request_service_rejects_inline_evidence_mismatch(
            self, tmp_path):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        aiv = next(
            component for component in config["components"]
            if component["name"] == "aiv_core")
        subcore = next(
            child for child in aiv["children"]
            if child["name"] == "subcore")
        vector = next(
            child for child in subcore["children"]
            if child["name"] == "vector_pipe")
        service = vector["exact_request_services"][
            "norm_sqrt_duplicate_div_serialized"]
        service["entries"][0]["service_time_ns"] += 1.0
        config_path = tmp_path / "invalid_exact_request_entries.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)

        with pytest.raises(ValueError, match="inline entries disagree"):
            HWSpec(config_path)

    def test_typed_vector_service_rates_are_per_physical_unit_si_rates(
            self, tmp_path):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        aiv = next(
            component for component in config["components"]
            if component["name"] == "aiv_core")
        subcore = next(
            child for child in aiv["children"]
            if child["name"] == "subcore")
        vector = next(
            child for child in subcore["children"]
            if child["name"] == "vector_pipe")
        vector["service_rates_per_unit_s"] = {
            "generic": 10e9,
            "exp": 2e9,
            "div": 1e9,
        }
        vector["service_setup_time_s"] = {
            "exp": 17e-9,
            "div": 19e-9,
        }
        config_path = tmp_path / "typed_ascend_910b.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)

        hw = HWSpec(config_path)
        resource = hw.resource_by_role("compute_vector")

        assert resource.service_rates_per_unit_s == {
            "generic": 10e9,
            "exp": 2e9,
            "div": 1e9,
        }
        assert resource.service_setup_time_s == {
            "exp": 17e-9,
            "div": 19e-9,
        }
        assert resource.effective_service_capacity(
            "exp", active_units=2, execution_domain_units=48
        ) == 4e9
        assert resource.effective_service_capacity(
            "exp", active_units=96, execution_domain_units=48
        ) == 96e9
        assert resource.setup_time_for_service("exp") == 17e-9

    def test_typed_setup_requires_a_matching_service_rate(self, tmp_path):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        aiv = next(
            component for component in config["components"]
            if component["name"] == "aiv_core")
        subcore = next(
            child for child in aiv["children"]
            if child["name"] == "subcore")
        vector = next(
            child for child in subcore["children"]
            if child["name"] == "vector_pipe")
        vector["service_setup_time_s"] = {"missing": 1e-9}
        config_path = tmp_path / "invalid_typed_setup.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)

        with pytest.raises(ValueError, match="has no matching rate"):
            HWSpec(config_path)

    def test_memory_component_rejects_typed_compute_service_rates(
            self, tmp_path):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        dram = next(
            component for component in config["components"]
            if component["name"] == "dram")
        dram["service_rates_per_unit_s"] = {"invalid": 1.0}
        config_path = tmp_path / "invalid_typed_memory.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)

        with pytest.raises(ValueError, match="non-compute component"):
            HWSpec(config_path)

    def test_non_vector_component_rejects_typed_service_rates(
            self, tmp_path):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        aic = next(
            component for component in config["components"]
            if component["name"] == "sm")
        subcore = next(
            child for child in aic["children"]
            if child["name"] == "subcore")
        scalar = next(
            child for child in subcore["children"]
            if child["name"] == "fma_pipe")
        scalar["service_rates_per_unit_s"] = {"generic": 1.0}
        config_path = tmp_path / "invalid_typed_scalar.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)

        with pytest.raises(ValueError, match="registered typed compute role"):
            HWSpec(config_path)

    def test_typed_service_rate_rejects_empirical_labels(self, tmp_path):
        with open("configs/hardware/ascend_910b.yaml") as stream:
            config = yaml.safe_load(stream)
        aiv = next(
            component for component in config["components"]
            if component["name"] == "aiv_core")
        subcore = next(
            child for child in aiv["children"]
            if child["name"] == "subcore")
        vector = next(
            child for child in subcore["children"]
            if child["name"] == "vector_pipe")
        vector["service_rates_per_unit_s"] = {
            "activation_shape_1024": 1.0}
        config_path = tmp_path / "invalid_typed_label.yaml"
        with config_path.open("w") as stream:
            yaml.safe_dump(config, stream)

        with pytest.raises(ValueError, match="typed service must be one"):
            HWSpec(config_path)

    def test_execution_domain_count_follows_explicit_task_kind(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.execution_units_for_task("aic") == 24
        assert hw.execution_units_for_task("mixed") == 24
        assert hw.execution_units_for_task("aiv") == 48
        with pytest.raises(ValueError, match="generic execution domain"):
            hw.execution_units_for_task("generic")

    def test_non_cache_buffer_preserves_declared_capacity(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.resource_by_role("ub").size_bytes == 196608
        assert hw.resource_by_role("l0a_buffer").size_bytes == 65536

    def test_vector_reduce_sum_lut_uses_mechanism_curve_and_bounded_interpolation(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        points, durations = hw.vector_reduce_sum_latency_lut
        assert points == tuple(2 ** exponent for exponent in range(5, 15))
        assert durations[6] == pytest.approx(95.044)
        assert hw.vector_reduce_sum_latency_ns(2560) == pytest.approx(108.3765)
        assert hw.vector_reduce_sum_latency_ns(5120) == pytest.approx(175.03925)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.vector_reduce_sum_latency_ns(16)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.vector_reduce_sum_latency_ns(32768)

    def test_exact_empty_task_luts_cover_all_physical_aic_and_aiv_cores(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.dispatch_overhead_ns == 0
        assert hw.aic_empty_kernel_duration_ns(1) == 520
        assert hw.aic_empty_kernel_duration_ns(24) == 2220
        assert hw.aiv_empty_kernel_duration_ns(1) == 520
        assert hw.aiv_empty_kernel_duration_ns(48) == 4700
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aic_empty_kernel_duration_ns(25)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aiv_empty_kernel_duration_ns(49)

    def test_nonempty_aiv_task_model_is_a_bounded_complete_max_floor(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        model = hw.aiv_nonempty_task_latency_model
        assert model["model"] == "isolated_nonempty_aiv_task_scheduler_v1"
        assert model["timer_domain"] == "msprof_task_time_complete_task"
        assert model["minimum_aiv_block_dim"] == 1
        assert model["maximum_aiv_block_dim"] == 48
        assert hw.aiv_nonempty_task_duration_ns(1) == pytest.approx(
            973.4693877551022)
        assert hw.aiv_nonempty_task_duration_ns(48) == pytest.approx(
            5166.530612244897)
        assert hw.aiv_worker_start_fill_model["model"] == (
            "isolated_aiv_worker_start_fill_v1")
        worker_start_evidence = Path(
            hw.aiv_worker_start_fill_model["evidence_ref"]).read_bytes()
        assert hashlib.sha256(worker_start_evidence).hexdigest() == (
            hw.aiv_worker_start_fill_model["evidence_sha256"])
        assert hw.aiv_worker_start_fill_ns(1) == 0.0
        assert hw.aiv_worker_start_fill_ns(43) == 1680.0
        assert hw.aiv_worker_start_fill_ns(48) == 1880.0
        assert hw.task_duration_floor_for_task(
            "aiv", 8, 1) == pytest.approx(1627.6295411781734)

        tiling_model = hw.aiv_tiling_materialization_latency_model
        assert tiling_model["model"] == (
            "shared_aiv_tiling_materialization_affine_v1")
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_v1", 1) == pytest.approx(
                102.53477542795081)
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_v1", 48) == pytest.approx(
                1526.0146022221856)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.task_prologue_duration_ns(
                "cann_aiv_tiling_materialization_v1", 49)
        rope_tiling_model = hw.aiv_tiling_materialization_48b_latency_model
        assert rope_tiling_model["model"] == (
            "rope_aiv_tiling_materialization_complete_max_v1")
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_48b_v1", 1) == pytest.approx(
                230.38614908854174)
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_48b_v1", 48) == pytest.approx(
                1808.960367838541)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.task_prologue_duration_ns(
                "cann_aiv_tiling_materialization_48b_v1", 49)
        rope_tiling_evidence_bytes = Path(
            rope_tiling_model["reference_path"]).read_bytes()
        assert hashlib.sha256(rope_tiling_evidence_bytes).hexdigest() == (
            rope_tiling_model["reference_sha256"])

        rope_tiling_evidence = json.loads(rope_tiling_evidence_bytes)
        assert rope_tiling_evidence["model"] == rope_tiling_model["model"]
        assert rope_tiling_evidence["parameters"] == {
            "serialized_ns_per_aiv":
                rope_tiling_model["serialized_ns_per_aiv"],
            "single_worker_completion_ns":
                rope_tiling_model["single_worker_completion_ns"],
        }
        assert rope_tiling_evidence["measurement_gate_passed"] is True
        assert rope_tiling_evidence[
            "production_or_evaluation_duration_consumed"] is False
        broadcast_tiling_model = (
            hw.aiv_tiling_materialization_56b_latency_model)
        assert broadcast_tiling_model["model"] == (
            "broadcast_one_dim_tiling_materialization_complete_max_v1")
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_56b_v1", 1) == pytest.approx(
                120.5314127604165)
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_56b_v1", 48) == pytest.approx(
                1809.5782470703125)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.task_prologue_duration_ns(
                "cann_aiv_tiling_materialization_56b_v1", 49)
        broadcast_tiling_evidence_bytes = Path(
            broadcast_tiling_model["reference_path"]).read_bytes()
        assert hashlib.sha256(broadcast_tiling_evidence_bytes).hexdigest() == (
            broadcast_tiling_model["reference_sha256"])
        broadcast_tiling_evidence = json.loads(broadcast_tiling_evidence_bytes)
        assert broadcast_tiling_evidence["parameters"] == {
            "serialized_ns_per_aiv": broadcast_tiling_model[
                "serialized_ns_per_aiv"],
            "single_worker_completion_ns": broadcast_tiling_model[
                "single_worker_completion_ns"],
        }
        assert broadcast_tiling_evidence[
            "production_or_evaluation_duration_consumed"] is False
        gather_tiling_model = (
            hw.aiv_tiling_materialization_192b_latency_model)
        assert gather_tiling_model["model"] == (
            "gather_v3_aiv_tiling_materialization_complete_max_v1")
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_192b_v1", 1) == pytest.approx(
                160.42968750000023)
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_192b_v1", 48) == pytest.approx(
                3080.14892578125)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.task_prologue_duration_ns(
                "cann_aiv_tiling_materialization_192b_v1", 49)
        gather_tiling_evidence_bytes = Path(
            gather_tiling_model["reference_path"]).read_bytes()
        assert hashlib.sha256(gather_tiling_evidence_bytes).hexdigest() == (
            gather_tiling_model["reference_sha256"])
        gather_tiling_evidence = json.loads(gather_tiling_evidence_bytes)
        assert gather_tiling_evidence["parameters"] == {
            "serialized_ns_per_aiv":
                gather_tiling_model["serialized_ns_per_aiv"],
            "single_worker_completion_ns":
                gather_tiling_model["single_worker_completion_ns"],
        }
        assert gather_tiling_evidence[
            "production_or_evaluation_duration_consumed"] is False
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_80b_v1", 1) == pytest.approx(
                104.18552310431747)
        assert hw.task_prologue_duration_ns(
            "cann_aiv_tiling_materialization_80b_v1", 48) == pytest.approx(
                1525.9614708178528)
        assert hw.task_duration_floor_for_task(
            "aiv", 48, 1) == pytest.approx(5285.901360544218)
        assert hw.task_duration_floor_for_task(
            "aiv", 48, 1,
            task_duration_floor_contract=(
                AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM),
        ) == pytest.approx(4700.0)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aiv_nonempty_task_duration_ns(49)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aiv_worker_start_fill_ns(49)

        memory_model = hw.aiv_memory_task_latency_model
        assert memory_model["model"] == "isolated_aiv_memory_task_floor_v1"
        assert hw.aiv_memory_task_duration_ns(1) == pytest.approx(
            987.4319727891159)
        assert hw.aiv_memory_task_duration_ns(48) == pytest.approx(
            5285.901360544218)
        assert hw.task_duration_floor_for_task(
            "aiv", 8, 1) == pytest.approx(1627.6295411781734)

    def test_nonempty_mix_aiv_task_model_is_a_distinct_complete_floor(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        model = hw.mix_aiv_nonempty_task_latency_model
        assert model["model"] == "isolated_nonempty_mix_aiv_task_scheduler_v1"
        assert model["duration_unit"] == (
            "ns_per_complete_nonempty_mix_aiv_task")
        assert model["minimum_aiv_block_dim"] == 1
        assert model["maximum_aiv_block_dim"] == 48
        assert hw.execution_units_for_task("mix_aiv") == 48
        assert hw.has_task_duration_floor_for_task("mix_aiv")
        assert hw.mix_aiv_nonempty_task_duration_ns(1) == pytest.approx(
            3120.195578231292)
        assert hw.task_duration_floor_for_task(
            "mix_aiv", 48, 1) == pytest.approx(7512.0960884353735)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.mix_aiv_nonempty_task_duration_ns(49)
        with pytest.raises(ValueError, match="does not support"):
            hw.task_duration_floor_for_task(
                "mix_aiv", 48, 1,
                task_duration_floor_contract="not-a-mix-aiv-contract")

    def test_c500_resident_cta_cohort_refill_uses_scheduler_services(self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        scheduler = hw.launch_latency_model

        assert hw.resident_cta_cohort_refill_ns(104, 512) == pytest.approx(
            104 * 4 * scheduler["cta_descriptor_ns"]
            + 4 * 8 * scheduler["multi_wave_round_ns"])

    def test_c500_binary_tensor_iterator_floor_replaces_generic_task_floor(
            self):
        hw = HWSpec("configs/hardware/metax_c500.yaml")
        task_class = PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS
        model = hw.runtime_task_latency_models[task_class]

        assert hw.runtime_task_class_floor_supports(task_class, 16384)
        assert not hw.runtime_task_class_floor_supports(task_class, 16385)
        assert hw.task_duration_floor_for_task(
            "generic", 16, 256, (16, 1, 1), (256, 1, 1),
            runtime_task_class=task_class,
            runtime_task_elements=16384,
        ) == pytest.approx(model["duration_ns"])
        assert model["duration_ns"] < hw.empty_kernel_duration_for_launch(
            16, 256, (16, 1, 1), (256, 1, 1))

    def test_a800_resident_cta_cohort_refill_uses_scheduler_rounds(self):
        hw = HWSpec("configs/hardware/nvidia_a800.yaml")
        scheduler = hw.launch_latency_model

        assert hw.resident_cta_cohort_refill_ns(108, 256) == pytest.approx(
            8 * max(
                scheduler["cta_round_ns"],
                8 * scheduler["wave_round_ns"],
            ))

    def test_aiv_memory_task_model_is_a_separate_complete_max_floor(
            self, tmp_path):
        import yaml

        with open("configs/hardware/ascend_910b.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["aiv_memory_task_latency_model"] = {
            "model": "isolated_aiv_memory_task_floor_v1",
            "duration_unit": "ns_per_complete_nonempty_aiv_task",
            "timer_domain": "msprof_task_time_complete_task",
            "software_revision": "cann_8.5.1",
            "minimum_aiv_block_dim": 1,
            "maximum_aiv_block_dim": 48,
            "submit_ns": 3000.0,
            "descriptor_ns_per_aiv": 50.0,
        }
        path = tmp_path / "aiv_memory_task_floor.yaml"
        path.write_text(yaml.safe_dump(cfg))

        hw = HWSpec(str(path))

        assert hw.aiv_memory_task_duration_ns(1) == pytest.approx(3050.0)
        assert hw.aiv_memory_task_duration_ns(48) == pytest.approx(5400.0)
        assert hw.task_duration_floor_for_task(
            "aiv", 8, 1) == pytest.approx(3400.0)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aiv_memory_task_duration_ns(49)

    def test_mixed_empty_task_lut_uses_complete_one_to_two_task_contract(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        points, aiv_ratio, durations = hw.mixed_empty_kernel_duration_lut
        assert points == tuple(range(1, 25))
        assert aiv_ratio == 2
        assert durations[0] == 3620
        assert hw.mixed_empty_kernel_duration_ns(24) == 9120
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.mixed_empty_kernel_duration_ns(25)

    def test_fia_cross_domain_program_span_uses_bounded_mechanism(self):
        import json

        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "fia_cross_domain_three_handoff_cann_8_5_1.json"))
        mechanism = hw.fia_cross_domain_three_handoff

        assert reference["production_or_evaluation_duration_consumed"] is False
        assert reference["mechanism_id"] == mechanism["model"]
        for name in (
                "minimum_task_count", "maximum_task_count",
                "minimum_active_aic_groups", "maximum_active_aic_groups",
                "aic_per_group", "aiv_per_group"):
            assert reference["bounds"][name] == mechanism[name]
        for name in (
                "startup_local_ns", "startup_shared_ns_per_active_group",
                "steady_local_ns", "steady_shared_ns_per_active_group"):
            assert reference["parameters_ns"][name] == mechanism[name]
        assert tuple(reference["single_task_duration_lut_ns"][
            "active_aic_groups"]) == mechanism[
                "published_single_task_active_aic_groups"]
        assert tuple(reference["single_task_duration_lut_ns"][
            "duration_ns"]) == mechanism[
                "published_single_task_duration_ns"]
        for name in (
                "source_commit", "source_snapshot_sha256", "binary_sha256",
                "report_path", "report_sha256"):
            assert reference["provenance"][name] == mechanism[name]
        assert tuple(reference["supplemental_single_task_evidence"]) == (
            mechanism["supplemental_single_task_evidence"])
        assert tuple(reference["extended_single_task_points"]) == (
            mechanism["extended_single_task_points"])
        # The consumed domain is the union of the two independently
        # provenanced campaigns, and every extension point carries its own
        # report digest rather than borrowing the published one.
        extension = {
            entry["active_aic_groups"]: entry["duration_ns"]
            for entry in mechanism["extended_single_task_points"]}
        assert set(extension).isdisjoint(
            mechanism["published_single_task_active_aic_groups"])
        for entry in mechanism["extended_single_task_points"]:
            assert entry["report_sha256"] != mechanism["report_sha256"]
            assert entry["production_or_evaluation_duration_consumed"] is (
                False)
        assert mechanism["single_task_active_aic_groups"] == tuple(sorted(
            set(mechanism["published_single_task_active_aic_groups"])
            | set(extension)))
        assert dict(zip(
            mechanism["single_task_active_aic_groups"],
            mechanism["single_task_duration_ns"])) == {
                **dict(zip(
                    mechanism["published_single_task_active_aic_groups"],
                    mechanism["published_single_task_duration_ns"])),
                **extension}
        assert hw.fia_cross_domain_three_handoff_time_ns((1,)) == 845.0
        assert hw.fia_cross_domain_three_handoff_time_ns((1,) * 4) == 960.0
        assert hw.fia_cross_domain_three_handoff_time_ns((1,) * 6) == 1320.0
        for groups, duration in extension.items():
            assert hw.fia_cross_domain_three_handoff_time_ns(
                (1,) * groups) == duration
        assert hw.fia_cross_domain_three_handoff_time_ns((2,)) == 870.0
        assert hw.fia_cross_domain_three_handoff_time_ns((2,) * 24) == (
            pytest.approx(4070.0))
        assert hw.fia_cross_domain_three_handoff_time_ns((64,) * 24) == (
            pytest.approx(118270.0))
        assert hw.fia_cross_domain_three_handoff_time_ns(
            (64,) + (1,) * 7) == pytest.approx(18565.0)
        assert hw.fia_cross_domain_three_handoff_time_ns(
            (128,) * 24) == pytest.approx(236153.87096774214)
        for task_count in (0, 129):
            with pytest.raises(ValueError, match="task count.*outside"):
                hw.fia_cross_domain_three_handoff_time_ns((task_count,))
        with pytest.raises(TypeError, match="non-empty tuple"):
            hw.fia_cross_domain_three_handoff_time_ns([])
        # Two active AIC groups stay outside the domain: that point was
        # measured but its six-run span exceeded the limit this mechanism
        # gates on, so the lookup still refuses it instead of interpolating.
        assert 2 not in mechanism["single_task_active_aic_groups"]
        with pytest.raises(ValueError, match="exact single-task"):
            hw.fia_cross_domain_three_handoff_time_ns((2, 2))

    def test_fia_aiv_instruction_cold_fill_uses_exact_evidence_domain(self):
        import json

        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "fia_aiv_instruction_cold_fill_cann_8_5_1.json"))
        mechanism = hw.fia_aiv_instruction_cold_fill

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert reference["mechanism_id"] == mechanism["model"]
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            11, 2048, 128) == 2480.0
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            11, 512, 128) == 3180.0
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            11, 2048, 64) is None
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            16, 2048, 128) == 3720.0
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            16, 1024, 64) == 5320.0
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            16, 1024, 128) is None
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            21, 2048, 128) == 7220.0
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            21, 2048, 64) is None
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            22, 512, 64) == 7520.0
        assert hw.fia_regular_aiv_instruction_cold_fill_time_ns(
            22, 512, 128) == 6120.0
        assert hw.fia_split_fuse_aiv_instruction_cold_fill_time_ns(
            24) == 8600.0
        with pytest.raises(ValueError, match="exact measured"):
            hw.fia_split_fuse_aiv_instruction_cold_fill_time_ns(23)

    def test_fia_mla_rejects_sparse_topk_scan_service(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        # CalcSinnerTopKBegin is implemented by SparseFlashAttentionMla,
        # not the FiaKernelNonQuantMla binary selected for MLA decode.
        assert hw.fia_mla_topk_scan_service is None
        assert hw.fia_mla_topk_scan_steady_service is None
        assert hw.fia_mla_topk_scan_times_ns(
            22, 512, "fia_block_cube_nonquant_mla_multibuffer", 6) is None

    def test_fia_aiv_sync_all_uses_independent_mechanism_evidence(self):
        import json

        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "fia_aiv_sync_all_cann_8_5_1.json"))
        mechanism = hw.fia_aiv_sync_all_service

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert reference["mechanism_id"] == mechanism["model"]
        assert reference["bounds"]["aiv_per_group"] == 2
        assert reference["parameters_ns"]["startup_ns"] == mechanism[
            "startup_ns"]
        assert reference["parameters_ns"][
            "per_active_aiv_worker_ns"] == mechanism[
                "per_active_aiv_worker_ns"]
        assert hw.fia_aiv_sync_all_time_ns(2) == pytest.approx(
            384.1, abs=0.1)
        assert hw.fia_aiv_sync_all_time_ns(48) == pytest.approx(
            3115.9, abs=0.1)
        for worker_count in (1, 49):
            with pytest.raises(ValueError, match="outside the measured"):
                hw.fia_aiv_sync_all_time_ns(worker_count)
        with pytest.raises(TypeError, match="must be an integer"):
            hw.fia_aiv_sync_all_time_ns(2.0)

    def test_fia_flash_decode_prelude_uses_independent_evidence(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = hw.fia_flash_decode_terminal_prelude_service

        assert mechanism["production_or_evaluation_duration_consumed"] is False
        assert mechanism["icache_preload_count"] == 2
        assert mechanism["aiv_buffer_count"] == 14
        assert mechanism["aiv_buffer_bytes"] == 97536
        assert hw.fia_flash_decode_terminal_prelude_time_ns(
            1) == pytest.approx(598.3333333333334)
        assert hw.fia_flash_decode_terminal_prelude_time_ns(
            16) == pytest.approx(2199.4202898550725)
        assert hw.fia_flash_decode_terminal_prelude_time_ns(
            24) == pytest.approx(3053.3333333333335)
        for active_groups in (0, 25):
            with pytest.raises(ValueError, match="outside the measured"):
                hw.fia_flash_decode_terminal_prelude_time_ns(active_groups)
        with pytest.raises(TypeError, match="must be an integer"):
            hw.fia_flash_decode_terminal_prelude_time_ns(16.0)

    def test_fia_tiling_materialization_uses_independent_evidence(self):
        import json

        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "fia_tiling_data_materialization_cann_8_5_1.json"))
        mechanism = hw.fia_tiling_data_materialization_service

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert reference["mechanism_id"] == mechanism["model"]
        for name in (
                "payload_bytes", "aiv_transfer_bytes", "aiv_per_group",
                "minimum_active_aic_groups", "maximum_active_aic_groups"):
            assert reference["bounds"][name] == mechanism[name]
        for name in (
                "aic_completion_ns", "aiv_startup_ns",
                "aiv_per_active_worker_ns"):
            assert reference["parameters_ns"][name] == mechanism[name]
        for name in (
                "source_commit", "source_snapshot_sha256", "binary_sha256",
                "raw_path", "report_path", "report_sha256"):
            assert reference["provenance"][name] == mechanism[name]
        assert tuple(reference["provenance"]["raw_sha256"]) == mechanism[
            "raw_sha256"]
        aic_ns, aiv_ns = hw.fia_tiling_data_materialization_times_ns(
            16, 1560, 1568)
        assert aic_ns == pytest.approx(12.2216796875)
        assert aiv_ns == pytest.approx(1234.2921557020718)
        with pytest.raises(ValueError, match="outside the measured"):
            hw.fia_tiling_data_materialization_times_ns(25, 1560, 1568)
        with pytest.raises(ValueError, match="payload byte count changed"):
            hw.fia_tiling_data_materialization_times_ns(16, 28, 1568)
        with pytest.raises(TypeError, match="must be an integer"):
            hw.fia_tiling_data_materialization_times_ns(16.0, 1560, 1568)

    def test_fia_mixed_task_outer_envelope_uses_independent_evidence(self):
        import json

        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "fia_mixed_task_outer_envelope_cann_8_5_1.json"))
        mechanism = hw.fia_mixed_task_outer_envelope

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert reference["mechanism_id"] == mechanism["model"]
        assert reference["bounds"]["aiv_per_group"] == 2
        assert reference["parameters_ns"]["startup_ns"] == mechanism[
            "startup_ns"]
        assert reference["parameters_ns"][
            "per_active_aiv_worker_ns"] == mechanism[
                "per_active_aiv_worker_ns"]
        for name in (
                "source_commit", "source_snapshot_sha256", "binary_sha256",
                "raw_path", "raw_sha256", "report_path", "report_sha256"):
            assert reference["provenance"][name] == mechanism[name]
        assert hw.fia_mixed_task_outer_envelope_time_ns(2) == pytest.approx(
            5324.367140345575)
        assert hw.fia_mixed_task_outer_envelope_time_ns(48) == pytest.approx(
            7885.694)
        for worker_count in (1, 49):
            with pytest.raises(ValueError, match="outside the measured"):
                hw.fia_mixed_task_outer_envelope_time_ns(worker_count)
        with pytest.raises(TypeError, match="must be an integer"):
            hw.fia_mixed_task_outer_envelope_time_ns(2.0)

    def test_aic_mte2_nd2nz_service_uses_measured_byte_and_group_axes(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = hw.aic_mte2_nd2nz_service

        assert hw.aic_mte2_nd2nz_serial_time_ns(32, 1) == pytest.approx(
            115.842041572)
        assert hw.aic_mte2_nd2nz_serial_time_ns(256, 24) == pytest.approx(
            178.469254176)
        assert hw.aic_mte2_nd2nz_serial_time_ns(768, 8) == pytest.approx(
            (244.241981506 + 253.938064575) / 2)
        assert hw.aic_mte2_nd2nz_stream_time_ns(512, 1) == pytest.approx(
            122.615610758)
        assert hw.aic_mte2_nd2nz_stream_time_ns(32768, 24) == pytest.approx(
            443.708496094)
        assert hw.aic_mte2_nd2nz_stream_time_ns(24576, 8) == pytest.approx(
            (197.742106120 + 323.164062500) / 2)
        large_reference_path = Path(
            "microbench/ascend_910b/references/"
            "aic_mte2_nd2nz_large_contiguous_resident_cann_8_5_1.json")
        large_mechanism = hw.aic_mte2_nd2nz_large_contiguous_service
        assert hashlib.sha256(large_reference_path.read_bytes()).hexdigest() == (
            large_mechanism["evidence_sha256"])
        assert hw.aic_mte2_nd2nz_large_contiguous_supports(128, 512, 22)
        assert hw.aic_mte2_nd2nz_large_contiguous_supports(256, 512, 16)
        assert not hw.aic_mte2_nd2nz_large_contiguous_supports(64, 512, 22)
        assert hw.aic_mte2_nd2nz_large_contiguous_serial_time_ns(
            128, 512, 22) == pytest.approx(827.0833333335)
        assert hw.aic_mte2_nd2nz_large_contiguous_stream_time_ns(
            256, 256, 16) == pytest.approx(
                596.904761905
                + (747.529761905 - 596.904761905) * 8 / 14)
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            98304, 24, stream=False) == pytest.approx(
                644.562872023625)
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            98304, 24, stream=True) == pytest.approx(
                556.428571428)
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            65536, 24, stream=True) > 0.0
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            32768, 24, stream=True) > 0.0
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            32767, 24, stream=True) is None
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            131072, 24, stream=True) == pytest.approx(
                738.0580357145)
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            262144, 24, stream=True) == pytest.approx(
                1464.575892857)
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            131072, 21, stream=True,
            dram_miss_fraction=0.31746031746031744) == pytest.approx(
                1078.557551493016)
        assert hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            98304, 19, stream=True,
            dram_miss_fraction=0.2) is None
        assert hw.aic_mte2_nd2nz_large_contiguous_cold_supports(
            128, 512, 23)
        assert not hw.aic_mte2_nd2nz_large_contiguous_cold_supports(
            128, 512, 21)
        assert hw.aic_mte2_nd2nz_large_contiguous_completion_time_ns(
            128, 512, 22, 0.0, stream=True) == pytest.approx(
                723.973214286)
        assert hw.aic_mte2_nd2nz_large_contiguous_completion_time_ns(
            128, 512, 22, 1.0, stream=True) == pytest.approx(
                1916.3839285715)
        assert hw.aic_mte2_nd2nz_large_contiguous_completion_time_ns(
            128, 512, 22, 0.5, stream=True) == pytest.approx(
            (723.973214286 + 1916.3839285715) / 2)
        gemm_reference_path = Path(
            "microbench/ascend_910b/references/"
            "aic_mte2_gemm_large_resident_descriptor_cann_8_5_1.json")
        gemm_mechanism = hw.aic_mte2_gemm_large_resident_service
        assert hashlib.sha256(gemm_reference_path.read_bytes()).hexdigest() == (
            gemm_mechanism["evidence_sha256"])
        assert hw.aic_mte2_gemm_large_resident_supports(
            16, 5120, 5120, 23)
        assert hw.aic_mte2_gemm_large_resident_supports(
            224, 256, 5120, 23)
        assert hw.aic_mte2_gemm_large_resident_supports(
            16, 4096, 4096, 24)
        assert hw.aic_mte2_gemm_large_resident_supports(
            176, 256, 4096, 24)
        assert not hw.aic_mte2_gemm_large_resident_supports(
            224, 256, 4096, 23)
        assert hw.aic_mte2_gemm_large_resident_serial_time_ns(
            16, 5120, 5120, 23) == pytest.approx(1012.5)
        assert hw.aic_mte2_gemm_large_resident_stream_time_ns(
            224, 256, 5120, 23) == pytest.approx(606.4583333334999)
        assert hw.aic_mte2_gemm_large_resident_serial_time_ns(
            16, 4096, 4096, 24) == pytest.approx(847.2023809525)
        assert hw.aic_mte2_gemm_large_resident_stream_time_ns(
            176, 256, 4096, 24) == pytest.approx(533.5119047620001)
        assert hw.aic_mte2_gemm_large_resident_program_supports(
            active_aic_groups=23,
            a_rows=16,
            a_columns=5120,
            a_source_row_stride_elements=5120,
            a_buffer_depth=1,
            a_descriptors_per_worker=1,
            b_rows=224,
            b_columns=256,
            b_source_row_stride_elements=5120,
            b_buffer_depth=2,
            b_descriptors_per_worker=20,
        )
        assert not hw.aic_mte2_gemm_large_resident_program_supports(
            active_aic_groups=23,
            a_rows=32,
            a_columns=256,
            a_source_row_stride_elements=5120,
            a_buffer_depth=2,
            a_descriptors_per_worker=40,
            b_rows=112,
            b_columns=256,
            b_source_row_stride_elements=5120,
            b_buffer_depth=2,
            b_descriptors_per_worker=40,
        )
        assert hw.aic_mte2_gemm_large_resident_program_supports(
            active_aic_groups=24,
            a_rows=32,
            a_columns=4096,
            a_source_row_stride_elements=4096,
            a_buffer_depth=1,
            a_descriptors_per_worker=1,
            b_rows=176,
            b_columns=256,
            b_source_row_stride_elements=4096,
            b_buffer_depth=2,
            b_descriptors_per_worker=16,
        )
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_nd2nz_large_contiguous_stream_time_ns(
                128, 512, 25)
        assert hw.aic_mte2_nd2nz_strided_serial_time_ns(
            256, 64, 512, 16) == pytest.approx(546.2671666665)
        assert hw.aic_mte2_nd2nz_strided_stream_time_ns(
            256, 128, 1024, 24) == pytest.approx(490.6865)
        assert hw.aic_mte2_nd2nz_strided_serial_time_ns(
            128, 128, 1024, 16) == pytest.approx(334.516)
        assert hw.aic_mte2_nd2nz_strided_stream_time_ns(
            128, 128, 1024, 8) == pytest.approx(245.118)
        miss_fraction = 8_421_376 / 9_273_344
        assert hw.aic_mte2_nd2nz_cold_completion_supports(
            4, 64, 64, 22)
        assert hw.aic_mte2_nd2nz_cold_completion_supports(
            256, 64, 512, 22)
        assert hw.aic_mte2_nd2nz_cold_completion_supports(
            4, 256, 512, 22)
        assert hw.aic_mte2_nd2nz_cold_completion_supports(
            4, 256, 512, 16)
        assert hw.aic_mte2_nd2nz_cold_completion_supports(
            256, 128, 1024, 16)
        assert hw.aic_mte2_nd2nz_cold_completion_supports(
            128, 128, 1024, 16)
        assert not hw.aic_mte2_nd2nz_cold_completion_supports(
            256, 64, 4096, 16)
        assert hw.aic_mte2_nd2nz_cold_completion_time_ns(
            256, 64, 512, 22, miss_fraction) == pytest.approx(
                551.218 + miss_fraction * 1107.302333333)
        assert hw.aic_mte2_nd2nz_cold_completion_time_ns(
            4, 256, 512, 22, 1.0) == pytest.approx(
                138.390666667 + 145.285666667)
        assert hw.aic_mte2_nd2nz_cold_stream_time_ns(
            15_753_216, 23) == pytest.approx(
                15_753_216
                / (29_592_640_612.76537 * 22) * 1.0e9)
        reference = json.loads(Path(
            "microbench/ascend_910b/references/"
            "aic_mte2_nd2nz_cold_stream_parallel_constraint_cann_8_5_1.json"
        ).read_text())
        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert reference["forward_parameters"][
            "parameter_fit_from_this_validation"] is False
        assert reference["forward_parameters"][
            "cold_completion_bandwidth_bytes_per_s_per_aic"] == (
                mechanism["cold_completion_bandwidth_bytes_per_s"])
        assert reference["mechanism_validation"][
            "analytical_shared_dram_constraint_ns"] == pytest.approx(
                hw.aic_mte2_nd2nz_cold_stream_time_ns(
                    reference["mechanism_validation"][
                        "aggregate_compulsory_operand_bytes"],
                    reference["mechanism_validation"][
                        "active_aic_groups"]))
        finite_validation = reference[
            "fia_paged_gqa_finite_program_validation"]
        assert finite_validation["maximum_descriptors_per_worker"] == 160
        assert finite_validation["mechanism_gate_passed"] is True
        assert finite_validation[
            "production_or_evaluation_duration_consumed"] is False
        assert finite_validation[
            "parameter_fit_from_this_validation"] is False
        assert max(
            result["maximum_run_median_deviation_pct"]
            for result in finite_validation["count_results"]
            if "maximum_run_median_deviation_pct" in result) < 5.0
        assert reference["excluded_diagnostic"]["status"] == "NO_GO"
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aic_mte2_nd2nz_serial_time_ns(16, 8)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aic_mte2_nd2nz_serial_time_ns(512, 25)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aic_mte2_nd2nz_stream_time_ns(256, 8)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_nd2nz_strided_serial_time_ns(128, 64, 512, 16)
        reference = json.loads(Path(
            "microbench/ascend_910b/references/"
            "aic_mte2_paged_gqa128_stride_cann_8_5_1.json").read_text())
        assert reference["production_or_evaluation_duration_consumed"] is False
        assert reference["parameters_ns"][
            "serial_duration_by_active_aic_group"] == [
                mechanism["strided_geometries"][(128, 128, 1024)][
                    "serial_duration_ns"][index]
                for index in range(3)]
        assert reference["stability_gate"]["passed"] is True
        assert hw.aic_mte2_nd2nz_cold_completion_time_ns(
            256, 128, 1024, 16, 1.0) == pytest.approx(
                max(
                    hw.aic_mte2_nd2nz_serial_time_ns(65536, 16),
                    557.3595)
                + 65536 * 16 / (29592640612.76537 * 22) * 1.0e9)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_nd2nz_cold_completion_time_ns(
                256, 64, 4096, 16, miss_fraction)
        with pytest.raises(ValueError, match="outside the measured range"):
            hw.aic_mte2_nd2nz_cold_stream_time_ns(4096, 25)

    def test_aic_mte2_33copy_group_service_is_exact_and_cold_dram_only(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference_path = Path(
            "microbench/ascend_910b/references/"
            "aic_mte2_nd2nz_33copy_large_stride_group_completion_cann_8_5_1.json"
        )
        mechanism = hw.aic_mte2_nd2nz_33copy_group_service

        assert mechanism["production_or_evaluation_duration_consumed"] is False
        assert mechanism["minimum_touched_bytes"] > hw.l2_size_bytes
        assert mechanism["reference_sha256"] == hashlib.sha256(
            reference_path.read_bytes()).hexdigest()
        assert hw.aic_mte2_nd2nz_33copy_group_time_ns(
            256, 24) == pytest.approx(35_614.0616666775)
        assert hw.aic_mte2_nd2nz_33copy_group_time_ns(
            384, 24) == pytest.approx(87_064.120000011)
        assert hw.memory_exact_request_service_time_s(
            "dram",
            "aic_mte2_nd2nz_33copy_large_stride_group_completion_24_workers",
            384,
        ) == pytest.approx(87_064.120000011e-9)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_nd2nz_33copy_group_time_ns(128, 24)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_nd2nz_33copy_group_time_ns(384, 23)
        with pytest.raises(ValueError, match="requires the DRAM path"):
            hw.memory_exact_request_service_time_s(
                "mte2",
                "aic_mte2_nd2nz_33copy_large_stride_group_completion_24_workers",
                384,
            )

    def test_deterministic_split_k_source_queue_service_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference_path = Path(
            "microbench/ascend_910b/references/"
            "aic_deterministic_split_k_source_queue_completion_cann_8_5_1.json"
        )
        mechanism = hw.aic_deterministic_split_k_source_queue_service

        assert mechanism["production_or_evaluation_duration_consumed"] is False
        assert mechanism["reference_sha256"] == hashlib.sha256(
            reference_path.read_bytes()).hexdigest()
        assert (
            mechanism["a1_buffer_slots"], mechanism["b1_buffer_slots"],
            mechanism["l0a_buffer_slots"], mechanism["l0b_buffer_slots"],
        ) == (1, 2, 2, 2)
        assert mechanism["traversal_order"] == "n_then_m"
        assert hw.aic_deterministic_split_k_source_queue_time_ns(
            16, 24) == pytest.approx(474_620.0)
        assert hw.aic_deterministic_split_k_source_queue_time_ns(
            128, 24) == pytest.approx(488_970.0)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_deterministic_split_k_source_queue_time_ns(144, 24)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_deterministic_split_k_source_queue_time_ns(16, 23)

    def test_deterministic_split_k_collective_service_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference_path = Path(
            "microbench/ascend_910b/references/"
            "aic_deterministic_split_k_collective_barrier_cann_8_5_1.json")
        mechanism = hw.aic_deterministic_split_k_collective_barrier_service

        assert mechanism["production_or_evaluation_duration_consumed"] is False
        assert mechanism["reference_sha256"] == hashlib.sha256(
            reference_path.read_bytes()).hexdigest()
        assert (
            mechanism["active_aic_groups"], mechanism["active_aiv_workers"],
            mechanism["n_slabs"],
        ) == (24, 48, 2)
        assert hw.aic_deterministic_split_k_collective_barrier_time_ns(
            24, 48, 2) == pytest.approx(
                mechanism["source_collective_complete_ns"]
                - mechanism["same_flag_and_pipe_control_ns"])
        with pytest.raises(ValueError, match="outside the measured source"):
            hw.aic_deterministic_split_k_collective_barrier_time_ns(24, 46, 2)

    def test_aic_mte2_operand_pair_stream_is_an_exact_long_program_service(
            self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = hw.aic_mte2_operand_pair_stream_service
        cold_mechanism = hw.aic_mte2_cold_operand_pair_stream_service
        overlap = hw.aic_mte2_operand_pair_overlap_domain

        assert hw.aic_mte2_operand_pair_stream_time_ns(
            128, 64, 64, 256, 24, 2432) == pytest.approx(
                2432 * 292.220052083)
        assert overlap["minimum_active_aic_groups"] == 22
        assert overlap["maximum_active_aic_groups"] == 24
        assert overlap["minimum_pairs_per_worker"] == 512
        assert overlap["maximum_pairs_per_worker"] == 4096
        overlap_reference = Path(overlap["reference_path"])
        assert hashlib.sha256(overlap_reference.read_bytes()).hexdigest() == (
            overlap["reference_sha256"])
        overlap_evidence = json.loads(overlap_reference.read_text())
        assert overlap_evidence["mechanism_validation"][
            "mechanism_gate_passed"] is True
        assert overlap_evidence[
            "production_or_evaluation_duration_consumed"] is False
        assert cold_mechanism["source_residency"] == "dram_cold"
        assert cold_mechanism["measured_pair_counts_per_worker"] == (
            512, 1024, 2048)
        assert hw.aic_mte2_cold_operand_pair_stream_time_ns(
            128, 64, 64, 256, 24, 640) == pytest.approx(
                640 * 806.341145833)
        cold_reference = Path(cold_mechanism["reference_path"])
        assert hashlib.sha256(cold_reference.read_bytes()).hexdigest() == (
            cold_mechanism["reference_sha256"])
        n_major = hw.aic_mte2_cold_n_major_b_stride_service
        assert n_major["model"] == (
            "cold_m16_pair_plus_n_major_b_serial_stride_delta_v1")
        expected_pair_ns = (
            621.907552083 + 706.6875
            - hw.aic_mte2_nd2nz_serial_time_ns(32768, 23))
        assert hw.aic_mte2_cold_n_major_b_pair_time_ns(
            16, 64, 64, 256, 28672, 23, 640) == pytest.approx(
                640 * expected_pair_ns)
        for path_name, sha_name in (
                ("pair_reference_path", "pair_reference_sha256"),
                ("stride_reference_path", "stride_reference_sha256")):
            reference = Path(n_major[path_name])
            assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
                n_major[sha_name])
        legacy_stride = hw.aic_mte2_cold_legacy_k14336_stride_service
        assert legacy_stride["model"] == (
            "cold_contiguous_envelope_plus_serial_stride_delta_v1")
        assert legacy_stride["active_aic_groups"] == 22
        assert legacy_stride["descriptors"]["a"] == {
            "rows": 16,
            "columns": 4096,
            "source_row_stride_elements": 14336,
            "copied_bytes_per_descriptor": 131072,
            "serial_completion_ns": pytest.approx(2319.880952381),
            "two_buffer_stream_ns": pytest.approx(2232.113095238),
        }
        assert legacy_stride["descriptors"]["b"] == {
            "rows": 96,
            "columns": 640,
            "source_row_stride_elements": 14336,
            "copied_bytes_per_descriptor": 122880,
            "serial_completion_ns": pytest.approx(2779.4047619045),
            "two_buffer_stream_ns": pytest.approx(2706.7708333335),
        }
        legacy_reference = Path(legacy_stride["reference_path"])
        assert hashlib.sha256(legacy_reference.read_bytes()).hexdigest() == (
            legacy_stride["reference_sha256"])
        v3_stride = hw.aic_mte2_cold_v3_k1536_n5120_stride_service
        assert set(v3_stride["descriptors"]) == {
            (224, 256, 5120, 23),
            (256, 256, 5120, 20),
            (256, 256, 5120, 24),
        }
        assert v3_stride["descriptors"][(256, 256, 5120, 24)][
            "serial_completion_ns"] == pytest.approx(2360.7589285715)
        v3_reference = Path(v3_stride["reference_path"])
        assert hashlib.sha256(v3_reference.read_bytes()).hexdigest() == (
            v3_stride["reference_sha256"])
        small_m_stride = hw.aic_mte2_cold_small_m_stride_service
        assert small_m_stride["descriptors"][(
            128, 512, 5120, 24)]["serial_completion_ns"] == pytest.approx(
                2264.83631)
        assert (256, 256, 13824, 24) not in small_m_stride["descriptors"]
        small_m_reference = Path(small_m_stride["reference_path"])
        assert hashlib.sha256(small_m_reference.read_bytes()).hexdigest() == (
            small_m_stride["reference_sha256"])
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_cold_n_major_b_pair_time_ns(
                16, 64, 64, 256, 8192, 23, 640)
        with pytest.raises(ValueError, match="geometry is outside"):
            hw.aic_mte2_operand_pair_stream_time_ns(
                32, 64, 64, 256, 24, 2432)
        with pytest.raises(ValueError, match="length is outside"):
            hw.aic_mte2_operand_pair_stream_time_ns(
                128, 64, 64, 256, 24, 48)
        with pytest.raises(ValueError, match="length is outside"):
            hw.aic_mte2_cold_operand_pair_stream_time_ns(
                128, 64, 64, 256, 24, 4096)

    def test_aic_mte2_finite_stream_and_bf16_terminal_fixpipe_services(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        finite = hw.aic_mte2_finite_stream_completion_service
        terminal = hw.aic_fixpipe_bf16_terminal_tile_service
        fixpipe_stream = hw.aic_fixpipe_bf16_finite_stream_service

        finite_reference = Path(finite["reference_path"])
        assert hashlib.sha256(finite_reference.read_bytes()).hexdigest() == (
            finite["reference_sha256"])
        assert finite["descriptor_bytes"] == 131072
        assert hw.aic_mte2_finite_stream_completion_time_ns(
            256, 256, 8192, 16, 64) == pytest.approx(269770.000404648)
        assert hw.aic_mte2_finite_stream_completion_time_ns(
            256, 256, 13824, 18, 60) == pytest.approx(269109.99517305)
        assert hw.aic_mte2_finite_stream_completion_time_ns(
            256, 256, 13824, 22, 60) == pytest.approx(301130.011677742)
        with pytest.raises(ValueError, match="stride is outside"):
            hw.aic_mte2_finite_stream_completion_time_ns(
                256, 256, 5120, 22, 60)
        with pytest.raises(ValueError, match="length is outside"):
            hw.aic_mte2_finite_stream_completion_time_ns(
                256, 256, 13824, 22, 73)

        terminal_reference = Path(terminal["reference_path"])
        assert hashlib.sha256(terminal_reference.read_bytes()).hexdigest() == (
            terminal["reference_sha256"])
        assert terminal["active_aic_groups"] == 1
        assert hw.aic_fixpipe_bf16_terminal_tile_time_ns(
            128, 256) == pytest.approx(442.96875)
        with pytest.raises(ValueError, match="tile is outside"):
            hw.aic_fixpipe_bf16_terminal_tile_time_ns(256, 128)

        fixpipe_reference = Path(fixpipe_stream["reference_path"])
        assert hashlib.sha256(fixpipe_reference.read_bytes()).hexdigest() == (
            fixpipe_stream["reference_sha256"])
        assert set(fixpipe_stream["cells"]) == {
            (16, 176, 22),
            (32, 256, 16), (32, 256, 20), (32, 256, 22),
            (48, 256, 22),
            (64, 256, 20), (64, 256, 22),
            (128, 256, 24),
        }
        assert hw.aic_fixpipe_bf16_finite_stream_time_ns(
            64, 256, 20, 1) == pytest.approx(287.7099609375)
        assert hw.aic_fixpipe_bf16_finite_stream_time_ns(
            32, 256, 22, 3) == pytest.approx(397.1728515625)
        assert hw.aic_fixpipe_bf16_finite_stream_time_ns(
            128, 256, 24, 72) == pytest.approx(745.6640625)
        assert hw.aic_fixpipe_bf16_finite_stream_time_ns(
            128, 256, 24, 1195) == pytest.approx(18618.75)
        with pytest.raises(ValueError, match="geometry is outside"):
            hw.aic_fixpipe_bf16_finite_stream_time_ns(64, 256, 24, 1)
        with pytest.raises(ValueError, match="outside the measured"):
            hw.aic_fixpipe_bf16_finite_stream_time_ns(64, 256, 24, 97)
        with pytest.raises(ValueError, match="outside the measured"):
            hw.aic_fixpipe_bf16_finite_stream_time_ns(128, 256, 24, 1196)

    def test_aic_gemm_source_order_completion_is_exact_and_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        collection = hw.aic_mte2_gemm_source_order_completion_service
        service = next(
            value for value in collection["programs"].values()
            if value["a_source_row_stride_elements"] == 8192)
        reference = Path(service["reference_path"])

        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            service["reference_sha256"])
        signature = {
            name: service[name]
            for name in (
                "dtype", "active_aic_groups", "operand_buffer_depth",
                "base_tile_m", "base_tile_n", "base_tile_k", "mte1_base_k",
                "a_rows", "a_columns", "a_source_row_stride_elements",
                "b_rows", "b_columns", "b_source_row_stride_elements",
                "m_groups", "n_groups", "output_tiles_per_worker",
                "k_slabs", "mte2_programs_per_worker",
                "mte1_pairs_per_worker")
        }
        assert hw.aic_mte2_gemm_source_order_completion_time_ns(
            **signature) == pytest.approx(781080.0)
        assert hw.aic_mte2_gemm_source_order_service(
            **signature)["mte2_only_ns"] == pytest.approx(578710.0)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_gemm_source_order_completion_time_ns(
                **{**signature, "output_tiles_per_worker": 9})

        k13824 = next(
            value for value in collection["programs"].values()
            if value["a_source_row_stride_elements"] == 13824)
        k13824_signature = {name: k13824[name] for name in signature}
        assert hw.aic_mte2_gemm_source_order_service(
            **k13824_signature)["mte2_only_ns"] == pytest.approx(321120.0)
        assert hw.aic_mte2_gemm_source_order_completion_time_ns(
            **k13824_signature) == pytest.approx(475330.0)
        capacities = tuple(
            cell["capacity_bytes_per_s"]
            for cell in collection["capacity_constrained_cells"])
        assert capacities == pytest.approx((
            108715176858.87578,
            121226905829.59642,
        ))
        assert collection[
            "capacity_constrained_bytes_per_s"] == pytest.approx(
                114971041344.2361)
        assert collection[
            "capacity_constrained_completion_bytes_per_s"] == pytest.approx(
                81222879717.65503)
        assert hw.aic_mte2_gemm_capacity_constrained_source_order_time_ns(
            128, 512, 304, 256, 256, 608, 24) == pytest.approx(
                1039719.7642325507)
        assert hw.aic_mte2_gemm_capacity_constrained_source_completion_time_ns(
            128, 512, 56, 128, 512, 336, 22) == pytest.approx(
                632583.1364094288)
        assert hw.aic_mte2_gemm_capacity_constrained_source_order_time_ns(
            128, 512, 304, 256, 256, 608, 23) == pytest.approx(
                1039719.7642325507)
        assert hw.aic_mte2_gemm_capacity_constrained_source_order_time_ns(
            128, 256, 608, 128, 256, 608, 24) == pytest.approx(
                693146.5094883671)
        assert hw.aic_mte2_gemm_capacity_constrained_source_order_time_ns(
            128, 1, 608, 128, 256, 608, 24) == pytest.approx(
                347927.1947282808)

    def test_aic_gemm_shared_source_queue_is_descriptor_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_mte2_gemm_shared_source_queue_service
        reference = Path(service["evidence_ref"])

        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            service["evidence_sha256"])
        assert hw.aic_mte2_gemm_shared_source_queue_supports(
            dtype="bf16",
            active_aic_groups=24,
            operand_buffer_depth=2,
            traversal_order="persistent_n_then_m",
            a_rows=128,
            a_columns=512,
            a_source_row_stride_elements=14336,
            b_rows=240,
            b_columns=256,
            b_source_row_stride_elements=14336,
        )
        assert not hw.aic_mte2_gemm_shared_source_queue_supports(
            dtype="bf16",
            active_aic_groups=23,
            operand_buffer_depth=2,
            traversal_order="persistent_n_then_m",
            a_rows=128,
            a_columns=512,
            a_source_row_stride_elements=14336,
            b_rows=240,
            b_columns=256,
            b_source_row_stride_elements=14336,
        )
        assert hw.aic_mte2_gemm_shared_source_descriptor_time_ns(
            128, 512, 14336, 0.0, stream=True) == pytest.approx(
                686.726190476)
        assert hw.aic_mte2_gemm_shared_source_descriptor_time_ns(
            128, 512, 14336, 1.0, stream=True) == pytest.approx(
                2639.4047619045)

    def test_aic_gemm_finite_ingress_is_pair_count_analytical(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_gemm_finite_ingress_completion_service
        for name in ("calibration", "cross_geometry_validation"):
            program = service[name]
            reference = Path(program["reference_path"])
            assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
                program["reference_sha256"])
        assert hw.aic_gemm_finite_ingress_completion_time_ns(
            completion_pairs_per_worker=48) == pytest.approx(
                22840.0)
        assert hw.aic_gemm_finite_ingress_completion_time_ns(
            completion_pairs_per_worker=96) == pytest.approx(
                43640.0)
        assert hw.aic_gemm_finite_ingress_completion_time_ns(
            completion_pairs_per_worker=672) == pytest.approx(293240.0)
        assert hw.aic_gemm_finite_ingress_completion_time_ns(
            completion_pairs_per_worker=720) == pytest.approx(314040.0)
        assert service["maximum_cross_geometry_rate_delta"] == 0.02
        assert service[
            "maximum_validated_completion_pairs_per_worker"] == 720
        assert service[
            "maximum_validated_source_descriptors_per_worker"] == 360
        transition_reference = Path(service["transition_reference_path"])
        assert hashlib.sha256(transition_reference.read_bytes()).hexdigest() \
            == service["transition_reference_sha256"]
        exact_programs = service["exact_transition_services"]
        assert len(exact_programs) == 10
        k8192 = next(
            program for program in exact_programs.values()
            if program["source_row_stride_elements"] == 8192
            and program["base_tile_n"] == 128
            and program["output_tiles_per_worker"] == 6)
        signature = {
            name: k8192[name]
            for name in (
                "dtype", "operand_buffer_depth", "active_aic_groups",
                "m_groups", "n_groups", "base_tile_m", "base_tile_n",
                "base_tile_k", "a_rows", "a_columns", "b_rows",
                "b_columns", "source_row_stride_elements",
                "output_tiles_per_worker", "k_slabs",
                "source_descriptors_per_worker",
                "completion_pairs_per_worker")
        }
        assert hw.aic_gemm_finite_transition_service(
            **signature)["mte2_only_ns"] == pytest.approx(214260.0)
        assert hw.aic_gemm_finite_transition_service(
            **{**signature, "source_row_stride_elements": 4096}) is None
        k8192_n256_long = next(
            program for program in exact_programs.values()
            if program["source_row_stride_elements"] == 8192
            and program["base_tile_n"] == 256
            and program["output_tiles_per_worker"] == 19)
        k8192_n256_long_signature = {
            name: k8192_n256_long[name] for name in signature}
        assert hw.aic_gemm_finite_transition_service(
            **k8192_n256_long_signature)["mte2_only_ns"] == pytest.approx(
                1071650.0)
        k13824 = next(
            program for program in exact_programs.values()
            if program["source_row_stride_elements"] == 13824)
        k13824_signature = {name: k13824[name] for name in signature}
        assert hw.aic_gemm_finite_transition_service(
            **k13824_signature)["mte2_only_ns"] == pytest.approx(224800.0)

    def test_aic_short_gemm_pipeline_is_output_tile_analytical(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_gemm_short_pipeline_completion_service
        reference = Path(service["reference_path"])

        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            service["reference_sha256"])
        assert service["minimum_output_tiles_per_worker"] == 1
        assert service["maximum_output_tiles_per_worker"] == 4
        assert hw.aic_gemm_short_pipeline_completion_time_ns(
            output_tiles_per_worker=1) == pytest.approx(17330.0)
        assert hw.aic_gemm_short_pipeline_completion_time_ns(
            output_tiles_per_worker=4) == pytest.approx(58100.0)
        assert hw.aic_gemm_short_pipeline_completion_time_ns(
            output_tiles_per_worker=5) is None

    def test_aic_gemm_persistent_issue_is_exact_and_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_mte2_gemm_persistent_issue_service
        reference = Path(service["reference_path"])

        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            service["reference_sha256"])
        signature = {
            name: service[name]
            for name in (
                "dtype", "active_aic_groups", "worker_grid_m",
                "worker_grid_n", "operand_buffer_depth", "base_tile_m",
                "base_tile_n", "base_tile_k", "a_rows", "a_columns",
                "a_source_row_stride_elements", "b_rows", "b_columns",
                "b_source_row_stride_elements", "output_tiles_per_worker",
                "k_slabs", "mte2_programs_per_worker")
        }
        assert hw.aic_mte2_gemm_persistent_issue_time_ns(
            **signature) == pytest.approx(754420.0)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_gemm_persistent_issue_time_ns(
                **{**signature, "k_slabs": 47})

    def test_aic_matmul_v2_dynamic_m_tail_control_is_mechanism_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_matmul_v2_dynamic_m_tail_control_service
        reference = Path(service["reference_path"])

        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            service["reference_sha256"])
        assert hw.aic_matmul_v2_dynamic_m_tail_control_time_ns() == (
            pytest.approx(37.6041640625))
        assert (
            hw.aic_matmul_v2_dynamic_m_tail_maximum_descriptor_k()
            == 256
        )
        assert HWSpec(
            "configs/hardware/nvidia_a800.yaml"
        ).aic_matmul_v2_dynamic_m_tail_control_service is None

    def test_aic_matmul_v2_base_k_lifecycle_is_mechanism_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_matmul_v2_base_k_lifecycle_service
        reference = Path(service["reference_path"])

        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            service["reference_sha256"])
        assert hw.aic_matmul_v2_base_k_lifecycle_time_ns(
            32, 2, 128) == pytest.approx(52741.36054421769)
        assert hw.aic_matmul_v2_base_k_lifecycle_supports(
            "bf16", 24, 96, 128, 32, 4096, 2,
            512, 256, 2, False)
        assert not hw.aic_matmul_v2_base_k_lifecycle_supports(
            "bf16", 22, 16, 96, 64, 14336, 2,
            256, 256, 2, False)
        with pytest.raises(ValueError, match="measured base-K domain"):
            hw.aic_matmul_v2_base_k_lifecycle_time_ns(16, 2, 128)
        assert HWSpec(
            "configs/hardware/nvidia_a800.yaml"
        ).aic_matmul_v2_base_k_lifecycle_service is None

    def test_aic_matmul_v2_small_m_base_k_control_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_matmul_v2_small_m_base_k_control_service
        reference = Path(service["reference_path"])

        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            service["reference_sha256"])
        assert hw.aic_matmul_v2_small_m_base_k_control_time_ns() == (
            pytest.approx(200.57823129251713))
        for workers, base_n, projection_k in (
                (22, 96, 11008),
                (22, 96, 14336),
                (23, 112, 13824)):
            assert hw.aic_matmul_v2_small_m_base_k_control_supports(
                "bf16", workers, 16, base_n, 64, projection_k, 2,
                512, 512, 2, True)
        assert not hw.aic_matmul_v2_small_m_base_k_control_supports(
            "bf16", 24, 16, 96, 64, 14336, 2,
            512, 512, 2, True)
        assert not hw.aic_matmul_v2_small_m_base_k_control_supports(
            "bf16", 22, 16, 96, 64, 14336, 2,
            512, 512, 2, False)
        assert HWSpec(
            "configs/hardware/nvidia_a800.yaml"
        ).aic_matmul_v2_small_m_base_k_control_service is None

    def test_aic_gemm_region_completion_is_exact_and_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_mte2_gemm_region_completion_service
        services = hw.aic_mte2_gemm_region_completion_services

        assert len(services) == 10
        assert service is services[0]
        for item in services:
            reference = Path(item["reference_path"])
            assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
                item["reference_sha256"])
        signature_fields = (
            "dtype", "active_aic_groups", "operand_buffer_depth",
            "base_tile_m", "base_tile_n", "base_tile_k", "a_rows",
            "a_columns", "a_source_row_stride_elements", "b_rows",
            "b_columns", "b_source_row_stride_elements", "m_tiles",
            "n_tiles", "m_regions", "n_regions", "m_tiles_per_region",
            "n_tiles_per_region", "maximum_output_tiles_per_worker",
            "k_slabs", "mte2_programs_per_worker",
            "mte1_pairs_per_worker")
        signature = {
            name: service[name]
            for name in signature_fields
        }
        assert hw.aic_mte2_gemm_region_completion_time_ns(
            **signature) == pytest.approx(2180810.0)
        supplemental = services[1]
        supplemental_signature = {
            name: supplemental[name]
            for name in signature_fields
        }
        assert hw.aic_mte2_gemm_region_completion_time_ns(
            **supplemental_signature) == pytest.approx(2200250.0)
        tail = next(
            item for item in services
            if (item["m_tiles"], item["n_tiles"], item["k_slabs"])
            == (32, 16, 28))
        tail_signature = {
            name: tail[name]
            for name in signature_fields
        }
        assert tail["mte2_only_ns"] == pytest.approx(1402740.0)
        assert hw.aic_mte2_gemm_region_completion_time_ns(
            **tail_signature) == pytest.approx(1927240.0)
        long_region = services[2]
        long_region_signature = {
            name: long_region[name]
            for name in signature_fields
        }
        assert tuple(long_region[name] for name in (
            "m_regions", "n_regions", "m_tiles_per_region",
            "n_tiles_per_region")) == (32, 8, 8, 14)
        assert long_region["mte2_only_ns"] == pytest.approx(47167770.0)
        assert hw.aic_mte2_gemm_region_completion_time_ns(
            **long_region_signature) == pytest.approx(61668590.0)
        single_region = next(
            item for item in services
            if (item["m_tiles"], item["n_tiles"], item["k_slabs"])
            == (16, 16, 8))
        single_region_signature = {
            name: single_region[name]
            for name in signature_fields
        }
        assert hw.aic_mte2_gemm_region_completion_time_ns(
            **single_region_signature) == pytest.approx(286870.0)
        two_by_two_region = next(
            item for item in services
            if (item["m_tiles"], item["n_tiles"], item["k_slabs"])
            == (16, 32, 16))
        two_by_two_region_signature = {
            name: two_by_two_region[name]
            for name in signature_fields
        }
        assert hw.aic_mte2_gemm_region_completion_time_ns(
            **two_by_two_region_signature) == pytest.approx(1125400.0)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_gemm_region_completion_time_ns(
                **{**signature, "m_regions": 2})

    def test_aic_gemm_region_pipeline_completion_is_source_bound(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_mte2_gemm_region_pipeline_completion_service
        reference = Path(service["reference_path"])

        assert service["model"] == (
            "aic_matmul_v3_l2_region_mte2_to_cube_completion_v1")
        assert service["mte2_only_ns"] == pytest.approx(10740020.0)
        assert service["ingress_completion_ns"] == pytest.approx(14718200.0)
        assert service["pipeline_completion_ns"] == pytest.approx(15210190.0)
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            service["reference_sha256"])

    def test_aic_mte2_fia_gqa_drained_count_law_is_bounded(self):
        """The count law widens descriptor count only, and stays fail-closed."""

        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_mte2_fia_gqa_finite_drain_service
        reference = Path(service["count_law_reference_path"])
        assert service["count_law_reference_sha256"] == hashlib.sha256(
            reference.read_bytes()).hexdigest()
        assert service["count_law_minimum_descriptor_count"] == 8
        assert service["count_law_maximum_descriptor_count"] == 128

        # Exact measured programs keep their own published durations and
        # always take precedence over the licensed anchor.
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 16, 8) == pytest.approx(14_480.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 16, 32) == pytest.approx(55_460.0)

        # Counts the exact grid never measured resolve through the licensed
        # per-descriptor anchor, which is count free by construction.
        anchor_ns = service["count_free_service"][(256, 128, 1024, 16)]
        for descriptor_count in (13, 40, 127, 128):
            assert hw.aic_mte2_fia_gqa_finite_drain_supports(
                256, 128, 1024, 16, descriptor_count)
            assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
                256, 128, 1024, 16, descriptor_count) == pytest.approx(
                    anchor_ns * descriptor_count)

        # Outside the validated count range, and for an unlicensed geometry
        # or concurrency, the mechanism refuses rather than extrapolating.
        assert not hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 16, 129)
        assert not hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 11, 16)
        assert not hw.aic_mte2_fia_gqa_finite_drain_supports(
            128, 128, 1024, 16, 16)
        with pytest.raises(ValueError):
            hw.aic_mte2_fia_gqa_finite_drain_time_ns(
                256, 128, 1024, 16, 129)

        # The d64 cells are licensed by the campaign but have no published
        # exact-count program at or above the validated minimum, so they must
        # not acquire an anchor.
        assert (256, 64, 512, 16) not in service["count_free_service"]

    def test_aic_mte2_fia_mha_finite_drain_is_exact_and_fail_loud(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_mte2_fia_mha_finite_drain_service
        reference = Path(service["reference_path"])

        assert service["model"] == "aic_mte2_fia_mha_finite_drain_v1"
        assert service["active_aic_groups"] == (16, 22, 24)
        assert service["reference_sha256"] == hashlib.sha256(
            reference.read_bytes()).hexdigest()
        assert hw.aic_mte2_fia_mha_finite_drain_supports(
            1, 128, 128, 22, 2)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            1, 128, 128, 22, 2) == pytest.approx(500.0)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            256, 128, 4096, 22, 24) == pytest.approx(53_170.0)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            256, 128, 4096, 24, 44) == pytest.approx(103_850.0)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            256, 128, 4096, 24, 42) == pytest.approx(99_440.0)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            1, 128, 128, 16, 2) == pytest.approx(1_240.0)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            256, 128, 4096, 16, 4) == pytest.approx(9_520.0)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            256, 128, 4096, 16, 8) == pytest.approx(14_900.0)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            256, 128, 4096, 16, 16) == pytest.approx(28_220.0)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            256, 128, 4096, 16, 32) == pytest.approx(56_560.0)
        assert hw.aic_mte2_fia_mha_finite_drain_supports(
            256, 128, 4096, 16, 64)
        assert hw.aic_mte2_fia_mha_finite_drain_time_ns(
            256, 128, 4096, 16, 64) == pytest.approx(111_600.0)
        assert not hw.aic_mte2_fia_mha_finite_drain_supports(
            256, 128, 4096, 22, 64)
        assert not hw.aic_mte2_fia_mha_finite_drain_supports(
            256, 128, 4096, 21, 24)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_fia_mha_finite_drain_time_ns(
                256, 128, 4096, 22, 64)
        with pytest.raises(TypeError, match="inputs must be integers"):
            hw.aic_mte2_fia_mha_finite_drain_supports(
                1, 128, 128, 22, 2.0)

    def test_aic_mte2_fia_gqa_finite_drain_is_exact_and_fail_loud(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        service = hw.aic_mte2_fia_gqa_finite_drain_service
        reference = Path(service["reference_path"])

        assert service["model"] == "aic_mte2_fia_gqa_finite_drain_v1"
        assert service["active_aic_groups"] == (8, 16, 22, 24)
        assert service["reference_sha256"] == hashlib.sha256(
            reference.read_bytes()).hexdigest()
        interleaved_reference = Path(service["interleaved_reference_path"])
        assert service["interleaved_reference_sha256"] == hashlib.sha256(
            interleaved_reference.read_bytes()).hexdigest()
        group22_count16_reference = Path(
            service["group22_count16_reference_path"])
        assert service["group22_count16_reference_sha256"] == hashlib.sha256(
            group22_count16_reference.read_bytes()).hexdigest()
        group16_count16_reference = Path(
            service["group16_count16_reference_path"])
        assert service["group16_count16_reference_sha256"] == hashlib.sha256(
            group16_count16_reference.read_bytes()).hexdigest()
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 64, 512, 16, 2)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 64, 512, 16, 2) == pytest.approx(4710.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 64, 512, 16, 4)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 64, 512, 16, 4) == pytest.approx(7_080.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 8, 32)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 8, 32) == pytest.approx(44_420.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 8, 16)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 8, 16) == pytest.approx(22_260.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 16, 32)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 16, 32) == pytest.approx(55_460.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 16, 8)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 16, 8) == pytest.approx(14_480.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 16, 16)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 16, 16) == pytest.approx(28_220.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 22, 4)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 22, 4) == pytest.approx(10_260.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 22, 8)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 22, 8) == pytest.approx(18_000.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 128, 1024, 22, 16)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 22, 16) == pytest.approx(35_040.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_supports(
            128, 128, 1024, 22, 8)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            128, 128, 1024, 22, 8) == pytest.approx(10_400.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 24, 2) == pytest.approx(6_300.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 24, 20) == pytest.approx(47_920.0)
        assert hw.aic_mte2_fia_gqa_finite_drain_time_ns(
            256, 128, 1024, 24, 22) == pytest.approx(51_900.0)
        assert not hw.aic_mte2_fia_gqa_finite_drain_supports(
            256, 64, 512, 16, 3)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_mte2_fia_gqa_finite_drain_time_ns(
                256, 64, 512, 16, 3)
        with pytest.raises(TypeError, match="inputs must be integers"):
            hw.aic_mte2_fia_gqa_finite_drain_supports(
                256, 64, 512, 16, 2.0)
    def test_grouped_matmul_finite_stream_completion_is_domain_bounded(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = hw.aic_grouped_matmul_finite_stream_completion_service

        for path, digest in zip(
                mechanism["reference_paths"],
                mechanism["reference_sha256s"]):
            reference = Path(path)
            assert hashlib.sha256(reference.read_bytes()).hexdigest() == digest
        assert mechanism["measured_pairs_per_worker"] == (
            16, 32, 64, 128, 256, 1024, 4096, 8192)
        assert hw.aic_grouped_matmul_finite_stream_completion_time_ns(
            16, 64, 64, 256, 24, 4, 16) == pytest.approx(12430.0)
        assert hw.aic_grouped_matmul_finite_stream_completion_time_ns(
            16, 64, 64, 256, 24, 4, 128) == pytest.approx(99440.0)
        assert hw.aic_grouped_matmul_finite_stream_completion_time_ns(
            16, 64, 64, 256, 24, 4, 256) == pytest.approx(178440.0)
        assert hw.aic_grouped_matmul_finite_stream_completion_time_ns(
            16, 64, 64, 256, 24, 4, 512) == pytest.approx(354100.0)
        assert hw.aic_grouped_matmul_finite_stream_completion_time_ns(
            16, 64, 64, 256, 24, 4, 8192) == pytest.approx(5368820.0)
        with pytest.raises(ValueError, match="geometry is outside"):
            hw.aic_grouped_matmul_finite_stream_completion_time_ns(
                32, 64, 64, 256, 24, 4, 16)
        with pytest.raises(ValueError, match="length is outside"):
            hw.aic_grouped_matmul_finite_stream_completion_time_ns(
                16, 64, 64, 256, 24, 4, 8)
        assert hw.aic_grouped_matmul_finite_stream_completion_time_ns(
            16, 64, 64, 256, 24, 4, 200) == pytest.approx(143877.5)

    def test_grouped_matmul_k2880_wide_base_tail_is_domain_bounded(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = (
            hw.aic_grouped_matmul_k2880_wide_base_tail_completion_service)

        reference = Path(mechanism["reference_path"])
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            mechanism["reference_sha256"])
        assert mechanism["projection_k"] == 2880
        assert mechanism["base_m_values"] == (64, 128)
        assert mechanism["active_m_values"] == (16, 32)
        assert mechanism["measured_pairs_per_worker"] == (
            512, 1024, 3072, 6144)
        assert hw.aic_grouped_matmul_k2880_wide_base_tail_time_ns(
            64, 16, 2880, 64, 64, 256, 24, 4, 512) == pytest.approx(
                359320.0)
        assert hw.aic_grouped_matmul_k2880_wide_base_tail_time_ns(
            128, 32, 2880, 64, 64, 256, 24, 4, 2048) == pytest.approx(
                1213370.0)
        assert not hw.aic_grouped_matmul_k2880_wide_base_tail_supports(
            64, 16, 2048, 64, 64, 256, 24, 4, 512)
        assert not hw.aic_grouped_matmul_k2880_wide_base_tail_supports(
            64, 16, 2880, 64, 64, 256, 24, 4, 256)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_grouped_matmul_k2880_wide_base_tail_time_ns(
                32, 16, 2880, 64, 64, 256, 24, 4, 512)

    def test_grouped_matmul_k2880_output_tile_is_domain_bounded(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = (
            hw.aic_grouped_matmul_k2880_output_tile_completion_service)

        reference = Path(mechanism["reference_path"])
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            mechanism["reference_sha256"])
        assert mechanism["projection_k"] == 2880
        assert mechanism["base_m_values"] == (128,)
        assert mechanism["active_m_values"] == (16, 32, 48)
        assert mechanism["output_tile_pairs"] == 45
        assert hw.aic_grouped_matmul_k2880_output_tile_time_ns(
            128, 16, 2880, 64, 64, 256, 24, 4, 45) == pytest.approx(
                33585.0)
        assert hw.aic_grouped_matmul_k2880_output_tile_time_ns(
            128, 32, 2880, 64, 64, 256, 24, 4, 45) == pytest.approx(
                34005.0)
        assert hw.aic_grouped_matmul_k2880_output_tile_time_ns(
            128, 48, 2880, 64, 64, 256, 24, 4, 45) == pytest.approx(
                33230.0)
        assert not hw.aic_grouped_matmul_k2880_output_tile_supports(
            64, 16, 2880, 64, 64, 256, 24, 4, 45)
        assert not hw.aic_grouped_matmul_k2880_output_tile_supports(
            128, 16, 2880, 64, 64, 256, 24, 4, 44)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_grouped_matmul_k2880_output_tile_time_ns(
                128, 64, 2880, 64, 64, 256, 24, 4, 45)

    def test_grouped_matmul_step4_worker_sequence_is_domain_bounded(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = (
            hw.aic_grouped_matmul_step4_worker_sequence_completion_service)

        reference = Path(mechanism["reference_path"])
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            mechanism["reference_sha256"])
        assert mechanism["active_aic_groups"] == hw.num_compute_units == 24
        assert mechanism["roles"]["w1"][
            "measured_output_tiles_per_worker"] == (47, 64, 80, 96, 128, 192)
        assert mechanism["roles"]["w2"][
            "measured_output_tiles_per_worker"] == (95, 128, 160, 176, 192,
                                                       224, 240, 288, 320, 352,
                                                       384)
        assert hw.aic_grouped_matmul_step4_worker_sequence_time_ns(
            "w1", "bf16", 24, 128, 16, 4096, 1024, 256, 64,
            8, 8, 4, 4, 2, 2, 1, 47) == pytest.approx(1_716_600.0)
        assert hw.aic_grouped_matmul_step4_worker_sequence_time_ns(
            "w1", "bf16", 24, 128, 16, 4096, 1024, 256, 64,
            8, 8, 4, 4, 2, 2, 1, 48) == pytest.approx(
                1_757_474.1176470588)
        assert hw.aic_grouped_matmul_step4_worker_sequence_time_ns(
            "w2", "bf16", 24, 128, 16, 1024, 4096, 256, 64,
            8, 8, 4, 4, 2, 2, 1, 160) == pytest.approx(1_521_320.0)
        assert not hw.aic_grouped_matmul_step4_worker_sequence_supports(
            "w1", "bf16", 24, 128, 32, 4096, 1024, 256, 64,
            8, 8, 4, 4, 2, 2, 1, 47)
        assert not hw.aic_grouped_matmul_step4_worker_sequence_supports(
            "w1", "bf16", 24, 128, 16, 4096, 1024, 256, 64,
            8, 8, 4, 4, 2, 2, 1, 193)
        assert hw.aic_grouped_matmul_step4_worker_sequence_supports(
            "w2", "bf16", 24, 128, 16, 1024, 4096, 256, 64,
            8, 8, 4, 4, 2, 2, 1, 384)
        assert not hw.aic_grouped_matmul_step4_worker_sequence_supports(
            "w2", "bf16", 24, 128, 16, 1024, 4096, 256, 64,
            8, 8, 4, 4, 2, 2, 1, 385)
        with pytest.raises(ValueError, match="role must be"):
            hw.aic_grouped_matmul_step4_worker_sequence_supports(
                "unknown", "bf16", 24, 128, 16, 4096, 1024, 256, 64,
                8, 8, 4, 4, 2, 2, 1, 47)

    def test_grouped_matmul_short_source_prelude_is_domain_bounded(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = hw.aic_grouped_matmul_short_source_prelude_control_service

        reference = Path(mechanism["reference_path"])
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            mechanism["reference_sha256"])
        assert mechanism["active_aic_groups"] == hw.num_compute_units == 24
        assert mechanism["k_rounds"] == (45, 96, 256)
        arguments = (
            "bf16", 24, 16, 16, 2880, 256, 64, 120, 8, 60, 4, 2, 2, 1,
        )
        assert hw.aic_grouped_matmul_short_source_prelude_control_supports(
            *arguments)
        assert hw.aic_grouped_matmul_short_source_prelude_control_time_ns(
            *arguments) == pytest.approx(696.112060546875)
        assert hw.aic_grouped_matmul_short_source_prelude_control_time_ns(
            "bf16", 24, 16, 16, 6144, 256, 64, 120, 8, 60, 4,
            2, 2, 1) == pytest.approx(1511.66748046875)
        assert hw.aic_grouped_matmul_short_source_prelude_control_time_ns(
            "bf16", 24, 16, 16, 16384, 256, 64, 120, 8, 60, 4,
            2, 2, 1) == pytest.approx(4089.44580078125)
        assert not hw.aic_grouped_matmul_short_source_prelude_control_supports(
            "bf16", 24, 16, 16, 2880, 256, 64, 8, 8, 4, 4, 2, 2, 1)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_grouped_matmul_short_source_prelude_control_time_ns(
                "bf16", 24, 16, 16, 4032, 256, 64, 120, 8, 60, 4,
                2, 2, 1)
        tail_mechanism = (
            hw.aic_grouped_matmul_short_source_tail_prelude_control_service)
        tail_reference = Path(tail_mechanism["reference_path"])
        assert hashlib.sha256(tail_reference.read_bytes()).hexdigest() == (
            tail_mechanism["reference_sha256"])
        assert tail_mechanism["logical_n_values"] == (64, 128)
        for logical_n in (64, 128):
            tail_arguments = (
                "bf16", 24, 16, 16, logical_n, 2880, 256, 64,
                120, 8, 60, 4, 2, 2, 1)
            assert hw.\
                aic_grouped_matmul_short_source_tail_prelude_control_supports(
                    *tail_arguments)
            assert hw.\
                aic_grouped_matmul_short_source_tail_prelude_control_time_ns(
                    *tail_arguments) == pytest.approx(696.112060546875)
        assert not hw.\
            aic_grouped_matmul_short_source_tail_prelude_control_supports(
                "bf16", 24, 16, 16, 32, 2880, 256, 64,
                120, 8, 60, 4, 2, 2, 1)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_grouped_matmul_short_source_tail_prelude_control_time_ns(
                "bf16", 24, 16, 16, 64, 6144, 256, 64,
                120, 8, 60, 4, 2, 2, 1)

    def test_grouped_matmul_m32_short_source_prelude_is_domain_bounded(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = (
            hw.aic_grouped_matmul_m32_short_source_prelude_control_service)

        reference = Path(mechanism["reference_path"])
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            mechanism["reference_sha256"])
        assert mechanism["active_aic_groups"] == hw.num_compute_units == 24
        assert mechanism["logical_n_values"] == (64, 128, 256)
        arguments = (
            "bf16", 24, 32, 32, 64, 2880, 256, 64,
            56, 8, 28, 4, 2, 2, 1,
        )
        assert hw.\
            aic_grouped_matmul_m32_short_source_prelude_control_supports(
                *arguments)
        assert hw.\
            aic_grouped_matmul_m32_short_source_prelude_control_time_ns(
                *arguments) == pytest.approx(705.279541015625)
        for logical_n in (128, 256):
            assert hw.\
                aic_grouped_matmul_m32_short_source_prelude_control_time_ns(
                    "bf16", 24, 32, 32, logical_n, 2880, 256, 64,
                    56, 8, 28, 4, 2, 2, 1) == pytest.approx(
                        705.279541015625)
        assert not hw.\
            aic_grouped_matmul_m32_short_source_prelude_control_supports(
                "bf16", 24, 16, 16, 64, 2880, 256, 64,
                120, 8, 60, 4, 2, 2, 1)
        assert not hw.\
            aic_grouped_matmul_m32_short_source_prelude_control_supports(
                "bf16", 24, 32, 32, 32, 2880, 256, 64,
                56, 8, 28, 4, 2, 2, 1)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.\
                aic_grouped_matmul_m32_short_source_prelude_control_time_ns(
                    "bf16", 24, 32, 32, 64, 2880, 256, 64,
                    56, 8, 4, 4, 2, 2, 1)

    def test_grouped_matmul_entry_metadata_pair_is_domain_bounded(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        mechanism = hw.aic_grouped_matmul_entry_metadata_pair_service

        reference = Path(mechanism["reference_path"])
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            mechanism["reference_sha256"])
        assert mechanism["active_aic_groups"] == hw.num_compute_units == 24
        assert mechanism["gmm_base_params_bytes"] == 1672
        assert mechanism["tcube_tiling_bytes"] == 200
        arguments = ("bf16", 24, "grouped_matmul", 6144, 16384)
        assert hw.aic_grouped_matmul_entry_metadata_pair_supports(*arguments)
        assert hw.aic_grouped_matmul_entry_metadata_pair_time_ns(
            *arguments) == pytest.approx(667.223307291667)
        assert not hw.aic_grouped_matmul_entry_metadata_pair_supports(
            "fp16", 24, "grouped_matmul", 6144, 16384)
        assert not hw.aic_grouped_matmul_entry_metadata_pair_supports(
            "bf16", 23, "grouped_matmul", 6144, 16384)
        assert not hw.aic_grouped_matmul_entry_metadata_pair_supports(
            "bf16", 24, "matmul", 6144, 16384)
        assert not hw.aic_grouped_matmul_entry_metadata_pair_supports(
            "bf16", 24, "grouped_matmul", 7168, 2048)
        assert not hw.aic_grouped_matmul_entry_metadata_pair_supports(
            "bf16", 24, "grouped_matmul", 2048, 7168)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_grouped_matmul_entry_metadata_pair_time_ns(
                "bf16", 24, "grouped_matmul", 7168, 2048)

    def test_fia_aiv_workspace_transfer_uses_measured_mode_and_group_axes(
            self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.fia_aiv_workspace_transfer_time_ns(
            "vec1_m4_s512_ordered_chain", 8) == pytest.approx(
                303.363444010417)
        assert hw.fia_aiv_workspace_transfer_time_ns(
            "vec1_m4_s1024_ordered_chain", 16) == pytest.approx(
                347.95166015625)
        assert hw.fia_aiv_workspace_transfer_time_ns(
            "vec2_m4_final_ordered_chain", 16) == pytest.approx(
                287.39013671875)
        with pytest.raises(ValueError, match="unknown FIA AIV workspace"):
            hw.fia_aiv_workspace_transfer_time_ns("unknown", 8)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.fia_aiv_workspace_transfer_time_ns(
                "vec1_m4_s512_ordered_chain", 4)
        with pytest.raises(TypeError, match="must be an integer"):
            hw.fia_aiv_workspace_transfer_time_ns(
                "vec1_m4_s512_ordered_chain", 8.0)

    def test_aic_mte1_load_pair_service_uses_independent_evidence(self):
        import json

        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "aic_mte1_load_pair_cann_8_5_1.json"))
        mechanism = hw.aic_mte1_load_pair_completion_service

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert reference["mechanism_id"] == mechanism["model"]
        assert tuple(reference["bounds"]["stage_kinds"]) == mechanism[
            "stage_kinds"]
        assert tuple(reference["bounds"]["head_dims"]) == mechanism[
            "head_dims"]
        assert tuple(reference["bounds"]["active_aic_groups"]) == mechanism[
            "active_aic_groups"]
        for stage_kind in mechanism["stage_kinds"]:
            assert tuple(reference["parameters_ns"][stage_kind]) == (
                mechanism["completion_ns_by_stage"][stage_kind])
        for name in (
                "measurement_baseline_commit", "source_commit",
                "source_snapshot_sha256", "binary_sha256", "raw_path",
                "raw_sha256"):
            assert reference["provenance"][name] == mechanism[name]
        assert hw.aic_mte1_load_pair_time_ns("mm1", 64, 8) == (
            pytest.approx(648.733333332))
        assert hw.aic_mte1_load_pair_time_ns("mm2", 128, 8) == (
            pytest.approx(1293.333333336))
        with pytest.raises(ValueError, match="stage kind.*outside"):
            hw.aic_mte1_load_pair_time_ns("unknown", 64, 1)
        with pytest.raises(ValueError, match="head dimension.*outside"):
            hw.aic_mte1_load_pair_time_ns("mm1", 256, 1)
        with pytest.raises(TypeError, match="must be an integer"):
            hw.aic_mte1_load_pair_time_ns("mm1", 64, 1.0)

    def test_grouped_matmul_mte1_pair_service_uses_independent_evidence(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "grouped_matmul_mte1_pair_cann_8_5_1.json"))
        geometry_reference = json.load(open(
            "microbench/ascend_910b/references/"
            "grouped_matmul_mte1_pair_geometry_cann_8_5_1.json"))
        mechanism = hw.aic_grouped_matmul_mte1_pair_completion_service

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert geometry_reference[
            "production_or_evaluation_duration_consumed"] is False
        assert reference["mechanism_id"].removesuffix("_v1") == (
            mechanism["model"].removesuffix("_v2"))
        assert tuple(reference["bounds"]["m_rows"]) == mechanism["m_rows"]
        assert reference["bounds"]["n_columns"] == mechanism["n_columns"]
        assert reference["bounds"]["k_columns"] == mechanism["k_columns"]
        n_values = tuple(
            row["n_columns"] for row in geometry_reference["measurements"])
        assert min(n_values) == mechanism["minimum_n_columns"]
        assert max(n_values) == mechanism["maximum_n_columns"]
        assert geometry_reference["affine_n_service"][
            "ns_per_n_column"] == mechanism["n_column_service_ns"]
        assert tuple(reference["parameters_ns"]) == mechanism[
            "completion_ns_by_m_rows"]
        assert hw.aic_grouped_matmul_mte1_pair_time_ns(
            16, 256, 64, 2) == pytest.approx(298.893229166)
        assert hw.aic_grouped_matmul_mte1_pair_time_ns(
            128, 256, 64, 1) == pytest.approx(184.479166667)
        assert hw.aic_grouped_matmul_mte1_pair_affine_extension_time_ns(
            48, 96, 64, 128) == pytest.approx(9027.31481490489)
        assert hw.aic_grouped_matmul_mte1_pair_affine_extension_time_ns(
            48, 80, 64, 128) is None
        with pytest.raises(ValueError, match="M rows.*outside"):
            hw.aic_grouped_matmul_mte1_pair_time_ns(24, 256, 64, 1)
        with pytest.raises(ValueError, match="N/K geometry.*outside"):
            hw.aic_grouped_matmul_mte1_pair_time_ns(16, 96, 64, 1)
        with pytest.raises(TypeError, match="must be an integer"):
            hw.aic_grouped_matmul_mte1_pair_time_ns(16, 256, 64, 1.0)

    def test_matmul_v3_split_k_resident_tile_uses_independent_evidence(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "aic_matmul_v3_split_k_resident_tile_completion_cann_8_5_1.json"))
        mechanism = hw.aic_matmul_v3_split_k_resident_tile_service

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert mechanism["reference_sha256"] == (
            "f04c7d29f170cca8bf625363dbca17ba0e6f5192a3da854e375a922cd4b6abc6")
        assert hw.aic_matmul_v3_split_k_resident_tile_time_ns(
            "bf16", 24, 128, 128, 384) == pytest.approx(853.359375)
        assert hw.aic_matmul_v3_split_k_resident_tile_time_ns(
            "bf16", 24, 128, 128, 256) == pytest.approx(
                568.9192708333334)
        assert hw.aic_matmul_v3_split_k_resident_tile_time_ns(
            "bf16", 22, 128, 128, 512) == pytest.approx(
                1137.8255208333335)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_matmul_v3_split_k_resident_tile_time_ns(
                "bf16", 24, 128, 128, 640)

    def test_matmul_v3_asw_mte2_pair_stream_uses_independent_evidence(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "matmul_v3_asw_mte2_pair_cann_8_5_1.json"))
        mechanism = hw.aic_matmul_v3_asw_mte2_pair_stream_service

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert mechanism["minimum_pairs_per_worker"] == reference[
            "measurement"]["epochs_per_aic"]
        assert mechanism["continuous_ns_per_pair"] == reference[
            "l2_resident_256x128_pair_service"]["pair_median_ns"]
        assert hw.aic_matmul_v3_asw_mte2_pair_time_ns(
            256, 128, 64, 24, 2560) == pytest.approx(
                2560 * 386.381161542)
        with pytest.raises(ValueError, match="shorter than"):
            hw.aic_matmul_v3_asw_mte2_pair_time_ns(
                256, 128, 64, 24, 1365)
        with pytest.raises(ValueError, match="outside the measured domain"):
            hw.aic_matmul_v3_asw_mte2_pair_time_ns(
                128, 256, 64, 24, 2560)

    def test_aic_fixpipe_full_tile_service_uses_source_geometry(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.aic_fixpipe_full_tile_time_ns(
            4, 128, 8192) == pytest.approx(
                138.708244067238 + 4 * 4.026384425584118)
        assert hw.aic_fixpipe_full_tile_time_ns(
            4, 128, 192) == pytest.approx(
                138.708244067238
                + 4 * 4.026384425584118
                + 35.75913489709218)
        assert hw.aic_fixpipe_full_tile_time_ns(
            4, 128, 160) == pytest.approx(
                138.708244067238
                + 4 * 4.026384425584118
                + 92.81642656375887)
        assert hw.aic_fixpipe_full_tile_time_ns(
            64, 128, 512) == pytest.approx(
                145.53617571059428 + 64 * 2.3778161337209314)
        with pytest.raises(ValueError, match="rows are outside"):
            hw.aic_fixpipe_full_tile_time_ns(2, 128, 128)
        with pytest.raises(ValueError, match="columns are outside"):
            hw.aic_fixpipe_full_tile_time_ns(4, 64, 128)
        with pytest.raises(ValueError, match="stride is outside"):
            hw.aic_fixpipe_full_tile_time_ns(4, 128, 130)

    def test_fia_block_reduce_sum_uses_source_branch_geometry(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.fia_block_reduce_sum_time_ns(
            32, 256, 32) == pytest.approx(281.16357421875)
        assert hw.fia_block_reduce_sum_time_ns(
            16, 512, 32) == pytest.approx(151.826171875)
        assert hw.fia_block_reduce_sum_time_ns(
            8, 1024, 32) == pytest.approx(918.9153869553999)
        assert hw.fia_block_reduce_sum_time_ns(
            16, 384, 32) == pytest.approx(368.17177886126075)
        with pytest.raises(ValueError, match="8-row-aligned domain"):
            hw.fia_block_reduce_sum_time_ns(7, 128, 32)
        with pytest.raises(ValueError, match="measured UB tile"):
            hw.fia_block_reduce_sum_time_ns(64, 256, 32)
        with pytest.raises(ValueError, match="exact measured row chunk"):
            hw.fia_block_reduce_sum_time_ns(24, 512, 16)

    def test_fia_block_reduce_max_uses_independent_source_service(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.fia_block_reduce_max_time_ns(
            32, 256, 32) == pytest.approx(269.24658203125)
        assert hw.fia_block_reduce_max_time_ns(
            16, 512, 32) == pytest.approx(140.32714843749994)
        assert hw.fia_block_reduce_max_time_ns(
            8, 1024, 32) == pytest.approx(795.2337186043231)
        assert hw.fia_block_reduce_max_time_ns(
            16, 384, 32) == pytest.approx(323.4257259636728)
        with pytest.raises(ValueError, match="8-row-aligned domain"):
            hw.fia_block_reduce_max_time_ns(7, 128, 32)
        with pytest.raises(ValueError, match="measured UB tile"):
            hw.fia_block_reduce_max_time_ns(64, 256, 32)
        with pytest.raises(ValueError, match="exact measured row chunk"):
            hw.fia_block_reduce_max_time_ns(24, 512, 16)

    def test_fia_brcb_uses_independent_source_service(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")
        reference = json.load(open(
            "microbench/ascend_910b/references/"
            "fia_brcb_cann_8_5_1.json"))

        assert reference[
            "production_or_evaluation_duration_consumed"] is False
        assert reference["model"] == hw.fia_brcb_service["model"]
        assert hw.fia_brcb_time_ns(8) == pytest.approx(9.9938)
        assert hw.fia_brcb_time_ns(16) == pytest.approx(10.549)
        assert hw.fia_brcb_time_ns(32) == pytest.approx(11.6594)
        assert hw.fia_brcb_time_ns(64) == pytest.approx(13.8802)
        with pytest.raises(ValueError, match="8-row-aligned domain"):
            hw.fia_brcb_time_ns(7)
        with pytest.raises(ValueError, match="8-row-aligned domain"):
            hw.fia_brcb_time_ns(72)
        with pytest.raises(TypeError, match="must be an integer"):
            hw.fia_brcb_time_ns(16.0)

    def test_task_kind_dispatch_selects_exact_npu_floor(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.empty_kernel_duration_for_task("aic", 8, 1) == 1040
        assert hw.empty_kernel_duration_for_task("aiv", 8, 1) == 1080
        assert hw.empty_kernel_duration_for_task("mixed", 8, 1) == 4960
        assert hw.task_duration_floor_for_task("aic", 8, 1) == 1040
        assert hw.task_duration_floor_for_task(
            "aiv", 8, 1) == pytest.approx(1627.6295411781734)
        assert hw.task_duration_floor_for_task("mixed", 8, 1) == 4960
        with pytest.raises(ValueError, match="block-size placeholder one"):
            hw.empty_kernel_duration_for_task("aic", 8, 2)

    def test_complete_softmax_algorithm_curve_is_not_a_hardware_parameter(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert not hasattr(hw, "aiv_softmax_vector_service_lut")
        assert not hasattr(hw, "aiv_softmax_vector_service_ns")

    def test_unproven_refetch_and_hccs_latency_parameters_are_disabled(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.memory_subsystem["gemm_l1_refetch_tile_rows"] is None
        assert hw.memory_subsystem["mte2_per_core_peak_gbs"] is None
        assert hw.dram_latency_ns is None
        assert hw.l2_latency_ns is None
        assert hw.cacheline_bytes is None
        assert hw.interconnect["nvlink_bw_bytes_s"] == pytest.approx(
            19.44395e9)
        assert hw.interconnect["nvlink_signal_latency_ns"] is None
        assert hw.interconnect["nccl_alpha_ns"] is None

    def test_gm_uses_shared_directional_capacities(self):
        hw = HWSpec("configs/hardware/ascend_910b.yaml")

        assert hw.dram_arbitration_mode == (
            "shared_directional_capacity")
        assert hw.dram_read_capacity_per_clk == 768
        assert hw.dram_write_capacity_per_clk == 768
        read = hw.dram_capacity_for_access(
            "read",
            active_units=hw.num_compute_units,
            sm_freq_hz=hw.nominal_sm_freq_hz,
            dram_freq_hz=hw.nominal_dram_freq_hz,
        )
        write = hw.dram_capacity_for_access(
            "write",
            active_units=hw.num_compute_units,
            sm_freq_hz=hw.nominal_sm_freq_hz,
            dram_freq_hz=hw.nominal_dram_freq_hz,
        )
        assert read == write


class TestHWSpecPIM:
    def test_loads_without_error(self):
        hw = HWSpec(
            "tests/kernel/hardware_specs/samsung_hbm_pim_fixture.yaml")
        assert hw is not None
        assert not hasattr(hw, "mshr_concurrency")


class TestHWSpecInvalidConfig:
    @staticmethod
    def _a800_tensor_component(cfg):
        return cfg["components"][0]["children"][0]["children"][0]

    def test_missing_kernel_memory_model_raises_schema_error(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        del cfg["features"]["kernel_memory_model"]
        path = tmp_path / "missing_memory_model.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="kernel_memory_model"):
            HWSpec(str(path))

    def test_unknown_kernel_memory_model_raises_value_error(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["features"]["kernel_memory_model"] = "card_name_switch"
        path = tmp_path / "unknown_memory_model.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="kernel_memory_model"):
            HWSpec(str(path))

    def test_gpu_memory_model_requires_store_completion_scope(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        del cfg["features"]["gpu_store_completion_scope"]
        path = tmp_path / "missing_store_completion_scope.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="gpu_store_completion_scope"):
            HWSpec(str(path))

    def test_target_memory_model_requires_explicit_arbitration_mode(
            self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        del cfg["memory_subsystem"]["dram_arbitration_mode"]
        path = tmp_path / "missing_arbitration.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="dram_arbitration_mode"):
            HWSpec(str(path))

    def test_directional_mode_requires_both_capacities(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["memory_subsystem"]["dram_write_capacity_per_clk"] = None
        path = tmp_path / "missing_write_capacity.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="requires explicit read and write"):
            HWSpec(str(path))

    def test_directional_read_capacity_must_match_resource(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["memory_subsystem"]["dram_read_capacity_per_clk"] = 1000
        path = tmp_path / "mismatched_read_capacity.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(
                ValueError, match="canonical DRAM resource capacity"):
            HWSpec(str(path))

    @pytest.mark.parametrize(
        "field,value,message",
        [
            (
                "l2_hit_latency_local_cycles", 200,
                "requires l2_local_fraction",
            ),
            (
                "dram_miss_extra_cycles", 100,
                "L2 hit plus DRAM miss extra cycles must equal dram_latency_ns",
            ),
        ],
    )
    def test_gpu_mshr_latency_chain_must_be_physically_consistent(
            self, tmp_path, field, value, message):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["memory_subsystem"][field] = value
        path = tmp_path / f"inconsistent_{field}.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match=message):
            HWSpec(str(path))

    def test_duration_calibration_feature_is_rejected(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["features"]["kernel_calibration_model"] = "cann_aic_aiv_v1"
        path = tmp_path / "obsolete_calibration.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="unknown=.*kernel_calibration_model"):
            HWSpec(str(path))

    def test_npu_empty_task_lut_rejects_sparse_interpolation(self, tmp_path):
        import yaml

        with open("configs/hardware/ascend_910b.yaml") as stream:
            cfg = yaml.safe_load(stream)
        del cfg["scheduling"]["aiv_empty_kernel_duration_lut"][
            "block_dims"][7]
        del cfg["scheduling"]["aiv_empty_kernel_duration_lut"][
            "duration_ns"][7]
        path = tmp_path / "sparse_aiv_launch_floor.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="every integer"):
            HWSpec(str(path))

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("submit_ns", 0.0, "finite and positive"),
            ("descriptor_ns_per_aiv", -1.0, "finite and positive"),
            ("maximum_aiv_block_dim", 47, "physical Vector-core count"),
        ),
    )
    def test_nonempty_aiv_task_model_rejects_invalid_services(
            self, tmp_path, field, value, message):
        import yaml

        with open("configs/hardware/ascend_910b.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["aiv_nonempty_task_latency_model"][field] = value
        path = tmp_path / f"invalid_nonempty_aiv_{field}.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match=message):
            HWSpec(str(path))

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        (
            ("submit_ns", 0.0, "finite and positive"),
            ("descriptor_ns_per_aiv", -1.0, "finite and positive"),
            ("maximum_aiv_block_dim", 47, "physical Vector-core count"),
        ),
    )
    def test_aiv_memory_task_model_rejects_invalid_services(
            self, tmp_path, field, value, message):
        import yaml

        with open("configs/hardware/ascend_910b.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["aiv_memory_task_latency_model"] = {
            "model": "isolated_aiv_memory_task_floor_v1",
            "duration_unit": "ns_per_complete_nonempty_aiv_task",
            "timer_domain": "msprof_task_time_complete_task",
            "software_revision": "cann_8.5.1",
            "minimum_aiv_block_dim": 1,
            "maximum_aiv_block_dim": 48,
            "submit_ns": 3000.0,
            "descriptor_ns_per_aiv": 50.0,
        }
        cfg["scheduling"]["aiv_memory_task_latency_model"][field] = value
        path = tmp_path / f"invalid_aiv_memory_task_{field}.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match=message):
            HWSpec(str(path))

    def test_mixed_empty_task_lut_requires_one_to_two_ratio(self, tmp_path):
        import yaml

        with open("configs/hardware/ascend_910b.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg["scheduling"]["mixed_empty_kernel_duration_lut"][
            "aiv_block_dim_ratio"] = 1
        path = tmp_path / "wrong_mixed_task_ratio.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="must be two"):
            HWSpec(str(path))

    def test_resource_leaf_requires_explicit_scope(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        del self._a800_tensor_component(cfg)["scope"]
        path = tmp_path / "missing_scope.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="requires an explicit scope"):
            HWSpec(str(path))

    def test_resource_leaf_requires_explicit_type(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        del self._a800_tensor_component(cfg)["type"]
        path = tmp_path / "missing_type.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="requires an explicit type"):
            HWSpec(str(path))

    def test_variant_component_requires_explicit_primary(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        del self._a800_tensor_component(cfg)["primary_variant"]
        path = tmp_path / "missing_primary_variant.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="requires a non-empty primary_variant"):
            HWSpec(str(path))

    def test_primary_variant_owns_canonical_resource_role(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        self._a800_tensor_component(cfg)["primary_variant"] = "bf16"
        path = tmp_path / "bf16_primary_variant.yaml"
        path.write_text(yaml.safe_dump(cfg))

        hardware = HWSpec(str(path))
        assert hardware.resource_by_role("compute_tensor").name == "tensor_core_bf16"

    def test_removed_mshr_fields_are_not_exported(self, tmp_path):
        import yaml

        src = "configs/hardware/nvidia_a800.yaml"
        with open(src) as f:
            cfg = yaml.safe_load(f)
        dst = tmp_path / "broken.yaml"
        with open(dst, "w") as f:
            yaml.dump(cfg, f)

        hardware = HWSpec(str(dst))
        assert not hasattr(hardware, "mshr_concurrency")
        assert not hasattr(hardware, "dram_outstanding_per_sm")

    def test_removed_mshr_field_fails_as_unknown(self, tmp_path):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as f:
            cfg = yaml.safe_load(f)

        cfg["memory_subsystem"]["mshr_concurrency"] = 0

        dst = tmp_path / "zero_mshr.yaml"
        with open(dst, "w") as f:
            yaml.dump(cfg, f)

        with pytest.raises(ValueError, match="unknown fields.*mshr_concurrency"):
            HWSpec(str(dst))

    @pytest.mark.parametrize(
        "section, field",
        [
            ("clock_domains", "sm"),
            ("memory_subsystem", "dram_latency_ns"),
            ("scheduling", "dispatch_overhead_ns"),
            ("interconnect", "pcie_bw_GB_s"),
        ],
    )
    def test_non_finite_hardware_scalar_fails_loudly(
            self, tmp_path, section, field):
        import math
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg[section][field] = math.nan
        path = tmp_path / "non_finite.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(ValueError, match="finite"):
            HWSpec(str(path))

    @pytest.mark.parametrize(
        "section, field, value, message",
        [
            ("clock_domains", "sm", "1410", "frequency must be numeric"),
            ("memory_subsystem", "cacheline_bytes", 128.5,
             "integer fields must be integers"),
            ("scheduling", "num_compute_units", 108.5,
             "must be an integer"),
            ("features", "cluster_size", 1.5, "must be an integer"),
            ("features", "kernel_memory_model", 1, "must be a string"),
            ("interconnect", "pcie_bw_GB_s", "32",
             "must be numeric or null"),
        ],
    )
    def test_hardware_scalar_type_errors_fail_loudly(
            self, tmp_path, section, field, value, message):
        import yaml

        with open("configs/hardware/nvidia_a800.yaml") as stream:
            cfg = yaml.safe_load(stream)
        cfg[section][field] = value
        path = tmp_path / "wrong_type.yaml"
        path.write_text(yaml.safe_dump(cfg))

        with pytest.raises(TypeError, match=message):
            HWSpec(str(path))
