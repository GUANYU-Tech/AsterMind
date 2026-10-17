"""Focused contracts for measured FusedMoE source-depth services."""

import math

import pytest

import src.kernel.predictors.fused_moe as fused_moe_predictor
from src.kernel.formula import PerformanceFormula
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.model_contract import (
    SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
)


def _projection_n(shape, variant):
    widths = {
        "w1": 2 * shape["intermediate"],
        "w2": shape["hidden"],
    }
    if variant not in widths:
        raise ValueError(f"unsupported grouped-MoE role: {variant!r}")
    return widths[variant]


@pytest.mark.parametrize((
    "experts", "topk", "hidden", "intermediate", "counts",
    "expected_topology", "variant", "expected_service",
    "expected_capacity",
), (
    (
        8, 2, 4096, 14336,
        (2,) + (1,) * 6 + (0,), (7, 8), "w1",
        "nvidia_bm16_e8_7_active_8_max_fused_moe_w1_operand_pipeline",
        1831031427428.9458,
    ),
    (
        8, 2, 4096, 14336,
        (2,) + (1,) * 6 + (0,), (7, 8), "w2",
        "nvidia_bm16_e8_7_active_8_max_fused_moe_w2_operand_pipeline",
        1704528963555.0315,
    ),
    (
        8, 2, 6144, 16384,
        (2,) + (1,) * 6 + (0,), (7, 8), "w1",
        "nvidia_bm16_e8_h6144_7_active_8_max_fused_moe_w1_operand_pipeline",
        1800596991973.8672,
    ),
    (
        8, 2, 6144, 16384,
        (2,) + (1,) * 6 + (0,), (7, 8), "w2",
        "nvidia_bm16_e8_h6144_7_active_8_max_fused_moe_w2_operand_pipeline",
        1537092890139.727,
    ),
    (
        128, 16, 2048, 768,
        (17,) * 49 + (16,) * 30 + (15,) * 49, (177, 248), "w1",
        "nvidia_bm16_e128_177_active_248_max_fused_moe_w1_operand_pipeline",
        2975495908026.122,
    ),
    (
        128, 16, 2048, 768,
        (17,) * 49 + (16,) * 30 + (15,) * 49, (177, 248), "w2",
        "nvidia_bm16_e128_177_active_248_max_fused_moe_w2_operand_pipeline",
        2803015376984.658,
    ),
))
def test_a800_bm16_exact_source_depth_is_role_specific(
        experts, topk, hidden, intermediate, counts, expected_topology,
        variant, expected_service, expected_capacity):
    shape = {
        "tokens": 4 if experts == 8 else 128,
        "experts": experts,
        "topk": topk,
        "hidden": hidden,
        "intermediate": intermediate,
    }
    assert len(counts) == experts
    assert sum(counts) == shape["tokens"] * topk
    attributes = {
        "tokens_per_expert": counts,
        "expert_parallel_size": 1,
    }
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FUSED_MOE,
        shape,
        "bf16",
        variant,
        attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        launch=launch,
        attributes={
            "tokens_per_expert": counts,
            "local_tokens_per_expert": counts,
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": tuple(range(experts)),
            "expert_weight_tile_rows": None,
        },
    )

    tile_m = launch.tiling.tile_shape["m"]
    tile_n = launch.tiling.tile_shape["n"]
    projection_n = _projection_n({
        "hidden": hidden, "intermediate": intermediate}, variant)
    assert (
        sum(math.ceil(count / tile_m) for count in counts if count),
        launch.grid_size // math.ceil(projection_n / tile_n),
    ) == expected_topology
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    assert resource_input.pipeline_plan is None
    operand_path, = tuple(
        path for path in resource_input.memory_paths
        if path.name == "operand_read")
    demand, = operand_path.access_class_demands
    traffic = fused_moe_predictor._gpu_traffic_components(kernel, hardware)

    assert demand.service == expected_service
    assert demand.replaces_ingress_path_service
    assert demand.logical_bytes == pytest.approx(
        traffic.activation_l2_read_bytes + traffic.weight_l2_read_bytes)
    prediction = PerformanceFormula(hardware).predict_kernel(resource_input)
    assert prediction.t_per_group[
        f"memory_access:{expected_service}"
    ] == pytest.approx(demand.logical_bytes / expected_capacity)


@pytest.mark.parametrize(("tokens", "expected_service"), (
    (8192, "grouped_fused_moe_operand_pipeline"),
    (16384, "nvidia_e128_deep_fused_moe_w2_operand_pipeline"),
))
def test_a800_e128_deep_w2_starts_after_two_source_waves(
        tokens, expected_service):
    shape = {
        "tokens": tokens,
        "experts": 128,
        "topk": 8,
        "hidden": 4096,
        "intermediate": 1536,
    }
    counts = (tokens * shape["topk"] // shape["experts"],) * 128
    attributes = {
        "tokens_per_expert": counts,
        "expert_parallel_size": 1,
    }
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FUSED_MOE,
        shape,
        "bf16",
        "w2",
        attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="w2",
        launch=launch,
        attributes={
            "tokens_per_expert": counts,
            "local_tokens_per_expert": counts,
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": tuple(range(shape["experts"])),
            "expert_weight_tile_rows": None,
        },
    )

    tile_n = launch.tiling.tile_shape["n"]
    maximum_m_tiles = launch.grid_size // math.ceil(
        shape["hidden"] / tile_n)
    assert maximum_m_tiles == (1150 if tokens == 8192 else 2174)
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    operand_path, = tuple(
        path for path in resource_input.memory_paths
        if path.name == "operand_read")
    demands = tuple(operand_path.access_class_demands)
    services = {demand.service for demand in demands}
    assert expected_service in services
    assert (
        "nvidia_e128_deep_fused_moe_w2_operand_pipeline" in services
    ) is (tokens == 16384)
    if tokens == 16384:
        demand, = demands
        traffic = fused_moe_predictor._gpu_traffic_components(
            kernel, hardware)
        assert demand.replaces_ingress_path_service
        assert demand.logical_bytes == pytest.approx(
            traffic.activation_l2_read_bytes
            + traffic.weight_l2_read_bytes)
        prediction = PerformanceFormula(hardware).predict_kernel(
            resource_input)
        assert prediction.t_per_group[
            f"memory_access:{expected_service}"
        ] == pytest.approx(
            demand.logical_bytes / 2031525634020.1377)


@pytest.mark.parametrize(("variant", "expected_capacity"), (
    ("w1", 3588120216141.3794),
    ("w2", 2102134847962.7576),
))
def test_a800_e128_h2048_topk16_uses_exact_source_census(
        variant, expected_capacity):
    shape = {
        "tokens": 32768,
        "experts": 128,
        "topk": 16,
        "hidden": 2048,
        "intermediate": 768,
    }
    census = (
        fused_moe_predictor
        ._NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_M_TILE_CENSUS)
    counts = [(m_tiles - 1) * 64 + 1 for m_tiles in census]
    remaining_rows = shape["tokens"] * shape["topk"] - sum(counts)
    for index, _ in enumerate(census):
        added_rows = min(remaining_rows, 63)
        counts[index] += added_rows
        remaining_rows -= added_rows
    counts = tuple(counts)
    assert remaining_rows == 0
    assert sum(counts) == shape["tokens"] * shape["topk"]
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    attributes = {
        "tokens_per_expert": counts,
        "expert_parallel_size": 1,
    }
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FUSED_MOE,
        shape,
        "bf16",
        variant,
        attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        launch=launch,
        attributes={
            "tokens_per_expert": counts,
            "local_tokens_per_expert": counts,
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": tuple(range(shape["experts"])),
            "expert_weight_tile_rows": None,
        },
    )

    projection_n = _projection_n(shape, variant)
    assert (
        sum(math.ceil(count / 64) for count in counts),
        launch.grid_size // math.ceil(projection_n / 64),
    ) == (8250, 8318)
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    operand_path, = tuple(
        path for path in resource_input.memory_paths
        if path.name == "operand_read")
    demand, = operand_path.access_class_demands
    expected_service = (
        fused_moe_predictor
        ._NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT[
            variant])
    assert demand.service == expected_service
    assert demand.replaces_ingress_path_service
    traffic = fused_moe_predictor._gpu_traffic_components(kernel, hardware)
    assert demand.logical_bytes == pytest.approx(
        traffic.activation_l2_read_bytes + traffic.weight_l2_read_bytes)
    prediction = PerformanceFormula(hardware).predict_kernel(resource_input)
    assert prediction.t_per_group[
        f"memory_access:{expected_service}"
    ] == pytest.approx(demand.logical_bytes / expected_capacity)


def test_a800_e256_h7168_w2_uses_2166_active_source_program():
    """The post-activation W2 source has one row per routed assignment."""

    shape = {
        "tokens": 16384,
        "experts": 256,
        "topk": 8,
        "hidden": 7168,
        "intermediate": 2048,
    }
    counts = (449,) * 119 + (512,) * 22 + (521,) + (576,) * 111 + (640,) * 3
    assert len(counts) == shape["experts"]
    assert sum(counts) == shape["tokens"] * shape["topk"]
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    attributes = {
        "tokens_per_expert": counts,
        "expert_parallel_size": 1,
    }
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FUSED_MOE,
        shape,
        "bf16",
        "w2",
        attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="w2",
        launch=launch,
        attributes={
            "tokens_per_expert": counts,
            "local_tokens_per_expert": counts,
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": tuple(range(shape["experts"])),
            "expert_weight_tile_rows": None,
        },
    )
    tile_n = launch.tiling.tile_shape["n"]
    assert (
        sum(math.ceil(count / 64) for count in counts),
        launch.grid_size // math.ceil(shape["hidden"] / tile_n),
    ) == (2166, 2300)
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    operand_path, = tuple(
        path for path in resource_input.memory_paths
        if path.name == "operand_read")
    demand, = operand_path.access_class_demands
    expected_service = (
        "nvidia_e256_h7168_2166_active_2300_max_"
        "fused_moe_w2_operand_pipeline")
    assert demand.service == expected_service
    assert demand.replaces_ingress_path_service
    traffic = fused_moe_predictor._gpu_traffic_components(kernel, hardware)
    assert demand.logical_bytes == pytest.approx(
        traffic.activation_l2_read_bytes + traffic.weight_l2_read_bytes)
    prediction = PerformanceFormula(hardware).predict_kernel(resource_input)
    assert prediction.t_per_group[
        f"memory_access:{expected_service}"
    ] == pytest.approx(
        demand.logical_bytes / 2455337326877.851)


@pytest.mark.parametrize(("variant", "expected_service", "capacity"), (
    (
        "w1",
        "nvidia_e256_h7168_mid_workspace_fused_moe_w1_operand_pipeline",
        2150550845514.6775,
    ),
    (
        "w2",
        "nvidia_e256_h7168_mid_workspace_fused_moe_w2_operand_pipeline",
        2111225044445.784,
    ),
))
def test_a800_e256_h7168_mid_workspace_ignores_route_census(
        variant, expected_service, capacity):
    """The 1,276-tile source workspace does not require one route census."""

    shape = {
        "tokens": 8192,
        "experts": 256,
        "topk": 8,
        "hidden": 7168,
        "intermediate": 2048,
    }
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    projection_n = 2 * shape["intermediate"] if variant == "w1" else shape["hidden"]

    def service_for(expert_counts):
        attributes = {
            "tokens_per_expert": expert_counts,
            "expert_parallel_size": 1,
        }
        launch = select_launch_config(
            hardware.hardware_id,
            KernelKind.FUSED_MOE,
            shape,
            "bf16",
            variant,
            attributes,
            hardware=hardware,
        )
        kernel = KernelSpec(
            KernelKind.FUSED_MOE,
            hardware.hardware_id,
            shape,
            dtype="bf16",
            variant=variant,
            launch=launch,
            attributes={
                "tokens_per_expert": expert_counts,
                "local_tokens_per_expert": expert_counts,
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": tuple(range(shape["experts"])),
                "expert_weight_tile_rows": None,
            },
        )
        maximum_m_tiles = launch.grid_size // math.ceil(projection_n / 64)
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        operand_path, = tuple(
            path for path in resource_input.memory_paths
            if path.name == "operand_read")
        demand, = operand_path.access_class_demands
        return kernel, maximum_m_tiles, demand

    # These valid synthetic routings have distinct active-tile censuses but
    # the identical source-selected padded workspace.  Neither is a dataset
    # task key or an ordered production route.
    for counts in ((256,) * 256, (257, 255) * 128):
        assert sum(counts) == shape["tokens"] * shape["topk"]
        kernel, maximum_m_tiles, demand = service_for(counts)
        assert maximum_m_tiles == 1276
        assert demand.service == expected_service
        assert demand.replaces_ingress_path_service
        traffic = fused_moe_predictor._gpu_traffic_components(kernel, hardware)
        assert demand.logical_bytes == pytest.approx(
            traffic.activation_l2_read_bytes + traffic.weight_l2_read_bytes)
        prediction = PerformanceFormula(hardware).predict_kernel(
            create_predictor(kernel.kind).resource_input(kernel, hardware)[0])
        assert prediction.t_per_group[
            f"memory_access:{expected_service}"
        ] == pytest.approx(demand.logical_bytes / capacity)


def test_a800_e256_h7168_2166_w1_requires_ordered_census():
    """The 2166/2300 W1 service cannot cross route-order boundaries."""

    shape = {
        "tokens": 16384,
        "experts": 256,
        "topk": 8,
        "hidden": 7168,
        "intermediate": 2048,
    }
    census = fused_moe_predictor._NVIDIA_E256_H7168_2166_W1_M_TILE_CENSUS
    counts = [m_tiles * 64 for m_tiles in census]
    excess_rows = sum(counts) - shape["tokens"] * shape["topk"]
    for index in range(len(counts)):
        trimmed_rows = min(excess_rows, 63)
        counts[index] -= trimmed_rows
        excess_rows -= trimmed_rows
    assert excess_rows == 0
    assert sum(counts) == shape["tokens"] * shape["topk"]
    assert tuple(math.ceil(count / 64) for count in counts) == census

    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    def demands_for(expert_counts):
        attributes = {
            "tokens_per_expert": expert_counts,
            "expert_parallel_size": 1,
        }
        launch = select_launch_config(
            hardware.hardware_id,
            KernelKind.FUSED_MOE,
            shape,
            "bf16",
            "w1",
            attributes,
            hardware=hardware,
        )
        kernel = KernelSpec(
            KernelKind.FUSED_MOE,
            hardware.hardware_id,
            shape,
            dtype="bf16",
            variant="w1",
            launch=launch,
            attributes={
                "tokens_per_expert": expert_counts,
                "local_tokens_per_expert": expert_counts,
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": tuple(range(shape["experts"])),
                "expert_weight_tile_rows": None,
            },
        )
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        operand_path, = tuple(
            path for path in resource_input.memory_paths
            if path.name == "operand_read")
        return kernel, operand_path.access_class_demands

    kernel, demands = demands_for(tuple(counts))
    demand, = demands
    expected_service = "nvidia_e256_h7168_2166_exact_w1_operand_pipeline"
    assert demand.service == expected_service
    assert demand.replaces_ingress_path_service
    traffic = fused_moe_predictor._gpu_traffic_components(kernel, hardware)
    assert demand.logical_bytes == pytest.approx(
        traffic.activation_l2_read_bytes + traffic.weight_l2_read_bytes)
    prediction = PerformanceFormula(hardware).predict_kernel(
        create_predictor(kernel.kind).resource_input(kernel, hardware)[0])
    assert prediction.t_per_group[
        f"memory_access:{expected_service}"
    ] == pytest.approx(demand.logical_bytes / 1900390333433.0642)

    _, fallback_demands = demands_for(tuple(reversed(counts)))
    assert all(demand.service != expected_service for demand in fallback_demands)


@pytest.mark.parametrize(("variant", "expected_service", "capacity"), (
    (
        "w1",
        "nvidia_e128_h2880_1084_active_1150_exact_"
        "w1_operand_pipeline",
        2553895510097.6865,
    ),
    (
        "w2",
        "nvidia_e128_h2880_1084_active_1150_exact_"
        "w2_operand_pipeline",
        1985887843680.5706,
    ),
))
def test_a800_e128_h2880_1084_source_requires_ordered_census(
        variant, expected_service, capacity):
    """A same-depth route order cannot reuse this source program."""

    shape = {
        "tokens": 16384,
        "experts": 128,
        "topk": 4,
        "hidden": 2880,
        "intermediate": 2880,
    }
    census = (
        fused_moe_predictor
        ._NVIDIA_E128_H2880_1084_ACTIVE_1150_M_TILE_CENSUS)
    counts = [m_tiles * 64 for m_tiles in census]
    excess_rows = sum(counts) - shape["tokens"] * shape["topk"]
    for index in range(len(counts)):
        trimmed_rows = min(excess_rows, 63)
        counts[index] -= trimmed_rows
        excess_rows -= trimmed_rows
    assert excess_rows == 0
    assert sum(counts) == shape["tokens"] * shape["topk"]
    assert tuple(math.ceil(count / 64) for count in counts) == census

    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    def service_for(expert_counts):
        attributes = {
            "tokens_per_expert": expert_counts,
            "expert_parallel_size": 1,
        }
        launch = select_launch_config(
            hardware.hardware_id,
            KernelKind.FUSED_MOE,
            shape,
            "bf16",
            variant,
            attributes,
            hardware=hardware,
        )
        kernel = KernelSpec(
            KernelKind.FUSED_MOE,
            hardware.hardware_id,
            shape,
            dtype="bf16",
            variant=variant,
            launch=launch,
            attributes={
                "tokens_per_expert": expert_counts,
                "local_tokens_per_expert": expert_counts,
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": tuple(range(shape["experts"])),
                "expert_weight_tile_rows": None,
            },
        )
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        operand_path, = tuple(
            path for path in resource_input.memory_paths
            if path.name == "operand_read")
        return kernel, operand_path.access_class_demands

    kernel, demands = service_for(tuple(counts))
    demand, = demands
    assert demand.service == expected_service
    assert demand.replaces_ingress_path_service
    traffic = fused_moe_predictor._gpu_traffic_components(kernel, hardware)
    assert demand.logical_bytes == pytest.approx(
        traffic.activation_l2_read_bytes + traffic.weight_l2_read_bytes)
    prediction = PerformanceFormula(hardware).predict_kernel(
        create_predictor(kernel.kind).resource_input(kernel, hardware)[0])
    assert prediction.t_per_group[
        f"memory_access:{expected_service}"
    ] == pytest.approx(demand.logical_bytes / capacity)

    _, fallback_demands = service_for(tuple(reversed(counts)))
    assert all(demand.service != expected_service for demand in fallback_demands)


@pytest.mark.parametrize(("variant", "expected_service", "capacity"), (
    (
        "w1",
        "nvidia_e256_h7168_deep_fused_moe_w1_operand_pipeline",
        1849843631119.7896,
    ),
    (
        "w2",
        "nvidia_e256_h7168_deep_fused_moe_w2_operand_pipeline",
        1947624778279.859,
    ),
))
def test_a800_e256_h7168_deep_source_requires_ordered_census(
        variant, expected_service, capacity):
    """The E256 deep source service is not reusable across route orders."""

    shape = {
        "tokens": 32768,
        "experts": 256,
        "topk": 8,
        "hidden": 7168,
        "intermediate": 2048,
    }
    census = fused_moe_predictor._NVIDIA_E256_H7168_DEEP_M_TILE_CENSUS
    counts = [m_tiles * 64 for m_tiles in census]
    excess_rows = sum(counts) - shape["tokens"] * shape["topk"]
    for index in range(len(counts)):
        trimmed_rows = min(excess_rows, 63)
        counts[index] -= trimmed_rows
        excess_rows -= trimmed_rows
    assert excess_rows == 0
    assert sum(counts) == shape["tokens"] * shape["topk"]
    assert tuple(math.ceil(count / 64) for count in counts) == census

    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    def service_for(expert_counts):
        attributes = {
            "tokens_per_expert": expert_counts,
            "expert_parallel_size": 1,
        }
        launch = select_launch_config(
            hardware.hardware_id,
            KernelKind.FUSED_MOE,
            shape,
            "bf16",
            variant,
            attributes,
            hardware=hardware,
        )
        kernel = KernelSpec(
            KernelKind.FUSED_MOE,
            hardware.hardware_id,
            shape,
            dtype="bf16",
            variant=variant,
            launch=launch,
            attributes={
                "tokens_per_expert": expert_counts,
                "local_tokens_per_expert": expert_counts,
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": tuple(range(shape["experts"])),
                "expert_weight_tile_rows": None,
            },
        )
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        operand_path, = tuple(
            path for path in resource_input.memory_paths
            if path.name == "operand_read")
        return kernel, operand_path.access_class_demands

    kernel, demands = service_for(tuple(counts))
    demand, = demands
    assert demand.service == expected_service
    assert demand.replaces_ingress_path_service
    traffic = fused_moe_predictor._gpu_traffic_components(kernel, hardware)
    assert demand.logical_bytes == pytest.approx(
        traffic.activation_l2_read_bytes + traffic.weight_l2_read_bytes)
    prediction = PerformanceFormula(hardware).predict_kernel(
        create_predictor(kernel.kind).resource_input(kernel, hardware)[0])
    assert prediction.t_per_group[
        f"memory_access:{expected_service}"
    ] == pytest.approx(demand.logical_bytes / capacity)

    swapped = tuple(reversed(counts))
    assert sum(math.ceil(count / 64) for count in swapped) == 4230
    _, fallback_demands = service_for(swapped)
    assert all(demand.service != expected_service for demand in fallback_demands)


@pytest.mark.parametrize(("tokens", "counts", "expected_topology"), (
    (
        16384,
        (4064, 4125, 4158, 4060, 4151, 4132, 4010, 4068),
        (515, 520),
    ),
    (
        32768,
        (8103, 8362, 8282, 8117, 8088, 8167, 8235, 8182),
        (1027, 1032),
    ),
))
def test_a800_e8_h4096_exact_w2_requires_ordered_census(
        tokens, counts, expected_topology):
    """The topk=1 W2 service must not escape its captured route order."""

    shape = {
        "tokens": tokens,
        "experts": 8,
        "topk": 2,
        "hidden": 4096,
        "intermediate": 14336,
    }
    assert sum(counts) == tokens * shape["topk"]
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    exact_services = set(
        fused_moe_predictor._NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY
        .values())

    def services_for(expert_counts):
        launch = select_launch_config(
            hardware.hardware_id,
            KernelKind.FUSED_MOE,
            shape,
            "bf16",
            "w2",
            {
                "tokens_per_expert": expert_counts,
                "expert_parallel_size": 1,
            },
            hardware=hardware,
        )
        kernel = KernelSpec(
            KernelKind.FUSED_MOE,
            hardware.hardware_id,
            shape,
            dtype="bf16",
            variant="w2",
            launch=launch,
            attributes={
                "tokens_per_expert": expert_counts,
                "local_tokens_per_expert": expert_counts,
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": tuple(range(shape["experts"])),
                "expert_weight_tile_rows": None,
            },
        )
        tile_m = launch.tiling.tile_shape["m"]
        tile_n = launch.tiling.tile_shape["n"]
        assert (
            sum(math.ceil(count / tile_m) for count in expert_counts),
            launch.grid_size // math.ceil(shape["hidden"] / tile_n),
        ) == expected_topology
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        operand_path, = tuple(
            path for path in resource_input.memory_paths
            if path.name == "operand_read")
        return tuple(
            demand.service for demand in operand_path.access_class_demands)

    assert services_for(counts) == (
        fused_moe_predictor._NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY[
            expected_topology],)
    assert not exact_services.intersection(services_for(tuple(reversed(counts))))








@pytest.mark.parametrize(("variant", "program_selected"), (
    ("w1", 0.0),
    ("w2", 1.0),
))
def test_c500_w2_routed_weight_program_has_no_synthetic_memcpy(
        variant, program_selected):
    shape = {
        "tokens": 1,
        "experts": 256,
        "topk": 8,
        "hidden": 2048,
        "intermediate": 512,
    }
    counts = (1,) * 8 + (0,) * 248
    attributes = {
        "tokens_per_expert": counts,
        "expert_parallel_size": 1,
    }
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FUSED_MOE,
        shape,
        "bf16",
        variant,
        attributes,
        hardware=hardware,
    )
    assert launch.parameters[
        SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER
    ] == program_selected
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        launch=launch,
        attributes={
            "tokens_per_expert": counts,
            "local_tokens_per_expert": counts,
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": tuple(range(shape["experts"])),
            "expert_weight_tile_rows": None,
        },
    )
    predictor = create_predictor(kernel.kind)
    resource_input, _ = predictor.resource_input(kernel, hardware)
    grouped_projection = PerformanceFormula(hardware).predict_kernel(
        resource_input)
    prediction = predictor.midlayer(kernel, hardware)
    assert prediction.t_total_s == pytest.approx(grouped_projection.t_total_s)
    assert "source_post_activation_copy" not in prediction.t_per_group


@pytest.mark.parametrize(("variant", "expected_alu", "phase_names"), (
    ("w1", 0.0, (
        "operand_ingress", "matrix_multiply", "output_writeback")),
    ("w2", 8.0 * 2048.0, (
        "operand_ingress", "matrix_multiply", "routed_weight_read",
        "routed_weight_scale", "output_writeback")),
))
def test_c500_w2_routed_weight_scale_is_source_ordered(
        variant, expected_alu, phase_names):
    shape = {
        "tokens": 1,
        "experts": 256,
        "topk": 8,
        "hidden": 2048,
        "intermediate": 512,
    }
    counts = (1,) * 8 + (0,) * 248
    attributes = {
        "tokens_per_expert": counts,
        "expert_parallel_size": 1,
    }
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FUSED_MOE,
        shape,
        "bf16",
        variant,
        attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        launch=launch,
        attributes={
            "tokens_per_expert": counts,
            "local_tokens_per_expert": counts,
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": tuple(range(shape["experts"])),
            "expert_weight_tile_rows": None,
        },
    )
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert resource_input.demand_per_resource["compute_alu"] == expected_alu
    assert tuple(phase.name for phase in resource_input.phase_ledger.phases) == (
        phase_names)
    traffic = fused_moe_predictor._gpu_traffic_components(kernel, hardware)
    routed_rows = sum(counts)
    _, tile_n, _ = fused_moe_predictor._tile(kernel)
    expected_router_l2 = (
        routed_rows * math.ceil(shape["hidden"] / tile_n) * 4.0
        if variant == "w2" else 0.0)
    expected_router_compulsory = (
        routed_rows * 4.0 if variant == "w2" else 0.0)
    assert traffic.routed_weight_l2_read_bytes == pytest.approx(
        expected_router_l2)
    assert traffic.routed_weight_compulsory_read_bytes == pytest.approx(
        expected_router_compulsory)
    assert traffic.routed_weight_dram_read_bytes == pytest.approx(
        expected_router_compulsory)
    router_paths = tuple(
        path for path in resource_input.memory_paths
        if path.name == "routed_weight_read")
    if variant == "w1":
        assert router_paths == ()
        return
    router_path, = router_paths
    router_nodes = {
        node.resource_role: node for node in router_path.nodes}
    assert router_nodes["l2_cache"].bytes == pytest.approx(
        expected_router_l2)
    assert router_nodes["dram"].bytes == pytest.approx(
        expected_router_compulsory)
    phases = {
        phase.name: phase for phase in resource_input.phase_ledger.phases}
    assert phases["routed_weight_read"].predecessors == (
        "matrix_multiply",)
    assert phases["routed_weight_scale"].predecessors == (
        "routed_weight_read",)
    assert {demand.path_name
            for demand in phases["routed_weight_read"].memory_demands} == {
                "routed_weight_read"}
