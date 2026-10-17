"""Zero-fitting mirror contract for the 910B compare/cast serialized service.

Every test here mirrors one rule of the routing-mask pricing contract: the
published HWSpec numbers are exactly the microbench reference's, the
predicted duration is the measured serialized-completion law (per-launch
setup plus one chip-level serialized stream, never parallelized across the
AIV workers), and any off-domain use fails closed.
"""

import json
from pathlib import Path

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor, predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.model_contract import ComputeServiceDemand

REFERENCE_PATH = Path(
    "microbench/ascend_910b/references/"
    "elementwise_compare_cast_serialized_service_cann_8_5_1.json")
REFERENCE = json.loads(REFERENCE_PATH.read_text())
OPERATION_TO_SERVICE = {
    "notequal_int32_bool": "notequal_int32_bool_serialized",
    "cast_bool_bf16": "cast_bool_bf16_serialized",
}


def _law(operation, elements):
    service = REFERENCE["services"][OPERATION_TO_SERVICE[operation]]
    return (
        service["setup_time_s_per_launch"]
        + elements
        / service["steady_rate_elements_per_s_serialized_stream"])


def _spec(hardware_id, operation, elements, extra_attributes=None):
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    shape = {"element_count": elements}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "elementwise",
        {
            "operation": operation,
            "reads": 1,
            "writes": 1,
            "launch_policy": "injected_launch",
        },
        hardware=hardware,
    )
    attributes = {
        "operation": operation,
        "reads": 1,
        "writes": 1,
    }
    if extra_attributes:
        attributes.update(extra_attributes)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch,
        attributes=attributes)
    return kernel, hardware


def test_reference_and_hardware_spec_publish_the_same_service_law():
    vector = HWSpec(
        "configs/hardware/ascend_910b.yaml").resource_by_role(
            "compute_vector")
    for operation, service_name in OPERATION_TO_SERVICE.items():
        published = REFERENCE["services"][service_name]
        assert published["service_publishable"] is True
        assert vector.service_rates_per_unit_s[service_name] == pytest.approx(
            published["steady_rate_elements_per_s_serialized_stream"])
        assert vector.service_setup_time_s[service_name] == pytest.approx(
            published["setup_time_s_per_launch"])
    assert REFERENCE["production_or_evaluation_duration_consumed"] is False
    assert REFERENCE["hardware_id"] == "ascend_910b"
    assert REFERENCE["validation_maximum_ape_percent"] == pytest.approx(
        max(
            service["maximum_validation_ape_percent"]
            for service in REFERENCE["services"].values()))
    sizes = REFERENCE["request_sizes_elements"]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert set(REFERENCE["calibration_run_ids"]).isdisjoint(
        REFERENCE["validation_run_ids"])
    for service in REFERENCE["services"].values():
        # The measured law stays publishable only while the holdout error
        # of every service stays inside the mechanism precedent ceiling.
        assert service["maximum_validation_ape_percent"] < 10.0


@pytest.mark.parametrize("operation", OPERATION_TO_SERVICE)
@pytest.mark.parametrize("elements", (104, 2848, 27648))
def test_device_time_never_drops_below_the_serialized_law(
        operation, elements):
    kernel, hardware = _spec("ascend_910b", operation, elements)
    prediction = predict(kernel, hardware)
    law_ns = _law(operation, elements) * 1e9
    device_ns = prediction.t_total_ns - prediction.t_launch_s * 1e9
    # The serialized stream is a floor: extra memory-path assembly may add
    # time, but no schedule may take credit for parallelizing it.
    assert device_ns >= law_ns - 1e-6
    smaller, _ = _spec("ascend_910b", operation, elements // 2 or 1)
    assert predict(smaller, hardware).t_total_ns < prediction.t_total_ns


@pytest.mark.parametrize(
    "operation, elements",
    (
        ("notequal_int32_bool", 104),
        ("notequal_int32_bool", 2848),
        ("notequal_int32_bool", 27648),
        ("cast_bool_bf16", 104),
        ("cast_bool_bf16", 2848),
    ),
)
def test_predicted_duration_is_the_serialized_completion_law(
        operation, elements):
    kernel, hardware = _spec("ascend_910b", operation, elements)
    prediction = predict(kernel, hardware)
    assert prediction.bottleneck_group == "vector_pipe"
    law_ns = _law(operation, elements) * 1e9
    device_ns = prediction.t_total_ns - prediction.t_launch_s * 1e9
    # Where the serialized service dominates the assembled device time, the
    # device time is the law alone: a fixed per-launch setup plus one
    # chip-level stream at the published steady rate, with no per-worker
    # parallelization credit.
    assert device_ns == pytest.approx(law_ns, rel=1e-6, abs=1.0)


def test_off_910b_pricing_fails_closed():
    with pytest.raises(ValueError):
        _spec("nvidia_a800", "notequal_int32_bool", 2848)


def test_worker_schedule_is_a_single_owner_serialized_stream():
    kernel, hardware = _spec("ascend_910b", "cast_bool_bf16", 2848)
    resource_input, _ = create_predictor(
        kernel.kind).resource_input(kernel, hardware)
    demand = next(
        demand for demand in resource_input.compute_demands
        if demand.resource_role == "compute_vector")
    (service,) = demand.service_demands
    assert service.service == "cast_bool_bf16_serialized"
    # Conservation: the service consumes exactly the element count the
    # demand carries, split onto one worker as a serialized stream.
    assert service.operations == 2848.0
    assert service.serial_invocations == 1
    assert sum(service.operations_by_worker) == 2848.0
    nonzero = [
        (operations, invocations)
        for operations, invocations in zip(
            service.operations_by_worker, service.serial_invocations_by_worker)
        if operations or invocations]
    assert nonzero == [(2848.0, 1)]


def test_compute_service_demand_validates_the_worker_contract():
    ComputeServiceDemand(
        "notequal_int32_bool_serialized", 100.0,
        serial_invocations=1,
        operations_by_worker=(100.0, 0.0),
        serial_invocations_by_worker=(1, 0))
    with pytest.raises(ValueError, match="must conserve the aggregate"):
        ComputeServiceDemand(
            "notequal_int32_bool_serialized", 100.0,
            serial_invocations=1,
            operations_by_worker=(99.0, 0.0),
            serial_invocations_by_worker=(1, 0))
    with pytest.raises(ValueError, match="critical-worker invocation"):
        ComputeServiceDemand(
            "cast_bool_bf16_serialized", 100.0,
            serial_invocations=0,
            operations_by_worker=(100.0, 0.0),
            serial_invocations_by_worker=(1, 0))
    with pytest.raises(ValueError, match="one of"):
        ComputeServiceDemand(
            "notequal_int32_unserialized", 100.0)


@pytest.mark.parametrize(
    "helper_launch, match",
    (
        ({"class": 1}, "non-empty 'class' string"),
        ({"class": ""}, "non-empty 'class' string"),
        ({"class": "run", "extra": 1}, "non-empty 'class' string"),
    ),
)
def test_kernel_spec_validates_the_helper_launch_attribute(
        helper_launch, match):
    with pytest.raises(ValueError, match=match):
        _spec(
            "ascend_910b", "notequal_int32_bool", 2848,
            extra_attributes={"helper_launch": helper_launch})
