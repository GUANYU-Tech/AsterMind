"""Both NORM variants must be priced by one per-element arithmetic tally.

The expected operation counts are read off the two vLLM source programs in
``vllm/csrc/libtorch_stable/layernorm_kernels.cu`` (with ``sum_squares()``
from ``csrc/type_convert.cuh``) on the production ``HasWeight`` path, not from
any measured duration:

* pass one squares each element -- one multiply and one accumulate;
* ``fused_add_rms_norm_kernel`` first adds the residual -- one more add;
* pass two scales each element by ``s_variance`` and the weight -- two
  multiplies.

so ``rms_norm`` costs four operations per element and ``fused_add_rms_norm``
costs five.  The residual writeback is a store and belongs to the byte
account, not to this tally.
"""

import math

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.predictors.vector import (
    _NORM_OUTPUT_MULTIPLY_OPERATIONS_PER_ELEMENT,
    _NORM_RESIDUAL_ADD_OPERATIONS_PER_ELEMENT,
    _NORM_SQUARE_ACCUMULATE_OPERATIONS_PER_ELEMENT,
    _norm_arithmetic_operations_per_element,
)

# Source instruction counts, one entry per pass of each kernel.
_SOURCE_TALLY = {
    "rmsnorm": {
        "square_accumulate": 2.0,
        "residual_add": 0.0,
        "output_multiply": 2.0,
    },
    "add_rmsnorm": {
        "square_accumulate": 2.0,
        "residual_add": 1.0,
        "output_multiply": 2.0,
    },
}


@pytest.mark.parametrize("variant", sorted(_SOURCE_TALLY))
def test_per_element_tally_matches_the_source_instruction_count(variant):
    tally = _SOURCE_TALLY[variant]
    assert (_NORM_SQUARE_ACCUMULATE_OPERATIONS_PER_ELEMENT
            == tally["square_accumulate"])
    assert (_NORM_OUTPUT_MULTIPLY_OPERATIONS_PER_ELEMENT
            == tally["output_multiply"])
    assert (_NORM_RESIDUAL_ADD_OPERATIONS_PER_ELEMENT[variant]
            == tally["residual_add"])
    assert (_norm_arithmetic_operations_per_element(variant)
            == math.fsum(tally.values()))


def test_the_two_variants_share_every_common_per_element_term():
    """Only the residual add may differ between the two source programs."""

    shared = {
        name: value for name, value in _SOURCE_TALLY["rmsnorm"].items()
        if name != "residual_add"
    }
    assert shared == {
        name: value
        for name, value in _SOURCE_TALLY["add_rmsnorm"].items()
        if name != "residual_add"
    }
    difference = (
        _norm_arithmetic_operations_per_element("add_rmsnorm")
        - _norm_arithmetic_operations_per_element("rmsnorm"))
    assert difference == _NORM_RESIDUAL_ADD_OPERATIONS_PER_ELEMENT[
        "add_rmsnorm"]


def _norm_resource_input(hardware_id: str, variant: str, rows: int,
                         hidden: int):
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    shape = {"rows": rows, "hidden": hidden}
    launch = select_launch_config(
        hardware_id, KernelKind.VECTOR, shape, "bf16", variant,
        {"launch_policy": "injected_launch"}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch, attributes={})
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    return resource_input


def _phase_operations(resource_input, phase_name: str, service: str):
    phase = next(
        phase for phase in resource_input.phase_ledger.phases
        if phase.name == phase_name)
    return math.fsum(
        demand.operations for demand in phase.compute_demands
        if demand.operation == service)


@pytest.mark.parametrize("variant", sorted(_SOURCE_TALLY))
def test_generic_norm_phases_charge_exactly_the_source_tally(variant):
    """A card without typed source rounds charges the tally, once."""

    hidden = 4096
    resource_input = _norm_resource_input(
        "ascend_910b", variant, rows=8, hidden=hidden)
    elements = 8 * hidden
    tally = _SOURCE_TALLY[variant]
    charged = {
        "square_accumulate": _phase_operations(
            resource_input, "input_compute", "square_accumulate"),
        "residual_add": _phase_operations(
            resource_input, "input_compute", "residual_add"),
        "output_multiply": _phase_operations(
            resource_input, "output_compute", "output_multiply"),
    }
    assert charged == {
        name: value * elements for name, value in tally.items()}
    assert math.fsum(charged.values()) == (
        _norm_arithmetic_operations_per_element(variant) * elements)
