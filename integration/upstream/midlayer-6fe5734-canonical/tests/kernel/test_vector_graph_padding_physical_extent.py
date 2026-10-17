"""The CANN FULL-graph re-binding keeps padded Vector specs on a real plan.

A padded main spec keeps the logical token shape while its recorded launch is
already the physical graph dispatch.  The source tiling contracts decide which
of the two extents the launch belongs to, and only a launch that demonstrably
belongs to the physical one is re-bound.
"""

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.predictors.vector import (
    _cann_full_graph_source_program_execution_kernel,
    _has_graph_padding,
)


HARDWARE = HWSpec("configs/hardware/ascend_910b.yaml")

LOGICAL_TOKENS = 10
PHYSICAL_TOKENS = 16
INTERMEDIATE = 9728
HIDDEN = 2560


def _shape(variant: str, tokens: int):
    if variant == "silu":
        return {
            "tok": tokens,
            "intermediate": INTERMEDIATE,
            "element_count": tokens * INTERMEDIATE,
        }
    return {"rows": tokens, "hidden": HIDDEN}


def _spec(variant: str, *, launch_tokens: int, shape_tokens: int):
    """Build a graph-padded spec whose launch is taken at ``launch_tokens``."""

    launch = select_launch_config(
        HARDWARE.hardware_id, KernelKind.VECTOR, _shape(variant, launch_tokens),
        "bf16", variant, {"launch_policy": "injected_launch"},
        hardware=HARDWARE)
    return KernelSpec(
        KernelKind.VECTOR, HARDWARE.hardware_id, _shape(variant, shape_tokens),
        dtype="bf16", variant=launch.execution_variant, launch=launch,
        attributes={
            "graph_logical_tokens": LOGICAL_TOKENS,
            "graph_model_input_tokens": PHYSICAL_TOKENS,
            "graph_runtime_mode": "FULL",
            "model_input_shape": _shape(variant, PHYSICAL_TOKENS),
        })


def test_physical_extent_launch_rebinds_every_gated_variant():
    for variant in ("silu", "rmsnorm", "add_rmsnorm"):
        padded = _spec(variant, launch_tokens=PHYSICAL_TOKENS,
                       shape_tokens=LOGICAL_TOKENS)
        assert _has_graph_padding(padded)

        bound = _cann_full_graph_source_program_execution_kernel(
            padded, HARDWARE)

        assert not _has_graph_padding(bound)
        for name, value in _shape(variant, PHYSICAL_TOKENS).items():
            assert bound.shape[name] == value


def test_rebound_spec_keeps_its_source_pipeline_plan():
    for variant in ("silu", "rmsnorm", "add_rmsnorm"):
        padded = _spec(variant, launch_tokens=PHYSICAL_TOKENS,
                       shape_tokens=LOGICAL_TOKENS)
        exact = _spec(variant, launch_tokens=LOGICAL_TOKENS,
                      shape_tokens=LOGICAL_TOKENS)
        exact = KernelSpec(
            exact.kind, exact.hardware_id, exact.shape, dtype=exact.dtype,
            variant=exact.variant, launch=exact.launch)

        predictor = create_predictor(KernelKind.VECTOR)
        padded_plan = predictor.resource_input(padded, HARDWARE)[0]
        exact_plan = predictor.resource_input(exact, HARDWARE)[0]

        assert exact_plan.pipeline_plan is not None
        assert padded_plan.pipeline_plan is not None


def test_logical_extent_launch_is_not_treated_as_physical_dispatch():
    for variant in ("silu", "rmsnorm", "add_rmsnorm"):
        padded = _spec(variant, launch_tokens=LOGICAL_TOKENS,
                       shape_tokens=LOGICAL_TOKENS)

        bound = _cann_full_graph_source_program_execution_kernel(
            padded, HARDWARE)

        assert bound is padded
        assert _has_graph_padding(bound)
