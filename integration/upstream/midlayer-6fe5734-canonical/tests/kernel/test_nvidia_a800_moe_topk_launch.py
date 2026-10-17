"""A800 source launch contract for vLLM's specialized MoE top-k kernel."""

from dataclasses import replace

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.nvidia_a800.vector_launch import (
    select_vector_launch,
)


_SHAPE = {
    "tokens": 129,
    "experts": 2,
    "topk": 1,
    "hidden": 64,
    "intermediate": 16,
}


def test_moe_topk_uses_vllm_source_block_geometry():
    launch = select_vector_launch(
        _SHAPE, "bf16", "moe_topk", {}, hardware=None)

    assert launch.grid == (2, 1, 1)
    assert launch.block == (32, 4, 1)
    assert launch.block_size == 128


def test_moe_topk_source_geometry_preserves_prediction():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    source_launch = select_vector_launch(
        _SHAPE, "bf16", "moe_topk", {}, hardware=hardware)
    flattened_launch = LaunchConfig(
        algorithm=source_launch.algorithm,
        grid=source_launch.grid,
        block=(128, 1, 1),
        execution_variant=source_launch.execution_variant,
    )
    source_kernel = KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id="nvidia_a800",
        shape=_SHAPE,
        dtype="bf16",
        variant="moe_topk",
        launch=source_launch,
    )
    flattened_kernel = replace(source_kernel, launch=flattened_launch)

    assert predict(source_kernel, hardware) == predict(flattened_kernel, hardware)
