"""H200 SGLang glue (MoE top-k append, EAGLE V2 bookkeeping): vector launch contract."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import (
    select_launch_config,
)

_H200 = HWSpec("configs/hardware/nvidia_h200.yaml")


def _spec(variant, dtype, elements, attributes):
    shape = {"element_count": elements}
    launch = select_launch_config(
        _H200.hardware_id, KernelKind.VECTOR, shape, dtype, variant,
        attributes, hardware=_H200)
    return KernelSpec(
        kind=KernelKind.VECTOR, hardware_id=_H200.hardware_id, shape=shape,
        dtype=dtype, variant=variant, launch=launch, attributes=attributes)


@pytest.mark.parametrize(
    "variant, dtype, elements, attributes, grid, block",
    [
        ("elementwise", "int32", 36,
         {"operation": "fill", "reads": 0, "writes": 1}, 1, 128),
        # PyTorch direct_copy launches nt=128, vt=2 for 4-byte dtypes.
        ("elementwise", "int32", 512,
         {"operation": "copy", "reads": 1, "writes": 1}, 2, 128),
        ("elementwise", "float32", 4,
         {"operation": "unary", "reads": 1, "writes": 1}, 1, 128),
        ("reduce", "float32", 4, {"reduction_factor": 8}, 1, 256),
        ("elementwise", "int64", 512,
         {"operation": "copy", "reads": 1, "writes": 1}, 2, 128),
        ("elementwise", "bool", 2048,
         {"operation": "fill", "reads": 0, "writes": 1}, 2, 128),
    ],
)
def test_glue_launches_are_selected_and_priced(
        variant, dtype, elements, attributes, grid, block):
    spec = _spec(variant, dtype, elements, attributes)
    assert spec.launch.grid_size == grid
    assert spec.launch.block_size == block
    assert predict(spec, _H200, "midlayer").t_total_s > 0


def test_scalar_multiply_reads_one_tensor():
    with pytest.raises(ValueError, match="reads/writes disagree"):
        _spec("elementwise", "float32", 4,
              {"operation": "unary", "reads": 2, "writes": 1})


def test_tensor_multiply_on_integers_stays_closed():
    with pytest.raises(ValueError):
        _spec("elementwise", "int64", 4,
              {"operation": "mul", "reads": 2, "writes": 1})


def test_logits_cast_reads_bf16_and_writes_fp32():
    cast = _spec("elementwise", "float32", 1 << 20,
                 {"operation": "cast_bf16_float32", "reads": 1, "writes": 1})
    copy = _spec("elementwise", "float32", 1 << 20,
                 {"operation": "copy", "reads": 1, "writes": 1})
    cast_time = predict(cast, _H200, "midlayer").t_total_s
    copy_time = predict(copy, _H200, "midlayer").t_total_s
    assert 0 < cast_time < copy_time
