"""H200 eager stream memcpy service: contract and fail-closed boundaries."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec, PhysicalTransferConfig

_H200 = HWSpec("configs/hardware/nvidia_h200.yaml")


def _copy(algorithm, direction, payload):
    link = "dram" if direction == "d2d" else "pcie"
    return KernelSpec(
        kind=KernelKind.MEMCPY, hardware_id="nvidia_h200",
        transfer=PhysicalTransferConfig(
            algorithm=algorithm, bytes=payload, direction=direction,
            link=link))


@pytest.mark.parametrize("algorithm, direction, key", [
    ("stream_copy_pageable", "h2d", "h2d_pageable"),
    ("stream_copy_pinned", "h2d", "h2d_pinned"),
    ("stream_copy_pinned", "d2h", "d2h_pinned"),
    ("stream_copy", "d2d", "d2d"),
])
def test_stream_copy_uses_its_measured_table(algorithm, direction, key):
    spec = _copy(algorithm, direction, 300)
    assert predict(spec, _H200, "midlayer").t_total_s == pytest.approx(
        _H200.stream_memcpy_service_ns(key, 300) * 1e-9)


def test_stream_copy_fails_closed_above_its_range():
    with pytest.raises(ValueError, match="measured only below"):
        predict(_copy("stream_copy", "d2d", 8 << 20), _H200, "midlayer")


def test_host_stream_copy_names_its_memory_kind():
    with pytest.raises(ValueError):
        _copy("stream_copy", "h2d", 64)
    with pytest.raises(ValueError):
        _copy("stream_copy_pinned", "d2d", 64)
