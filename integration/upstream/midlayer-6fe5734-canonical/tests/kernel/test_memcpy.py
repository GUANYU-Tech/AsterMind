"""Physical transfer pricing tests for the canonical MEMCPY predictor."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)


def _d2d_spec(byte_count, hardware_id="nvidia_a800"):
    return KernelSpec(
        kind=KernelKind.MEMCPY,
        hardware_id=hardware_id,
        transfer=PhysicalTransferConfig(
            algorithm="dma",
            bytes=byte_count,
            direction="d2d",
            link="dram",
        ),
    )


def test_a800_d2d_roofline_uses_directional_read_and_write_service():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    byte_count = 256 * 1024 * 1024
    read_capacity = hardware.dram_capacity_for_access(
        "read",
        active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )
    write_capacity = hardware.dram_capacity_for_access(
        "write",
        active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )

    prediction = predict(_d2d_spec(byte_count), hardware, "roofline")

    assert prediction.t_total_s == pytest.approx(
        byte_count / read_capacity + byte_count / write_capacity)
    assert prediction.t_total_s != pytest.approx(
        2.0 * byte_count / read_capacity, rel=1e-6)


def test_a800_d2d_midlayer_fails_without_independent_fixed_floor():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    with pytest.raises(
            KeyError,
            match="MEMCPY requires memory_subsystem.memcpy_dtod_floor_ns"):
        predict(_d2d_spec(1024), hardware, "midlayer")


def test_c500_d2d_midlayer_adds_measured_floor_to_directional_payload():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    byte_count = 8192
    read_capacity = hardware.dram_capacity_for_access(
        "read",
        active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )
    write_capacity = hardware.dram_capacity_for_access(
        "write",
        active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )

    prediction = predict(
        _d2d_spec(byte_count, hardware.hardware_id), hardware, "midlayer")

    assert prediction.t_total_ns == pytest.approx(
        9612.988764927777
        + byte_count / read_capacity * 1e9
        + byte_count / write_capacity * 1e9)


def test_c500_native_flash_attention_copy_uses_medium_task_plateau():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    byte_count = 256 * 1024
    read_capacity = hardware.dram_capacity_for_access(
        "read",
        active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )
    write_capacity = hardware.dram_capacity_for_access(
        "write",
        active_units=hardware.num_compute_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )
    spec = KernelSpec(
        kind=KernelKind.MEMCPY,
        hardware_id=hardware.hardware_id,
        transfer=PhysicalTransferConfig(
            algorithm="native_d2d_flash_attention_copy",
            bytes=byte_count,
            direction="d2d",
            link="dram",
        ),
    )

    prediction = predict(spec, hardware, "midlayer")

    assert prediction.t_total_ns == pytest.approx(
        35735.33924058255
        + byte_count / read_capacity * 1e9
        + byte_count / write_capacity * 1e9)
