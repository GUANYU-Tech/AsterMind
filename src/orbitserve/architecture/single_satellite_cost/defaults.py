"""Default profiles anchored to the parameter document.

Purchase prices are intentionally not embedded in accelerator defaults because
the parameter document treats COTS GPU price as quote/index driven rather than
a stable public fact.
"""

from __future__ import annotations

from .models import AcceleratorProfile, StorageDevice


PARAMETER_DOC = "docs/research/parameters/single_satellite_cost_model_parameter_values.md"


ACCELERATOR_PROFILES: dict[str, AcceleratorProfile] = {
    "jetson_agx_orin_64gb": AcceleratorProfile(
        name="jetson_agx_orin_64gb",
        peak_compute_tflops=0.275,
        fast_mem_gb=64.0,
        fast_mem_bw_gbs=204.8,
        peak_power_w=60.0,
        mass_kg=0.5,
        intra_bw_gbs=64.0,
        mass_source="module proxy sensitivity",
        source_ids=("S21", "S13"),
    ),
    "a100_80gb": AcceleratorProfile(
        name="a100_80gb",
        peak_compute_tflops=312.0,
        fast_mem_gb=80.0,
        fast_mem_bw_gbs=2000.0,
        peak_power_w=400.0,
        mass_kg=16.25,
        intra_bw_gbs=600.0,
        mass_source="DGX/HGX-class system proxy",
        source_ids=("S7", "S13"),
    ),
    "h100_sxm": AcceleratorProfile(
        name="h100_sxm",
        peak_compute_tflops=989.0,
        fast_mem_gb=80.0,
        fast_mem_bw_gbs=3350.0,
        peak_power_w=700.0,
        mass_kg=16.25,
        intra_bw_gbs=900.0,
        mass_source="DGX/HGX-class system proxy",
        source_ids=("S5", "S13"),
    ),
    "h200_sxm": AcceleratorProfile(
        name="h200_sxm",
        peak_compute_tflops=989.0,
        fast_mem_gb=141.0,
        fast_mem_bw_gbs=4800.0,
        peak_power_w=700.0,
        mass_kg=16.25,
        intra_bw_gbs=900.0,
        mass_source="DGX/HGX-class system proxy",
        source_ids=("S6", "S13"),
    ),
    "b200_hgx_per_gpu": AcceleratorProfile(
        name="b200_hgx_per_gpu",
        peak_compute_tflops=2250.0,
        fast_mem_gb=175.0,
        fast_mem_bw_gbs=7500.0,
        peak_power_w=1000.0,
        mass_kg=20.0,
        intra_bw_gbs=1800.0,
        mass_source="HGX system proxy sensitivity",
        source_ids=("S20", "S13"),
    ),
}


STORAGE_PROFILES: dict[str, StorageDevice] = {
    "none": StorageDevice(
        name="none",
        capacity_tb=0.0,
        bandwidth_gbs=0.0,
        power_w=0.0,
        mass_kg=0.0,
        cost_per_gb_usd=0.0,
    ),
    "nvme_4tb": StorageDevice(
        name="nvme_4tb",
        capacity_tb=4.0,
        bandwidth_gbs=7.0,
        power_w=25.0,
        mass_kg=0.10,
        cost_per_gb_usd=0.10,
        source_ids=("S11", "S23"),
    ),
    "nvme_8tb": StorageDevice(
        name="nvme_8tb",
        capacity_tb=8.0,
        bandwidth_gbs=7.0,
        power_w=25.0,
        mass_kg=0.10,
        cost_per_gb_usd=0.10,
        source_ids=("S11", "S23"),
    ),
    "nvme_32tb": StorageDevice(
        name="nvme_32tb",
        capacity_tb=32.0,
        bandwidth_gbs=7.0,
        power_w=25.0,
        mass_kg=0.25,
        cost_per_gb_usd=0.10,
        source_ids=("S11", "S23"),
    ),
}


def accelerator_profile(name: str) -> AcceleratorProfile:
    try:
        return ACCELERATOR_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(ACCELERATOR_PROFILES))
        raise KeyError(f"unknown accelerator profile '{name}'. Known profiles: {known}") from exc


def storage_profile(name: str) -> StorageDevice:
    try:
        return STORAGE_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(STORAGE_PROFILES))
        raise KeyError(f"unknown storage profile '{name}'. Known profiles: {known}") from exc


