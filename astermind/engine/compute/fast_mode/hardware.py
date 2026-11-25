"""Bind Roofline inputs to the selected Cell hardware profile."""

from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True, slots=True)
class Hardware:
    bf16_flops: float = 989.4e12
    fp8_flops: float | None = 1978.8e12
    # The pinned H200 read rate for standalone formula probes. The CLI binds
    # the actual Cell hardware profile at startup.
    hbm_bytes: float = 1478.7609855270666 * 3.125e9
    tensor_efficiency: float = .70
    hbm_efficiency: float = 1.
    fp16_flops: float | None = None
    widen_fp8_to_bf16: bool = False

    @classmethod
    def from_hardware_spec(cls, spec, *, compute_dtype="bf16", widen_fp8_to_bf16=False):
        read_bandwidth = spec.dram_capacity_for_access(
            "read", active_units=1, sm_freq_hz=spec.nominal_sm_freq_hz,
            dram_freq_hz=spec.nominal_dram_freq_hz,
        )
        if spec.hardware_id == "nvidia_h200":
            return cls(hbm_bytes=read_bandwidth)

        def tensor_peak(dtype):
            try:
                resource = spec.resource_by_role(f"compute_tensor_{dtype}")
            except KeyError:
                resource = spec.resource_by_role("compute_tensor")
                if not resource.name.endswith(f"_{dtype}"):
                    return None
            if not isfinite(resource.capacity) or resource.capacity <= 0:
                raise ValueError(f"{spec.hardware_id} has an invalid {dtype} tensor peak")
            return resource.capacity

        bf16 = tensor_peak("bf16")
        if bf16 is None:
            raise ValueError(f"{spec.hardware_id} does not declare a BF16 tensor peak")
        fp16 = tensor_peak("fp16") if compute_dtype == "fp16" else None
        if compute_dtype == "fp16" and fp16 is None:
            raise ValueError(f"{spec.hardware_id} does not declare an FP16 tensor peak")
        return cls(bf16_flops=bf16, fp8_flops=tensor_peak("fp8"), fp16_flops=fp16,
                   hbm_bytes=read_bandwidth, widen_fp8_to_bf16=widen_fp8_to_bf16)

    def nominal_hbm_bytes(self, hardware_id):
        # H200 SXM is rated at 4.8 TB/s. Other profiles use their declared read bandwidth.
        return 4.8e12 if hardware_id == "nvidia_h200" else self.hbm_bytes

    def with_roofline(self, hardware_id, settings):
        if settings is None:
            return self
        return replace(self, hbm_bytes=self.nominal_hbm_bytes(hardware_id),
                       tensor_efficiency=settings.tensor_percent / 100, hbm_efficiency=settings.hbm_percent / 100)

    def roofline(self, flops, memory, *, fp8=False):
        if fp8:
            # Widen finite, already-quantized E4M3 operands exactly inside a
            # tile. This reference charges packed HBM bytes and BF16 products.
            # Conversion, block scaling and intermediate tile traffic are omitted.
            peak = self.bf16_flops if self.widen_fp8_to_bf16 else self.fp8_flops
        else:
            peak = self.fp16_flops or self.bf16_flops
        if peak is None:
            raise ValueError("the selected hardware has no FP8 tensor profile")
        return max(flops / (peak * self.tensor_efficiency),
                   memory / (self.hbm_bytes * self.hbm_efficiency))
