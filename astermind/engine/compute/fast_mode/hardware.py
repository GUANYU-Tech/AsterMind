"""Bind Roofline inputs to the selected Cell hardware profile."""

from dataclasses import dataclass, replace
import json
from pathlib import Path


_PROFILES = json.loads(Path(__file__).with_name("hardware_profiles.json").read_text(encoding="utf-8"))
_H200 = _PROFILES["nvidia_h200"]
KV_HBM_UTILIZATION_PERCENT = 90
KV_RUNTIME_RESERVE_BYTES = 2 * 2**30


def fast_hbm_capacity_bytes(hardware_id):
    return _PROFILES[hardware_id]["hbm_capacity_gib"] * 2**30


@dataclass(frozen=True, slots=True)
class Hardware:
    bf16_flops: float = _H200["bf16_flops"]
    fp8_flops: float | None = _H200["fp8_flops"]
    # Peak bytes/s and efficiency are independent of HBM capacity in bytes.
    hbm_bytes: float = _H200["hbm_bytes_per_s"]
    tensor_efficiency: float = .70
    hbm_efficiency: float = .96
    fp16_flops: float | None = None
    widen_fp8_to_bf16: bool = False

    @classmethod
    def from_hardware_spec(cls, spec, *, compute_dtype="bf16", widen_fp8_to_bf16=False):
        profile = _PROFILES[spec.hardware_id]
        default_hbm = .96 if spec.hardware_id == "nvidia_h200" else 1.
        return cls(bf16_flops=profile["bf16_flops"], fp8_flops=profile["fp8_flops"],
                   fp16_flops=profile["fp16_flops"] if compute_dtype == "fp16" else None,
                   hbm_bytes=profile["hbm_bytes_per_s"], widen_fp8_to_bf16=widen_fp8_to_bf16,
                   hbm_efficiency=default_hbm)

    def nominal_hbm_bytes(self, hardware_id):
        return _PROFILES[hardware_id]["hbm_bytes_per_s"]

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
