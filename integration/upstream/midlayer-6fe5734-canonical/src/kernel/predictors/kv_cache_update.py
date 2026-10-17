"""Source-work analytical prediction for typed device-cache writes."""

from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.common import (
    build_aggregate_gpu_input,
    build_forward_demand,
    prediction_from_ns,
    roofline_ns,
)
from src.kernel.predictors.vector import (
    forward_analytical_resource_input,
    paged_kv_cache_resource_input,
)


def _target_cache_input(kernel: KernelSpec, hardware):
    variant = kernel.variant
    tokens = int(kernel.shape["seq_len"])
    slot_bytes = float(8 * tokens)
    if variant == "mla_kv_cache":
        elements = int(kernel.shape["element_count"])
        tensor_bytes = float(elements * int(kernel.shape["dtype_bytes"]))
        read_bytes = tensor_bytes + slot_bytes
        write_bytes = tensor_bytes
        flops = {
            "compute_tensor": 0.0,
            "compute_fma": 0.0,
            "compute_alu": float(elements + tokens),
            "compute_sfu": 0.0,
        }
        limitations = (
            "mla_cache_append_cta_pipeline_schedule_unavailable",
            "production_duration_calibration_not_used",
        )
    elif variant == "dsa_fp8_index_cache":
        key_dim = int(kernel.shape["key_dim"])
        scale_bytes = int(kernel.shape["scale_bytes_per_token"])
        key_elements = tokens * key_dim
        scale_count = tokens * scale_bytes // 4
        read_bytes = float(2 * key_elements) + slot_bytes
        write_bytes = float(key_elements + tokens * scale_bytes)
        # The fixed CUDA source reduces one absolute maximum per quantization
        # block, derives one UE8M0 scale, then converts every BF16 key element
        # to E4M3. Count each source-level action once.
        flops = {
            "compute_tensor": 0.0,
            "compute_reduction": float(key_elements),
            "compute_fma": float(key_elements),
            "compute_alu": float(key_elements + tokens),
            "compute_sfu": float(scale_count),
        }
        limitations = (
            "fp8_quantization_warp_pipeline_schedule_unavailable",
            "production_duration_calibration_not_used",
        )
    else:
        raise ValueError(
            f"unknown KV_CACHE variant: {kernel.variant!r}")
    total_bytes = read_bytes + write_bytes
    demand = build_forward_demand(total_bytes, total_bytes, True, flops)
    inp = build_aggregate_gpu_input(
        kernel,
        hardware,
        demand,
        write_l2_bytes=write_bytes,
        write_dram_bytes=write_bytes,
        limitations=limitations,
    )
    return inp, total_bytes


def _model(kernel: KernelSpec, hardware):
    if kernel.variant == "paged_kv_cache":
        return paged_kv_cache_resource_input(kernel, hardware)
    return _target_cache_input(kernel, hardware)


class KVCacheUpdatePredictor(KernelPredictor):
    kind = KernelKind.KV_CACHE

    def resource_input(self, kernel: KernelSpec, hardware):
        return _model(kernel, hardware)

    def midlayer(self, kernel: KernelSpec, hardware):
        if kernel.variant == "paged_kv_cache":
            inp, _ = forward_analytical_resource_input(kernel, hardware)
        elif kernel.variant in {"mla_kv_cache", "dsa_fp8_index_cache"}:
            inp, _ = self.resource_input(kernel, hardware)
        else:
            raise ValueError(
                f"unknown KV_CACHE variant: {kernel.variant!r}")
        return PerformanceFormula(hardware).predict_kernel(inp)

    def roofline(self, kernel: KernelSpec, hardware):
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory),
            "roofline",
        )
