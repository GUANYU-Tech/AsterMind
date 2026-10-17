"""Complete-spec KV cache tier load and eviction prediction."""

from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.common import prediction_from_ns


class KVOffloadPredictor(KernelPredictor):
    kind = KernelKind.KV_OFFLOAD

    def _latency(self, kernel: KernelSpec, hardware, include_latency: bool) -> float:
        transfer = kernel.transfer
        link = transfer.link.lower()
        bandwidth_key = f"{link}_bw_bytes_s"
        latency_key = f"{link}_latency_ns"
        if link not in ("pcie", "nvlink"):
            raise ValueError(f"unknown KV_OFFLOAD link: {transfer.link!r}")
        bandwidth = hardware.interconnect[bandwidth_key]
        if bandwidth is None:
            raise KeyError(f"KV_OFFLOAD requires interconnect.{bandwidth_key}")
        latency_value = hardware.interconnect[latency_key]
        if include_latency and latency_value is None:
            raise KeyError(f"KV_OFFLOAD requires interconnect.{latency_key}")
        latency = float(latency_value) if include_latency else 0.0
        return latency + transfer.bytes / float(bandwidth) * 1e9

    def midlayer(self, kernel: KernelSpec, hardware):
        return prediction_from_ns(self._latency(kernel, hardware, True), "kv_offload")

    def roofline(self, kernel: KernelSpec, hardware):
        return prediction_from_ns(self._latency(kernel, hardware, False), "kv_offload")
