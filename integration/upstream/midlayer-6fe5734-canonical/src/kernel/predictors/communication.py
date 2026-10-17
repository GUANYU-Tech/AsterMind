"""Zero-fit analytical communication Roofline and backend dispatch."""

from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.common import prediction_from_ns
from src.kernel.predictors.protobridge.adapter import predict_protobridge


def _required_interconnect(hardware, key: str) -> float:
    value = hardware.interconnect[key]
    if value is None:
        raise KeyError(f"COMMUNICATION requires interconnect.{key}")
    return float(value)


class CommunicationPredictor(KernelPredictor):
    kind = KernelKind.COMMUNICATION

    def _analytical_roofline(self, kernel: KernelSpec, hardware):
        transfer = kernel.transfer
        ranks = len(transfer.participants)
        if transfer.link == "nvlink":
            bandwidth_key = "nvlink_bw_bytes_s"
        elif transfer.link == "infiniband":
            bandwidth_key = "infiniband_bw_bytes_s"
        else:
            raise ValueError(
                "analytical COMMUNICATION requires nvlink or infiniband, "
                f"got {transfer.link!r}")
        bandwidth = _required_interconnect(hardware, bandwidth_key)
        if transfer.operation == "all_reduce":
            network_bytes = (
                2.0 * (ranks - 1) / ranks * transfer.bytes)
        elif transfer.operation == "all_gather":
            network_bytes = (ranks - 1) * transfer.bytes
        elif transfer.operation == "send_recv":
            network_bytes = transfer.bytes
        elif transfer.operation == "all_to_all":
            network_bytes = (
                (ranks - 1) / ranks * transfer.bytes)
        else:
            raise ValueError(
                f"unknown COMMUNICATION operation: {transfer.operation!r}")
        data_ns = network_bytes / bandwidth * 1e9
        if (transfer.implementation
                == "sglang_flashinfer_allreduce_residual_rmsnorm"):
            # One fused kernel: all-reduce, residual add and RMSNorm.
            service_ns = hardware.sglang_flashinfer_allreduce_fusion_ns(
                ranks, transfer.bytes)
            return prediction_from_ns(
                max(service_ns, data_ns), "communication")
        if transfer.implementation == "sglang_custom_all_reduce_v2":
            # SGLang CustomAllReduceV2 below its max size; above it SGLang
            # falls back to NCCL, priced by the bandwidth term.
            service_ns = hardware.sglang_custom_all_reduce_service_ns(
                ranks, transfer.bytes)
            if service_ns is not None:
                return prediction_from_ns(
                    max(service_ns, data_ns), "communication")
            return prediction_from_ns(data_ns, "communication")
        if transfer.operation == "all_reduce" and transfer.link == "nvlink":
            # Pinned vLLM runs small NVLink all-reduces through its IPC
            # custom kernel; its measured service includes the cross-rank
            # synchronisation that a bandwidth-only term omits.
            service_ns = hardware.custom_all_reduce_service_ns(
                ranks, transfer.bytes)
            if service_ns is not None:
                return prediction_from_ns(
                    max(service_ns, data_ns), "communication")
        return prediction_from_ns(data_ns, "communication")

    def _predict_selected_backend(self, kernel: KernelSpec, hardware):
        backend = kernel.transfer.backend
        if backend == "analytical_roofline_v1":
            return self._analytical_roofline(kernel, hardware)
        elif backend == "protobridge":
            return predict_protobridge(kernel, hardware)
        else:
            raise ValueError(
                f"unknown COMMUNICATION backend: {backend!r}")

    def midlayer(self, kernel: KernelSpec, hardware):
        return self._predict_selected_backend(kernel, hardware)

    def roofline(self, kernel: KernelSpec, hardware):
        return self._predict_selected_backend(kernel, hardware)
