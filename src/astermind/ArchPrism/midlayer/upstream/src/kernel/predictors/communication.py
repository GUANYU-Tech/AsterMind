"""Zero-fit analytical communication Roofline and backend dispatch."""

from collections import Counter

from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.common import prediction_from_ns
from src.kernel.predictors.protobridge.adapter import predict_protobridge


class CommunicationPredictor(KernelPredictor):
    kind = KernelKind.COMMUNICATION

    def _analytical_roofline(self, kernel: KernelSpec, hardware):
        transfer = kernel.transfer
        from src.kernel.memory_communication import has_communication_source
        if has_communication_source(kernel):
            from src.kernel.memory_access_builder import communication_memory_access_program
            from src.kernel.memory_path_solver import collective_source_work
            payload, ranks, operation = collective_source_work(
                communication_memory_access_program(kernel, hardware))
            if operation != transfer.operation:
                raise ValueError("collective source operation does not match the call")
        else:
            # Fused residual/RMSNorm has extra operands and remains outside
            # this source domain; preserve its existing support.
            payload, ranks = transfer.bytes, len(transfer.participants)
        bandwidth = hardware.analytical_link_bandwidth_bytes_s(transfer.link)
        if transfer.operation == "all_reduce":
            network_bytes = (
                2.0 * (ranks - 1) / ranks * payload)
        elif transfer.operation == "all_gather":
            network_bytes = (ranks - 1) * payload
        elif transfer.operation == "send_recv":
            network_bytes = payload
        elif transfer.operation == "all_to_all":
            network_bytes = (
                (ranks - 1) / ranks * payload)
        else:
            raise ValueError(
                f"unknown COMMUNICATION operation: {transfer.operation!r}")
        data_ns = network_bytes / bandwidth * 1e9
        if transfer.link == "roce":
            # A full-duplex NIC has independent TX/RX capacity. Each ring
            # participant sends and receives the same logical byte volume;
            # max NIC load, rather than rank count or switch port count,
            # sets the bandwidth term. IDs describe physical placement.
            if not hardware.roofline_only:
                raise NotImplementedError("RoCE requires a declared aggregate NIC model")
            loads = Counter(transfer.participant_nic_ids)
            if max(loads.values()) > hardware.interconnect["roce_devices_per_nic"]:
                raise ValueError("RoCE NIC placement exceeds declared devices per NIC")
            nic_by_rank = dict(zip(transfer.participants, transfer.participant_nic_ids))
            if transfer.operation in ("all_reduce", "all_gather"):
                ring = transfer.ring_order
                if any(nic_by_rank[rank] == nic_by_rank[ring[(i + 1) % ranks]]
                       for i, rank in enumerate(ring)):
                    raise NotImplementedError("RoCE ring includes an unknown same-NIC local transfer path")
            elif transfer.operation in ("send_recv", "all_to_all"):
                if len(loads) != ranks:
                    raise NotImplementedError("RoCE transfer includes an unknown same-NIC local transfer path")
            else:
                raise AssertionError("validated RoCE operation disappeared")
            network_ns = data_ns * max(loads.values())
            signal_ns = hardware.interconnect["roce_signal_latency_ns"]
            if signal_ns is not None:
                # One-way propagation per dependent algorithm round. Startup
                # and staging DDR are distinct services, not included here.
                if transfer.operation == "all_reduce":
                    rounds = 2 * (ranks - 1)
                elif transfer.operation in ("all_gather", "all_to_all"):
                    rounds = ranks - 1
                else:
                    rounds = 1
                network_ns += rounds * signal_ns
            return prediction_from_ns(network_ns, "communication")
        if (transfer.implementation
                == "sglang_flashinfer_allreduce_residual_rmsnorm"):
            # One fused kernel: all-reduce, residual add and RMSNorm.
            service_ns = hardware.sglang_flashinfer_allreduce_fusion_ns(
                ranks, payload)
            return prediction_from_ns(
                max(service_ns, data_ns), "communication")
        if transfer.implementation == "sglang_custom_all_reduce_v2":
            # SGLang CustomAllReduceV2 below its max size; above it SGLang
            # falls back to NCCL, priced by the bandwidth term.
            service_ns = hardware.sglang_custom_all_reduce_service_ns(
                ranks, payload)
            if service_ns is not None:
                return prediction_from_ns(
                    max(service_ns, data_ns), "communication")
            return prediction_from_ns(data_ns, "communication")
        if transfer.operation == "all_reduce" and transfer.link == "nvlink":
            # Pinned vLLM runs small NVLink all-reduces through its IPC
            # custom kernel; its measured service includes the cross-rank
            # synchronisation that a bandwidth-only term omits.
            service_ns = hardware.custom_all_reduce_service_ns(
                ranks, payload)
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
