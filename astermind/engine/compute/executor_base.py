"""Shared placement and task adaptation for the runtime batch executor."""
from astermind.engine.comm.operator import CommunicationOperator
from astermind.engine.comm.native_intra import native_enabled, bind_native_context, bind_native_broadcast_context, ERROR_PREFIX


class NetworkBatchExecutor:
    """Resolve rank placement and preserve the source communication contract."""

    def __init__(self, cell_id, loop, comm, placements, *, on_batch_boundary=None,
                 on_batch_complete=None):
        self.cell_id = cell_id
        self.loop = loop
        self.comm = comm
        self._on_batch_boundary = on_batch_boundary
        self._on_batch_complete = on_batch_complete
        self.placements = {}
        device_ids = set()
        for placement in placements:
            if placement.cell_id != cell_id:
                raise ValueError("rank placement belongs to another Cell")
            identity = (cell_id, placement.pool_name, placement.rank)
            if identity in self.placements or placement.gpu_device_id in device_ids:
                raise ValueError("duplicate rank placement or shared physical GPU")
            self.placements[identity] = placement
            device_ids.add(placement.gpu_device_id)
        self._active = set()
        self.records = []
        self.input_distributions = []

    def _device(self, pool, rank):
        return self.placements[(self.cell_id, pool, rank)].gpu_device_id

    def _operator(self, call, execution=None, step_id=None):
        try:
            return self._build_operator(call, execution, step_id)
        except Exception as error:
            if native_enabled(getattr(self.comm, 'policy', None), self.cell_id) and not str(error).startswith(ERROR_PREFIX):
                raise ValueError(ERROR_PREFIX + f'{type(error).__name__}: {error}') from error
            raise

    def _build_operator(self, call, execution=None, step_id=None):
        use_native = native_enabled(getattr(self.comm, 'policy', None), self.cell_id)
        transfer = call.kernel_spec.transfer
        if transfer is None:
            raise ValueError("communication kernel has no transfer contract")
        if not use_native and (transfer.implementation or transfer.destination_hardware_id):
            raise NotImplementedError("specialized source hardware needs a verified surrogate")
        if not use_native and len(set(transfer.participant_nic_ids))!=len(transfer.participant_nic_ids):
            raise NotImplementedError("shared source NIC needs a bound capacity model")
        ranks = transfer.participants
        if transfer.operation == "send_recv":
            ranks = (transfer.source_rank, transfer.destination_rank)
        elif transfer.operation == "broadcast" and transfer.source_rank is not None:
            ranks = (transfer.source_rank, *(rank for rank in ranks if rank != transfer.source_rank))
        task = CommunicationOperator(
            operation=transfer.operation, algorithm=None,
            participants=tuple(self._device(call.pool_name, rank) for rank in ranks),
            bytes=transfer.bytes, bytes_semantics=transfer.bytes_semantics,
            ring_order=(),
            reduction=transfer.reduction,
            partition_bytes=getattr(transfer, "partition_bytes", getattr(transfer, "owner_sizes", ())),
            traffic_matrix_bytes=getattr(transfer, "traffic_matrix_bytes", ()),
            cell_id=self.cell_id,
            task_id=f'{self.cell_id}/{call.pool_name}/{step_id}/{call.call_id}',
        )
        if use_native:
            if execution is None:
                raise ValueError('ArchPrism原生roofline通信预测不支持: missing execution hardware context')
            bind_native_context(self.comm.network, task, call.kernel_spec,
                                execution.hardware_by_id[call.kernel_spec.hardware_id])
        return task

    def _bind_input_distribution(self, task, execution, pool):
        if native_enabled(getattr(self.comm, 'policy', None), self.cell_id):
            try:
                bind_native_broadcast_context(self.comm.network, task, execution.hardware_by_id[pool.hardware_id])
            except Exception as error:
                raise ValueError(ERROR_PREFIX + str(error)) from error
