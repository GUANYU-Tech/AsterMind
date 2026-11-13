"""Explicit Comm adapter for ArchPrism's native communication roofline.

Source KernelSpec/HWSpec objects stay in a side registry on the owning Network;
they never enter task JSON. The native predictor is invoked only by Comm.
"""
from dataclasses import replace
from itertools import permutations
from math import ceil, isfinite
from .estimator import Estimate

ERROR_PREFIX = 'ArchPrism原生roofline通信预测不支持: '


def native_enabled(policy, cell_id):
    selected = getattr(policy, 'cell_policies', {}).get(cell_id)
    return (getattr(selected, 'intra_mode', None) or getattr(policy, 'intra_mode', 'surrogate')) == 'IntraSatelliteRoofline'


def bind_native_context(network, task, kernel_spec, hardware):
    if not task.cell_id or not task.task_id:
        raise ValueError(ERROR_PREFIX + 'native source context requires cell_id and task_id')
    if not hasattr(network, '_comm_native_contexts'):
        network._comm_native_contexts = {}
    network._comm_native_contexts[task.cell_id, task.task_id] = (kernel_spec, hardware)


def bind_native_broadcast_context(network, task, hardware):
    """Comm-created input broadcast has no source kernel; retain real HWSpec."""
    if task.operation != 'broadcast':
        raise ValueError(ERROR_PREFIX + 'synthetic context is only for input broadcast')
    bind_native_context(network, task, None, hardware)


def clear_native_contexts(network, cell_id, task_prefix):
    registry = getattr(network, '_comm_native_contexts', {})
    for key in tuple(registry):
        if key[0] == cell_id and key[1].startswith(task_prefix):
            registry.pop(key)


def estimate_native(task, routes, network):
    """Call the real native predictor, preserving service tables and failures.

The caller must select this adapter explicitly. This adapter never falls back
to surrogate. It retains the native hardware profile's link assumptions;
physical routes are checked for same-satellite reachability, not calibrated.
"""
    try:
        if len(task.participants) < 2 or len(set(task.participants)) != len(task.participants):
            raise ValueError('native communication requires distinct participants')
        if type(task.bytes) is not int or task.bytes < 0:
            raise ValueError('native communication requires nonnegative integer bytes')
        if not task.cell_id or not task.task_id:
            raise ValueError('missing Cell/task source identity')
        context = getattr(network, '_comm_native_contexts', {}).get((task.cell_id, task.task_id))
        if context is None:
            raise ValueError('missing original ArchPrism KernelSpec/HWSpec context')
        source, hardware = context
        devices = routes.topology.devices
        if len({devices[g].satellite for g in task.participants}) != 1:
            raise ValueError('IntraSatelliteRoofline accepts only whole same-satellite tasks')
        pairs = ((task.participants[0], b) for b in task.participants[1:]) if task.operation == 'broadcast' else permutations(task.participants, 2)
        paths = tuple(routes.route(a, b, task.bytes) for a, b in pairs)
        owner = devices[task.participants[0]].satellite
        if any(devices[n].satellite != owner for path in paths for n in path['path']):
            raise ValueError('intra route leaves the source satellite')
        if task.operation == 'broadcast':
            if source is not None or task.algorithm not in ('direct', 'pairwise') or task.bytes_semantics != 'payload':
                raise ValueError('native input broadcast requires Comm direct/pairwise fanout and payload bytes')
            if task.reduction is not None or task.partition_bytes or task.traffic_matrix_bytes or task.ring_order:
                raise ValueError('native broadcast does not accept reduction/partitions/matrix/ring')
            from src.kernel.kernel_spec import KernelKind, KernelSpec, PhysicalTransferConfig
            from src.kernel.predictors.communication import CommunicationPredictor
            durations = []
            for _ in paths:
                if task.bytes == 0:
                    durations.append(0.)
                    continue
                transfer = PhysicalTransferConfig(algorithm='send_recv', bytes=task.bytes,
                    direction='point_to_point', link='nvlink', topology='point_to_point',
                    participants=(0, 1), operation='send_recv', bytes_semantics='payload',
                    source_rank=0, destination_rank=1, backend='analytical_roofline_v1',
                    resource_id='comm-input-distribution')
                kernel = KernelSpec(KernelKind.COMMUNICATION, hardware.hardware_id, transfer=transfer)
                durations.append(CommunicationPredictor().predict(kernel, hardware, method='roofline').t_total_s * 1e9)
            ns = max(durations, default=0.)
            if not isfinite(ns) or ns < 0:
                raise ValueError('native predictor returned invalid broadcast duration')
            return Estimate(ceil(ns), 'intra_archprism_native_broadcast_fanout', int(bool(paths)), len(paths), 0, 0, paths)
        transfer = source.transfer
        if transfer is None or transfer.operation != task.operation or transfer.bytes != task.bytes:
            raise ValueError('native source task does not match communication operation/bytes')
        if len(transfer.participants) != len(task.participants):
            raise ValueError('native source participant count mismatch')
        if transfer.bytes_semantics != task.bytes_semantics:
            raise ValueError('native source byte semantics mismatch')
        if transfer.reduction != task.reduction:
            raise ValueError('native source reduction mismatch')
        if tuple(transfer.partition_bytes) != tuple(task.partition_bytes) or tuple(transfer.traffic_matrix_bytes) != tuple(task.traffic_matrix_bytes):
            raise ValueError('native source partition/matrix mismatch')
        algorithm = task.algorithm
        if algorithm == 'hierarchical':
            raise NotImplementedError('native communication predictor has no hierarchical algorithm; configure a native-supported algorithm')
        if task.operation == 'send_recv':
            algorithm, topology, order = 'send_recv', 'point_to_point', ()
        elif task.operation in ('all_reduce', 'all_gather'):
            topology = {'ring':'ring', 'tree':'tree', 'fullmesh':'fully_connected', 'fc2':'fully_connected', 'mesh':'fully_connected'}[algorithm]
            order = tuple(transfer.participants) if algorithm == 'ring' else ()
        elif task.operation in ('all_to_all', 'all_gatherv', 'reduce_scatterv'):
            if algorithm not in ('pairwise', 'direct'):
                raise NotImplementedError(f'unsupported native algorithm {algorithm}')
            algorithm, topology, order = 'pairwise', 'fully_connected', ()
        else:
            raise NotImplementedError(f'no native source operation {task.operation}')
        selected = replace(transfer, algorithm=algorithm, topology=topology,
                           ring_order=order, backend='analytical_roofline_v1', backend_config_sha256=None)
        kernel = replace(source, transfer=selected)
        from src.kernel.predictors.communication import CommunicationPredictor
        result = CommunicationPredictor().predict(kernel, hardware, method='roofline')
        ns = result.t_total_s * 1e9
        if not isfinite(ns) or ns < 0:
            raise ValueError('native predictor returned invalid duration')
        return Estimate(ceil(ns), 'intra_archprism_native_roofline', 0, 0, 0, 0, paths)
    except Exception as error:
        getattr(network, '_comm_native_contexts', {}).pop((task.cell_id, task.task_id), None)
        if str(error).startswith(ERROR_PREFIX):
            raise
        raise ValueError(ERROR_PREFIX + f'{type(error).__name__}: {error}') from error
