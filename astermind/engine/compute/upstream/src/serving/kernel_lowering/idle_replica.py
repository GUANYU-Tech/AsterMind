"""Logical payload estimates for bounded idle-worker source operations."""

from math import prod

from src.kernel.kernel_spec import KernelKind, KernelSpec


def price_idle_operator(node, hardware_id, memory):
    """Keep dummy dependencies and avoid charging nonexistent cache/state access."""
    attrs = node.attributes
    if attrs.get('source_actual_request_count') != 0 or hardware_id != 'nvidia_h200':
        return None
    component = attrs.get('source_component')
    if component == 'vllm_0_30_symbolic_v4_swa_insert':
        if (attrs.get('cache_write_semantics') != 'all_PAD_no_store'
                or attrs.get('symbolic_slot_mapping') != (-1,) or attrs.get('cache_write_valid_rows') != 0):
            raise ValueError('Idle V4 SWA requires one PAD slot and no cache write')
        shape = dict(node.shape)
        shape['element_count'] = shape.pop('elements')
        return (KernelSpec(KernelKind.KV_CACHE, hardware_id, shape, node.dtype, node.variant,
            attributes={**{key: attrs[key] for key in ('cache_layout', 'cache_write_semantics',
                'quantization', 'slot_mapping_all_valid')}, 'valid_token_count': 0,
                'mathematical_program': 'v4_idle_swa_insert_v1'},
            launch=None, execution_support='unsupported', unsupported_reason='launch_selector_unavailable'),)

    if attrs.get('operation') == 'fill' and node.node_id.endswith((
            'dummy:zero-IDs', 'dummy:zero-positions', 'dummy:PAD-slots',
            'dummy:positions-zero', 'dummy:input-IDs-zero')):
        expected = -1 if node.node_id.endswith('dummy:PAD-slots') else 0
        if (attrs.get('source_constant'), attrs.get('reads'), attrs.get('writes')) != (expected, 0, 1):
            raise ValueError('Idle worker input fill has a different source value')
        read_ports, output_count = (), 1
    elif component == 'vllm_0_30_symbolic_v4_sparse' and attrs.get('all_slot_mapping_PAD') is True:
        if node.variant == 'dsv4_dummy_swa_indices':
            if (attrs.get('indices_fill'), attrs.get('used_lengths_fill')) != (-1, 0):
                raise ValueError('Idle V4 indices must be empty')
            read_ports, output_count = (0,), 2
        elif node.variant == 'dsv4_sparse_decode':
            if (attrs.get('effective_KV_rows'), attrs.get('cache_read_bytes'), attrs.get('output_value')) != (0, 0, 0):
                raise ValueError('Idle V4 sparse decode must have empty visible cache')
            # Lengths determine the empty-list branch before any Q or cache reads.
            read_ports, output_count = (2,), 1
        else:
            return None
    elif component == 'vllm_0_30_symbolic_v4_compressor' and attrs.get('cache_write_semantics') == 'all_PAD_no_store':
        if attrs.get('cache_write_valid_rows') != 0:
            raise ValueError('Idle V4 compressor cannot write the request cache')
        read_ports, output_count = ((2,) if node.variant == 'dsv4_compressor_state_cache' else (0,)), 0
    elif component == 'vllm_0_30_kimi_k3_triton_decode' and attrs.get('source_null_state_skip') is True:
        if attrs.get('state_indices') != (0,):
            raise ValueError('Idle K3 skip requires the reserved null state')
        read_ports = (0,)
        output_count = 0 if attrs.get('stage') == 'convolution' else 1
        if output_count and (attrs.get('stage'), attrs.get('source_output_constant')) != ('recurrent', 0):
            raise ValueError('Idle K3 recurrence must write zeros')
    else:
        return None

    source = memory._by_node[node.node_id] if memory is not None else None
    if (source is None or len(source.outputs) != output_count
            or any(index >= len(source.inputs) for index in read_ports)):
        raise ValueError('Idle operator requires its current typed memory ports')
    reads = tuple(source.inputs[index] for index in read_ports)
    if any(view.dtype not in ('bf16', 'int32', 'int64') for view in (*reads, *source.outputs)):
        raise ValueError('Idle operator has an unsupported payload type')
    return tuple(KernelSpec(KernelKind.VECTOR, hardware_id,
        {'element_count': prod(view.shape)}, view.dtype, 'elementwise',
        attributes={'operation': operation, 'reads': int(operation == 'read'),
            'writes': int(operation == 'fill'), 'mathematical_program': 'vllm_idle_replica_payload_v1'},
        launch=None, execution_support='unsupported', unsupported_reason='launch_selector_unavailable')
        for operation, views in (('read', reads), ('fill', source.outputs)) for view in views)
