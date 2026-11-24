"""Compile the shared Cell editor against the satellite and ground IDs of a scene."""
from copy import deepcopy
from itertools import accumulate


def editor_layout(identity, members, row, base, requests, catalog):
    from astermind.builder.config.deployment import SceneDeployment, ServingCell
    from astermind.builder.scene import CellLayout
    from cell_adapter import compile_serving_cell, schema_validate
    from compute_options import su_groups

    value = deepcopy(row['configuration'])
    value.pop('requests', None)  # Legacy physical models may still contain a workload.
    schema = catalog['schema']
    schema_validate(value, schema={**schema, 'required': [key for key in schema['required'] if key != 'requests']})
    hw = value['hardware']
    ids = hw['satellite_ids']
    if len(ids) != len(members) or set(ids) != set(members) or hw['satellite_count'] != len(ids):
        raise ValueError('编队卫星清单与物理场景不一致')
    if value['compute_mode'] != base.compute_mode:
        raise ValueError('编队与全场景的计算模式必须一致')
    if value['topology']['intra_satellite']['type'] != 'profile_su':
        raise ValueError('全场景算力编队使用 SU 分域互联')
    profile = catalog['su_profiles'][hw['default_hardware_id']]
    cards = hw['accelerators_per_satellite']
    if not 1 <= cards <= 64:
        raise ValueError('全场景每星 GPU 数必须为 1 到 64')
    limit = profile['fast_gpu_limit_per_satellite']
    sizes = su_groups(hw['su_group_sizes'], cards, limit)
    stops = list(accumulate(sizes, initial=0))
    hierarchy = {sat: {'su_groups': [[f'{sat}/gpu{i}' for i in range(a, b)]
                                     for a, b in zip(stops, stops[1:])]} for sat in ids}
    longest = max((r['input_tokens'] + r['output_tokens'] for r in requests
                   if r.get('cell') in (None, identity)), default=0)
    value['ep_strategy'] = base.communication.get('ep_strategy', 'auto')
    cell, comm, method = compile_serving_cell(value, {'intra_hierarchies': hierarchy}, longest,
                                              cell_id=identity, scene=True, catalog=catalog)
    cell['detail_prediction_method'] = method
    policy = comm['cell_policies'][identity]
    policy['intra_mode'] = comm['intra_mode']
    deploy = SceneDeployment(compute_mode=base.compute_mode, model=cell['model'], hardware=cell['hardware'],
                             hardware_id=cell['hardware_id'], dp=cell['parallel']['dp'], gpus_per_satellite=cards,
                             gpu_link_bandwidth_gbps=profile['gpu_link_bandwidth_gbps'] or hw['intra_satellite_bandwidth_gbps'],
                             gpu_link_latency_ns=profile['gpu_link_latency_ns'],
                             communication={'cell_policies': {identity: policy}})
    uplink = (hw['intra_satellite_bandwidth_gbps'], hw['intra_satellite_latency_ns']) if len(sizes) > 1 else None
    return CellLayout(deploy, dict.fromkeys(ids, cards), scale_up_groups=sizes, scale_out=uplink,
                      su_limit=limit, serving_cell=ServingCell.model_validate(cell))
