"""Personal, versioned physical models and complete experiment scenes."""
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import re
import uuid
from zoneinfo import ZoneInfo


def key(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', value):
        raise ValueError('invalid local document ID')
    return value


def load(root, kind, identity):
    if kind not in {'physical', 'scenes'}:
        raise ValueError('unknown document collection')
    from scene_validation import strict_json
    return strict_json((Path(root) / 'local/inputs' / kind / (key(identity) + '.json')).read_text(encoding='utf-8-sig'))


def catalog(root, kind):
    directory = Path(root) / 'local/inputs' / kind
    rows = []
    for path in sorted(directory.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if kind == 'scenes':
                from maintenance import scene_problem
                if scene_problem(root,data):
                    continue
        except (OSError,ValueError,KeyError,TypeError):
            continue
        if all(k in data for k in ('id','name','saved_at')):
            rows.append({k: data[k] for k in ('id', 'name', 'saved_at')})
    return rows


def physical_catalog(root):
    rows = catalog(root, 'physical')
    inputs = Path(root) / 'local/inputs'
    registration = inputs / 'examples/case-study-iii.json'
    if not registration.is_file():
        return rows
    try:
        example = json.loads(registration.read_text(encoding='utf-8'))
        scene_id = key(example['scene_id'])
        model_id = key(example['physical_model_id'])
    except (OSError, ValueError, KeyError, TypeError):
        return rows
    if (inputs / 'scenes' / (scene_id + '.json')).is_file() and (
            inputs / 'physical-data' / (model_id + '.json')).is_file():
        for row in rows:
            if row['id'] == model_id:
                row['case_study_iii_scene_id'] = scene_id
                break
    return rows


def physical_data(root, identity):
    """Persist dataset ownership separately from immutable configuration revisions."""
    identity = key(identity)
    reference = Path(root) / 'local/inputs/physical-data' / (identity + '.json')
    if not reference.is_file():
        return dict(state='missing', ready=False, physical_model_id=identity)
    value = json.loads(reference.read_text(encoding='utf-8'))
    folder = Path(root) / 'local/results/network' / key(value['run_id'])
    from maintenance import dataset_folder, dataset_islsim_problem, dataset_problem
    import console
    if Path(root).resolve() == console.ROOT.resolve():
        job = console.run_status(value['run_id'])
    else:
        job = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
        if job['state'] == 'running':
            job['state'] = 'interrupted'
    folder = dataset_folder(folder)
    problem = (dataset_problem(folder) or dataset_islsim_problem(folder)) if job['state']=='completed' else None
    ready = job['state'] == 'completed' and problem is None
    result = dict(value, state=job['state'], ready=ready, physical_dataset=str(folder),
                  progress_url=f"/runs/{value['run_id']}/progress")
    if problem:
        result.update(state='outdated',reason=problem)
    if ready:
        data = json.loads((folder / 'physical_dataset.json').read_text(encoding='utf-8'))
        result['input_sha256'] = data['input_sha256']
        result['physics_version'] = data['physics_version']
    return result


def bind_dataset(root, value):
    from astermind.builder.physical.physical_dataset import validate_dataset_input, input_fingerprint
    from astermind.builder.physical.network_inputs import snapshot_network_input
    experiment = value['experiment']
    physical = deepcopy(experiment['physical_network'])
    # Scene network settings are authoritative. The outer key is read only
    # for older saved scenes that did not yet have the canonical field.
    settings = physical['network'].get('random_faults', value.get('random_faults', {}))
    physical['network']['fault_timing'] = 'continuous'
    physical['network']['random_faults'] = deepcopy(settings)
    validate_faults(settings, float(physical['time']['duration_s']), float(physical['time']['step_s']))
    from types import SimpleNamespace
    from astermind.engine.setup.faults import validate_fault_boundaries
    duration, step = (float(physical['time'][k]) for k in ('duration_s', 'step_s'))
    validate_fault_boundaries(SimpleNamespace(fault_timing='continuous', faults=physical['network'].get('faults', ()),
                                             random_faults=settings),
                              tuple(i * step for i in range(round(duration / step) + 1)))
    identity = value.get('physical_model_id')
    if identity:
        model = load(root, 'physical', identity)
        if input_fingerprint(snapshot_network_input(physical, root)) != input_fingerprint(snapshot_network_input(model['physical_network'], root)):
            raise ValueError('物理配置与数据包不匹配，请返回第一步重新生成')
        data = physical_data(root, identity)
        if not data['ready']:
            raise ValueError('请先在第一步生成完整物理数据，再配置和运行第二步')
        experiment['physical_dataset'] = data['physical_dataset']
    reference = experiment.get('physical_dataset')
    if not reference:
        raise ValueError('请选择第一步已生成的完整物理数据包')
    validate_dataset_input(reference, physical, root)
    return physical


def validate_model(physical, root):
    from astermind.builder.physical.network_inputs import materialize_network_input
    from astermind.engine import NetworkSpec, candidate_links
    concrete = materialize_network_input(physical, root)
    spec = NetworkSpec.from_value(concrete['network'])
    edges = candidate_links(spec)
    from astermind.engine.setup.radio_capacity import hop_pools
    nodes = {n.node_id: n for n in spec.nodes}
    for edge in edges:
        for source, target in ((edge['endpoint_a'], edge['endpoint_b']), (edge['endpoint_b'], edge['endpoint_a'])):
            hop_pools(nodes, source, target)
    ids = {row['link_id'] for row in edges}
    if any(f['target_type'] == 'link' and f['target_id'] not in ids for f in spec.faults):
        raise ValueError('固定故障链路不属于当前拓扑，请清除旧故障或修改目标')
    duration, step = (float(physical['time'][k]) for k in ('duration_s', 'step_s'))
    if not math.isfinite(duration) or not math.isfinite(step) or duration <= 0 or step <= 0:
        raise ValueError('实验时长和采样间隔必须为有限正数')
    from astermind.engine.setup.faults import validate_fault_boundaries
    count = round(duration / step)
    validate_fault_boundaries(spec, tuple(i * step for i in range(count + 1)))
    grounds=physical['ground_ingresses']
    if not grounds or len({g['ingress_id'] for g in grounds})!=len(grounds):
        raise ValueError('至少需要一个地面站，且 ID 不能重复')
    for ground in grounds:
        lat,lon=ground['lat_deg'],ground['lon_deg']
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (lat,lon,ground.get('altitude_km',0),ground.get('min_elevation_deg',10))) or not -90<=lat<=90 or not -180<=lon<=180 or not 0<=ground.get('min_elevation_deg',10)<=90:
            raise ValueError('地面站坐标、仰角必须有限且在有效范围内')
    return concrete, spec, edges


def validate_faults(settings, duration, step=None):
    from astermind.engine.setup.faults import validate_random_faults, random_fault_window, random_faults_enabled
    validate_random_faults(settings)
    if not random_faults_enabled(settings):
        return
    if step is not None:
        random_fault_window(settings, tuple(i * step for i in range(round(duration / step) + 1)))
        return
    if 'start_step' in settings:
        raise ValueError('校验时间片编号需要物理模型的 step_s')
    start, end = (float(settings[k]) for k in ('start_s', 'end_s'))
    if not 0 <= start < end <= duration:
        raise ValueError('随机故障时间须在实验范围内且开始早于结束')


def realize_faults(physical, settings, root):
    physical = deepcopy(physical)
    from astermind.engine.setup.faults import random_faults_enabled
    if not random_faults_enabled(settings):
        return physical
    concrete, spec, edges = validate_model(physical, root)
    from dataclasses import replace
    from astermind.engine.setup.faults import realize_runtime_faults
    duration, step = (float(physical['time'][k]) for k in ('duration_s', 'step_s'))
    boundaries = tuple(i * step for i in range(round(duration / step) + 1))
    realized = realize_runtime_faults(replace(spec, random_faults=settings), boundaries, edges)
    physical['network']['faults'] = list(realized.faults)
    return physical


def save(root, kind, payload):
    value = deepcopy(payload)
    if not isinstance(value.get('name'),str) or not value['name'].strip():
        raise ValueError('请填写名称')
    if kind == 'physical':
        from astermind.engine import physical_network_settings
        physical = value['physical_network']
        physical['network'] = physical_network_settings(physical['network'])
        value.pop('random_faults', None)
        compute = value.get('compute_configuration', {})
        if 'link_bandwidth_factors' in compute:
            from astermind.builder.config.communication import CommPolicy
            if compute.get('mode') not in ('fast_mode', 'batch_roofline'):
                raise ValueError('有效可用带宽比例仅适用于 Fast Mode')
            compute['link_bandwidth_factors'] = CommPolicy(
                link_bandwidth_factors=compute['link_bandwidth_factors']).link_bandwidth_factors
        validate_model(physical, root)
    elif kind == 'scenes':
        from scene_validation import validate_scene
        value, _ = validate_scene(root, value)
    else:
        raise ValueError('unknown collection')
    # New immutable revision on every save; an existing scene is never overwritten.
    value.update(id=uuid.uuid4().hex, saved_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), schema_version=f'orbitserve.local.{kind}.v1')
    path = Path(root) / 'local/inputs' / kind / (value['id'] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return value


def scene_input(root, identity):
    from scene_validation import validate_scene
    _, experiment = validate_scene(root, load(root, 'scenes', identity))
    return experiment


def physical_input(root, identity):
    from astermind.engine import physical_network_settings
    document = load(root, 'physical', identity)
    physical = deepcopy(document['physical_network'])
    # Reusable physics never bakes runtime failure outcomes into the dataset.
    physical['network'] = physical_network_settings(physical['network'])
    # Interactive probes are queried on demand; avoid a route matrix per frame.
    physical['route_probes'] = []
    return physical
