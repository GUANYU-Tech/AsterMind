"""The restored scene editor calls only V1 builder contracts."""
from copy import deepcopy
from pathlib import Path


def validate_document(root, document):
    from runtime_environment import activate
    activate()
    from astermind.builder.config import SimulationConfig
    from studio import physical_data
    value = deepcopy(document)
    experiment = value.get('experiment')
    if not isinstance(experiment, dict):
        raise ValueError('experiment 必须是完整 V1 配置')
    retired = {'communication_mode', 'timeline_mode', 'protobridge_startup_us'} & experiment.keys()
    if retired:
        raise ValueError('这是旧执行配置；V1 使用 compute_mode 与 communication，请重新生成配置：' + ', '.join(sorted(retired)))
    if value.get('physical_model_id'):
        data = physical_data(root, value['physical_model_id'])
        if not data['ready'] or Path(data['physical_dataset']).resolve() != Path(experiment.get('physical_dataset', '')).resolve():
            raise ValueError('场景引用的物理数据与所选已完成模型不一致')
    config = SimulationConfig.model_validate(experiment)
    config.topology()
    value['experiment'] = config.model_dump(mode='json')
    return value, value['experiment']


def generate(root, payload):
    from runtime_environment import enable_device_native
    upstream = enable_device_native()
    from astermind.builder.scene import build_scene
    from astermind.builder.config.deployment import SceneDeployment
    from astermind.builder.catalog import validate_deployment_choice
    from studio import physical_data
    if set(payload) - {'physical_model_id', 'deployment', 'requests', 'migrations', 'runtime_network'}:
        raise ValueError('生成配置包含未知字段')
    data = physical_data(root, payload['physical_model_id'])
    if not data['ready']:
        raise ValueError('请先生成完整物理数据包')
    options=deepcopy(payload.get('deployment', {}))
    options.setdefault('communication', {}).setdefault('intra_mode','IntraSatelliteRoofline')
    deployment = SceneDeployment.model_validate(options)
    hardware = validate_deployment_choice(deployment, upstream)
    defaults = {key: hardware[key] for key in ('gpu_link_bandwidth_gbps','gpu_link_latency_ns')
                if key not in deployment.model_fields_set and hardware[key] is not None}
    deployment = deployment.model_copy(update=defaults)
    return build_scene(data['physical_dataset'], deployment, requests=payload.get('requests',()),
                       migrations=payload.get('migrations',()), runtime_network=payload.get('runtime_network')).model_dump(mode='json')
