"""Read-only preflight for editable scenes; never starts a simulation."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'JSON 字段重复：{key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'JSON 不允许非有限数值：{value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def finite_json(value, path='配置'):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{path} 必须是有限数值')
    if isinstance(value, dict):
        for key, child in value.items():
            finite_json(child, f'{path}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_json(child, f'{path}[{index}]')


def validate_scene(root, payload):
    if not isinstance(payload, dict):
        raise ValueError('场景必须是 JSON 对象')
    finite_json(payload)
    if not isinstance(payload.get('name'), str) or not payload['name'].strip():
        raise ValueError('请填写场景名称')
    experiment = payload.get('experiment', {})
    if experiment.get('schema_version') not in (1, 'orbitserve.native_cells.v1') or 'cells' not in experiment:
        raise ValueError('该输入属于旧 Serving 引擎；请用当前物理数据与部署表单生成 V1 场景，原输入不会被改写')
    from native_scene import validate_document
    return validate_document(root, deepcopy(payload))
