"""Observe public configuration contracts through the original import paths."""
import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import get_type_hints

from structure_test_support import run_suites, write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from pydantic import BaseModel, ValidationError
    from astermind.common import to_jsonable
    from astermind.builder.config import SimulationConfig, Parallelism, Request
    from astermind.engine.setup.network_spec import NetworkSpec
    from tests.unit.runtime.test_dynamic_network import network
    names = {
        'astermind.builder.config': ('Parallelism', 'Placement', 'ServingCell', 'Request', 'Migration', 'CellLinkAddition', 'SimulationConfig'),
        'astermind.builder.models': ('Cell', 'Satellite', 'Device', 'AvailabilityWindow', 'OSReferenceLayout', 'LinkPolicyConfig', 'Link', 'Template', 'Scenario'),
        'astermind.builder.scene': ('SceneDeployment',),
        'astermind.engine.setup.network_spec': ('NetworkNodeSpec', 'LinkProfile', 'NetworkSpec'),
        'astermind.engine.comm.frame_surrogate': ('CommPolicy',),
        'astermind.engine.comm.policy': ('CellCommPolicy',),
    }
    definitions = {}
    for module, symbols in names.items():
        imported = importlib.import_module(module)
        for name in symbols:
            cls = getattr(imported, name)
            assert pickle.loads(pickle.dumps(cls)) is cls
            key = module+'.'+name
            value = dict(module=cls.__module__, qualname=cls.__qualname__,
                type_hints={key:repr(value) for key,value in get_type_hints(cls, include_extras=True).items()})
            if issubclass(cls, BaseModel):
                value['schema'] = cls.model_json_schema()
            else:
                value['fields'] = [f.name for f in dataclasses.fields(cls)]
            definitions[key] = value
    configs = {}
    for path in sorted((args.repo/'examples/native').glob('*.json')):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or not {'physical', 'cells', 'requests'} <= raw.keys():
            continue
        config = SimulationConfig.model_validate(raw)
        restored = pickle.loads(pickle.dumps(config))
        assert type(restored) is SimulationConfig and restored.model_dump() == config.model_dump()
        configs[path.name] = config.model_dump(mode='json')
    assert configs
    spec = network().spec
    assert NetworkSpec.from_value(spec.to_dict()) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec
    failures = []
    for cls, raw in [(Parallelism, {'sp': 2}), (Parallelism, {'tp': 2, 'ep': 3}),
                     (Request, dict(id='r', user_device='g', input_tokens=1, context_id='c'))]:
        try:
            cls.model_validate(raw)
        except ValidationError as error:
            failures.append(json.loads(error.json(include_url=False)))
        else:
            raise AssertionError('invalid configuration was accepted')
    modules = list(names)
    if (args.repo/'astermind/builder/config').is_dir():
        # Start each new domain in a fresh process as well as the old API roots.
        for name in ('scenario','resources','network','deployment','requests','communication'):
            subprocess.run([sys.executable, '-B', '-c',
                f'import astermind.builder.config.{name}; import astermind.engine; import astermind.builder.scene'],
                cwd=args.repo, check=True)
    for order in (modules, list(reversed(modules))):
        subprocess.run([sys.executable, '-B', '-c',
            'import importlib; [importlib.import_module(m) for m in '+repr(order)+']'],
            cwd=args.repo, check=True)
    result = dict(definitions=definitions, examples=configs, network=to_jsonable(spec),
        rejected=failures, request_input_bytes_default=Request(id='r', user_device='g', input_tokens=1).input_bytes,
        suites=run_suites(['tests.unit.runtime.test_request_batch',
            'tests.unit.runtime.test_request_prerequisites.PrerequisiteConfigTests',
            'tests.unit.runtime.test_scene_catalog', 'tests.unit.runtime.test_islsim_config',
            'tests.smoke.test_public_examples', 'tests.unit.test_module_boundaries']))
    write_result(args.output, result)


if __name__ == '__main__':
    main()
