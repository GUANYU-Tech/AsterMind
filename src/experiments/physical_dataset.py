"""Publish and reuse complete, backend-independent physical datasets."""

import hashlib
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

from .network_inputs import snapshot_network_input

SCHEMA = "orbitserve.physical_dataset.v2"
LEGACY_SCHEMA = "orbitserve.physical_dataset.v1"
PHYSICS_VERSION = "sample_hold_directional_solar_visibility_islsim_v4"


def input_fingerprint(concrete, *, legacy=False):
    value = deepcopy(concrete)
    if legacy:
        value.get('network', {}).pop('transport_model', None)
    else:
        from orbitserve.runtime import physical_network_settings, freeze_network_profiles
        # Raw inputs omitting the default model and already-frozen inputs must
        # identify the same physics, including upstream parameters/source hashes.
        value['network'] = physical_network_settings(freeze_network_profiles(value.get('network', {})))
    value.pop("route_probes", None)
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_dataset(root, concrete, *, source_input=None, file_hashes=None):
    root = Path(root)
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and (
            p.parts[len(root.parts)] in {"orbit_trace", "state"}
            or p.name
            in {
                "core_result.json",
                "materialized_input.json",
                "physical_state.sqlite",
            }
        )
    ]
    value = {
        "schema_version": SCHEMA,
        "physics_version": PHYSICS_VERSION,
        "input_sha256": input_fingerprint(concrete),
        "source_sha256": input_fingerprint(
            source_input if source_input is not None else concrete
        ),
        "files": {p.relative_to(root).as_posix(): (file_hashes or {}).get(str(p.resolve())) or digest(p)
                  for p in sorted(files) if not p.name.endswith('.tmp')},
        "optional_exports": "physical_exports.json",
    }
    temporary = root / "physical_dataset.json.tmp"
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(root / "physical_dataset.json")
    return value


def publish_exports(root, result, file_hashes):
    from orbitserve.common import write_json
    root = Path(root)
    value = {'schema_version': 'orbitserve.physical_exports.v1',
             'trace_id': result['trace_id'], 'artifacts': result['artifacts'],
             'timing_s': result['timing_s'],
             'files': {Path(p).relative_to(root.resolve()).as_posix(): sha
                       for p, sha in sorted(file_hashes.items())}}
    write_json(root / 'physical_exports.json', value)
    return value


def _verify_files(root, files):
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"physical dataset missing or corrupt: {name}")


def load_dataset(reference, concrete=None, *, verify_exports=True):
    root = Path(reference)
    if root.is_file():
        root = root.parent
    index = root / "physical_dataset.json"
    if not index.is_file():
        raise ValueError("完整物理数据尚未就绪，请在第一步完成物理预计算")
    value = json.loads(index.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") not in {SCHEMA, LEGACY_SCHEMA}
        or value.get("physics_version") != PHYSICS_VERSION
    ):
        raise ValueError("物理数据版本已过期，请在第一步重新生成")
    files = value["files"]
    required = {
            "physical_state.sqlite",
            "materialized_input.json",
            "orbit_trace/trace_manifest.json",
    }
    legacy = value['schema_version'] == LEGACY_SCHEMA
    if legacy:
        required.update({'network_result.json', 'node_positions.jsonl.gz', 'link_attributes.jsonl.gz',
                         'link_events.jsonl.gz', 'astermind_network.jsonl.gz'})
    else:
        required.update({'core_result.json', 'state/index.json', 'state/timeline.json'})
    if not required <= files.keys():
        raise ValueError("incomplete physical dataset")
    _verify_files(root, files)
    stored = json.loads((root / "materialized_input.json").read_text(encoding="utf-8"))
    if value['input_sha256'] not in {input_fingerprint(stored), input_fingerprint(stored, legacy=True)}:
        raise ValueError("physical dataset input identity mismatch")
    if concrete is not None and input_fingerprint(stored) != input_fingerprint(concrete):
        raise ValueError("物理配置与数据包不匹配，请返回第一步重新生成")
    result = json.loads((root / ('network_result.json' if legacy else 'core_result.json')).read_text(encoding="utf-8"))
    from .network import TERMINAL_ALLOCATION_MODEL

    if (
        result.get("provenance", {}).get("terminal_allocation_model")
        != TERMINAL_ALLOCATION_MODEL
        or result.get("provenance", {}).get("between_samples") != "sample_hold"
    ):
        raise ValueError("物理模型语义已过期，请在第一步重新生成")
    if not legacy and verify_exports and (root / 'physical_exports.json').is_file():
        exports = json.loads((root / 'physical_exports.json').read_text(encoding='utf-8'))
        if exports.get('schema_version') != 'orbitserve.physical_exports.v1' or exports.get('trace_id') != result['trace_id']:
            raise ValueError('physical exports identity mismatch')
        _verify_files(root, exports['files'])
        result['artifacts'].update(exports['artifacts'])
        result['timing_s'].update(exports['timing_s'])
        result['snapshots'] = result['artifacts']['frames'][:-1]
        files = {**files, **exports['files']}
    if any(frame["artifact"] not in files for frame in result["artifacts"]["frames"]):
        raise ValueError("incomplete physical dataset frames")
    return root.resolve(), value, result


def reuse_dataset(reference, destination, concrete, *, base_dir=".", include_exports=True):
    source, manifest, result = load_dataset(Path(base_dir) / reference, concrete, verify_exports=include_exports)
    target = Path(destination)
    if source != target.resolve():
        target.mkdir(parents=True, exist_ok=True)
        # Share immutable bytes on the same filesystem; portable copy fallback.
        names = [*manifest['files'], 'physical_dataset.json']
        if include_exports and (source / 'physical_exports.json').is_file():
            exports = json.loads((source / 'physical_exports.json').read_text(encoding='utf-8'))
            names.extend([*exports['files'], 'physical_exports.json'])
        for name in dict.fromkeys(names):
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                if digest(output) != digest(source / name):
                    raise ValueError(
                        "output already contains a different physical dataset"
                    )
                continue
            try:
                os.link(source / name, output)
            except OSError:
                shutil.copy2(source / name, output)
        if not (target / 'network_result.json').exists():
            from orbitserve.common import write_json
            write_json(target / 'network_result.json', result)
    result = deepcopy(result)
    result["provenance"]["physical_dataset"] = {
        "input_sha256": manifest["input_sha256"],
        "source": str(source),
        "reused": True,
    }
    return result


def validate_dataset_input(reference, physical, base_dir="."):
    root, manifest, result = load_dataset(Path(base_dir) / reference, verify_exports=False)
    requested = input_fingerprint(snapshot_network_input(physical, base_dir))
    if requested not in {manifest.get("source_sha256"), manifest["input_sha256"]}:
        stored = json.loads((root / 'materialized_input.json').read_text(encoding='utf-8'))
        if requested != input_fingerprint(stored):
            # Older manifests hashed Runtime fault controls as physical inputs.
            from .network_inputs import materialize_network_input
            if input_fingerprint(materialize_network_input(physical, base_dir)) != input_fingerprint(stored):
                raise ValueError("物理配置与数据包不匹配，请返回第一步重新生成")
    return root, manifest, result


def dataset_physical_input(reference, physical, base_dir="."):
    root, _, _ = validate_dataset_input(reference, physical, base_dir)
    concrete = json.loads(
        (root / "materialized_input.json").read_text(encoding="utf-8")
    )
    from orbitserve.runtime import RUNTIME_NETWORK_FIELDS
    for key in RUNTIME_NETWORK_FIELDS:
        concrete['network'].pop(key, None)
        if key in physical['network']:
            concrete['network'][key] = deepcopy(physical['network'][key])
    return concrete
