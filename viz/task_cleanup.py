"""Permanently remove a terminal run's own files, never its dataset sources."""
import json
from pathlib import Path
import re
import shutil


def _owned(root, relative):
    target = root / relative
    # Validate every ancestor as well as the target before recursive removal.
    cursor = target
    while cursor != root:
        if cursor.is_symlink() or (hasattr(cursor, 'is_junction') and cursor.is_junction()):
            raise ValueError('任务目录包含链接，不能安全删除：' + str(cursor))
        cursor = cursor.parent
    if not target.resolve().is_relative_to(root):
        raise ValueError('任务目录超出工作区')
    return target


def _references(value, output, base, run_id, resolved=None):
    if resolved is None:
        resolved = {}
    if isinstance(value, dict):
        # This typed graph contains device IDs such as satellite/gpu0, not
        # filesystem references. Resolving every endpoint as a path caused
        # tens of thousands of Windows-mount metadata operations per delete.
        native = value.get('schema_version') in (1, 'orbitserve.native_cells.v1')
        return any((k in {'run_id', 'active_run_id'} and v == run_id) or
                   _references(v, output, base, run_id, resolved)
                   for k, v in value.items() if not (native and k == 'physical'))
    if isinstance(value, list):
        return any(_references(v, output, base, run_id, resolved) for v in value)
    if not isinstance(value, str) or not value:
        return False
    # Only path-shaped strings; titles and arbitrary configuration values are not paths.
    if ('/' not in value and '\\' not in value) or '://' in value:
        return False
    path = Path(value)
    candidate = path if path.is_absolute() else base / path
    if candidate not in resolved:
        resolved[candidate] = candidate.resolve()
    return resolved[candidate].is_relative_to(output)


def deletion_paths(root, run_id):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
        raise ValueError('invalid run ID')
    root = Path(root).resolve()
    output = _owned(root, Path('local/results/network') / run_id)
    inputs = _owned(root, Path('local/inputs/network') / run_id)
    if not output.is_dir():
        raise FileNotFoundError('任务不存在或已删除')
    # Input manifests are small. Never load potentially huge result payloads here.
    candidates = list((root/'local/inputs/network').glob('*/manifest.json'))
    for name in ('scenes', 'experiments'):
        candidates.extend((root/'local/inputs'/name).glob('*.json'))
    candidates.extend((root/'local/results/comparisons').glob('*/status.json'))
    candidates.extend((root/'local/results/comparisons').glob('*/comparison.json'))
    resolved = {}
    for path in candidates:
        if path.is_relative_to(inputs):
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        if _references(value, output, path.parent, run_id, resolved) or _references(value, output, root, run_id, resolved):
            raise ValueError('该任务的数据仍被其他场景或运行引用，不能删除：' + str(path.relative_to(root)))
    bindings = []
    for path in (root/'local/inputs/physical-data').glob('*.json'):
        if json.loads(path.read_text(encoding='utf-8')).get('run_id') == run_id:
            bindings.append(_owned(root, path.relative_to(root)))
    return output, inputs, bindings


def delete_files(root, run_id):
    from tasks import processes
    if processes(root, run_id):
        raise ValueError('任务仍在运行，请先停止并等待退出')
    output, inputs, bindings = deletion_paths(root, run_id)
    # Remove only run-owned paths. A hard-linked/copied input dataset's original
    # remains intact; symlinks inside these trees are unlinked by rmtree.
    if inputs.exists():
        shutil.rmtree(inputs)
    for path in bindings:
        path.unlink()
    shutil.rmtree(output)
    return {'id': run_id, 'state': 'deleted'}
