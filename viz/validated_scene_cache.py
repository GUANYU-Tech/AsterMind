"""Reuse native preflight only while configuration and referenced files match."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

_entries = OrderedDict()
_lock = threading.Lock()


def identity(value, root, midlayer):
    paths = set()
    dataset = value.get('physical_dataset')
    if not isinstance(dataset, str):
        return None
    folder = Path(dataset)
    if not folder.is_absolute():
        folder = Path(root)/folder
    if not folder.is_dir():
        return None
    paths.update(p for p in folder.rglob('*') if p.is_file())
    for cell in value.get('cells', ()):
        for field in ('model','hardware','memory_policy_file','host_launch_profile'):
            name = cell.get(field, {'model':'configs/model/qwen3-4b.json',
                                    'hardware':'configs/hardware/nvidia_a800.yaml'}.get(field))
            if name:
                paths.add(Path(midlayer)/name)
    # Include local validator code; cached approvals never survive changed rules.
    paths.update(Path(__file__).parent/name for name in ('native_scene.py','scene_validation.py'))
    paths.update(Path(root)/'simulator'/name for name in ('config.py','scene.py','physics.py','resources.py'))
    paths.add(Path(root)/'simulator/catalog.py')
    paths.update(Path(root)/'integration'/name for name in ('midlayer_cell.py','latest_vllm_backend.py'))
    try:
        signatures = [(str(p.resolve()), (s:=p.stat()).st_size, s.st_mtime_ns, s.st_ctime_ns)
                      for p in sorted(paths)]
    except OSError:
        return None
    canonical = json.dumps([value, signatures],sort_keys=True,allow_nan=False,separators=(',',':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def get(key):
    with _lock:
        value = _entries.get(key)
        if value is not None:
            _entries.move_to_end(key)
            return deepcopy(value)


def remember(key, value):
    if key is None:
        return
    with _lock:
        _entries[key] = deepcopy(value)
        _entries.move_to_end(key)
        while len(_entries)>32:
            _entries.popitem(last=False)
