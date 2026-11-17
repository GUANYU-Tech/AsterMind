"""The explicitly started experiment; never inferred from history ordering."""
import json
from pathlib import Path
import threading
from studio import key

_lock = threading.Lock()


def select(root, identity):
    identity = key(identity)
    path = Path(root)/'local/inputs/current-experiment.json'
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'id':identity}), encoding='utf-8')
        temp.replace(path)


def current(root):
    path = Path(root)/'local/inputs/current-experiment.json'
    if not path.is_file():
        return {'id':None}
    value=json.loads(path.read_text(encoding='utf-8'))
    identity=key(value['id'])
    if not (Path(root)/'local/results/network'/identity/'status.json').is_file():
        return {'id':None}
    return {'id':identity}
