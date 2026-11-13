"""Content identities for external inputs used by resumable regression runs."""
import hashlib
import json
from pathlib import Path
import subprocess


def hash_tree(root, suffixes=('.py', '.json', '.yaml', '.toml', '.lock', '.so')):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and path.suffix in suffixes and not {'.git', '__pycache__'} & set(path.parts):
            digest.update(str(path.relative_to(root)).encode()+b'\0')
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def python_identity(executable):
    # RECORD content binds the installed distribution versions/builds. VLLM
    # adapter source is separately hashed because it may be locally patched.
    script = '''import hashlib, importlib.metadata as m, json, pathlib, sys
rows=[]
for d in m.distributions():
 rows.append((d.metadata['Name'],d.version,hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest()))
root=pathlib.Path(m.distribution('vllm').locate_file('vllm'))
h=hashlib.sha256()
for p in sorted(root.rglob('*.py')):
 h.update(str(p.relative_to(root)).encode()+b'\\0'+hashlib.sha256(p.read_bytes()).digest())
print(json.dumps(dict(version=sys.version, executable_sha256=hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest(), distributions=sorted(rows), vllm_source_sha256=h.hexdigest())))
'''
    output = subprocess.check_output([str(executable), '-B', '-c', script], text=True)
    return json.loads(output.splitlines()[-1])
