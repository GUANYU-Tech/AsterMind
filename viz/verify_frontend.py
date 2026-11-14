"""Read-only HTTP / source checks for the running local V1 frontend.

Writes only viz/.runtime/verification/frontend-checks.json. Does not run UC1–UC6.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

VIZ = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8766'


def fetch(path):
    with urllib.request.urlopen(BASE+path, timeout=30) as response:
        body=response.read()
        return dict(path=path,status=response.status,bytes=len(body)),body


def main():
    checks=[]
    routes=['/','/workspace','/configuration','/physical','/scene','/live','/results',
        '/experiments','/cell/','/cell/usage','/assets/guide.html','/health',
        '/physical-catalog','/scene-catalog','/physical-datasets','/task-catalog',
        '/cell/api/capabilities','/cell/api/usage','/islsim-defaults','/radio-profiles']
    with ThreadPoolExecutor(max_workers=6) as executor:
        for row,body in executor.map(fetch,routes):
            assert row['status']==200 and body,row
            checks.append(row)
    health=json.loads(fetch('/health')[1])
    assert health['platform']=='linux' and health['old_backend'] is False
    assert '/envs/OS/' in health['python']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VIZ.parent,text=True).strip()
    assert health['commit']==head
    syntax=0
    for path in VIZ.rglob('*.py'):
        if '.runtime' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path));syntax+=1
    reports=[]
    for directory in sorted((VIZ/'.runtime/workspace/local/results/network').glob('native-scene-*')):
        native=directory/'native_result.json'
        if not native.is_file():continue
        result=json.loads(native.read_text())
        if result['status']!='completed':continue
        prefix='/runs/'+directory.name+'/'
        for suffix in ['analysis','environment-index','native-replay/index.html','native-replay/replay.js',
                       'native-replay/catalog.json','native-replay/transmission-recording.json']:
            row,body=fetch(prefix+suffix);assert row['status']==200;checks.append(row)
        analysis=json.loads(fetch(prefix+'analysis')[1])
        assert analysis['summary']['completed']==sum(r['status']=='completed' for r in result['requests'].values())
        recording=json.loads(fetch(prefix+'native-replay/transmission-recording.json')[1])
        assert recording['event_count']>0
        reports.append(dict(id=directory.name,status=result['status'],audit=result.get('audit'),
            replay_events=recording['event_count']))
    baseline=json.loads((VIZ/'.runtime/outside-before.json').read_text())
    changed=[]
    for name,digest in baseline.items():
        path=VIZ.parent/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:changed.append(name)
    result=dict(time=datetime.now(timezone.utc).isoformat(),health=health,http=checks,
        python_syntax_files=syntax,completed_scenes=reports,outside_files_checked=len(baseline),outside_changed=changed)
    (VIZ/'.runtime/verification/frontend-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('http','completed_scenes')},ensure_ascii=False))
    assert not changed,changed
    assert reports,'No completed V1 scene evidence'
    print('PASS',len(checks),'HTTP checks;',len(reports),'completed V1 scene reports')


if __name__=='__main__':main()
