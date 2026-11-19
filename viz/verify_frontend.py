"""Read-only HTTP / source checks for the running local V1 frontend.

Writes only viz/.runtime/verification/frontend-checks.json. Does not run UC1–UC6.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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
            if row['path'] in {'/', '/workspace', '/configuration', '/cell/', '/cell/usage'}:
                assert b'data-platform-control' not in body,row
            checks.append(row)
    health=json.loads(fetch('/health')[1])
    assert health['status']=='ok' and health['platform']=='linux' and health['old_backend'] is False
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VIZ.parent,text=True).strip()
    assert health['commit']==head
    from runtime_environment import source_identity
    expected,_=source_identity()
    assert health['source_fingerprint']==expected['source_fingerprint'], 'Server is running stale backend code'
    assert health['modified_files']==expected['modified_files']
    scene_html=fetch('/scene')[1].decode()
    assert 'id="routingPolicy"' in scene_html and 'id="customRoutingPolicy"' in scene_html
    assert 'id="cellSelection"' in scene_html and 'value="estimated_arrival"' in scene_html
    assert 'update_interval_s' not in scene_html
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
    changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=VIZ.parent,text=True).splitlines()
    result=dict(time=datetime.now(timezone.utc).isoformat(),health=health,http=checks,
        python_syntax_files=syntax,completed_scenes=reports,changed_files=changed)
    (VIZ/'.runtime/verification/frontend-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('http','completed_scenes')},ensure_ascii=False))
    assert reports,'No completed V1 scene evidence'
    print('PASS',len(checks),'HTTP checks;',len(reports),'completed V1 scene reports')


if __name__=='__main__':main()
