"""Explicit local platform selection; legacy imports never enter the V1 process."""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

from paths import VIZ, RUNTIME, REPOSITORY

LYZ_COMMIT='cf96341e52ee2bdbb249ce54ffd3aef97aa9250e'
LYZ_PORT=8798
START_LOCK=threading.Lock()


def health():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{LYZ_PORT}/health',timeout=.5) as response:
            value=json.load(response)
        return value if value.get('simulation_platform')=='lyz' and value.get('commit')==LYZ_COMMIT else None
    except (OSError,ValueError):
        return None


def select(platform, v1_port=8766):
    if platform not in {'lyz','v1'}:
        raise ValueError('未知仿真平台')
    if platform=='v1':
        return dict(platform='v1',url=f'http://localhost:{int(v1_port)}/')
    with START_LOCK:
        if health() is None:
            # Refuse to take over another application's port.
            import socket
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                try:probe.bind(('127.0.0.1',LYZ_PORT))
                except OSError as error:raise ValueError('LYZ 本地端口已被其他服务占用') from error
            folder=RUNTIME/'platforms/lyz';folder.mkdir(parents=True,exist_ok=True)
            environment=dict(os.environ,ORBITSERVE_V1_PORT=str(int(v1_port)),TMPDIR='/tmp')
            with (folder/'server.log').open('ab') as log:
                process=subprocess.Popen([sys.executable,'-B',str(VIZ/'lyz_server.py'),'--port',str(LYZ_PORT)],
                    cwd=REPOSITORY,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                    start_new_session=True)
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                if health() is not None:break
                if process.poll() is not None:raise ValueError('LYZ 平台启动失败，请查看平台诊断日志')
                time.sleep(.2)
            else:raise ValueError('LYZ 平台仍在准备依赖，请稍后重试')
        return dict(platform='lyz',url=f'http://localhost:{LYZ_PORT}/',commit=LYZ_COMMIT)


def control(content, platform):
    """Identical accessible selector in both independently hosted frontends."""
    if b'data-platform-control' in content:return content
    selected=lambda name:' selected' if name==platform else ''
    markup=('<label class="platform-switch" data-platform-control>仿真平台'
        '<select aria-label="仿真平台" id="simulationPlatform">'
        f'<option value="lyz"{selected("lyz")}>LYZ · 原版计算 / FlowSim</option>'
        f'<option value="v1"{selected("v1")}>V1.0 · Runtime Comm</option></select></label>'
        '<span id="platformSwitchStatus" role="status" aria-live="polite"></span>').encode()
    close=content.find(b'</header>')
    if close!=-1:content=content[:close]+markup+content[close:]
    assets=b'<link rel="stylesheet" href="/assets/platform_selector.css"><script src="/assets/platform_selector.js" defer></script>'
    if b'</body>' in content:return content.replace(b'</body>',assets+b'</body>',1)
    # The complete historical result templates also include HTML fragments.
    if b'</html>' in content:return content.replace(b'</html>',assets+b'</html>',1)
    return content+assets
