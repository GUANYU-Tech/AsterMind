"""Isolated runtime-dynamic-links platform, using the preserved LYZ UI contracts.

Only this process imports orbitserve and the original EmbeddedSession Mid-Layer.
The V1 server and its saved inputs/results remain independent.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

VIZ=Path(__file__).resolve().parent
REPOSITORY=VIZ.parent


def prepare():
    from simulation_platforms import LYZ_COMMIT
    from runtime_environment import source_git, extract_source, snapshot_lock
    root=VIZ/'.runtime/platforms/lyz/source'
    with snapshot_lock(root):
        marker=root/'.lyz-source.json'
        if not marker.is_file():
            raw=subprocess.check_output([source_git(),'archive','--format=zip',LYZ_COMMIT,
                'src','simulator','network_sim','integration','examples'],cwd=REPOSITORY)
            root.mkdir(parents=True,exist_ok=True)
            extract_source(raw,root)
            marker.write_text(json.dumps({'commit':LYZ_COMMIT}))
        elif json.loads(marker.read_text()).get('commit')!=LYZ_COMMIT:
            raise RuntimeError('LYZ 源码快照不匹配')
        # Match the old wheel's setuptools.package-dir mapping without an
        # editable installation or modifying either repository checkout.
        import shutil
        for package in ('experiments','dse','baselines'):
            target=root/'src/orbitserve'/package
            if not target.exists():shutil.copytree(root/'src'/package,target)
        if not (root/'tests/unit/runtime/test_runtime_engine.py').is_file():
            raw=subprocess.check_output([source_git(),'archive','--format=zip',LYZ_COMMIT,'tests'],cwd=REPOSITORY)
            extract_source(raw,root)
        dependencies=root/'third_party';dependencies.mkdir(exist_ok=True)
        for name in ('astermind',):
            target=dependencies/name
            if not target.exists():target.symlink_to(REPOSITORY/'third_party'/name,target_is_directory=True)
    frontend=REPOSITORY/'local/archive/frontend-starguide-original'
    midlayer=REPOSITORY/'third_party/astermind/third_party/midlayer'
    if not (frontend/'native_live.py').is_file():raise RuntimeError('缺少原版完整前端备份')
    if not (midlayer/'midlayer/runtime.py').is_file():raise RuntimeError('缺少 LYZ 原有 Mid-Layer 本地依赖')
    return root,frontend,midlayer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8798)
    args=parser.parse_args()
    root,frontend,midlayer=prepare()
    if Path(sys.prefix).name=='OS':os.environ['CONDA_DEFAULT_ENV']='OS'
    settings=REPOSITORY/'local/config/communication-modules.json'
    if settings.is_file():
        configured=json.loads(settings.read_text())
        for module in ('islsim','gslsim'):
            for field,suffix in (('root','ROOT'),('config','CONFIG'),('rf_config','RF_CONFIG'),('optical_config','OPTICAL_CONFIG')):
                value=configured.get(module,{}).get(field)
                if value:
                    if len(value)>2 and value[1]==':':value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
                    os.environ.setdefault('ORBITSERVE_'+module.upper()+'_'+suffix,value)
    from simulation_platforms import LYZ_COMMIT
    # Do not inherit V1's package search path or upstream selection.
    sys.path[:]=[p for p in sys.path if p not in {str(VIZ),str(REPOSITORY)} and '/.runtime/sources/' not in p]
    initial=[str(frontend),str(root/'src'),str(root),str(midlayer)]
    sys.path[:0]=initial
    os.environ.update(ORBITSERVE_MIDLAYER_ROOT=str(midlayer),ORBITServe_MIDLAYER_ROOT=str(midlayer),
                      PYTHONPATH=os.pathsep.join(initial),TMPDIR='/tmp',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import native_environment
    native_environment.ROOT=root
    native_environment.NATIVE=REPOSITORY/'local/native'
    # This original frontend predates the vLLM 0.31 native-scene integration.
    # Its EmbeddedSession uses the retained AsterMind/midlayer packages.
    cache=Path('/tmp/orbitserve-native-scene-031-517')
    if not cache.is_dir():cache=None

    native_environment.enable_local_flowsim=lambda:None
    def enable_classic_inprocess():
        repository=REPOSITORY/'third_party/astermind'
        paths=[REPOSITORY/'local/native',repository/'src',repository/'third_party/midlayer',
            root/'src',root,root/'examples/astermind_joint',cache,
            REPOSITORY/'local/validation/astermind-demo/venv/lib/python3.12/site-packages']
        paths=[str(p) for p in paths if p is not None and Path(p).is_dir()]
        sys.path[:]=paths+[p for p in sys.path if p not in paths and p not in {str(REPOSITORY),str(VIZ)}]
        os.environ['ORBIT_ASTERMIND_NATIVE']=str(REPOSITORY/'local/native')
        import midlayer
        if str(REPOSITORY/'local/native') not in midlayer.__path__:
            midlayer.__path__.insert(0,str(REPOSITORY/'local/native'))
    native_environment.enable_native_inprocess=enable_classic_inprocess
    enable_classic_inprocess()
    import astermind.embedded
    # Reuse the original, complete physical cache byte-for-byte, rather than
    # making an isolated source directory regenerate the 72x22 constellation.
    physical=REPOSITORY/'local/results/full-serving/physical-v1-tp_pp.json'
    cached=root/'local/results/full-serving/physical-v1-tp_pp.json'
    if physical.is_file() and not cached.is_file():
        import shutil
        cached.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(physical,cached)
    import console
    # The historical inputs/results are kept in their existing local namespace.
    console.ROOT=REPOSITORY
    original_status=console.status
    def status(job,state,**extra):
        return original_status(job,state,simulation_platform='lyz',source_commit=LYZ_COMMIT,**extra)
    console.status=status
    from urllib.parse import urlparse
    from http.server import ThreadingHTTPServer
    import importlib.util
    theme_spec=importlib.util.spec_from_file_location('platform_theme',VIZ/'workspace.py')
    theme=importlib.util.module_from_spec(theme_spec);theme_spec.loader.exec_module(theme)
    def decorate(content,path):
        return theme.decorate_document(content,path,platform='lyz')
    __import__('workspace').decorate_document=decorate

    class Handler(console.Handler):
        def do_GET(self):
            path=urlparse(self.path).path
            if not self.allowed_authority():return self.respond({'error':'Host not allowed'},code=403)
            if path=='/health':
                return self.respond(dict(status='ok',simulation_platform='lyz',commit=LYZ_COMMIT,
                    backend='LYZ original EmbeddedSession',communication='original FlowSim',
                    frontend_backup='frontend-starguide-original',
                    computation='astermind.embedded / midlayer.runtime',
                    frontend_source=console.__file__,compute_source=astermind.embedded.__file__,
                    runtime_source=sys.modules['midlayer.runtime'].__file__,
                    source=str(root),midlayer=str(midlayer)))
            if path=='/':return self.respond((VIZ/'portal.html').read_bytes(),'text/html')
            if path in {'/workspace','/configuration'}:
                return self.respond((frontend/'home.html').read_bytes(),'text/html')
            if path=='/results':
                self.send_response(302);self.send_header('Location','/experiments')
                self.send_header('Content-Length','0');self.end_headers();return
            if path in {'/cell','/cell/'}:
                self.send_response(302);self.send_header('Location','http://localhost:4173/')
                self.send_header('Content-Length','0');self.end_headers();return
            if path.startswith('/cell/assets/design/'):
                target=(VIZ/path.lstrip('/')).resolve()
                if target.is_relative_to(VIZ/'cell/assets/design') and target.is_file() and target.suffix in {'.svg','.png','.jpg','.webp'}:
                    import mimetypes
                    return self.respond(target.read_bytes(),mimetypes.guess_type(str(target))[0])
            # Preserve the reference artwork and dark UI while retaining the
            # historical JS/API contracts and actual legacy execution.
            if path.startswith('/assets/') and (path.endswith(('.png','.jpg','.webp','.svg')) or path.rsplit('/',1)[-1] in
                    {'platform_selector.js','platform_selector.css','workspace.css','starguide.css','mission.css','design-system.css'}):
                target=(VIZ/path.lstrip('/')).resolve()
                if target.is_relative_to(VIZ/'assets') and target.is_file():
                    import mimetypes
                    kind=mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
                    return self.respond(target.read_bytes(),kind)
            return super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path!='/simulation-platform':return super().do_POST()
            host=self.allowed_authority()
            if not host or (self.headers.get('Origin') and self.headers['Origin'].lower()!=f'http://{host}'):
                return self.respond({'error':'same origin required'},code=403)
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=1024:raise ValueError('无效的平台请求')
                value=json.loads(self.rfile.read(length))
                if set(value)!={'platform'} or value['platform'] not in {'lyz','v1'}:raise ValueError('未知仿真平台')
                port=args.port if value['platform']=='lyz' else int(os.environ.get('ORBITSERVE_V1_PORT','8766'))
                return self.respond(dict(platform=value['platform'],url=f'http://localhost:{port}/'))
            except Exception as error:return self.respond({'error':str(error)},code=400)

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.allowed_hosts={f'localhost:{args.port}',f'127.0.0.1:{args.port}'}
    print(json.dumps(dict(platform='lyz',commit=LYZ_COMMIT,port=args.port)),flush=True)
    server.serve_forever()


if __name__=='__main__':main()
