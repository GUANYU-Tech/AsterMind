"""One same-origin server for the restored scene and Cell frontends."""
import argparse
import json
import mimetypes
from pathlib import Path
import uuid
from urllib.parse import urlparse,parse_qs,unquote
from http.server import ThreadingHTTPServer

from runtime_environment import activate,identity
activate()
from paths import VIZ,RUNTIME,WORKSPACE
from console import Handler as SceneHandler


class Handler(SceneHandler):
    def raw(self,data,content_type):
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-cache')
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        path=unquote(urlparse(self.path).path)
        if not self.allowed_authority(): return self.respond({'error':'Host not allowed'},code=403)
        try:
            if path=='/health': return self.respond({'status':'ok',**identity()})
            if path=='/cell':
                self.send_response(302);self.send_header('Location','/cell/');self.send_header('Content-Length','0');self.end_headers();return
            if path.startswith('/cell/api/'):
                return self.cell_get(path.removeprefix('/cell'),parse_qs(urlparse(self.path).query))
            if path.startswith('/cell/'):
                relative=path.removeprefix('/cell/') or 'index.html'
                if relative=='usage': relative='usage.html'
                target=(VIZ/'cell'/relative).resolve()
                if not target.is_relative_to(VIZ/'cell') or not target.is_file() or target.suffix not in ('.html','.js','.mjs','.css','.png','.jpg','.svg','.woff','.woff2','.otf','.ico'):
                    return self.respond({'detail':'资源不存在'},code=404)
                return self.raw(target.read_bytes(),mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
            if path.startswith('/assets/') and path.endswith('.html'):
                target=(VIZ/path.lstrip('/')).resolve()
                if target.parent==VIZ/'assets': return self.raw(target.read_bytes(),'text/html; charset=utf-8')
            if path=='/favicon.ico':
                self.send_response(204);self.send_header('Content-Length','0');self.end_headers();return
            return super().do_GET()
        except Exception as error:
            return self.respond({'error':str(error),'detail':str(error)},code=400)

    def cell_get(self,path,query):
        import cell_jobs as jobs
        from cell_adapter import capabilities
        if path=='/api/capabilities': return self.respond(capabilities())
        if path=='/api/usage': return self.respond(jobs.usage(query))
        parts=path.strip('/').split('/')
        if len(parts)==3 and parts[:2]==['api','usage']: return self.respond(jobs.detail(parts[2]))
        if len(parts)>=3 and parts[:2]==['api','simulations']:
            job=parts[2]
            if len(parts)==3:return self.respond(jobs.status(job))
            name=parts[3]
            if name in ('configuration','result'):
                return self.respond(json.loads((jobs.folder(job)/(name+'.json')).read_text()))
            if name=='artifacts': return self.respond(jobs.artifacts(job))
            if name=='artifact':
                key=query.get('key',[''])[0]
                data=jobs.artifact(job,key,max(0,int(query.get('offset',[0])[0])),max(1,min(100,int(query.get('limit',[100])[0]))))
                if query.get('download')==['true']: return self.respond_file(jobs.folder(job)/key,attachment=True)
                return self.respond(data)
        return self.respond({'detail':'接口不存在'},code=404)

    def do_POST(self):
        path=urlparse(self.path).path
        if not path.startswith('/cell/api/'): return super().do_POST()
        try:
            host=self.allowed_authority()
            if not host or (self.headers.get('Origin') and self.headers['Origin'].lower()!=f'http://{host}'):
                return self.respond({'detail':'same origin required'},code=403)
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=16*1024*1024:raise ValueError('请求必须非空，且不超过 16 MiB')
            raw=self.rfile.read(length)
            if path.startswith('/cell/api/uploads/'):
                kind=path.rsplit('/',1)[-1]
                if kind not in ('tle','requests','kv_replay'): raise ValueError('未知资源类型')
                value={'path':str(RUNTIME/'uploads'/(uuid.uuid4().hex+('.tle' if kind=='tle' else '.json')))}
                if kind=='tle':
                    from astermind.builder.orbit import tle_elements
                    bundle=tle_elements(raw.decode(),name='cell-upload')
                    value['satellite_ids']=[s.satellite_id for s in bundle.satellites]
                elif kind=='kv_replay':json.loads(raw)
                target=Path(value['path']);target.parent.mkdir(exist_ok=True);target.write_bytes(raw)
                return self.respond(value)
            from scene_validation import strict_json
            config=strict_json(raw)
            from cell_adapter import topology,convert,request_stream
            if path=='/cell/api/topology/preview':return self.respond(topology(config)[1])
            if path=='/cell/api/config/validate':
                effective,_=convert(config)
                return self.respond({'valid':True,'effective_configuration':effective,'backend':identity()})
            if path=='/cell/api/requests/identity':
                stream=request_stream(config)
                from src.serving.request_gen.request_stream import request_stream_sha256
                return self.respond({'request_stream_sha256':request_stream_sha256(stream),'request_count':len(stream.requests)})
            if path=='/cell/api/simulations':
                from cell_jobs import start
                return self.respond(start(config,self.client_address[0]),code=202)
            return self.respond({'detail':'接口不存在'},code=404)
        except Exception as error:
            return self.respond({'detail':str(error),'error':str(error)},code=400)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--host',default='127.0.0.1')
    args=parser.parse_args()
    for name in ('network','physical','scenes','physical-data'):
        (WORKSPACE/'local/inputs'/name).mkdir(parents=True,exist_ok=True)
    (WORKSPACE/'local/results/network').mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    server.allowed_hosts={f'localhost:{args.port}',f'127.0.0.1:{args.port}'}
    print(json.dumps(identity()),flush=True)
    print(f'Open http://localhost:{args.port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
