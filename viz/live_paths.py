"""Bounded end-to-end path context reconstructed from actual hop records."""
from collections import deque
import json
from pathlib import Path


def flow_id(event):
    if event.get('cell_transfer_id'):
        return event['cell_transfer_id']
    request=event.get('request_id') or ''
    if request.startswith('__cell_transfer__/'):
        return request.removeprefix('__cell_transfer__/')
    if event['kind']=='cell_transfer_terminal':
        return event.get('id')
    return None


class FlowPathWindow:
    def __init__(self, recent_limit=64, archive=None):
        self.paths={};self.arrivals={};self.completed_requests=set()
        self.recent=deque();self.recent_limit=recent_limit;self.cursor=0
        from flow_archive import FlowArchive
        self.archive=FlowArchive(archive) if archive is not None else None

    def observe(self, event, order):
        kind=event['kind'];request=event.get('request_id')
        if kind in {'v1_admit','v1_p2p_complete','v1_retry','v1_failed'}:
            identity=event.get('id')
            if kind=='v1_admit':
                self.paths[identity]=dict(id=identity,request_id=request,stage='V1 整条路径',
                    time_semantics='whole_path_shared_resources',start_order=order,start_s=event['time_s'],
                    start_ticks=event.get('time_ticks'),hops=[],segments=event.get('path_legs',[]),
                    source=event.get('source_node_id'),target=event.get('target_node_id'),
                    raw_admission=event)
            elif identity in self.paths:
                row=self.paths[identity]
                row.update(terminal_order=order,terminal_s=event['time_s'],terminal_ticks=event.get('time_ticks'),
                           terminal='completed' if kind=='v1_p2p_complete' else 'failed')
                self.recent.append(identity)
                if self.archive is not None:self.archive.append(row)
            self.cursor=order+1
            return
        if kind=='request_arrival':
            self.arrivals.setdefault(event['time_s'],[]).append(request)
        if kind in {'request_complete','request_failed'}:
            self.completed_requests.add(request)
        identity=flow_id(event)
        if identity and (kind in {'transfer_queued','transmit_start','transmit_complete','hop_arrival','transfer_cancelled','cell_transfer_terminal','route_plan'}):
            row=self.paths.setdefault(identity,dict(id=identity,request_id=None,stage=event.get('stage',event.get('operation')),start_order=order,start_s=event['time_s'],hops=[]))
            if event.get('time_ticks') is not None:row.setdefault('start_ticks',event['time_ticks'])
            correlation=event.get('business_request_id') or event.get('correlation_id')
            if correlation:
                if row['request_id'] and row['request_id']!=correlation:
                    raise ValueError('Conflicting end-to-end request identity')
                row['request_id']=correlation
            # Earlier native logs omitted correlation on in-flight hops. An
            # input starts at its arrival event; use only a unique exact-time
            # match, and cross-check it against the terminal correlation later.
            if not row['hops'] and event.get('stage')=='request_input' and not row['request_id']:
                arrivals=self.arrivals.get(event['time_s'],[])
                if len(arrivals)==1:row['request_id']=arrivals[0]
            for name in ('source','target'):
                if event.get('flow_'+name+'_node_id'):row[name]=event['flow_'+name+'_node_id']
            if kind=='route_plan':
                row['planned']=event.get('path',[]);row['plan_order']=order
            elif kind=='cell_transfer_terminal':
                row.update(terminal_order=order,terminal_s=event['time_s'],terminal=event.get('terminal_kind','completed'))
                if event.get('time_ticks') is not None:row['terminal_ticks']=event['time_ticks']
                self.recent.append(identity)
                if self.archive is not None:self.archive.append(row)
            elif event.get('source_node_id') and event.get('target_node_id'):
                hop=next((h for h in row['hops'] if h['transfer_id']==event.get('transfer_id')),None)
                if hop is None:
                    hop={k:event.get(k) for k in ('transfer_id','link_id','source_node_id','target_node_id')}
                    hop.update(start_order=order,start_s=event['time_s']);row['hops'].append(hop)
                    row.setdefault('source',hop['source_node_id'])
                field={'transmit_start':'sent','transmit_complete':'propagating','hop_arrival':'arrived','transfer_cancelled':'cancelled'}.get(kind)
                if field:
                    hop[field+'_order']=order;hop[field+'_s']=event['time_s']
                    if event.get('time_ticks') is not None:hop[field+'_ticks']=event['time_ticks']
        self.cursor=order+1

    def capture(self):
        if self.archive is not None:self.archive.flush()
        while len(self.recent)>self.recent_limit:
            identity=self.recent.popleft();row=self.paths.get(identity)
            if row and (row.get('stage')!='request_input' or row.get('request_id') in self.completed_requests):
                self.paths.pop(identity,None)
        recent=set(self.recent)
        self.paths={k:r for k,r in self.paths.items() if k in recent or 'terminal_order' not in r or r.get('request_id') not in self.completed_requests}
        return dict(schema='orbitserve.live_paths.v1',through_event_count=self.cursor,paths=list(self.paths.values()))

    def save(self, output, *, force=False):
        folder=Path(output)/'native-replay';folder.mkdir(exist_ok=True)
        target=folder/'live-paths.json';temporary=folder/'live-paths.json.tmp'
        temporary.write_text(json.dumps(self.capture(),ensure_ascii=False),encoding='utf8')
        temporary.replace(target)
        if self.archive is not None:self.archive.publish(force=force)
