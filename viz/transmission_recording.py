"""Communication-only index and a viewing clock derived from actual busy intervals."""
import base64
import gzip
import json
from pathlib import Path
import sqlite3

KINDS={'transfer_queued','transmit_start','transmit_complete','hop_arrival','transfer_cancelled','transfer_dropped','transfer_retried','external_flow_done','route_recovery','fault_injected','fault_restored','topology_command'}
KINDS.update({'v1_admit','v1_service','v1_p2p_complete','v1_retry','v1_failed'})

def is_transmission_event(event):
    return event['kind'] in KINDS and (not event.get('source_node_id') or not event.get('target_node_id') or event['source_node_id']!=event['target_node_id'])

def read_page(path):
    data=json.loads(path.read_text())
    return json.loads(gzip.decompress(base64.b64decode(data['data']))) if data.get('encoding')=='gzip-base64' else data

def build_transmission_index(output):
    folder=Path(output)/'native-replay';original=json.loads((folder/'scene-recording.json').read_text())
    pages=[];offset=completed=0;end=0;v1_starts={};v1_intervals=[];v1_done=[];v1_failed=[];v1_attempts=0
    for descriptor in original['pages']:
        data=read_page(folder/descriptor['file']);events=data['events'];count=sum(is_transmission_event(e) for e in events)
        pages.append(dict(descriptor,offset=offset,count=count,completed_before=completed))
        offset+=count;completed+=sum(e['kind']=='external_flow_done' for e in events)
        for event in events:
            if is_transmission_event(event):end=max(end,event['time_s'])
            if event['kind']=='v1_admit' and is_transmission_event(event):
                v1_starts[event['id']]=event;v1_attempts+=1
            if event['kind'] in {'v1_p2p_complete','v1_retry','v1_failed'}:
                matches=[event['id']] if event['id'] in v1_starts else [key for key,row in v1_starts.items() if event['kind']=='v1_failed' and row.get('task')==event['id']]
                for key in matches:
                    start=v1_starts.pop(key)
                    v1_intervals.append((start['time_s'],event['time_s']))
                    (v1_done if event['kind']=='v1_p2p_complete' else v1_failed).append(event)
    intervals=[];completion_times=[];failure_times=[];completion_ticks=[];failure_ticks=[]
    archive=folder/'flow-paths.sqlite3'
    if archive.is_file():
        with sqlite3.connect(archive.as_uri()+'?mode=ro',uri=True) as db:
            for (raw,) in db.execute('SELECT data FROM paths'):
                path=json.loads(raw);physical_hops=[h for h in path['hops'] if h.get('source_node_id')!=h.get('target_node_id')]
                if physical_hops and path.get('terminal_s') is not None:
                    (failure_times if path.get('terminal') in {'failed','cancelled'} else completion_times).append(path['terminal_s'])
                    if path.get('terminal_ticks') is not None:
                        (failure_ticks if path.get('terminal') in {'failed','cancelled'} else completion_ticks).append(path['terminal_ticks'])
                for hop in physical_hops:
                    start=hop.get('sent_s');finish=hop.get('arrived_s',hop.get('cancelled_s'))
                    if start is not None and finish is not None and finish>=start and hop.get('source_node_id')!=hop.get('target_node_id'):intervals.append((start,finish))
    intervals.sort();merged=[]
    if v1_attempts:
        intervals=sorted(v1_intervals+[(row['time_s'],end) for row in v1_starts.values()])
        completion_times=[e['time_s'] for e in v1_done];failure_times=[e['time_s'] for e in v1_failed]
        completion_ticks=[e['time_ticks'] for e in v1_done if e.get('time_ticks') is not None]
        failure_ticks=[e['time_ticks'] for e in v1_failed if e.get('time_ticks') is not None]
    for start,finish in intervals:
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],finish)
        else:merged.append([start,finish])
    outcomes=dict(completion_times=sorted(completion_times),failure_times=sorted(failure_times)) if completion_times or failure_times else {}
    result=dict(schema='orbitserve.transmission_recording.v1',pages=pages,event_count=offset,transfer_count=len(completion_times)+len(failure_times) if outcomes else completed,end_s=end,busy_intervals=merged,**outcomes)
    if v1_attempts:result.update(transfer_count=v1_attempts,time_semantics='whole_path_shared_resources',unit='通信尝试')
    if original.get('clock'):
        result.update(clock=original['clock'],end_ticks=original['end_ticks'],completion_ticks=sorted(completion_ticks,key=int),failure_ticks=sorted(failure_ticks,key=int))
    temporary=folder/'transmission-recording.tmp';temporary.write_text(json.dumps(result,separators=(',',':')));temporary.replace(folder/'transmission-recording.json')
    return result
