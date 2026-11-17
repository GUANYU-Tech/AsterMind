"""Export saved scene events in bounded pages, with causal state at each boundary."""
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sqlite3

from event_catalog import annotate, is_playback_event
from live_replay import LiveReplayWindow


def page_paths(output, data):
    archive=Path(output)/'native-replay'/'flow-paths.sqlite3'
    if not archive.is_file():return []
    identities=set()
    for event in [*data['initial'].get('active_transfers',[]),*data['events']]:
        request=event.get('request_id') or ''
        identity=(event.get('v1_flow_id') or event.get('id')) if event.get('time_semantics')=='whole_path_shared_resources' else event.get('cell_transfer_id') or (request[len('__cell_transfer__/'):] if request.startswith('__cell_transfer__/') else None)
        if identity:identities.add(identity)
    rows=[];identities=sorted(identities)
    with sqlite3.connect(archive.as_uri()+'?mode=ro',uri=True) as db:
        for start in range(0,len(identities),256):
            part=identities[start:start+256]
            rows.extend(json.loads(row[0]) for row in db.execute('SELECT data FROM paths WHERE id IN ('+','.join('?' for _ in part)+')',part))
    return rows


def build_recording(output, page_size=4096, *, force=False):
    output=Path(output);folder=output/'native-replay';folder.mkdir(exist_ok=True)
    target=folder/'scene-recording.json'
    if target.is_file() and not force:
        if not (folder/'transmission-recording.json').is_file():
            from transmission_recording import build_transmission_index
            build_transmission_index(output.resolve())
        return json.loads(target.read_text())
    state=LiveReplayWindow();links={};commands={};pages=[];events=[];visible=0;offset=0;last_time=0
    from runtime_events import project
    from live_paths import FlowPathWindow
    saved=json.loads((output/'native_result.json').read_text())
    owners={d['id']:d['satellite'] for d in saved['configuration']['physical']['devices']}
    paths=FlowPathWindow(archive=output)
    def context():
        return deepcopy(dict(active_transfers=list(state.flows.values()),route_plans=list(state.plans.values()),
                    faults=list(state.faults.values()),recovery_events=list(state.recoveries.values()),
                    loss_events=list(state.losses.values())))
    initial=context();initial_links=[];initial_commands=[]
    def flush():
        nonlocal offset,visible,initial,initial_links,initial_commands
        if not events:return
        name=f'scene-events-{len(pages):05d}.json'
        data=dict(initial=initial,events=events,link_states=initial_links,commands=initial_commands)
        paths.archive.flush()
        data['paths']=page_paths(output.resolve(),data)
        raw=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()
        payload=dict(encoding='gzip-base64',data=base64.b64encode(gzip.compress(raw,compresslevel=1)).decode())
        (folder/name).write_text(json.dumps(payload,separators=(',',':')))
        pages.append(dict(file=name,offset=offset,count=visible,start_s=events[0]['time_s'],end_s=events[-1]['time_s'],start_ticks=events[0].get('time_ticks'),end_ticks=events[-1].get('time_ticks'),first_order=events[0]['event_order'],last_order=events[-1]['event_order']))
        offset+=visible;visible=0;events.clear();initial=context()
        initial_links=list(links.values());initial_commands=list(commands.values())
    with gzip.open(output/'runtime_events.jsonl.gz','rt',encoding='utf-8') as stream:
        for order,line in enumerate(stream):
            raw=json.loads(line);last_time=raw['time_s']
            if raw.get('time_semantics')=='whole_path_shared_resources':
                raw=project({**raw,'kind':raw['kind'].removeprefix('v1_')},owners)
            row=annotate(dict(raw,event_order=order))
            paths.observe(row,order)
            # Keep topology state, but advance the player only on semantic events.
            if row['category']!='diagnostic':
                events.append(row);visible+=is_playback_event(row)
            state.observe(raw)
            if raw['kind']=='link_state' and raw.get('link_id'):links[raw['link_id']]=raw
            if raw['kind']=='topology_command':commands[raw.get('target_id',raw.get('link_id',''))]=raw
            if visible>=page_size or len(events)>=page_size*2:flush()
    flush()
    paths.save(output)
    result=dict(schema='orbitserve.scene_recording.v1',pages=pages,event_count=offset,end_s=last_time,end_ticks=raw.get('time_ticks'))
    if raw.get('time_ticks') is not None:
        from astermind.engine.clock import CLOCK
        result['clock']=CLOCK
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(result,separators=(',',':')));temporary.replace(target)
    from transmission_recording import build_transmission_index
    build_transmission_index(output.resolve())
    return result
