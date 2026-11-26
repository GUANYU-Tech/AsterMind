"""Read-only V1 event projection for the preserved live/replay presentation."""
from collections import Counter


def project(raw, owner):
    row=dict(raw)
    row['time_s']=int(row['time_ticks'])/1e18 if row.get('time_ticks') else row.get('time_ns',0)/1e9
    kind=row.get('kind')
    if kind=='link_state' and not row.get('link_id'):
        row['kind']='v1_physical_boundary'
    if kind in ('admit','service','p2p_complete','retry','failed','comm_plan','allocation'):
        row['kind']='v1_'+kind
        row['cell_transfer_id']=row.get('task',row.get('id'))
        row['transfer_id']=row.get('id')
        row['request_id']=row.get('task',row.get('id'))
        row['time_semantics']='whole_path_shared_resources'
        if row.get('path'):
            row['device_path']=list(row['path'])
            row['path_nodes']=list(dict.fromkeys(owner.get(d,d) for d in row['path']))
            row['source_node_id']=row['path_nodes'][0]
            row['target_node_id']=row['path_nodes'][-1]
            row['path_legs']=[dict(source_node_id=a,target_node_id=b,
                transfer_id=f"{row['id']}/segment/{i}",cell_transfer_id=row['cell_transfer_id'],
                request_id=row['request_id'],time_semantics='whole_path_shared_resources')
                for i,(a,b) in enumerate(zip(row['path_nodes'],row['path_nodes'][1:])) if a!=b]
    elif 'status' in row and 'request_id' in row:
        row['kind']={'completed':'request_complete','failed':'request_failed','input':'request_arrival'}.get(row['status'],'v1_request_state')
    for key in ('source_node_id','target_node_id','resume_node_id','original_source_node_id','compute_node_id'):
        if key in row:row[key]=owner.get(row[key],row[key])
    return row


def result_events(result):
    owner={d['id']:d['satellite'] for d in result['configuration']['physical']['devices']}
    rows=[project(e,owner) for e in [*result['network']['event_log'],*result['communication_events'],*result['request_state_events']]]
    rows.sort(key=lambda e:int(e.get('time_ticks') or round(e['time_s']*1e18)))
    from event_catalog import annotate
    rows=[annotate(dict(e,event_order=i)) for i,e in enumerate(rows)]
    horizon = result['configuration']['horizon_ns']
    end_ns = result['measurement']['simulated_time_ns'] if horizon is None else horizon
    return dict(events=rows,end_s=end_ns/1e9,
        disabled_links=result['configuration']['disabled_physical_links'],
        category_counts=dict(Counter(e['category'] for e in rows)),
        event_counts=dict(Counter(e['kind'] for e in rows)),initial_events=sum(e['time_s']==0 for e in rows))
