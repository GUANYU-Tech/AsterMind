"""Audit causal records of the source compute/Comm pipeline in exact ticks."""
import argparse
import json
from pathlib import Path
from astermind.engine.clock import Time
from astermind.engine.timeline import nanoseconds


def migration_barrier_valid(migration, terminals):
    """Compare one clock representation; legacy ns displays round upward."""
    try:
        if not terminals or any(event is None or event['kind'] != 'completed' for event in terminals):
            return False
        if 'network_complete_time_ticks' in migration:
            barrier = int(migration['network_complete_time_ticks'])
            return (barrier >= 0
                    and int(migration['completed_time_ticks']) >= barrier
                    and all(0 <= int(event['time_ticks']) <= barrier for event in terminals))
        # Old files have no exact migration stamp. Their integer ns completion
        # fields must be compared to the same ceil-ns projection of the barrier.
        barrier = nanoseconds(migration['network_complete_time_s'])
        return (migration['completed_time_s'] >= migration['network_complete_time_s']
                and all(event['completion_time_ns'] <= barrier for event in terminals))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def audit_estimate(estimate):
    assert all(type(value) is int and value>=0 for value in estimate.values()),estimate
    assert estimate['cross_satellite_flows']<=estimate['flow_count'],estimate
    assert estimate['rounds']<=estimate['flow_count'] or estimate['duration_ns']==0,estimate


def audit(result):
    transfers={row['id']:row for row in result['transfers']}
    assert len(transfers)==len(result['transfers']), 'duplicate transfer terminal event'
    counts=dict(requests=len(result['requests']),kernels=0,communications=0,dependencies=0,
                transfers=len(transfers),allocations=0,failed_batches=0)
    requested={r['id']:r for r in result['configuration']['requests']}
    for identity,state in result['requests'].items():
        assert state['status'] in ('completed','failed'), state
        if state['status']=='failed':continue
        assert state['received_tokens']==requested[identity]['output_tokens'], state
        assert int(state['completed_ticks'])>=int(state['compute_complete_ticks']), state
        assert state['cell_ready_ns']>=state['arrival_ns'],state
        if state['first_token_received_ns'] is not None:
            assert state['first_token_received_ns']>=state['cell_ready_ns'],state
    events = {event['sequence']: event for event in result['event_log']}
    statuses = {identity: 'pending' for identity in result['requests']}
    for change in result['request_state_events']:
        identity = change['request_id']
        assert change['previous'] == statuses[identity], change
        statuses[identity] = change['status']
        if change['event_sequence'] is not None:
            event = events[change['event_sequence']]
            assert int(change['time_ticks']) == int(event['time_ticks']), change
    assert statuses == {identity: state['status'] for identity, state in result['requests'].items()}
    contexts={(r.get('session_id'),r.get('context_id') or r['id']):r['id'] for r in requested.values()}
    for identity,state in result['requests'].items():
        if 'cell_ready_ticks' not in state:
            assert state['status'] == 'failed', state
            continue
        ready = int(state['cell_ready_ticks'])
        assert 'input_ready_ticks' in state, state
        request=requested[identity]
        if request.get('parent_context_id') is not None:
            parent=contexts[request.get('session_id'),request['parent_context_id']]
            assert 'parent_ready_ticks' in state, state
            assert int(state['parent_ready_ticks'])==int(result['requests'][parent]['compute_complete_ticks']),state
        for migration in result['configuration'].get('migrations',()):
            if migration.get('before_request')!=identity:continue
            committed=result['migrations'][migration['id']]
            assert committed['state']=='completed' and 'kv_ready_ticks' in state,state
            assert int(state['kv_ready_ticks'])>=int(committed['completed_time_ticks']),state
        for key in ('input_ready_ticks', 'parent_ready_ticks', 'kv_ready_ticks'):
            if key in state:
                assert ready >= int(state[key]), state
    for batch in result['batches']:
        failed=batch.get('status')=='failed'
        counts['failed_batches']+=failed
        assert batch.get('native_communication_duration_charged_s',0)==0
        calls={(c['call_id'],c['occurrence']):c for c in batch['calls']}
        for call in calls.values():
            if call['end_time_ns'] is None:
                assert failed, 'uncompleted call in successful batch'
                continue
            start,end=int(call['start_time_ticks']),int(call['end_time_ticks'])
            assert end>=start>=int(batch['start_time_ticks']),call
            assert start>=int(call['host_ready_time_ticks']),call
            for dep in call['dependencies']:
                predecessor=calls[tuple(dep)]
                assert predecessor['end_time_ns'] is not None,call
                assert start>=int(predecessor['end_time_ticks'])+int(call['launch_gap_ticks']),call
                counts['dependencies']+=1
            if call['communication']:
                assert 'communication_estimate' in call,call
                audit_estimate(call['communication_estimate'])
                projected=(end-start+10**9-1)//10**9
                assert call['communication_estimate']['duration_ns']==projected,call
                counts['communications']+=1
            else:
                assert end-start==Time(call['native_prediction_s']).ticks,call
                counts['kernels']+=1
        if not failed:
            assert int(batch['end_time_ticks'])==max(int(c['end_time_ticks']) for c in calls.values())
    for row in result.get('communication_events',()):
        if row['kind']!='allocation':continue
        for resource,capacity in row['capacities'].items():
            used=sum(f['rate_bytes_per_ns']*f['demand'].get(resource,0) for f in row['flows'].values())
            assert used<=capacity+max(1e-12,capacity*1e-10),(resource,used,capacity)
        counts['allocations']+=1
    for migration in result.get('migrations',{}).values():
        if migration['state']!='completed':continue
        events=[transfers[p['transfer_id']] for p in migration['partitions']]
        assert all(e['kind']=='completed' for e in events),migration
        assert migration_barrier_valid(migration,events),migration
        assert sum(e['size_bytes'] for e in events)==migration['bytes'],migration
    batches={(b['cell_id'],b['pool_name'],b['step_id']):b for b in result['batches']}
    for distribution in result.get('input_distributions',()):
        batch=batches[distribution['cell_id'],distribution['pool_name'],distribution['step_id']]
        assert int(distribution['completed_ticks'])<=int(batch['start_time_ticks']),distribution
    for cell in result['cells'].values():
        for request_id,handoff in cell.get('pd_handoffs',{}).items():
            if handoff['end_time_ns'] is None:continue
            events=[transfers[p['id']] for p in handoff['partitions']]
            assert len(events)==len(handoff['partitions']) and events,handoff
            if handoff['failures']:
                assert result['requests'][request_id]['status']=='failed',handoff
                continue
            assert all(event['kind']=='completed' for event in events),handoff
            assert all(event['size_bytes']==part['bytes'] for event,part in zip(events,handoff['partitions'],strict=True)),handoff
            assert max(int(e['time_ticks']) for e in events)==int(handoff['end_time_ticks']),handoff
            decode=[step for step in cell['steps'] if step['step']['schedule']['pool_name']=='decode'
                    and any(row['request_id']==request_id for row in step['step']['schedule']['requests'])]
            assert decode and int(decode[0]['start_time_ticks'])>=int(handoff['end_time_ticks']),handoff
    return dict(passed=True,**counts)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reports',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={p.stem:audit(json.loads(p.read_text())) for p in args.reports}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
