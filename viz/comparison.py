"""Personal paired-run evidence, derived only from executed Runtime records."""
import json
from pathlib import Path


def write_comparison(queue_dir, simgrid_dir):
    roots={'queue':Path(queue_dir),'simgrid':Path(simgrid_dir)}
    result={}
    requests={}
    for mode,folder in roots.items():
        runtime=json.loads((folder/'experiment_result.json').read_text())['runtime']
        assert runtime['provenance']['network_transport']['backend']==mode
        requests[mode]=[r['request'] for r in runtime['request_records']]
        flows=[]
        for record in runtime['request_records']:
            leg=next(l for l in record['routing']['selected_path'] if l['stage']=='input' and l['link_id'].startswith('sgl:'))
            flows.append({'request_id':record['request']['request_id'],'ready_s':record['request']['arrival_time_s'],**{k:leg[k] for k in ('source_node_id','target_node_id','start_s','finish_s','payload_bytes','segments')},'e2e_s':record['timing']['e2e_s'],'ttft_s':record['timing']['ttft_s']})
        result[mode]={'run_id':folder.name,'backend':runtime['provenance']['network_transport']['backend'],'completed':runtime['request_summary']['completed_request_count'],'flows':flows}
    first=result['simgrid']['flows'][0]
    assert requests['queue']==requests['simgrid'], 'comparison requires identical requests'
    assert all((f['source_node_id'],f['target_node_id'])==(first['source_node_id'],first['target_node_id']) for m in result.values() for f in m['flows'])
    assert [f['payload_bytes'] for f in result['queue']['flows']]==[f['payload_bytes'] for f in result['simgrid']['flows']]
    segments=[s for flow in result['simgrid']['flows'] for s in flow['segments']]
    peak=max(segments,key=lambda s:(s['concurrent_flows'],s['finish_s']-s['start_s']))
    peak_flows=[{'request_id':flow['request_id'],**s} for flow in result['simgrid']['flows'] for s in flow['segments'] if s['start_s']==peak['start_s'] and s['finish_s']==peak['finish_s']]
    value={'schema_version':'orbitserve.local_comparison.v1','source':'two_actual_runtime_runs_same_input_except_transport_model','queue':result['queue'],'simgrid':result['simgrid'],'peak':{**peak,'flows':peak_flows},'source_node_id':first['source_node_id'],'target_node_id':first['target_node_id']}
    for folder in roots.values():(folder/'comparison.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value
