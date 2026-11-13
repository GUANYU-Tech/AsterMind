"""Eight moving satellites, two grounds and four TP2 Cells: correctness only."""
import json
from pathlib import Path


def scene(compute_mode='midlayer',case='normal'):
    physics=json.loads(Path(__file__).with_name('dynamic_physics.json').read_text())
    template=physics['elements']['satellites'][0]
    satellites=[f'sat{i}' for i in range(8)];grounds=['ground0','ground1']
    physics['time'].update(duration_s=3.,step_s=.05)
    physics['elements']['satellites']=[dict(template,satellite_id=name,satnum=11000+i,
        mean_anomaly_deg=.05*i) for i,name in enumerate(satellites)]
    ground=physics['ground_ingresses'][0]
    physics['ground_ingresses']=[dict(ground,ingress_id=name,lon_deg=ground['lon_deg']+.02*i)
                                for i,name in enumerate(grounds)]
    pairs=list(zip(satellites,satellites[1:]))+[(grounds[0],satellites[0]),(grounds[1],satellites[-1])]
    physics['network'].update(nodes=[dict(node_id=name,role='compute',isl_terminals=2,sgl_terminals=1) for name in satellites]+
        [dict(node_id=name,role='ground',isl_terminals=0,sgl_terminals=1) for name in grounds],explicit_pairs=pairs)
    physics['network']['link_profiles']={
        'communication_isl':dict(capacity_model='fixed',bandwidth_bps=100e9,setup_time_s=.001,
                                capacity_schedule=[[.08,.5],[.18,1.]]),
        'sgl':dict(capacity_model='fixed',bandwidth_bps=1e9,setup_time_s=.001)}
    groups=[dict(id=f'cell{i}',satellites=satellites[i*2:i*2+2]) for i in range(4)]
    physical=dict(cells=groups+[dict(id='ground-resources',satellites=grounds)],
        satellites=[dict(id=name) for name in satellites+grounds],devices=[],links=[],dynamic_external=True)
    for owner in satellites+grounds:
        physical['devices'] += [dict(id=owner+'-hub',satellite=owner,kind='switch',capacity_gbps=100000),
            dict(id=owner+'-gpu',satellite=owner,kind='gpu')]
        physical['links'].append(dict(id=owner+'-local',source=owner+'-gpu',target=owner+'-hub',bandwidth_gbps=1000))
    for index,(a,b) in enumerate(pairs):
        terminals=[]
        for owner in (a,b):
            terminal=f'{owner}-port-{index}';terminals.append(terminal)
            physical['devices'].append(dict(id=terminal,satellite=owner,kind='terminal',external_access=True))
            physical['links'].append(dict(id=terminal+'-access',source=owner+'-hub',target=terminal,bandwidth_gbps=1000))
        physical['links'].append(dict(id=f'edge-{index}',source=terminals[0],target=terminals[1],bandwidth_gbps=1000))
    cells=[];requests=[]
    for index,group in enumerate(groups):
        owners=group['satellites'];gpus=[owner+'-gpu' for owner in owners]
        cells.append(dict(id=group['id'],model='configs/model/qwen3-8b.json',hardware='configs/hardware/nvidia_h200.yaml',
            hardware_id='nvidia_h200',parallel=dict(tp=2),placements=[dict(rank=r,device=gpu) for r,gpu in enumerate(gpus)],
            ingress_device=gpus[0],output_device=gpus[0],prefix_caching=True,num_kv_blocks=64,
            max_model_len=64,max_batched_tokens=32,max_requests=4,
            external_terminals=[d['id'] for d in physical['devices'] if d['satellite'] in owners and d['kind']=='terminal'],
            egress_mode='multi_gateway'))
        requests.append(dict(id=f'r{index}',cell=group['id'],user_device=grounds[index%2]+'-gpu',
            arrival_ns=0,input_tokens=2,output_tokens=2,input_bytes=100000,output_bytes_per_token=16))
    requests += [dict(id='parent',cell='cell0',user_device='ground0-gpu',arrival_ns=0,input_tokens=16,
        output_tokens=2,input_bytes=128,session_id='session',context_id='p'),
        dict(id='child',cell='cell1',user_device='ground1-gpu',arrival_ns=0,input_tokens=20,
        output_tokens=2,input_bytes=128,session_id='session',context_id='c',parent_context_id='p',prefix_hit_tokens=16)]
    raw=dict(physical=physical,physical_network=physics,ground_owners=grounds,cells=cells,requests=requests,
        migrations=[dict(id='kv',after_request='parent',before_request='child',source_cell='cell0',target_cell='cell1',
                         session_id='session',context_id='p',mode='copy')],
        compute_mode=compute_mode,communication=dict(intra_mode='surrogate'),horizon_ns=3_000_000_000)
    if case=='fault':raw['runtime_network']=dict(fault_timing='continuous',faults=[dict(target_type='link',
        target_id='communication_isl:sat0:sat1',start_s=.1,end_s=.15)])
    if case=='timeout':raw['communication']['task_timeout_s']=1e-9
    return raw
