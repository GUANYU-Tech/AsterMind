"""Deterministic two-satellite acceptance scenarios for the imported stack.

These are synthetic correctness fixtures, not calibrated constellation designs.
"""
import json
from pathlib import Path


def scene(case, *, compute_mode='midlayer', intra_mode='surrogate'):
    if case.startswith('DS06'):
        raw=json.loads((Path(__file__).with_name('session_migration_tp2.json')).read_text())
        for cell in raw['cells']:
            cell.update(hardware='configs/hardware/nvidia_h200.yaml',hardware_id='nvidia_h200')
        raw['requests'][1]['arrival_ns']=0
        raw['migrations'][0]['before_request']=raw['requests'][1]['id']
        raw['migrations'][0]['mode']='move' if case.endswith('move') else 'copy'
        raw.update(compute_mode=compute_mode,communication=dict(intra_mode=intra_mode))
        return raw
    wide=case.startswith('DS07');pd=case.startswith('DS08');hier=case=='DS05'
    tp=8 if wide or pd or hier else 2
    pp=2 if case in ('DS03','DS07-pp2') else 1
    dp=4 if wide else 1
    one=case in ('DS03','DS04') or wide or pd
    workers=tp*pp*dp
    per_star=workers if pd else workers//2 if one else tp
    if case=='DS04':per_star=1
    owners=('satA','satB')
    groups=[dict(id='cellA',satellites=list(owners))] if one else [dict(id=f'cell{x}',satellites=[s]) for x,s in zip('AB',owners)]
    physical=dict(cells=groups+[dict(id='ground-resources',satellites=['ground'])],
        satellites=[dict(id=s) for s in (*owners,'ground')],
        devices=[dict(id='user',satellite='ground',kind='terminal',external_access=True,capacity_gbps=100,max_external_links=2)],links=[])
    for owner in owners:
        terminal=f'{owner}-terminal'
        physical['devices'].append(dict(id=terminal,satellite=owner,kind='terminal',external_access=True,capacity_gbps=100,max_external_links=2))
        physical['links'].append(dict(id=f'ground-{owner}',source='user',target=terminal,bandwidth_gbps=100,latency_ns=1000))
        switches=[f'{owner}-switch-{i}' for i in range(2 if hier else 1)]
        for switch in switches:
            physical['devices'].append(dict(id=switch,satellite=owner,kind='switch',capacity_gbps=100000))
            physical['links'].append(dict(id=switch+'-terminal',source=switch,target=terminal,bandwidth_gbps=100))
        for rank in range(per_star):
            gpu=f'{owner}-gpu-{rank}'
            physical['devices'].append(dict(id=gpu,satellite=owner,kind='gpu'))
            physical['links'].append(dict(id=gpu+'-local',source=gpu,target=switches[rank//4 if hier else 0],bandwidth_gbps=1000,latency_ns=100))
        if hier:
            physical['links'].append(dict(id=owner+'-su-bridge',source=switches[0],target=switches[1],bandwidth_gbps=400,latency_ns=500))
            physical.setdefault('intra_hierarchies',{})[owner]=dict(
                su_groups=[[f'{owner}-gpu-{r}' for r in range(i*4,(i+1)*4)] for i in range(2)],su_switches=switches)
    physical['links'].append(dict(id='isl',source='satA-terminal',target='satB-terminal',bandwidth_gbps=100,latency_ns=1000))
    if case=='DS08-kv-failure':physical['links'][-1]['capacity_schedule_ns']=[(1,0)]
    model='glm-5.2' if wide else 'kimi-k2.6' if pd else 'qwen3-30b-a3b' if case=='DS04' else 'qwen3-8b'
    cells=[]
    for group in groups:
        members=group['satellites']
        devices=[f'{owner}-gpu-{r}' for owner in members for r in range(per_star)]
        placements=[dict(rank=r,device=g,pool=('prefill' if r<workers else 'decode') if pd else 'main') for r,g in enumerate(devices)]
        cell=dict(id=group['id'],model=f'configs/model/{model}.json',hardware='configs/hardware/nvidia_h200.yaml',hardware_id='nvidia_h200',
            placements=placements,parallel=dict(tp=tp,pp=pp,dp=dp,ep=tp*dp if wide or pd or case=='DS04' else 1),
            ingress_device=devices[0],output_device=devices[workers if pd else tp*(pp-1)],
            external_terminals=[f'{owner}-terminal' for owner in members],egress_mode='multi_gateway' if len(members)>1 else 'single_gateway',
            max_model_len=32,max_batched_tokens=32,max_requests=4,num_kv_blocks=64)
        if wide or pd or case=='DS04':cell['expert_routing_seed']=42
        if wide:cell.update(wide_ep_enabled=True,scheduler_adapter='vllm_0_30')
        if hier or case=='DS03':cell['scheduler_adapter']='vllm_0_30'
        if pd:cell.update(deployment='pd_disagg',scheduler_adapter='vllm_0_11')
        cells.append(cell)
    requests=[]
    for cell in cells:
        for index in range(4 if wide and not case.endswith('idle') else 2):
            requests.append(dict(id=f"{cell['id']}-r{index}",cell=cell['id'],user_device='user',
                arrival_ns=0,input_tokens=2+index,output_tokens=1 if case=='DS08-one-token' else 2,input_bytes=1000,output_bytes_per_token=16))
    return dict(physical=physical,cells=cells,requests=requests,ground_owners=['ground'],
        horizon_ns=30_000_000_000,compute_mode=compute_mode,communication=dict(intra_mode=intra_mode,trace_service=not wide))


def cross_cell_competition(*, compute_mode='midlayer', intra_mode='surrogate'):
    """Two independent schedulers over disjoint ranks on the same two stars.

    This is a compute/Comm integration fixture. Both branches' physical Builder
    requires unique Cell ownership per satellite, so it must not masquerade as
    a supported two-Cell SimulationConfig deployment.
    """
    raw=scene('DS03',compute_mode=compute_mode,intra_mode=intra_mode)
    raw['requests']=[]
    placements={f'cell{index}':(f'satA-gpu-{index}',f'satB-gpu-{index}') for index in range(2)}
    return raw,placements
