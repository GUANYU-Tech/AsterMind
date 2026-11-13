"""Sampled 24h geometry fed to the UNMODIFIED ae6ee46 optical allocation block.
Not a full offline build, not an online request simulation. Fixed-LVLH motion
uses the local addition; the allocation block is read verbatim from base ZIP.
"""
import sys,json,time,zipfile,hashlib,heapq
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import textwrap
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parents[1]))
from sgp4.api import SatrecArray
from astermind.builder.physical.network_inputs import materialize_network_input
from astermind.builder.orbit.sgp4.elements import load_sgp4_element_bundle
from astermind.builder.orbit.sgp4.propagation import _load_sgp4_api,_initialize_satellite,_utc_to_julian,propagate_sgp4_teme
from astermind.engine import NetworkSpec,candidate_links
from astermind.engine.policies.connection import connection_priority
from dataclasses import replace

t0=time.perf_counter()
original=(R.parents[1]/'astermind/engine/state/network.py').read_text()
block=original[original.index('            used = {}'):original.index('            changed.update(k for k in previous_dynamic')]
assert 'optical_terminal_quotas' not in block
code=compile(textwrap.dedent(block),'ae6ee46-optical-allocation-block','exec')
p=json.loads((R/'scenario.json').read_text());p['ground_ingresses']=[]
for formation in p['constellations']['compute_formations']:
    formation.pop('max_external_links',None)
c=materialize_network_input(p)
# Existing low-level per-node configuration: no new production terminal rule.
for node in c['network']['nodes']:
    if node['role']=='compute':
        node['isl_terminals']=3 if node['cell_ingress'] else 2
(R/'materialized-network.json').write_text(json.dumps(c,indent=2))
spec=NetworkSpec.from_value(c['network'])
assert not any('max_external_links' in n for n in c['network']['nodes'])
edges=list(candidate_links(spec));ids=[n.node_id for n in spec.nodes];index={s:i for i,s in enumerate(ids)}
a=np.array([index[e['endpoint_a']] for e in edges]);b=np.array([index[e['endpoint_b']] for e in edges])
maxd=np.array([spec.link_profiles[e['kind']].max_distance_km for e in edges]);internal=np.array([e['kind']=='cell_isl' for e in edges]);cross=np.array([e['kind']=='cross_layer' for e in edges]);backbone=np.array([e['kind']=='communication_isl' for e in edges])
bundle=load_sgp4_element_bundle(c['elements']);cb=replace(bundle,satellites=tuple(e for e in bundle.satellites if not e.satellite_id.startswith('custom-comm')))
times=np.arange(0,86401,60,dtype=float);epoch=p['time']['epoch_utc']
cp=propagate_sgp4_teme(cb,simulation_epoch_utc=epoch,sample_times_s=times.tolist())
sorted_compute=sorted(e.satellite_id for e in cb.satellites)
compute_pos=np.array([s.position_teme_km for s in cp.samples]).reshape(len(times),40,3)
comm=tuple(e for e in bundle.satellites if e.satellite_id.startswith('custom-comm'))
api=_load_sgp4_api();sa=SatrecArray([_initialize_satellite(api,e) for e in comm]);jd,fr=_utc_to_julian(epoch)
ci=[index[e.satellite_id] for e in comm];pi=[index[s] for s in sorted_compute]
exits=sorted(n.node_id for n in spec.nodes if n.role=='compute' and spec.external_member(n))
exitset=set(exits);ext_by_exit=[[k for k,e in enumerate(edges) if e['kind']=='cross_layer' and s in (e['endpoint_a'],e['endpoint_b'])] for s in exits]
dummy=SimpleNamespace(spec=spec,nodes={n.node_id:n for n in spec.nodes},link_frames=defaultdict(list),allocation_checks={'max_pool_usage':{}},end_s=86401.)
states={};prev=set();rowcounts=[];rowready=[];backbone_lost=[];excess=[];new_count=np.zeros(10,int)
scope={'heapq':heapq,'connection_priority':connection_priority}
edgeids=[e['link_id'] for e in edges];eid_index={key:k for k,key in enumerate(edgeids)}
for k,t in enumerate(times):
    er,r,v=sa.sgp4(np.array([jd]),np.array([fr+t/86400.]));assert not er.any()
    pos=np.zeros((len(ids),3));pos[ci]=r[:,0];pos[pi]=compute_pos[k]
    delta=pos[b]-pos[a];d2=np.einsum('ij,ij->i',delta,delta);dist=np.sqrt(d2)
    fraction=np.clip(-np.einsum('ij,ij->i',pos[a],delta)/d2,0,1)
    closest=pos[a]+fraction[:,None]*delta
    valid=(dist<=maxd)&(np.linalg.norm(closest,axis=1)>6478.137)
    eligible={edgeids[j]:{**edges[j],'distance_km':float(dist[j])} for j in np.flatnonzero(valid)}
    states={key:value for key,value in states.items() if key in eligible}
    before=set(states)
    ns=dict(self=dummy,t=float(t),states=states,eligible=eligible,latest=eligible,selected_sgl=set(),previous_dynamic={},changed=set(),queue=[])
    ns.update(scope)
    exec(code,ns,ns)
    count=[];up=[]
    for j,indices in enumerate(ext_by_exit):
        keys=[edgeids[x] for x in indices if edgeids[x] in states]
        count.append(len(keys));up.append(sum(states[key]['ready_s']<=t for key in keys));new_count[j]+=sum(key not in before for key in keys)
    rowcounts.append(count);rowready.append(up)
    allocated_indices=np.array([eid_index[key] for key in states],dtype=int)
    allowed_backbone=set(np.flatnonzero(valid&backbone));active_backbone={j for j in allocated_indices if backbone[j]}
    backbone_lost.append(len(allowed_backbone-active_backbone))
    extcounts=defaultdict(int)
    for j in allocated_indices:
        if cross[j]:
            node=edges[j]['endpoint_b'] if edges[j]['endpoint_a'] in exitset else edges[j]['endpoint_a'];extcounts[node]+=1
    excess.append(max(extcounts.values(),default=0))
    internal_indices=set(np.flatnonzero(internal))
    assert internal_indices.issubset(set(allocated_indices)), 'internal ring allocation missing'
    np.testing.assert_allclose(dist[internal],1.,atol=1e-8)
    usage=np.bincount(np.concatenate((a[allocated_indices],b[allocated_indices])),minlength=len(ids))
    assert all(usage[index[node.node_id]]<=node.isl_terminals for node in spec.nodes)
    assert all(x<=1 for x in count),'single exit assumption failed'
    if k%120==0:print('hour',t/3600,'wall',round(time.perf_counter()-t0,2),flush=True)
arr=np.array(rowcounts);up=np.array(rowready);rows=[]
for j,s in enumerate(exits):
    missing=np.flatnonzero(up[:,j]==0);runs=[x for x in np.split(missing,np.flatnonzero(np.diff(missing)>1)+1) if len(x)]
    rows.append(dict(exit=s,allocated_sample_fraction=float(np.mean(arr[:,j]==1)),ready_sample_fraction=float(np.mean(up[:,j]==1)),new_associations=int(new_count[j]),longest_no_ready_sample_span_s=max((float(times[x[-1]]-times[x[0]]) for x in runs),default=0)))
out=dict(step_s=60,epoch=epoch,cells=rows,wall_s=time.perf_counter()-t0,
    allocation_source='unmodified ae6ee46 allocation block in isolated worktree',allocation_sha256=hashlib.sha256(block.encode()).hexdigest(),
    reference_motion='local fixed-LVLH addition, NOT original-branch capability',
    max_access_links_on_one_comm_satellite=int(max(excess)),max_geometry_eligible_backbone_edges_not_allocated=int(max(backbone_lost)),
    all_exits_always_allocated_sampled=bool(np.all(arr==1)),
    scope='Sampled optical admission, 6 shared ports with no category quotas, exit 3 ports, others 2; no ground pool (independent), no online traffic, no continuous visibility proof. Setup readiness sampled at 60s; not continuous-time availability.')
(R/'shared-optical-scan.json').write_text(json.dumps(out,indent=2));np.savez_compressed(R/'shared-optical-scan.npz',times=times,allocated=arr,ready=up)
print(json.dumps(out,indent=2),flush=True)
