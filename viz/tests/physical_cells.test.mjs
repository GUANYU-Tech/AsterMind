import assert from 'node:assert/strict';
import test from 'node:test';
import {placeMemberLabel} from '../cell/scene-geometry.mjs';
import {formationPairs} from '../cell/config-edit.mjs';
import {newPhysicalPreset,applyCell,cellCount,cellGeometry,cellIndex,cellKeys,cellLayout,deploymentFromPhysical,loadPhysicalCompute,memberIds,physicalPayload,satelliteCount} from '../assets/physical-cells.js';

const formation=()=>({name:'east',source:{type:'walker_delta',planes:1,clusters_per_plane:3,altitude_km:500},
  cluster:{satellites_per_cluster:2,grid_shape:'2x1',formation_pitch_m:500},topology:'ring'});
const layout=members=>({satellite_gpus:Object.fromEntries(members.map(id=>[id,2])),
  rank_devices:[members[1]+'/gpu1',members[0]+'/gpu0'],deployment:{hardware_id:'nvidia_h200',gpus_per_satellite:2}});

test('physical saves store static bandwidth factors only in Fast Mode',()=>{
  const east=formation();
  east.compute_parameters={template:{configuration:{compute_mode:'fast_mode',fast_link_contention:true}}};
  const network={constellations:{compute_formations:[east]},network:{}};
  const factors={cross_layer:0.95,cell_isl:0.9};
  const fast=physicalPayload(network,'fast_mode','auto',null,factors);
  assert.deepEqual(fast.compute_configuration.link_bandwidth_factors,factors);
  assert.equal(fast.compute_configuration.formations.east.template.configuration.fast_link_contention,undefined);
  assert.equal(fast.physical_network.network.link_bandwidth_factors,undefined);
  const detailed=physicalPayload({...network,constellations:{compute_formations:[]}},'detail_mode','auto',null,factors);
  assert.equal(detailed.compute_configuration.link_bandwidth_factors,undefined);
});

test('a Cell template preserves custom rank slots across physical save, reload and expansion',()=>{
  const east=formation(),west={...formation(),name:'west'};
  const [a,b,c]=cellKeys(east);
  const first=cellGeometry(east,a);
  const template={...first,satellites_per_cluster:4,grid_shape:'2x2',topology:'fully_connected'};
  const members=memberIds(east,b,template);
  applyCell(east,b,template,layout(members),true);
  const override={...template,satellites_per_cluster:3,grid_shape:'3x1'};
  applyCell(east,c,override,layout(memberIds(east,c,override)),false);
  east.source.clusters_per_plane++;
  assert.equal(satelliteCount(east),15);
  assert.equal(cellGeometry(east,a).grid_shape,'2x2');
  assert.equal(cellGeometry(west,a).grid_shape,'2x1');
  const doc=physicalPayload({constellations:{compute_formations:[east,west]}},'detail_mode','auto');
  assert.equal(doc.physical_network.constellations.compute_formations[0].compute_parameters,undefined);
  const actual=Object.fromEntries([east,west].flatMap(f=>cellKeys(f).map(k=>[f.name+'-'+k,memberIds(f,k,cellGeometry(f,k))])));
  const result=deploymentFromPhysical(JSON.parse(JSON.stringify(doc)),actual);
  for(const key of cellKeys(east)){
    const id=east.name+'-'+key;
    assert.deepEqual(result[id].rank_devices,[actual[id][1]+'/gpu1',actual[id][0]+'/gpu0']);
  }
  assert.equal(Object.keys(result).length,4);
  east.source.clusters_per_plane=2;
  const reduced=physicalPayload({constellations:{compute_formations:[east]}},'detail_mode','auto');
  assert.deepEqual(reduced.compute_configuration.formations.east.cells,{});
  assert.deepEqual(reduced.physical_network.constellations.compute_formations[0].cell_overrides,{});
});

test('TLE templates remap by catalog member order and retain observed geometry',()=>{
  const f=formation();
  f.source={type:'tle',cell_members:{z:[90,10],a:[12,11]}};
  applyCell(f,'z',{topology:'fully_connected'},layout(memberIds(f,'z',{})),true);
  assert.deepEqual(cellLayout(f,'a',memberIds(f,'a',{})).rank_devices,['east-11/gpu1','east-12/gpu0']);
  assert.deepEqual(cellGeometry(f,'a'),{topology:'fully_connected'});
  assert.equal(satelliteCount(f),4);
});

test('editing one Cell preserves imported link overrides when restoring the scene budget',()=>{
  const f=formation();
  const [a,b]=cellKeys(f);
  const geometry=cellGeometry(f,a),members=memberIds(f,a,geometry);
  const sourceEdge={source:members[0],target:members[1],mode:'fixed',bandwidth_gbps:123};
  const row={...layout(members),custom_links:true,configuration:{compute_mode:'detail_mode',hardware:{},deployment:{placements:[]},
    topology:{inter_satellite:{links:[sourceEdge]}}}};
  applyCell(f,a,geometry,row,false);
  const [source,target]=memberIds(f,b,cellGeometry(f,b));
  const imported=[{source,target,profile_id:'imported'}];
  const network={constellations:{compute_formations:[f]},network:{
    link_profiles:{cell_isl:{setup_time_s:2},imported:{bandwidth_bps:2e9}},
    cell_link_overrides:{['east-'+b]:imported},
  }};
  const first=physicalPayload(network,'detail_mode','auto');
  const spec=first.physical_network.network;
  const derived=spec.cell_link_overrides['east-'+a][0].profile_id;
  assert.equal(spec.link_profiles[derived].bandwidth_bps,123e9);
  assert.equal(spec.link_profiles[derived].setup_time_s,2);
  loadPhysicalCompute(first);
  first.physical_network.constellations.compute_formations[0].compute_parameters.cells[a].custom_links=false;
  const second=physicalPayload(first.physical_network,'detail_mode','auto').physical_network.network;
  assert.deepEqual(second.cell_link_overrides,{['east-'+b]:imported});
  assert.ok(!Object.hasOwn(second.link_profiles,derived));
  assert.equal(second.link_profiles.imported.bandwidth_bps,2e9);
});

test('paged Cell IDs match backend plane and slot numbering; legacy models require no compute metadata',()=>{
  const f=formation();
  f.source.planes=1000;f.source.clusters_per_plane=1000;
  assert.equal(cellCount(f),1000000);
  assert.deepEqual(cellKeys(f,999,1001),['p000-s0999','p001-s0000']);
  assert.equal(cellIndex(f,'p001-s0000'),1000);
  assert.equal(cellIndex(f,'p001-s1000'),-1);
  f.source.planes=f.source.clusters_per_plane=1;
  assert.deepEqual(deploymentFromPhysical({physical_network:{constellations:{compute_formations:[f]}}},{}),{});
});

test('chain preview and saved template follow the same ordered open path',()=>{
  const f=formation(), [first,second]=cellKeys(f);
  const geometry={...cellGeometry(f,first),topology:'chain',satellites_per_cluster:4,grid_shape:'4x1'};
  const members=memberIds(f,first,geometry);
  applyCell(f,first,geometry,layout(members),true);
  const saved=physicalPayload({constellations:{compute_formations:[f]}},'detail_mode','auto');
  const reloaded=saved.physical_network.constellations.compute_formations[0];
  const sibling=cellGeometry(reloaded,second);
  assert.equal(sibling.topology,'chain');
  assert.deepEqual(formationPairs([3,0,2,1],sibling.topology),[[3,0],[0,2],[2,1]]);
  assert.deepEqual(formationPairs([0],'chain'),[]);
  assert.deepEqual(formationPairs([0,1],'chain'),[[0,1]]);
});

test('crowded member labels stay distinct and inside the visible canvas',()=>{
  const viewport={left:10,top:20,right:150,bottom:130}, visible=[];
  for(let i=0;i<12;i++){
    const box=placeMemberLabel(75+i%2,70,42,20,viewport,visible);
    if(!box)continue;
    assert.ok(box.left>=viewport.left && box.right<=viewport.right && box.top>=viewport.top && box.bottom<=viewport.bottom);
    for(const other of visible)assert.ok(box.right<=other.left || other.right<=box.left || box.bottom<=other.top || other.bottom<=box.top);
    visible.push(box);
  }
  assert.ok(visible.length>=2 && visible.length<12);
  assert.equal(placeMemberLabel(15,25,200,30,viewport,[]),null);
});

test('new presets and applied templates persist the same static formation the editor previews',()=>{
  const f=formation(), network={constellations:{compute_formations:[f]}};
  newPhysicalPreset(network);
  const [first,second]=cellKeys(f);
  const ring=cellGeometry(f,first);
  assert.equal(ring.motion_model,'static_formation');
  assert.ok(!('grid_shape' in ring));
  applyCell(f,first,{...ring,topology:'chain'},layout(memberIds(f,first,ring)),true);
  const saved=physicalPayload(network,'detail_mode','auto').physical_network.constellations.compute_formations[0];
  assert.equal(cellGeometry(saved,second).topology,'chain');
  assert.equal(cellGeometry(saved,second).motion_model,'static_formation');
});
