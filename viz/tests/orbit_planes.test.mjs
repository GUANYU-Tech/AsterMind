import assert from 'node:assert/strict';
import test from 'node:test';
import {createOrbitSampleLoader,orbitPlanes} from '../assets/orbit_planes.js';

const radius=7000, spin=7.292115e-5, inclination=98*Math.PI/180;
const point=(id,role,r,extra={})=>({node_id:id,role,position_itrs_km:r,...extra});
const sat=point('m0','compute',[radius,0,0],{shell_id:'dawn',plane:null,slot:null});
const direction=[0,Math.cos(inclination),Math.sin(inclination)];
function checkPlane(result) {
  assert.equal(result.missing,0);
  assert.equal(result.guides.length,1);
  const {normal,radius:r}=result.guides[0];
  assert.equal(r,radius);
  assert.ok(Math.abs(normal[0])<1e-10);
  assert.ok(Math.abs(normal[1]*direction[1]+normal[2]*direction[2])<1e-9);
}

test('one dawn-dusk compute satellite defines a plane from its ITRS motion, independently of sunlight',()=>{
  const velocity=[0,7.5*direction[1]-radius*spin,7.5*direction[2]];
  checkPlane(orbitPlanes([sat],0,{time_s:0,nodes:[{...sat,velocity_itrs_km_s:velocity}]}));
});

test('sparse legacy frames recover the inertial plane when seeking forward or backward',()=>{
  for (const dt of [-30,30]) {
    const phase=dt*.001, angle=-spin*dt;
    const x=radius*Math.cos(phase), y=radius*Math.sin(phase)*direction[1];
    const pos=[x*Math.cos(angle)-y*Math.sin(angle),x*Math.sin(angle)+y*Math.cos(angle),radius*Math.sin(phase)*direction[2]];
    checkPlane(orbitPlanes([sat],0,{time_s:dt,nodes:[{...sat,position_itrs_km:pos}]}));
  }
  for (const sample of [undefined,{time_s:30,nodes:[sat]}]) {
    assert.deepEqual(orbitPlanes([sat],0,sample),{guides:[],missing:1});
  }
});

test('separate shell planes and roles survive; ground nodes never produce orbits',()=>{
  const nodes=[point('g','ground',[6400,0,0])];
  for (const role of ['communication','compute']) {
    for (const shell_id of ['A','B']) {
      nodes.push(point(role+shell_id+'0',role,[radius,0,0],{plane:0,shell_id}),
        point(role+shell_id+'1',role,[0,0,radius],{plane:0,shell_id}));
    }
  }
  const {guides,missing}=orbitPlanes(nodes,0);
  assert.equal(missing,0);assert.equal(guides.length,4);
  assert.ok(guides.every(g=>g.members.length===2));
});

test('legacy sample lookup uses a distinct frame at both ends of playback',async()=>{
  const frames=[{time_s:0,nodes:[sat]},{time_s:2,nodes:[sat]}];
  const load=createOrbitSampleLoader('',{snapshots:frames});
  assert.equal(await load(0),frames[1]);assert.equal(await load(2),frames[0]);
});
