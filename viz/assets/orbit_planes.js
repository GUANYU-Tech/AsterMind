import {createFrameCache} from './runtime_frames.js';

const earthRotation = 7.292115e-5; // rad/s about ITRS +Z
const cross = ([x,y,z],[a,b,c]) => [y*c-z*b,z*a-x*c,x*b-y*a];
const length = v => Math.hypot(...v);
const valid = v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);

// Express the other epoch's position in the current epoch's axes before
// taking the plane normal. An ITRS velocity also needs omega cross r.
function motionNormal(node, sample, dt) {
  const r = node.position_itrs_km;
  const velocity = node.velocity_itrs_km_s;
  if (valid(velocity)) {
    return cross(r, [velocity[0]-earthRotation*r[1], velocity[1]+earthRotation*r[0], velocity[2]]);
  }
  if (!valid(sample?.position_itrs_km)) return null;
  const [x,y,z] = sample.position_itrs_km, a = earthRotation*dt;
  if (valid(sample.velocity_itrs_km_s)) {
    const [vx,vy,vz]=sample.velocity_itrs_km_s;
    const [nx,ny,nz]=cross([x,y,z],[vx-earthRotation*y,vy+earthRotation*x,vz]);
    return [nx*Math.cos(a)-ny*Math.sin(a),nx*Math.sin(a)+ny*Math.cos(a),nz];
  }
  // Event frames can repeat a held position with a new business timestamp.
  if (!dt || (x===r[0] && y===r[1] && z===r[2])) return null;
  return cross(r, [x*Math.cos(a)-y*Math.sin(a), x*Math.sin(a)+y*Math.cos(a), z]);
}

export function orbitPlanes(nodes, time, sample) {
  const groups = new Map(), guides = new Map();
  const neighbors = new Map(sample?.nodes.map(n => [n.node_id, n]));
  for (const n of nodes) {
    if (!['compute','communication'].includes(n.role) || !valid(n.position_itrs_km) || !length(n.position_itrs_km)) continue;
    const placed = Number.isInteger(n.plane) && n.plane >= 0;
    const key = JSON.stringify([n.role,n.shell_id,placed ? n.plane : n.node_id]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(n);
  }
  let missing = 0;
  for (const [key, members] of groups) {
    const first = members[0], r = first.position_itrs_km, radius = length(r);
    let normal = motionNormal(first, neighbors.get(first.node_id), sample?.time_s-time);
    if (!normal || length(normal) < 1e-6) {
      // Widely spaced members define a plane even in a single stored snapshot.
      let best = 0;
      for (const other of members.slice(1)) {
        const p = other.position_itrs_km, candidate = cross(r,p), separation = length(candidate)/(radius*length(p));
        if (separation > Math.max(best,.01)) { normal = candidate; best = separation; }
      }
    }
    if (!normal || length(normal) < 1e-6) { missing++; continue; }
    const norm = length(normal);
    const dominant = normal.reduce((a,b) => Math.abs(a)>Math.abs(b) ? a : b);
    normal = normal.map(x => x*Math.sign(dominant)/norm);
    // Unassigned and formation members may share the same apparent ring.
    // Merge only nearly coincident guides within one shell and role.
    const placed = Number.isInteger(first.plane) && first.plane >= 0;
    const id = placed ? key : JSON.stringify([first.role,first.shell_id,normal.map(x=>Math.round(x*1000)),Math.round(radius/3)]);
    if (!guides.has(id)) guides.set(id,{role:first.role,normal,radius,members:[]});
    const ids = guides.get(id).members;
    for (const n of members) ids.push(n.node_id);
  }
  return {guides:[...guides.values()],missing};
}

export function createOrbitSampleLoader(base, model) {
  const motion = model.motion_frames || [], frames = model.artifacts?.frames || model.snapshots || [];
  const cache = createFrameCache(async row => {
    const response = await fetch(base+row.artifact);
    if (!response.ok) throw Error('Orbit sample unavailable');
    return response.json();
  }, 2);
  function load(rows,time,adjacent) {
    if (!rows.length) return null;
    let lo=0, hi=rows.length;
    while (lo<hi) { const mid=(lo+hi)>>>1; if (rows[mid].time_s<=time) lo=mid+1; else hi=mid; }
    let i=Math.max(0,lo-1);
    if (adjacent && rows[i].time_s===time) i = i+1<rows.length ? i+1 : Math.max(0,i-1);
    const row=rows[i];
    return row.nodes ? row : cache.get(row);
  }
  return async time => {
    if (motion.length) {
      try { return await load(motion,time,false); }
      catch (error) { if (!frames.length) throw error; }
    }
    return load(frames,time,true);
  };
}
