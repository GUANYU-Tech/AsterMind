// Rendering-only projection. Distances/links remain authoritative on the backend.
export function sceneData(config, preview = null) {
  const ids = config.hardware?.satellite_ids || [];
  let edges = (config.topology?.inter_satellite?.links || []).map(link => [ids.indexOf(link.source), ids.indexOf(link.target)]);
  let positions = null;
  if (preview?.topology) {
    const topology = preview.topology;
    const nodes = new Map(topology.nodes.map(node => [node.id, node.satellite_id]));
    edges = topology.links.filter(link => link.level === "inter_satellite" && link.enabled !== false)
      .map(link => [ids.indexOf(nodes.get(link.source)), ids.indexOf(nodes.get(link.target))]);
    const states = preview.geometry?.satellites || topology.satellites;
    const selected = ids.map(id => states.find(state => state.satellite_id === id));
    const coordinate = ["teme_position_km", "ecef_position_km"].find(key => selected.length && selected.every(state => Array.isArray(state?.[key]) && state[key].length === 3 && state[key].every(Number.isFinite)));
    if (coordinate) {
      const vectors = selected.map(state => state[coordinate]);
      const center = [0, 1, 2].map(axis => vectors.reduce((sum, point) => sum + point[axis], 0) / vectors.length);
      const relative = vectors.map(point => point.map((v, axis) => v - center[axis]));
      const radius = Math.max(...relative.map(point => Math.hypot(...point)), 1e-12);
      positions = relative.map(point => point.map(v => v / radius));
    }
  }
  const unique = new Map();
  for (const [a, b] of edges) if (a >= 0 && b >= 0 && a !== b) unique.set([a,b].sort((x,y) => x-y).join("-"), [a,b]);
  return { edges: [...unique.values()], positions };
}

// A rigid rotation for the local preview. Preserve every relative distance and
// depth; orient a planar formation towards the viewer instead of edge-on.
export function orientPositions(points) {
  if (!points?.length) return null;
  const dot = (a,b) => a.reduce((sum,v,i) => sum + v*b[i], 0);
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const unit = a => { const length = Math.hypot(...a); return length > 1e-12 ? a.map(v => v/length) : null; };
  const direction = points.at(-1).map((v,i) => v-points[0][i]);
  const x = unit(direction) || unit(points.find(p => Math.hypot(...p) > 1e-12) || []) || [1,0,0];
  const candidates = points.map(point => cross(x,point));
  let normal = candidates.reduce((best,v) => Math.hypot(...v) > Math.hypot(...best) ? v : best, [0,0,0]);
  if (Math.hypot(...normal) < 1e-12) {
    const basis = [[1,0,0],[0,1,0],[0,0,1]].sort((a,b) => Math.abs(dot(x,a))-Math.abs(dot(x,b)))[0];
    normal = cross(x,basis);
  }
  const z = unit(normal), y = unit(cross(z,x));
  return points.map(point => [dot(point,x),dot(point,y),dot(point,z)]);
}

// Fit both the Earth and the foreground formation; Earth stays the orbit target.
export function earthViewDistance(points, aspect, fov = 42) {
  const tangent = Math.tan(fov * Math.PI / 360);
  const horizontal = tangent * Math.max(.1, aspect);
  const halfAngle = Math.atan(Math.min(tangent, horizontal));
  let distance = 2.55 / Math.sin(halfAngle);
  for (const [x,y,z] of points) {
    distance = Math.max(distance, z + (Math.abs(x) + .3) / horizontal,
      z + (Math.abs(y) + .3) / tangent);
  }
  return distance * 1.08;
}
