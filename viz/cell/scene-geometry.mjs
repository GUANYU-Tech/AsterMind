// Display geometry only. Link budgets and propagation use backend kilometre coordinates.
export const EARTH_RADIUS_KM = 6371;
export const EARTH_RADIUS = 2;
export const ALTITUDE_GAIN = 2;
export const MAX_FORMATION_RADIUS = .62;

export function formationDisplay(count, layout, spacingKm, altitudeKm) {
  const spacing = Math.max(0, Number(spacingKm) || 0);
  const altitude = Math.max(0, Number(altitudeKm) || 0);
  const orbitRadius = EARTH_RADIUS * (1 + ALTITUDE_GAIN * altitude / EARTH_RADIUS_KM);
  const physicalRadius = count < 2 ? 0 : (layout === 'ring' || layout === 'fully_connected')
    ? spacing / (2 * Math.sin(Math.PI / count)) : spacing * (count - 1) / 2;
  // A compact enlargement with a hard asymptotic bound, shared by both layouts.
  // Increasing distance never changes orientation or camera framing.
  const displayRadius = count < 2 ? 0 : .18 + (MAX_FORMATION_RADIUS - .18) * physicalRadius / (physicalRadius + 6);
  const arc = Math.asin(Math.min(.9, displayRadius / orbitRadius));
  const positions = Array.from({length: count}, (_, index) => {
    if (count < 2) return [0, 0, 0];
    if (layout === 'chain') {
      const angle = (index / (count - 1) * 2 - 1) * arc;
      return [orbitRadius * Math.sin(angle), 0, orbitRadius * Math.cos(angle) - orbitRadius];
    }
    const angle = index / count * Math.PI * 2 - Math.PI / 2;
    return [Math.cos(angle) * displayRadius, Math.sin(angle) * displayRadius,
      Math.sqrt(orbitRadius ** 2 - displayRadius ** 2) - orbitRadius];
  });
  let separation = Infinity;
  for (let a=0; a<positions.length; a++) for (let b=a+1; b<positions.length; b++) {
    separation = Math.min(separation, Math.hypot(...positions[a].map((v,i)=>v-positions[b][i])));
  }
  return {positions, orbitRadius, displayRadius, physicalRadius,
    satelliteScale: Math.min(.38, separation * 1.2)};
}

export function orbitDisplay(config, preview) {
  const ids = config.hardware?.satellite_ids || [];
  const states = preview?.geometry?.satellites || preview?.topology?.satellites || [];
  const selected = ids.map(id => states.find(state => state.satellite_id === id));
  if (!selected.length || !selected.every(state => Array.isArray(state?.ecef_position_km)
    && state.ecef_position_km.length === 3 && state.ecef_position_km.every(Number.isFinite))) return null;
  // ECEF Z is north; Three.js uses Y up. Keep geocentric scale, without recentering.
  return selected.map(({ecef_position_km:[x,y,z]}) => [x,z,-y].map(v=>v*EARTH_RADIUS/EARTH_RADIUS_KM));
}

export function focusedViewDistance(points, aspect, fov = 42) {
  const vertical = Math.tan(fov * Math.PI / 360);
  const horizontal = vertical * Math.max(.1, aspect);
  return Math.max(2.4, ...points.map(([x,y,z]) =>
    z + Math.max((Math.abs(x)+.18)/horizontal, (Math.abs(y)+.18)/vertical))) * 1.08;
}

// Try nearby positions in a stable order as the camera moves. Bounds are in CSS pixels.
export function placeMemberLabel(x, y, width, height, viewport, occupied) {
  const gap = 5;
  const rightSide = x + gap, leftSide = x - gap - width;
  const above = y - gap - height, below = y + gap, middle = x - width / 2;
  const centeredY = y - height / 2;
  const candidates = [
    [rightSide,below], [leftSide,below], [rightSide,above], [leftSide,above],
    [middle,above], [middle,below], [rightSide,centeredY], [leftSide,centeredY],
  ];
  for (const [candidateX, candidateY] of candidates) {
    const left = Math.round(candidateX), top = Math.round(candidateY);
    const box = {left,top,right:left + width,bottom:top + height};
    if (left < viewport.left || box.right > viewport.right || top < viewport.top || box.bottom > viewport.bottom) continue;
    if (occupied.some(other => left < other.right + 2 && box.right + 2 > other.left &&
      top < other.bottom + 2 && box.bottom + 2 > other.top)) continue;
    return box;
  }
  return null;
}
