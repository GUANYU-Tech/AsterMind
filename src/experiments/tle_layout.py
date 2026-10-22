"""Place observed TLE satellites on a prescribed sparse grid, without filling holes.

This is a configurable grid fit, not a reproduction of the paper's automatic
multimodal shell/spacing recovery. An explicit catalog-to-slot map is preferred
when orbital planes are irregular or their nominal positions are known.
"""
import math

from orbitserve.orbit import propagate_sgp4_teme


def _phase(values, spacing):
    candidates = [v % spacing for v in values]
    return min(candidates, key=lambda p: (sum(min((v-p) % spacing, (p-v) % spacing) for v in values), p)) if candidates else 0.


def sparse_tle_layout(bundle, source, epoch):
    rows = [s.to_dict() for s in bundle.satellites]
    planes, slots = source.get('planes'), source.get('satellites_per_plane')
    if any(type(v) is not int or v < 1 for v in (planes, slots)):
        raise ValueError('TLE Grid/Honeycomb requires nominal planes and satellites_per_plane; holes must not be compressed')
    explicit = source.get('grid_positions')
    if explicit is not None:
        known = {str(r['satnum']) for r in rows}
        if not isinstance(explicit, dict) or set(explicit) - known:
            raise ValueError('grid_positions keys must be catalog numbers present in TLE')
        layout = []
        for row in rows:
            pair = explicit.get(str(row['satnum']))
            if pair is None:
                layout.append((None, None))
            elif (isinstance(pair, (list, tuple)) and len(pair) == 2 and
                  all(type(v) is int for v in pair) and 0 <= pair[0] < planes and 0 <= pair[1] < slots):
                layout.append(tuple(pair))
            else:
                raise ValueError('grid_positions values must be null or [plane, slot] within nominal grid')
        occupied = [p for p in layout if p != (None, None)]
        if len(set(occupied)) != len(occupied):
            raise ValueError('duplicate TLE grid position')
        return rows, layout, 'explicit_catalog_grid_positions'

    tolerance = source.get('grid_tolerance_fraction', .2)
    if not math.isfinite(tolerance) or not 0 < tolerance < .5:
        raise ValueError('grid_tolerance_fraction must be in (0,0.5)')
    # Evaluate all observations at one epoch before assigning angular positions.
    states = propagate_sgp4_teme(bundle, simulation_epoch_utc=epoch, sample_times_s=(0.,))
    angles = {}
    for sample in states.samples:
        r, v = sample.position_teme_km, sample.velocity_teme_km_s
        h = (r[1]*v[2]-r[2]*v[1], r[2]*v[0]-r[0]*v[2], r[0]*v[1]-r[1]*v[0])
        hn = math.sqrt(sum(x*x for x in h))
        node = (-h[1], h[0], 0.)
        nn = math.hypot(*node[:2])
        if nn < 1e-9:
            raise ValueError('equatorial TLE requires explicit grid_positions')
        a = tuple(x / nn for x in node)
        b = ((h[1]*a[2]-h[2]*a[1])/hn, (h[2]*a[0]-h[0]*a[2])/hn, (h[0]*a[1]-h[1]*a[0])/hn)
        angles[sample.satellite_id] = (math.degrees(math.atan2(node[1], node[0])) % 360,
            math.degrees(math.atan2(sum(x*y for x,y in zip(r,b)), sum(x*y for x,y in zip(r,a)))) % 360)
    dp, ds = 360 / planes, 360 / slots
    raan0 = source.get('raan0_deg', _phase([a for a,b in angles.values()], dp))
    if not math.isfinite(raan0):
        raise ValueError('raan0_deg must be finite')
    groups = {}
    layout = [(None, None)] * len(rows)
    for i, row in enumerate(rows):
        raan, u = angles[row['satellite_id']]
        p = round(((raan-raan0) % 360) / dp) % planes
        residual = abs((raan-raan0-p*dp+180) % 360 - 180)
        if residual <= tolerance * dp:
            groups.setdefault(p, []).append((i, u))
    for p, members in groups.items():
        phase = source.get('u0_deg', _phase([u for _,u in members], ds))
        if not math.isfinite(phase):
            raise ValueError('u0_deg must be finite')
        claimed = {}
        for i, u in members:
            s = round(((u-phase) % 360) / ds) % slots
            error = abs((u-phase-s*ds+180) % 360 - 180)
            if error <= tolerance * ds:
                claimed.setdefault(s, []).append((error, rows[i]['satnum'], i))
        for s, contenders in claimed.items():
            layout[min(contenders)[2]] = (p, s)
    return rows, layout, 'prescribed_uniform_grid_fit_common_epoch_v1'
