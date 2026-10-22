"""Inspect one TLE shell and propose a sparse-grid scaffold from angular modes.

This is an explicitly labeled estimate, not an operator slot assignment or the
full multimodal regularization procedure in Ronchetti et al.
"""
from collections import Counter
import math
from statistics import median

from orbitserve.orbit import tle_elements, propagate_sgp4_teme


def _spacing(gaps):
    usable = [g for g in gaps if 3 <= g <= 180]
    if not usable:
        raise ValueError('Too few distinct angular positions to infer a grid; supply nominal dimensions in JSON')
    histogram = Counter(round(g * 2) / 2 for g in usable)
    peak = min(histogram, key=lambda k: (-histogram[k], k))
    refined = median(g for g in usable if abs(g-peak) <= .75)
    count = max(1, round(360/refined))
    return count, refined, histogram[peak]


def inspect_tle_constellation(text, *, max_satellites=2500):
    count = sum(line.startswith('1 ') for line in text.splitlines())
    if count > max_satellites:
        raise ValueError(f'TLE contains {count} satellites; import one shell (limit {max_satellites}), not the entire Starlink catalog')
    bundle = tle_elements(text, name='inspection')
    rows = bundle.satellites
    epoch = sorted(s.epoch_utc for s in rows)[len(rows)//2]
    samples = propagate_sgp4_teme(bundle, simulation_epoch_utc=epoch, sample_times_s=(0.,)).samples
    angles = []
    for sample in samples:
        r,v = sample.position_teme_km,sample.velocity_teme_km_s
        h = (r[1]*v[2]-r[2]*v[1], r[2]*v[0]-r[0]*v[2], r[0]*v[1]-r[1]*v[0])
        hn,nn = math.sqrt(sum(x*x for x in h)), math.hypot(h[0],h[1])
        if nn < 1e-9:
            raise ValueError('Equatorial orbits require an explicit nominal grid')
        a = (-h[1]/nn,h[0]/nn,0.)
        b = ((h[1]*a[2]-h[2]*a[1])/hn, (h[2]*a[0]-h[0]*a[2])/hn, (h[0]*a[1]-h[1]*a[0])/hn)
        angles.append((math.degrees(math.atan2(h[0],-h[1])) % 360,
            math.degrees(math.atan2(sum(x*y for x,y in zip(r,b)),sum(x*y for x,y in zip(r,a)))) % 360))
    groups = []
    for raan,u in sorted(angles):
        if not groups or raan-groups[-1][-1][0] > 2.:
            groups.append([])
        groups[-1].append((raan,u))
    if len(groups)>1 and groups[0][0][0]+360-groups[-1][-1][0] <= 2.:
        groups[0] = [(raan-360,u) for raan,u in groups.pop()] + groups[0]
    centers = sorted((sum(a for a,u in g)/len(g)) % 360 for g in groups)
    planes,raan_gap,raan_support = _spacing([(b-a) % 360 for a,b in zip(centers,centers[1:]+centers[:1])]) if len(centers)>1 else (1,360.,1)
    slot_gaps = []
    for group in groups:
        us = sorted(u for a,u in group)
        if len(us)>1:
            slot_gaps.extend((b-a) % 360 for a,b in zip(us,us[1:]+us[:1]))
    slots,u_gap,u_support = _spacing(slot_gaps)
    altitudes = [(398600.8/(s.mean_motion_rev_per_day*2*math.pi/86400)**2)**(1/3)-6378.135 for s in rows]
    inclinations = [s.inclination_deg for s in rows]
    if max(altitudes)-min(altitudes)>30 or max(inclinations)-min(inclinations)>1:
        raise ValueError('TLE spans multiple altitude/inclination shells; extract one shell before importing')
    source={'type':'tle','tle_text':text,'planes':planes,'satellites_per_plane':slots}
    from .tle_layout import sparse_tle_layout
    _,layout,_=sparse_tle_layout(bundle,source,epoch)
    placed=sum(p is not None for p,s in layout)
    return dict(satellite_count=len(rows),simulation_epoch_utc=epoch,
        epoch_range_utc=[min(s.epoch_utc for s in rows),max(s.epoch_utc for s in rows)],
        mean_altitude_km=[min(altitudes),max(altitudes)],inclination_deg=[min(inclinations),max(inclinations)],
        observed_plane_groups=len(groups),inferred_grid_shape=[planes,slots],
        dominant_raan_gap_deg=raan_gap,dominant_slot_gap_deg=u_gap,
        angular_mode_support={'raan_gaps':raan_support,'slot_gaps':u_support},
        grid_placed_count=placed,grid_unplaced_count=len(rows)-placed,
        nominal_grid_empty_positions=planes*slots-placed,
        grid_inference='dominant_angular_modes_at_common_epoch_not_authoritative_slots',
        suggested_source=source)
