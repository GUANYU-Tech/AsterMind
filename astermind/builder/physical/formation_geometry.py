"""Equal-altitude static Cell formations, in the reference orbit's local frame."""

import math


def static_formation_offsets(count, layout, spacing_km, altitude_km):
    """Radial, along-track and cross-track offsets in km; spacing is an adjacent chord."""
    if type(count) is not int or count < 1 or layout not in ('ring', 'chain', 'fully_connected'):
        raise ValueError('invalid static formation count or layout')
    if not all(math.isfinite(x) and x > 0 for x in (spacing_km, altitude_km)):
        raise ValueError('static formation spacing and altitude must be positive and finite')
    if count == 1:
        return [(0., 0., 0.)]
    radius = 6371.0 + altitude_km
    if layout == 'chain':
        if spacing_km > 2 * radius:
            raise ValueError('static formation spacing exceeds the available orbit')
        step = 2 * math.asin(spacing_km / (2 * radius))
        angles = ((i - (count - 1) / 2) * step for i in range(count))
        return [(radius * (math.cos(a) - 1), radius * math.sin(a), 0.) for a in angles]
    ring_radius = spacing_km / (2 * math.sin(math.pi / count))
    if ring_radius > radius:
        raise ValueError('static formation ring exceeds the orbit radius')
    radial = math.sqrt(radius ** 2 - ring_radius ** 2) - radius
    angles = (2 * math.pi * i / count - math.pi / 2 for i in range(count))
    return [(radial, ring_radius * math.cos(a), ring_radius * math.sin(a)) for a in angles]
