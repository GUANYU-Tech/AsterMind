"""Direct solar incidence at a receiver, including opaque WGS-84 Earth."""

import math

SUN_GEOMETRY_VERSION = "receiver_visible_sun_wgs84_v1"
_EARTH_AXES_KM = (6378.137, 6378.137, 6356.752314245)


def sun_visible(observer, sun):
    """Whether the segment to the Sun centre avoids the WGS-84 ellipsoid.

    A surface receiver looking below its local horizon is occulted as well.
    Exact limb tangency is treated as visible; the finite solar disc, refraction
    and atmospheric attenuation are outside this geometric exclusion model.
    """
    origin = tuple(x / radius for x, radius in zip(observer, _EARTH_AXES_KM))
    direction = tuple((s - x) / radius for x, s, radius in zip(observer, sun, _EARTH_AXES_KM))
    length2 = sum(x * x for x in direction)
    if not length2:
        return False
    fraction = max(0.0, min(1.0, -sum(x * y for x, y in zip(origin, direction)) / length2))
    clearance2 = sum((x + fraction * y) ** 2 for x, y in zip(origin, direction))
    return clearance2 >= 1.0 - 1e-12


def receiver_sun_separation(observer, transmitter, sun):
    """Signal/Sun angle at this receiver, or None when Earth blocks the Sun."""
    if not sun_visible(observer, sun):
        return None
    signal = tuple(t - x for t, x in zip(transmitter, observer))
    solar = tuple(s - x for s, x in zip(sun, observer))
    norm = math.sqrt(sum(x * x for x in signal) * sum(x * x for x in solar))
    if not norm:
        return None
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(signal, solar)) / norm))
    return math.degrees(math.acos(cosine))
