"""Synthetic Walker mean elements and checksum-validated observed TLE inputs."""

import math
from datetime import datetime, timezone
from astermind.builder.orbit.sgp4.elements import Sgp4ElementBundle


def canonical_epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("epoch requires an explicit UTC offset")
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _bundle(rows, kind, source):
    return Sgp4ElementBundle.from_value(
        dict(
            schema_version="orbitserve.sgp4_element_bundle.v1",
            element_semantics="sgp4_mean_elements",
            gravity_model="wgs72",
            ops_mode="improved",
            time_scale="utc",
            satellites=rows,
            provenance=dict(
                source_kind=kind,
                source_name=source,
                source_version="1",
                observationally_fitted=kind
                == "observationally_fitted_general_perturbations",
            ),
        )
    )


def walker_delta_elements(
    *,
    name,
    altitude_km,
    inclination_deg,
    planes,
    satellites_per_plane,
    epoch_utc,
    walker_phase=0,
    raan0_deg=0.0,
    u0_deg=0.0,
    start_satnum=1,
):
    """Altitude defines nominal mean semimajor axis, not an observed fitted orbit."""
    if any(type(x) is not int or x < 1 for x in (planes, satellites_per_plane)):
        raise ValueError("planes and satellites_per_plane must be positive integers")
    if type(walker_phase) is not int or not 0 <= walker_phase < planes:
        raise ValueError("walker_phase must be an integer in [0,planes)")
    if not math.isfinite(altitude_km) or altitude_km <= 0:
        raise ValueError("altitude_km must be positive and finite")
    if not math.isfinite(inclination_deg) or not 0 <= inclination_deg <= 180:
        raise ValueError("inclination_deg must be in [0,180]")
    if not all(math.isfinite(x) for x in (raan0_deg, u0_deg)):
        raise ValueError("invalid Walker angular offsets")
    epoch = canonical_epoch(epoch_utc)
    radius = 6378.135 + altitude_km
    motion = math.sqrt(398600.8 / radius**3) * 86400 / (2 * math.pi)
    total = planes * satellites_per_plane
    rows = []
    for p in range(planes):
        for s in range(satellites_per_plane):
            rows.append(
                dict(
                    satellite_id=f"{name}-p{p:03d}-s{s:04d}",
                    satnum=start_satnum + len(rows),
                    classification="U",
                    epoch_utc=epoch,
                    inclination_deg=inclination_deg,
                    raan_deg=(raan0_deg + 360 * p / planes) % 360,
                    eccentricity=0.0,
                    argument_of_perigee_deg=0.0,
                    mean_anomaly_deg=(
                        u0_deg
                        + 360 * s / satellites_per_plane
                        + 360 * walker_phase * p / total
                    )
                    % 360,
                    mean_motion_rev_per_day=motion,
                    mean_motion_dot_rev_per_day2=0.0,
                    mean_motion_ddot_rev_per_day3=0.0,
                    bstar_inv_earth_radii=0.0,
                    ephemeris_type=0,
                    element_set_number=1,
                    revolution_number_at_epoch=1,
                )
            )
    return _bundle(rows, "synthetic_sgp4_mean_elements", "walker-delta-mean-elements")


def dawn_dusk_elements(*, name, altitude_km, satellites_per_plane, epoch_utc,
                       planes=1, ltan_hours=6.0, plane_spread_deg=0.0,
                       walker_phase=0, u0_deg=0.0):
    """First-order circular J2 SSO design, with epoch mean-Sun LTAN 06/18.

    Multiple planes occupy a user-selected narrow RAAN band around the
    terminator. SGP4 subsequently propagates each orbit without station keeping;
    this initializer does not guarantee constant LTAN or absence of eclipses.
    """
    from dataclasses import replace
    from astropy.time import Time

    if ltan_hours not in (6., 18.):
        raise ValueError('dawn_dusk ltan_hours must be 6 or 18')
    if not math.isfinite(plane_spread_deg) or not 0 <= plane_spread_deg <= 30:
        raise ValueError('dawn_dusk plane_spread_deg must be in [0,30]')
    if not math.isfinite(altitude_km) or altitude_km <= 0:
        raise ValueError('altitude_km must be positive and finite')
    epoch = canonical_epoch(epoch_utc)
    radius = 6378.135 + altitude_km
    n = math.sqrt(398600.8 / radius**3)
    cos_i = -(2 * math.pi / (365.2422 * 86400)) / (1.5 * 1.082616e-3 * n * (6378.135 / radius)**2)
    if not -1 <= cos_i <= 1:
        raise ValueError('altitude does not admit this circular J2 sun-synchronous design')
    # Mean solar longitude referred to the equinox of date (Meeus low-order).
    centuries = (Time(epoch, scale='utc').tt.jd - 2451545.) / 36525.
    mean_sun_ra = (280.460 + 36000.770 * centuries) % 360
    raan = (mean_sun_ra + (ltan_hours - 12) * 15) % 360
    base = walker_delta_elements(name=name, altitude_km=altitude_km,
                                inclination_deg=math.degrees(math.acos(cos_i)),
                                planes=planes, satellites_per_plane=satellites_per_plane,
                                epoch_utc=epoch, walker_phase=walker_phase, u0_deg=u0_deg)
    rows = []
    for i, satellite in enumerate(base.satellites):
        p = i // satellites_per_plane
        offset = plane_spread_deg * (p / (planes - 1) - .5) if planes > 1 else 0.
        # Stagger co-planar satellites as well, avoiding duplicate initial states.
        row = replace(satellite, raan_deg=(raan + offset) % 360,
                      mean_anomaly_deg=(u0_deg + 360 * (i % satellites_per_plane) / satellites_per_plane
                                        + 360 * p / (planes * satellites_per_plane)) % 360)
        rows.append(row.to_dict())
    return _bundle(rows, 'synthetic_sgp4_mean_elements', 'dawn-dusk-j2-mean-sun-ltan-design-v1')


def tle_elements(text, *, name):
    """Read mean elements directly from TLE, preserving their epoch and derivatives."""
    from sgp4.api import Satrec, WGS72
    from sgp4.conveniences import sat_epoch_datetime

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    rows = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("1 "):
            i += 1
        if i + 1 >= len(lines):
            raise ValueError("missing TLE element lines")
        first, second = lines[i : i + 2]
        i += 2
        if not first.startswith("1 ") or not second.startswith("2 "):
            raise ValueError("expected TLE line 1 followed by line 2")
        for line in (first, second):
            if len(line) != 69 or not line[68].isdigit():
                raise ValueError("TLE lines must have 69 columns including checksum")
            checksum = (
                sum(int(c) if c.isdigit() else 1 if c == "-" else 0 for c in line[:68])
                % 10
            )
            if checksum != int(line[68]):
                raise ValueError("TLE checksum mismatch")
        if first[2:7] != second[2:7]:
            raise ValueError("TLE satellite numbers differ")
        sat = Satrec.twoline2rv(first, second, WGS72)
        if sat.error:
            raise ValueError(f"invalid TLE initialization: {sat.error}")
        rows.append(
            dict(
                satellite_id=f"{name}-{sat.satnum}",
                satnum=sat.satnum,
                classification=sat.classification,
                epoch_utc=canonical_epoch(sat_epoch_datetime(sat).isoformat()),
                inclination_deg=math.degrees(sat.inclo),
                raan_deg=math.degrees(sat.nodeo),
                eccentricity=sat.ecco,
                argument_of_perigee_deg=math.degrees(sat.argpo),
                mean_anomaly_deg=math.degrees(sat.mo),
                mean_motion_rev_per_day=sat.no_kozai * 1440 / (2 * math.pi),
                mean_motion_dot_rev_per_day2=sat.ndot * 1440**2 / (2 * math.pi),
                mean_motion_ddot_rev_per_day3=sat.nddot * 1440**3 / (2 * math.pi),
                bstar_inv_earth_radii=sat.bstar,
                ephemeris_type=sat.ephtype,
                element_set_number=sat.elnum,
                revolution_number_at_epoch=sat.revnum,
            )
        )
    if not rows:
        raise ValueError("TLE input is empty")
    return _bundle(
        rows, "observationally_fitted_general_perturbations", "user-provided-tle"
    )
