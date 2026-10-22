"""Recover a public Orbit Sun ephemeris using the trace's pinned provenance."""

from ...orbit import (
    Sgp4EarthFixedResult,
    Sgp4ItrsSample,
    compute_sgp4_sun_positions_itrs,
)


def sun_positions_from_trace(trace):
    provenance = trace.trace_manifest.provenance
    transform = provenance["earth_fixed_transform"]
    earth_fixed = Sgp4EarthFixedResult(
        element_set_id=provenance["element_input"]["element_set_id"],
        simulation_epoch_utc=trace.time["epoch"],
        samples=tuple(
            Sgp4ItrsSample(
                float(row["time_s"]),
                str(row["satellite_id"]),
                tuple(row["position_itrs_km"]),
                tuple(row["velocity_itrs_km_s"]),
            )
            for row in trace.ephemeris_samples
            if str(row["satellite_id"])==str(trace.satellite_catalog[0]["satellite_id"])
        ),
        **{
            key: transform[key]
            for key in (
                "iers_a_sha256",
                "iers_readme_sha256",
                "leap_seconds_sha256",
                "earth_orientation_quality",
                "iers_data_release",
                "astropy_iers_data_version",
            )
        },
    )
    return tuple(
        row.to_dict() for row in compute_sgp4_sun_positions_itrs(earth_fixed).samples
    )
