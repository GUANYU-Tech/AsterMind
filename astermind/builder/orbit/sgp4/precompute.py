"""Bounded orbit batches with one process pool and an in-memory handoff."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from time import perf_counter


def generate(
    output_dir,
    bundle,
    epoch,
    times,
    ground,
    pairs,
    paper,
    warnings,
    workers,
    report,
    timings,
    bundle_sink,
    compact=False,
    file_hashes=None,
):
    from astermind.builder.orbit.trace import write_orbit_trace
    from astermind.builder.orbit.sgp4.frames import compute_sgp4_sun_positions_itrs
    from astermind.builder.orbit.sgp4.trace import _geometry_partition, _limit_worker, _merge_sample_windows, _sgp4_trace_payload, _state_partition, _uniform_step

    per_time = len(bundle.satellites) * max(1, len(ground)) + len(pairs)
    size = max(
        2,
        min(
            121,
            100_000 // max(1, len(bundle.satellites)),
            1_000_000 // max(1, per_time),
        ),
    )
    offsets = list(range(0, len(times) - 1, size - 1)) or [0]
    ephemeris = []
    windows = {"sgl_windows": [], "isl_windows": [], "sunlight_windows": []}
    first = None
    pool = (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_limit_worker,
        )
        if workers > 1
        else None
    )
    try:
        for index, offset in enumerate(offsets):
            part_times = times[offset : offset + size]
            if len(offsets) > 1:
                report(
                    f"Orbit batch {index + 1}/{len(offsets)}: {part_times[0]:g}–{part_times[-1]:g} s"
                )
            report("SGP4 state precompute")
            start = perf_counter()
            count = min(workers, len(bundle.satellites))
            chunks = [
                replace(bundle, satellites=bundle.satellites[i::count])
                for i in range(count)
            ]
            jobs = [(chunk, epoch, part_times) for chunk in chunks]
            parts = (
                list(pool.map(_state_partition, jobs))
                if pool
                else [_state_partition(jobs[0])]
            )
            propagation = replace(
                parts[0][0],
                element_set_id=bundle.element_set_id,
                samples=tuple(
                    sorted(
                        (s for p, _ in parts for s in p.samples),
                        key=lambda s: (s.time_s, s.satellite_id),
                    )
                ),
            )
            earth = replace(
                parts[0][1],
                element_set_id=bundle.element_set_id,
                samples=tuple(
                    sorted(
                        (s for _, e in parts for s in e.samples),
                        key=lambda s: (s.time_s, s.satellite_id),
                    )
                ),
            )
            timings["position_s"] = (
                timings.get("position_s", 0.0) + perf_counter() - start
            )
            report("Visibility, sunlight and candidate geometry")
            start = perf_counter()
            sun = compute_sgp4_sun_positions_itrs(earth)
            count = min(workers, max(1, len(pairs) + len(ground)))
            # Only one job evaluates sunlight; all others evaluate disjoint links.
            jobs = [
                (earth, ground[i::count], pairs[i::count], sun, i == 0)
                for i in range(count)
            ]
            pieces = (
                list(pool.map(_geometry_partition, jobs))
                if pool
                else [_geometry_partition(jobs[0])]
            )
            geometry = replace(
                pieces[0],
                ground_ingresses=tuple(
                    sorted(
                        (x for p in pieces for x in p.ground_ingresses),
                        key=lambda x: x.ingress_id,
                    )
                ),
                isl_pairs=tuple(sorted(x for p in pieces for x in p.isl_pairs)),
                sgl_samples=tuple(
                    sorted(
                        (x for p in pieces for x in p.sgl_samples),
                        key=lambda x: (x.time_s, x.ingress_id, x.satellite_id),
                    )
                ),
                isl_samples=tuple(
                    sorted(
                        (x for p in pieces for x in p.isl_samples),
                        key=lambda x: (x.time_s, x.endpoint_a, x.endpoint_b),
                    )
                ),
                sgl_windows=tuple(
                    sorted(
                        (x for p in pieces for x in p.sgl_windows),
                        key=lambda x: (x.ingress_id, x.satellite_id, x.start_s),
                    )
                ),
                isl_windows=tuple(
                    sorted(
                        (x for p in pieces for x in p.isl_windows),
                        key=lambda x: (x.endpoint_a, x.endpoint_b, x.start_s),
                    )
                ),
            )
            timings["candidate_geometry_s"] = (
                timings.get("candidate_geometry_s", 0.0) + perf_counter() - start
            )
            start = perf_counter()
            payload = _sgp4_trace_payload(
                bundle, propagation, earth, geometry, paper, warnings, True
            )
            if first is None:
                first = payload
            ephemeris.extend(
                row
                for row in payload["ephemeris_samples"]
                if not index or row["time_s"] > part_times[0]
            )
            for key, rows in windows.items():
                rows.extend(payload[key])
            timings["orbit_assembly_s"] = (
                timings.get("orbit_assembly_s", 0.0) + perf_counter() - start
            )
        if len(offsets) > 1:
            first["provenance"]["precompute"] = {
                "time_batches": len(offsets),
                "max_batch_samples": size,
            }
        first["time"].update(
            start_s=times[0],
            end_s=times[-1],
            duration_s=times[-1] - times[0],
            step_s=_uniform_step(times),
        )
        first["ephemeris_samples"] = ephemeris
        for key, rows in windows.items():
            first[key] = _merge_sample_windows(rows, key) if len(offsets) > 1 else rows
        report("Canonical orbit trace export")
        start = perf_counter()
        manifest = write_orbit_trace(output_dir, **first, _bundle_sink=bundle_sink,
                                    _compact=compact, _file_hashes=file_hashes)
        timings["orbit_export_s"] = perf_counter() - start
        return manifest
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
