"""Physical-network evaluation and machine-readable position/link audit output."""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
from time import perf_counter

from orbitserve.orbit import generate_orbit_trace
from orbitserve.runtime import (
    NetworkSpec,
    NetworkState,
    NetworkUnavailable,
    candidate_links,
    SUN_GEOMETRY_VERSION,
    sun_positions_from_trace,
)

TERMINAL_ALLOCATION_MODEL = 'causal_configurable_handover_service_quotas_v4'


def run_network_experiment(
    manifest: dict,
    output_dir: str | Path,
    *,
    base_dir: str | Path = ".",
    _workers: int | None = None,
    progress=None,
    export_mode: str = 'full',
    export_workers: int | None = None,
    compression_level: int = 1,
    chunk_samples: int = 32,
) -> dict:
    if export_mode not in {'core', 'full'}:
        raise ValueError('export_mode must be core or full')
    if type(compression_level) is not int or not 0 <= compression_level <= 9:
        raise ValueError('compression_level must be an integer from 0 to 9')
    if type(chunk_samples) is not int or chunk_samples < 1:
        raise ValueError('chunk_samples must be a positive integer')
    if export_workers is not None and (type(export_workers) is not int or export_workers < 1):
        raise ValueError('export_workers must be a positive integer')
    if (Path(output_dir) / "network_result.json").exists():
        raise FileExistsError(
            "completed experiment output already exists; choose a new output directory"
        )
    report = progress or (lambda stage: None)
    report("Materializing physical nodes")
    from .network_inputs import materialize_network_input, snapshot_network_input

    source_input = snapshot_network_input(manifest, base_dir)
    manifest = materialize_network_input(source_input, base_dir)
    unknown = set(manifest) - {
        "schema_version",
        "time",
        "elements",
        "network",
        "ground_ingresses",
        "route_probes",
        "initialization_provenance",
    }
    if unknown or manifest.get("schema_version") != "orbitserve.network_experiment.v1":
        raise ValueError(f"invalid network experiment schema/fields: {sorted(unknown)}")
    time = manifest["time"]
    duration, step = float(time["duration_s"]), float(time["step_s"])
    if not all(math.isfinite(x) and x > 0 for x in (duration, step)):
        raise ValueError("duration_s and step_s must be positive finite values")
    count = round(duration / step)
    if count < 1 or not math.isclose(count * step, duration, abs_tol=1e-9):
        raise ValueError("duration_s must be an integer multiple of step_s")
    times = tuple(i * step for i in range(count + 1))
    spec = NetworkSpec.from_value(manifest["network"])
    edges = candidate_links(spec)
    elements = manifest["elements"]
    if isinstance(elements, str):
        elements = Path(base_dir) / elements
    gateways = tuple(manifest["ground_ingresses"])
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "materialized_input.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report('Compute formation clearance preflight')
    from .formation_validation import validate_formation_input
    start = perf_counter()
    formation_checks = validate_formation_input(manifest, times)
    formation_check_s = perf_counter() - start
    trace_dir = Path(output_dir) / "orbit_trace"
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    budget = max(1, available - 1)
    # The backend owns resource allocation. The private override is used only
    # by serial/parallel equivalence tests, never read from an input manifest.
    workers = budget if _workers is None else min(_workers, budget)
    if type(workers) is not int or workers < 1:
        raise ValueError("invalid internal worker count")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    timings = {'formation_preflight_s': formation_check_s}
    file_hashes = {}
    trace_bundle = []
    generate_orbit_trace(
        trace_dir,
        elements=elements,
        simulation_epoch_utc=time["epoch_utc"],
        sample_times_s=times,
        ground_ingresses=gateways,
        isl_pairs=[
            (edge["endpoint_a"], edge["endpoint_b"])
            for edge in edges
            if edge["kind"] != "sgl"
        ],
        workers=workers,
        progress=report,
        _timings=timings,
        _bundle_sink=trace_bundle,
        _compact=True,
        _file_hashes=file_hashes,
    )
    trace = trace_bundle.pop()
    start = perf_counter()
    network = NetworkState(trace, spec, gateways, _defer_compile=True)
    if not network.sun_positions:
        # Sun geometry is physical scene metadata even when no link profile
        # enables exclusion. It is inexpensive compared with per-link budgets.
        network.sun_positions = [(float(row['time_s']), tuple(row['position_itrs_km']))
                                 for row in sun_positions_from_trace(trace)]
        network.sun_times = tuple(t for t, _ in network.sun_positions)
    timings['network_setup_s'] = perf_counter() - start
    report("Persisting complete backend-independent physical attributes")
    report(f"Visibility screening: {len(edges)} structural pairs -> {len(network.edges)} horizon candidates")
    network.save_physical_state(output_root / 'physical_state.sqlite', workers=workers, progress=report)
    timings.update(network.precompute_timings)
    online_origin = network
    network = network.audit_timeline()
    keepouts = {
        row["name"]: row.get("keepout_m", 0.0)
        for row in manifest.get("initialization_provenance", [])
    }
    report("Core physical validation and snapshot store")
    start = perf_counter()
    checks = validate_network(network, _validation_snapshots(network, times), keepouts)
    checks['formation_preflight'] = formation_checks
    timings['validation_s'] = perf_counter() - start
    from .snapshot_store import write_store
    start = perf_counter()
    file_hashes.update(write_store(network, output_root, times, chunk_samples=chunk_samples))
    timings['snapshot_store_s'] = perf_counter() - start
    report("Route probes")
    routes = []
    for probe in manifest.get("route_probes", ()):
        probe_network = online_origin.fork()
        observations = iter(probe_network.fault_observations())
        upcoming = next(observations, None)
        policy = probe.get("policy", "earliest_arrival")
        for t in times[:-1]:
            while upcoming and upcoming[0] <= t:
                observed_at, index, restored, fault = upcoming
                probe_network.observe_fault(index, fault, observed_at, restored=restored)
                upcoming = next(observations, None)
            row = {**probe, "time_s": t, "policy": policy}
            try:
                arrival, path = probe_network.transfer(
                    probe["source"],
                    probe["target"],
                    t,
                    float(probe.get("payload_bytes", 0)),
                    policy,
                )
                row.update(completed=True, arrival_s=arrival, path=path)
            except NetworkUnavailable as exc:
                row.update(completed=False, reason=str(exc), path=[])
            routes.append(row)
    result = {
        "schema_version": "orbitserve.network_result.v1",
        "trace_id": trace.trace_id,
        "time": trace.time,
        "network": spec.to_dict(),
        "snapshots": [],
        "boundary_sample_times_s": times,
        "events": network.events,
        "routes": routes,
        "sun_positions": [
            {"time_s": t, "position_itrs_km": p} for t, p in network.sun_positions
        ],
        "checks": checks,
        "timing_s": timings,
        "physical_counts": network.physical_counts,
        "warnings": trace.warnings,
        "candidate_links": network.edges,
        "provenance": {
            "sun_geometry_version": SUN_GEOMETRY_VERSION,
            "requested_transport_model": spec.transport_model,
            "transport_execution": "not_run_geometry_and_independent_route_probes_only",
            "initialization": manifest.get("initialization_provenance", []),
            "position_source": "per_physical_satellite_sgp4",
            "execution": {
                "workers": workers,
                "cpu_count": available,
                "reserved_cores": 1 if available > 1 else 0,
            },
            "between_samples": "sample_hold",
            "link_window_boundaries": "sample_boundaries",
            "duplex": "independent_directions_full_duplex",
            "forwarding": "whole_message_store_and_forward",
            "range_changes": "sample_boundaries",
            "transmission_failure": "plan_reject_or_wait_for_complete_future_contact",
            "routing_knowledge": "deterministic_future_contact_timeline",
            "route_probes": "independent_non_reserving_queries",
            'terminal_allocation_model': TERMINAL_ALLOCATION_MODEL,
            'ground_association': {'algorithm': spec.ground_matching,
                                   'ground_preference': spec.ground_access_policy,
                                   'satellite_preference': spec.satellite_access_policy},
            'topology_repair_reference': 'https://doi.org/10.1145/3789240.3827597',
            'radio_capacity_model': 'directional_satellite_rf_pools_v1',
            'fault_knowledge': 'audit_replays_observations_probes_only_observe_past_faults',
        },
    }
    result['artifacts'] = {
        'orbit_trace': 'orbit_trace/trace_manifest.json',
        'physical_input': 'materialized_input.json',
        'snapshot_store': 'state/index.json',
        'frames': [],
        'optional_exports': 'physical_exports.json',
        'frame_semantics': 'complete_candidate_links_at_each_boundary_including_horizon_end',
    }
    from orbitserve.common import write_json

    from .physical_dataset import publish_dataset
    started = perf_counter()
    persisted = {**result, 'events': [], 'candidate_links': []}
    index = output_root / 'core_result.json'
    file_hashes[str(index.resolve())] = write_json(index, persisted)
    write_json(output_root / 'network_result.json', persisted)
    publish_dataset(output_root, manifest, source_input=source_input, file_hashes=file_hashes)
    timings['core_publish_s'] = perf_counter() - started
    report('Physical core dataset ready')
    if export_mode == 'full':
        from .network_exports import export_network_artifacts
        report('Parallel snapshot artifact export')
        exports = export_network_artifacts(
            output_root, workers=min(workers, export_workers or 8),
            compression_level=compression_level, chunk_samples=chunk_samples,
            _core_result=result)
        result['artifacts'].update(exports['artifacts'])
        timings.update(exports['timing_s'])
        frame_index = result['artifacts']['frames']
        if count <= 32:
            snapshots = []
            for row in frame_index[:-1]:
                with gzip.open(output_root / row['artifact'], 'rt', encoding='utf-8') as stream:
                    frame = json.load(stream)
                frame.pop('events_until_next_boundary', None)
                frame['links'] = [edge for edge in frame['links'] if edge['state'] != 'DOWN']
                snapshots.append(frame)
            result['snapshots'] = snapshots
        else:
            result['snapshots'] = frame_index[:-1]
    return result


def _validation_snapshots(network, times):
    """Validate all samples without expanding RF diagnostics or inactive edges."""
    access = [edge for edge in network.edges if edge['kind'] == 'sgl']
    import bisect
    for t in times[:-1]:
        links = []
        for edge in access:
            rows = network.link_frames[edge['link_id']]
            index = bisect.bisect_right(network.link_times[edge['link_id']], t) - 1
            if index >= 0 and rows[index]['state'] != 'DOWN':
                links.append(rows[index])
        yield {'nodes': [{'node_id': n.node_id, 'position_itrs_km': network.position(n.node_id, t)}
                         for n in network.spec.nodes], 'links': links}


def validate_network(
    network: NetworkState, snapshots: list[dict], keepouts=None
) -> dict:
    errors = []
    if not network.allocation_checks["passed"]:
        errors.append("terminal over-allocation")
    for history in network.link_frames.values():
        for edge in history:
            if edge["state"] != "DOWN" and edge["reason"] is not None:
                errors.append("blocked active edge")
            if edge["state"] == "UP" and edge["time_s"] < edge["ready_s"]:
                errors.append("premature UP")
    ids = set(network.nodes)
    snapshot_count = 0
    for snapshot in snapshots:
        snapshot_count += 1
        if {n["node_id"] for n in snapshot["nodes"]} != ids:
            errors.append("incomplete snapshot node inventory")
        if not all(
            math.isfinite(x) for n in snapshot["nodes"] for x in n["position_itrs_km"]
        ):
            errors.append("nonfinite node position")
        service_counts = {}
        for edge in snapshot['links']:
            if edge['kind'] != 'sgl' or edge['state'] == 'DOWN':
                continue
            a, b = edge['endpoint_a'], edge['endpoint_b']
            ground, satellite = (a, b) if network.nodes[a].role == 'ground' else (b, a)
            service = network.nodes[ground].ground_kind
            key = satellite, service
            service_counts[key] = service_counts.get(key, 0) + 1
        for (satellite, service), count in service_counts.items():
            limit = getattr(network.nodes[satellite], 'max_' + service + '_links')
            if limit is not None and count > limit:
                errors.append(f'{satellite}: {service} association quota exceeded')
    degrees = {n.node_id: 0 for n in network.spec.nodes if n.role == "communication"}
    for edge in network.edges:
        if edge["kind"] == "communication_isl":
            for node in (edge["endpoint_a"], edge["endpoint_b"]):
                degrees[node] = degrees.get(node, 0) + 1
    for shell in network.spec.communication_shells:
        members = [n for n in network.spec.nodes if n.role == 'communication' and n.shell_id == shell['shell_id']]
        shape = shell.get('grid_shape')
        complete = (shape and min(shape) >= 3 and len(members) == math.prod(shape)
                    and all(n.plane is not None and n.isl_terminals > 0 for n in members))
        if (complete and shell['topology'] in {'grid','honeycomb'}
            and shell.get("wrap_planes", True) and shell.get("wrap_slots", True)):
            expected = 4 if shell["topology"] == "grid" else 3
            if any(
                degrees[n.node_id] != expected
                for n in network.spec.nodes
                if n.role == "communication" and n.shell_id == shell["shell_id"]
            ):
                errors.append(
                    f"incorrect candidate degree in shell {shell['shell_id']}"
                )
    cell_minima = {}
    for cell in sorted({n.cell_id for n in network.nodes.values() if n.cell_id}):
        members = [n for n in network.nodes.values() if n.cell_id == cell]
        minimum = math.inf
        times = sorted({t for n in members for t, _ in network.samples[n.node_id]})
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                for t in times:
                    minimum = min(
                        minimum,
                        math.dist(
                            network.position(a.node_id, t),
                            network.position(b.node_id, t),
                        )
                        * 1000,
                    )
        if math.isfinite(minimum):
            cell_minima[cell] = minimum
            required = (keepouts or {}).get(members[0].shell_id, 0.0)
            if minimum < required:
                errors.append(
                    f"cell {cell} violates configured keepout: {minimum:.3f} < {required:.3f} m"
                )
    result = {
        "passed": not errors,
        "cell_min_separation_m": cell_minima,
        "errors": errors,
        "snapshot_count": snapshot_count,
        "physical_satellites": sum(n.role != "ground" for n in network.spec.nodes),
        "ground_nodes": sum(n.role == "ground" for n in network.spec.nodes),
        "communication_candidate_degrees": degrees,
        "event_frame_count": len(network.frames),
    }
    if errors:
        raise ValueError(f"network invariant failure: {errors[:5]}")
    return result
