"""Full 72x22 communication shell and real Midlayer parallel inference."""

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

if os.environ.get("ORBIT_ASTERMIND_NATIVE"):
    import midlayer

    midlayer.__path__.insert(0, os.environ["ORBIT_ASTERMIND_NATIVE"])

import numpy as np
from orbitserve.orbit import (
    walker_delta_elements,
    dawn_dusk_elements,
    propagate_sgp4_teme,
    transform_sgp4_teme_to_itrs,
    evaluate_sgp4_geometry,
    compute_sgp4_sun_positions_itrs,
    OrbitTraceBundle,
    OrbitTraceManifest,
)
from orbitserve.orbit.sgp4.frames import Sgp4ItrsSample
from orbitserve.runtime import (
    NetworkNodeSpec,
    NetworkSpec,
    LinkProfile,
    RuntimeStrategySpec,
    run_simulation,
    candidate_links,
)
from orbitserve.runtime.state.network import NetworkState
from orbitserve.runtime.engine.astermind_bridge import AsterMindRuntime, AsterMindCell
from tests.unit.runtime.test_runtime_engine import _trace, _scenario, _request
from parallel_configs import serving_config
from replay_data import build_replay

ROOT = Path(__file__).resolve().parents[2]
ASTER = ROOT / "third_party/astermind"
OUTPUT = ROOT / "local/results/full-serving"
EPOCH = "2026-07-20T00:00:00.000000Z"


def physical(parallel):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / ("physical-v1-" + parallel + ".json")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf8"))
    config = serving_config(ASTER, parallel)
    ranks = config.world_size
    times = tuple(float(t) for t in range(0, 61, 10))
    comm = walker_delta_elements(
        name="comm",
        altitude_km=550,
        inclination_deg=53,
        planes=72,
        satellites_per_plane=22,
        walker_phase=1,
        epoch_utc=EPOCH,
    )
    dawn = dawn_dusk_elements(
        name="compute",
        altitude_km=800,
        satellites_per_plane=2,
        epoch_utc=EPOCH,
        ltan_hours=6,
    )
    dawn = replace(
        dawn,
        satellites=tuple(
            replace(s, mean_anomaly_deg=i * 10.0) for i, s in enumerate(dawn.satellites)
        ),
    )
    comm_fixed = transform_sgp4_teme_to_itrs(
        propagate_sgp4_teme(comm, simulation_epoch_utc=EPOCH, sample_times_s=times)
    )
    compute_fixed = transform_sgp4_teme_to_itrs(
        propagate_sgp4_teme(dawn, simulation_epoch_utc=EPOCH, sample_times_s=times)
    )
    cluster = []
    for state in compute_fixed.samples:
        replica = next(
            i
            for i, s in enumerate(dawn.satellites)
            if s.satellite_id == state.satellite_id
        )
        p, v = np.array(state.position_itrs_km), np.array(state.velocity_itrs_km_s)
        radial = p / np.linalg.norm(p)
        along = v - radial * np.dot(v, radial)
        along /= np.linalg.norm(along)
        normal = np.cross(radial, along)
        offsets = (
            ((-40, -40), (40, -40), (40, 40), (-40, 40))
            if ranks == 4
            else ((-40, 0), (40, 0))
        )
        for rank, (x, y) in enumerate(offsets):
            cluster.append(
                Sgp4ItrsSample(
                    state.time_s,
                    f"compute-{replica}-{rank}",
                    tuple(p + x * along + y * normal),
                    state.velocity_itrs_km_s,
                )
            )
    fixed = replace(
        comm_fixed,
        samples=tuple(
            sorted(
                (*comm_fixed.samples, *cluster),
                key=lambda s: (s.time_s, s.satellite_id),
            )
        ),
    )
    nodes = []
    for p in range(72):
        for s in range(22):
            nodes.append(
                NetworkNodeSpec(
                    f"comm-p{p:03d}-s{s:04d}",
                    "communication",
                    6,
                    8,
                    plane=p,
                    slot=s,
                    shell_id="communication-72x22",
                )
            )
    for replica in range(2):
        for rank in range(ranks):
            nodes.append(
                NetworkNodeSpec(
                    f"compute-{replica}-{rank}",
                    "compute",
                    6,
                    0,
                    cell_id=f"replica-{replica}",
                    cell_ingress=rank == 0,
                    compute_target=rank == 0,
                )
            )
    # All four user positions are present in every run, keeping physical input identical.
    target = next(
        s for s in cluster if s.time_s == 0 and s.satellite_id == "compute-0-0"
    )
    x, y, z = target.position_itrs_km
    target_lon = math.degrees(math.atan2(y, x))
    gateways = []
    for i, offset in enumerate((-40.0, -25.0, 25.0, 40.0)):
        gateways.append(
            {
                "ingress_id": f"user-{i}",
                "lat_deg": 10.0 + i * 4.0,
                "lon_deg": (target_lon + offset + 180) % 360 - 180,
                "min_elevation_deg": 5.0,
            }
        )
        nodes.append(
            NetworkNodeSpec(f"user-{i}", "ground", sgl_terminals=1, ground_kind="user")
        )
    network = NetworkSpec(
        tuple(nodes),
        {
            "communication_isl": LinkProfile(80e6, 0.0, max_distance_km=5000.0),
            "sgl": LinkProfile(16e6, 0.0, max_distance_km=3500.0),
            "cross_layer": LinkProfile(40e6, 0.0, max_distance_km=5000.0),
            "cell_isl": LinkProfile(32e6, 0.0, max_distance_km=200.0),
        },
        communication_topology="grid",
        communication_grid_shape=(72, 22),
        cell_topology="ring",
        ground_roles=("communication",),
        cell_egress_policies={"replica-0": "single", "replica-1": "single"},
    )
    edges = candidate_links(network)
    pairs = [(e["endpoint_a"], e["endpoint_b"]) for e in edges if e["kind"] != "sgl"]
    sun = compute_sgp4_sun_positions_itrs(fixed)
    print(
        "geometry: 72x22 shell,",
        len(fixed.samples),
        "states,",
        len(pairs),
        "ISL candidates",
        flush=True,
    )
    geometry = evaluate_sgp4_geometry(
        fixed,
        ground_ingresses=gateways,
        isl_pairs=pairs,
        sun_positions=sun,
        sample_hold=True,
    )
    trace = _trace()
    trace = replace(
        trace,
        trace_manifest=replace(
            trace.trace_manifest,
            trace_id="full-shell-" + parallel,
            trace_model="sgp4-walker-72x22-and-dawn-dusk-lvlh",
            time={
                "epoch": EPOCH,
                "start_s": 0.0,
                "end_s": 60.0,
                "duration_s": 60.0,
                "step_s": 10.0,
            },
        ),
        satellite_catalog=tuple(
            {"satellite_id": n.node_id, "compute_capable": n.role == "compute"}
            for n in nodes
            if n.role != "ground"
        ),
        ephemeris_samples=tuple(s.to_dict() for s in fixed.samples),
        sgl_windows=tuple(s.to_dict() for s in geometry.sgl_windows),
        isl_windows=tuple(s.to_dict() for s in geometry.isl_windows),
        sunlight_windows=tuple(s.to_dict() for s in geometry.sunlight_windows),
    )
    print("compile physical network", flush=True)
    net = NetworkState(trace, network, gateways)
    frames = [net.snapshot(t) for t in times]
    payload = {
        "trace": asdict(trace),
        "network": network.to_dict(),
        "gateways": gateways,
        "frames": frames,
        "sun_positions": [s.to_dict() for s in sun.samples],
        "ranks_per_replica": ranks,
        "shell_satellites": 1584,
        "replicas": 2,
    }
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf8")
    temporary.replace(path)
    return payload


def run(users, backend, parallel):
    source = physical(parallel)
    raw = source["trace"]
    trace = OrbitTraceBundle(
        **{**raw, "trace_manifest": OrbitTraceManifest(**raw["trace_manifest"])}
    )
    network = NetworkSpec.from_value({**source["network"], "transport_model": backend})
    count = 1 if users == "single" else 4
    requests = []
    for turn in range(1 if users == "single" else 2):
        for i in range(count):
            ground = source["gateways"][i]
            requests.append(
                {
                    **_request(
                        f"user-{i}-request-{turn}",
                        2.0 + turn * 0.6 + i * 0.06,
                        session_id=f"independent-{i}-{turn}",
                    ),
                    **{k: ground[k] for k in ("ingress_id", "lat_deg", "lon_deg")},
                    "prompt_tokens": 8,
                    "output_tokens": 2,
                    "input_bytes": 256_000.0,
                    "output_bytes": 512.0,
                }
            )
    base = _scenario(
        RuntimeStrategySpec(
            latency_weight=0.01,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.99,
        ),
        requests=requests,
        trace=trace,
    )
    spec = replace(
        base, network=network, gateways=tuple(source["gateways"]), time_horizon_s=60.0
    )
    config = serving_config(ASTER, parallel)
    cells = {
        f"compute-{replica}-0": AsterMindCell(
            config,
            {i: f"compute-{replica}-{i}" for i in range(config.world_size)},
            ASTER,
        )
        for replica in range(2)
    }
    kernels = Counter()
    native = []

    def emit(node, kind, row):
        if kind == "execution_timeline":
            kernels[node] += 1
        if kind == "requests" and row.get("event") in ("arrival", "completion"):
            native.append(
                {
                    "node": node,
                    "kind": row["event"],
                    "time_s": row["time_s"],
                    "request_id": row.get("request_id") or row["request"].request_id,
                }
            )

    bridge = AsterMindRuntime(cells, emit=emit, record_packets=backend == "ns3")
    print("run:", users, backend, parallel, len(requests), "requests", flush=True)
    result = run_simulation(spec, compute_runtime=bridge)
    if result.request_summary["completed_request_count"] != len(requests):
        raise RuntimeError(f"incomplete full-serving run: {result.request_summary}")
    log = result.state_summary["event_log"]
    replay = build_replay(log, requests, network.link_profiles, backend)
    records = list(result.request_records)
    for record in records:
        path = record["routing"]["selected_path"]
        kinds = {leg.get("link_kind", leg.get("kind")) for leg in path}
        if not {"sgl", "cross_layer", "communication_isl"} <= kinds:
            raise RuntimeError(
                "request did not traverse the communication shell end to end"
            )
        if record["timing"]["e2e_s"] is None:
            raise RuntimeError("reply did not reach the user")
    replicas_used = sorted({job["node"] for job in bridge.jobs.values()})
    if users == "multi" and len(replicas_used) != 2:
        raise RuntimeError("multi-user run did not exercise both independent replicas")
    primitives = [e for e in log if e["kind"] == "astermind_flow"]
    operation_counts = Counter(
        "pipeline"
        if "pipeline" in e["flow_id"] or "pp-" in e["flow_id"]
        else "all_reduce"
        if "all-reduce" in e["flow_id"]
        else "all_gather"
        if "all-gather" in e["flow_id"]
        else "other"
        for e in primitives
    )
    packets = getattr(bridge.engine.transport, "packet_trace", [])
    if backend == "ns3":
        counters = result.provenance["network_transport"]["native_counters"]
        if (
            len(packets) != counters["packets_started"]
            or sum(p["wire_bytes"] for p in packets) != counters["wire_bytes_started"]
        ):
            raise RuntimeError("native packet trace does not match wire counters")
    return {
        "users": users,
        "backend": backend,
        "parallel": parallel,
        "summary": result.request_summary,
        "physical_artifact": "physical-v1-" + parallel + ".json",
        "records": records,
        "hops": replay["hops"],
        "events": [
            e
            for e in log
            if e["kind"]
            in (
                "request_arrival",
                "input_done",
                "astermind_first_token",
                "astermind_compute_complete",
                "request_complete",
            )
        ],
        "kernels": dict(kernels),
        "primitive_count": len(primitives),
        "operation_counts": dict(operation_counts),
        "native_requests": native,
        "replicas_used": replicas_used,
        "provenance": result.provenance["network_transport"],
        "packets": packets,
        "payload_hop_bytes": sum(h["bytes"] for h in replay["hops"]),
        "checks": {
            "communication_shell_satellites": 1584,
            "end_to_end_path": True,
            "reply_delivered": True,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", choices=("single", "multi"), default="single")
    parser.add_argument("--backend", choices=("flowsim", "ns3"), default="flowsim")
    parser.add_argument("--parallel", choices=("tp_pp", "ep"), default="tp_pp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    payload = (
        physical(args.parallel)
        if args.prepare_only
        else run(args.users, args.backend, args.parallel)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, allow_nan=False), encoding="utf8")
    print("done", args.users, args.backend, args.parallel, flush=True)
