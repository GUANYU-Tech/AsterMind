"""Four-rank training communication over OrbitServe's shared physical network.

This models a training communication graph, not model accuracy or a calibrated
training compute predictor. Every network primitive is executed by the backend.
"""

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path

if os.environ.get("ORBIT_ASTERMIND_NATIVE"):
    import midlayer

    midlayer.__path__.insert(0, os.environ["ORBIT_ASTERMIND_NATIVE"])

import numpy as np
from astermind.collective_runtime import CollectiveRuntime
from midlayer.kernel.kernel_spec import PhysicalTransferConfig
from orbitserve.orbit import (
    dawn_dusk_elements,
    propagate_sgp4_teme,
    transform_sgp4_teme_to_itrs,
    evaluate_sgp4_geometry,
    compute_sgp4_sun_positions_itrs,
)
from orbitserve.orbit.sgp4.frames import Sgp4ItrsSample
from orbitserve.runtime import (
    NetworkSpec,
    NetworkNodeSpec,
    LinkProfile,
    RuntimeStrategySpec,
    run_simulation,
)
from orbitserve.runtime.engine.astermind_bridge import _HostSimulator
from tests.unit.runtime.test_runtime_engine import _trace, _scenario, _request
from replay_data import build_replay


class TrainingRuntime:
    serving_rate_model = "analytical_background_requests_with_midlayer_collective_graph"

    def __init__(self):
        self.jobs = {}
        self.finished = []
        self.frames = []

    def initialize(self, engine, requests):
        self.engine = engine
        service = {
            "request": {"request_id": "training-step-0"},
            "done": False,
            "path": [],
            "ledger": [],
            "external_service": True,
        }
        engine.service_jobs["training-step-0"] = service
        host = _HostSimulator(engine, service)

        def emit(kind, row):
            engine.log.append({"kind": kind, "request_id": "training-step-0", **row})

        self.collectives = CollectiveRuntime(
            host, {i: f"sat-{i}" for i in range(4)}, emit
        )
        operations = [
            ("all_reduce", 8_000_000, "ring"),
            ("reduce_scatter", 8_000_000, "ring"),
            ("all_gather", 2_000_000, "ring"),
            ("reduce", 4_000_000, "tree"),
        ]

        def launch(index):
            if index == len(operations):
                return
            operation, amount, algorithm = operations[index]
            transfer = PhysicalTransferConfig(
                algorithm=algorithm,
                bytes=amount,
                direction="bidirectional",
                link="isl",
                participants=(0, 1, 2, 3),
                operation=operation,
                ring_order=(0, 1, 2, 3),
                backend="flowsim",
                reduction="sum" if operation != "all_gather" else None,
            )

            def complete():
                self.finished.append(operation)
                # Configured local optimizer/compute barrier, not predicted GPU time.
                host.schedule(500_000_000, lambda: launch(index + 1), 0)

            self.collectives.submit("step-0/" + operation, transfer, complete)

        host.schedule(10e9, lambda: launch(0), 0)
        # Capture exactly the engine's network state at its ordinary sample boundaries.
        for time in engine.net.environment_times:
            engine.push(
                time,
                1,
                "external_callback",
                payload=lambda time=time: self.frames.append(engine.net.snapshot(time)),
            )

    def submit(self, engine, job):
        engine.on_analytical_input_done(job, None)

    def close(self):
        pass


def run(mode, backend):
    epoch = "2026-07-20T00:00:00.000000Z"
    times = tuple(float(t) for t in range(0, 241, 10))
    base = dawn_dusk_elements(
        name="dawn",
        altitude_km=600,
        satellites_per_plane=4,
        epoch_utc=epoch,
        ltan_hours=6,
    )
    sats = tuple(
        replace(row, satellite_id=f"sat-{i}", mean_anomaly_deg=10.0 * i)
        for i, row in enumerate(base.satellites)
    )
    bundle = replace(base, satellites=sats)
    fixed = transform_sgp4_teme_to_itrs(
        propagate_sgp4_teme(bundle, simulation_epoch_utc=epoch, sample_times_s=times)
    )
    if mode == "ring4":
        # A prescribed LVLH square around an SGP4-propagated reference, 80 km sides.
        centers = [s for s in fixed.samples if s.satellite_id == "sat-0"]
        formation = []
        for sample in centers:
            p, v = (
                np.array(sample.position_itrs_km),
                np.array(sample.velocity_itrs_km_s),
            )
            radial = p / np.linalg.norm(p)
            along = v - radial * np.dot(v, radial)
            along /= np.linalg.norm(along)
            normal = np.cross(radial, along)
            for i, (x, y) in enumerate(((-40, -40), (40, -40), (40, 40), (-40, 40))):
                q = p + x * along + y * normal
                formation.append(
                    Sgp4ItrsSample(
                        sample.time_s, f"sat-{i}", tuple(q), sample.velocity_itrs_km_s
                    )
                )
        fixed = replace(
            fixed,
            samples=tuple(sorted(formation, key=lambda s: (s.time_s, s.satellite_id))),
        )
    reference = next(
        s for s in fixed.samples if s.time_s == 10 and s.satellite_id == "sat-0"
    )
    x, y, z = reference.position_itrs_km
    lat, lon = (
        math.degrees(math.atan2(z, math.hypot(x, y))),
        math.degrees(math.atan2(y, x)),
    )
    ground = {
        "ingress_id": "gateway-0",
        "lat_deg": lat,
        "lon_deg": lon,
        "min_elevation_deg": 0.0,
    }
    pairs = tuple((f"sat-{a}", f"sat-{b}") for a in range(4) for b in range(a + 1, 4))
    sun = compute_sgp4_sun_positions_itrs(fixed)
    geometry = evaluate_sgp4_geometry(
        fixed,
        ground_ingresses=(ground,),
        isl_pairs=pairs,
        sun_positions=sun,
        sample_hold=True,
    )
    trace = _trace()
    trace = replace(
        trace,
        trace_manifest=replace(
            trace.trace_manifest,
            trace_id="dawn-dusk-" + mode,
            trace_model="sgp4-dawn-dusk-lvlh" if mode == "ring4" else "sgp4-dawn-dusk",
            time={
                "epoch": epoch,
                "start_s": 0.0,
                "end_s": 240.0,
                "duration_s": 240.0,
                "step_s": 10.0,
            },
        ),
        satellite_catalog=tuple(
            {"satellite_id": f"sat-{i}", "compute_capable": True} for i in range(4)
        ),
        ephemeris_samples=tuple(s.to_dict() for s in fixed.samples),
        sgl_windows=tuple(s.to_dict() for s in geometry.sgl_windows),
        isl_windows=tuple(s.to_dict() for s in geometry.isl_windows),
        sunlight_windows=tuple(s.to_dict() for s in geometry.sunlight_windows),
    )
    members = tuple(
        NetworkNodeSpec(
            f"sat-{i}",
            "compute",
            3,
            1,
            cell_id="formation-A" if mode == "ring4" else f"cell-{i}",
            cell_ingress=True,
            compute_target=i == 2 if mode == "ring4" else True,
        )
        for i in range(4)
    )
    network = NetworkSpec(
        (*members, NetworkNodeSpec("gateway-0", "ground", sgl_terminals=1)),
        {
            "cell_isl": LinkProfile(4e6, 0.0, capacity_schedule=((30.0, 0.5),)),
            "inter_cell": LinkProfile(4e6, 0.0, capacity_schedule=((30.0, 0.5),)),
            "sgl": LinkProfile(2e6, 0.0),
        },
        communication_topology="explicit",
        ground_roles=("compute",),
        cell_topology="ring",
        inter_cell_links=mode == "multi",
        cell_egress_policies={"formation-A": "free"} if mode == "ring4" else {},
        transport_model=backend,
    )
    requests = tuple(
        {
            **_request(f"new-request-{i}", t, session_id=f"session-{i}"),
            "lat_deg": lat,
            "lon_deg": lon,
            "input_bytes": 1_000_000.0,
            "prompt_tokens": 8,
            "output_tokens": 2,
        }
        for i, t in enumerate((12.0, 18.0, 35.0))
    )
    spec = _scenario(
        RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
        ),
        requests=requests,
        trace=trace,
    )
    spec = replace(spec, network=network, gateways=(ground,), time_horizon_s=240.0)
    runtime = TrainingRuntime()
    result = run_simulation(spec, compute_runtime=runtime)
    if runtime.finished != ["all_reduce", "reduce_scatter", "all_gather", "reduce"]:
        raise RuntimeError(
            f"training communication graph unfinished: {runtime.finished}"
        )
    log = result.state_summary["event_log"]
    replay = build_replay(log, list(requests), network.link_profiles, backend)
    replay["links"] = [
        {**{k: e[k] for k in ("link_id", "endpoint_a", "endpoint_b", "kind")}}
        for e in runtime.frames[0]["links"]
    ]
    starts = {e["id"]: e for e in log if e["kind"] == "primitive_start"}
    done = {e["id"]: e["time_s"] for e in log if e["kind"] == "primitive_complete"}
    for event in starts.values():
        if any(done[parent] > event["time_s"] for parent in event["dependencies"]):
            raise RuntimeError("collective dependency executed before delivery")
    allreduce = [e for e in starts.values() if e["operation"] == "all_reduce"]
    if len(allreduce) != 24 or {
        (e["source_rank"], e["target_rank"]) for e in allreduce
    } != {(0, 1), (1, 2), (2, 3), (3, 0)}:
        raise RuntimeError(
            "four-rank ring must execute six rounds of four adjacent sends"
        )
    if (
        sum(
            h["stage"] == "astermind_collective" and h["start_s"] <= 12 < h["finish_s"]
            for h in replay["hops"]
        )
        < 4
    ):
        raise RuntimeError("new request did not arrive during four-rank communication")
    crossing = [
        h
        for h in replay["hops"]
        if h["stage"] == "astermind_collective" and h["start_s"] < 30 < h["finish_s"]
    ]
    if not crossing:
        raise RuntimeError(
            "no collective retained across the 30-second capacity update"
        )
    for hop in crossing:
        if not any(
            s["interval_end_s"] <= 30 and s["link_capacity_bps"] == 4e6
            for s in hop["segments"]
        ):
            raise RuntimeError("missing old-capacity progress")
        if not any(
            s["interval_start_s"] >= 30 and s["link_capacity_bps"] == 2e6
            for s in hop["segments"]
        ):
            raise RuntimeError("missing new-capacity progress")
    return {
        "mode": mode,
        "backend": backend,
        "summary": result.request_summary,
        "collectives": [
            e
            for e in log
            if e["kind"]
            in (
                "collective_start",
                "collective_complete",
                "primitive_start",
                "primitive_complete",
            )
        ],
        "hops": replay["hops"],
        "events": [e for e in log if e["kind"] == "request_arrival"],
        "frames": runtime.frames,
        "sun_positions": [s.to_dict() for s in sun.samples],
        "network": network.to_dict(),
        "time": dict(trace.time),
        "orbit": {
            "ltan_hours": 6,
            "inclination_deg": sats[0].inclination_deg,
            "model": "SGP4 reference + prescribed LVLH formation"
            if mode == "ring4"
            else "independent SGP4 dawn-dusk orbits",
        },
        "checks": {
            "dependency_order": True,
            "four_rank_ring": True,
            "arrival_during_collective": True,
            "retained_across_capacity_change": True,
            "primitive_count": len(starts),
            "completed_count": len(done),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ring4", "multi"), required=True)
    parser.add_argument("--backend", choices=("flowsim", "ns3"), default="flowsim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = run(args.mode, args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf8"
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "backend": args.backend,
                "checks": data["checks"],
                "summary": data["summary"],
            }
        )
    )
