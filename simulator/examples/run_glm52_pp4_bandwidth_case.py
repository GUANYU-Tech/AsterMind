"""Run and audit one fixed-link GLM-5.2 case with bounded trace retention."""

import argparse
from collections import Counter
from functools import partial
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

from network_sim.comm.runtime import CommunicationRuntime

from ..__main__ import json_default
from ..audit import audit
from ..config import SimulationConfig
from ..engine import Simulation


def latency_ms(values):
    ordered = sorted(values)
    return {"mean": mean(ordered) * 1000, "p50": ordered[math.ceil(len(ordered) * 0.5) - 1] * 1000,
            "p95": ordered[math.ceil(len(ordered) * 0.95) - 1] * 1000}


class ProfilePrefixComplete(Exception):
    pass


def critical_path(batch):
    """Attribute elapsed device time along one chain of latest-ready predecessors."""
    by_id = {(call["call_id"], call["occurrence"]): call for call in batch["calls"]}
    last = max(batch["calls"], key=lambda call: call["end_time_ns"])
    elapsed = Counter()
    local_kinds = Counter()
    while last is not None:
        duration = last["end_time_ns"] - last["start_time_ns"]
        if last["communication"]:
            operation = last["communication_operation"]
            category = "pp_inter_satellite" if operation == "send_recv" else "intra_satellite_collective"
        else:
            category = "local_gpu"
            local_kinds[last["kind"]] += duration
        elapsed[category] += duration
        previous = max((by_id[tuple(dep)] for dep in last["dependencies"]),
                       key=lambda call: call["end_time_ns"], default=None)
        previous_end = previous["end_time_ns"] if previous else batch["start_time_ns"]
        elapsed["dependency_gap"] += last["start_time_ns"] - previous_end
        last = previous
    assert sum(elapsed.values()) == batch["end_time_ns"] - batch["start_time_ns"]
    return dict(critical_ns=dict(elapsed), local_critical_ns_by_kind=dict(local_kinds))


def run_case(config, *, flow_library=None, profile=False, profile_prefix=None):
    if (config.backend != "flowsim" or config.orbit is not None or config.migrations
            or len(config.cells) != 1 or config.cells[0].deployment != "colocated"):
        raise ValueError("the bounded trace runner requires a single colocated Cell and fixed FlowSim links")
    if profile_prefix is not None and config.cells[0].scheduler_adapter != "vllm_0_30":
        raise ValueError("prefix profiling requires the isolated vLLM 0.30 scheduler")
    counts = Counter()
    batch_rows = []
    started = perf_counter()
    network_factory = partial(CommunicationRuntime, library=flow_library)
    with Simulation(config, network_factory=network_factory) as sim:
        if sim.network.controller is not None:
            raise ValueError("the bounded trace runner requires static links")
        cell, = sim.cells.values()
        executor, = sim.executors.values()
        boundary = executor._on_batch_boundary
        original_events = sim.loop.events
        request_event_index = 0
        request_timeline_events = {}
        timeline_events = 0
        carried_last_event = 0

        def archive_events():
            nonlocal request_event_index, timeline_events, carried_last_event
            ids = {event["event_sequence"] for event in sim.request_state_events[request_event_index:]}
            request_event_index = len(sim.request_state_events)
            request_timeline_events.update((event["sequence"], event) for event in original_events
                                           if event["sequence"] in ids)
            timeline_events += len(original_events) - carried_last_event
            original_events[:] = original_events[-1:]
            carried_last_event = len(original_events)

        def archive_batch():
            boundary()
            batch, = executor.records
            stats = audit({"requests": {}, "batches": executor.records,
                           "input_distributions": executor.input_distributions,
                           "transfers": sim.transport.transfer_events})
            counts.update({name: stats[name] for name in ("kernels", "communications", "dependencies", "transfers")})
            for call in batch["calls"]:
                if call["communication"]:
                    estimate = call["communication_estimate"]
                    counts["estimated_operator_flows"] += estimate["flow_count"]
                    if call["communication_operation"] == "send_recv":
                        counts["pp_cross_satellite_flows"] += estimate["cross_satellite_flows"]
                        counts["pp_cross_satellite_bytes"] += estimate["cross_satellite_bytes"]
            schedule = cell.steps[-1]["step"]["schedule"]
            row = {
                "step_id": batch["step_id"], "start_time_ns": batch["start_time_ns"],
                "end_time_ns": batch["end_time_ns"], "kernel_calls": len(batch["calls"]),
                "prefill_tokens": sum(r["input_tokens"] for r in schedule["requests"]),
                "decode_requests": sum(r["phase"] == "decode" for r in schedule["requests"]),
                "scheduled_requests": len(schedule["requests"]),
            }
            if profile or profile_prefix is not None:
                row.update(critical_path(batch))
                row["input_distribution_ns"] = sum(r["completed_ns"] - r["ready_ns"]
                                                   for r in executor.input_distributions)
            batch_rows.append(row)
            if len(batch_rows) == 1 or len(batch_rows) % 5 == 0:
                print(f"Batch {len(batch_rows)} · {batch['end_time_ns'] / 1e9:.3f} s simulated · "
                      f"{perf_counter() - started:.1f} s elapsed", flush=True)
            if profile_prefix == len(batch_rows):
                raise ProfilePrefixComplete
            executor.records.clear()
            executor.input_distributions.clear()
            cell.steps.clear()
            sim.transport.transfer_events.clear()
            archive_events()

        executor._on_batch_boundary = archive_batch
        try:
            report = sim.run()
        except ProfilePrefixComplete:
            # Prefix profiling deliberately leaves live requests. The normal
            # close RPC requires a drained scheduler, so shut down its worker.
            cell.backend._control_plane._force_close()
            return {
                "status": "profile_prefix_completed", "batches": batch_rows,
                "configuration": config.model_dump(), "audit": dict(passed=True, **counts),
                "execution_performance": cell.execution.performance_stats(),
                "wall_s": perf_counter() - started,
                "flow_library_sha256": sha256(flow_library.read_bytes()).hexdigest() if flow_library else None,
            }
        archive_events()
        counts["transfers"] += len(sim.transport.transfer_events)
        report.update(event_log=list(request_timeline_events.values()), batches=[], input_distributions=[])
        audit(report)
        if report["status"] != "completed":
            raise RuntimeError("requests failed to complete")

    count = len(config.requests)
    completions = report["compute_completions"]
    requests = report["requests"]
    assert len(completions) == count
    assert sum(row["prefill_tokens"] for row in batch_rows) == sum(r.input_tokens for r in config.requests)
    assert counts["pp_cross_satellite_flows"] > 0
    first = min(request["arrival_ns"] for request in requests.values()) / 1e9
    last = max(request["completed_ns"] for request in requests.values()) / 1e9
    first_ready = min(row["cell_ready_time_s"] for row in completions)
    last_compute = max(row["completion_time_s"] for row in completions)
    tokens = sum(request["received_tokens"] for request in requests.values())
    return {
        "status": report["status"], "requests": count, "output_tokens": tokens,
        "prefill_ttft_cell_ms": latency_ms(row["first_token_time_s"] - row["cell_ready_time_s"]
                                          for row in completions),
        "prefill_active_ms": latency_ms(row["prefill_active_s"] for row in completions),
        "decode_tpot_cell_ms": latency_ms(
            (row["last_token_time_s"] - row["first_token_time_s"]) / (row["generated_tokens"] - 1)
            for row in completions),
        "decode_active_ms": latency_ms(row["decode_active_s"] for row in completions),
        "ttft_user_ms": latency_ms((r["first_token_received_ns"] - r["arrival_ns"]) / 1e9
                                   for r in requests.values()),
        "decode_tpot_user_ms": latency_ms(
            (r["last_token_received_ns"] - r["first_token_received_ns"]) / 1e9 / (r["received_tokens"] - 1)
            for r in requests.values()),
        "output_tokens_per_s_cell": tokens / (last_compute - first_ready),
        "output_tokens_per_s_end_to_end": tokens / (last - first),
        "simulated_service_s": last - first, "wall_s": perf_counter() - started,
        "audit": {"passed": True, **counts, "timeline_events": timeline_events},
        "scheduler_runtime": next(iter(report["cells"].values()))["scheduler_runtime"],
        "execution_performance": next(iter(report["cells"].values()))["execution_performance"],
        "batches": batch_rows, "request_timings": requests, "compute_completions": completions,
        "configuration": report["configuration"], "provenance": report["provenance"],
        "flow_library_sha256": sha256(flow_library.read_bytes()).hexdigest() if flow_library else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--flow-library", type=Path, help="use a separately built FlowSim library")
    parser.add_argument("--profile", action="store_true", help="record the critical path of every batch")
    parser.add_argument("--profile-prefix", type=int, help="profile and stop after the specified number of batches")
    args = parser.parse_args()
    if args.profile_prefix is not None and args.profile_prefix < 1:
        parser.error("--profile-prefix must be positive")
    gc.set_threshold(*gc.get_threshold()[:2], 400)
    result = run_case(SimulationConfig.load(args.scenario), flow_library=args.flow_library,
                      profile=args.profile, profile_prefix=args.profile_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n",
                   encoding="utf-8")
    tmp.replace(args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
