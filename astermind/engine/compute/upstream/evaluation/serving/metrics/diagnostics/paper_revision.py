"""Controlled CAL revision studies; interventions stay in evaluation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import sys
from unittest.mock import patch

import yaml

from evaluation.serving.metrics import sharegpt_compact_evaluation as compact
from evaluation.serving.metrics.diagnostics import mechanism_ablation
from evaluation.serving.metrics.diagnostics import graph_padding_paired_steps
from evaluation.serving.simulation import simulation_runner
from src.serving.kernel_lowering import kernel_predict




def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_case(root, case_id, load_compact_case):
    matches = [
        (config, cell)
        for config, cells in compact.official_metrics_suite_runs(root)
        for cell in cells
        if compact._canonical_cell_id(cell) == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one official serving case for {case_id}")
    config, cell = matches[0]
    case = load_compact_case(compact.compact_case_path(root, config, cell))
    return config, cell, case


def run_study(root, case_id, kind, factor, output, load_compact_case):
    """Run one independent closed loop, or a paired fixed-schedule study."""
    root, output = Path(root), Path(output)
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("intervention factor must be finite positive")
    if kind in ("baseline", "ladder", "padding") and factor != 1.0:
        raise ValueError("this study kind requires the neutral factor 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    config, cell, case = load_case(root, case_id, load_compact_case)
    if kind == "ladder":
        result = mechanism_ablation.run_mechanism_ablation(
            root, case_id, load_compact_case)
        write_json(output, result)
        return
    if kind == "padding":
        result = graph_padding_paired_steps.run_paired_steps(
            root, config, cell, case)
        write_json(output, result)
        return
    bandwidth_roles = {
        "dram_bandwidth_scale": "dram",
        "l2_bandwidth_scale": "l2_cache",
    }
    if kind not in ("baseline", "kernel_scale", "l2_scale", "reserve", *bandwidth_roles):
        raise ValueError(f"unknown intervention {kind}")
    hardware_id = cell.hardware.hardware_id
    host = compact.resolve_host_launch_profile_paths(root, [], {hardware_id})
    scheduler = compact.resolve_scheduler_host_profile_paths(
        root, [], {hardware_id})
    argv = compact.build_simulation_argv(
        root, config, cell, case,
        serving_timing_mode="device_plus_launch_plus_scheduler",
        host_launch_profile_path=host[hardware_id],
        scheduler_host_profile_path=scheduler[hardware_id])
    # Retain compact schedule rows through an observer; no predictor formulas
    # are duplicated, and production configuration files are never edited.
    original_first = simulation_runner.engine_clock_first_schedule_start_s
    original_predict = kernel_predict.predict
    trajectory = []
    unique_kernel_samples = []
    intervention = {"kind": kind, "factor": factor}
    if kind in ("l2_scale", "reserve", *bandwidth_roles):
        policy_index = argv.index("--runtime-memory-policy") + 1
        policy = json.loads(Path(argv[policy_index]).read_text())
        if kind in ("l2_scale", *bandwidth_roles):
            hw_index = argv.index("--hardware-config") + 1
            hardware = yaml.safe_load(Path(argv[hw_index]).read_text())
            resources = list(hardware["components"])
            changed = 0
            for resource in resources:
                if "children" in resource:
                    resources.extend(resource["children"])
                if kind == "l2_scale" and "role" in resource and resource["role"] == "l2_cache":
                    before = resource["size_bytes"]
                    resource["size_bytes"] = round(before * factor)
                    intervention["before_bytes"] = before
                    intervention["after_bytes"] = resource["size_bytes"]
                    changed += 1
                elif (kind in bandwidth_roles and "role" in resource
                      and resource["role"] == bandwidth_roles[kind]):
                    before = resource["throughput_per_clk"]
                    resource["throughput_per_clk"] = before * factor
                    intervention["before_bytes_per_cycle"] = before
                    intervention["after_bytes_per_cycle"] = before * factor
                    clock_hz = hardware["clock_domains"][resource["clock_domain"]]
                    intervention["before_bytes_per_s"] = before * clock_hz
                    intervention["after_bytes_per_s"] = before * factor * clock_hz
                    changed += 1
            if changed != 1:
                raise ValueError("expected exactly one modified memory resource")
            if kind == "dram_bandwidth_scale":
                directional = {}
                for field in ("dram_read_capacity_per_clk", "dram_write_capacity_per_clk"):
                    before = hardware["memory_subsystem"][field]
                    if before is None:
                        raise ValueError("bandwidth study requires directional capacities")
                    hardware["memory_subsystem"][field] = before * factor
                    directional[field] = {"before": before, "after": before * factor}
                intervention["directional_capacities_bytes_per_cycle"] = directional
            hw_path = output.with_suffix(".hardware.yaml")
            hw_path.write_text(yaml.safe_dump(hardware, sort_keys=False))
            argv[hw_index] = str(hw_path)
            digest = hashlib.sha256(hw_path.read_bytes()).hexdigest()
            for pool in policy["scheduler_pools"]:
                pool["binding"]["hardware_config_sha256"] = digest
        else:
            # Explicit hypothetical extra graph reserve in GiB. This is a
            # capacity sensitivity input, not an observed allocation.
            updates = add_graph_reserve(policy, factor)
            intervention["pool_reserve_updates"] = updates
        policy_path = output.with_suffix(".memory.json")
        write_json(policy_path, policy)
        argv[policy_index] = str(policy_path)

    def observe(trace, requests):
        first = original_first(trace, requests)
        arrivals = {r.request_id: r.arrival_time_s for r in requests}
        seen, finished = set(), set()
        for index, entry in enumerate(trace.entries):
            schedule = entry.backend_step.schedule
            seen.update(r.request_id for r in schedule.requests)
            finished.update(entry.backend_step.finished_request_ids)
            trajectory.append({
                "step_index": index,
                "start_s": entry.start_time_s,
                "end_s": entry.end_time_s,
                "batch_requests": len(schedule.requests),
                "scheduled_tokens": schedule.token_count,
                "prefill_tokens": sum(r.input_tokens for r in schedule.requests),
                "decode_tokens": sum(r.output_tokens for r in schedule.requests),
                "arrived_not_yet_scheduled": sum(
                    arrival <= entry.start_time_s and rid not in seen
                    for rid, arrival in arrivals.items()),
                "finished_requests": len(finished),
                "device_queue_wait_s": entry.device_queue_wait_s,
                "scheduler_plan_cpu_s": entry.scheduler_plan_cpu_s,
                "scheduler_update_cpu_s": entry.scheduler_update_cpu_s,
            })
        return first

    def intervene(spec, hardware, method):
        prediction = original_predict(spec, hardware, method)
        unique_kernel_samples.append({
            "kind": spec.kind.value,
            "variant": spec.variant,
            "shape": dict(spec.shape),
            "time_s": prediction.t_total_s,
            "resource_time_s": dict(prediction.t_per_resource),
        })
        if kind != "kernel_scale":
            return prediction
        return replace(
            prediction,
            t_total_s=prediction.t_total_s * factor,
            t_launch_s=prediction.t_launch_s * factor,
            t_per_group={k: v * factor for k, v in prediction.t_per_group.items()},
            t_per_resource={
                k: v * factor for k, v in prediction.t_per_resource.items()},
            diagnostics=None)

    started = time.perf_counter()
    with patch.object(
            simulation_runner, "engine_clock_first_schedule_start_s", observe
    ), patch.object(kernel_predict, "predict", intervene):
        payload = simulation_runner.run_cli_args(argv)
    observed = compact._observed_metrics(case)
    comparison = compact._compare_prediction(observed, payload, len(case.requests))
    metrics = payload["metrics"]
    summary = {
        "ttft_ms": statistics.fmean(
            r["ttft_s"] * 1000 for r in metrics["requests"].values()),
        "tpot_ms": statistics.fmean(
            r["tpot_s"] * 1000 for r in metrics["requests"].values()),
        "request_throughput_per_s": metrics["request_throughput_per_s"],
        "output_token_throughput_per_s": metrics["output_token_throughput_per_s"],
        "steps": payload["num_steps"],
        "schedule_end_time_s": payload["schedule_end_time_s"],
    }
    result = {
        "schema_version": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "case_id": case_id,
        "intervention": intervention,
        "interpretation": (
            "independent closed-loop prediction; counterfactual interventions "
            "are sensitivity studies, not measured hardware validation"),
        "host_costs_unchanged": True,
        "wall_time_s": time.perf_counter() - started,
        "argv": argv,
        "summary": summary,
        "comparison": comparison,
        "trajectory": trajectory,
        "unique_kernel_samples": unique_kernel_samples,
        "kernel_samples_scope": "cache misses only, not occurrence-weighted",
        "kv_capacity": payload["kv_capacity"],
    }
    write_json(output, result)


def add_graph_reserve(policy, factor):
    """Change exactly one graph reserve per pool or reject the intervention."""
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("intervention factor must be finite positive")
    pools = policy["scheduler_pools"]
    if not pools:
        raise ValueError("reserve intervention requires scheduler pools")
    reserves = []
    for pool in pools:
        matches = [reserve for reserve in pool["non_kv_reserves"]
                   if reserve["kind"] == "cuda_graph"]
        if len(matches) != 1:
            raise ValueError("reserve intervention requires exactly one cuda_graph reserve per pool")
        reserves.append(matches[0])
    updates = []
    for index, reserve in enumerate(reserves):
        before = reserve["bytes"]
        reserve["bytes"] += round(factor * 2**30)
        reserve["provenance"] = {
            "source_kind": "runtime_configuration",
            "source_reference": (
                "CAL revision counterfactual additional "
                f"graph reserve: {factor} GiB"),
            "source_sha256": None,
        }
        updates.append({"pool_index": index, "before_bytes": before,
                        "after_bytes": reserve["bytes"]})
    return updates


def validate_retained_study(result, case_id, kind, factor):
    """Match experiment identity without relabeling its recorded source."""
    if (result["case_id"] != case_id
            or result["intervention"]["kind"] != kind
            or result["intervention"]["factor"] != factor):
        raise ValueError("incompatible retained revision result")


def run_campaign(root, jobs):
    """Run the explicit revision matrix in independent processes."""
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("campaign jobs must be a positive integer")
    root = Path(root)
    output = root / "experiment_logs/paper_revision/sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for kind, hardware, factors in (
            ("l2_scale", "nvidia_a800", (0.5, 1.0, 2.0)),
            ("kernel_scale", "ascend_910b", (0.9, 1.0, 1.1))):
        for qps in (4, 16, 64):
            for factor in factors:
                tasks.append((kind, hardware, qps, factor))
    for qps in (4, 16, 64):
        tasks.append(("reserve", "nvidia_a800", qps, 1.0))

    def run(task):
        kind, hardware, qps, factor = task
        name = f"{kind}_{hardware}_qps{qps}_x{factor:g}"
        path = output / f"{name}.json"
        case = f"{hardware}/qwen3-4b/tp1_qps{qps}"
        if path.exists():
            validate_retained_study(
                json.loads(path.read_text()), case, kind, factor)
            return {"name": name, "status": "reused", "returncode": 0}
        command = [sys.executable,
                   str(root / "scripts/paper_plot/cal26/run_revision_study.py"),
                   "--case", case, "--kind", kind, "--factor", str(factor),
                   "--output", str(path)]
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        return {"name": name, "status": "ran", "returncode": result.returncode}

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        results = list(executor.map(run, tasks))
    write_json(output / "campaign.json", {"jobs": jobs, "results": results})
    return results
