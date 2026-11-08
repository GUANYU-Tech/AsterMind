"""Actual continuous D-only windows for the declared FP4 hardware scenarios."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind
from src.serving.workload.model_config import load_model_config
from src.serving.workload.traced_flash_decode_scenario import FlashDecodeScenario
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.kernel_lowering.execution_timeline import build_execution_timeline, with_earliest_ready_resource_order
from src.serving.kv_cache.model_residency import flash_fp4_decode_capacity

HARDWARE = ("hygon_bw1100", "nvidia_rtx_pro_5500")
SCHEMES = ("tp8_ep8", "dp8_ep8", "dp8_fpga")


def scenario_hardware(bw216=False):
    return {name: HWSpec(f"configs/evaluation/experiments/{name}_216gb_fp4_decode.yaml" if bw216 and name == "hygon_bw1100" else f"configs/evaluation/experiments/{name}_fp4_decode.yaml")
        for name in (*HARDWARE, "intel_agmf039r47")}


def case_settings():
    return tuple((hw, scheme, mb) for hw in HARDWARE for scheme in SCHEMES
        for mb in ((1,) if scheme == "tp8_ep8" else (1, 2, 3)))


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    files = ("src/serving/workload/traced_flash_decode_scenario.py",
        "src/serving/workload/traced_models.py", "src/serving/workload/traced_operators.py",
        "src/serving/workload/traced_forward.py", "src/serving/deployment/deployment_config.py",
        "src/serving/kernel_lowering/heterogeneous_moe.py", "src/serving/kernel_lowering/execution_timeline.py",
        "src/serving/kernel_lowering/lower_workload.py", "src/serving/kv_cache/model_residency.py",
        "src/kernel/kernel_spec.py", "src/kernel/hardware_spec.py", "src/kernel/kernel_base.py",
        "src/kernel/predictors/gemm.py", "src/kernel/predictors/communication.py", "src/kernel/predictors/vector.py",
        "evaluation/serving/metrics/flash_decode_steady.py", "configs/model/deepseek-v4-flash.json",
        "configs/model/deepseek-v4-pro.json", "configs/evaluation/experiments/hygon_bw1100_216gb_fp4_decode.yaml",
        *(f"configs/evaluation/experiments/{name}_fp4_decode.yaml" for name in (*HARDWARE, "intel_agmf039r47")))
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}


def steady_metrics(completions, *, scheme, microbatches, steps, simulation_wall_s):
    if steps < 24:
        raise ValueError("steady report requires at least24 actual tokens including warmup and drain")
    # Fixed in advance: discard first8 tokens and last8 tokens of every
    # request. The intersection gives a common observed steady time interval.
    begin_index, end_index = 7, steps - 9
    lanes = tuple((mb, rank) for mb in range(microbatches) for rank in ((0,) if scheme == "tp8_ep8" else range(8)))
    start = max(completions[(mb, begin_index, rank)] for mb, rank in lanes)
    end = min(completions[(mb, end_index, rank)] for mb, rank in lanes)
    if end <= start:
        raise ValueError("no common steady interval; extend the actual decode window")
    tokens, intervals = 0, []
    cadence = []
    for mb, rank in lanes:
        rows = 128 if scheme == "tp8_ep8" else 16
        times = tuple(completions[(mb, step, rank)] for step in range(steps))
        tokens += rows * sum(start < t <= end for t in times)
        for previous, current in zip(times, times[1:]):
            if previous >= start and current <= end:
                intervals.extend([current - previous] * rows)
        cadence.append({"microbatch_index": mb, "rank": rank, "completion_times_s": times,
            "first_measured_half_mean_s": (times[15] - times[7]) / 8,
            "second_measured_half_mean_s": (times[23] - times[15]) / 8})
    if not intervals:
        raise ValueError("steady interval contains no consecutive per-request completion pair")
    average = sum(intervals) / len(intervals)
    return {"steady_window_start_s": start, "steady_window_end_s": end,
        "steady_window_output_tokens": tokens, "steady_output_tokens_per_s": tokens / (end - start),
        "steady_tpot_mean_s": average, "steady_tpot_min_s": min(intervals), "steady_tpot_max_s": max(intervals),
        "concurrency": microbatches * 128, "microbatches": microbatches,
        "tokens_excluded_at_start_per_request": 8, "tokens_excluded_at_end_per_request": 8,
        "actual_decode_tokens_per_request": steps, "simulation_wall_s": simulation_wall_s,
        "request_tpot_interval_samples": len(intervals), "cadence": cadence,
        "throughput_definition": "actual output tokens in the common observed steady time interval / its elapsed time",
        "tpot_definition": "same-request consecutive output completion intervals fully inside the common steady interval"}


def evaluate_case(*, hardware_id, scheme, microbatches, output_dir, steps=32, seed=0,
                  layer_count=None, html=True, model_name="flash", bw216=False, fpga_cards=16):
    if (hardware_id, scheme, microbatches) not in case_settings():
        raise ValueError("case is outside the frozen14-point matrix")
    if type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError("actual decode window must contain1..128 tokens")
    if model_name not in ("flash", "pro"):
        raise ValueError("invalid model")
    if layer_count is None:
        layer_count = 43 if model_name == "flash" else 61
    if type(layer_count) is not int or not 1 <= layer_count <= (43 if model_name == "flash" else 61):
        raise ValueError("invalid target layer count")
    started = time.perf_counter()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    hardware = scenario_hardware(bw216)
    model = load_model_config(f"configs/model/deepseek-v4-{model_name}.json")
    capacity = flash_fp4_decode_capacity(model, deployment=scheme, concurrency=128 * microbatches,
        hbm_capacity_bytes=hardware[hardware_id].resource_by_role("dram").size_bytes,
        initial_context_tokens=100000, decode_tokens=steps, fpga_cards=fpga_cards)
    _write_json(path / "capacity.json", capacity)
    config = {"model_name": model_name, "bw216": bw216, "hardware_id": hardware_id, "scheme": scheme, "microbatches": microbatches,
        "concurrency": 128 * microbatches, "initial_context_tokens": 100000,
        "actual_decode_tokens_per_request": steps, "target_layers": layer_count, "seed": seed,
        "collective_startup_us": 50, "switch_transport": "direct_fully_connected_no_ring",
        "weight_storage": "all_matrix_packed_FP4_group32_E8M0_scale_no_resident_FP8_copy",
        "gpu_matrix_compute": "FP8" if hardware_id == HARDWARE[0] else "native_FP4",
        "fpga_cards": fpga_cards if scheme == "dp8_fpga" else 0,
        "fpga_chips": fpga_cards * 2 if scheme == "dp8_fpga" else 0,
        "pcie_scenario": ("two4GPU_PCIe5_switches_each_one_x16_uplink_common_nonblocking_root"
            if hardware_id == HARDWARE[1] and scheme != "dp8_fpga" else None),
        "no_prefill_or_PD": True, "no_real_hardware": True}
    _write_json(path / "configuration.json", config)
    if capacity["status"] == "capacity_rejected":
        result = {**config, "status": "capacity_rejected", "steady_metrics": None, "cpu_wall_s": time.perf_counter() - started}
        _write_json(path / "result.json", result)
        return result
    sources = tuple(FlashDecodeScenario(model, hardware_id=hardware_id, scheme=scheme,
        microbatch_index=mb, max_steps=steps, root_seed=seed, layer_count=layer_count, fpga_cards=fpga_cards) for mb in range(microbatches))
    calls, predictions, receipts = [], {}, []
    previous_samplers = {}
    sampler_ids = {}
    for step in range(steps):
        for mb, source in enumerate(sources):
            stage_started = time.perf_counter()
            graph, state = source.step()
            built = time.perf_counter()
            plan = lower_workload(graph, source.deployment, hardware)
            lowered = time.perf_counter()
            if {n.node_id for n in graph.nodes} != {n for c in plan.calls for n in c.source_node_ids}:
                raise RuntimeError("complete decoder source contains unlowered work")
            current_samplers = {}
            for call in plan.calls:
                if step and not call.dependencies:
                    dependencies = ((previous_samplers[(mb, call.rank)],) if call.rank < 8 else
                        tuple(previous_samplers[(mb, rank)] for rank in range(8)))
                    call = replace(call, dependencies=dependencies)
                calls.append(call)
                predictions[call.call_id] = predict(call.kernel_spec, hardware[call.kernel_spec.hardware_id], "roofline")
                if (call.kernel_spec.variant == "vllm_plain_greedy_sample"
                        and call.kernel_spec.attributes["stage"] == "output_int32"):
                    sampler_ids[(mb, step, call.rank)] = call.call_id
                    current_samplers[(mb, call.rank)] = call.call_id
            if set(rank for group, rank in current_samplers if group == mb) != set(range(8)):
                raise RuntimeError("every actual decode step must complete all8 GPU samplers")
            previous_samplers.update(current_samplers)
            receipt = {"step": step, "microbatch": mb, "kv_length": state["kv_length"],
                "nodes": len(graph.nodes), "calls": len(plan.calls), "all_calls_priced": True,
                "source_wall_s": built - stage_started, "lower_wall_s": lowered - built,
                "predict_wall_s": time.perf_counter() - lowered,
                "global_histograms": state["global_histograms"]}
            receipts.append(receipt)
            _write_json(path / "progress.json", {**config, "status": "pricing", "step": step,
                "microbatch": mb, "priced_calls": len(predictions), "cpu_wall_s": time.perf_counter() - started})
            source.retire()
            del graph, plan
    priced = time.perf_counter()
    _write_json(path / "progress.json", {**config, "status": "scheduling", "priced_calls": len(predictions),
        "cpu_wall_s": priced - started})
    combined = KernelPlan(calls)
    combined = with_earliest_ready_resource_order(combined, predictions)
    timeline = build_execution_timeline(combined, predictions)
    completed_by_id = {i.call_id: i.end_time_s for i in timeline.intervals}
    completions = {key: completed_by_id[identifier] for key, identifier in sampler_ids.items()}
    if scheme == "tp8_ep8":
        for step in range(steps):
            completions[(0, step, 0)] = max(completions[(0, step, rank)] for rank in range(8))
    metrics = (steady_metrics(completions, scheme=scheme, microbatches=microbatches, steps=steps,
        simulation_wall_s=time.perf_counter() - started) if steps >= 24 else None)
    result = {**config, "status": "analytical_completed" if metrics is not None else "complete_step_checkpoint",
        "steady_metrics": metrics, "priced_calls": len(predictions), "source_steps": len(receipts),
        "timeline_duration_s": timeline.duration_s, "cpu_pricing_wall_s": priced - started,
        "cpu_timeline_wall_s": time.perf_counter() - priced, "source_identity": identity,
        "kernel_kind_counts": {kind.value: sum(c.kernel_spec.kind is kind for c in combined.calls) for kind in KernelKind
            if any(c.kernel_spec.kind is kind for c in combined.calls)},
        "normal_capacity_accepted": False, "native_provider_accepted": False,
        "limitations": ["analytical_declared_hardware_scenario_not_native_provider_or_measured_accuracy",
            "GPU_vector_scalar_SFU_memory_only_approximation",
            "unknown_runtime_workspace_reserve_not_assumed_zero",
            "PCIe_scenario_is_explicit_common_root_switch_design_not_identified_board",
            "full_mesh_allreduce_broadcast_and_local_reduction_not_a_ring_or_in_switch_reduction"]}
    if html and metrics is not None:
        from evaluation.reporting.flash_decode_steady import write_flash_decode_steady_html
        detail_start = metrics["steady_window_start_s"] + .25 * (metrics["steady_window_end_s"] - metrics["steady_window_start_s"])
        detail_end = min(metrics["steady_window_end_s"], detail_start + 2 * metrics["steady_tpot_mean_s"])
        result["html"] = write_flash_decode_steady_html(combined, predictions, timeline, metrics,
            output=path / "timeline.html", configuration=config, detail_interval=(detail_start, detail_end))
    if source_identity() != identity:
        raise RuntimeError("prediction source changed during the experiment; result cannot be published")
    _write_json(path / "source_receipts.json", receipts)
    result["cpu_wall_s"] = time.perf_counter() - started
    _write_json(path / "result.json", result)
    return result


def _run(arguments):
    hw, scheme, mb, output, steps, html = arguments
    try:
        return evaluate_case(hardware_id=hw, scheme=scheme, microbatches=mb, output_dir=output, steps=steps, html=html)
    except Exception:
        result = {"hardware_id": hw, "scheme": scheme, "microbatches": mb, "status": "failed", "error": traceback.format_exc()}
        _write_json(Path(output) / "result.json", result)
        return result


def evaluate_matrix(*, output_dir, workers=4, steps=32, html=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen = source_identity()
    settings = case_settings()
    started = time.perf_counter()
    manifest = {"purpose": "user_DSV4_Flash_FP4_100000_decode_steady_14points_and_capacity18",
        "canonical_data_changes": False, "retention": "regenerable_simulation_results",
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_identity": frozen, "workers": workers, "actual_decode_tokens": steps,
        "cases": [{"hardware_id": h, "scheme": s, "microbatches": m} for h, s, m in settings]}
    _write_json(output / "run_manifest.json", manifest)
    jobs = [(hw, scheme, mb, str(output / f"{hw}__{scheme}__mb{mb}"), steps, html) for hw, scheme, mb in settings]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({name: result[name] for name in ("hardware_id", "scheme", "microbatches", "status")}), flush=True)
            _write_json(output / "matrix_progress.json", {"completed": len(results), "total": len(jobs),
                "cpu_wall_s": time.perf_counter() - started, "cases": results})
    if frozen != source_identity():
        raise RuntimeError("matrix source changed during the run")
    summary = {"status": "completed" if all(r["status"] == "analytical_completed" for r in results) else "checkpoint",
        "cases": results, "cpu_wall_s": time.perf_counter() - started, "source_identity": frozen}
    hardware, capacities = scenario_hardware(), []
    model = load_model_config("configs/model/deepseek-v4-flash.json")
    for hw in HARDWARE:
        for scheme in SCHEMES:
            for mb in (1, 2, 3):
                capacities.append({"hardware_id": hw, "scheme": scheme, "microbatches": mb,
                    "capacity": flash_fp4_decode_capacity(model, deployment=scheme, concurrency=128 * mb,
                        hbm_capacity_bytes=hardware[hw].resource_by_role("dram").size_bytes,
                        initial_context_tokens=100000, decode_tokens=steps)})
    _write_json(output / "capacity_table.json", capacities)
    from evaluation.reporting.flash_decode_steady import write_flash_decode_experiment_report
    summary["report"] = write_flash_decode_experiment_report(output, results, capacities)
    _write_json(output / "matrix_result.json", summary)
    return summary
