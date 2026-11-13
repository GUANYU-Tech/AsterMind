"""Bounded CPU evaluation of complete fixed-concurrency Flash requests.

The schedule is explicit: fixed TP8 prefill groups drain, then synchronous
decode rounds. GPU work retains each full DP-lane batch; only FPGA expert
work is divided into receive/compute/send token microbatches.
This is an analytical schedule, not a claim to run a native PD Scheduler.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

from src.serving.execution import ServingExecutionContext
from src.serving.workload.model_config import load_model_config
from src.serving.workload.traced_heterogeneous_moe import MICROBATCH_SCHEDULE
from .heterogeneous_flash_moe import _hardware_specs, GPU_APPROXIMATION_LIMITATIONS


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _case(arguments):
    from src.serving.workload.traced_heterogeneous_request import FlashHeterogeneousBatchRequest
    (concurrency, microbatches, cards, prompt, outputs, seed, approximate,
        gpu_utilization, fpga_utilization, gpu_hardware_config, directory) = arguments
    started = time.monotonic()
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    identity = {"microbatch_schedule": MICROBATCH_SCHEDULE, "concurrency": concurrency, "microbatches_requested": microbatches,
        "fpga_cards": cards, "fpga_chips": cards * 2, "prompt_tokens": prompt,
        "output_tokens_per_request": outputs, "seed": seed,
        "gpu_vector_memory_only": approximate,
        "gpu_hardware_config": gpu_hardware_config,
        "gpu_hardware_config_sha256": hashlib.sha256(Path(gpu_hardware_config).read_bytes()).hexdigest(),
        "gpu_utilization": gpu_utilization, "fpga_utilization": fpga_utilization, "GPU_used": False, "worker_pid": os.getpid()}
    identity["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    _write_json(path / "progress.json", {**identity, "status": "running", "decode_rounds": 0})
    try:
        hardware = _hardware_specs(approximate, gpu_utilization=gpu_utilization,
            fpga_utilization=fpga_utilization, gpu_hardware_config=gpu_hardware_config)
        gpu_ids = tuple(name for name in hardware if name != "intel_agmf039r47")
        if len(gpu_ids) != 1:
            raise ValueError("heterogeneous request requires exactly one GPU scenario identity")
        identity["gpu_hardware_id"] = gpu_ids[0]
        identity["gpu_parameter_source"] = hardware[gpu_ids[0]].parameter_source
        model = load_model_config("configs/model/deepseek-v4-flash.json")
        program = FlashHeterogeneousBatchRequest(model, prompt_tokens=prompt,
            output_tokens=outputs, concurrency=concurrency, microbatches=microbatches,
            fpga_cards=cards, root_seed=seed, gpu_hardware_id=gpu_ids[0])
        execution = ServingExecutionContext.for_traced_device_sources(model, program.deployment, hardware,
            device_resource_order="earliest_ready")
        identity["execution_owner"] = "ServingExecutionContext.execute_source_workload"
        elapsed, total_calls, total_nodes, fpga_calls = 0.0, 0, 0, 0
        prefill_groups, rounds = [], []
        request_ids = ()

        def execute_source(builder, *, retire_source=None):
            nonlocal request_ids
            def build():
                nonlocal request_ids
                traced = builder()
                request_ids = tuple(i for batch in traced.microbatch_token_ids for i in batch)
                return traced.graph, traced.deployment
            return execution.execute_source_workload(build,
                retire_source=retire_source)

        def receipt(components, stage_started):
            nonlocal elapsed, total_calls, total_nodes, fpga_calls
            plan, predictions = components.kernel_plan, components.predictions
            graph, timeline = components.workload, components.execution_timeline
            covered = {node for call in plan.calls for node in call.source_node_ids}
            if covered != {node.node_id for node in graph.nodes}:
                raise RuntimeError("fixed-concurrency source graph contains unlowered work")
            duration = timeline.duration_s
            if duration <= 0 or len(predictions) != len(plan.calls):
                raise RuntimeError("complete fixed-concurrency stage must be fully priced and positive")
            elapsed += duration
            total_calls += len(plan.calls)
            total_nodes += len(graph.nodes)
            fpga_calls += sum(call.kernel_spec.hardware_id == "intel_agmf039r47" for call in plan.calls)
            return {"duration_s": duration, "kernel_calls": len(plan.calls),
                "workload_nodes": len(graph.nodes), "completion_s": elapsed,
                "cpu_wall_s": time.monotonic() - stage_started}

        preparing = iter(program.prepare())
        while True:
            stage_started = time.monotonic()
            try:
                components = execute_source(lambda: next(preparing),
                    retire_source=program.retire_prepared_group)
            except StopIteration:
                break
            prefill_groups.append({**receipt(components, stage_started), "request_ids": request_ids})
            del components
            _write_json(path / "progress.json", {**identity, "status": "running",
                "phase": "prefill_and_pd", "prefill_groups_completed": len(prefill_groups),
                "decode_rounds": 0, "target_decode_rounds": outputs - 1,
                "predicted_elapsed_s": elapsed, "kernel_calls": total_calls,
                "cpu_wall_s": time.monotonic() - started,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
        prepared_s = elapsed
        for index in range(outputs - 1):
            stage_started = time.monotonic()
            components = execute_source(program.decode, retire_source=program.retire_step)
            rounds.append({"round": index + 1, "kv_length": prompt + index + 1,
                **receipt(components, stage_started)})
            del components
            if index == 0 or (index + 1) % 16 == 0:
                _write_json(path / "progress.json", {**identity, "status": "running",
                    "decode_rounds": index + 1, "target_decode_rounds": outputs - 1,
                    "predicted_elapsed_s": elapsed, "kernel_calls": total_calls,
                    "cpu_wall_s": time.monotonic() - started,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
        if program.emitted_tokens != outputs:
            raise RuntimeError("fixed-concurrency program did not emit its complete output length")
        result = {**identity, "status": "completed", "layer_count": model.num_hidden_layers,
            "completed_requests": concurrency, "generated_tokens": concurrency * outputs,
            "microbatches_nonempty": min(concurrency, microbatches),
            "prefill_groups": prefill_groups, "decode_rounds": rounds,
            "kernel_calls": total_calls, "priced_calls": total_calls, "unpriced_calls": 0,
            "all_graph_nodes_lowered": True, "workload_nodes": total_nodes,
            "fpga_calls": fpga_calls, "handoff_bytes": program.handoff_bytes,
            "prefill_and_pd_prepare_s": prepared_s, "full_batch_latency_s": elapsed,
            "qps": concurrency / elapsed, "output_tokens_per_second": concurrency * outputs / elapsed,
            "mean_decode_round_s": (None if outputs == 1 else (elapsed - prepared_s) / (outputs - 1)),
            "cpu_wall_s": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "normal_capacity_accepted": False, "native_provider_accepted": False,
            "scheduler": "declared_fixed_concurrency_synchronous_rounds_not_native_pd",
            "prefill_batching": "one_fixed_TP8_group_per_DP_lane",
            "decode_gpu_batching": "fixed_DP_lane_batch_attention_router_shared_expert",
            "fpga_batching": "token_microbatches_receive_compute_send",
            "resource_scheduling": "non_delay_earliest_ready_device_only",
            "execution_cache_stats": execution.cache_stats(),
            "execution_performance_stats": execution.performance_stats(),
            "decode_source_reuse": {name: sum(trace._module_frame_stats[name] for trace in program.traces)
                for name in program.traces[0]._module_frame_stats},
            "stage_cpu_wall_scope": "source_construction_lowering_prediction_scheduling_receipt",
            "step_boundary": "previous_round_all_calls_complete_before_next_round",
            "model_config_sha256": model.config_sha256}
    except Exception as exc:
        result = {**identity, "status": "failed", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "full_batch_latency_s": None, "qps": None,
            "cpu_wall_s": time.monotonic() - started}
    _write_json(path / "result.json", result)
    _write_json(path / "progress.json", {key: value for key, value in result.items()
        if key not in ("prefill_groups", "decode_rounds", "traceback")})
    return result


def _schedule_identity(record):
    # Missing identity belongs to the original unversioned schedule. It is
    # retained as unknown, never upgraded to the corrected FPGA-only scope.
    return record["microbatch_schedule"] if "microbatch_schedule" in record else None


def _current_sweep_schedule(result):
    return (_schedule_identity(result) == MICROBATCH_SCHEDULE
        and all(_schedule_identity(case) == MICROBATCH_SCHEDULE for case in result["cases"]))


def render_request_sweep(result):
    current_schedule = _current_sweep_schedule(result)
    gpu_ids = sorted({case["gpu_hardware_id"] for case in result["cases"] if "gpu_hardware_id" in case})
    gpu_description = "/".join(gpu_ids) if gpu_ids else "GPU (identity retained in original configuration)"
    lines = ["# Flash GPU＋FPGA 固定并发 Roofline", "",
        f'运行状态：`{result["status"]}`。', "",
        f'已完成{sum(case["status"] == "completed" for case in result["cases"])}/{len(result["cases"])}组；CPU并行上限{result["workers"]}个进程，CPU亲和范围`{result["cpu_affinity"]}`，不运行GPU。', "",
        f"8张{gpu_description} Prefill TP8；另8张同型号 Decode DP8；FPGA只执行Decode路由专家。MTP关闭，均匀无放回top6随机路由、记录种子。", "",
        ("调度：固定每个DP lane的一组TP8 Prefill；GPU attention、router和shared expert保持整批，microbatch只切分FPGA routed expert的收、算、发。随后执行同步Decode轮次，共同资源/依赖时间线按数据就绪和资源空闲启动，允许收算发重叠；每轮结束设全局屏障。未调用原生PD Scheduler，未计未知软件和协议开销。"
            if current_schedule else "**已被替代的旧调度结果：schedule身份缺失或不同，MB可能同时改变Prefill和GPU整批划分，不能作为仅改变FPGA microbatch的对比。保留原执行结果，不重新标记为新调度。**"), "",
        "| FPGA卡 | 并发 | MB | 状态 | 输出/请求 | 全调用计价 | 整批时延(s) | QPS | 输出token/s |", "|---:|---:|---:|---|---:|---|---:|---:|---:|"]
    for case in result["cases"]:
        if case["status"] == "completed":
            values = f'{case["priced_calls"]}/{case["kernel_calls"]} | {case["full_batch_latency_s"]:.6f} | {case["qps"]:.6f} | {case["output_tokens_per_second"]:.3f}'
        else:
            values = "— | — | — | —"
        lines.append(f'| {case["fpga_cards"]} | {case["concurrency"]} | {case["microbatches_requested"]} | {case["status"]} | {case["output_tokens_per_request"]} | {values} |')
    gpu_utilization = result["gpu_utilization"] if "gpu_utilization" in result else 1.0
    fpga_utilization = result["fpga_utilization"] if "fpga_utilization" in result else 1.0
    lines.extend(("", f"有效算力及本地带宽：GPU {gpu_utilization:.0%}，FPGA {fpga_utilization:.0%}（含HBM与DDR）；主频、固定延时、网络带宽及延时不变。", ""))
    if "nvidia_rtx_pro_5500" in gpu_ids:
        lines.extend(("RTX PRO5500显存/带宽 + 用户指定PRO5000 dense算力proxy；不是RTX PRO5500官方算力峰值或原生provider预测。", ""))
    lines.extend(("", ("并发1时MB2/3仍只有一个非空FPGA microbatch；Prefill和GPU整批划分固定。"
        if current_schedule else "旧结果只描述其原调度；Prefill/GPU分组可能随MB变化，不满足当前FPGA-only设置。"), ""))
    lines.extend("- " + limitation for limitation in GPU_APPROXIMATION_LIMITATIONS)
    lines.append("- 本报告不证明normal容量、实际provider、硬件性能精度或checkpoint数值输出；虚拟Tensor仅元数据。")
    for case in result["cases"]:
        if "reused_from" in case:
            lines.append(f'- FPGA{case["fpga_cards"]}/C{case["concurrency"]}/MB{case["microbatches_requested"]}复用已完成结果：`{case["reused_from"]}`；原执行snapshot保留在JSON，不重标成本轮运行。')
    for case in result["cases"]:
        if case["status"] == "failed":
            lines.append(f'- FPGA{case["fpga_cards"]}/C{case["concurrency"]}/MB{case["microbatches_requested"]}失败：{case["error_type"]}: {case["error"]}')
    return "\n".join(lines) + "\n"


def evaluate_request_sweep(*, output_dir, concurrencies, microbatches, fpga_cards,
        prompt_tokens, output_tokens, seed=0, gpu_vector_memory_only=False, workers=8, snapshot,
        reuse_completed_from=None, gpu_utilization=1.0, fpga_utilization=1.0,
        gpu_hardware_config="configs/hardware/hygon_bw1100.yaml"):
    # Validate the explicit scenario before launching any CPU workers.
    hardware = _hardware_specs(gpu_vector_memory_only, gpu_utilization=gpu_utilization,
        fpga_utilization=fpga_utilization, gpu_hardware_config=gpu_hardware_config)
    gpu_ids = tuple(name for name in hardware if name != "intel_agmf039r47")
    if len(gpu_ids) != 1:
        raise ValueError("heterogeneous sweep requires exactly one GPU scenario identity")
    gpu_config_sha256 = hashlib.sha256(Path(gpu_hardware_config).read_bytes()).hexdigest()
    if type(workers) is not int or workers < 1:
        raise ValueError("request sweep workers must be a positive integer")
    if (not concurrencies or not microbatches or not fpga_cards
            or any(type(n) is not int or n < 1 for n in concurrencies)
            or any(type(n) is not int or n not in (1, 2, 3) for n in microbatches)
            or any(type(n) is not int or n not in (8, 16) for n in fpga_cards)):
        raise ValueError("invalid fixed-concurrency request settings")
    if type(prompt_tokens) is not int or prompt_tokens < 1 or type(output_tokens) is not int or output_tokens < 1:
        raise ValueError("request token lengths must be positive integers")
    settings = tuple(product(concurrencies, microbatches, fpga_cards))
    if len(set(settings)) != len(settings):
        raise ValueError("duplicate request settings are not allowed")
    available = sorted(os.sched_getaffinity(0))
    workers = min(workers, len(settings), len(available))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = {"scope": "request_sweep", "method": "roofline", "snapshot": snapshot,
        "microbatch_schedule": MICROBATCH_SCHEDULE,
        "workers": workers, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {name: os.environ[name] if name in os.environ else None
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "GPU_used": False, "real_model_tensors": False, "status": "running",
        "gpu_utilization": gpu_utilization, "fpga_utilization": fpga_utilization,
        "gpu_hardware_id": gpu_ids[0], "gpu_hardware_config": gpu_hardware_config,
        "gpu_hardware_config_sha256": gpu_config_sha256,
        "gpu_parameter_source": hardware[gpu_ids[0]].parameter_source,
        "utilization_scope": "compute_and_local_HBM_DDR_rates_only_clocks_delays_network_unchanged",
        "cases": [{"microbatch_schedule": MICROBATCH_SCHEDULE,
            "concurrency": concurrency, "microbatches_requested": microbatches,
            "fpga_cards": cards, "output_tokens_per_request": output_tokens, "status": "queued"}
            for concurrency, microbatches, cards in settings]}
    reused = {}
    if reuse_completed_from is not None:
        source = Path(reuse_completed_from)
        if source.resolve() == directory.resolve():
            raise ValueError("completed-result reuse requires a distinct immutable source directory")
        previous = json.loads((source / "result.json").read_text())
        if previous["scope"] != "request_sweep" or previous["method"] != "roofline":
            raise ValueError("completed-result source has a different evaluation contract")
        if _schedule_identity(previous) != MICROBATCH_SCHEDULE:
            raise ValueError("completed-result reuse requires identical current microbatch schedule")
        previous_gpu = previous["gpu_utilization"] if "gpu_utilization" in previous else 1.0
        previous_fpga = previous["fpga_utilization"] if "fpga_utilization" in previous else 1.0
        if (previous_gpu, previous_fpga) != (gpu_utilization, fpga_utilization):
            raise ValueError("completed-result reuse requires identical utilization scenarios")
        if ("gpu_hardware_config_sha256" in previous
                and previous["gpu_hardware_config_sha256"] != gpu_config_sha256):
            raise ValueError("completed-result reuse requires identical GPU hardware configuration")
        if "gpu_hardware_id" in previous and previous["gpu_hardware_id"] != gpu_ids[0]:
            raise ValueError("completed-result reuse requires identical GPU hardware identity")
        if "gpu_hardware_id" not in previous and gpu_ids[0] != "hygon_bw1100":
            raise ValueError("legacy BW1100 execution cannot be reused for another GPU identity")
        for case in previous["cases"]:
            if case["status"] != "completed":
                continue
            key = (case["concurrency"], case["microbatches_requested"], case["fpga_cards"])
            if key not in settings:
                continue
            if _schedule_identity(case) != MICROBATCH_SCHEDULE:
                raise ValueError("completed-result reuse requires identical current microbatch schedule for every case")
            if "gpu_hardware_config_sha256" not in previous:
                # The historical runner selected only this canonical BW1100
                # file. Recover its bytes from the original execution commit,
                # rather than treating an absent identity as the current GPU.
                original_snapshot = previous["snapshot"]
                if ("uncommitted_changes" not in original_snapshot
                        or any("configs/hardware/hygon_bw1100.yaml" in row
                            for row in original_snapshot["uncommitted_changes"])):
                    raise ValueError("legacy reuse requires a clean original BW1100 configuration snapshot")
                original_config = subprocess.check_output(["git", "show",
                    original_snapshot["commit"] + ":configs/hardware/hygon_bw1100.yaml"])
                if hashlib.sha256(original_config).hexdigest() != gpu_config_sha256:
                    raise ValueError("legacy reuse requires identical original GPU configuration bytes")
            if ("gpu_hardware_id" in case and case["gpu_hardware_id"] != gpu_ids[0]
                    or "gpu_hardware_config_sha256" in case
                    and case["gpu_hardware_config_sha256"] != gpu_config_sha256):
                raise ValueError("completed-result case has a different GPU configuration identity")
            if (case["prompt_tokens"] != prompt_tokens or case["output_tokens_per_request"] != output_tokens
                    or case["seed"] != seed or case["gpu_vector_memory_only"] is not gpu_vector_memory_only
                    or case["generated_tokens"] != case["concurrency"] * output_tokens
                    or case["completed_requests"] != case["concurrency"]
                    or case["unpriced_calls"] != 0 or case["priced_calls"] != case["kernel_calls"]
                    or not case["all_graph_nodes_lowered"] or len(case["decode_rounds"]) != output_tokens - 1):
                raise ValueError("completed-result reuse requires identical settings and complete priced execution")
            execution_snapshot = case["execution_snapshot"] if "execution_snapshot" in case else previous["snapshot"]
            reuse_chain = (*case["reuse_chain"], str(source)) if "reuse_chain" in case else (
                (case["reused_from"], str(source)) if "reused_from" in case else (str(source),))
            reused[key] = {**case, "reused_from": str(source), "execution_snapshot": execution_snapshot,
                "reuse_chain": reuse_chain,
                "reuse_basis": "explicit reuse with identical schedule and settings; original execution identity retained"}
        result["cases"] = [reused[(case["concurrency"], case["microbatches_requested"], case["fpga_cards"])]
            if (case["concurrency"], case["microbatches_requested"], case["fpga_cards"]) in reused else case
            for case in result["cases"]]
    def publish():
        _write_json(directory / "result.json", result)
        (directory / "report.md").write_text(render_request_sweep(result))
    arguments = tuple((*setting, prompt_tokens, output_tokens, seed, gpu_vector_memory_only, gpu_utilization, fpga_utilization, gpu_hardware_config,
        str(directory / f"fpga{setting[2]}_c{setting[0]}_mb{setting[1]}")) for setting in settings if setting not in reused)
    publish()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = tuple(executor.submit(_case, args) for args in arguments)
        for future in as_completed(futures):
            completed = future.result()
            key = tuple(completed[field] for field in ("concurrency", "microbatches_requested", "fpga_cards"))
            result["cases"] = [completed if tuple(case[field] for field in
                ("concurrency", "microbatches_requested", "fpga_cards")) == key else case
                for case in result["cases"]]
            result["cases"].sort(key=lambda c: (c["fpga_cards"], c["concurrency"], c["microbatches_requested"]))
            publish()
    result["status"] = "completed" if all(c["status"] == "completed" for c in result["cases"]) else "failed"
    publish()
    return result


def _completed_request_receipt(case, snapshot, model):
    """Check each completed request independently of its comparison partner."""
    outputs, prompt = case["output_tokens_per_request"], case["prompt_tokens"]
    rounds = case["decode_rounds"]
    if (case["status"] != "completed" or type(outputs) is not int or outputs < 1
            or type(prompt) is not int or prompt < 1
            or case["layer_count"] != model.num_hidden_layers
            or case["model_config_sha256"] != model.config_sha256
            or case["completed_requests"] != case["concurrency"]
            or case["generated_tokens"] != case["concurrency"] * outputs
            or case["unpriced_calls"] != 0 or case["priced_calls"] != case["kernel_calls"]
            or case["all_graph_nodes_lowered"] is not True
            or case["GPU_used"] is not False or case["normal_capacity_accepted"] is not False
            or case["native_provider_accepted"] is not False
            or len(rounds) != outputs - 1):
        raise ValueError("completed request requires the complete declared source and all-priced output")
    if (tuple(row["round"] for row in rounds) != tuple(range(1, outputs))
            or tuple(row["kv_length"] for row in rounds) != tuple(range(prompt + 1, prompt + outputs))):
        raise ValueError("completed request decode rounds and KV lineage must be contiguous")
    stages = (*case["prefill_groups"], *rounds)
    if (not case["prefill_groups"]
            or any(not math.isfinite(row["duration_s"]) or row["duration_s"] <= 0
                or row["kernel_calls"] <= 0 or row["workload_nodes"] <= 0 for row in stages)
            or sum(row["kernel_calls"] for row in stages) != case["kernel_calls"]
            or sum(row["workload_nodes"] for row in stages) != case["workload_nodes"]
            or not math.isclose(sum(row["duration_s"] for row in stages),
                case["full_batch_latency_s"], rel_tol=1e-12, abs_tol=0.0)
            or not math.isclose(sum(row["duration_s"] for row in case["prefill_groups"]),
                case["prefill_and_pd_prepare_s"], rel_tol=1e-12, abs_tol=0.0)):
        raise ValueError("completed request stages must conserve calls, nodes and full duration")
    return {"microbatch_schedule": _schedule_identity(case),
        "fpga_cards": case["fpga_cards"], "concurrency": case["concurrency"],
        "microbatches": case["microbatches_requested"], "generated_tokens": case["generated_tokens"],
        "priced_calls": case["priced_calls"], "decode_rounds": len(rounds),
        "kv_length_first": rounds[0]["kv_length"] if rounds else None,
        "kv_length_last": rounds[-1]["kv_length"] if rounds else None,
        "execution_snapshot": case["execution_snapshot"] if "execution_snapshot" in case else snapshot,
        "status": "passed_complete_request_invariants"}


def _write_comparison_plot(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concurrencies = sorted({row["concurrency"] for row in result["cases"]})
    xs = tuple(range(len(concurrencies)))
    cards = sorted({row["fpga_cards"] for row in result["cases"]})
    microbatches = sorted({row["microbatches_requested"] for row in result["cases"]})
    figure, axes = plt.subplots(2, len(cards), figsize=(6 * len(cards), 8), sharex=True, squeeze=False)
    markers = {1: "o", 2: "s", 3: "^"}
    styles = {1: "-", 2: "--", 3: ":"}
    for column, card in enumerate(cards):
        selected = { (row["concurrency"], row["microbatches_requested"]): row
            for row in result["cases"] if row["fpga_cards"] == card}
        for label, color, run in zip(("baseline", "scenario"), ("#2068b1", "#df7d23"), result["runs"]):
            title = f'{run["gpu_hardware_id"]}, GPU {run["gpu_utilization"]:.0%} / FPGA {run["fpga_utilization"]:.0%}'
            for mb in microbatches:
                for index, metric in enumerate(("output_tokens_per_second", "full_batch_latency_s")):
                    values = [selected[(c, mb)][label][metric]
                        if (c, mb) in selected and selected[(c, mb)][label] is not None
                        else float("nan") for c in concurrencies]
                    axes[index, column].plot(xs, values, color=color, linestyle=styles[mb],
                        marker=markers[mb], markersize=6, label=f"{title}, MB{mb}")
        axes[0, column].set_title(f"{card} FPGA cards")
        for axis in axes[:, column]:
            axis.set_xticks(xs, concurrencies)
            axis.grid(alpha=0.25)
            axis.set_ylim(bottom=0)
        axes[1, column].set_xlabel("Concurrent requests")
    axes[0, 0].set_ylabel("Total output throughput (token/s)")
    axes[1, 0].set_ylabel("Full-batch completion time (s)")
    lengths = ", ".join(f"{prompt} input / {outputs} output" for prompt, outputs in result["request_lengths"])
    figure.suptitle(f"DSV4 Flash | {lengths} | Roofline | MTP off"
        + ("" if result["fpga_only_microbatch_comparison"] else "\nLEGACY / DIFFERENT SCHEDULE: not FPGA-only microbatch comparison"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    figure.text(0.5, 0.005, "Missing points are unfinished cases; no interpolation or zero filling.",
        ha="center", fontsize=9)
    if any(run["gpu_hardware_id"] == "nvidia_rtx_pro_5500" for run in result["runs"]):
        figure.text(0.5, 0.025, "RTX PRO5500 memory/bandwidth + user-specified PRO5000 dense compute proxy; not official PRO5500 compute peaks.",
            ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"comparison.{suffix}", dpi=160)
    plt.close(figure)


def compare_request_sweeps(baseline_dir, scenario_dir, output_dir):
    """Publish paired results without changing either run's execution identity."""
    directories = (Path(baseline_dir), Path(scenario_dir))
    runs = tuple(json.loads((directory / "result.json").read_text()) for directory in directories)
    for run in runs:
        if run["scope"] != "request_sweep" or run["method"] != "roofline":
            raise ValueError("comparison requires two request Roofline sweeps")
    model = load_model_config("configs/model/deepseek-v4-flash.json")
    validated = [{"mode": directory.name, **_completed_request_receipt(case, run["snapshot"], model)}
        for directory, run in zip(directories, runs) for case in run["cases"]
        if case["status"] == "completed"]
    keys = ("fpga_cards", "concurrency", "microbatches_requested")
    indexed = tuple({tuple(case[key] for key in keys): case for case in run["cases"]} for run in runs)
    if (not indexed[0] or set(indexed[0]) != set(indexed[1])
            or any(len(table) != len(run["cases"]) for table, run in zip(indexed, runs))):
        raise ValueError("comparison requires identical unique request settings")
    cases = []
    for key in sorted(indexed[0]):
        pair = tuple(table[key] for table in indexed)
        row = dict(zip(keys, key))
        row["microbatch_schedules"] = tuple(_schedule_identity(case) for case in pair)
        row["matching_declared_schedule"] = (row["microbatch_schedules"][0] is not None
            and row["microbatch_schedules"][0] == row["microbatch_schedules"][1]
            and all(_schedule_identity(run) == schedule
                for run, schedule in zip(runs, row["microbatch_schedules"])))
        row["fpga_only_microbatch_comparison"] = (row["matching_declared_schedule"]
            and row["microbatch_schedules"][0] == MICROBATCH_SCHEDULE)
        row["statuses"] = tuple(case["status"] for case in pair)
        row["paired_complete"] = all(case["status"] == "completed" for case in pair)
        for label, case in zip(("baseline", "scenario"), pair):
            row[label] = ({field: case[field] for field in (
                "full_batch_latency_s", "prefill_and_pd_prepare_s", "qps", "output_tokens_per_second",
                "kernel_calls", "output_tokens_per_request", "normal_capacity_accepted", "native_provider_accepted")}
                if case["status"] == "completed" else None)
        row["latency_ratio"] = None
        if row["paired_complete"]:
            fields = ("prompt_tokens", "output_tokens_per_request", "seed", "gpu_vector_memory_only",
                "model_config_sha256", "layer_count")
            if row["matching_declared_schedule"]:
                fields += ("kernel_calls", "workload_nodes", "handoff_bytes")
            for field in fields:
                if pair[0][field] != pair[1][field]:
                    raise ValueError(f"comparison settings/work differ: {key}: {field}")
            for case in pair:
                if (case["unpriced_calls"] != 0 or case["priced_calls"] != case["kernel_calls"]
                        or not case["all_graph_nodes_lowered"]
                        or case["completed_requests"] != case["concurrency"]
                        or case["generated_tokens"] != case["concurrency"] * case["output_tokens_per_request"]
                        or len(case["decode_rounds"]) != case["output_tokens_per_request"] - 1):
                    raise ValueError("comparison requires complete all-priced requests")
            if row["matching_declared_schedule"]:
                row["latency_ratio"] = pair[1]["full_batch_latency_s"] / pair[0]["full_batch_latency_s"]
        cases.append(row)
    result = {"status": "completed" if all(run["status"] == "completed" for run in runs)
            and all(row["paired_complete"] for row in cases) else "incomplete",
        "microbatch_schedule": (MICROBATCH_SCHEDULE if all(_current_sweep_schedule(run) for run in runs) else None),
        "fpga_only_microbatch_comparison": all(_current_sweep_schedule(run) for run in runs),
        "runs": [{"directory": str(directory.resolve()), "snapshot": run["snapshot"],
            "microbatch_schedule": _schedule_identity(run),
            "gpu_hardware_id": run["gpu_hardware_id"] if "gpu_hardware_id" in run else "hygon_bw1100",
            "gpu_identity_source": "declared" if "gpu_hardware_id" in run else "historical_BW1100_only_runner",
            "gpu_hardware_config_sha256": run["gpu_hardware_config_sha256"] if "gpu_hardware_config_sha256" in run else None,
            "gpu_utilization": run["gpu_utilization"] if "gpu_utilization" in run else 1.0,
            "fpga_utilization": run["fpga_utilization"] if "fpga_utilization" in run else 1.0,
            "status": run["status"], "completed": sum(case["status"] == "completed" for case in run["cases"])}
            for directory, run in zip(directories, runs)],
        "cases": cases, "GPU_used": False, "method": "roofline"}
    completed_baseline = tuple(case for case in runs[0]["cases"] if case["status"] == "completed")
    lengths = sorted({(case["prompt_tokens"], case["output_tokens_per_request"]) for case in completed_baseline})
    result["request_lengths"] = lengths
    length_description = "、".join(f"{prompt}输入/{outputs}输出" for prompt, outputs in lengths) if lengths else "长度见原始运行配置"
    requests_per_mode = sum(case["concurrency"] for case in runs[0]["cases"])
    schedule_notice = ("固定Prefill及GPU attention/router/shared整批；MB只切分FPGA routed expert收、算、发。"
        if result["fpga_only_microbatch_comparison"] else
        "**已被替代的旧调度／不同调度结果：不能作为FPGA-only microbatch对比。原snapshot及schedule身份原样保留；未声明相同schedule的配对不计算时延倍数。**")
    lines = ["# DSV4 Flash GPU＋FPGA：配对参数场景对比", "", schedule_notice, "",
        f'状态：{result["status"]}；两组完成数：' + " / ".join(f'{run["completed"]}/{len(cases)}' for run in result["runs"]) + "。", "",
        f"同一43层Flash；{length_description}，Prefill TP8 GPU、Decode attention DP8 GPU，Decode routed experts在FPGA；MTP关闭。", "",
        "| 模式 | GPU硬件身份 | GPU算力/本地显存带宽 | FPGA算力/HBM/DDR |", "|---|---|---:|---:|"]
    for label, run in zip(("基线", "配对场景"), result["runs"]):
        lines.append(f'| {label} | {run["gpu_hardware_id"]} | {run["gpu_utilization"]:.0%} | {run["fpga_utilization"]:.0%} |')
    if any(run["gpu_hardware_id"] == "nvidia_rtx_pro_5500" for run in result["runs"]):
        lines.extend(("", "RTX PRO5500显存/带宽 + 用户指定PRO5000 dense算力proxy；不是RTX PRO5500官方算力峰值或原生provider预测。"))
    lines.extend(("", f"每组请求数等于并发数；每种模式共{requests_per_mode}个完整请求，两种共{2 * requests_per_mode}个。空白表示该项尚未完成，已完成的单侧结果照常列出。", "",
        "![吞吐与整批时延](comparison.png)", "", "[下载CSV表](comparison.csv) · [下载SVG图](comparison.svg)", "",
        "GPU规格及GPU/FPGA有效速率按上表各自配置。模型、精度、路由种子、调度、FPGA设计、RoCE带宽/延时保持配对；从算子服务重新计算资源竞争和收算发重叠。", "",
        "| FPGA卡 | 并发 | MB | 基线时延(s) | 利用率场景时延(s) | 时延倍数 | 基线输出token/s | 场景输出token/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for row in cases:
        values = " | ".join((
            f'{row["baseline"]["full_batch_latency_s"]:.3f}' if row["baseline"] is not None else "",
            f'{row["scenario"]["full_batch_latency_s"]:.3f}' if row["scenario"] is not None else "",
            f'{row["latency_ratio"]:.4f}' if row["latency_ratio"] is not None else "",
            f'{row["baseline"]["output_tokens_per_second"]:.3f}' if row["baseline"] is not None else "",
            f'{row["scenario"]["output_tokens_per_second"]:.3f}' if row["scenario"] is not None else ""))
        lines.append(f'| {row["fpga_cards"]} | {row["concurrency"]} | {row["microbatches_requested"]} | {values} |')
    lines.extend(("", "同步Decode轮次及完整arena PD复制的解析Roofline场景；GPU向量采用用户批准的仅访存近似。未知软件、控制器、协议启动未计；不是实际GPU/FPGA测量、native provider或normal容量验收。", "",
        schedule_notice + " 原始result.json保留每轮、调用覆盖率、执行snapshot及复用来源。", ""))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "completed_validation.json", {
        "microbatch_schedule": result["microbatch_schedule"],
        "fpga_only_microbatch_comparison": result["fpga_only_microbatch_comparison"],
        "validated_cases": len(validated), "total_requested_cases": sum(len(run["cases"]) for run in runs),
        "completed": result["status"] == "completed", "GPU_used": False,
        "normal_capacity_accepted": False, "native_provider_accepted": False, "cases": validated})
    _write_json(directory / "result.json", result)
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("fpga_cards", "concurrency", "microbatches", "baseline_latency_s", "scenario_latency_s",
            "latency_ratio", "baseline_output_token_s", "scenario_output_token_s"))
        for row in cases:
            writer.writerow((row["fpga_cards"], row["concurrency"], row["microbatches_requested"],
                row["baseline"]["full_batch_latency_s"] if row["baseline"] is not None else "",
                row["scenario"]["full_batch_latency_s"] if row["scenario"] is not None else "",
                row["latency_ratio"] if row["latency_ratio"] is not None else "",
                row["baseline"]["output_tokens_per_second"] if row["baseline"] is not None else "",
                row["scenario"]["output_tokens_per_second"] if row["scenario"] is not None else ""))
    _write_comparison_plot(result, directory)
    (directory / "report.md").write_text("\n".join(lines))
    return result
