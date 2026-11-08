"""Evaluation of complete, consecutive pure-GPU Flash decode source steps."""
import json
import gzip
import hashlib
import subprocess
from collections import Counter
from dataclasses import asdict
import os
from pathlib import Path
from time import perf_counter
import yaml

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind
from src.kernel.predictors.fused_moe import dsv4_marlin_moe_source_work
from src.kernel.predictors.sparse_attn import dsv4_attention_mathematical_work
from src.kernel.predictors.sparse_attn_indexer import dsv4_indexer_logical_work
from src.kernel.predictors.gemm import _unquantized_matmul_math
from src.serving.kv_cache.model_residency import flash_gpu_decode_capacity
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.kernel_lowering.execution_timeline import build_execution_timeline, ExecutionResourceCalendar
from src.serving.workload.model_config import load_model_config
from src.serving.workload.traced_flash_gpu_decode import FlashGPUDecode


def evaluate_case(*, scheme, concurrency, steps, seed, output_dir):
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError("decode window must have 1..64 actual steps")
    started = perf_counter()
    model = load_model_config("configs/model/deepseek-v4-flash.json")
    config = yaml.safe_load(Path("configs/hardware/hygon_bw1100.yaml").read_text())
    config["features"]["gpu_vector_memory_only"] = True
    hw = HWSpec(config)
    hardware = {hw.hardware_id: hw}
    source = FlashGPUDecode(model, concurrency=concurrency, scheme=scheme, max_steps=steps, root_seed=seed)
    records, elapsed = [], 0.0
    peak_workspace = peak_persistent = peak_arena = 0
    source_files = ("src/serving/workload/traced_flash_gpu_decode.py",
        "src/serving/workload/traced_models.py", "src/serving/workload/traced_operators.py",
        "src/serving/kernel_lowering/heterogeneous_moe.py", "src/serving/kernel_lowering/lower_workload.py",
        "src/serving/kv_cache/model_residency.py", "evaluation/serving/metrics/flash_gpu_decode.py",
        "configs/model/deepseek-v4-flash.json", "configs/hardware/hygon_bw1100.yaml")
    identity = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files}
    with Path(output_dir, "steps.jsonl").open("w") as stream:
        stream.write("")
    for index in range(steps):
        begin = perf_counter()
        graph, state = source.step()
        traced = perf_counter()
        plan = lower_workload(graph, source.deployment, hardware)
        lowered = perf_counter()
        predictions = {c.call_id: predict(c.kernel_spec, hw, "roofline") for c in plan.calls}
        priced = perf_counter()
        timeline = build_execution_timeline(plan, predictions, shared_resource_calendar=ExecutionResourceCalendar())
        scheduled = perf_counter()
        if {n.node_id for n in graph.nodes} != {n for c in plan.calls for n in c.source_node_ids}:
            raise RuntimeError("D-only lowering omitted source work")
        sampler_ids = {c.call_id: c.rank for c in plan.calls if
            c.kernel_spec.variant == "vllm_plain_greedy_sample" and c.kernel_spec.attributes["stage"] == "output_int32"}
        rank_completion = {sampler_ids[i.call_id]: elapsed + i.end_time_s for i in timeline.intervals
            if i.call_id in sampler_ids}
        if set(rank_completion) != set(range(8)):
            raise RuntimeError("D-only must complete every rank's actual sampler")
        elapsed += timeline.duration_s
        capacity_rows = []
        for rank, trace in enumerate(source.traces):
            cache_ids = {t.view.allocation_id for layer in source.caches[rank] for _, t in layer}
            persistent_ids = {t.view.allocation_id for t in (source.topks[rank], *source.ropes[rank].values())}
            arena = sum(trace._allocations[n].size_bytes for n in cache_ids)
            persistent = sum(trace._allocations[n].size_bytes for n in persistent_ids)
            # Sum of all non-cache/non-parameter source requests is a safe
            # known capacity envelope, not measured allocator residency.
            workspace = sum(a.size_bytes for n, a in trace._allocations.items()
                if n not in cache_ids | persistent_ids and (n not in trace._persistent_ids
                    or trace._references[n] > 0))
            capacity_rows.append({"rank": rank, "known_source_workspace_request_sum_bytes": workspace,
                "known_source_persistent_bytes": persistent, "source_virtual_arena_bytes": arena})
        peak_workspace = max(peak_workspace, *(r["known_source_workspace_request_sum_bytes"] for r in capacity_rows))
        peak_persistent = max(peak_persistent, *(r["known_source_persistent_bytes"] for r in capacity_rows))
        peak_arena = max(peak_arena, *(r["source_virtual_arena_bytes"] for r in capacity_rows))
        physics = _physical_ledger(plan, predictions, hw)
        if timeline.duration_s < max(physics["compute_lower_bound_s"],
                physics["major_operand_bandwidth_lower_bound_s"]) - 1e-12:
            raise RuntimeError("D-only timeline violates an owner-derived physical lower bound")
        if index == 0:
            with gzip.open(Path(output_dir, "first_step_calls.json.gz"), "wt") as stream:
                json.dump([{"id": c.call_id, "rank": c.rank, "dependencies": c.dependencies,
                    "kind": c.kernel_spec.kind.value, "variant": c.kernel_spec.variant,
                    "shape": dict(c.kernel_spec.shape), "dtype": c.kernel_spec.dtype,
                    "attributes": dict(c.kernel_spec.attributes),
                    "transfer": None if c.kernel_spec.transfer is None else asdict(c.kernel_spec.transfer),
                    "predicted_s": predictions[c.call_id].t_total_s} for c in plan.calls], stream)
        record = {**state, "step_latency_s": timeline.duration_s, "completion_s": elapsed,
            "nodes": len(graph.nodes), "calls": len(plan.calls), "all_nodes_lowered": True,
            "hardware_ids": sorted({c.kernel_spec.hardware_id for c in plan.calls}),
            "communication_payload_bytes": sum(c.kernel_spec.transfer.bytes for c in plan.calls
                if c.kernel_spec.kind is KernelKind.COMMUNICATION),
            "rank_completion_times_s": rank_completion,
            "roles": dict(Counter(n.kind.value for n in graph.nodes)),
            "variants": dict(Counter(str(c.kernel_spec.variant) for c in plan.calls)),
            "physics": physics, "source_capacity": capacity_rows,
            "cpu_wall_s": {"source": traced - begin, "lower": lowered - traced,
                "predict": priced - lowered, "timeline": scheduled - priced}}
        records.append(record)
        with Path(output_dir, "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({"step": index, "latency_s": timeline.duration_s,
            "wall_s": scheduled - begin, "scheme": scheme, "C": concurrency}), flush=True)
        del graph, plan, timeline, predictions
        source.retire()
    subsequent = sum(r["step_latency_s"] for r in records[1:])
    capacity = flash_gpu_decode_capacity(model, scheme=scheme, concurrency=concurrency,
        context_tokens=131072 + steps, hbm_capacity_bytes=hw.resource_by_role("dram").size_bytes,
        source_persistent_bytes=peak_persistent, source_workspace_bytes=peak_workspace,
        source_logical_arena_bytes=peak_arena)
    capacity["source_workspace_semantics"] = "sum_of_known_source_requests_conservative_envelope_not_native_peak"
    capacity["native_workspace_and_runtime_reserve_remain_unbound"] = True
    return {"status": "analytical_completed", "scheme": scheme, "concurrency": concurrency,
        "microbatch": 1, "initial_context_tokens": 131072, "layers": 43, "seed": seed,
        "actual_steps": len(records), "window_s": elapsed, "capacity": capacity,
        "source_identity_sha256": identity, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "cpu_phase_wall_s": {phase: sum(r["cpu_wall_s"][phase] for r in records)
            for phase in ("source", "lower", "predict", "timeline")},
        "output_tokens_per_s": concurrency * steps / elapsed,
        "first_step_s": records[0]["step_latency_s"],
        "subsequent_mean_tpot_s": subsequent / (steps - 1) if steps > 1 else None,
        "cpu_wall_s": perf_counter() - started, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "normal_capacity_accepted": False, "native_provider_bound": False, "GPU_used": False,
        "steps": records}


def _physical_ledger(plan, predictions, hw):
    """Aggregate owner work ledgers; evaluation introduces no kernel price."""
    fields = ("serial_kernel_service_s", "routed_tensor_flops_bf16", "routed_logical_bytes",
        "routed_active_weight_bytes", "linear_tensor_flops_fp8", "linear_tensor_flops_bf16",
        "indexer_tensor_flops_bf16", "indexer_full_scan_pairs", "indexer_logical_bytes",
        "indexer_cache_read_bytes", "attention_tensor_flops_bf16", "attention_logical_bytes", "selected_attention_pairs")
    rows = [{f: 0.0 for f in fields} for _ in range(8)]
    for call in plan.calls:
        spec, target = call.kernel_spec, rows[call.rank]
        target["serial_kernel_service_s"] += predictions[call.call_id].t_total_s
        if spec.kind is KernelKind.FUSED_MOE:
            work = dsv4_marlin_moe_source_work(spec)
            target["routed_tensor_flops_bf16"] += sum(s["demand"]["compute_tensor"] for s in work["stages"])
            target["routed_logical_bytes"] += sum(sum(s["read_write_bytes"].values()) for s in work["stages"])
            target["routed_active_weight_bytes"] += sum(sum(s["weight_payload_per_active_expert_bytes"].values())
                * work["active_experts"] for s in work["stages"])
        elif spec.kind is KernelKind.GEMM:
            work = _unquantized_matmul_math(spec.shape, spec.dtype, 2, "matrix")
            fp8 = spec.dtype == "fp8" or ("mathematical_program" in spec.attributes
                and spec.attributes["mathematical_program"] in ("block_fp8_fp32_scales_v1",
                    "batched_fp8_quantized_input_fp32_scales_v1"))
            target["linear_tensor_flops_fp8" if fp8 else "linear_tensor_flops_bf16"] += work["mathematical_tensor_flops"]
        elif spec.kind is KernelKind.SPARSE_ATTN_INDEXER:
            work = dsv4_indexer_logical_work(spec)
            target["indexer_tensor_flops_bf16"] += work["demand"]["compute_tensor"]
            target["indexer_full_scan_pairs"] += work["active_pairs"]
            target["indexer_logical_bytes"] += work["logical_bytes"]
            target["indexer_cache_read_bytes"] += sum(v["bytes"] for s in work["stages"] for v in s["operands"]
                if v["role"] in ("key_role_uses", "key_scale_role_uses"))
        elif spec.kind is KernelKind.SPARSE_ATTN:
            work = dsv4_attention_mathematical_work(spec)
            target["attention_tensor_flops_bf16"] += work["demand"]["compute_tensor"]
            target["attention_logical_bytes"] += work["logical_bytes"]
            target["selected_attention_pairs"] += sum(work["pair_capacities"].values())
    return {"per_rank": tuple(dict(r) for r in rows),
        "compute_lower_bound_s": max((r["routed_tensor_flops_bf16"] + r["linear_tensor_flops_bf16"]
            + r["indexer_tensor_flops_bf16"] + r["attention_tensor_flops_bf16"])
            / hw.resource_by_role("compute_tensor").capacity
            + r["linear_tensor_flops_fp8"] / hw.resource_by_role("compute_tensor_fp8").capacity for r in rows),
        "major_operand_bandwidth_lower_bound_s": max((r["routed_logical_bytes"] + r["indexer_logical_bytes"]
            + r["attention_logical_bytes"]) / hw.resource_by_role("dram").capacity for r in rows),
        "scope": "logical_role_uses_not_observed_DRAM; lower_bounds_exclude_other_work_and_startup"}


def decode_window_metrics(result, count):
    """Use recorded completion intervals and the complete chosen denominator."""
    records = result["steps"][:count]
    if len(records) != count or count < 2 or any(r["step_index"] != i for i, r in enumerate(records)):
        raise ValueError("a decode window requires every consecutive executed step")
    if result["scheme"] != "tp8":
        raise NotImplementedError("per-request DP lane metrics await the EP source")
    completion = [max(r["rank_completion_times_s"].values()) for r in records]
    duration = records[-1]["completion_s"]
    if duration <= 0 or any(b <= a for a, b in zip(completion, completion[1:])):
        raise ValueError("decode window completion times must strictly advance")
    return {"actual_steps": count, "output_tokens": result["concurrency"] * count,
        "window_s": duration, "first_step_s": completion[0],
        "subsequent_31_or_63_span_s": completion[-1] - completion[0],
        "subsequent_mean_tpot_s": (completion[-1] - completion[0]) / (count - 1),
        "all_step_mean_s": duration / count,
        "output_tokens_per_s": result["concurrency"] * count / duration,
        "completion_semantics": "all_TP_rank_sampler_completion_per_request; explicit_whole_batch_step_barrier"}


def write_checkpoint_report(output_dir):
    """Keep the original four-point denominator, including unimplemented EP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.serving.kv_cache.model_residency import dsv4_model_buffer_bytes
    from src.serving.parallelism.parallel_config import ParallelConfig

    output = Path(output_dir)
    model = load_model_config("configs/model/deepseek-v4-flash.json")
    cases = []
    for scheme in ("tp8", "dp8_ep8"):
        for concurrency in (128, 192):
            if scheme == "tp8":
                result = json.loads((output / f"{scheme}_c{concurrency}" / "result.json").read_text())
                if (result["concurrency"] != concurrency or result["scheme"] != scheme
                        or result["layers"] != 43 or result["initial_context_tokens"] != 131072
                        or result["microbatch"] != 1 or result["actual_steps"] not in (32, 64)
                        or len(result["steps"]) != result["actual_steps"]
                        or result["status"] != "analytical_completed"
                        or type(result["seed"]) is not int or result["seed"] < 0):
                    raise ValueError("formal matrix input disagrees with the frozen decode workload")
                metrics = decode_window_metrics(result, 32)
                cases.append({"scheme": scheme, "concurrency": concurrency,
                    "status": "analytical_32_step_completed_reserve_unbound", "metrics": metrics, "seed": result["seed"],
                    "capacity": result["capacity"], "actual_case_cpu_wall_s": result["cpu_wall_s"],
                    "first_32_core_cpu_wall_s": sum(sum(r["cpu_wall_s"].values()) for r in result["steps"][:32]),
                    "cpu_phase_wall_s": result["cpu_phase_wall_s"],
                    "physics_first_step": result["steps"][0]["physics"],
                    "source_identity_sha256": result["source_identity_sha256"],
                    "first_step_communication_payload_bytes": result["steps"][0]["communication_payload_bytes"],
                    "trend_64": decode_window_metrics(result, 64) if result["actual_steps"] == 64 else None})
            else:
                # The complete static model buffer inventory is an explicitly
                # named estimate, not the missing EP source's measured scratch.
                buffers = dsv4_model_buffer_bytes(model, ParallelConfig(tensor_parallel_size=1),
                    max_batched_tokens=concurrency // 8)
                capacity = flash_gpu_decode_capacity(model, scheme=scheme, concurrency=concurrency,
                    context_tokens=131104, hbm_capacity_bytes=144_000_000_000,
                    source_persistent_bytes=sum(buffers.values()), source_workspace_bytes=None)
                capacity["persistent_scope"] = "existing_full_checkpoint_model_buffer_inventory_estimate; EP_source_not_executed"
                cases.append({"scheme": scheme, "concurrency": concurrency,
                    "status": "not_run_requires_shared_core_patch_freeze", "metrics": None,
                    "capacity": capacity, "actual_case_cpu_wall_s": None})
    cohort_seed = _formal_cohort_seed(cases[:2])
    summary = {"status": "checkpoint", "formal_matrix_denominator": 4, "priced_formal_cases": 2,
        "seed": cohort_seed, "initial_context_tokens": 131072, "microbatch": 1, "layers": 43,
        "case_processes_concurrent": 2, "GPU_used": False, "cases": cases,
        "gates": {"D1": "partial_TP8_only", "D2": "partial_TP8_only_EP_core_scope_blocked",
            "D3": "partial_2_of_4_timing; all_four_static_capacity_inventories",
            "D4": "partial_TP8_physics_and_C128_32_to_64", "D5": "checkpoint_not_final_acceptance"}}
    (output / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = [f'{c["scheme"]}\nC{c["concurrency"]}' for c in cases]
    for ax, key, scale, ylabel in ((axes[0], "output_tokens_per_s", 1, "Output token/s"),
            (axes[1], "subsequent_mean_tpot_s", 1000, "Per-request TPOT (ms)")):
        for index, case in enumerate(cases):
            if case["metrics"] is not None:
                value = case["metrics"][key] * scale
                ax.bar(index, value, color="#3465a4")
                ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(index, 0.05, "not run", transform=ax.get_xaxis_transform(),
                    ha="center", rotation=90, fontsize=9)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("BW1100 x8 / Flash / 128K / MB1 / 32 executed steps — checkpoint")
    fig.tight_layout()
    fig.savefig(output / "throughput_tpot.png", dpi=150)
    plt.close(fig)
    lines = ["# BW1100 Flash D-only checkpoint", "",
        "正式分母仍为四点；TP8两个点已实际连续执行32步，C128继续执行到64步。EP8未运行，吞吐/TPOT为null，不以局部专家价替代。D1–D5尚未全部通过，Goal未完成。", "",
        "同一原始DeepSeek-V4-Flash、43层、初始已prefill的131072-token KV声明输入、首个D输入token、q_len=1、MB1。8张BW1100，无P/PD/FPGA/MTP；CPU虚拟Tensor解析场景，未验证BW原生provider、数值模型或实卡性能。", "",
        "| 方案 | 总并发 | 实际正式步数 | 实例输出token/s | 首步ms | 后续31步TPOT ms | 窗口s | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for case in cases:
        m = case["metrics"]
        if m is None:
            lines.append(f'| {case["scheme"]} | {case["concurrency"]} | 0 | null | null | null | null | 共享core待冻结 |')
        else:
            lines.append(f'| {case["scheme"]} | {case["concurrency"]} | 32 | {m["output_tokens_per_s"]:.3f} | {m["first_step_s"]*1000:.3f} | {m["subsequent_mean_tpot_s"]*1000:.3f} | {m["window_s"]:.6f} | 解析完成/预留未绑定 |')
    lines.extend(("", "![吞吐与TPOT](throughput_tpot.png)", "",
        "吞吐=C×实际步数/完整D窗口。TPOT由同一请求连续的实际sampler完成时间差取得；TP8使用所有rank sampler完成的显式整批barrier。step latency未除以C。首步与后续31步固定分开，不选最快step。", "",
        "| 方案/C | 权重GB/卡 | resident KV页GB/卡 | 已知persistent GB | source工作区GB | 已知占用GB/卡 | 144GB余量GB |",
        "|---|---:|---:|---:|---:|---:|---:|"))
    for case in cases:
        r = case["capacity"]["ranks"][0]
        workspace = "unbound" if r["source_workspace_bytes"] is None else f'{r["source_workspace_bytes"]/1e9:.4f}'
        lines.append(f'| {case["scheme"]}/{case["concurrency"]} | {r["weight_payload_bytes"]/1e9:.4f} | {r["cache_page_budget_bytes"]/1e9:.4f} | {r["known_persistent_bytes"]/1e9:.4f} | {workspace} | {r["known_required_lower_bound_bytes"]/1e9:.4f} | {r["known_inventory_headroom_bytes"]/1e9:.4f} |')
    lines.extend(("", "TP8 source工作区列为全部已知非cache/非参数source请求的保守总和包络，不是native必要峰值或测得allocator占用；其加总口径单列在JSON。EP8只给静态loader/生命周期页与既有完整checkpoint buffer估算，source工作区未知。所有点的native workspace、软件/allocator reserve均unbound，normal容量未验收。TP8逻辑virtual arena分别约5.47/8.21TB/卡，是按全context描述SWA/partial-state的source地址空间，未等同resident页或实际分配，未据此盲拒绝请求。完整参数与7类KV分组见matrix.json和各result.json。", "",
        "| TP8/C | routed BF16 TFLOP/卡 | linear FP8 TFLOP/卡 | linear BF16 TFLOP/卡 | indexer全扫描GB/卡 | 活跃专家权重GB/卡 | 主要字节下界ms | 计算下界ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"))
    for case in cases[:2]:
        p = case["physics_first_step"]
        r = p["per_rank"][0]
        lines.append(f'| {case["concurrency"]} | {r["routed_tensor_flops_bf16"]/1e12:.4f} | {r["linear_tensor_flops_fp8"]/1e12:.4f} | {r["linear_tensor_flops_bf16"]/1e12:.4f} | {r["indexer_cache_read_bytes"]/1e9:.4f} | {r["routed_active_weight_bytes"]/1e9:.4f} | {p["major_operand_bandwidth_lower_bound_s"]*1000:.4f} | {p["compute_lower_bound_s"]*1000:.4f} |')
    lines.extend(("", "账本消费既有kernel owner，按354 BF16/708 FP8 TF/s和2.4TB/s检查；W4A16使用BF16算力。C4 indexer扫描所有已完成压缩KV（首步每请求32768行），不是只扫描最终512个被选中项。Sparse attention沿原source padded64-head几何，并保留SWA/C4/C128、mHC、shared/routed专家、量化和target链；完整shape/依赖保存在first_step_calls.json.gz，所有步骤动态histogram、KV、物理账本保存在steps.jsonl。", "",
        "TP8采用声明ring 0/2/4/6/1/3/5/7，每两卡共享一条每方向50GB/s NIC；timeline预留4条NIC各自TX/RX资源，未把交换机串成单链。只计该合法网络路径的带宽component；BW配置的传播与collective软件startup是null，未挪用FPGA的20us。未知启动/协议、compiled/cache/launch/SM、native workspace和服务控制面未绑定。向量/scalar/SFU只计逻辑访存，矩阵计算保留。网络HBM事务和实际collective scratch仍未绑定，不能称为完整provider服务。", ""))
    trend = cases[0]["trend_64"]
    first = cases[0]["metrics"]
    lines.extend((f'C128 32→64真实窗口：{first["output_tokens_per_s"]:.3f}→{trend["output_tokens_per_s"]:.3f} token/s，后续31/63间隔TPOT {first["subsequent_mean_tpot_s"]*1000:.4f}→{trend["subsequent_mean_tpot_s"]*1000:.4f} ms。变化来自逐步route和压缩边界/KV推进。C192的工作量增大、权重复用改善，吞吐高于C128而每请求TPOT上升；未据趋势调参数。', "",
        "| TP8/C | 实际case CPU wall s | 正式前32步source/lower/predict/timeline合计s | source s | lower s | predict s | timeline s |",
        "|---|---:|---:|---:|---:|---:|---:|"))
    for case in cases[:2]:
        wall = case["cpu_phase_wall_s"]
        lines.append(f'| {case["concurrency"]} | {case["actual_case_cpu_wall_s"]:.3f} | {case["first_32_core_cpu_wall_s"]:.3f} | {wall["source"]:.3f} | {wall["lower"]:.3f} | {wall["predict"]:.3f} | {wall["timeline"]:.3f} |')
    lines.extend(("", "C128 case实际含64步，C192含32步；两case起始并行进程2，C192结束后C128继续。未更改CPU affinity/OMP/MKL；完整实际affinity在result.json。阶段wall不含额外证据序列化/整理；case wall含这些开销，绘图/汇总另算。wall不进入预测时延，未作一次墙钟回归结论。", "",
        "EP8阻塞：现有共享KernelSpec仅放行Flash TP8/EP-off/I256；plain MXFP4 MM也只允许FPGA。现有variable collective只绑定direct-pairwise/non-RoCE，不能伪用InfiniBand或FPGA。需主agent先冻结proposed_ep8_core.patch及Kernel验证scope，再接纯GPU EP8 source与合法all_gatherv/reduce_scatterv共享NIC路线；当前没有应用共享patch、没有删除或重定义任何gate。详见ep8_core_review.md。", "",
        "重跑（仅CPU，解释器为既有Conda环境）：", "", "```bash",
        "python scripts/evaluate_bw1100_decode.py --scheme tp8 --concurrency 128 --steps 64 --output-dir experiment_logs/bw1100_decode_only_20261104/tp8_c128",
        "python scripts/evaluate_bw1100_decode.py --scheme tp8 --concurrency 192 --steps 32 --output-dir experiment_logs/bw1100_decode_only_20261104/tp8_c192",
        "python scripts/evaluate_bw1100_decode.py --summarize --output-dir experiment_logs/bw1100_decode_only_20261104",
        "python scripts/run_tests.py -q -n 0 tests/serving/test_flash_gpu_decode.py tests/serving/test_flash_gpu_decode_capacity.py",
        "```", "", "测试与普通Serving验证状态见validation.json；最终精确commit全维护测试与EP8性能/趋势仍须完成。未改HWSpec/predictor/launch公式或canonical，无raw采集，无GPU/Modal，无新增docs，无push。"))
    (output / "serving_report.md").write_text("\n".join(lines) + "\n")
    return summary


def _formal_cohort_seed(cases):
    """A report may combine C128/C192 only from one seed and source cohort."""
    if len(cases) != 2:
        raise ValueError("formal TP8 cohort requires both concurrency cases")
    seeds = tuple(case["seed"] for case in cases)
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 1:
        raise ValueError("formal cases must share the same explicit nonnegative seed")
    if cases[0]["source_identity_sha256"] != cases[1]["source_identity_sha256"]:
        raise ValueError("formal cases must share the same source identity")
    return seeds[0]


def inspect_unpriced_case(*, concurrency, steps, seed, output_dir):
    """Execute the whole EP source while retaining the explicit core blocker.

    This is not a partial-price fallback. Every non-expert source node is
    lowered through its existing owner; the unchanged expert KernelSpec must
    reject the proposed EP8 geometry. No full latency or throughput is formed.
    """
    from src.serving.kernel_lowering.heterogeneous_moe import (
        heterogeneous_moe_specs, validate_heterogeneous_graph)
    from src.serving.workload.workload_graph import WorkloadKind
    if concurrency not in (128, 192) or type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError("EP source inspection preserves formal concurrency and at most64 actual steps")
    started = perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("configs/hardware/hygon_bw1100.yaml").read_text())
    config["features"]["gpu_vector_memory_only"] = True
    hw, model = HWSpec(config), load_model_config("configs/model/deepseek-v4-flash.json")
    source = FlashGPUDecode(model, concurrency=concurrency, scheme="dp8_ep8", max_steps=steps, root_seed=seed)
    source_paths = ("src/serving/workload/traced_flash_gpu_decode.py",
        "src/serving/kernel_lowering/heterogeneous_moe.py", "src/serving/workload/traced_models.py",
        "src/serving/workload/traced_operators.py", "src/kernel/kernel_spec.py",
        "src/kernel/predictors/fused_moe.py", "configs/hardware/hygon_bw1100.yaml",
        "configs/model/deepseek-v4-flash.json", "evaluation/serving/metrics/flash_gpu_decode.py",
        "scripts/evaluate_bw1100_decode.py", "src/kernel/predictors/communication.py", "src/kernel/predictors/vector.py")
    source_identity = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}
    if source.pool.hardware_id != hw.hardware_id:
        raise ValueError("EP inspection hardware must match the actual deployment pool")
    records = []
    with (output / "source_steps.jsonl").open("w") as stream:
        for index in range(steps):
            begin = perf_counter()
            graph, state = source.step()
            traced = perf_counter()
            validate_heterogeneous_graph(graph, source.deployment)
            expert_nodes = tuple(n for n in graph.nodes if n.kind is WorkloadKind.FUSED_MOE)
            if len(expert_nodes) != model.num_hidden_layers * 8:
                raise RuntimeError("full EP source omitted an expert layer or rank")
            failures, priced, bytes_by_stage = [], Counter(), Counter()
            nic_bytes, nic_service = Counter(), Counter()
            nic_hbm_by_rank = [0] * 8
            nonexpert_nodes = 0
            for node in graph.nodes:
                if node.kind is WorkloadKind.FUSED_MOE:
                    try:
                        heterogeneous_moe_specs(node, hw.hardware_id)
                    except ValueError as exc:
                        if str(exc) != "mathematical experts require their exact unpadded source program":
                            raise
                        failures.append({"node_id": node.node_id, "rank": node.rank, "reason": str(exc)})
                    else:
                        raise RuntimeError("EP core scope changed: use complete evaluation, not the unpriced inspector")
                else:
                    specs = heterogeneous_moe_specs(node, hw.hardware_id)
                    nonexpert_nodes += 1
                    for spec in specs:
                        prediction = predict(spec, hw, "roofline")
                        priced[spec.kind.value] += 1
                        if node.kind is WorkloadKind.P2P:
                            a = node.attributes
                            bytes_by_stage[a["collective_operation"] + ":" + a["collective_stage"]] += a["bytes"]
                            nic_bytes[a["participant_nic_ids"][0] + ":tx"] += a["bytes"]
                            nic_bytes[a["participant_nic_ids"][1] + ":rx"] += a["bytes"]
                            nic_service[a["participant_nic_ids"][0] + ":tx"] += prediction.t_total_s
                            nic_service[a["participant_nic_ids"][1] + ":rx"] += prediction.t_total_s
                            nic_hbm_by_rank[a["source_rank"]] += a["nic_hbm_role_bytes"][0]
                            nic_hbm_by_rank[a["destination_rank"]] += a["nic_hbm_role_bytes"][1]
            finished = perf_counter()
            record = {**state, "layers": model.num_hidden_layers, "nodes": len(graph.nodes),
                "roles": dict(Counter(n.kind.value for n in graph.nodes)),
                "local_output_shapes": tuple(t.shape for t in source.tokens),
                "all_source_nodes_accounted": nonexpert_nodes + len(failures) == len(graph.nodes),
                "priced_nonexpert_calls_by_kind": dict(priced), "unpriced_expert_nodes": len(failures),
                "communication_bytes_by_stage": dict(bytes_by_stage),
                "shared_nic_bytes": dict(nic_bytes), "shared_nic_service_sums_s": dict(nic_service),
                "nic_hbm_unpriced_role_bytes_by_rank": nic_hbm_by_rank,
                "full_step_latency_s": None, "output_tokens_per_s": None, "per_request_tpot_s": None,
                "status": "full_source_executed_core_experts_unpriced",
                "cpu_wall_s": {"source": traced - begin, "validate_lower_predict_nonexpert": finished - traced}}
            if index == 0:
                with gzip.open(output / "first_step_source.json.gz", "wt") as detail:
                    json.dump({"nodes": [{"id": n.node_id, "rank": n.rank, "kind": n.kind.value,
                        "shape": dict(n.shape), "dtype": n.dtype, "variant": n.variant,
                        "attributes": dict(n.attributes)} for n in graph.nodes],
                        "edges": [asdict(e) for e in graph.edges]}, detail)
                (output / "unpriced_experts.json").write_text(json.dumps(failures, indent=2) + "\n")
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps({"scheme": "dp8_ep8", "C": concurrency, "source_step": index,
                "unpriced_experts": len(failures), "priced_nonexpert_calls": sum(priced.values()),
                "wall_s": finished - begin, "full_step_latency_s": None}), flush=True)
            del graph, expert_nodes
            source.retire()
    if {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths} != source_identity:
        raise RuntimeError("EP inspection source changed during execution; snapshot is not complete")
    result = {"status": "checkpoint_full_source_timing_partial", "scheme": "dp8_ep8", "concurrency": concurrency,
        "initial_context_tokens": 131072, "microbatch": 1, "layers": model.num_hidden_layers, "seed": seed,
        "actual_source_steps": len(records), "actual_fully_priced_steps": 0,
        "full_window_s": None, "output_tokens_per_s": None, "per_request_tpot_s": None,
        "cpu_wall_s": perf_counter() - started, "steps": records,
        "hardware_ids": (hw.hardware_id,), "GPU_used": False, "fpga_used": False,
        "source_identity_sha256": source_identity,
        "limits": ("Original shared KernelSpec rejects EP8/I2048; proposal not applied.",
            "Every source node is retained, but complete EP timing/throughput remain null.",
            "Ring wire payload service uses shared50GB/s NICs; software startup and native provider are unbound.",
            "NIC endpoint HBM role bytes are exported separately, not silently priced as zero or as fake copies.")}
    (output / "source_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
