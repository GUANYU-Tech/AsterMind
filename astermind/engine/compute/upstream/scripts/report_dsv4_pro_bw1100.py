#!/usr/bin/env python3
"""Assemble the declared DSV4 decode campaign from completed case receipts."""
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT / "experiment_logs/bw1100_pro5500_fpga/pro"
OLD = REPOSITORY_ROOT / "experiment_logs/bw1100_pro5500_fpga/flash"

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    old_matrix = OLD / "formal/matrix_result.json"
    old_capacity = OLD / "formal/capacity_table.json"
    new_capacity = read(ROOT / "new_capacity_table.json")
    capacities = [{"model": "flash", "bw216": False, "origin": "reused", **c} for c in read(old_capacity)]
    capacities += [{**c, "origin": "new"} for c in new_capacity]
    cases = [{**c, "model_name": "flash", "bw216": False, "origin": "reused"} for c in read(old_matrix)["cases"]]
    groups = (("pro", "hygon_bw1100", False), ("pro", "nvidia_rtx_pro_5500", False),
              ("pro", "hygon_bw1100", True), ("flash", "hygon_bw1100", True))
    for model, hw, bw in groups:
        for scheme in ("tp8_ep8", "dp8_ep8", "dp8_fpga"):
            for mb in ((1,) if scheme == "tp8_ep8" else (1, 2, 3)):
                name = f'{model}__{hw}__{"216" if bw else "original"}__{scheme}__mb{mb}'
                path = ROOT / "jobs" / name / "result.json"
                case = read(path) if path.exists() else {"model_name": model, "hardware_id": hw,
                    "bw216": bw, "scheme": scheme, "microbatches": mb, "status": "pending", "steady_metrics": None}
                cases.append({**case, "origin": "new", "case_dir": str(path.parent)})
    (ROOT / "capacity_table.json").write_text(json.dumps(capacities, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "matrix_result.json").write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n")
    run_manifest = []
    for case in cases:
        if case["origin"] != "new":
            continue
        directory = Path(case["case_dir"])
        exit_file = directory / "exit_code"
        retry_file = directory / "rerun_capacity_exit_code"
        run_manifest.append({"model": case["model_name"], "hardware_id": case["hardware_id"],
            "bw216": case["bw216"], "scheme": case["scheme"], "microbatches": case["microbatches"],
            "status": case["status"], "case_dir": str(directory),
            "exit_code_raw": exit_file.read_text().strip() if exit_file.exists() else None,
            "capacity_retry_exit_code": retry_file.read_text().strip() if retry_file.exists() else None})
    (ROOT / "manifest.json").write_text(json.dumps({"cases": run_manifest,
        "exit_code_note": "xargs quoting wrote trailing literal n; 0n and 1n denote status 0 and 1; two early capacity-only CLI errors were retried after CLI correction"}, indent=2)+"\n")
    rows = []
    for c in cases:
        metric = c["steady_metrics"]
        match = next(x for x in capacities if x["model"] == c["model_name"] and x["hardware_id"] == c["hardware_id"] and x["bw216"] == c["bw216"] and x["scheme"] == c["scheme"] and x["microbatches"] == c["microbatches"])
        cap = match["capacity"]; rank = cap["gpu_ranks"][0]
        rows.append({"model": c["model_name"], "hardware": c["hardware_id"], "bw216": c["bw216"],
            "scheme": c["scheme"], "microbatches": c["microbatches"], "origin": c["origin"], "status": c["status"],
            "tpot_ms": None if metric is None else 1000 * metric["steady_tpot_mean_s"],
            "tokens_per_s": None if metric is None else metric["steady_output_tokens_per_s"],
            "simulation_wall_s": c["cpu_wall_s"] if "cpu_wall_s" in c else None,
            "capacity_status": cap["status"], "weight_bytes_per_gpu": rank["weight_payload_bytes"],
            "kv_bytes_per_gpu": rank["kv_page_budget_bytes"], "persistent_bytes_per_gpu": rank["known_model_persistent_buffer_bytes"],
            "gpu_limit_bytes": cap["hbm_capacity_bytes_per_rank"],
            "max_initial_context_conditional": cap["conditional_static_max_initial_context_tokens"]})
    with (ROOT / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), constrained_layout=True)
    labels = [f'{r["model"]} {"BW216" if r["bw216"] else ("BW" if r["hardware"] == "hygon_bw1100" else "RTX")}\n{r["scheme"]} m{r["microbatches"]}' for r in rows]
    for ax, key, ylabel in zip(axes, ("tpot_ms", "tokens_per_s"), ("稳态 TPOT (ms/token)", "输出吞吐 (tokens/s)"), strict=True):
        for i, row in enumerate(rows):
            if row[key] is not None:
                ax.bar(i, row[key], color="#396eaa" if row["origin"] == "reused" else "#e18e39")
        ax.set_ylabel(ylabel); ax.set_xticks(range(len(rows)), labels, rotation=90, fontsize=6); ax.grid(axis="y", alpha=.2)
    fig.suptitle("DSV4 decode: 蓝色 reused / 橙色 new；缺柱为容量拒绝或待完成")
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(ROOT / "figures" / f"comparison.{ext}", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    xs = list(range(len(capacities)))
    weights = [x["capacity"]["gpu_ranks"][0]["weight_payload_bytes"] / 1e9 for x in capacities]
    kv = [x["capacity"]["gpu_ranks"][0]["kv_page_budget_bytes"] / 1e9 for x in capacities]
    buffers = [x["capacity"]["gpu_ranks"][0]["known_model_persistent_buffer_bytes"] / 1e9 for x in capacities]
    ax.bar(xs, weights, label="权重 GB/GPU")
    ax.bar(xs, kv, bottom=weights, label="KV GB/GPU")
    ax.bar(xs, buffers, bottom=[a+b for a,b in zip(weights,kv,strict=True)], label="已知 buffer GB/GPU")
    ax.scatter(xs, [x["capacity"]["hbm_capacity_bytes_per_rank"] / 1e9 for x in capacities], marker="_", color="red", label="设备上限 GB/GPU")
    labels_cap = [f'{x["model"]} {"BW216" if x["bw216"] else ("BW" if x["hardware_id"] == "hygon_bw1100" else "RTX")} {x["scheme"]} m{x["microbatches"]} ({x["origin"]})' for x in capacities]
    ax.set_xticks(xs, labels_cap, rotation=90, fontsize=6); ax.set_ylabel("已知静态容量 (GB/GPU)"); ax.legend()
    for ext in ("png", "svg"):
        fig.savefig(ROOT / "figures" / f"capacity.{ext}", dpi=160)
    plt.close(fig)
    done = sum(r["status"] == "analytical_completed" for r in rows if r["origin"] == "new")
    rejected = sum(r["status"] == "capacity_rejected" for r in rows if r["origin"] == "new")
    representative = [r for r in rows if r["model"] == "pro" and r["hardware"] == "hygon_bw1100" and not r["bw216"] and r["scheme"] == "dp8_ep8" and r["microbatches"] == 1 and r["tpot_ms"] is not None]
    conclusion = (f'代表点 Pro / 原 BW / DP8-EP8 / 微批 1：TPOT {representative[0]["tpot_ms"]:.2f} ms，吞吐 {representative[0]["tokens_per_s"]:.2f} tokens/s。旧 Flash 同设置为 27.36 ms 和 4678.37 tokens/s；两者模型和源码版本不同。' if representative else "代表点仍在计算。")
    lines = ["# DSV4 Flash / Pro decode 对照", "", conclusion, "", f"新增性能：完成 {done}/28，容量拒绝 {rejected}/28；其余为待完成或失败。原 Flash 14 项性能和 18 项容量复用。",
        "", "[性能图](figures/comparison.svg) · [容量图](figures/capacity.svg) · [原实验总览与第一层单步时间线](" + str(OLD / "visualizations/overview_20261105.html") + ") · [CSV](results.csv)",
        "", "蓝色为原结果，橙色为新仿真；无柱不表示零性能。容量通过仅为静态上限，workspace 和 allocator reserve 未知。",
        "", "100000 初始上下文、32 个实际 decode 步、每微批 128 请求；丢弃前后各 8 步，计算公共稳态区间；FP4 group32 存储，BW GPU FP8 计算，50 μs direct fully-connected collective，无 prefill/PD。",
        "", "旧预测源 f90f227a18bb1aeacd300c05c2ca03c5ac8b4a0e；新基线 46563ce8f5cee590bd16a5b78ace7297c1f3aa4c 加本工作树改动，版本差异使数值比较受限。无真实设备精度验证；BW GPU vector memory-only 近似，FPGA 运行时空间未知。",
        "", "## 性能及容量", "", "|模型|硬件|方案|微批|来源|状态|TPOT ms|tokens/s|仿真 s|权重 GB/GPU|KV GB/GPU|GPU GB|", "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(v): return "—" if v is None else f"{v:.2f}"
    for r in rows:
        hw = "BW216/4TB/s" if r["bw216"] else ("BW144/1.2TB/s" if r["hardware"] == "hygon_bw1100" else "RTX 5500")
        lines.append(f'|{r["model"]}|{hw}|{r["scheme"]}|{r["microbatches"]}|{r["origin"]}|{r["status"]}|{fmt(r["tpot_ms"])}|{fmt(r["tokens_per_s"])}|{fmt(r["simulation_wall_s"])}|{r["weight_bytes_per_gpu"]/1e9:.2f}|{r["kv_bytes_per_gpu"]/1e9:.2f}|{r["gpu_limit_bytes"]/1e9:.2f}|')
    profiles = []
    for profile_path in (ROOT / "profile_pro_bw_dp.json", ROOT / "profile_pro_bw216_fpga.json"):
        if profile_path.exists():
            profile = read(profile_path)
            busiest = max(profile["per_rank_kind_service_s"], key=lambda row: row["busy_s"])
            profiles.append(f'{profile["scheme"]} / {"BW216" if profile["bw216"] else "原 BW"}：首步总时长 {profile["timeline_s"]*1000:.2f} ms；最大单 rank 单类累计服务为 {busiest["pool"]} rank {busiest["rank"]} 的 {busiest["kind"]} {busiest["busy_s"]*1000:.2f} ms。')
    lines += ["", "## 资源占用与瓶颈线索", "", " ".join(profiles) if profiles else "首步资源统计运行中。", "这些是首步单 rank 的累计服务，不是多资源可相加的关键路径归因；详细数值见 profile_*.json。稳态各 case 的精确关键路径尚未分解。", "", "容量分解和全部 54 项容量（包括 TP 方案微批 2/3）见 capacity_table.json；已知 persistent buffer 单列于 results.csv。Pro FPGA 专家权重每芯片 25.69 GB，对应 HBM 上限 32 GB；额外 workspace 及 DDR staging 未知。GPU 可用不等同 FPGA 或实际部署已验收。量级检查：216 GB 以 4 TB/s 完整读取一次的理想带宽时间为 54 ms；实际权重并非每步全量读取，不能将此作为时延或瓶颈定论。", "", "## 复现", "", "命令清单见 jobs/commands.txt，逐 case 输出在 jobs/；容量由现有 src capacity owner 计算。仿真结果及图表均是可再生数据。"]
    (ROOT / "report.md").write_text("\n".join(lines)+"\n")
    table = "".join("<tr>"+"".join(f"<td>{html.escape(str(r[k]))}</td>" for k in ("model","hardware","bw216","scheme","microbatches","origin","status","tpot_ms","tokens_per_s","capacity_status"))+"</tr>" for r in rows)
    (ROOT / "index.html").write_text('<meta charset="utf-8"><title>DSV4 decode 对照</title><h1>DSV4 decode 对照</h1><p><a href="report.md">中文报告</a> · <a href="results.csv">CSV</a> · <a href="'+str(OLD / 'visualizations/overview_20261105.html')+'">原第一层时间线</a></p><img width="100%" src="figures/comparison.svg"><img width="100%" src="figures/capacity.svg"><table border="1"><tr><th>模型</th><th>硬件</th><th>216GB</th><th>方案</th><th>微批</th><th>来源</th><th>状态</th><th>TPOT ms</th><th>tokens/s</th><th>容量</th></tr>'+table+'</table>')
    new_identities = [c["source_identity"] for c in cases if c["origin"] == "new" and "source_identity" in c]
    old_identity = read(old_matrix)["source_identity"]
    changed_sources = {name: {"old_sha256": old_identity[name], "new_sha256": new_identities[0][name]}
        for name in old_identity if new_identities and name in new_identities[0] and old_identity[name] != new_identities[0][name]}
    repro = {"new_run_source_identity": new_identities[0] if new_identities else None,
        "new_source_identity_consistent": all(x == new_identities[0] for x in new_identities) if new_identities else None,
        "changed_source_hashes_against_original": changed_sources,
        "original_reused_files": {str(p): digest(p) for p in (old_matrix, old_capacity, OLD / "reproduction.json")},
        "baseline_commit": "46563ce8f5cee590bd16a5b78ace7297c1f3aa4c",
        "report_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
        "old_prediction_commit": "f90f227a18bb1aeacd300c05c2ca03c5ac8b4a0e", "commands": str(ROOT / "jobs/commands.txt"),
        "capacity_command": f"{sys.executable} scripts/capacity_dsv4_pro_bw1100.py",
        "profile_command": f"{sys.executable} scripts/profile_dsv4_decode_step.py --scheme dp8_ep8 --output experiment_logs/bw1100_pro5500_fpga/pro/profile_pro_bw_dp.json",
        "report_command": f"{sys.executable} scripts/report_dsv4_pro_bw1100.py",
        "input_sha256": {str(path): digest(path) for path in (
            REPOSITORY_ROOT / "configs/model/deepseek-v4-flash.json",
            REPOSITORY_ROOT / "configs/model/deepseek-v4-pro.json",
            REPOSITORY_ROOT / "configs/evaluation/experiments/hygon_bw1100_fp4_decode.yaml",
            REPOSITORY_ROOT / "configs/evaluation/experiments/hygon_bw1100_216gb_fp4_decode.yaml",
            REPOSITORY_ROOT / "configs/evaluation/experiments/nvidia_rtx_pro_5500_fp4_decode.yaml",
            REPOSITORY_ROOT / "configs/evaluation/experiments/intel_agmf039r47_fp4_decode.yaml",
        )},
        "new_performance_completed": done, "new_performance_capacity_rejected": rejected,
        "new_capacity_cases": len(new_capacity), "old_performance_reused": 14, "old_capacity_reused": 18}
    (ROOT / "reproduction.json").write_text(json.dumps(repro, indent=2)+"\n")

if __name__ == "__main__":
    main()
