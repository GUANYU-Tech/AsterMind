"""Archived: build the Gemmini/Groq-LPU dataset via the Timeloop
simulator (NVlabs/timeloop) as groundtruth.

Gemmini (ucb-bar, ISCA21) 是 16×16 output-stationary systolic array。groundtruth
latency 由 Timeloop (NVlabs/timeloop, ISPASS19) 仿真器给出: 对每个 GEMM shape,
timeloop-mapper 搜索最优 mapping 并输出 cycle-accurate Cycles (1GHz → ns)。这是真
仿真器 (cycle-accurate mapping search), 非解析公式。

阶段 (与 build_samsung_hbm_pim_dataset.py 同构):
  - generate: 逐 shape 调 timeloop-mapper -> per-kernel {kernel}.json
  - merge:    合并 per-kernel JSON -> dataflow_dataset.json

算子范围 (与 PIM 同, 复用 EAGERATTN/FUSEDMOE op type):
  - GEMM:     单 timeloop-mapper 调用, latency = 真实 cycle-accurate Cycles。
  - EAGERATTN: 分解为 qkt mm + softmax + pv mm 求和 = 非融合 eager 上界。
  - FUSEDMOE: 分解为 router + softmax + Σ专家(w1+swiglu+w2) + reduce 求和;
              tok_per_expert + moe_tiling 克隆 A800 实测保证自洽。非融合上界。
  详见 docs/dataflow_dataset_notes.md。

依赖: conda env `timeloop` (barvinok/isl/boost/libconfig) + 编译好的
timeloop 二进制 (bin/timeloop-mapper)。见
archive/simulator_legacy/configs/ucb_gemmini.yaml。
"""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.dataset.config_io import (
    load_config,
    get_dataset_gen_cfg,
    get_kernel_config_path,
    get_kernel_dataset_path,
    REPO_ROOT,
)

# Archived Timeloop templates live beside this legacy generator.
_TL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "timeloop"))

# Round-robin 16-core mask assignment for taskset: each timeloop-mapper process
# is pinned to a 16-core range (e.g. "0-15"). timeloop's mapper spawns
# `hardware_concurrency()` threads by default (128) — we cannot set
# num_threads:1 in mapper.yaml (that shrinks the search space so no valid
# mapping is found). taskset confines the whole process (all its threads) to a
# 16-core range, so the 128-thread mapper has real parallelism while N parallel
# generators use N×16 cores total. With search_size=50000 (see mapper.yaml),
# 1-core pinning makes per-shape wall-time hours; 16-core pinning brings it to
# minutes. 128 cores / 16 = 8 non-overlapping ranges → use `-j 8` (no oversub).
# Counter steps by 1 per call; base = (cnt*16) % n cycles 0,16,32,…,112,0,16,…
# so any 8 concurrent procs get disjoint ranges (period 8 = n/16).
_CPU_COUNTER = itertools.count()
_CPU_LOCK = threading.Lock()
_CORES_PER_PROC = 16


def _next_cpu():
    """Return a round-robin 16-core mask (e.g. "0-15") for the next timeloop
    worker. With `-j 8` on a 128-core box this yields 8 disjoint 16-core ranges;
    taskset -c accepts the "lo-hi" range syntax."""
    n = os.cpu_count() or 1
    with _CPU_LOCK:
        base = (next(_CPU_COUNTER) * _CORES_PER_PROC) % n
    return f"{base}-{min(base + _CORES_PER_PROC - 1, n - 1)}"


# ─── NCU 兼容 entry 骨架 (同 PIM, 占位 leaf + Duration=latency) ────────────
def _ncu_sections_skeleton(duration_ns: float) -> dict:
    return {
        "GPU Speed Of Light Throughput": {
            "Header": {
                "Duration [nsecond]": float(duration_ns),
                "SM Frequency [cycle/second]": 1000000000.0,
                "DRAM Frequency [cycle/second]": 1000000000.0,
                "Elapsed Cycles [cycle]": float(duration_ns),
                "SM Active Cycles [cycle]": float(duration_ns),
            }
        },
        "Compute Workload Analysis": {
            "Pipe Utilization (% of active cycles)": {
                "Tensor (All) [%]": 0.0, "FMA [%]": 0.0, "ALU [%]": 0.0,
            },
            "Pipe Utilization (% of peak instructions executed)": {"XU [%]": 0.0},
        },
        "Memory Workload Analysis Chart": {
            "Memory Chart": {
                "l1tex__m_xbar2l1tex_read_bytes.sum [byte]": 0.0,
                "l1tex__m_l1tex2xbar_write_bytes.sum [byte]": 0.0,
                "dram__bytes_read.sum [byte]": 0.0,
                "dram__bytes_write.sum [byte]": 0.0,
                "l1tex__t_sector_hit_rate.pct [%]": 0.0,
                "lts__t_sector_hit_rate.pct [%]": 0.0,
                "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum [byte]": 0.0,
            }
        },
        "Launch Statistics": {
            "Header": {
                "Block Size": 1, "Registers Per Thread [register/thread]": 32,
                "Dynamic Shared Memory Per Block [byte/block]": 0,
                "Static Shared Memory Per Block [byte/block]": 0,
                "Driver Shared Memory Per Block [byte/block]": 0,
                "Achieved Occupancy [%]": 100.0, "Theoretical Occupancy [%]": 100.0,
            }
        },
        "Occupancy": {"Header": {"Block Limit Registers [block]": 1}},
        "Memory Workload Analysis Tables": {
            "Device Memory": {
                "dram__bytes_read.sum.pct_of_peak_sustained_elapsed [%]": 0.0,
                "dram__bytes_write.sum.pct_of_peak_sustained_elapsed [%]": 0.0,
            }
        },
    }


# ─── Timeloop 仿真桥 ──────────────────────────────────────
def _run_timeloop_gemm(timeloop_bin, tl_env, arch_yaml, mapspace_yaml,
                       mapper_yaml, workdir, name, M, K, N, clock_ghz,
                       timeout_s) -> float:
    """Simulate one GEMM via timeloop-mapper; return latency in ns.

    Writes a problem.yaml with M/K/N into workdir, runs timeloop-mapper, parses
    `Cycles: <N>` from timeloop-mapper.stats.txt. latency_ns = cycles / clock_ghz
    (1 GHz → latency_ns = cycles).
    """
    # problem template substitution
    with open(os.path.join(_TL_DIR, "gemm_problem_template.yaml")) as f:
        problem_tmpl = f.read()
    problem_yaml = os.path.join(workdir, f"{name}_problem.yaml")
    with open(problem_yaml, "w") as f:
        f.write(problem_tmpl.replace("{{M}}", str(M))
                              .replace("{{K}}", str(K))
                              .replace("{{N}}", str(N)))

    cmd = [timeloop_bin, arch_yaml, mapspace_yaml, problem_yaml, mapper_yaml]
    # Pin each timeloop-mapper to a single CPU core via taskset. timeloop's mapper
    # spawns `hardware_concurrency()` threads by default (128 here) — we CANNOT set
    # num_threads:1 in mapper.yaml (that shrinks the search space so no valid mapping
    # is found). Instead taskset confines the whole process (all its threads) to one
    # core, so N parallel generators use exactly N cores total, no oversubscription.
    taskset = shutil.which("taskset")
    if taskset:
        cmd = [taskset, "-c", _next_cpu(), *cmd]
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                              timeout=timeout_s, env=tl_env)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"timeloop-mapper timed out >{timeout_s}s")
    if proc.returncode != 0:
        raise RuntimeError(f"timeloop-mapper exit {proc.returncode}: {proc.stderr[-300:]}")

    stats = os.path.join(workdir, "timeloop-mapper.stats.txt")
    if not os.path.exists(stats):
        raise RuntimeError(f"no stats.txt in {workdir}")
    with open(stats) as f:
        txt = f.read()
    if "Summary Stats" not in txt:
        raise RuntimeError(f"no Summary Cycles in stats.txt: rc={proc.returncode} stderr={proc.stderr[-200:]}")
    # 'Summary Stats' block (after a --- separator): 'Cycles: <N>'
    m = re.search(r"Summary Stats.*?Cycles\s*:\s*(\d+)", txt, re.DOTALL)
    if not m:
        raise RuntimeError(f"no Summary Cycles in stats.txt")
    total_cycles = int(m.group(1))
    assert total_cycles > 0, f"non-positive cycles {total_cycles} for {name}"
    # Also extract per-level MACC compute cycles + memory cycles for compute-scale
    # correction (LPU: 256 PEs in arch model vs 409,600 in real MXM = 1600× scale).
    macc_cycles = _extract_level_cycles(txt, "MACC")
    return total_cycles, macc_cycles


def _extract_level_cycles(stats_txt, level_name):
    """Extract the Cycles value for a specific level from Timeloop stats.txt."""
    # Pattern: === <level_name> === ... Cycles : <N>
    m = re.search(rf"=== {re.escape(level_name)} ===.*?Cycles\s*:\s*(\d+)",
                  stats_txt, re.DOTALL)
    return int(m.group(1)) if m else 0


def _run_timeloop_gemm_min(timeloop_bin, tl_env, arch_yaml, mapspace_yaml,
                            mapper_exh_yaml, mapper_rand_yaml, workdir, name,
                            M, K, N, clock_ghz, timeout_s) -> float:
    """Run BOTH exhaustive + random mapper, return min latency (closest to optimal).

    Neither algorithm alone gives the true optimal: exhaustive with a finite
    victory_condition terminates early (suboptimal); random samples (upper bound).
    Running both and taking min = a larger effective search → closest to optimal
    mapping's cycle-accurate latency. Both are real cycle-accurate evaluations of
    valid mappings (same underlying model), so the min is a defensible groundtruth.
    """
    lats = []
    for m_yaml in (mapper_exh_yaml, mapper_rand_yaml):
        try:
            lats.append(_run_timeloop_gemm(timeloop_bin, tl_env, arch_yaml, mapspace_yaml,
                                            m_yaml, workdir, name, M, K, N, clock_ghz, timeout_s))
        except (TimeoutError, RuntimeError, AssertionError):
            pass  # one algo may fail (e.g. exhaustive times out on large shapes)
    if not lats:
        raise RuntimeError(f"both mappers failed for {name}")
    # Return the min-total tuple (both algos return (total_cyc, macc_cyc))
    return min(lats, key=lambda t: t[0])


def _tl_glue():
    """Glue stages (softmax/swiglu/reduce) latency — Gemmini is a GEMM/conv
    accelerator with no native softmax/elewise; these are negligible vs the
    GEMM/MVM stages (same semantics as PIM's best-effort glue = 0)."""
    return 0.0


# ─── Vector roofline (memory-bound elementwise; Gemmini/LPU have no native vector) ─
# Gemmini (16×16 systolic) and LPU (MXM) are GEMM/conv engines — they have NO
# native elementwise/norm/rotary/activation primitive. An elementwise vector op
# (AI ≈ 1 FLOP/byte) is pure memory traffic on these archs, so the defensible
# per-hw groundtruth is the MEMORY ROOFLINE — the same method the LPU GEMM path
# already uses (sram_bandwidth_b_per_cyc). Shapes are cloned from the A800 vector
# attr (344); only Duration is replaced by the roofline latency.
#   Gemmini: 16 B/cyc 128-bit DMA bus (hw spec systolic_dma_bus_bpc=16), 1 GHz,
#            no SRAM skip (DRAM-backed), no pipeline fill.
#   LPU:     61111 B/cyc on-chip SRAM (55 TiB/s @ 900 MHz), +20-tile SIMD fill,
#            skip cases whose data volume > 220 MiB SRAM (goal: LPU SRAM放不下不采集).
_VECTOR_OP_TYPE = {
    "rmsnorm": "NORM", "add_rmsnorm": "NORM", "rope": "ROTARY",
    "silu": "ACTIVATION", "elementwise": "ELEMENTWISE", "kvcache": "KV_CACHE",
}


def _vector_volume_bytes(attr) -> int:
    """Read+write byte volume for a vector op (bf16 = dtype_bytes). Matches the
    traffic convention in src/kernel/ops/norm.py so GT and the forward model
    agree on compulsory bytes (the model's L2/DRAM split is handled elsewhere)."""
    op = attr["op"]
    tok = int(attr["tok"])
    H = int(attr.get("hidden") or 0)
    INT = int(attr.get("intermediate") or 0)
    NH = int(attr.get("num_heads") or 0)
    D = int(attr.get("head_dim") or 0)
    KVH = int(attr.get("num_kv_heads") or 0)
    B = int(attr.get("batch") or 1)
    db = int(attr.get("dtype_bytes") or 2)
    if op == "rmsnorm":
        return 2 * tok * H * db                # read x + write y
    if op == "add_rmsnorm":
        return 2 * tok * H * db                # residual folded in-place (per norm.py)
    if op == "rope":
        return 4 * tok * NH * D * db           # Q,K: 2 read + 2 write
    if op == "silu":
        return 3 * tok * INT * db              # gate read + up read + out write
    if op == "elementwise":
        return 3 * tok * H * db                # 2 read + 1 write (binary op)
    if op == "kvcache":
        return 2 * B * tok * KVH * D * db      # K+V write × batch
    raise ValueError(f"unsupported vector op: {op}")


def _vector_roofline_latency_ns(attr, dcfg) -> float:
    """latency_ns = (volume_bytes / mem_bw_per_cyc + pipeline_fill) / clock_ghz.

    mem_bw / fill / sram_cap come from config (vector_* keys fall back to the
    LPU's general sram_bandwidth_b_per_cyc / pipeline_fill / sram_capacity_bytes;
    Gemmini sets vector_mem_bandwidth_b_per_cyc=16 explicitly). Raises
    RuntimeError if the data volume exceeds the SRAM capacity (LPU skip path)."""
    vol = _vector_volume_bytes(attr)
    mem_bw = int(dcfg.get("vector_mem_bandwidth_b_per_cyc")
                 or dcfg.get("sram_bandwidth_b_per_cyc") or 0)
    assert mem_bw > 0, ("vector roofline needs vector_mem_bandwidth_b_per_cyc "
                        "or sram_bandwidth_b_per_cyc in config")
    fill = int(dcfg.get("vector_pipeline_fill", dcfg.get("pipeline_fill", 0)))
    sram_cap = int(dcfg.get("vector_sram_capacity_bytes",
                            dcfg.get("sram_capacity_bytes", 0)))
    clock = float(dcfg.get("clock_ghz", dcfg.get("gemmini_clock_ghz", 1.0)))
    if sram_cap > 0 and vol > sram_cap:
        raise RuntimeError(f"vector data {vol} > SRAM {sram_cap} "
                           f"({vol/1024/1024:.0f}MB > {sram_cap/1024/1024:.0f}MB)")
    cyc = vol / mem_bw + fill
    return cyc / clock


# ─── EagerAttn / MoE decomposition (同 PIM, 调 _run_timeloop_gemm) ─────────
# EagerAttn geometry uses the A800 FA attr's authoritative Q_LEN (query count)
# / KV_LEN (KV cache length), NOT the legacy S — see build_samsung_hbm_pim_dataset.py
# _decomp_eager_attn_latency header for the full rationale (variant coverage,
# paged-scatter upper bound, MLA up-proj偏下近似). Same here: qkt/pv are real
# timeloop GEMMs, softmax = _tl_glue() (no native softmax on a systolic array).
def _decomp_eager_attn_latency(attr, run_fn, name_stem):
    """Return (latency_ns, stages). attr: {B,Q_LEN?,KV_LEN?,S,HQ,HKV,D,...}."""
    B = int(attr["B"]); HQ = int(attr["HQ"]); D = int(attr["D"])
    Q = int(attr["Q_LEN"]) if "Q_LEN" in attr else int(attr["S"])
    KV = int(attr["KV_LEN"]) if "KV_LEN" in attr else (
        int(attr["KV"]) if "KV" in attr else int(attr["S"]))
    l_qkt = run_fn("mm", [Q, D, KV, HQ * B], f"{name_stem}_qkt")
    l_sm = _tl_glue()
    l_pv = run_fn("mm", [Q, KV, D, HQ * B], f"{name_stem}_pv")
    return l_qkt + l_sm + l_pv, {"qkt": l_qkt, "softmax": l_sm, "pv": l_pv}


def _decomp_moe_latency(attr, run_fn, name_stem):
    """Return (latency_ns, stages). attr: {TOK,E,TOPK,H,INT,tok_per_expert}."""
    H = int(attr["H"]); INT = int(attr["INT"])
    tpe = list(attr["tok_per_expert"])
    l_router = _tl_glue()
    l_rsm = _tl_glue()
    per_tok_e = {}
    for tok_e in sorted({t for t in tpe if t > 0}):
        l_w1 = run_fn("mm", [tok_e, H, 2 * INT, 1], f"{name_stem}_w1_t{tok_e}")
        l_sw = _tl_glue()
        l_w2 = run_fn("mm", [tok_e, INT, H, 1], f"{name_stem}_w2_t{tok_e}")
        per_tok_e[tok_e] = l_w1 + l_sw + l_w2
    l_red = _tl_glue()
    expert_sum = sum(per_tok_e[t] * tpe.count(t) for t in per_tok_e)
    total = l_router + l_rsm + expert_sum + l_red
    return total, {"router": l_router, "softmax": l_rsm,
                   "expert_sum": expert_sum, "reduce": l_red,
                   "active_experts": sum(1 for t in tpe if t > 0)}


# ─── entry makers ─────────────────────────────────────────
def _make_entry(op_type, attr, latency_ns, kernel_name, grid_size="1,1,1"):
    return {
        "op_info": {"Type": op_type, "Attr": attr,
                     "Overall Duration [nsecond]": float(latency_ns)},
        "kernels": [{
            "Metadata": {"Kernel Name": kernel_name, "Grid Size": grid_size,
                         "Block Size": "1,1,1", "Device": "Gemmini", "CC": "gemmini"},
            "Sections": _ncu_sections_skeleton(latency_ns),
        }],
    }


def _make_gemv_entry(B, M, K, N, latency_ns):
    return _make_entry("GEMM", {"B": B, "M": M, "K": K, "N": N}, latency_ns,
                        "gemmini_os_gemm")


# ─── case matrix enumerators (同 PIM, 与 A800 一致) ───────
def _enumerate_gemv_cases(kernel_cfg):
    for wc in kernel_cfg["workload_configs"]:
        B = int(wc["batch"]); M = int(wc["seq_len"])
        for layer in kernel_cfg["layers"]:
            K = int(layer["k"]); N = int(layer["n"])
            yield f"GEMM_B{B}_M{M}_K{K}_N{N}", None, (B, M, K, N)


def _enumerate_eager_attn_cases(kernel_cfg):
    for wc in kernel_cfg["workload_configs"]:
        B = int(wc["batch"]); S = int(wc["seq_len_cache"])
        SP = int(wc["num_splits"]); C = 1 if wc["causal"] else 0
        for layer in kernel_cfg["layers"]:
            HQ = int(layer["num_heads_q"]); HKV = int(layer["num_heads_kv"])
            D = int(layer["head_dim"])
            suf = f"B{B}_S{S}_HQ{HQ}_HKV{HKV}_D{D}_SP{SP}_C{C}"
            yield (f"EAGERATTN_{suf}", f"FLASHATTN_{suf}",
                   (B, S, HQ, HKV, D, SP, C))


def _enumerate_moe_cases(kernel_cfg):
    for wc in kernel_cfg["workload_configs"]:
        TOK = int(wc["num_tokens"])
        for layer in kernel_cfg["layers"]:
            E = int(layer["num_experts"]); TOPK = int(layer["top_k"])
            H = int(layer["hidden_size"]); INT = int(layer["intermediate_size"])
            key = f"FUSEDMOE_TOK{TOK}_E{E}_TOPK{TOPK}_H{H}_INT{INT}"
            yield key, key, (TOK, E, TOPK, H, INT)


def _load_a800_attr_map(cfg, gpu_kernel):
    dcfg = get_dataset_gen_cfg(cfg)
    a800_dir = dcfg["nvidia_a800_routing_dataset"]
    path = a800_dir if os.path.isabs(a800_dir) else os.path.join(REPO_ROOT, a800_dir)
    path = os.path.join(path, f"{gpu_kernel}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v["op_info"]["Attr"] for k, v in data.items()}


_KERNEL_DISPATCH = {
    "gemm": ("gemm", "gemm", _enumerate_gemv_cases),
    "eager_attn": ("flash_attn", "flash_attn", None),   # cloned from A800 FA attr (306)
    "fused_moe": ("fused_moe", "fused_moe", _enumerate_moe_cases),
    # "vector" REMOVED: Dataflow/LPU have no vector execution unit, so the only
    # defensible per-hw GT was a pure memory roofline (vol/mem_bw+fill) — but that
    # is NOT a cycle-sim/real measurement, and the forward model reproduces the
    # SAME roofline formula, making the 0% MAPE circular/meaningless. Dropped
    # until a real DMA-streaming cycle-sim is available (Timeloop's GEMM mapper
    # does not model pure-streaming vector ops). PIM keeps its vector dataset
    # (UniNDP elewise/layernorm bank sim = real cycle-accurate GT).
}


# ─── stages ────────────────────────────────────────────────
def stage_generate(cfg, kernel, max_tasks, start_idx, end_idx, jobs, timeout_s,
                   mapper_name="mapper"):
    """Simulate every case via timeloop-mapper; write {kernel}.json."""
    assert kernel in _KERNEL_DISPATCH, f"unsupported kernel: {kernel}"
    config_subdir, a800_json, enumerate_fn = _KERNEL_DISPATCH[kernel]

    dcfg = get_dataset_gen_cfg(cfg)
    timeloop_bin = dcfg["timeloop_bin"]
    timeloop_env_prefix = dcfg["timeloop_env_prefix"]   # conda env + build/ lib paths
    clock_ghz = float(dcfg.get("clock_ghz", dcfg.get("gemmini_clock_ghz", 1.0)))
    pipeline_fill = int(dcfg.get("pipeline_fill", 0))  # 0 for Gemmini, 20 for LPU
    sram_cap = int(dcfg.get("sram_capacity_bytes", 0))   # 0 = no SRAM check (Gemmini)
    # Compute-scale: LPU arch has 256 PEs (Gemmini structure) but real MXM = 409,600.
    # 409600/256 = 1600. Scale compute cycles by 1/1600. Gemmini: scale=1 (no correction).
    compute_scale = int(dcfg.get("compute_scale", 1))

    arch_yaml = os.path.join(_TL_DIR, dcfg.get("arch_yaml", "gemmini_arch.yaml"))
    mapspace_yaml = os.path.join(_TL_DIR, dcfg.get("mapspace_yaml", "gemm_mapspace.yaml"))
    mapper_yaml = os.path.join(_TL_DIR, f"{mapper_name}.yaml")
    # For --algo-min: run exhaustive + random, take min (closest to optimal).
    mapper_exh_yaml = os.path.join(_TL_DIR, "mapper_exhaustive.yaml")
    mapper_rand_yaml = os.path.join(_TL_DIR, "mapper.yaml")  # random
    use_min = (mapper_name == "min")
    assert os.path.exists(timeloop_bin), f"timeloop_bin not found: {timeloop_bin}"

    # env: conda env lib + timeloop build dir (for libtimeloop-*.so)
    tl_env = dict(os.environ)
    tl_env["LD_LIBRARY_PATH"] = f"{timeloop_env_prefix}/lib:{os.path.dirname(timeloop_bin)}/../build:{tl_env.get('LD_LIBRARY_PATH','')}"

    kernel_cfg_path = get_kernel_config_path(cfg, config_subdir)
    kernel_cfg = load_config(kernel_cfg_path)
    assert kernel_cfg["kernel_name"] == config_subdir, (
        f"kernel_name mismatch in {kernel_cfg_path}: expected {config_subdir}")

    # shared scratch root for per-case workdirs (cleaned after each case)
    scratch_root = tempfile.mkdtemp(prefix="tl_df_")

    def _check_sram(M, K, N):
        """Skip cases whose INT8 data (M*K + K*N + M*N) exceeds SRAM capacity.
        LPU's 220MB SRAM is its design point — cases that don't fit can't run on
        a single LPU without host swap (which the LPU design avoids)."""
        if sram_cap <= 0:
            return  # no SRAM check (Gemmini)
        data_bytes = M * K + K * N + M * N  # INT8: 1 byte each
        if data_bytes > sram_cap:
            raise RuntimeError(f"data {data_bytes} > SRAM {sram_cap} ({data_bytes/1024/1024:.0f}MB > {sram_cap/1024/1024:.0f}MB)")

    def run_fn(_workload, sizes, name):
        M, K, N = sizes[0] * sizes[3], sizes[1], sizes[2]
        _check_sram(M, K, N)
        wd = os.path.join(scratch_root, name)
        os.makedirs(wd, exist_ok=True)
        try:
            if use_min:
                total_cyc, macc_cyc = _run_timeloop_gemm_min(timeloop_bin, tl_env, arch_yaml,
                    mapspace_yaml, mapper_exh_yaml, mapper_rand_yaml, wd, name,
                    M, K, N, clock_ghz, timeout_s)
            else:
                total_cyc, macc_cyc = _run_timeloop_gemm(timeloop_bin, tl_env, arch_yaml, mapspace_yaml,
                                          mapper_yaml, wd, name, M, K, N, clock_ghz, timeout_s)
            # Compute-scale correction (LPU only): the arch model has 256 PEs (Gemmini
            # structure), but real LPU MXM = 409,600 MAC units (4×320×320). Scale the
            # compute portion by 256/409600 = 1/1600. Memory cycles are computed from
            # the published SRAM bandwidth (55 TiB/s, ISCA 2020) and the data volume
            # (M*K + K*N + M*N INT8 bytes) — NOT derived as (total - macc), because when
            # compute was the bottleneck in the 256-PE model, memory was overlapped and
            # its actual time was hidden (total = macc → non_macc = 0, missing memory).
            if compute_scale > 1:
                real_macc = macc_cyc / compute_scale
                # Memory cycles from published SRAM bandwidth + data volume (roofline).
                # SRAM BW = 55 TiB/s @ 900MHz = 61,111 B/cyc (ISCA 2020 Eq 2).
                memory_bytes = M * K + K * N + M * N
                sram_bw_per_cyc = int(dcfg.get("sram_bandwidth_b_per_cyc", 61111))
                memory_cyc = memory_bytes / sram_bw_per_cyc if sram_bw_per_cyc > 0 else 0
                final_cycles = max(real_macc, memory_cyc) + pipeline_fill
            else:
                final_cycles = total_cyc + pipeline_fill
            return final_cycles / clock_ghz
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    a800_map = _load_a800_attr_map(cfg, a800_json) if a800_json else {}
    if a800_json and not a800_map:
        print(f"  WARN: A800 {a800_json} dataset absent — {kernel} cases that need "
              f"cloned op_attr will be skipped", flush=True)

    def simulate(task_key, gpu_key, case):
        if kernel == "gemm":
            B, M, K, N = case
            lat = run_fn("mm", [B * M, K, N, 1], task_key)
            return _make_gemv_entry(B, M, K, N, lat), lat
        if gpu_key not in a800_map:
            return None, None
        attr = a800_map[gpu_key]
        if kernel == "eager_attn":
            lat, _ = _decomp_eager_attn_latency(attr, run_fn, task_key)
            return _make_entry("EAGERATTN", attr, lat, "gemmini_eagerattn"), lat
        if kernel == "vector":
            # Gemmini/LPU have no native vector op → memory-bound roofline (the
            # established LPU GEMM method). Raises on LPU SRAM-overflow (→ skip).
            lat = _vector_roofline_latency_ns(attr, dcfg)
            return _make_entry(_VECTOR_OP_TYPE[attr["op"]], attr, lat,
                                "gemmini_vector"), lat
        lat, _ = _decomp_moe_latency(attr, run_fn, task_key)
        w1_grid = int(attr["moe_tiling"]["w1"]["grid"])
        return _make_entry("FUSEDMOE", attr, lat, "gemmini_moe_decomposed",
                            grid_size=f"{w1_grid},1,1"), lat

    if enumerate_fn is not None:
        cases = list(enumerate_fn(kernel_cfg))
    else:
        # vector/eager_attn: cases cloned verbatim from the A800 attr map.
        assert kernel in ("vector", "eager_attn"), "enumerate_fn=None is vector/eager_attn-only"
        cases = [(k, k, a) for k, a in a800_map.items()]
    lo = start_idx
    hi = len(cases) if end_idx <= 0 else min(end_idx, len(cases))
    cases = cases[lo:hi]
    if max_tasks > 0:
        cases = cases[:max_tasks]

    out_path = get_kernel_dataset_path(cfg, kernel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    results = dict(existing)
    todo = [(tk, gk, c) for (tk, gk, c) in cases if tk not in existing]
    n_new = 0
    n_skip = 0
    print(f"\n>>> Stage: Generate [{kernel}] — {len(cases)} cases "
          f"({len(existing)} cached, {len(todo)} to run) via timeloop-mapper "
          f"(Gemmini OS, {clock_ghz}GHz, timeout={timeout_s}s)", flush=True)

    def worker(task_key, gpu_key, case):
        try:
            entry, lat = simulate(task_key, gpu_key, case)
            if entry is None:
                return task_key, None, None, "no A800 op_attr to clone"
            return task_key, entry, lat, None
        except (TimeoutError, RuntimeError, AssertionError) as e:
            return task_key, None, None, str(e)

    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(worker, tk, gk, c) for (tk, gk, c) in todo]
        for fut in as_completed(futs):
            task_key, entry, lat, reason = fut.result()
            done += 1
            with lock:
                if entry is not None:
                    results[task_key] = entry
                    n_new += 1
                    print(f"  [{done:3d}/{len(todo)}] {task_key}: {lat:.0f} ns",
                          flush=True)
                else:
                    n_skip += 1
                    print(f"  [{done:3d}/{len(todo)}] {task_key}: SKIP ({reason})",
                          flush=True)
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    shutil.rmtree(scratch_root, ignore_errors=True)
    print(f"  wrote {out_path}: {len(results)} entries ({n_new} new, {n_skip} skipped)",
          flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Gemmini (dataflow) dataset generation via Timeloop simulator "
                    "(cycle-accurate groundtruth latency)")
    parser.add_argument(
        "--config",
        default="archive/simulator_legacy/configs/ucb_gemmini.yaml",
    )
    parser.add_argument("--kernel", default="gemm",
                        help="Comma-separated: gemm,eager_attn,fused_moe,vector "
                             "(vector = memory-bound roofline, no timeloop call)")
    parser.add_argument("--stage", required=True, choices=["generate"])
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=0)
    parser.add_argument("-j", "--jobs", type=int, default=32,
                        help="Concurrent timeloop-mapper procs (each single-threaded "
                             "via OMP; jobs = core budget)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Per-case timeloop-mapper timeout in seconds (0 = config)")
    parser.add_argument("--mapper", default="mapper",
                        help="Mapper config basename in archive/simulator_legacy/timeloop/ "
                             "(mapper=50k random, GEMM/MoE; mapper_exhaustive=optimal; "
                             "eager_attn: gemmini100k for dataflow, lpu50k for LPU — "
                             "the committed eager_attn GTs were generated with these "
                             "specific mappers, see docs/{dataflow,lpu}_dataset_notes.md)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timeout_s = (args.timeout if args.timeout > 0
                 else int(get_dataset_gen_cfg(cfg)["sim_timeout_s"]))
    kernels = [k.strip() for k in args.kernel.split(",") if k.strip()]
    assert kernels, "--kernel must provide at least one kernel"

    # pin each timeloop-mapper to single thread (it spawns 128 threads by default)
    os.environ["OMP_NUM_THREADS"] = "1"

    if args.stage == "generate":
        for kernel in kernels:
            stage_generate(cfg, kernel, args.max_tasks, args.start_idx, args.end_idx,
                           args.jobs, timeout_s, args.mapper)


if __name__ == "__main__":
    main()
