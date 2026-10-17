#!/usr/bin/env python3
"""Archived diagnostic: random search-size sweep for Timeloop mapping quality.

Exhaustive can't find valid mappings on this mapspace (fanout violations), so we
measure search-suboptimality a different way: increase random search_size and see
if more samples find lower cycles. If search_size=200k << 2k → better mappings exist
(search suboptimality, recoverable by bigger random search = shippable 方案 A). If
saturated (200k ≈ 2k) → random has exhausted what it can find; gap is mapspace/physical
floor (方案 B or mapspace rework).
"""
import json, os, re, subprocess, tempfile, time, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
TL_DIR = str(ARCHIVE_ROOT / "timeloop")
TIMELOOP_BIN = "/share_data/yangxinhao/chip_modeling/timeloop/bin/timeloop-mapper"
ENV_PREFIX = "/home/yangxinhao_hermes/.conda/envs/timeloop"
ARCH = f"{TL_DIR}/gemmini_arch.yaml"
MAPSPACE = f"{TL_DIR}/gemm_mapspace.yaml"
PROBLEM_TMPL = f"{TL_DIR}/gemm_problem_template.yaml"

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = f"{ENV_PREFIX}/lib:{os.path.dirname(TIMELOOP_BIN)}/../build:{env.get('LD_LIBRARY_PATH','')}"

SIZES = [2000, 50000, 200000]
SIZE_TIMEOUT = {2000: 120, 50000: 500, 200000: 1200}

SHAPES = [
    (32,   256,  256),    # 1 tiny
    (128,  512,  512),    # 2 small
    (512,  1024, 1024),   # 3 med
    (1,    4096, 4096),   # 4 decode mem-bound
    (2048, 1536, 5120),   # 5 prefill
    (2048, 4096, 4096),   # 6 prefill
    (2048, 4096, 11008),  # 7 prefill large
    (4096, 4096, 4096),   # 8 large prefill
]

NCPU = os.cpu_count() or 128
def cpu_mask(idx):
    base = (idx * 16) % NCPU; end = min(base + 15, NCPU - 1); return f"{base}-{end}"

def make_mapper(search_size):
    """Write a temp random mapper yaml with given search_size; victory_condition = search_size
    so it samples the FULL budget (doesn't stop after 1000 valid)."""
    d = tempfile.mkdtemp(prefix="tl_map_")
    p = os.path.join(d, "mapper.yaml")
    open(p, "w").write(
        f"mapper:\n  algorithm: random\n  optimization_metrics:\n  - delay\n"
        f"  search_size: {search_size}\n  victory_condition: {search_size}\n"
        f"  timeout: {SIZE_TIMEOUT[search_size]}\n")
    return p

def run_one(job_idx, M, K, N, search_size):
    mp = make_mapper(search_size)
    wd = tempfile.mkdtemp(prefix=f"tl_sw_")
    tmpl = open(PROBLEM_TMPL).read()
    problem_yaml = os.path.join(wd, "problem.yaml")
    open(problem_yaml, "w").write(tmpl.replace("{{M}}", str(M)).replace("{{K}}", str(K))
                                          .replace("{{N}}", str(N)))
    cmd = [TIMELOOP_BIN, ARCH, MAPSPACE, problem_yaml, mp]
    ts = shutil.which("taskset")
    if ts: cmd = [ts, "-c", cpu_mask(job_idx), *cmd]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                              timeout=SIZE_TIMEOUT[search_size], env=env)
    except subprocess.TimeoutExpired:
        shutil.rmtree(wd, ignore_errors=True); shutil.rmtree(os.path.dirname(mp), ignore_errors=True)
        return {"M": M, "K": K, "N": N, "ss": search_size, "status": "TIMEOUT", "cyc": None, "wall": SIZE_TIMEOUT[search_size]}
    wall = time.time() - t0
    stats = os.path.join(wd, "timeloop-mapper.stats.txt")
    cyc = None
    if os.path.exists(stats):
        txt = open(stats).read()
        m = re.search(r"Summary Stats.*?Cycles\s*:\s*(\d+)", txt, re.DOTALL)
        if m: cyc = int(m.group(1))
    shutil.rmtree(wd, ignore_errors=True); shutil.rmtree(os.path.dirname(mp), ignore_errors=True)
    st = "OK" if cyc else ("NO_STATS" if not os.path.exists(stats) else "NO_CYCLES")
    return {"M": M, "K": K, "N": N, "ss": search_size, "status": st, "cyc": cyc, "wall": wall}

def forward_peak(M, K, N):
    cc = 2*M*K*N/512.0; mc = (M*K+K*N+M*N)/16.0; return max(cc, mc)

def main():
    os.makedirs("/tmp/tl_diag", exist_ok=True)
    jobs = [(i, M, K, N, s) for i, (M, K, N) in enumerate(SHAPES) for s in SIZES]
    print(f"running {len(jobs)} jobs (8 shapes × 3 search_sizes), max 8 parallel, 16 cores each")
    print(f"{'shape':<20}{'ss':>8}{'status':>10}{'cyc':>16}{'wall_s':>9}{'fwd_peak':>14}{'util%':>8}")
    print("-"*85)
    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, i, M, K, N, s): (i, M, K, N, s) for i, M, K, N, s in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            key = (r["M"], r["K"], r["N"])
            res.setdefault(key, {})[r["ss"]] = r
            peak = forward_peak(*key)
            util = (peak/r["cyc"]*100) if r["cyc"] else None
            sk = f"M{key[0]}K{key[1]}N{key[2]}"
            print(f"{sk:<20}{r['ss']:>8}{r['status']:>10}{str(r['cyc'] or '-'):>16}"
                  f"{r.get('wall',0):>9.1f}{peak:>14.0f}{(f'{util:.1f}' if util else '-'):>8}", flush=True)

    print("\n" + "="*85)
    print("SWEEP SUMMARY: cycles by search_size (util% = forward_lower_bound / GT)")
    print("="*85)
    print(f"{'shape':<20}{'2k_cyc':>14}{'50k_cyc':>14}{'200k_cyc':>14}{'Δ(200k-2k)%':>14}{'2k_util':>9}{'200k_util':>10}")
    print("-"*91)
    rows = []
    for M, K, N in SHAPES:
        key = (M, K, N); r = res.get(key, {}); peak = forward_peak(M, K, N)
        c2 = r.get(2*1000, {}).get("cyc"); c50 = r.get(50000, {}).get("cyc"); c200 = r.get(200000, {}).get("cyc")
        delta = ((c200 - c2)/c2*100) if (c2 and c200) else None
        u2 = (peak/c2*100) if c2 else None; u200 = (peak/c200*100) if c200 else None
        sk = f"M{M}K{K}N{N}"
        print(f"{sk:<20}{str(c2 or '-'):>14}{str(c50 or '-'):>14}{str(c200 or '-'):>14}"
              f"{(f'{delta:+.1f}' if delta is not None else '-'):>14}"
              f"{(f'{u2:.1f}' if u2 else '-'):>9}{(f'{u200:.1f}' if u200 else '-'):>10}")
        rows.append({"shape": sk, "M":M,"K":K,"N":N,"fwd_peak":peak,
                     "c2k":c2,"c50k":c50,"c200k":c200,"delta_200k_vs_2k_pct":delta,
                     "u2k":u2,"u200k":u200})
    ok = [r for r in rows if r["delta_200k_vs_2k_pct"] is not None]
    if ok:
        import statistics
        deltas = [r["delta_200k_vs_2k_pct"] for r in ok]
        u2s = [r["u2k"] for r in ok]; u2ks = [r["u200k"] for r in ok]
        print(f"\nVERDICT (n={len(ok)}):")
        print(f"  mean Δ(200k vs 2k) = {statistics.mean(deltas):+.1f}%  (negative = bigger search found LOWER cycles)")
        print(f"  mean util: 2k={statistics.mean(u2s):.1f}%  →  200k={statistics.mean(u2ks):.1f}%")
        print(f"  → if 200k notably < 2k: search suboptimality recoverable (方案 A: raise search_size)")
        print(f"  → if 200k ≈ 2k: random saturated; gap is mapspace/physical floor (方案 B)")
    json.dump({"rows": rows}, open("/tmp/tl_diag/sweep_result.json","w"), indent=2)
    print("\nsaved /tmp/tl_diag/sweep_result.json")

if __name__ == "__main__":
    main()
