#!/usr/bin/env python3
"""Archived validation of mapper.yaml @ search_size=50000 on the largest real
dataset shapes — the worst case for (a) wall-time timeout and (b) util% fix.

Goal (docs/goal_gemmini_sub10.md §3 risk): before the multi-hour full run,
confirm 50k (i) completes within the per-case timeout budget on the biggest
shapes and (ii) lifts util% toward 100% (the actual MAPE fix). The §2 sweep
only went up to M4096K4096N4096 (268M cycles); the real dataset tops out at
~503G cycles (B8_M4096_K8192_N28672) — 1800× bigger cycle output, so wall-time
scaling must be measured, not extrapolated.

Uses a GENEROUS timeout (3600s) to measure TRUE wall time even if production
timeout (600s) would kill it — so we learn whether production 600s is enough.
"""
import json, os, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
TL_DIR = str(ARCHIVE_ROOT / "timeloop")
TIMELOOP_BIN = "/share_data/yangxinhao/chip_modeling/timeloop/bin/timeloop-mapper"
ENV_PREFIX = "/home/yangxinhao_hermes/.conda/envs/timeloop"
ARCH = f"{TL_DIR}/gemmini_arch.yaml"
MAPSPACE = f"{TL_DIR}/gemm_mapspace.yaml"
PROBLEM_TMPL = f"{TL_DIR}/gemm_problem_template.yaml"
# Use the PRODUCTION mapper.yaml (edited to 50k) — but override timeout to 3600
# so we measure true wall time instead of getting a 600s-kill with no data.
PROD_MAPPER = f"{TL_DIR}/mapper.yaml"

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = (f"{ENV_PREFIX}/lib:{os.path.dirname(TIMELOOP_BIN)}/../build:"
                          f"{env.get('LD_LIBRARY_PATH','')}")
os.environ["OMP_NUM_THREADS"] = "1"

SEARCH = 50000
SUBPROC_TIMEOUT = 3700   # generous, wraps the 3600 mapper-internal timeout

# Worst-case real shapes (eff M = B*M, K, N) — the biggest cycle outputs in the
# shipped 2k dataset. These are the prefill cases driving the 29% MAPE bucket.
SHAPES = [
    ("B8_M4096_K8192_N28672",  8*4096, 8192,  28672),  # 2k stored 503G cyc, util 6%  ← worst
    ("B8_M4096_K28672_N8192", 8*4096, 28672, 8192),   # 2k stored 68G cyc,  util 44%
    ("B4_M4096_K28672_N8192", 4*4096, 28672, 8192),    # 2k stored 30G cyc,  util 50%
    # one diagnostic shape re-confirmed with the production file (expect 100%):
    ("diag_M4096K4096N4096",  4096,   4096,  4096),
]

NCPU = os.cpu_count() or 128
def cpu_mask(idx):
    base = (idx * 16) % NCPU; end = min(base + 15, NCPU - 1); return f"{base}-{end}"

def make_mapper(search_size, timeout):
    d = tempfile.mkdtemp(prefix="tl_valmap_")
    p = os.path.join(d, "mapper.yaml")
    open(p, "w").write(
        f"mapper:\n  algorithm: random\n  optimization_metrics:\n  - delay\n"
        f"  search_size: {search_size}\n  victory_condition: {search_size}\n"
        f"  timeout: {timeout}\n")
    return p

def forward_peak(M, K, N):
    cc = 2*M*K*N/512.0; mc = (M*K+K*N+M*N)/16.0; return max(cc, mc)

def run_one(idx, name, M, K, N):
    mp = make_mapper(SEARCH, 3600)
    wd = tempfile.mkdtemp(prefix="tl_val_")
    tmpl = open(PROBLEM_TMPL).read()
    problem_yaml = os.path.join(wd, "problem.yaml")
    open(problem_yaml, "w").write(tmpl.replace("{{M}}", str(M)).replace("{{K}}", str(K))
                                          .replace("{{N}}", str(N)))
    cmd = [TIMELOOP_BIN, ARCH, MAPSPACE, problem_yaml, mp]
    ts = shutil.which("taskset")
    if ts: cmd = [ts, "-c", cpu_mask(idx), *cmd]
    t0 = time.time()
    status, cyc = "OK", None
    try:
        proc = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                              timeout=SUBPROC_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        status = f"SUBPROC_TIMEOUT(>{SUBPROC_TIMEOUT}s)"
    else:
        stats = os.path.join(wd, "timeloop-mapper.stats.txt")
        if os.path.exists(stats):
            txt = open(stats).read()
            m = re.search(r"Summary Stats.*?Cycles\s*:\s*(\d+)", txt, re.DOTALL)
            if m: cyc = int(m.group(1))
            else: status = "NO_CYCLES"
        else:
            status = "NO_STATS"
            if proc.returncode != 0:
                status = f"RC={proc.returncode}:{proc.stderr[-120:]}"
    wall = time.time() - t0
    shutil.rmtree(wd, ignore_errors=True); shutil.rmtree(os.path.dirname(mp), ignore_errors=True)
    peak = forward_peak(M, K, N)
    util = (peak/cyc*100) if cyc else None
    return {"name": name, "M":M,"K":K,"N":N, "status":status, "cyc":cyc, "wall":wall,
            "fwd_peak":peak, "util":util}

def main():
    os.makedirs("/tmp/tl_diag", exist_ok=True)
    print(f"validate 50k on {len(SHAPES)} worst-case shapes (16 cores each, parallel, "
          f"timeout={SUBPROC_TIMEOUT}s wrapper / 3600s mapper-internal)", flush=True)
    print(f"{'name':<26}{'M':>7}{'K':>7}{'N':>7}{'status':>16}{'cyc':>16}{'wall_s':>9}"
          f"{'fwd_peak':>14}{'util%':>8}", flush=True)
    print("-"*96, flush=True)
    res = []
    with ThreadPoolExecutor(max_workers=len(SHAPES)) as ex:
        futs = {ex.submit(run_one, i, *s): s for i, s in enumerate(SHAPES)}
        for fut in as_completed(futs):
            r = fut.result()
            res.append(r)
            util_s = f"{r['util']:.1f}" if r['util'] else "-"
            print(f"{r['name']:<26}{r['M']:>7}{r['K']:>7}{r['N']:>7}{r['status']:>16}"
                  f"{str(r['cyc'] or '-'):>16}{r['wall']:>9.1f}{r['fwd_peak']:>14.0f}"
                  f"{util_s:>8}", flush=True)
    json.dump(res, open("/tmp/tl_diag/validate_50k_huge.json","w"), indent=2)
    print("\nsaved /tmp/tl_diag/validate_50k_huge.json", flush=True)
    # verdict
    ok = [r for r in res if r["cyc"]]
    maxwall = max(r["wall"] for r in res)
    print(f"\nMAX wall = {maxwall:.0f}s  → production mapper-internal timeout 600s "
          f"{'ENOUGH' if maxwall < 590 else 'TOO SHORT — raise timeout'}", flush=True)
    if ok:
        tagged = [f"{r['name']}={r['util']:.1f}%" for r in ok
                  if r['name'].startswith(('B', 'diag'))]
        print(f"util on fixed shapes: {tagged}", flush=True)

if __name__ == "__main__":
    main()
