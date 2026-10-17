#!/usr/bin/env python3
"""Build the 5-part GEMM LUT for 32B shapes by running cuBLAS under nsys.

For each 32B-specific (M,N,K,has_bias=0,dtype=fp16) shape, run F.linear
(cuBLAS) on the GPU, captured by nsys (cudaProfilerApi range). Parse the
nsys sqlite: the captured GEMM kernels are in EXECUTION ORDER == shape order
(single-stream in-order program, no async, no CDR), so kernel[i] <-> shape[i]
DIRECTLY (no index-pairing, no mis-attribution). nsys gives the FULL kernel
info (name + gridX/Y/Z + block + smem + splitKreduce epilogue).

Only 32B-SPECIFIC shapes (not also in TP1 op-records) are replaced — shared
shapes keep the existing 5-part entry (TP1 majority-vote is correct for them,
since TP1 is in-order). This avoids regressing TP1.

Run:
  python scripts/serve_predict/build_lut_nsys.py
"""
import json, os, sqlite3, subprocess, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LUT_PATH = ROOT / "src" / "serve_trace" / "gemm_kernel_lut_a800.json"
RUNNER = ROOT / "scripts" / "serve_predict" / "gemm_runner.py"
R2R_PY = "/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python"
NSYS = "/opt/nvidia/nsight-compute/2023.2.2/host/target-linux-x64/nsys"
OUT_DIR = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / "lut_nsys_build"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEMM_SUBS = ("s16816gemm", "xmma", "gemvx", "gemv2t", "splitkreduce")


def get_shapes(case_ids, rank=0):
    """Distinct (M,N,K,position) linear shapes from op-records. position is
    needed because lm_head's op-record N is the FULL vocab, but TP2 actually
    runs the GEMM on vocab/TP (per-rank) — the cuBLAS shape must be sharded."""
    shapes = set()
    for cid in case_ids:
        cd = ROOT / "serve_exp" / "eval_serving_sharegpt" / cid
        for rfn in cd.glob(f"*_operators.jsonl.rank{rank}"):
            with open(rfn) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("op_type") != "linear":
                        continue
                    a = r.get("attrs", {})
                    M, N, K = a.get("M"), a.get("N"), a.get("K")
                    if M is None or N is None or K is None:
                        continue
                    shapes.add((int(M), int(N), int(K), r.get("position", "")))
    return shapes


def main():
    s32b_cases = [f"ShareGPT-Qwen3-32B_TP2_n128_qps{q}" for q in (1, 4, 16)]
    tp1_cases = ([f"ShareGPT-Qwen3-4B_TP1_n128_qps{q}" for q in (4, 16, 64)]
                  + [f"ShareGPT-Qwen3-8B_TP1_n128_qps{q}" for q in (4, 16, 64)]
                  + [f"ShareGPT-Llama-3.1-8B_TP1_n128_qps{q}" for q in (4, 16, 64)])
    s32b = get_shapes(s32b_cases)
    tp1 = get_shapes(tp1_cases)
    s32b = get_shapes(s32b_cases)
    tp1 = get_shapes(tp1_cases)
    s32b_specific = sorted(s32b - tp1)  # (M,N,K,position), 32B-only
    print(f"32B shapes: {len(s32b)} | TP1 shapes: {len(tp1)} | 32B-specific: {len(s32b_specific)}")

    # 32B is TP2. lm_head op-record N is the FULL vocab, but TP2 runs the GEMM
    # on vocab/2 (per-rank) — the cuBLAS shape must be sharded. Other GEMMs
    # (qkv/o_proj/gate_up/down_proj) op-record is already per-rank (verified
    # gate_up N=25600 == nsys). So only lm_head needs N//TP sharding.
    TP = 2
    shapes_for_runner = []  # [[M, N_cublas, K]] — the shape cuBLAS actually runs
    lut_keys = []            # [(M, N_lut, K)] — the op-record key the dispatch looks up
    for M, N, K, pos in s32b_specific:
        N_cublas = N // TP if pos == "lm_head" else N
        shapes_for_runner.append([M, N_cublas, K])
        lut_keys.append((M, N, K))
    n_lmhead = sum(1 for _,_,_,p in s32b_specific if p == "lm_head")
    print(f"  lm_head shapes sharded N//{TP}: {n_lmhead}")

    shapes_file = OUT_DIR / "shapes.json"
    json.dump(shapes_for_runner, open(shapes_file, "w"))

    # 32B captures are INCONSISTENT: qps1/qps16 captured fp16, qps4 bf16 (model
    # is bf16 per config.json, so qps4 is the correct capture). Build LUT for
    # BOTH dtypes so each case matches its own nsys dtype.
    lut = json.load(open(LUT_PATH))
    for dtype in ("fp16", "bf16"):
        nsys_out = str(OUT_DIR / f"lut_build_{dtype}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        cmd = [NSYS, "profile", "--capture-range=cudaProfilerApi",
               "--capture-range-end=stop", "--trace=cuda", "--export=sqlite",
               "-f", "true", "-o", nsys_out, R2R_PY, str(RUNNER),
               str(shapes_file), dtype]
        print(f"[nsys] running {len(shapes_for_runner)} shapes dtype={dtype}...")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(f"[nsys] {dtype} FAILED (rc={r.returncode})\n{r.stderr[-1500:]}")
            continue
        sqlite_path = Path(nsys_out + ".sqlite")
        if not sqlite_path.exists():
            print(f"[nsys] {dtype} no sqlite at {sqlite_path}"); continue

        conn = sqlite3.connect(str(sqlite_path))
        rows = conn.execute(
            "SELECT k.start, s.value, k.gridX, k.gridY, k.gridZ, "
            "k.blockX, k.blockY, k.blockZ, k.staticSharedMemory+k.dynamicSharedMemory "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
            "WHERE k.deviceId=0 ORDER BY k.start").fetchall()
        conn.close()
        filt = [(s, nm, (gx, gy, gz), (bx, by, bz), smem)
                for s, nm, gx, gy, gz, bx, by, bz, smem in rows
                if any(sub in nm.lower() for sub in GEMM_SUBS)]
        groups = []
        i = 0
        while i < len(filt):
            _s, nm, grid, block, smem = filt[i]
            if "splitkreduce" in nm.lower():
                i += 1; continue
            g = [[nm, list(grid), [block[0], block[1]], int(smem)]]
            if i + 1 < len(filt) and "splitkreduce" in filt[i + 1][1].lower():
                _s2, nm2, grid2, block2, smem2 = filt[i + 1]
                g.append([nm2, list(grid2), [block2[0], block2[1]], int(smem2)])
                i += 2
            else:
                i += 1
            groups.append(g)
        print(f"[nsys] {dtype}: parsed {len(groups)} shape-groups")

        n_repl = 0
        for (M, N, K), g in zip(lut_keys, groups):
            key = f"[{M},{N},{K},0,{dtype}]"  # 5-part, has_bias=0, op-record N (full for lm_head)
            if key in lut:
                n_repl += 1
            lut[key] = g
        print(f"[nsys] {dtype}: replaced {n_repl} entries")

    json.dump(lut, open(LUT_PATH, "w"), indent=0, sort_keys=True)
    print(f"LUT saved. Total entries: {len([k for k in lut if k != '_meta'])}")


if __name__ == "__main__":
    main()
