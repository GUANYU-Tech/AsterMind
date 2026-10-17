"""Aggregate the 12 in-scope TP1/TP2 910B serving cases' distilled GEMM grid
LUTs into ONE committed static LUT, mirroring the A800 gemm_kernel_lut file:

    archive/serving_legacy/kernel_synthesis/data/gemm_kernel_lut_huawei_910b.json
        {"[M,N,K]": [["aclnnMatmul_.._MatMulV2", [block_dim,1,1], [1,1,1], 0]]}

Schema mirrors A800 (single-element list, 1-to-1 per shape). 910B semantics:
  - grid[0] = block_dim = AIC cores launched (the launch geometry; CANN
    deterministic per (M,N,K)). grid[1..2] are 1 (910B has no CUDA-style
    multi-axis grid for cube GEMM).
  - block triple = [1,1,1] placeholder (cube tile is fixed 16^3, not per-shape).
  - smem = 0 (910B uses L1/L2, not CUDA shared mem).

Per shape the real op_name (MatMulV2 vs MatMulV3) is recovered from each
case's op_profile.json (dominant by count across the 12 cases) so the entry
carries the REAL kernel name, like the A800 LUT — not just the synthetic
emit's hard-coded V2 name.

Sources per case (both produced by distill_msprof_op_summary.py at capture):
  gemm_grid_lut.json  -> {(M,N,K): block_dim}        (the grid)
  op_profile.json     -> {op_name|input_shapes|bd}:  (real op_name per shape)

The 12 cases are the TP1+TP2 ShareGPT serving captures (30B-A3B EP and 70B TP4
cases are out of scope by design — their lm_head is a different op / not a
standalone aclnnMatmul).
"""
import json
import os
from collections import Counter, defaultdict

CASE_ROOT = "/mnt/gpfs/midlayer_yxh/serve_exp/serve_exp_910b/eval_serving_sharegpt"

# TP1 + TP2 cases (the 12 in scope).
CASES = [
    "ShareGPT-Llama-3.1-8B_TP1_n128_qps4",
    "ShareGPT-Llama-3.1-8B_TP1_n128_qps16",
    "ShareGPT-Llama-3.1-8B_TP1_n128_qps64",
    "ShareGPT-Qwen3-4B_TP1_n128_qps4",
    "ShareGPT-Qwen3-4B_TP1_n128_qps16",
    "ShareGPT-Qwen3-4B_TP1_n128_qps64",
    "ShareGPT-Qwen3-8B_TP1_n128_qps4",
    "ShareGPT-Qwen3-8B_TP1_n128_qps16",
    "ShareGPT-Qwen3-8B_TP1_n128_qps64",
    "ShareGPT-Qwen3-32B_TP2_n128_qps1",
    "ShareGPT-Qwen3-32B_TP2_n128_qps4",
    "ShareGPT-Qwen3-32B_TP2_n128_qps16",
]

OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "archive", "serving_legacy",
    "kernel_synthesis", "data", "gemm_kernel_lut_huawei_910b.json",
))


def _parse_mnk(input_shapes):
    """op_profile 'input_shapes' "M,K;N,K" -> (M,N,K). None if unparseable."""
    if not input_shapes:
        return None
    s = input_shapes.strip().strip('"').strip()
    parts = [p for p in s.split(";") if p.strip()]
    if len(parts) < 2:
        return None
    try:
        a = [int(float(x)) for x in parts[0].split(",")[:2]]
        b = [int(float(x)) for x in parts[1].split(",")[:2]]
        if len(a) < 2 or len(b) < 2:
            return None
        return (a[0], b[0], a[1])  # (M, N, K)
    except Exception:
        return None


def collect():
    # block_dim source: each case's gemm_grid_lut.json — this is the SAME
    # distillation the checker built (dense aclnnMatmul rows from op_summary),
    # and it is the authoritative per-(M,N,K) grid (verified to cover every
    # synthetic emit shape after the lm_head TP-shard recipe fix; gap=0).
    # op_name source: op_profile.json — carries the real MatMulV2/V3 name per
    # shape (best-effort; op_profile drops a few shapes the grid LUT keeps, so
    # fall back to the synthetic-emit default V2 name when absent).
    bd_lut = {}               # (M,N,K) -> block_dim
    nm_acc = defaultdict(Counter)   # (M,N,K) -> Counter(op_name)
    for c in CASES:
        cd = os.path.join(CASE_ROOT, c)
        gjf = os.path.join(cd, "gemm_grid_lut.json")
        if os.path.exists(gjf):
            for k, v in json.load(open(gjf)).items():
                try:
                    bd_lut[tuple(int(x) for x in k.split(","))] = int(v)
                except Exception:
                    continue
        pjf = os.path.join(cd, "op_profile.json")
        if not os.path.exists(pjf):
            continue
        for pk, pv in json.load(open(pjf)).items():
            parts = pk.split("|")
            if len(parts) < 2:
                continue
            nm = parts[0]
            # dense aclnnMatmul family only (MatMulV2/V3); startswith selects
            # exactly those, excluding batch/grouped + oddballs (PpMatmul...).
            if not nm.lower().startswith("aclnnmatmul_"):
                continue
            mnk = _parse_mnk(parts[1])
            if mnk is None:
                continue
            cnt = pv.get("count", 1) if isinstance(pv, dict) else 1
            nm_acc[mnk][nm] += cnt

    out = {}
    for mnk, bd in bd_lut.items():
        nm = nm_acc[mnk].most_common(1)[0][0] if mnk in nm_acc \
            else "aclnnMatmul_MatMulCommon_MatMulV2"
        out[f"[{mnk[0]},{mnk[1]},{mnk[2]}]"] = [
            [nm, [bd, 1, 1], [1, 1, 1], 0]
        ]
    return out


def main():
    lut = collect()
    with open(OUT, "w") as f:
        json.dump(lut, f, indent=1, sort_keys=False)
    # quick op-name breakdown
    from collections import Counter as C
    names = C(v[0][0] for v in lut.values())
    print(f"wrote {os.path.abspath(OUT)}")
    print(f"  entries: {len(lut)}")
    print(f"  op_name breakdown:")
    for n, c in names.most_common():
        print(f"    {c:>5}  {n}")


if __name__ == "__main__":
    main()
