"""910B kernel-launch tile/grid alignment checker (goal part C).

Compares the SYNTHETIC kernel emits (operator_records._op_to_kernels, the
Ascend branches) against the REAL msprof-captured launch geometry (Op Name +
Block Dim / Mix Block Dim from op_summary*.csv). On 910B the "tile" of a cube
GEMM/GroupedMatmul is fixed 16³ (CANN INI, no autotuner), so the launch-geometry
contract is the GRID (Block Dim = num AIC cores = 24 for cube, 24+48 MIX_AIC
for FA, 48 AIV for vector/norm/act). This checker verifies the synthetic emit's
kernel_name + grid match the real msprof Op Name + Block Dim.

Hooked into run_prediction right after the regenerated per_rank_traces are
built (both the synthetic kernels and the case_dir msprof are in scope).

mismatch → print [grid_mismatch] / [name_mismatch] error lines + a final
tile_match X/Y (Z%) summary. CANN tiling is deterministic per shape on 910B
(no cuBLAS autotuner nondeterminism), so a grid mismatch is a real emit bug.
"""
import csv
import glob
import os
from pathlib import Path


_LEGACY_GEMM_LUT = (
    Path(__file__).resolve().parents[3]
    / "archive/serving_legacy/kernel_synthesis/data/gemm_kernel_lut_huawei_910b.json"
)

# Synthetic kernel_name substring → (msprof Op-Name keyword, expected Block Dim,
# "mix" flag). The checker finds the real msprof Op Name containing the keyword
# and compares the synthetic grid[0] to the real Block Dim.
#
# cube ops: Block Dim 24 (AIC). FA: 24 + Mix 48 (MIX_AIC). vector/norm/act: 48 (AIV).
_EXPECTED = {
    # (substring in synthetic name, msprof keyword, expected block_dim, needs_mix48)
    # NOTE: synthetic kernel names differ by model family — Llama emits the
    # vllm-ascend aclnn names (ReshapeAndCacheNdKernel / SwiGlu / _triton_rope /
    # RmsNorm / AddRmsNormBias), Qwen3 emits lowercase_* names (reshape_and_cache
    # / act_and_mul / rotary_embedding). Substrings must match BOTH conventions,
    # so no underscores that would miss the no-underscore variant (the old
    # "rms_norm"/"fused_add_rms_norm" keys silently skipped all Norm emits).
    ("aclnnmatmul", "matmul"): (24, False),       # dense GEMM
    ("aclnngroupedmatmul", "groupedmatmul"): (24, False),  # MoE grouped GEMM
    ("fusedinferattentionscore", "fusedinferattentionscore"): (24, True),  # FA (MIX_AIC)
    ("addrmsnormbias", "addrmsnormbias"): (48, False),   # AddRmsNormBias (AIV)
    ("rmsnorm", "rmsnorm"): (48, False),                 # RmsNorm (AIV)
    ("swiglu", "swiglu"): (48, False),                  # SwiGlu (AIV) — Llama
    ("act_and_mul", "act_and_mul"): (48, False),         # act_and_mul (AIV) — Qwen3
    ("_triton_rope", "_triton_rope"): (48, False),      # rotary (AIV) — Llama
    ("rotary_embedding", "_triton_rope"): (48, False),   # rotary (AIV) — Qwen3
    ("reshapeandcachend", "reshapeandcachend"): (48, False),  # kv_cache — Llama
    ("reshape_and_cache", "reshapeandcachend"): (48, False),  # kv_cache — Qwen3
    ("moegatingtopk", "moegatingtopk"): (48, False),
    ("moeinitrouting", "moeinitrouting"): (48, False),
    ("moetokenunpermute", "moetokenunpermute"): (48, False),
}


def _msprof_op_rows(case_dir):
    """Yield per-invocation op rows {op_name, op_type, block_dim, mix_block_dim,
    input_shapes} from msprof op_summary*.csv, OR (when CSV absent) from the
    ai_core_op_summary.db sqlite `ge_summary` table.

    The sqlite is kept when the capture deleted raw data (commit 020ea87 "keep
    device sqlite for TTFT serial, delete raw data only") — so a case with no
    op_summary CSV can STILL be rebuilt from its sqlite (e.g. 30B-A3B EP2 qps4).
    Returns nothing if neither present (case had no msprof capture at all)."""
    import sqlite3
    csvs = glob.glob(os.path.join(str(case_dir), "msprof", "PROF_*",
                                  "mindstudio_profiler_output", "op_summary*.csv"))
    if csvs:
        for f in csvs:
            try:
                for r in csv.DictReader(open(f)):
                    yield {"op_name": (r.get("Op Name") or "").strip(),
                           "op_type": (r.get("OP Type") or "").strip(),
                           "block_dim": r.get("Block Dim"),
                           "mix_block_dim": r.get("Mix Block Dim"),
                           "input_shapes": r.get("Input Shapes", "")}
            except Exception:
                continue
        return
    # CSV absent — rebuild from the kept sqlite ge_summary table.
    dbs = glob.glob(os.path.join(str(case_dir), "msprof", "PROF_*",
                                 "device_*", "sqlite", "ai_core_op_summary.db"))
    for db in dbs:
        try:
            c = sqlite3.connect(db)
            for op_name, op_type, bd, mb, ish in c.execute(
                    "SELECT op_name, op_type, block_dim, mix_block_dim, "
                    "input_shapes FROM ge_summary"):
                yield {"op_name": (op_name or "").strip(),
                       "op_type": (op_type or "").strip(),
                       "block_dim": bd, "mix_block_dim": mb,
                       "input_shapes": ish or ""}
        except Exception:
            continue


def _load_msprof_grids(case_dir):
    """{op_name_lower: {"block_dim": int, "mix_block_dim": int}} from op_summary
    CSV (or rebuilt from the ai_core_op_summary.db sqlite when CSV absent)."""
    out = {}
    any_row = False
    for r in _msprof_op_rows(case_dir):
        any_row = True
        nm = r["op_name"]
        if not nm:
            continue
        bd = r["block_dim"]
        mb = r["mix_block_dim"]
        try:
            bdi = int(float(bd)) if bd not in (None, "") else 0
        except Exception:
            bdi = 0
        try:
            mbi = int(float(mb)) if mb not in (None, "") else 0
        except Exception:
            mbi = 0
        if bdi > 0:
            # first non-zero wins (constant per Op Name)
            out.setdefault(nm.lower(), {"block_dim": bdi, "mix_block_dim": mbi})
    if out:
        return out
    # 3rd source: distilled op_grids.json (csv+sqlite absent).
    jf = os.path.join(str(case_dir), "op_grids.json")
    if os.path.exists(jf):
        try:
            import json as _json
            d = _json.load(open(jf))
            return {k.lower(): v for k, v in d.items()} if isinstance(d, dict) else None
        except Exception:
            pass
    return None


def _build_gemm_grid_lut(case_dir=None):
    """{(M,N,K): block_dim} for 910B dense aclnnMatmul, from the committed
    archived synthesis LUT `archive/serving_legacy/kernel_synthesis/data/`
    `gemm_kernel_lut_huawei_910b.json` (aggregated from
    the 12 TP1/TP2 ShareGPT cases' distilled op_summary by
    scripts/kernel/validation/gen_ascend_910b_runtime_kernel_lut.py).

    On 910B the aclnnMatmul Block Dim (AIC cores used) is a deterministic
    function of (M,N,K) — CANN allocates cores by work size (e.g.
    M=63,K=2048: N=2560->23, N=2048->22, N=128->16). The LUT key MUST be the
    full (M,N,K) (NOT just (M,N) — same (M,N) at different K can map to
    different grids, the A800 o_proj/down_proj lesson).

    Static + committed (mirrors the A800 gemm_kernel_lut convention): the emit
    path (operator_records) loads this and RAISES on an unseen (M,N,K) — no
    silent warn+fallback-to-24. Coverage was verified gap=0 across the 12
    in-scope cases after the lm_head TP-shard recipe fix. case_dir is accepted
    for call-site compat but ignored (the LUT is no longer per-case).
    """
    import json as _json
    from pathlib import Path
    jf = _LEGACY_GEMM_LUT
    if not jf.exists():
        raise FileNotFoundError(
            f"910B GEMM grid LUT not found: {jf}. Run "
            "scripts/kernel/validation/gen_ascend_910b_runtime_kernel_lut.py to build it.")
    d = _json.load(open(jf))
    lut = {}
    for k, v in d.items():
        # k = "[M,N,K]"; v = [[name, [bd,1,1], [1,1,1], smem]] (A800-style
        # single-element list; bd is grid[0] = AIC cores launched).
        try:
            mnk = tuple(int(x) for x in k.strip("[]").split(","))
            bd = int(v[0][1][0])
        except Exception:
            continue
        lut[mnk] = bd
    if not lut:
        raise ValueError(f"910B GEMM grid LUT empty: {jf}")
    return lut


def _build_gemm_name_lut():
    """{(M,N,K): kernel_name} for 910B dense aclnnMatmul, from the SAME committed
    static LUT as _build_gemm_grid_lut. The LUT carries the REAL aclnnMatmul
    op_name (MatMulV2 vs MatMulV3) per (M,N,K) — CANN's V2/V3 dispatch is NOT a
    clean M-threshold (32B: V3 fires only at M∈{8192,1255,1159,1234,1590} +
    specific N,K; V2 at neighboring M like 1018/987). The operator_records emit
    uses this to pick the kernel name instead of the coarse M>128 gate (which
    over-emits V3 on 32B prefill, ~12k name_mm)."""
    import json as _json
    from pathlib import Path
    jf = _LEGACY_GEMM_LUT
    d = _json.load(open(jf))
    lut = {}
    for k, v in d.items():
        try:
            mnk = tuple(int(x) for x in k.strip("[]").split(","))
            name = v[0][0]
        except Exception:
            continue
        lut[mnk] = name
    return lut


def _entry_grid0(entry):
    """Synthetic kernel's grid[0] (gx) — the AIC/AIV core count on 910B."""
    g = entry.get("grid") or entry.get("grid_size")
    if isinstance(g, str):
        g = g.strip("()").split(",")
        try:
            return int(float(g[0]))
        except Exception:
            return 0
    if isinstance(g, (list, tuple)) and g:
        try:
            return int(float(g[0]))
        except Exception:
            return 0
    return 0


def check_tile_alignment(case, case_dir, per_rank_traces, verbose=True):
    """Compare synthetic emits (per_rank_traces) to msprof Op Name + Block Dim.

    Returns (n_match, n_mismatch, n_name_mismatch). Prints [grid_mismatch] /
    [name_mismatch] lines + a summary. No-op (returns 0,0,0) if no msprof or no
    ascend traces.
    """
    case_dir = Path(case_dir)
    msprof = _load_msprof_grids(case_dir)
    if not msprof:
        if verbose:
            print(f"[tile_check] {case}: no msprof op_summary — skipping")
        return 0, 0, 0
    msprof_keys = list(msprof.keys())

    # Is this an Ascend (910B) trace? detect by any aclnn*/FusedInfer op in msprof
    is_ascend = any(k.startswith("aclnn") or "fusedinfer" in k for k in msprof_keys)
    if not is_ascend:
        return 0, 0, 0  # only meaningful on 910B

    n_match = n_grid_mismatch = n_name_mismatch = n_checked = 0
    # accumulate per-(synthetic_name) to avoid per-invocation spam; track a count
    per_name = {}
    # GEMM grid is a per-(M,N,K) function on 910B (CANN allocates AIC cores by
    # work size — NOT constant 24). Verify per-(M,N,K) using the op_summary LUT
    # built from Input Shapes + Block Dim (the user directive: full (M,N,K) key,
    # never (M,N) alone — same (M,N) at different K can be different grids).
    gemm_lut = _build_gemm_grid_lut(case_dir)
    per_mnk = {}  # (kn, mnk, real_bd, gx) -> count

    for rank_id, trace in (per_rank_traces or {}).items():
        if not trace:
            continue
        for e in trace:
            kn = (e.get("kernel_name") or e.get("name") or "").strip()
            if not kn:
                continue
            knl = kn.lower()
            # GEMM per-(M,N,K) path: aclnnMatmul with op_attr M,N,K. Only when
            # the op_summary LUT is non-empty (the case HAS an msprof op_summary
            # capture); when absent there's no per-shape Block-Dim ground truth
            # to verify against, so skip (the emit uses the default grid=24 +
            # the checker notes "no op_summary" below).
            oa = e.get("op_attr") or {}
            if knl.startswith("aclnnmatmul") and all(
                    oa.get(d) is not None for d in ("M", "N", "K")) and gemm_lut:
                mnk = (int(oa["M"]), int(oa["N"]), int(oa["K"]))
                real_bd = gemm_lut.get(mnk)
                gx = _entry_grid0(e)
                n_checked += 1
                key = (kn, mnk, real_bd, gx)
                per_mnk[key] = per_mnk.get(key, 0) + 1
                continue
            # find the expected (msprof keyword, block_dim, mix) for this synthetic name
            exp = None
            for (sub, kw), v in _EXPECTED.items():
                if sub in knl:
                    exp = (kw, v[0], v[1])
                    break
            if exp is None:
                continue  # comm/memcpy/elementwise glue — not in the 910B grid contract
            kw, exp_bd, needs_mix = exp
            n_checked += 1
            # Prefer an EXACT name match (synthetic name == real Op Name) so a
            # MatMulV2 emit pairs with the real aclnnMatmul...MatMulV2 (bd=24),
            # NOT with aclnnBatchMatMul (bd=6) — catches op-type mismatches too.
            # Fall back to the category keyword if no exact match.
            real = knl if knl in msprof else next(
                (k for k in msprof_keys if kw in k), None)
            gx = _entry_grid0(e)
            # expected block_dim = the SPECIFIC real op's Block Dim when we have
            # an exact name match; else the category default (exp_bd).
            if real is not None and knl in msprof:
                exp_bd = msprof[real]["block_dim"] or exp_bd
            key = (kn, real, exp_bd, gx)
            slot = per_name.setdefault(key, 0)
            per_name[key] = slot + 1

    # Report — aggregate by (synthetic_name, real_op, status) so a kernel
    # launched at many token counts (e.g. rms_norm grid=tokens) prints ONE
    # line, not one per distinct grid. status ∈ {match, grid_mismatch,
    # name_mismatch}.
    case_id = case.get("id", str(case)) if isinstance(case, dict) else str(case)
    # group: (synthetic_name, real_op, status) -> [count, set(distinct grid0)]
    groups = {}
    for (kn, real, exp_bd, gx), cnt in per_name.items():
        if real is None:
            st = "name_mismatch"
        elif gx != exp_bd:
            st = "grid_mismatch"
        else:
            st = "match"
        k = (kn, real or "—", st)
        g = groups.setdefault(k, [0, set()])
        g[0] += cnt
        if st == "grid_mismatch":
            g[1].add(gx)

    for (kn, real, st), (cnt, grids) in sorted(groups.items(),
                                               key=lambda kv: (kv[0][2] != "match", -kv[1][0], kv[0][0])):
        if st == "name_mismatch":
            print(f"[name_mismatch] case={case_id} kernel='{kn}' count={cnt} "
                  f"reason='no msprof Op Name matches; synthetic name not launched on 910B'")
        elif st == "grid_mismatch":
            actual_bd = msprof[real]["block_dim"]
            samp = sorted(grids)[:6]
            print(f"[grid_mismatch] case={case_id} kernel='{kn}' count={cnt} "
                  f"distinct_pred_grids={len(grids)} sample={samp} "
                  f"expected_block_dim={msprof.get(real,{}).get('block_dim')} "
                  f"actual_op='{real}' actual_block_dim={actual_bd}")
        else:
            print(f"[tile_match]    case={case_id} kernel='{kn}' actual_op='{real}' count={cnt}")

    # GEMM per-(M,N,K) grid report — the CANN aclnnMatmul Block Dim varies by
    # (M,N,K); compare the synthetic grid (currently a constant 24) to the real
    # per-shape Block Dim from the op_summary LUT. Mismatch = the emit's fixed
    # grid doesn't match the shape-dependent CANN selection (the emit-side LUT
    # fix is the follow-up the checker surfaces). unseen_mnk = the synthetic
    # (M,N,K) wasn't in the op_summary LUT (rare — emit shapes come from the
    # same capture; warns so no silent gap).
    g_match = g_mismatch = g_unseen = 0
    for (kn, mnk, real_bd, gx), cnt in sorted(per_mnk.items(),
                                               key=lambda kv: (kv[0][2] is not None, -kv[1])):
        M, N, K = mnk
        if real_bd is None:
            g_unseen += 1
            # don't spam: only print if many unseen or verbose
        elif gx == real_bd:
            g_match += 1
        else:
            g_mismatch += 1
            print(f"[grid_mismatch] case={case_id} kernel='aclnnMatmul' M={M} N={N} K={K} "
                  f"count={cnt} predicted_grid0={gx} actual_block_dim={real_bd} "
                  f"(CANN allocates {real_bd} AIC cores for this shape, emit used {gx})")
    if g_unseen:
        print(f"[tile_check] case={case_id}: GEMM unseen (M,N,K) not in op_summary LUT: "
              f"{g_unseen} shape(s) — cannot verify grid")
    if per_mnk:
        gtot = g_match + g_mismatch
        gpct = 100.0 * g_match / gtot if gtot else 0.0
        print(f"[tile_check] case={case_id}: GEMM per-(M,N,K) grid match {g_match}/{gtot} "
              f"({gpct:.1f}%)  mismatch={g_mismatch}  unseen={g_unseen}")
    elif not gemm_lut:
        print(f"[tile_check] case={case_id}: GEMM grid UNVERIFIABLE — no msprof "
              f"op_summary*.csv AND no ai_core_op_summary.db sqlite for this "
              f"case (re-run serving with msprof op_summary to capture); emit "
              f"uses default grid=24.")

    n_match = sum(1 for _, _, st in groups if st == "match") + g_match
    n_grid_mismatch = sum(1 for _, _, st in groups if st == "grid_mismatch") + g_mismatch
    n_name_mismatch = sum(1 for _, _, st in groups if st == "name_mismatch")
    total = n_match + n_grid_mismatch + n_name_mismatch
    pct = 100.0 * n_match / total if total else 0.0
    print(f"[tile_check] case={case_id}: tile_match {n_match}/{total} "
          f"({pct:.1f}%)  grid_mismatch={n_grid_mismatch}  name_mismatch={n_name_mismatch}  "
          f"(checked {n_checked} invocations)")
    return n_match, n_grid_mismatch, n_name_mismatch
