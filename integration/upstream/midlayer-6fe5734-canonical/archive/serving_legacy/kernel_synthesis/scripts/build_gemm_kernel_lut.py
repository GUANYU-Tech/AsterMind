#!/usr/bin/env python3
"""Build the unified GEMM/GEMV kernel decision table from nsys (A800, bf16).

Key = (M_tok, N_out, K) EXACT (no pad64 — cuBLAS picks per-exact-M, pad64 collapsed
M=65..128 and picked the wrong dominant tile for ~5% of shapes).
Value = [kernel_name, [gx,gy,gz], [bx,by,bz], smem] = the DOMINANT nsys kernel for
that (M,N,K). Covers s16816 + internal_gemvx + gemv2T + sm80_xmma_gemm (whatever
cuBLAS actually launched). Replaces _select_tc_tile + the hardcoded GEMV/gemv2T
thresholds + the wrong gemvx grid ceil(N/4).

Built per-step (operator-record linear op (M,N,K) <-> nsys GEMM-family kernel, joined
on step_id==sidecar iter_id), dominant kernel per (M,N,K). Unseen -> assert at runtime.
"""
import json, re, sqlite3, sys
from collections import defaultdict, Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.build_gemm_tile_lut_mnk import (
    _load_sidecar, _nsys_bounds, _ts_boundaries, _sync_boundaries, EXP_ROOT)
from scripts.serve_predict.validate_trace_alignment import load_nsys_all as _load_nsys_all
from src.serve_predict.model_config import load_model_config as _load_mc_path


def _load_model_config(case_dir):
    """Load ModelConfig from the case's meta.json `model` path (for has_bias)."""
    import json as _json
    meta = _json.load(open(case_dir / "meta.json"))
    return _load_mc_path(meta["model"])

LUT_OUT = ROOT / "src" / "serve_trace" / "gemm_kernel_lut_a800.json"
# Substrings (lowercased) selecting the cuBLAS GEMM-family on the nsys side. Use
# bare tokens that match the demangled names: nsys gemvx is "internal::gemvx"
# (colons, not the recipe's "internal_gemvx" underscore), xmma is
# "sm80_xmma_gemm_*". "gemvx" does not cross-match "gemv2t" (no 'x').
GEMM_SUBS = ["s16816gemm", "s1688gemm", "gemvx", "gemv2t", "xmma"]
TILE_RE = re.compile(r'(\d+)x(\d+)')


def load_all_linear(case_dir):
    """All rank-0 linear operator records (M,N,K) with step_id, in op_idx order.
    NOT TC-filtered (the unified table must cover GEMV-band decode shapes too, which
    the _is_tc_dispatch filter excluded)."""
    rf = case_dir / "plugin_trace.json_operators.jsonl.rank0"
    if not rf.exists():
        rf = sorted(case_dir.glob("*_operators.jsonl.rank*"))[0]
    out = []
    with open(rf) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("op_type") != "linear":
                continue
            a = r.get("attrs") or {}
            if a.get("M") is None or a.get("N") is None or a.get("K") is None:
                continue
            out.append((r.get("step_id"), int(a["M"]), int(a["N"]), int(a["K"]),
                        r.get("position", "")))
    return out


def load_nsys_gemm(case_dir):
    """nsys GEMM-family kernels (device 0, ordered by start) with name+grid+block+smem."""
    p = case_dir / "nsys_profile.sqlite"
    if not p.exists():
        return None, None, None
    conn = sqlite3.connect(str(p))
    idset = set()
    for sub in GEMM_SUBS:
        for r in conn.execute(
            "SELECT id FROM StringIds WHERE value LIKE ?", (f"%{sub}%",)).fetchall():
            idset.add(r[0])
    if not idset:
        conn.close()
        return [], None, None
    ph = ",".join("?" * len(idset))
    rows = conn.execute(
        f"SELECT k.start, s.value, k.gridX, k.gridY, k.gridZ, "
        f"k.blockX, k.blockY, k.staticSharedMemory + k.dynamicSharedMemory "
        f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
        f"WHERE k.demangledName IN ({ph}) AND k.deviceId=0 ORDER BY k.start",
        list(idset)).fetchall()
    conn.close()
    kernels = [(r[0], r[1], (r[2], r[3], r[4]), (r[5], r[6], 1), r[7] or 0)
               for r in rows]
    iters, ts = _load_sidecar(case_dir)
    n0, le = _nsys_bounds(case_dir)
    # Prefer per-step cudaStreamSynchronize completion boundaries (physical,
    # no extrapolation drift) over ts extrapolation; _sync_boundaries returns
    # iters trimmed to its sync count (capture tail dropped) so the downstream
    # zip stays iter↔boundary-aligned. Fall back to ts when sync unavailable
    # (no runtime table / multi-rank / sync count mismatch — TP1-only for now).
    sb = _sync_boundaries(case_dir, iters) if iters else None
    if sb is not None:
        bounds, iters = sb
    elif iters and ts and n0:
        bounds = _ts_boundaries(ts, n0, le)
    else:
        bounds = None
    return kernels, iters, bounds


def _partition(kernels, bounds, iters):
    if bounds is None or not iters:
        return {None: list(kernels)}
    by = {it: [] for it in iters}
    ki = 0
    n = len(kernels)
    for idx, it in enumerate(iters):
        lo = bounds[idx]
        hi = bounds[idx + 1] if idx + 1 < len(bounds) else float("inf")
        while ki < n and kernels[ki][0] < lo:
            ki += 1
        j = ki
        while j < n and kernels[j][0] < hi:
            by[it].append(kernels[j])
            j += 1
        ki = j
    return by


def _consistent(N, M, kname, gx, gy):
    """Gate: nsys grid self-consistent with op-record (N,M) for tile-bearing families
    (s16816/xmma). GEMV (gemvx/gemv2T) has no parseable tile -> accept (per-step order
    already aligns it)."""
    if "s16816" in kname or "xmma" in kname:
        m = TILE_RE.search(kname)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 32 <= a <= 512 and 32 <= b <= 512:
                import math
                if (N <= gx * a < N + a) and (M <= gy * b < M + b):
                    return True
                if gy == 1 and gx == math.ceil(N / a) * math.ceil(M / b):
                    return True
                return False
    return True


def _dtype_of(kname):
    """Detect GEMM dtype from the cuBLAS/cutlass kernel name."""
    n = kname or ""
    if "bf16" in n or "bfloat16" in n:
        return "bf16"
    if "fp16" in n or "_f16" in n or "float16" in n:
        return "fp16"
    return "fp16"  # default (cutlass xmma is f16)


def _has_bias_of(kname):
    """bias -> ampere_s16816gemm_relu epilogue; no bias -> plain s16816/xmma."""
    return 1 if "relu" in (kname or "") else 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Only scan these case-dir name substrings (default: all A800). "
                         "Merge mode: re-derives only these cases' shapes, keeps others.")
    a = ap.parse_args()
    dist = defaultdict(Counter)   # (M,N,K,has_bias,dtype) -> Counter((name,grid,block))
    smem_dist = defaultdict(lambda: defaultdict(Counter))  # -> {(name,grid,block): Counter(smem)}
    epilogue = defaultdict(Counter)  # (M,N,K,has_bias,dtype) -> Counter((skr_ker, tc_ker))
    shapes = set()
    for cd in sorted(EXP_ROOT.iterdir()):
        if a.cases and not any(c in cd.name for c in a.cases):
            continue
        meta = cd / "meta.json"
        if not meta.exists():
            continue
        if "A800" not in json.load(open(meta)).get("hw_config", ""):
            continue
        regen = load_all_linear(cd)
        nsys_kernels, iters, bounds = load_nsys_gemm(cd)
        if not regen or not nsys_kernels:
            continue
        # Model config for has_bias: cuBLAS picks a `cutlass::Kernel2<...gemm_relu>`
        # variant for SOME small-M shapes EVEN when the model has no bias, so the
        # kernel NAME's "relu" is NOT a reliable bias signal (a no-bias Qwen3
        # decode-M down GEMM is named ...gemm_relu...). The has_bias LUT key must
        # match what the regen queries (model-config-driven, see operator_records
        # _op_to_kernels has_bias). Source: vLLM QKVParallelLinear bias iff
        # config.attention_bias; RowParallelLinear o_proj iff attention_bias /
        # down iff mlp_bias; ColumnParallelLinear gate_up iff mlp_bias; lm_head
        # never. The op-record position is generic ("row"=o|down, "col"=gate_up),
        # so OR the two config biases for row/col (exact for the no-bias dense
        # models — Llama-3 / Qwen3 set both False).
        mc = _load_model_config(cd)
        def _cfg_has_bias(pos):
            if pos == "lm_head":
                return False
            if pos == "qkv_proj":
                return bool(mc.attention_bias)
            if pos == "col":
                return bool(mc.mlp_bias)
            if pos == "row":
                return bool(mc.attention_bias) or bool(mc.mlp_bias)
            return False
        regen_by_step = defaultdict(list)
        for st, M, N, K, pos in regen:
            regen_by_step[st].append((M, N, K, pos))
        nsys_by_step = _partition(nsys_kernels, bounds, iters)
        # Full-kernel stream (ALL kernels, not just GEMM-family) to detect the
        # splitKreduce epilogue that follows a splitK GEMM (it is NOT a GEMM-
        # family kernel, so the GEMM-filtered nseq misses it).
        full_kernels, _fi, _fb = _load_nsys_all(cd)
        full_start_idx = {fk[0]: i for i, fk in enumerate(full_kernels)} if full_kernels else {}
        # PER-STEP pairing: within each step, the op-idx order of linear ops
        # (qkv, o, gate_up, down, lm_head) matches the nsys GEMM start order
        # (same execution order), and per-step counts match exactly (verified
        # 1:1 for Qwen3-4B: 1532/1532 steps with zero bleed). The earlier
        # GLOBAL position-zip mis-paired when op-step-order ≠ nsys-start-order
        # globally (warmup offset / async launch). Per-step pairing keeps the
        # (M,N,K,has_bias,dtype) label exact. cuBLAS is DETERMINISTIC per
        # (M,N,K,bias,dtype), so the per-step dominant vote is the real kernel.
        for it in iters:
            ops = regen_by_step.get(it, [])
            nks = nsys_by_step.get(it, [])
            if not ops or not nks:
                continue
            for j in range(min(len(ops), len(nks))):
                M, N, K, pos = ops[j]
                _s, kname, grid, block, smem = nks[j]
                dt = _dtype_of(kname)
                hb = 1 if _cfg_has_bias(pos) else 0
                shapes.add((M, N, K, hb, dt))
                # NOTE: smem excluded from the Counter key so it doesn't split
                # the count and change which (name,grid,block) wins most_common
                # (the old code included smem → static-only 49152 collapsed all
                # dynamic variants into one bucket, but static+dynamic splits them
                # → a different kernel could win → GEMM dispatch regression).
                dist[(M, N, K, hb, dt)][(kname, grid, block)] += 1
                # track smem separately per (name,grid,block) for the dominant
                smem_dist[(M, N, K, hb, dt)][(kname, grid, block)][smem] += 1
                gz = grid[2] if len(grid) > 2 else 1
                if gz > 1:
                    fi = full_start_idx.get(_s)
                    nxt1 = full_kernels[fi + 1] if (fi is not None and fi + 1 < len(full_kernels)) else None
                    nxt2 = full_kernels[fi + 2] if (fi is not None and fi + 2 < len(full_kernels)) else None
                    has_skr = nxt1 is not None and "splitKreduce" in nxt1[1]
                    has_tc = has_skr and nxt2 is not None and ("_typeConvert" in nxt2[1] or "typeConvert" in nxt2[1])
                    skr_ker = None
                    tc_ker = None
                    if has_skr and nxt1 is not None:
                        g = nxt1[2]
                        skr_ker = (nxt1[1], (g[0], g[1], g[2] if len(g) > 2 else 1), (32, 16), 0)
                    if has_tc and nxt2 is not None:
                        g = nxt2[2]
                        tc_ker = (nxt2[1], (g[0], g[1], g[2] if len(g) > 2 else 1), (128, 1), 0)
                    epilogue[(M, N, K, hb, dt)][(skr_ker, tc_ker)] += 1
        print(f"  [kernel-lut] {cd.name}", flush=True)
    # MERGE into existing LUT: keep the high-quality 4-part fp16 entries
    # (built by extend_gemm_lut, all shapes) and add the 5-part dtype-keyed
    # entries (bf16 serving shapes from nsys). fp16 queries fall back to the
    # 4-part entries; bf16 queries use the 5-part entries.
    existing = {}
    if LUT_OUT.exists():
        try:
            existing = json.loads(LUT_OUT.read_text())
        except Exception:
            existing = {}
    lut = dict(existing)
    n_added = n_updated = 0
    for k, cc in dist.items():
        # Add 5-part dtype-keyed entries for BOTH fp16 and bf16. The 4-part fp16
        # LUT (from extend_gemm_lut) stays as a dtype-less fallback, but the 5-part
        # entries carry the nsys-captured gridZ (splitK) + splitKreduce epilogue —
        # which the 4-part LUT lacks (it stored gridZ=1, no splitK). fp16 models
        # with splitK (e.g. Qwen3-4B fp16) need the 5-part entries.
        key = f"[{k[0]},{k[1]},{k[2]},{k[3]},{k[4]}]"
        name, grid, block = cc.most_common(1)[0][0]
        # take the most-common smem for this (name,grid,block) — static+dynamic
        smem = smem_dist[k][(name, grid, block)].most_common(1)[0][0]
        # wrap as 1-element list of kernels (the dispatch iterates kernels).
        kernels = [[name, list(grid), list(block), int(smem)]]
        # If this shape's splitK GEMMs usually have a splitKreduce epilogue
        # (majority vote), append it as a 2nd kernel. cuBLAS emits splitKreduce
        # per-run non-deterministically (70B 100%, Llama-8B ~1.6%); only store it
        # when the shape consistently has it, so we don't over-emit.
        # per-shape epilogue (splitKreduce only; typeConvert DISABLED — cross-case
        # contamination: a shape shared between Qwen3-4B (has typeConvert) and
        # Llama-8B (no typeConvert) gets majority typeConvert, but cuBLAS is
        # deterministic per shape, so the shape must differ in some dim (K? leading)
        # not captured. splitKreduce alone is safe (same shape → same splitK
        # behavior across cases). typeConvert needs per-case LUT to fix.
        ep = epilogue.get(k)
        if ep:
            (skr_ker, tc_ker), _ = ep.most_common(1)[0]
            if skr_ker is not None:
                kernels.append([skr_ker[0], list(skr_ker[1]), list(skr_ker[2]), int(skr_ker[3])])
        val = kernels
        if key in lut:
            lut[key] = val; n_updated += 1
        else:
            lut[key] = val; n_added += 1
    LUT_OUT.write_text(json.dumps(lut, indent=0, sort_keys=True))
    missing = [s for s in shapes if f"[{s[0]},{s[1]},{s[2]},{s[3]},{s[4]}]" not in lut]
    print(f"\nLUT: {len(lut)} entries (added {n_added}, updated {n_updated}) | shapes: {len(shapes)} | missing: {len(missing)}")
    for s in missing[:15]:
        print("  MISSING:", s)


if __name__ == "__main__":
    main()
