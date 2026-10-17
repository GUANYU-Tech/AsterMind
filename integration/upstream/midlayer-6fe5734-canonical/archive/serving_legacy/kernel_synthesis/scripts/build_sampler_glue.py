#!/usr/bin/env python3
"""Build per-step sampler_glue.jsonl from nsys (A800, multi-rank-safe).

The lm_head sampler pipeline (post-lm_head, physically executed in the NEXT sync
window under vLLM async scheduling) is NOT a fixed 11-kernel block: its
middle section varies by sampling variant. nsys ground truth (decode the
front-of-step block up to the first vllm::rms_norm_kernel):

  FillFunctor -> [1 or 2 triton_poi_fused_* kernels] -> gather
               -> [1 elementwise_kernel] -> cross_device_reduce

  - 0 triton  : TP1 dense models (Llama-8B/Qwen3-4B/Qwen3-8B) — the simple
    sampler (no triton, no post-gather EW).
  - 1 triton  : 32B multinomial / mixed — 'triton_poi_fused_add_bitwise_and_
    bitwise_not_bitwise_or_ge_lt_mul_sub_0' (WITH 'bitwise_not', suffix _0).
  - 2 triton  : 32B greedy+B=1 with mask — the no-not '_0' THEN the
    '_1' (triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1).

The gather is indexSelectSmallIndex (greedy) or vectorized_gather
(multinomial); its grid and the post-gather EW grid vary per step.

This builder records, per step (attributed to the LM_HEAD's iter = the
PREVIOUS step, since the sampler lands in the next sync window — see
build_fill_grid's matching shift): n_triton, the exact triton kernel names,
gather_type, gather_grid, ew_present, ew_grid. operator_records looks it up
to emit the sampler glue nsys-grounded instead of a hardcoded 11-kernel block.

Uses the CORRECT partitioner (validate_trace_alignment.load_nsys_all +
_partition) — NOT the raw cudaStreamSynchronize.end list, which for TP2
doubles and interleaves non-monotonically (build_sampling_mode's bug).

Usage:
  python scripts/serve_predict/build_sampler_glue.py --case <name>
  python scripts/serve_predict/build_sampler_glue.py   # all A800
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.validate_trace_alignment import load_nsys_all, _partition
from scripts.serve_predict.build_gemm_tile_lut_mnk import _load_sidecar, EXP_ROOT


def _argmax_pos(seq):
    """Position (fraction 0..1) of the sampler argmax (reduce_kernel with
    MaxOps) within the step. <0.1 = front (remapped from prev step, dense/TP2);
    >0.9 = end (in-step, 30B-A3B-EP2). None if not found."""
    n = len(seq)
    for i, r in enumerate(seq):
        nm = r[1]
        if "reduce_kernel" in nm and "MaxOps" in nm:
            return i / max(1, n)
    return None


def _sampler_block_front(seq):
    """Front-of-step kernels up to (not incl) the first vllm::rms_norm_kernel
    (the input_layernorm that starts the transformer body). For remap=True
    cases (dense/TP2) where the sampler lands in the NEXT sync window's front."""
    out = []
    for r in seq:
        nm = r[1]
        if "vllm::rms_norm_kernel" in nm and out:
            break
        out.append(r)
    return out


def _sampler_block_tail(seq):
    """Tail-of-step kernels from the lm_head GEMM (last s16816/s1688) to the
    step end. For remap=False cases (30B-A3B-EP2) where the sampler is in-step
    at the END (after lm_head). Includes the lm_head AllGather + sampler."""
    lh_idx = None
    for i in range(len(seq) - 1, -1, -1):
        nm = seq[i][1]
        if "s16816gemm" in nm or "s1688gemm" in nm or "xmma_gemm" in nm:
            lh_idx = i; break
    if lh_idx is None:
        return []
    return seq[lh_idx + 1:]


def _extract_glue(block):
    """From a sampler block (front or tail), extract the triton kernels (name +
    block), gather type/grid, and post-gather elementwise. The triton_poi_fused
    block varies (32 or 128, torch-inductor num_warps heuristic — not
    source-derivable), so capture it per-step from nsys (the glue is already
    nsys-grounded for the kernel names + gather grid)."""
    tritons = [r for r in block if "triton_poi_fused" in r[1]]  # full tuples (need r[3]=block)
    gather_type = ""
    gather_grid = [0, 0, 0]
    for r in block:
        if "indexSelectSmallIndex" in r[1]:
            gather_type = "idx_sel"; g = r[2]; gather_grid = [g[0], g[1], g[2]]; break
        if "vectorized_gather" in r[1]:
            gather_type = "vec_gather"; g = r[2]; gather_grid = [g[0], g[1], g[2]]; break
    ew_present = False
    ew_grid = [0, 0, 0]
    seen_gather = False
    for r in block:
        if not seen_gather:
            if "indexSelectSmallIndex" in r[1] or "vectorized_gather" in r[1]:
                seen_gather = True
            continue
        if "elementwise_kernel" in r[1] and "direct_copy" not in r[1]:
            ew_present = True; g = r[2]; ew_grid = [g[0], g[1], g[2]]; break
    # triton block (r[3] = (blockX, blockY, blockZ)); default (128,1,1) if missing
    def _blk(r):
        b = r[3] if len(r) > 3 else (128, 1, 1)
        return [int(b[0]), int(b[1]), int(b[2] if len(b) > 2 else 1)]
    t1_block = _blk(tritons[0]) if len(tritons) >= 1 else [128, 1, 1]
    t2_block = _blk(tritons[1]) if len(tritons) >= 2 else [128, 1, 1]
    return len(tritons), (tritons[0][1] if len(tritons) >= 1 else ""), \
        (tritons[1][1] if len(tritons) >= 2 else ""), gather_type, gather_grid, \
        ew_present, ew_grid, t1_block, t2_block


def build(case_dir, device_id=0):
    nk, iters, bounds = load_nsys_all(case_dir, device_id=device_id)
    if not nk or not iters or bounds is None:
        return None
    by_step = _partition(nk, bounds, iters)
    # Per-case sampler_remap: argmax (MaxOps reduce) position. <0.5 = front
    # (remapped from prev step, dense/TP2) -> remap=True. >=0.5 = end (in-step,
    # 30B-A3B-EP2) -> remap=False (sampler stays in its own step).
    pos = []
    for it in iters:
        p = _argmax_pos(by_step.get(it, []))
        if p is not None:
            pos.append(p)
    import statistics
    med = statistics.median(pos) if pos else 0.0
    sampler_remap = med < 0.5
    out = []
    for idx, it in enumerate(iters):
        seq = by_step.get(it, [])
        if not seq:
            continue
        if sampler_remap:
            block = _sampler_block_front(seq)
            owner = iters[idx - 1] if idx > 0 else it
        else:
            block = _sampler_block_tail(seq)
            owner = it  # in-step: attribute to the lm_head's own step
        if not block:
            # still emit a (zero) record so operator_records knows the step
            n_tri, t1, t2, gt, gg, ew, eg, t1b, t2b = 0, "", "", "", [0, 0, 0], False, [0, 0, 0], [128, 1, 1], [128, 1, 1]
        else:
            n_tri, t1, t2, gt, gg, ew, eg, t1b, t2b = _extract_glue(block)
        out.append({
            "iter_id": owner,
            "sampler_remap": sampler_remap,
            "n_triton": n_tri,
            "triton1": t1,
            "triton2": t2,
            "triton1_block": t1b,
            "triton2_block": t2b,
            "gather_type": gt,
            "gather_grid": gg,
            "ew_present": ew,
            "ew_grid": eg,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--root", default=None,
                    help="Case dir root (default: A800 EXP_ROOT). For C500 use "
                         "serve_exp/eval_serving_sharegpt_c500.")
    a = ap.parse_args()
    exp_root = Path(a.root) if a.root else EXP_ROOT
    cases = [exp_root / a.case] if a.case else sorted(exp_root.iterdir())
    for cd in cases:
        if not (cd / "meta.json").exists():
            continue
        # Don't hard-filter on A800 — allow C500 (macaprof) cases too. The
        # partitioner (load_nsys_all) handles both sqlite (A800) and the
        # nsys_kernels_full.json + mctracer-sync fallback (C500).
        out = build(cd)
        if out is None:
            print(f"[sampler_glue] {cd.name}: SKIP")
            continue
        with open(cd / "sampler_glue.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        c = Counter(r["n_triton"] for r in out)
        print(f"[sampler_glue] {cd.name}: {len(out)} steps, n_triton dist {dict(c)}")


if __name__ == "__main__":
    main()
