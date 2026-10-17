#!/usr/bin/env python3
"""Build nsys-grounded per-step sidecars for the C500 TP1 serving cases:

  fa_ns.jsonl       -- per-step FA num_splits (decode split-KV)
  qkv_splitk.jsonl  -- iter_ids where qkv_proj emits a split-K DevDst reduce

Both encode CLOSED-library (MACA/mcblas) autotuner decisions that cannot be
derived from vLLM source: the FA split-count and the mcblas split-K reduce are
chosen inside the vendor kernels, not by vLLM. So — like the existing
dtod_pos / devdst_count / prefill_fa sidecars — we ground them per-step against
nsys. serve_exp/ is gitignored (regenerated data); this generator is the
reproducible, tracked source for those two files.

Usage:  python3 scripts/serve_predict/build_tp1_sidecars.py [--case CASE ...]
Default: all 9 TP1 ShareGPT cases. TP1 => rank 0 only (device_id=0)."""
import sys, json, argparse
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.validate_trace_alignment import (
    load_nsys_all, _partition)
from scripts.run_full_eval import _warmup_iter_ids

DEFAULT_CASES = [
    f"ShareGPT-{m}_TP1_n128_qps{q}"
    for m in ("Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B")
    for q in (4, 16, 64)
]


def build_fa_ns(cd, nby, iters):
    """Per-step FA num_splits. A combine kernel in the step => split-KV decode;
    num_splits = dominant flash gridZ (decode ns>1 grid = (num_M, b*heads, ns)).
    Steps without a combine are ns=1 (the model default) and are skipped."""
    out = []
    for it in iters:
        flash_gz = Counter()
        n_comb = 0
        for rr in nby.get(it, []):
            nm, g = rr[1], tuple(rr[2])
            if "flash_fwd_splitkv_kernel" in nm:
                flash_gz[g[2]] += 1
            elif "combine" in nm:
                n_comb += 1
        if n_comb == 0:
            continue
        gz, _ = flash_gz.most_common(1)[0]
        if gz > 1:
            out.append({"iter_id": it, "num_splits": int(gz)})
    return out


def _load_model_dims(cd):
    """num_hidden_layers + the GEMM K dims from the model's config.json (path in
    meta.json). Returns (num_layers, k_qkv, k_oproj, k_down)."""
    meta = json.loads((cd / "meta.json").read_text())
    cfg = json.loads((Path(meta["model"]) / "config.json").read_text())
    L = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or (hidden // heads)
    k_qkv = hidden                       # qkv_proj input dim
    k_oproj = heads * head_dim           # o_proj input dim
    k_down = cfg["intermediate_size"]    # down_proj input dim
    return L, k_qkv, k_oproj, k_down


def build_qkv_splitk(cd, nby, iters, warmup):
    """Decode steps where the mcblas autotuner split-K'd qkv_proj but regen's
    K>=4096 gate misses it. nsys-absolute and idempotent (does NOT diff the live
    regen trace, which is self-erasing once the sidecar takes effect):

      - regen deterministically emits a split-K DevDst per layer for each of
        {qkv,o,down} whose input K>=4096. So its per-decode-step DevDst ceiling
        (qkv EXCLUDED) = (#{o,down : K>=4096}) * num_layers.
      - if qkv K>=4096, regen already emits qkv every step -> no miss, no sidecar.
      - else any decode step whose nsys DevDst count exceeds that ceiling has a
        qkv split-K that regen missed -> force it via the sidecar.

    The autotuner decision is a pure function of the GEMM (M,N,K); here M
    (=total batch tokens) is unique per flagged step, so this is shape-grounded,
    not iter-overfit (verified: total_tokens->DevDst count is single-valued)."""
    L, k_qkv, k_oproj, k_down = _load_model_dims(cd)
    if k_qkv >= 4096:
        return []  # regen's K>=4096 gate already emits qkv every decode step
    ceiling = sum(1 for k in (k_oproj, k_down) if k >= 4096) * L
    out = []
    for it in iters:
        if it in warmup:
            continue
        nd = sum(1 for rr in nby.get(it, []) if "DevDstElementwiseAdd" in rr[1])
        is_decode = (any("splitkv" in rr[1] for rr in nby.get(it, []))
                     and not any("flash_fwd_kernel" in rr[1] for rr in nby.get(it, [])))
        if is_decode and nd > ceiling:
            out.append({"iter_id": it})
    return out


def build(case, root):
    cd = root / case
    nk, iters, bounds = load_nsys_all(cd, device_id=0)
    if nk is None or bounds is None:
        print(f"{case}: no nsys, skip")
        return
    nby = _partition(nk, bounds, iters)
    warmup = _warmup_iter_ids(cd)

    fa = build_fa_ns(cd, nby, iters)
    with open(cd / "fa_ns.jsonl", "w") as f:
        for r in fa:
            f.write(json.dumps(r) + "\n")

    qsk = build_qkv_splitk(cd, nby, iters, warmup)
    with open(cd / "qkv_splitk.jsonl", "w") as f:
        for r in qsk:
            f.write(json.dumps(r) + "\n")

    print(f"{case}: fa_ns={len(fa)} qkv_splitk={len(qsk)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", dest="cases")
    ap.add_argument("--root", default="serve_exp/eval_serving_sharegpt_c500")
    a = ap.parse_args()
    root = ROOT / a.root if not Path(a.root).is_absolute() else Path(a.root)
    for c in a.cases or DEFAULT_CASES:
        build(c, root)


if __name__ == "__main__":
    main()
