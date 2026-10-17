#!/usr/bin/env python3
"""Build per-step mask_pipeline.jsonl for 910B Qwen3-32B from msprof.

The 32B lm_head sampler runs a 12-kernel top-p/top-k mask pipeline between the
token-gather (IndexSelect+Repeat pair) and the next-input Embedding, then a
trailing aclnnInplaceMaskedFillScalar after the Embedding:

  [argmax] -> Fill [+ B>1 scatter] -> Subs -> IndexSelect, Repeat,
              [InplaceCopy_Slice if B>1] -> IndexSelect, Repeat,
              [InplaceCopy_Slice if B>1] ->
    GeScalar_Cast, GeScalar_GreaterEqual, LtScalar_Cast, LtScalar_Less,
    BitwiseAndTensor_LogicalAnd, Muls_Cast, Muls_Mul, Sub_Cast, Sub_Sub,
    Mul_Cast, Mul_Mul, BitwiseNot_LogicalNot   (the "mask12")
  -> Embedding -> InplaceMaskedFillScalar_MaskedFill

This is 32B-ONLY: Qwen3-4B/8B (TP1) and Llama-8B have ZERO mask kernels
(verified). It fires on EVERY decode step (incl. greedy/temperature=0, incl.
B=1) — the gate is NOT sampling_mode, it is the 32B capture's sampler
structure. The regen therefore emits the mask12 + trailing MaskedFill whenever
this sidecar is present (built only when msprof actually contains
aclnnGeScalar_GreaterEqual, so 4B/8B/Llama produce no file → no emit).

Per step we record, per kernel, the msprof-observed Block Dim (for phase-2
grid_mm). The 12 kernels are always contiguous and in this exact order right
before the embedding (verified 100% across all 3 32B qps cases, both ranks).

Writes <case_dir>/mask_pipeline.jsonl with {iter_id, present, bds} where
bds = [bd]*12 (mask12 order) + [bd_masked_fill]. present=False for steps
without the pipeline (prefill, or non-32B). Consumed by build_eval_dataset's
gen.set_step_mask_pipeline.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.validate_trace_alignment import (
    load_msprof_all, _partition_910b_by_fa)


# The 12 mask kernels in msprof order, immediately before the embedding.
MASK12 = [
    "aclnnGeScalar_CastAiCore_Cast",
    "aclnnGeScalar_GreaterEqual_GreaterEqual",
    "aclnnLtScalar_CastAiCore_Cast",
    "aclnnLtScalar_LessAiCore_Less",
    "aclnnBitwiseAndTensor_LogicalAndAiCore_LogicalAnd",
    "aclnnMuls_CastAiCore_Cast",
    "aclnnMuls_MulAiCore_Mul",
    "aclnnSub_CastAiCore_Cast",
    "aclnnSub_SubAiCore_Sub",
    "aclnnMul_CastAiCore_Cast",
    "aclnnMul_MulAiCore_Mul",
    "aclnnBitwiseNot_LogicalNotAiCore_LogicalNot",
]
EMBED_NAME = "aclnnEmbedding_GatherV2AiCore_GatherV2"
MASKEDFILL_NAME = "aclnnInplaceMaskedFillScalar_MaskedFillAiCore_MaskedFill"


def build(case_dir, device_id=0):
    case_dir = Path(case_dir)
    nk, iters, bounds = load_msprof_all(case_dir, device_id=device_id)
    if nk is None or not iters:
        return 0
    nby = _partition_910b_by_fa(nk, iters)
    out_path = case_dir / "mask_pipeline.jsonl"
    n = 0
    with open(out_path, "w") as f:
        for it in iters:
            entries = nby.get(it, [])
            # Find the step's embedding (sampler end). The mask12 is the 12
            # kernels immediately before it; MaskedFill is immediately after.
            emb_i = None
            for i, e in enumerate(entries):
                if len(e) > 1 and e[1] == EMBED_NAME:
                    emb_i = i      # last embedding wins (decode has 1; take last to be safe)
            if emb_i is None or emb_i < 12:
                f.write(json.dumps({"iter_id": it, "present": False,
                                    "bds": []}) + "\n")
                n += 1
                continue
            seq12 = [entries[emb_i - 12 + k][1] for k in range(12)
                     if len(entries[emb_i - 12 + k]) > 1]
            if seq12 != MASK12:
                # Not the mask-pipeline structure (e.g. prefill chunk) — skip.
                f.write(json.dumps({"iter_id": it, "present": False,
                                    "bds": []}) + "\n")
                n += 1
                continue
            bds = [int(entries[emb_i - 12 + k][2][0]) for k in range(12)]
            # trailing MaskedFill (immediately after the embedding)
            mf_bd = 1
            if emb_i + 1 < len(entries) and \
                    len(entries[emb_i + 1]) > 1 and \
                    entries[emb_i + 1][1] == MASKEDFILL_NAME:
                mf_bd = int(entries[emb_i + 1][2][0] or 1)
            bds.append(mf_bd)
            # B>1 scatter: some decode steps scatter the sampled token back via
            # aclnnIndex + InplaceScatter×2 right after Fill (before Subs). The
            # gate is not a clean sidecar signal (85/767 B>1 steps), so record
            # the per-step scatter presence + bds from msprof. The 3 kernels are
            # at [fill+1 .. fill+3] when present (Fill, Index, InplaceScatter,
            # InplaceScatter, Subs). Non-scatter: Fill, Subs directly.
            scatter = None
            # walk back from mask12[0] (GeScalar_Cast at emb_i-12) to find Fill
            fill_i = None
            for k in range(emb_i - 13, max(-1, emb_i - 30), -1):
                if k >= 0 and len(entries[k]) > 1 and \
                        entries[k][1] == "aclnnInplaceFillScalar_FillAiCore_Fill":
                    fill_i = k
                    break
            if fill_i is not None and fill_i + 3 < emb_i:
                nxt = [entries[fill_i + 1 + j][1] for j in range(3)
                       if len(entries[fill_i + 1 + j]) > 1]
                if (len(nxt) >= 3 and nxt[0] == "aclnnIndex_IndexAiCore_Index"
                        and "InplaceScatter_LinearIndex" in nxt[1]
                        and "InplaceScatter_ScatterElements" in nxt[2]):
                    scatter = {
                        "bds": [int(entries[fill_i + 1 + j][2][0] or 1)
                                for j in range(3)]}
            rec = {"iter_id": it, "present": True, "bds": bds}
            if scatter is not None:
                rec["scatter"] = scatter
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    a = ap.parse_args()
    n = build(a.case_dir, a.device_id)
    print(f"[mask_pipeline] wrote {n} steps -> {a.case_dir}/mask_pipeline.jsonl")


if __name__ == "__main__":
    main()
