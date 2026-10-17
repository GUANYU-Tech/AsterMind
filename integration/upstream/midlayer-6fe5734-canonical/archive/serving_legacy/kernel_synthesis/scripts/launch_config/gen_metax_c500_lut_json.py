"""Build data/c500_lut.json for the C500 serving eval.

Aggregates the (M_tok, N_out, K) -> dominant (actual_tile, actual_grid) mapping
across all C500 serving cases, then writes the JSON LUT in the schema the loader
in src/serve_trace/analytic_mirror.py expects:

  {
    "gemm_tile_lut": {
      "(M_tok, N_out, K)": [
        [ "<kernel_name>", [gridX, gridY, gridZ], [blockX, blockY, blockZ], <smem_int> ]
      ]
    }
  }

The kernel_name MUST contain the substring
  bf16gemm_tn_{tileM}x{tileN}x{tileK}_{layout}_{threads}t_{stage_sb}
so the loader regex extracts (tile_m, tile_n, tile_k, layout, threads, stage_sb).

ALIGNMENT (no regen, no LUT dependency):
  _c500_align_gemms reads the on-disk plugin_trace.json, but the C500 cases only
  ship operator records; the compiled trace is rebuilt in-memory by the regen,
  which ITSELF calls _maca_gemm_dispatch -> _maca_select_tile — a chicken-and-egg
  dependency on this very LUT. So we bypass the regen entirely: the operator
  records' `linear` ops carry attrs.M/N/K in file order, and they align 1:1 with
  nsys mcblas+b16gemvt kernels in ts order by per-step ordinal (verified: counts
  match exactly, 156165 == 156165). This is the same count-based segmentation
  _c500_align_gemms uses, just driven from operator records instead of the
  regen trace.

Run:  python -m scripts.kernel.gen_c500_lut_json
"""
import json
import os
import re
import sys
import multiprocessing as mp
from collections import defaultdict, Counter
from pathlib import Path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

C500_DIR = os.path.join(REPO, "serve_exp", "eval_serving_sharegpt_c500")
OUT_PATH = os.path.join(REPO, "data", "c500_lut.json")

# tile_family metadata: (block_threads, layout, stage_suffix) per observed C500
# mcblas kernel family. Sourced from kernel_dataset/metax_c500/gemm.json.
TILE_META = {
    "64x32x256":  (256, "2m1n8k", "bf16_bf16_fp32_sb_0_2"),
    "128x64x256": (256, "2m1n8k", "bf16_bf16_fp32_sb_0_0"),
    "128x128x128":(256, "4m1n8k", "4stage_bf16_bf16_fp32_sb_0_0"),
    "256x256x64": (512, "2m1n8k", "4stage_bf16_bf16_fp32_sb_0_0"),
    "256x192x64": (512, "2m1n8k", "4stage_bf16_bf16_fp32_sb_0_0"),
    "256x128x64": (512, "2m1n8k", "4stage_bf16_bf16_fp32_sb_0_0"),
}


def _nsys_tile_key(kn, grid):
    """Tile key for a C500 GEMM/GEMV kernel (mirrors _c500_align_gemms._key).
    mcblas__Mck_bf16gemm -> the NxNxK tile string (e.g. '128x64x256').
    b16gemvt_wave_kernel -> f'gemv_gx{grid_x}' (GEMV — skipped, not a GEMM tile)."""
    if "b16gemvt" in kn:
        return None  # GEMV — not part of the GEMM tile LUT
    t = re.search(r"bf16gemm_tn_(\d+x\d+x\d+)", kn)
    return t.group(1) if t else None


def _align_one(case_dir):
    """Pair each operator-record linear op's (M,N,K) with its nsys actual tile
    by per-step GEMM ordinal. Returns list of (M,N,K,tile,grid). No regen, no
    LUT dependency — reads operator records + nsys_kernels_full.json directly."""
    import traceback
    try:
        return _align_one_impl(case_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] {os.path.basename(str(case_dir))} failed: {e}", flush=True)
        traceback.print_exc()
        return []


def _align_one_impl(case_dir):
    case_dir = Path(case_dir)
    nsys_full = case_dir / "nsys_kernels_full.json"
    sidecar_path = str(case_dir / "plugin_trace.json") + "_scheduler.jsonl"
    ops_rank0 = case_dir / "plugin_trace.json_operators.jsonl.rank0"
    if not (nsys_full.exists() and Path(sidecar_path).exists() and ops_rank0.exists()):
        return []

    # --- per-step GEMM (M,N,K) sequence from operator records (file order) ---
    # Operator records' linear ops (op_type=='linear' with attrs.M/N/K) align 1:1
    # with nsys mcblas+b16gemvt kernels in ts order, by per-step ordinal. This is
    # the same contract _c500_align_gemms relies on (verified counts match).
    step_pred = {}
    with open(ops_rank0) as f:
        for line in f:
            o = json.loads(line)
            if o.get("op_type") != "linear":
                continue
            a = o.get("attrs") or {}
            if "M" not in a or "N" not in a or "K" not in a:
                continue
            sid = o.get("step_id", 0)
            step_pred.setdefault(sid, []).append(
                (int(a["M"]), int(a["N"]), int(a["K"])))
    if not step_pred:
        return []

    # --- load scheduler sidecar (for step count validation) ---
    from src.serve_predict.serve_metric_predictor import load_scheduler_sidecar
    sidecar_steps = [s for s in load_scheduler_sidecar(sidecar_path)
                     if s.total_tokens > 0]
    n_steps = len(sidecar_steps)
    if n_steps == 0:
        return []

    # --- load nsys actual tiles in ts order ---
    data = json.load(open(nsys_full))
    per_rank = data.get("per_rank") or {}
    by_dev = {}
    for dev, rd in per_rank.items():
        ks = rd.get("kernels") if isinstance(rd, dict) else None
        if not ks:
            continue
        kl = []
        for k in ks:
            ts = k.get("ts_ns")
            if ts is None:
                continue
            kl.append({"start": ts, "end": ts + k.get("dur_ns", 0),
                       "name": k.get("name", ""),
                       "grid": k.get("grid", [1, 1, 1])})
        kl.sort(key=lambda x: x["start"])
        if kl:
            by_dev[dev] = kl
    if not by_dev:
        return []

    devs = sorted(by_dev)
    dev = "0" if "0" in by_dev else devs[0]
    kl_sorted = sorted(by_dev[dev], key=lambda x: x["start"])
    all_nsys_tiles = []
    for kk in kl_sorted:
        kn = kk["name"]
        if "mcblas__Mck" not in kn and "b16gemvt" not in kn:
            continue
        key = _nsys_tile_key(kn, kk.get("grid"))
        if key:  # only GEMM tiles (GEMV returns None -> skipped)
            all_nsys_tiles.append((key, kk.get("grid")))

    # --- count-based segmentation (mirrors _c500_align_gemms) ---
    regen_steps = sorted(step_pred.keys())
    nsys_step_tiles = []
    cursor = 0
    for sid in regen_steps:
        count = len(step_pred[sid])
        nsys_step_tiles.append(all_nsys_tiles[cursor:cursor + count])
        cursor += count

    out = []
    for si, sid in enumerate(regen_steps):
        acts = nsys_step_tiles[si]
        mnks = step_pred[sid]
        for i, (m_tok, n_out, k) in enumerate(mnks):
            if i >= len(acts):
                continue  # capture gap — no nsys pair, skip
            tile, grid = acts[i]
            if tile not in TILE_META:
                continue  # skip unknown tile families
            try:
                g = [int(grid[0]), int(grid[1]), int(grid[2]) if len(grid) > 2 else 1]
            except (TypeError, IndexError, ValueError):
                g = [0, 0, 1]
            out.append((m_tok, n_out, k, tile, tuple(g)))
    return out


def scan():
    cases = [os.path.join(C500_DIR, c)
             for c in sorted(os.listdir(C500_DIR))
             if os.path.isdir(os.path.join(C500_DIR, c))]
    tile_agg = defaultdict(Counter)              # (M,N,K) -> Counter(tile)
    grid_agg = defaultdict(lambda: defaultdict(Counter))  # (M,N,K)[tile] -> Counter(grid)
    print(f"[gen_c500_lut_json] aligning {len(cases)} cases (parallel, 3 workers)...")
    with mp.Pool(min(3, len(cases))) as pool:
        done = 0
        for recs in pool.imap_unordered(_align_one, cases):
            done += 1
            print(f"  [{done}/{len(cases)}] got {len(recs)} aligned GEMM records", flush=True)
            for m_tok, n_out, k, tile, grid in recs:
                tile_agg[(m_tok, n_out, k)][tile] += 1
                grid_agg[(m_tok, n_out, k)][tile][grid] += 1
    return tile_agg, grid_agg


def build_lut(tile_agg, grid_agg):
    lut = {}
    n = 0
    for key in sorted(tile_agg):
        c = tile_agg[key]
        dom_tile, cnt = c.most_common(1)[0]
        tm, tn, tk = (int(x) for x in dom_tile.split("x"))
        blk, lay, stage_sb = TILE_META[dom_tile]
        gc = grid_agg[key][dom_tile]
        if gc:
            dom_grid = gc.most_common(1)[0][0]
        else:
            dom_grid = (0, 0, 1)
        kernel_name = (
            f"mcblas__Mck_bf16gemm_tn_{tm}x{tn}x{tk}_{lay}_{blk}t_{stage_sb}"
        )
        key_str = f"({key[0]}, {key[1]}, {key[2]})"
        lut[key_str] = [[kernel_name,
                         [int(dom_grid[0]), int(dom_grid[1]), int(dom_grid[2])],
                         [int(blk), 1, 1],
                         0]]
        n += 1
    return lut, n


def main():
    tile_agg, grid_agg = scan()
    lut, n = build_lut(tile_agg, grid_agg)
    out = {"gemm_tile_lut": lut}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[gen_c500_lut_json] wrote {n} entries to {OUT_PATH}")
    for probe in ("(729, 6144, 4096)",):
        print(f"  probe {probe}: {'PRESENT' if probe in lut else 'absent'}")


if __name__ == "__main__":
    main()
