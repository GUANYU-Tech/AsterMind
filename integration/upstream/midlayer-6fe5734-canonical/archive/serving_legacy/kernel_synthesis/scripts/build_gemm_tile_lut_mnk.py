#!/usr/bin/env python3
"""Build a K-aware cuBLAS tile LUT for A800 from real data.

The tile/grid a cuBLAS autotuner picks is a function of (M, N, K, dtype,
layout, hw). For fixed input the autotuner is deterministic (modulo the rare
genuine cross-run tie). The previous LUT keyed only on (M, N) and dropped K,
so two projections sharing (M, N) but differing in K (e.g. Llama-70B TP4:
o_proj M=768 N=8192 K=2048 -> 128x256; down_proj M=768 N=8192 K=7168 ->
256x128) collided into one dominant tile and the other was flagged a mismatch.

This builder reconstructs the (M, N, K) -> dominant-tile mapping directly from
the serving operator records (real M/N/K per linear op, captured by the
platform plugin) correlated with the nsys-captured kernel name (which encodes
the tile). Correlation is by global stream order (the regenerated trace IS the
operator-record -> kernel synthesis, in step order), keeping only pairs whose
nsys grid is consistent with the op-record (M, N) under the chosen tile
(self-consistency, tile-orientation-agnostic).

  dtype = bf16, layout = tn (the serving path; vLLM dense projections run
  ampere_bf16_s16816gemm_bf16_*_tn on A800). These are FIXED for this LUT, so
  they are not separate key dimensions — but the principle still holds: the
  ncu_dataset_a800 microbenchmark is fp16 (ampere_fp16_*_tn), a DIFFERENT
  dtype, so it is NOT merged here. Mixing fp16 (offline) and bf16 (serving)
  tiles under one (M,N,K) key was the original conflation bug — the autotuner
  is a function of dtype too.

Key = (M_tok_padded64, N_out_padded64, K). Unseen (M, N, K) at query time ->
AssertionError (no silent fallback).

Outputs:
  src/serve_trace/gemm_tile_lut_a800.json   {"key":[tm,tn], ...}
  + a mismatch report: NEW-LUT tile vs nsys actual tile per GEMM kernel (the
    goal's validation), and a coverage check.

Usage:
  python scripts/serve_predict/build_gemm_tile_lut_mnk.py
"""
import json
import os
import re
import sys
import gzip
import pickle
import glob
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXP_ROOT = ROOT / "serve_exp" / "eval_serving_sharegpt"
NCU_GEMM = ROOT / "kernel_dataset" / "a800" / "gemm.json"
LUT_OUT = ROOT / "src" / "serve_trace" / "gemm_tile_lut_a800.json"

TILE_RE = re.compile(r'(\d+)x(\d+)')


def parse_tile(name):
    """Parse (tile_m, tile_n) from an ampere/cutlass s16816 kernel name.

    ampere_fp16_s16816gemm_fp16_256x128_ldg8...  -> (256, 128)
    ampere_bf16_s16816gemm_bf16_128x256_ldg8...  -> (128, 256)
    cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x6_tn -> (128, 64)
    Returns (None, None) if no tile dims found.
    """
    m = TILE_RE.search(name or "")
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 32 <= a <= 512 and 32 <= b <= 512:
            return a, b
    return None, None


def pad64(x):
    return ((x + 63) // 64) * 64 if x else 0


def consistent(N, M, ntm, ntn, gx, gy, gz=1):
    """True if nsys grid is self-consistent with op-record (N, M) for tile
    (ntm, ntn). cuBLAS emits BOTH 2D grids (gx=ceil(N/ntm), gy=ceil(M/ntn))
    and 1D flattened grids (gy=1, gx=ceil(N/ntm)*ceil(M/ntn), gz=split-K), so
    accept either. This is only a sanity filter — per-step ORDER alignment
    (regen[i] <-> nsys[i] within a step) is the primary correctness mechanism."""
    import math
    if not ntm or not ntn or not gx:
        return False
    # 2D grid.
    if (N <= gx * ntm < N + ntm) and (M <= gy * ntn < M + ntn):
        return True
    # 1D flattened grid (gy==1; gz is a K-split, ignored).
    if gy == 1 and gx == math.ceil(N / ntm) * math.ceil(M / ntn):
        return True
    return False


def _is_tc_dispatch(m_tok, n_out, k):
    """Replicate _cublas_gemm_dispatch's TC-vs-GEMV decision for an A800 dense
    linear op (cuBLAS args: m=N_out, n=M_tok, k=K). Prefill (M_tok>1) is always
    TC; single-token decode (M_tok<=1) is GEMV for the narrow/small-K band and
    TC otherwise. This selects exactly the matmuls the recipe emits as s16816,
    so the operator-record TC set aligns with nsys's s16816 stream (modulo the
    rare xmma divergence, bounded per step)."""
    if m_tok > 1:
        return True
    # decode M_tok<=1: GEMV band -> not TC
    if (n_out <= 2048 and k <= 4096) or (n_out <= 5120 and k <= 2048):
        return False
    return True


def load_regen(case_dir):
    """Return list of (step_id, M, N, K) for TC-dispatched dense linear ops
    from the rank-0 operator records, ordered by (step_id, file order = op_idx).

    Reads the operator records DIRECTLY (the real M/N/K the platform plugin
    captured — the source the goal names) and filters to the TC-dispatched
    subset via _is_tc_dispatch, which equals the recipe's s16816 emission set.
    This (a) is faithful to the goal, (b) is independent of the regen cache
    (absent for never-eval'd cases, stale after a code change), (c) is far
    cheaper for MoE cases (no recipe expansion into millions of kernels), and
    (d) keeps the TC-only property that aligns with nsys s16816. step_id = the
    vLLM forward-step index == scheduler-sidecar iter_id, the join key to nsys's
    per-step partition."""
    rf = case_dir / "plugin_trace.json_operators.jsonl.rank0"
    if not rf.exists():
        cands = sorted(case_dir.glob("*_operators.jsonl.rank*"))
        if not cands:
            return None
        rf = cands[0]
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
            M, N, K = a.get("M"), a.get("N"), a.get("K")
            if M is None or N is None or K is None:
                continue
            if not _is_tc_dispatch(int(M), int(N), int(K)):
                continue
            out.append((r.get("step_id"), int(M), int(N), int(K)))
    return out


def _load_sidecar(case_dir):
    """Read the scheduler sidecar -> (sorted_iters, ts_us_list).

    Each sidecar line is {iter_id, ts_us, ...} (ts_us = epoch-µs wall time at
    the step boundary). Returns iters sorted ascending with matching ts_us."""
    p = case_dir / "plugin_trace.json_scheduler.jsonl"
    if not p.exists():
        return None, None
    entries = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            it = r.get("iter_id")
            ts = r.get("ts_us")
            if it is not None and ts:
                entries.append((int(it), int(ts)))
    entries.sort()
    return [e[0] for e in entries], [e[1] for e in entries]


def _idle_iter_set(case_dir):
    """Set of sidecar iter_ids that scheduled NO tokens (total_tokens==0).

    An idle engine step runs no model forward, so it emits no per-step
    cudaStreamSynchronize / mcEventSynchronize and the regen recipe produces no
    kernels for it. It must therefore NOT occupy a partition slot: the sync
    stream is 1:1 with the FORWARD-executing (non-idle) iters only. A mid-stream
    idle iter left in the positional sync<->iter mapping shifts every later step
    by one (masked in homogeneous decode regions, surfaces as a full-step
    type_mm at the next prefill/compose-change boundary)."""
    p = case_dir / "plugin_trace.json_scheduler.jsonl"
    if not p.exists():
        return set()
    idle = set()
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                it = r.get("iter_id")
                if it is not None and r.get("ts_us") and not r.get("total_tokens", 0):
                    idle.add(int(it))
    except Exception:
        return set()
    return idle


def _load_nsys_full_json(case_dir, device_id=0):
    """Fallback nsys loader for cards whose capture is exported as the
    pre-extracted nsys_kernels_full.json instead of nsys_profile.sqlite
    (C500/macaprof: no sqlite shipped; the JSON carries the same fields).

    Returns the start-ordered kernel list for `device_id`, or None when the
    JSON is absent. Each entry mirrors the sqlite query shape:
      (ts_ns, name, (gx, gy, gz), (bx, by, bz), smem).

    Handles TWO formats:
    1. per_rank format (Qwen3-4B/8B/Llama-8B): {per_rank: {0: {kernels: [...]}}
       with ts_ns per kernel.
    2. flat-list format (Qwen3-32B-TP2): {kernels: [...]} without ts_ns —
       all kernels loaded as rank 0 with ts_ns=0 (count-based partition)."""
    p = case_dir / "nsys_kernels_full.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    pr = d.get("per_rank") or {}
    key = str(device_id) if str(device_id) in {str(k) for k in pr} else next(iter(pr), None)
    if key is None:
        # Flat-list format (32B-TP2): no per_rank, load all as rank 0.
        ks = d.get("kernels") or []
    else:
        rank = pr[key]
        ks = rank["kernels"] if isinstance(rank, dict) else rank
    out = []
    for e in ks:
        g = e.get("grid") or [0, 0, 0]
        b = e.get("block") or [0, 0, 0]
        out.append((e.get("ts_ns", 0), e.get("name", ""),
                    (g[0], g[1], g[2]), (b[0], b[1], b[2]),
                    e.get("smem", 0)))
    out.sort(key=lambda x: x[0])
    return out


def _nsys_bounds(case_dir, device_id=None):
    """Return (nsys0_ns, last_end_ns) — the wall-time window used to anchor
    sidecar step boundaries into nsys time.

    nsys0 = start of the first REAL model kernel (vllm::rms_norm), NOT the
    absolute first kernel: capture includes a warmup step whose sampler tail
    (CUDAFill/FillFunctor/direct_copy/gather) bleeds into the window before the
    first active step's rms_norm. Anchoring at rms_norm skips that warmup
    residue so step 0 (prefill) aligns regen (which starts at rms_norm).

    device_id: when set, anchor to that device's kernels (multi-rank).

    Anchor choice:
    - TP (nccl all-reduce present): first nccl kernel — TP steps START with
      all-reduce (the previous step's reduce), so step 0's first kernel is the
      warmup-window's all-reduce. Anchoring there keeps it in step 0.
    - TP1 (no nccl): first rms_norm — skips the warmup sampler tail
      (CUDAFill/FillFunctor/direct_copy/gather) that bleeds before step 0's
      rms_norm (regen has no warmup, so align at the first real model kernel)."""
    import sqlite3
    p = case_dir / "nsys_profile.sqlite"
    if not p.exists():
        # JSON fallback (C500): derive the anchor + last_end from the
        # pre-extracted nsys_kernels_full.json using the same name anchors.
        ks = _load_nsys_full_json(case_dir, device_id if device_id is not None else 0)
        if not ks:
            return None, None
        nsys0 = None
        for st, nm, _g, _b, _s in ks:
            if "cross_device_reduce" in nm:
                nsys0 = st; break
        if nsys0 is None:
            for st, nm, _g, _b, _s in ks:
                if "rms_norm_kernel" in nm or "fused_add_rms_norm" in nm:
                    nsys0 = st; break
        if nsys0 is None and ks:
            nsys0 = ks[0][0]
        # last_end = max(ts + dur). dur_ns lives in nsys_kernels.json (per-kernel),
        # but nsys_kernels_full.json has no dur — approximate last_end as max(ts).
        last_end = max(st for st, *_ in ks)
        return nsys0, last_end
    conn = sqlite3.connect(str(p))
    dev = int(device_id) if device_id is not None else 0
    # Anchor at the first cross_device_reduce (the leading all-reduce at the
    # FRONT of step 0) for TP cases; else first rms_norm (TP1). The leading
    # all-reduce (prev step's down_proj reduce) opens every TP step, so step 0
    # (the first prefill) opens with the warmup's down reduce. Anchoring there
    # puts the FULL prefill (leading reduce + body + lm_head AllGather+cast) in
    # win0, matching regen (which emits the leading all_reduce as op[0]).
    # Earlier anchors were wrong: first nccl AllGather = the lm_head AllGather
    # at the prefill's END (skips the prefill body, 1-kernel split mismatch);
    # first rms_norm = the input_layernorm (skips the leading all_reduce that
    # regen emits as op[0] -> full-cascade mismatch). The leading all_reduce is
    # the step's true first kernel for TP. Decode steps' leading reduces land
    # in their own sync windows (after sync_end[i-1]), unaffected by nsys0.
    anchor_row = conn.execute(
        f"SELECT MIN(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        f"JOIN StringIds s ON k.demangledName=s.id "
        f"WHERE k.deviceId={dev} AND s.value LIKE '%cross_device_reduce%'").fetchone()
    if anchor_row is None or anchor_row[0] is None:
        # TP1 (no cross_device_reduce): first rms_norm (skip warmup sampler tail)
        anchor_row = conn.execute(
            f"SELECT MIN(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            f"JOIN StringIds s ON k.demangledName=s.id "
            f"WHERE k.deviceId={dev} AND (s.value LIKE '%rms_norm_kernel%' "
            f"OR s.value LIKE '%fused_add_rms_norm%')").fetchone()
    nsys0 = anchor_row[0] if anchor_row and anchor_row[0] is not None else conn.execute(
        f"SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId={dev}"
        ).fetchone()[0]
    last_end = conn.execute(
        f"SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId={dev}"
        ).fetchone()[0]
    conn.close()
    return nsys0, last_end



def _ts_boundaries(sidecar_ts_us, nsys0_ns, last_end_ns):
    """nsys-time step boundaries (ns) from sidecar µs timestamps (verbatim logic
    from build_eval_dataset._ts_boundaries_from_sidecar). The capture is
    bracketed to start at the first active step, so nsys0 anchors there; the
    constant scheduler->GPU dispatch lag cancels in the deltas."""
    ts0 = sidecar_ts_us[0]
    bounds = [nsys0_ns + (t - ts0) * 1000.0 for t in sidecar_ts_us]
    bounds.append(max(last_end_ns + 1.0, bounds[-1] + 1.0))
    return bounds


def _mctracer_sync_se(case_dir, device_id=0):
    """C500 fallback for _sync_boundaries when nsys_profile.sqlite is absent:
    extract per-step sync completion (start, end) timestamps from the macaprof
    (mctracer) chrome-JSON trace.

    mctracer records mcEventSynchronize (≈1/step, the vLLM per-step sync — plays
    the role of cudaStreamSynchronize in nsys) plus a final mcDeviceSynchronize
    (the engine's last torch.cuda.synchronize), with ts+dur on the same ns clock
    as the kernel events, so its `.end` is the step's physical completion
    boundary (no ts extrapolation, no CPU↔GPU dispatch drift). Falls back to
    mcStreamSynchronize when no Event/Device sync is present.

    Single-rank (TP1) only for now: returns ALL mcEventSynchronize events (mctracer
    on a 1-card run emits one process). For multi-rank TP/EP the per-rank split
    needs tid->rank mapping (TODO). Built once, cached to
    <case_dir>/.mctracer_sync_ends_<rank>.json (serve_exp/ is gitignored)."""
    import glob, json, os, re
    cd = Path(case_dir)
    rank = device_id if device_id is not None else 0
    cache = cd / f".mctracer_sync_ends_{rank}.json"
    if cache.exists():
        try:
            data = json.load(open(cache))
            if data:
                return [tuple(x) for x in data]
        except Exception:
            pass
    mdir = cd / "mctracer_out"
    cands = sorted(glob.glob(str(mdir / "mctracer-*.json")),
                   key=os.path.getsize, reverse=True)
    # Keep only the real per-rank traces (the large files; the small mctracer-*.json
    # are init/metadata shards). Multi-rank TP/EP: one large file per rank, keyed
    # by the dominant `device_id` in its kernel events. Pick the file whose
    # device_id == rank (sample first ~6MB) so each rank's syncs come from its
    # own trace. Falls back to the largest file when only one rank exists.
    # TP1 (single large file): use cands[0] directly (verified 8B 100% align) —
    # the big-filter + device_id sampling only kicks in for multi-rank.
    big = [c for c in cands if os.path.getsize(c) > 50_000_000]
    if not big:
        big = cands
    if len(big) <= 1:
        p = cands[0] if cands else None
    else:
        p = big[0]
    if not p:
        return None
    if len(big) > 1 and device_id is not None:
        devp = re.compile(rb'"device_id":\s*(\d+)')
        best = None; best_score = -1
        for c in big:
            counts = {}
            with open(c, "rb", buffering=1 << 20) as f:
                sampled = 0
                for raw in f:
                    for m in devp.finditer(raw):
                        d = int(m.group(1))
                        counts[d] = counts.get(d, 0) + 1
                        sampled += 1
                    if sampled > 20000:
                        break
            score = counts.get(device_id, 0)
            if score > best_score:
                best, best_score = c, score
        if best is not None and best_score > 0:
            p = best
    TSPAT = re.compile(r'"ts":\s*(\d+)'); DUR = re.compile(r'"dur":\s*(\d+)')
    NAM = re.compile(r'"name":\s*"([^"]+)"')
    want = {"mcEventSynchronize", "mcDeviceSynchronize", "mcStreamSynchronize"}
    events = []  # (name, ts, dur)
    last_ts = last_dur = None
    with open(p, "rb", buffering=1 << 20) as f:
        for raw in f:
            line = raw.decode("utf-8", "replace")
            s = line.strip()
            if s in ("{", ",{"):
                last_ts = last_dur = None
            m = NAM.search(line)
            if m and m.group(1) in want and last_ts is not None:
                events.append((m.group(1), int(last_ts), int(last_dur or 0)))
            else:
                t = TSPAT.search(line)
                if t: last_ts = int(t.group(1))
                d = DUR.search(line)
                if d: last_dur = int(d.group(1))
    # Per-step syncs = mcEventSynchronize (1/step) + the final mcDeviceSynchronize.
    ev = [(e[1], e[1] + e[2]) for e in events
          if e[0] in ("mcEventSynchronize", "mcDeviceSynchronize")]
    if not ev:
        ev = [(e[1], e[1] + e[2]) for e in events if e[0] == "mcStreamSynchronize"]
    ev.sort()
    json.dump(ev, open(cache, "w"))
    return ev or None


def _sync_boundaries(case_dir, iters, device_id=None):
    """Structural step boundaries from per-step cudaStreamSynchronize completion
    timestamps — REPLACES the time-stamp extrapolation in _ts_boundaries.

    vLLM ends every step with a torch.cuda.synchronize() (-> cudaStreamSynchronize
    on the host) that waits for that step's GPU work to finish. The sync event's
    `.end` (ns, host clock; nsys calibrates runtime + kernel events to one clock)
    is therefore the step's physical completion boundary — no extrapolation, no
    CPU↔GPU dispatch drift. Verified on Qwen3-8B TP1: sync count == plugin step
    count; sync.end lags the step's last GPU kernel.end by a stable ~9-12µs;
    partitioning the full kernel stream yields 0 empty steps, 0 cross-boundary
    overlaps, vs the ts method's ~1.7% mis-assigned kernels.

    TP1 (single stream) only: one cudaStreamSynchronize per step, 1:1 with the
    sidecar iters. The sidecar may carry one extra trailing iter with no plugin
    ops (capture tail); drop it so len(sync) == len(active iters). Returns None
    when the runtime table is absent or the sync count does not match the iter
    count (±1) — callers fall back to _ts_boundaries (and multi-rank / MoE-EP,
    whose per-rank syncs and comm noise need a device-split variant, also fall
    back until that is built). Output format is identical to _ts_boundaries:
    [nsys0, sync_end[0], ..., sync_end[n-1], +inf].

    Multi-rank (TP/EP): cudaStreamSynchronize is per-rank-process (each rank
    syncs its own stream). device_id selects which rank's syncs to use: map
    each globalTid to its dominant deviceId (via cudaLaunchKernel->kernel
    deviceId), keep syncs whose globalTid owns device_id. nsys0 anchored at
    that device's first rms_norm."""
    import sqlite3
    p = case_dir / "nsys_profile.sqlite"
    sync_se = None
    from_mctracer = False
    if p.exists():
        conn = sqlite3.connect(str(p))
        try:
            if device_id is not None:
                # Map globalTid -> dominant deviceId via cudaLaunchKernel kernels.
                gt_dev = {}
                for gt, dev, cnt in conn.execute(
                        "SELECT r.globalTid, k.deviceId, COUNT(*) "
                        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                        "JOIN StringIds s ON r.nameId = s.id "
                        "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId "
                        "WHERE s.value LIKE 'cudaLaunchKernel%' "
                        "GROUP BY r.globalTid, k.deviceId").fetchall():
                    if gt_dev.get(gt, (0, 0))[1] < cnt:
                        gt_dev[gt] = (dev, cnt)
                owning_gts = {gt for gt, (dev, _) in gt_dev.items() if dev == device_id}
                if not owning_gts:
                    sync_se = None
                else:
                    ph = ",".join("?" * len(owning_gts))
                    sync_se = [tuple(r) for r in conn.execute(
                        f"SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                        f"JOIN StringIds s ON r.nameId = s.id "
                        f"WHERE s.value LIKE 'cudaStreamSynchronize%' "
                        f"AND r.globalTid IN ({ph}) ORDER BY r.start",
                        list(owning_gts)).fetchall()]
            else:
                sync_se = [tuple(r) for r in conn.execute(
                    "SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                    "JOIN StringIds s ON r.nameId = s.id "
                    "WHERE s.value LIKE 'cudaStreamSynchronize%' "
                    "ORDER BY r.start").fetchall()]
        except sqlite3.OperationalError:
            sync_se = None
        finally:
            conn.close()
    if not sync_se:
        # C500 fallback: macaprof (mctracer) per-step sync events (no sqlite).
        sync_se = _mctracer_sync_se(case_dir, device_id)
        from_mctracer = bool(sync_se)
        if sync_se:
            # mctracer captures process init (mcInit/mcGetDevice…) BEFORE the
            # first model kernel, so drop warmup syncs that complete before nsys0
            # (the first rms_norm / all-reduce) and any past last_end (capture
            # tail). This is the macaprof analog of the nsys capture window.
            nsys0_w, last_end_w = _nsys_bounds(case_dir, device_id)
            if nsys0_w is not None:
                sync_se = [se for se in sync_se if se[1] >= nsys0_w]
            if last_end_w is not None:
                sync_se = [se for se in sync_se if se[0] <= last_end_w]
    if not sync_se or not iters:
        return None
    # Idle engine steps (total_tokens==0) run no forward -> no sync, no regen
    # kernels. Drop them from the sync<->iter mapping so the sync stream aligns
    # to the FORWARD-executing iters only (a retained mid-stream idle iter shifts
    # every later step by one — see _idle_iter_set).
    idle = _idle_iter_set(case_dir)
    if idle:
        iters = [it for it in iters if it not in idle]
        if not iters:
            return None
    # EP multi-rank (MoE-EP): vLLM's custom_all_reduce issues ~192 signal
    # cudaStreamSynchronize per step (per cross_device_reduce signal phase), so
    # the raw sync count (~193x iters) breaks the 1:1 sync<->iter assumption.
    # The cudaStreamSynchronize events are vLLM source syncs: 192/step are EP
    # routing barriers (intra-forward), 1/step is the step-end (after the
    # sampler) = the forward-iteration boundary. Source-faithful partition: pick
    # the step-end sync (the LAST gap>3ms candidate before each ~193-sync gap —
    # it's followed by the next step's scheduler tick). This makes step i =
    # [forward i: 48 layers + MoE + sampler] (MoE+sampler in-step, matching the
    # source vLLM forward structure), so the regen emits MoE+sampler in-step
    # (sampler_remap=False). NOT the layer-0 routing barrier (the FIRST gap>3ms
    # candidate), which yields a 1-layer-shifted step inconsistent with vLLM.
    if len(sync_se) > 3 * len(iters):
        STEP_END_GAP_NS = 3_000_000
        cand_idx = [i for i in range(len(sync_se) - 1)
                    if sync_se[i + 1][0] - sync_se[i][1] >= STEP_END_GAP_NS]
        if not cand_idx:
            return None
        MIN_SYNC_SPACING = 180
        kept_idx = []
        for i in cand_idx:
            if kept_idx and i - kept_idx[-1] < MIN_SYNC_SPACING:
                continue
            kept_idx.append(i)
        kept = [sync_se[i][1] for i in kept_idx]
        if not (0.9 * len(iters) <= len(kept) <= 1.1 * len(iters)):
            return None
        sync_ends = kept
        ep_filtered = True
    else:
        sync_ends = [se[1] for se in sync_se]
        ep_filtered = False
    # TP1: 1 sync per step. Sidecar may carry a capture-tail iter beyond the last
    # sync (no plugin ops) — tolerate off-by-one, else bail to ts fallback. The EP
    # filtered sync count is approximate (custom_all_reduce signal noise), so it
    # skips the exact-match check (the +/-10% gate above is its tolerance).
    # C500 mctracer path tolerates ±2 (a couple chunked-prefill / capture-tail
    # iters may lack a per-step mcEventSynchronize); it then trims iters to the
    # sync count so iter[i] <-> sync_end[i].
    if not ep_filtered:
        # from_mctracer: the nsys/macaprof capture window can END BEFORE the run
        # finishes (ring-buffer full / profiler stopped early). The mctracer
        # syncs then cover a clean 1:1 PREFIX of the forward steps — sync[k] <->
        # iter[k] for the first len(sync_ends) iters — and every trailing iter
        # ran UNCAPTURED (verified on Llama-3.1-8B qps4/16/64: per-time-decile
        # sync counts == iter counts within the captured window; sync_end[k]-ts[k]
        # is a stable ~55ms CPU->GPU offset with no mid-stream jump; regen kernel
        # count for iters[:n] == captured nsys total to the kernel). So accept
        # ANY len(sync_ends) <= len(iters): iters_use = iters[:len(sync_ends)]
        # keeps the captured prefix and drops the uncaptured tail (symmetric to
        # the warmup/capture-tail skips in the alignment loop). The old 0.85
        # floor wrongly rejected sub-85% early-capture-end cases (qps64 = 65%),
        # forcing the drift-prone _ts_boundaries fallback and a whole-run
        # category cascade (405k type_mm). A mid-stream skipped step (which WOULD
        # break the prefix assumption) is not observed on any C500 mctracer
        # trace; if it occurred the sync_end[k]-ts[k] offset would ramp up at the
        # gap rather than stay flat.
        ok = (len(sync_ends) == len(iters) or len(sync_ends) == len(iters) - 1
              or (from_mctracer and 2 <= len(sync_ends) <= len(iters)))
        if not ok:
            return None
    # Drop the trailing iters that have no matching sync (capture tail), keeping
    # the leading len(sync_ends) iters so iter[i] <-> sync_end[i].
    iters_use = list(iters[:len(sync_ends)])
    nsys0, last_end = _nsys_bounds(case_dir, device_id)
    if nsys0 is None:
        return None
    # bounds[idx] = lo of step iters_use[idx] = completion of step idx-1 (sync_end
    # of the previous step); bounds[idx+1] = hi = sync_end[idx]. The first step's
    # lo is nsys0 (first GPU kernel = first step's first kernel); the last step's
    # hi is the last sync_end, after which any kernels are capture tail (excluded,
    # matching the verified sum=total-tail). Length = n_steps+1, identical shape
    # to _ts_boundaries.
    bounds = [nsys0] + sync_ends
    return bounds, iters_use


def load_nsys(case_dir):
    """Return (kernels, iters, boundaries) where kernels is a start-ordered list
    of (start_ns, nsys_tile, gx, gy) for s16816 GEMMs on device 0, and
    (iters, boundaries) are the per-step partition keys/windows from the
    scheduler sidecar. Falls back to nsys_kernels_full.json when sqlite is
    absent (then boundaries are None -> global-order correlation)."""
    import sqlite3
    sqlite_path = case_dir / "nsys_profile.sqlite"
    iters, ts_us = (None, None)
    boundaries = None
    if sqlite_path.exists():
        conn = sqlite3.connect(str(sqlite_path))
        # Pre-fetch the s16816 StringIds (small set), then select kernels with
        # demangledName IN (...). This uses the kernel-table index instead of a
        # LIKE-join full scan (the scan was minutes on the 70B/32B cases).
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM StringIds WHERE value LIKE '%s16816gemm%'").fetchall()]
        placeholders = ",".join("?" * len(ids))
        q = (f"SELECT k.start, s.value, k.gridX, k.gridY "
             f"FROM CUPTI_ACTIVITY_KIND_KERNEL k "
             f"JOIN StringIds s ON k.demangledName = s.id "
             f"WHERE k.demangledName IN ({placeholders}) AND k.deviceId = 0 "
             f"ORDER BY k.start")
        rows = conn.execute(q, ids).fetchall()
        conn.close()
        kernels = [(r[0], parse_tile(r[1]), r[2], r[3]) for r in rows]
        iters, ts_us = _load_sidecar(case_dir)
        nsys0, last_end = _nsys_bounds(case_dir)
        # Prefer the structural per-step cudaStreamSynchronize completion
        # boundaries (physical, no extrapolation drift) over ts extrapolation.
        # _sync_boundaries also returns the iters aligned to its sync count
        # (dropping any capture-tail sidecar iter); on the TP1 path that trimmed
        # iters MUST be returned so downstream zip stays iter↔boundary-aligned.
        sb = _sync_boundaries(case_dir, iters) if iters else None
        if sb is not None:
            boundaries, iters = sb
        elif iters and ts_us and nsys0 is not None:
            boundaries = _ts_boundaries(ts_us, nsys0, last_end)
        return kernels, iters, boundaries
    # Fallback: pre-extracted nsys_kernels_full.json (no per-step partition).
    p = case_dir / "nsys_kernels_full.json"
    if not p.exists():
        return None, None, None
    nk = json.load(open(p))["kernels"]
    kernels = []
    for k in nk:
        nm = k.get("name") or ""
        if "s16816gemm" not in nm:
            continue
        g = k.get("grid") or [0, 0, 0]
        kernels.append((0, parse_tile(nm), g[0], g[1]))
    return kernels, None, None


def _partition_nsys_by_step(kernels, boundaries, iters):
    """Partition start-ordered nsys kernels into per-step lists keyed by sidecar
    iter_id. Returns {iter_id: [kernels...]}. kernels without boundaries (no
    sidecar) -> a single synthetic 'global' step."""
    if boundaries is None or not iters:
        return {None: list(kernels)}
    by_step = {it: [] for it in iters}
    ki = 0
    n = len(kernels)
    for idx, it in enumerate(iters):
        lo = boundaries[idx]
        hi = boundaries[idx + 1] if idx + 1 < len(boundaries) else float("inf")
        while ki < n and kernels[ki][0] < lo:
            ki += 1
        j = ki
        while j < n and kernels[j][0] < hi:
            by_step[it].append(kernels[j])
            j += 1
        ki = j
    return by_step


def correlate_serving(case_dir):
    """Correlate op-record (M,N,K) with nsys tiles for one case, PER STEP.

    The operator-record TC ops (with real M/N/K and step_id) and the nsys s16816
    kernels (with the actual tile) are aligned per step: the scheduler sidecar
    bridges nsys time -> step iter_id (the same vLLM step index the operator
    record carries as step_id), so each step's TC-op sequence zips with that
    step's nsys kernel sequence in execution order. Per-step (not global)
    alignment is required because the recipe's TC set can differ from nsys's
    s16816 set (the recipe emits s16816 where nsys ran an xmma kernel, or a TC
    where nsys ran a GEMV), which makes the two streams different lengths and
    would drift a global zip — but prefill steps (where K actually changes the
    tile) have no GEMV, so they align cleanly and the K-collisions resolve.

    Returns (tile_dist, pairs, all_shapes) or (None, None, None):
      tile_dist: {(Mp,Np,K): Counter((ntm,ntn))}  over CONSISTENT pairs
      pairs: list of (Mp, Np, K, nsys_tile) for every consistent pair
      all_shapes: set of (Mp,Np,K) the eval will query (from operator records)
    """
    regen = load_regen(case_dir)
    nsys_kernels, iters, boundaries = load_nsys(case_dir)
    if regen is None or nsys_kernels is None:
        return None, None, None
    # regen grouped by step_id (in op order within each step).
    regen_by_step = defaultdict(list)
    for step_id, M, N, K in regen:
        regen_by_step[step_id].append((M, N, K))
    nsys_by_step = _partition_nsys_by_step(nsys_kernels, boundaries, iters)
    tile_dist = defaultdict(Counter)
    pairs = []
    all_shapes = set()
    # Join on iter_id == step_id, but ONLY for steps the nsys capture actually
    # covers (nsys is bracketed to start after warmup, so operator records
    # contain pre-bracket warmup steps nsys never captured — their shapes are
    # NOT queried at eval time and must not count toward coverage). A step is
    # in-window iff its nsys partition is non-empty.
    if boundaries is not None:
        steps = [s for s in iters
                 if regen_by_step.get(s) and nsys_by_step.get(s)]
    else:
        steps = [None]
        regen_by_step[None] = [v for step_id in regen_by_step
                               for v in regen_by_step[step_id]]
    for step in steps:
        rseq = regen_by_step.get(step, [])
        nseq = nsys_by_step.get(step, [])
        n = min(len(rseq), len(nseq))
        for i in range(n):
            M, N, K = rseq[i]
            _start, (ntm, ntn), gx, gy = nseq[i]
            Mp, Np = pad64(M), pad64(N)
            all_shapes.add((Mp, Np, K))
            if consistent(N, M, ntm, ntn, gx, gy):
                tile_dist[(Mp, Np, K)][(ntm, ntn)] += 1
                pairs.append((Mp, Np, K, (ntm, ntn)))
    return tile_dist, pairs, all_shapes


def load_lut(path=LUT_OUT):
    """Load the K-aware LUT JSON -> {(Mp,Np,K): (tile_m, tile_n)}."""
    if not Path(path).exists():
        raise FileNotFoundError(f"gemm tile LUT not found: {path}; run "
                                f"build_gemm_tile_lut_mnk.py first")
    raw = json.load(open(path))
    lut = {}
    for k, v in raw.items():
        Mp, Np, K = k.strip("[]").split(",")
        lut[(int(Mp), int(Np), int(K))] = tuple(v)
    return lut


def validate_gemm_tile_lut(case_dir, lut=None, verbose=True):
    """The goal's validation, per case: for each nsys GEMM kernel, look up the
    LUT with the op-record (M,N,K) and compare to the nsys actual tile.

    Prints mismatches and returns (n_pairs, n_mismatch, n_nondeterministic).
    A mismatch where the nsys tile is NOT one of the tiles observed for that
    (M,N,K) is a genuine surprise (LUT wrong / coverage gap); a mismatch where
    it IS an observed tile is just cuBLAS autotuner nondeterminism (the LUT's
    dominant tile differs from this one occurrence) — irreducible, not a bug.
    """
    if lut is None:
        lut = load_lut()
    tile_dist, pairs, _ = correlate_serving(case_dir)
    if not pairs:
        if verbose:
            print(f"  [tile-validate] {case_dir.name}: no op-record/nsys pairs")
        return 0, 0, 0
    mism = 0
    nondet = 0
    surprises = []
    for Mp, Np, K, ntile in pairs:
        ltile = lut.get((Mp, Np, K))
        if ltile is None:
            continue  # uncovered shapes are reported by the builder's coverage check
        if ltile != ntile:
            mism += 1
            if ntile in tile_dist[(Mp, Np, K)]:
                nondet += 1
            elif len(surprises) < 12:
                surprises.append((Mp, Np, K, ltile, ntile))
    n = len(pairs)
    if verbose:
        print(f"  [tile-validate] {case_dir.name}: {n} kernels | "
              f"mismatch {mism} ({100*mism/max(1,n):.2f}%) | "
              f"of which nsys-nondeterministic {nondet}, genuine surprises "
              f"{mism - nondet}")
        for Mp, Np, K, lt, nt in surprises:
            print(f"    SURPRISE (M={Mp},N={Np},K={K}): LUT={lt} nsys={nt}")
    return n, mism, nondet


def load_ncu():
    """Return {(Mp,Np,K): Counter(tile)} from the offline ncu microbenchmark."""
    if not NCU_GEMM.exists():
        return {}
    d = json.load(open(NCU_GEMM))
    dist = defaultdict(Counter)
    for v in d.values():
        a = v["op_info"]["Attr"]
        Mtok = a["B"] * a["M"]   # effective tokens (B batches of M each)
        K, N = a["K"], a["N"]
        Mp, Np = pad64(Mtok), pad64(N)
        kn = v["kernels"][0]["Metadata"]["Kernel Name"]
        t = parse_tile(kn)
        if t[0]:
            dist[(Mp, Np, K)][t] += 1
    return dist


def main():
    # --- serving correlation across all A800 cases ---
    # Collect (Mp,Np,K) -> Counter(nsys tile) over CONSISTENT op-record/nsys
    # pairs, plus the per-kernel (LUT-tile, nsys-tile) comparison used by the
    # post-build validation. The regen cache carries op_attr (real M/N/K) on
    # each GEMM entry; it is independent of the tile LUT, so existing caches
    # (built with the old LUT) are still valid for building the new one.
    serving_dist = defaultdict(Counter)
    serving_pairs = []        # (Mp, Np, K, nsys_tile) for every consistent pair
    serving_shapes = set()
    total_clean = 0
    per_case = []
    for case_dir in sorted(EXP_ROOT.iterdir()):
        meta = case_dir / "meta.json"
        if not meta.exists():
            continue
        hw = json.load(open(meta)).get("hw_config", "")
        if "A800" not in hw:
            continue
        tile_dist, pairs, all_shapes = correlate_serving(case_dir)
        if tile_dist is None:
            continue
        print(f"  [build] {case_dir.name}: {len(pairs)} pairs, "
              f"{len(all_shapes)} shapes", flush=True)
        serving_pairs.extend(pairs)
        for k, cc in tile_dist.items():
            serving_dist[k] += cc
        serving_shapes |= all_shapes
        c_clean = sum(sum(c.values()) for c in tile_dist.values())
        total_clean += c_clean
        per_case.append((case_dir.name, c_clean, len(pairs)))

    print("=== per-case consistent op-record/nsys GEMM pairs ===")
    for name, cl, cp in per_case:
        print(f"  {name:45s} clean_pairs={cp:8d}")
    print(f"  {'TOTAL':45s} clean_pairs={sum(p[2] for p in per_case):8d}")

    # --- dominant tile per (M,N,K) from serving (bf16) data only ---
    lut = {}
    multi = 0
    for k, cc in serving_dist.items():
        lut[k] = cc.most_common(1)[0][0]
        if len(cc) > 1:
            multi += 1
    print(f"\nfinal LUT: {len(lut)} (M,N,K) entries (bf16/tn, serving) | "
          f"{multi} with >1 observed tile (nondeterministic ties)")

    # --- coverage check: every serving-queried shape must be in the LUT ---
    missing = [s for s in serving_shapes if s not in lut]
    print(f"coverage: {len(serving_shapes)} serving shapes, {len(missing)} MISSING "
          f"-> these would assert at eval time")
    for s in missing[:20]:
        print("  MISSING:", s)

    # --- VALIDATION (the goal's check): for each nsys GEMM kernel, look up the
    #     LUT with the op-record (M,N,K) and compare to the nsys actual tile. ---
    mism = 0
    tied = 0   # mismatches where nsys itself was non-deterministic (LUT's dom
               # tile differs from this occurrence but IS one of the observed tiles)
    mis_examples = []
    for Mp, Np, K, ntile in serving_pairs:
        ltile = lut.get((Mp, Np, K))
        if ltile is None:
            continue  # uncovered (reported above)
        if ltile != ntile:
            mism += 1
            cc = serving_dist[(Mp, Np, K)]
            if ntile in cc:
                tied += 1
            else:
                if len(mis_examples) < 12:
                    mis_examples.append((Mp, Np, K, ltile, ntile))
    n_val = len(serving_pairs)
    print(f"\n=== VALIDATION: new LUT tile vs nsys actual tile ===")
    print(f"  {n_val} consistent GEMM kernels | mismatch={mism} ({100*mism/max(1,n_val):.2f}%)")
    print(f"  of which nsys-nondeterministic (LUT dominant != this occurrence, but "
          f"occurrence is an observed tile): {tied}")
    print(f"  genuine surprises (nsys tile NOT observed for this (M,N,K)): "
          f"{mism - tied}")
    for Mp, Np, K, lt, nt in mis_examples:
        print(f"    (M={Mp},N={Np},K={K}): LUT={lt} nsys={nt}")

    # --- collision report: (M,N) with multiple K giving different tiles ---
    by_mn = defaultdict(list)
    for (Mp, Np, K), (tm, tn) in lut.items():
        by_mn[(Mp, Np)].append((K, (tm, tn)))
    print("\n=== (M,N) collisions: same (M,N), K changes the tile ===")
    ncol = 0
    for (Mp, Np), ks in sorted(by_mn.items()):
        tiles = {t for _, t in ks}
        if len(ks) > 1 and len(tiles) > 1:
            ncol += 1
            print(f"  (M={Mp}, N={Np}): " + ", ".join(f"K={K}->{t}" for K, t in ks))
    print(f"  ({ncol} collision (M,N) pairs — K-aware key resolves these; the old "
          f"(M,N) LUT mis-predicted all but the dominant one)")

    # --- write LUT (JSON keys must be strings) ---
    serial = {f"[{Mp},{Np},{K}]": [int(tm), int(tn)]
              for (Mp, Np, K), (tm, tn) in lut.items()}
    LUT_OUT.write_text(json.dumps(serial, indent=0, sort_keys=True))
    print(f"\nwrote {LUT_OUT} ({len(serial)} entries)")


if __name__ == "__main__":
    main()
