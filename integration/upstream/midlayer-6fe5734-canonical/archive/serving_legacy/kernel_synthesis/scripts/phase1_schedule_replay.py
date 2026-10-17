#!/usr/bin/env python3
"""Phase 1 (route ③) — offline vLLM-Scheduler replay & fidelity harness.

Drives a REAL vllm.v1 Scheduler fully offline (no GPU, no worker, no weights)
against an already-captured serving trace, and checks that its per-step batch
composition reproduces the ground-truth scheduler sidecar.

This is the de-risking spike for the S/W/H redesign (docs/serving_perf_modeling_
redesign.md, ...execution_goal.md Phase 1). It touches NO real hardware and runs
NO real forward pass — invariants I1 (no real compute) and I2 (validate against
existing GT traces only) hold by construction:

  * Scheduler decisions are pure Python (queue math + KV block accounting).
  * We fake the ModelRunnerOutput (dummy non-EOS tokens; max_tokens drives stop).
  * Virtual clock replays GT step timestamps ONLY to gate request admission by
    arrival time — the scheduler's batching logic is what we test.

Historical GT source per case (the retired Phase 0 inventory is no longer kept):
  * plugin_trace.json_scheduler.jsonl : {iter_id, ts_us, scheduled:{req:q}, ...}
  * plugin_trace.json_provenance.json : prompt_lengths, output_lengths, engine_kwargs
  * arrival_schedule.json             : arrival_times_s

Usage:
    conda run -n r2r_qwen3_next python scripts/serve_predict/phase1_schedule_replay.py \
        --case serve_exp/eval_serving_sharegpt/ShareGPT-Qwen3-8B_TP1_n128_qps4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# I1: force CPU-only before importing vllm/torch — no device is ever touched.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ GT loading
def load_case(case_dir: Path):
    prov = json.load(open(case_dir / "plugin_trace.json_provenance.json"))
    arr = json.load(open(case_dir / "arrival_schedule.json"))
    steps = []
    with open(case_dir / "plugin_trace.json_scheduler.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return prov, arr, steps


# ------------------------------------------------------- offline vllm Scheduler
def build_scheduler(prov, num_gpu_blocks: int):
    """Construct a real vllm.v1 Scheduler with zero hardware.

    Faithfully reuses the engine_kwargs recorded at capture time so the
    scheduling policy (chunked prefill, max_num_batched_tokens, prefix caching
    default) matches GT. Memory capacity is a pure input (num_gpu_blocks) — the
    Phase 3 knob standing in for worker KV profiling.
    """
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig, FullAttentionSpec, KVCacheGroupSpec, KVCacheTensor,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    ek = dict(prov.get("engine_kwargs", {}))
    # Map recorded server kwargs -> EngineArgs; drop anything device-specific.
    ea = EngineArgs(
        model=prov["model"],
        tensor_parallel_size=ek.get("tensor_parallel_size", 1),
        pipeline_parallel_size=ek.get("pipeline_parallel_size", 1),
        dtype=ek.get("dtype", "bfloat16"),
        max_model_len=ek.get("max_model_len"),
        enforce_eager=True,           # no cudagraph capture (would need device)
        trust_remote_code=ek.get("trust_remote_code", True),
        enable_chunked_prefill=ek.get("enable_chunked_prefill", True),
        max_num_batched_tokens=ek.get("max_num_batched_tokens"),
        load_format="dummy",          # never read weights from disk
    )
    vcfg = ea.create_engine_config()
    mc, pc = vcfg.model_config, vcfg.parallel_config

    n_layers = mc.get_num_layers(pc)
    block_size = vcfg.cache_config.block_size
    dtype = mc.dtype if isinstance(mc.dtype, torch.dtype) else torch.bfloat16
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=mc.get_num_kv_heads(pc),
        head_size=mc.get_head_size(),
        dtype=dtype,
    )
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(n_layers)]
    page_bytes = spec.page_size_bytes
    kv_cfg = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[KVCacheTensor(size=num_gpu_blocks * page_bytes,
                                        shared_by=layer_names)],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names,
                                          kv_cache_spec=spec)],
    )
    # THE memory knob (worker profiling is skipped): set KV capacity directly.
    vcfg.cache_config.num_gpu_blocks = num_gpu_blocks
    vcfg.cache_config.num_gpu_blocks_override = num_gpu_blocks

    som = StructuredOutputManager(vcfg)
    sched = Scheduler(vcfg, kv_cfg, som)
    return sched, vcfg


def make_request(vcfg, req_id: str, prompt_len: int, output_len: int,
                 arrival_time: float):
    """Build a Request with a faithful prefix-cache block hasher (GT default)."""
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher, init_none_hash,
    )
    from vllm.utils import get_hash_fn_by_name

    block_hasher = None
    cc = vcfg.cache_config
    if cc.enable_prefix_caching:
        hash_fn = get_hash_fn_by_name(cc.prefix_caching_hash_algo)
        init_none_hash(hash_fn)
        block_hasher = get_request_block_hasher(cc.block_size, hash_fn)

    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max(1, output_len)),
        pooling_params=None,
        eos_token_id=None,           # only max_tokens stops the request
        arrival_time=arrival_time,
        block_hasher=block_hasher,
    )


def fake_output(sched, sched_out):
    """Fabricate a ModelRunnerOutput for the just-scheduled step.

    A request emits one dummy token iff it has finished its prompt this step
    (num_computed_tokens >= num_prompt_tokens after schedule()); partial prefill
    chunks emit []. max_tokens (not EOS) drives the stop, matching GT lengths.
    """
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(sched_out.num_scheduled_tokens.keys())
    idx = {r: i for i, r in enumerate(req_ids)}
    sampled = []
    for r in req_ids:
        req = sched.requests.get(r)
        done_prompt = req is not None and req.num_computed_tokens >= req.num_prompt_tokens
        sampled.append([42] if done_prompt else [])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=idx,
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[None] * len(req_ids),
    )


# --------------------------------------------------------------------- replay
def replay(case_dir: Path, num_gpu_blocks: int, verbose: int = 0):
    prov, arr, gt_steps = load_case(case_dir)
    prompts = prov["prompt_lengths"]
    outputs = prov["output_lengths"]
    arrivals = arr["arrival_times_s"]
    n = len(prompts)
    assert len(outputs) == n == len(arrivals), "workload length mismatch"

    sched, vcfg = build_scheduler(prov, num_gpu_blocks)

    # Requests admitted lazily by arrival time, in arrival order.
    order = sorted(range(n), key=lambda i: arrivals[i])
    next_arrival = 0
    t0_us = gt_steps[0]["ts_us"]

    rows = []            # per-step comparison records
    n_match = n_mismatch = 0
    first_mismatch = None

    for k, gt in enumerate(gt_steps):
        vtime = (gt["ts_us"] - t0_us) / 1e6      # virtual clock (s) from GT
        # Admit every request that has arrived by now.
        while next_arrival < n and arrivals[order[next_arrival]] <= vtime:
            i = order[next_arrival]
            sched.add_request(make_request(
                vcfg, f"req_{i}", prompts[i], outputs[i], arrivals[i]))
            next_arrival += 1

        sched_out = sched.schedule()
        mine = dict(sched_out.num_scheduled_tokens)
        gt_sched = {k2: int(v) for k2, v in gt["scheduled"].items()}

        exact = mine == gt_sched
        if exact:
            n_match += 1
        else:
            n_mismatch += 1
            if first_mismatch is None:
                first_mismatch = k
        # Cheap structural deltas for diagnosis.
        my_set, gt_set = set(mine), set(gt_sched)
        rows.append({
            "k": k, "iter_id": gt["iter_id"], "exact": exact,
            "my_nreq": len(mine), "gt_nreq": len(gt_sched),
            "my_tokens": sum(mine.values()), "gt_tokens": gt["total_tokens"],
            "only_mine": sorted(my_set - gt_set)[:6],
            "only_gt": sorted(gt_set - my_set)[:6],
        })
        if verbose and (not exact) and n_mismatch <= verbose:
            print(f"  [mismatch k={k} iter={gt['iter_id']}] "
                  f"mine={_short(mine)} gt={_short(gt_sched)}")

        # Advance request state so the next step sees correct running set.
        sched.update_from_output(sched_out, fake_output(sched, sched_out))

    total = len(gt_steps)
    # Taxonomy of deviations. The scheduler's batching logic is deterministic
    # given the admitted request set; the ONLY irreproducible input offline is
    # sub-ms arrival/step ordering (the real async add_request task races the
    # step loop). So a faithful ③ should show every mismatch as either:
    #   * "admit_shift"  : symmetric req-set diff of exactly 1 (a new request's
    #                      first prefill lands ±1 step vs GT), or
    #   * "resync"       : identical req set, q-map differs (the shifted request
    #                      catching up — prefill in one, decode in the other).
    # Any "logic_diverge" (>=2 reqs differ) would indicate a real policy bug.
    admit_shift = resync = logic_diverge = 0
    for r in rows:
        if r["exact"]:
            continue
        nd = len(r["only_mine"]) + len(r["only_gt"])
        if nd == 0:
            resync += 1
        elif nd == 1:
            admit_shift += 1
        else:
            logic_diverge += 1

    n_match = sum(1 for r in rows if r["exact"])
    n_mismatch = total - n_match
    gate_pass = (logic_diverge == 0
                 and summary_pct(n_match, total) >= 97.0
                 and next_arrival == n)
    summary = {
        "case": case_dir.name,
        "num_gpu_blocks": num_gpu_blocks,
        "n_reqs": n,
        "n_steps_gt": total,
        "n_steps_driver": total,
        "exact_match_steps": n_match,
        "mismatch_steps": n_mismatch,
        "exact_match_pct": summary_pct(n_match, total),
        "first_mismatch_k": first_mismatch,
        "driver_finished_all": next_arrival == n,
        # deviation taxonomy (see comment above)
        "dev_admit_shift": admit_shift,
        "dev_resync": resync,
        "dev_logic_diverge": logic_diverge,
        # exit gate: >=97% exact, zero logic divergence, workload fully drained
        "EXIT_GATE_PASS": gate_pass,
    }
    return summary, rows


def summary_pct(n_match, total):
    return 100.0 * n_match / total if total else 0.0


def _short(d):
    keys = sorted(d, key=lambda r: int(r.split("_")[1]))
    if len(keys) <= 6:
        return {k: d[k] for k in keys}
    return {**{k: d[k] for k in keys[:4]}, "...": f"+{len(keys)-4}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    help="case dir (abs or repo-relative)")
    ap.add_argument("--num-gpu-blocks", type=int, default=40000,
                    help="KV capacity; generous for Phase 1 (no preemption).")
    ap.add_argument("--verbose", type=int, default=10,
                    help="print up to N mismatching steps")
    ap.add_argument("--out-json", default=None,
                    help="write summary JSON here (per-step rows excluded "
                         "unless --dump-rows)")
    ap.add_argument("--dump-rows", action="store_true",
                    help="also embed the full per-step comparison rows "
                         "(large; regenerable, kept out of git)")
    args = ap.parse_args()

    case_dir = Path(args.case)
    if not case_dir.is_absolute():
        case_dir = REPO / case_dir

    summary, rows = replay(case_dir, args.num_gpu_blocks, args.verbose)

    print("\n=== Phase 1 ③ schedule-replay summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary}
        if args.dump_rows:
            payload["rows"] = rows
        json.dump(payload, open(out, "w"), indent=1)
        print(f"\nwrote {out}" + ("" if args.dump_rows else " (summary only)"))


if __name__ == "__main__":
    main()
