"""910B serving workload driver for msprof timeline capture (Ascend).

Reuses the A800 methodology's workload definition — ShareGPT real prompts,
fixed seed, N requests — via vLLM's built-in ShareGPTDataset sampler, then runs
LLM.generate on vllm-ascend. NO CUDA sidecar / LD_PRELOAD (Ascend has none);
the kernel timeline is captured by wrapping THIS script under `msprof`.

Run standalone (Phase 0 style):
    ASCEND_RT_VISIBLE_DEVICES=0,1 python3 serve_driver_910b.py \
        --model /mnt/gpfs/models/Qwen3-30B-A3B --tp 2 --num-requests 16

Run under msprof (Phase 1, timeline ground truth):
    msprof --application="python3 serve_driver_910b.py ..." \
        --task-time=on --output=<dir> --delay=<skip load+warmup s>
"""
import argparse, os, time, json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--ep", type=int, default=1)  # recorded in meta; vLLM-ascend EP via env
    p.add_argument("--num-requests", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset",
                   default="/mnt/gpfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--meta-out", default=None, help="write timing/meta JSON here")
    a = p.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    # Same workload source as A800: ShareGPT real prompt/output lengths, fixed seed.
    from vllm.benchmarks.datasets import ShareGPTDataset
    ds = ShareGPTDataset(dataset_path=a.dataset, random_seed=a.seed)
    samples = ds.sample(tok, num_requests=a.num_requests)
    prompts = [s.prompt for s in samples]
    out_lens = [s.expected_output_len for s in samples]
    print(f"[workload] {len(prompts)} ShareGPT prompts, "
          f"prompt_len[min/max]={min(s.prompt_len for s in samples)}/"
          f"{max(s.prompt_len for s in samples)}, "
          f"out_len[min/max]={min(out_lens)}/{max(out_lens)}", flush=True)

    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=a.tp,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem_util,
              trust_remote_code=True, enforce_eager=True)
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s", flush=True)

    sps = [SamplingParams(max_tokens=ol, temperature=0.0, ignore_eos=True)
           for ol in out_lens]

    # Warmup (1 request) so msprof --delay can skip compile/load.
    llm.generate([prompts[0]], [SamplingParams(max_tokens=8, temperature=0.0)])
    print("[warmup] done", flush=True)

    t1 = time.time()
    gen_start_epoch = t1
    outs = llm.generate(prompts, sps)
    gen_end_epoch = time.time()
    gen_s = gen_end_epoch - t1
    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[generate] {gen_s:.2f}s, {total_out} out tokens, "
          f"{total_out/gen_s:.1f} tok/s", flush=True)
    # Absolute epoch window of the profiled generate() — msprof Task Start
    # Time(us) is Unix-epoch µs on the SAME clock, so the parser filters the
    # kernel timeline to exactly this inference window (excludes load/warmup).
    print(f"[window] gen_start_epoch_us={gen_start_epoch*1e6:.0f} "
          f"gen_end_epoch_us={gen_end_epoch*1e6:.0f}", flush=True)
    print("DRIVER_OK", flush=True)

    if a.meta_out:
        json.dump({"model": a.model, "tp": a.tp, "ep": a.ep,
                   "num_requests": a.num_requests, "seed": a.seed,
                   "load_s": load_s, "generate_s": gen_s,
                   "gen_start_epoch_us": gen_start_epoch * 1e6,
                   "gen_end_epoch_us": gen_end_epoch * 1e6,
                   "total_out_tokens": total_out,
                   "prompt_lens": [s.prompt_len for s in samples],
                   "out_lens": out_lens}, open(a.meta_out, "w"), indent=2)


if __name__ == "__main__":
    main()
