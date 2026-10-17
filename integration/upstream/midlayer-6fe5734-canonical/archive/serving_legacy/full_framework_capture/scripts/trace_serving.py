#!/usr/bin/env python3
"""
Run vLLM serving workload under CUDA emulation and capture kernel trace.

This script uses vllm.LLM (documented offline Python API) with batch
submission to trigger scheduler-driven continuous batching and chunked prefill.
It supports:
- Offline batch mode: LLM.generate(prompts_list, sampling_params_list)
- Explicit refusal for unsupported staggered arrivals
- Chunked prefill observation with long prompts
- Inference-window trace filtering (excludes model loading/warmup)

Usage:
    # Basic batch of 10 requests with 512-token prompts, 8 decode steps each
    python scripts/serve_trace/trace_serving.py \
        --model /share_data/public_models/Qwen3-0.6B \
        --num-requests 10 --prompt-len 512 --max-tokens 8

    # Chunked prefill: long prompt (4096) with small batch budget (512)
    python scripts/serve_trace/trace_serving.py \
        --model /share_data/public_models/Qwen3-0.6B \
        --num-requests 3 --prompt-len 4096 --max-tokens 8 \
        --enable-chunked-prefill --max-num-batched-tokens 512

    # Staggered arrival is refused unless implemented with a true async/server path
    python scripts/serve_trace/trace_serving.py \
        --model /share_data/public_models/Qwen3-0.6B \
        --num-requests 5 --prompt-len 1024 --max-tokens 8 \
        --stagger 0.5

    # With trace gating (NOT SUPPORTED for serving — will error)
    CUDA_TRACE_REQUIRE_PROFILER=1 python scripts/serve_trace/trace_serving.py ...

Environment:
    - LD_PRELOAD must include libcudart.so.12, libcublas.so.12
    - VLLM_WORKER_MULTIPROC_METHOD=fork (required for LD_PRELOAD inheritance)
    - CUDA_TRACE_OUTPUT: base path for trace files (default: serve_exp/serving_trace.json)
    - CUDA_TRACE_REQUIRE_PROFILER: MUST NOT be set to 1 (incompatible with vLLM V1
      multi-process architecture; use inference-window filtering instead)

Outputs:
    - {CUDA_TRACE_OUTPUT}: merged trace (all PIDs combined, sorted by timestamp)
    - {CUDA_TRACE_OUTPUT}_provenance.json: metadata sidecar
"""
import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

# ── Environment defaults ──────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)
from src.kernel.validation.dataset_io import get_emulation_output_dir as _get_emu_dir
_emu_output = _get_emu_dir()
os.makedirs(_emu_output, exist_ok=True)
os.environ.setdefault("CUDA_TRACE_OUTPUT",
                       os.path.join(_emu_output, "serving_trace.json"))

# ── Scheduler sidecar: per-step request attribution ──────────────
# Monkey-patches vLLM V1 Scheduler.schedule() to emit JSONL recording
# which requests participate in each scheduling step and how many tokens
# each contributes.  Applied before `from vllm import LLM` so the patch
# propagates to the forked EngineCore child process (where the scheduler
# actually runs).
_SIDECAR_PATH = None  # set in main() before engine creation
_SIDECAR_PTH_PATH = None  # set by _install_scheduler_sidecar()


def _install_scheduler_sidecar():
    """Patch Scheduler.schedule() to write per-step JSONL.

    Works in both fork and spawn modes:
    - Sets CUDA_SIDECAR_PATH env var (inherited by child processes)
    - Installs a .pth file in site-packages that auto-imports the patch
      module when any Python process starts (including vLLM spawn children)
    - Also patches immediately for the current process
    """
    sidecar_path = os.environ.get("CUDA_SIDECAR_PATH")
    if not sidecar_path:
        return None

    # Install .pth file so spawn children auto-load the patch
    global _SIDECAR_PTH_PATH
    from src.serve_trace.sidecar_lifecycle import install_sidecar_pth
    _root = Path(__file__).parent.parent.parent
    _SIDECAR_PTH_PATH = install_sidecar_pth(project_root=_root)

    # Also patch immediately for the current process
    from src.serve_trace._sidecar_patch import install_sidecar_patch
    install_sidecar_patch()
    return _SIDECAR_PTH_PATH

# Reject profiler gating: vLLM V1 EngineCore child process does all GPU work,
# so cudaProfilerStart in the main process cannot propagate to the worker.
if os.environ.get("CUDA_TRACE_REQUIRE_PROFILER") == "1":
    print("[ERROR] CUDA_TRACE_REQUIRE_PROFILER=1 is incompatible with vLLM V1 serving mode.\n"
          "The EngineCore child process handles all GPU kernels; profiler gating in\n"
          "the main process cannot propagate to the forked worker. Unset this variable\n"
          "or set CUDA_TRACE_REQUIRE_PROFILER=0. Warmup exclusion uses post-hoc\n"
          "inference-window filtering instead.", file=sys.stderr)
    sys.exit(1)


def timeout_handler(signum, frame):
    print("\n[TIMEOUT] Exiting", file=sys.stderr)
    sys.exit(1)


def generate_prompt(length_tokens: int, tokenizer=None) -> str:
    """Generate a prompt of approximately `length_tokens` tokens.

    Uses a fixed vocabulary of common words to get predictable token counts.
    If tokenizer is provided, re-samples until the tokenized length is within
    10% of the target.
    """
    vocab = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "artificial", "intelligence", "machine", "learning", "neural",
        "network", "transformer", "attention", "mechanism", "language",
        "model", "generation", "inference", "serving", "batching",
        "throughput", "latency", "optimization", "accelerator", "GPU",
        "tensor", "parallel", "data", "pipeline", "distributed", "system",
    ]
    # Rough estimate: 1 token ≈ 0.75 words on average for English
    num_words = int(length_tokens * 0.75)
    prompt = " ".join(random.choices(vocab, k=num_words))

    if tokenizer is not None:
        # Refine: re-sample if token count is way off
        for _ in range(10):
            toks = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if abs(len(toks) - length_tokens) <= max(10, length_tokens // 10):
                break
            num_words = int(num_words * length_tokens / max(len(toks), 1))
            prompt = " ".join(random.choices(vocab, k=num_words))
    return prompt


def load_prompts_and_update_request_count(args, tokenizer):
    """Load prompts and return (prompts, prompt_lens, output_lens).

    output_lens is None when all requests share args.max_tokens,
    or a list of per-request int when using dataset with real output lengths.
    """
    output_lens = None
    if args.prompts_file:
        with open(args.prompts_file) as f:
            workload = json.load(f)
        prompts = workload["prompts"]
        actual_lens = workload.get("prompt_lengths")
        if actual_lens is None:
            actual_lens = [
                len(tokenizer(p, add_special_tokens=False)["input_ids"])
                for p in prompts
            ]
        args.num_requests = len(prompts)
        print(f"[workload] Loaded {len(prompts)} exact prompts from {args.prompts_file}")
    elif args.dataset:
        from vllm.benchmarks.datasets import ShareGPTDataset
        ds = ShareGPTDataset(dataset_path=args.dataset, random_seed=args.seed)
        samples = ds.sample(tokenizer, num_requests=args.num_requests,
                            output_len=None)
        prompts = [s.prompt for s in samples]
        actual_lens = [s.prompt_len for s in samples]
        output_lens = [s.expected_output_len for s in samples]
        args.num_requests = len(prompts)
        print(f"[workload] Loaded {len(prompts)} prompts from ShareGPT dataset "
              f"(per-request output lens: min={min(output_lens)}, "
              f"max={max(output_lens)}, median={sorted(output_lens)[len(output_lens)//2]})")
    else:
        prompts = [generate_prompt(args.prompt_len, tokenizer) for _ in range(args.num_requests)]
        actual_lens = [
            len(tokenizer(p, add_special_tokens=False)["input_ids"])
            for p in prompts
        ]
        args.num_requests = len(prompts)
    return prompts, actual_lens, output_lens


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run vLLM serving workload under CUDA emulation"
    )
    parser.add_argument("--model", default="/share_data/public_models/Qwen3-0.6B",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Number of requests to submit")
    parser.add_argument("--prompt-len", type=int, default=512,
                        help="Target prompt length in tokens")
    parser.add_argument("--max-tokens", type=int, default=8,
                        help="Max tokens to generate per request")
    parser.add_argument("--workload-file", default=None,
                        help="JSON with per-request {prompt_lens:[...], "
                             "out_lens:[...]} — analytic mode uses these exact "
                             "variable lengths so prediction matches the real "
                             "ShareGPT sample the msprof GT ran (workload align). "
                             "Absent → uniform prompt_len/max_tokens (unchanged).")
    parser.add_argument("--seq-len", type=int, default=8192,
                        help="Max model sequence length")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Pipeline parallel size (analytic mode: models one "
                             "rank running layers/pp; non-last stages omit lm_head)")
    parser.add_argument("--expert-parallel-size", type=int, default=1,
                        help="Expert parallel size for MoE. The naive EP backend "
                             "(A800) issues no all-to-all: it places a local expert "
                             "subset and reduces with the post-FFN all-reduce, so "
                             "EP>1 adds an all-reduce, not all-to-all. Real all-to-all "
                             "(pplx/deepep) is captured, not synthesized, in plugin mode.")
    parser.add_argument("--engine-args-json", default=None,
                        help="JSON object of additional vLLM LLM engine kwargs")
    parser.add_argument("--stagger", type=float, default=0.0,
                        help="Seconds to sleep between request submissions (0 = batch)")
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                        help="Enable chunked prefill in scheduler")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Override scheduler max_num_batched_tokens (default: model_max)")
    parser.add_argument("--hw-config", default="configs/hardware/nvidia_a800.yaml",
                        help="Path to HW config YAML")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds")
    parser.add_argument("--dataset", default=None,
                        help="Path to ShareGPT-format dataset JSON for realistic prompts")
    parser.add_argument("--prompts-file", default=None,
                        help="Path to workload JSON with exact prompt strings (bypasses generation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt generation")
    parser.add_argument("--trace-output", default=None,
                        help="Override CUDA_TRACE_OUTPUT path")
    parser.add_argument("--mode", choices=["real", "analytic", "plugin"], default="plugin",
                        help="Trace source (default 'plugin'). 'plugin' (recommended, "
                             "approach D) runs REAL vLLM and captures the real "
                             "scheduler sidecar + real ModelRunner operator sequence, "
                             "then synthesizes the kernel trace via the recipe "
                             "operator->kernel last hop; 'real' runs vLLM under CUDA "
                             "emulation (LD_PRELOAD fake libcudart) and is kept as the "
                             "NVIDIA oracle/cross-check; 'analytic' mirrors the vLLM "
                             "scheduler and synthesizes the trace from the model config "
                             "without GPU or vLLM.")
    parser.add_argument("--nsys-bracket", action="store_true",
                        help="Add cudaProfilerStart/Stop around main generate() for "
                             "nsys --capture-range=cudaProfilerApi. Includes a warmup "
                             "step before the profiler window. Plugin mode only.")
    parser.add_argument("--warmup-reset", action="store_true",
                        help="Reset operator/sidecar files after warmup (like --nsys-bracket "
                             "but without cudaProfiler calls). For external profilers "
                             "like mcTracer that capture the full process.")
    parser.add_argument("--qps", type=float, default=0.0,
                        help="Queries per second for Poisson arrival (0 = offline batch). "
                             "When >0, uses AsyncLLMEngine with staggered request submission.")
    parser.add_argument("--schedule-file", default=None,
                        help="Path to arrival_schedule.json. If given, load arrival times "
                             "from it; otherwise generate from --qps + --seed.")
    return parser


def validate_args(args, parser):
    if args.stagger > 0 and args.qps <= 0:
        parser.error(
            "--stagger > 0 is not supported: vllm.LLM.generate() in this "
            "offline integration serializes one-prompt calls, so it cannot "
            "produce true staggered continuous batching. Use --qps for "
            "Poisson arrival serving mode, or --stagger 0 for offline batch."
        )
    if args.qps > 0 and args.mode != "plugin":
        parser.error("--qps > 0 (serving mode) is only supported with --mode plugin")


def _run_analytic(args):
    """Analytic mode: mirror vLLM scheduling and generate the kernel trace +
    scheduler sidecar from the model config alone — no GPU, no vLLM, no
    LD_PRELOAD. Produces the same on-disk artifacts as the real path so all
    downstream consumers (ServingPredictor, compare_trace_with_nsys) are
    mode-agnostic.
    """
    import json
    from src.serve_predict.model_config import load_model_config
    from src.kernel.hardware_spec import HWSpec
    from src.serve_trace.analytic_mirror import (
        MirrorRequest, AnalyticTraceGenerator, write_analytic_trace)

    trace_base = args.trace_output or os.environ["CUDA_TRACE_OUTPUT"]
    Path(trace_base).parent.mkdir(parents=True, exist_ok=True)
    print(f"[analytic] mode=analytic (no vLLM/GPU)")
    print(f"[analytic] model={args.model} requests={args.num_requests} "
          f"prompt_len={args.prompt_len} max_tokens={args.max_tokens} "
          f"tp={args.tensor_parallel_size}")

    model_cfg = load_model_config(args.model)
    hw = HWSpec(args.hw_config)
    gen = AnalyticTraceGenerator(model_cfg, hw,
                                 tp_size=args.tensor_parallel_size,
                                 pp_size=args.pipeline_parallel_size,
                                 ep_size=args.expert_parallel_size)

    # Per-request lengths. With --workload-file, use the EXACT variable prompt/
    # output lengths of the real ShareGPT sample (so analytic prediction matches
    # the workload the msprof ground truth actually ran). Else uniform.
    if getattr(args, "workload_file", None):
        wl = json.load(open(args.workload_file))
        pls, ols = wl["prompt_lens"], wl["out_lens"]
        requests = [MirrorRequest(req_id=str(i), prompt_len=int(pls[i]),
                                  output_len=int(ols[i])) for i in range(len(pls))]
        args.num_requests = len(requests)
        print(f"[analytic] workload-file: {len(requests)} reqs, "
              f"prompt[min/max]={min(pls)}/{max(pls)}, out[min/max]={min(ols)}/{max(ols)}")
    else:
        requests = [MirrorRequest(req_id=str(i),
                                  prompt_len=args.prompt_len,
                                  output_len=args.max_tokens)
                    for i in range(args.num_requests)]

    trace_entries, sched_records = gen.generate(
        requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill)

    trace_path, sched_path = write_analytic_trace(
        trace_entries, sched_records, trace_base)

    provenance = {
        "trace_output": trace_path,
        "scheduler_sidecar": sched_path,
        "mode": "analytic",
        "model": args.model,
        "hw_config": args.hw_config,
        "num_requests": args.num_requests,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    with open(trace_base + "_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"[analytic] {len(trace_entries)} kernel entries → {trace_path}")
    print(f"[analytic] {len(sched_records)} scheduler steps → {sched_path}")
    print("SUCCESS: analytic mirror trace generated!")
    return 0


def _run_plugin(args):
    """Plugin mode (approach D): run REAL vLLM on GPU, capture the real scheduler
    sidecar + the real per-step ModelRunner operator sequence (exact shapes),
    then synthesize the kernel trace via the recipe operator->kernel last hop.

    No LD_PRELOAD / no fake CUDA: vLLM runs real compute on the GPU. The kernel
    trace is SYNTHESIZED from the captured real operator sequence (it is never a
    CUDA-launch capture). Produces the same on-disk artifacts (trace JSON +
    scheduler sidecar JSONL) as the other modes so downstream consumers are
    mode-agnostic; provenance records producer=recipe-from-real-operators.
    """
    import json
    from src.serve_predict.model_config import load_model_config
    from src.kernel.hardware_spec import HWSpec
    from src.serve_trace.operator_records import (
        read_operator_records, OperatorTraceGenerator)
    from src.serve_trace.sidecar_lifecycle import (
        install_sidecar_pth, cleanup_sidecar_pth)

    trace_base = args.trace_output or os.environ["CUDA_TRACE_OUTPUT"]
    Path(trace_base).parent.mkdir(parents=True, exist_ok=True)
    op_path = trace_base + "_operators.jsonl"
    sched_path = trace_base + "_scheduler.jsonl"
    import glob as _glob
    for p in [sched_path] + _glob.glob(op_path + ".*"):
        if os.path.exists(p):
            os.remove(p)

    # Enable the in-child capture patches (sidecar + operator) via env + .pth so
    # they load in the spawned EngineCore worker.  For fork mode, also install
    # directly in this process so forked children inherit the patched state.
    # Strip emulation env vars first — if the shell ran LD_PRELOAD mode before,
    # these would make child processes load fake CUDA/NCCL libraries. If
    # LD_PRELOAD was active when THIS process started, the fake libs are already
    # loaded and cannot be unloaded; fail early so the user re-launches cleanly.
    _preload = os.environ.get("LD_PRELOAD", "")
    if any(lib in _preload for lib in ("libcudart", "libcublas", "libnccl")):
        print("[FATAL] LD_PRELOAD contains fake CUDA/NCCL libraries — these are "
              "already loaded into this process and cannot be unloaded. Plugin "
              "mode requires real GPU libraries. Re-launch without LD_PRELOAD.",
              file=sys.stderr)
        sys.exit(1)
    for key in ("LD_PRELOAD", "CUDA_EMULATE_CONFIG", "VLLM_NCCL_SO_PATH"):
        os.environ.pop(key, None)
    os.environ["CUDA_OPERATOR_TRACE_PATH"] = op_path
    os.environ["CUDA_SIDECAR_PATH"] = sched_path
    _root = Path(__file__).parent.parent.parent
    pth = install_sidecar_pth(project_root=_root)
    from src.serve_trace._sidecar_patch import install_sidecar_patch
    from src.serve_trace._operator_patch import install_operator_patch
    install_sidecar_patch()
    install_operator_patch()
    print(f"[plugin] mode=plugin (real vLLM on GPU; approach-D operator capture)")
    print(f"[plugin] operator trace → {op_path}")
    print(f"[plugin] scheduler sidecar → {sched_path}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        prompts, actual_lens, output_lens = load_prompts_and_update_request_count(args, tokenizer)

        # Build engine kwargs consistent with the real path: forward extra engine
        # args, chunked-prefill controls, and REAL expert parallelism so the vLLM
        # engine actually runs the claimed TP/PP/EP cell (not just the post-hoc
        # trace generator). enable_expert_parallel is required for EP>1.
        engine_kwargs = {
            "model": args.model, "dtype": "bfloat16",
            "gpu_memory_utilization": 0.8, "enforce_eager": True,
            "max_model_len": max(512, args.seq_len),
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "trust_remote_code": True,
            "enable_chunked_prefill": args.enable_chunked_prefill,
        }
        if args.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
        if args.expert_parallel_size > 1:
            engine_kwargs["enable_expert_parallel"] = True
        if args.engine_args_json:
            extra = json.loads(args.engine_args_json)
            engine_kwargs.update(extra)
        print(f"[plugin] engine_kwargs: { {k: v for k, v in engine_kwargs.items() if k != 'model'} }")

        from vllm import LLM, SamplingParams
        llm = LLM(**engine_kwargs)
        if output_lens is not None:
            params = [SamplingParams(temperature=0.0, max_tokens=ol,
                                     min_tokens=ol, ignore_eos=True)
                      for ol in output_lens]
        else:
            params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                    min_tokens=args.max_tokens, ignore_eos=True)

        if args.nsys_bracket or args.warmup_reset:
            import torch
            _ = llm.generate(["test"], SamplingParams(temperature=0.0, max_tokens=1))
            torch.cuda.synchronize()
            import glob as _gl
            # Only clear warmup records if --warmup-reset (NOT for
            # --nsys-bracket alone): operator-clock TTFT needs warmup as iter0.
            if args.warmup_reset:
                for f in [sched_path] + _gl.glob(op_path + ".*"):
                    open(f, "w").close()
            if args.nsys_bracket:
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.synchronize()
                print("[plugin] nsys bracket: warmup done, profiler started" +
                      (" (warmup retained)" if not args.warmup_reset else " (files reset)"))
            elif args.warmup_reset:
                print("[plugin] warmup done, files reset")

        t0 = time.time()
        out = llm.generate(prompts, params)
        wall_s = time.time() - t0

        if args.nsys_bracket:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"WALL_MS:{wall_s * 1000:.3f}")

        print(f"[plugin] generated {sum(len(o.outputs[0].token_ids) for o in out)} tokens "
              f"over {len(out)} requests ({wall_s:.2f}s)")
    finally:
        cleanup_sidecar_pth(project_root=_root)

    # Convert the captured real operator sequence -> per-op priced kernel trace
    # (records_to_trace_direct: each OperatorRecord priced via its recipe leaf).
    # The operator patch writes one PID-suffixed file per worker rank; the model
    # forward runs in the EngineCore worker(s), not the main process. For multi-
    # GPU we compare rank 0 (consistent with rank-0 nsys comparison), so pick the
    # worker file with the most records as the representative rank.
    import glob
    op_files = sorted(glob.glob(op_path + ".*"))
    if not op_files:
        print("[plugin] ERROR: no operator trace files captured — the operator "
              "patch did not load in the EngineCore worker.", file=sys.stderr)
        return 1
    op_counts = {f: sum(1 for _ in open(f)) for f in op_files}
    op_counts = {f: n for f, n in op_counts.items() if n > 0}
    if not op_counts:
        print("[plugin] ERROR: operator trace files are empty.", file=sys.stderr)
        return 1
    # Compare rank 0 (matches capture_nsys's rank-0 convention). The operator
    # patch names files by global rank (`.rank0`, `.rank1`, ...); prefer rank0,
    # falling back to the only/most-populated file for single-process runs.
    rank0_candidates = [f for f in op_counts if f.endswith(".rank0")]
    if rank0_candidates:
        rank0_file = rank0_candidates[0]
    else:
        rank0_file = max(op_counts, key=op_counts.get)
    print(f"[plugin] operator files: { {Path(f).name: n for f, n in op_counts.items()} }; "
          f"using {Path(rank0_file).name}")
    records = read_operator_records(rank0_file)
    model_cfg = load_model_config(args.model)
    hw = HWSpec(args.hw_config)
    gen = OperatorTraceGenerator(model_cfg, hw,
                                 tp_size=engine_kwargs.get("tensor_parallel_size",
                                                           args.tensor_parallel_size),
                                 pp_size=engine_kwargs.get("pipeline_parallel_size",
                                                           args.pipeline_parallel_size),
                                 ep_size=engine_kwargs.get("expert_parallel_size",
                                                           args.expert_parallel_size))
    trace_entries = gen.records_to_trace_direct(records)
    with open(trace_base, "w") as f:
        json.dump(trace_entries, f)

    n_steps = len({r.step_id for r in records})
    provenance = {
        "trace_output": trace_base,
        "scheduler_sidecar": sched_path,
        "operator_trace": rank0_file,
        "mode": "plugin",
        "producer": "recipe-from-real-operators",
        "scheduler": "real-vLLM",
        "operators": "real-ModelRunner",
        "validation_reference": "nsys",
        "model": args.model,
        "hw_config": args.hw_config,
        "num_requests": args.num_requests,
        "prompt_lengths": actual_lens,
        "output_lengths": output_lens if output_lens is not None else [args.max_tokens] * args.num_requests,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "expert_parallel_size": args.expert_parallel_size,
        "engine_kwargs": {k: v for k, v in engine_kwargs.items() if k != "model"},
    }
    with open(trace_base + "_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"[plugin] {len(records)} operator records over {n_steps} steps")
    print(f"[plugin] {len(trace_entries)} priced kernel entries → {trace_base}")
    print("SUCCESS: plugin (approach-D) trace generated!")
    return 0


def _generate_poisson_schedule(num_requests, qps, seed=42):
    """Generate Poisson inter-arrival times.

    Returns a list of absolute arrival times in seconds, starting from 0.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    intervals = rng.exponential(1.0 / qps, size=num_requests)
    arrivals = [0.0]
    for gap in intervals[1:]:
        arrivals.append(arrivals[-1] + gap)
    return arrivals


def _run_serving(args):
    """Serving mode: real vLLM via AsyncLLMEngine + Poisson arrival.

    Same plugin hooks as _run_plugin (operator capture + scheduler sidecar),
    but requests arrive with Poisson timing instead of batch submission.
    Uses AsyncLLMEngine.start_profile()/stop_profile() for nsys bracket.
    """
    import asyncio
    import json
    import numpy as np
    from src.serve_predict.model_config import load_model_config
    from src.kernel.hardware_spec import HWSpec
    from src.serve_trace.operator_records import (
        read_operator_records, OperatorTraceGenerator)
    from src.serve_trace.sidecar_lifecycle import (
        install_sidecar_pth, cleanup_sidecar_pth)

    trace_base = args.trace_output or os.environ["CUDA_TRACE_OUTPUT"]
    Path(trace_base).parent.mkdir(parents=True, exist_ok=True)
    op_path = trace_base + "_operators.jsonl"
    sched_path = trace_base + "_scheduler.jsonl"
    import glob as _glob
    for p in [sched_path] + _glob.glob(op_path + ".*"):
        if os.path.exists(p):
            os.remove(p)

    # Strip emulation env vars (same safety as _run_plugin)
    _preload = os.environ.get("LD_PRELOAD", "")
    if any(lib in _preload for lib in ("libcudart", "libcublas", "libnccl")):
        print("[FATAL] LD_PRELOAD contains fake CUDA/NCCL libraries.", file=sys.stderr)
        sys.exit(1)
    for key in ("LD_PRELOAD", "CUDA_EMULATE_CONFIG", "VLLM_NCCL_SO_PATH"):
        os.environ.pop(key, None)
    os.environ["CUDA_OPERATOR_TRACE_PATH"] = op_path
    os.environ["CUDA_SIDECAR_PATH"] = sched_path
    _root = Path(__file__).parent.parent.parent
    pth = install_sidecar_pth(project_root=_root)
    from src.serve_trace._sidecar_patch import install_sidecar_patch
    from src.serve_trace._operator_patch import install_operator_patch
    install_sidecar_patch()
    install_operator_patch()
    print(f"[serving] mode=serving (AsyncLLMEngine + Poisson QPS={args.qps})")
    print(f"[serving] operator trace → {op_path}")
    print(f"[serving] scheduler sidecar → {sched_path}")

    # Generate or load arrival schedule
    if args.schedule_file and os.path.exists(args.schedule_file):
        with open(args.schedule_file) as f:
            schedule_data = json.load(f)
        arrivals = schedule_data["arrival_times_s"]
        args.num_requests = len(arrivals)
        print(f"[serving] loaded schedule from {args.schedule_file} "
              f"({len(arrivals)} requests)")
    else:
        arrivals = _generate_poisson_schedule(
            args.num_requests, args.qps, args.seed)
        schedule_data = {
            "qps": args.qps,
            "seed": args.seed,
            "num_requests": args.num_requests,
            "arrival_times_s": arrivals,
        }
        if args.schedule_file:
            Path(args.schedule_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.schedule_file, "w") as f:
                json.dump(schedule_data, f, indent=2)
            print(f"[serving] saved schedule to {args.schedule_file}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        prompts, actual_lens, output_lens = load_prompts_and_update_request_count(args, tokenizer)

        mml = max(512, args.seq_len)
        engine_kwargs = {
            "model": args.model, "dtype": "bfloat16",
            "gpu_memory_utilization": 0.8, "enforce_eager": True,
            "max_model_len": mml,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "trust_remote_code": True,
            "enable_chunked_prefill": args.enable_chunked_prefill,
        }
        if args.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
        if args.expert_parallel_size > 1:
            engine_kwargs["enable_expert_parallel"] = True
        if args.engine_args_json:
            extra = json.loads(args.engine_args_json)
            engine_kwargs.update(extra)
        print(f"[serving] engine_kwargs: "
              f"{ {k: v for k, v in engine_kwargs.items() if k != 'model'} }")

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm import SamplingParams

        ea = AsyncEngineArgs(**engine_kwargs)
        engine = AsyncLLMEngine.from_engine_args(ea)
        if output_lens is not None:
            params_list = [SamplingParams(temperature=0.0, max_tokens=ol,
                                          min_tokens=ol, ignore_eos=True)
                           for ol in output_lens]
        else:
            params_list = [SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                          min_tokens=args.max_tokens, ignore_eos=True)
                           ] * len(prompts)

        async def _warmup_and_run():
            # Warmup
            warmup_prompt = " ".join(["hello"] * 64)
            wp = SamplingParams(temperature=0.0, max_tokens=1)
            async for _ in engine.generate(warmup_prompt, wp, request_id="warmup_0"):
                pass
            print("[serving] warmup done")

            if args.nsys_bracket or args.warmup_reset:
                import torch as _torch
                import glob as _gl
                # Only clear warmup records if --warmup-reset (NOT for
                # --nsys-bracket alone): the operator-clock TTFT needs warmup
                # as iter0 (iteration_ends_us[0] ≈ driver t0). The nsys
                # profiler starts AFTER warmup regardless (cudaProfilerStart),
                # so retaining warmup in the sidecar doesn't affect the nsys
                # capture range — it only keeps the warmup step in the sidecar
                # for the operator-clock t0 anchor.
                if args.warmup_reset:
                    for f in [sched_path] + _gl.glob(op_path + ".*"):
                        open(f, "w").close()
                if args.nsys_bracket:
                    _torch.cuda.cudart().cudaProfilerStart()
                    _torch.cuda.synchronize()
                    print("[serving] nsys bracket: profiler started" +
                          (" (warmup retained)" if not args.warmup_reset else " (files reset)"))
                elif args.warmup_reset:
                    print("[serving] warmup done, files reset")

            # Submit requests with Poisson timing
            results = {}
            gen_start_epoch_us = time.time() * 1e6   # Unix-epoch, same clock as
                                                     # msprof Task Start Time(us)
            t0 = time.perf_counter()

            async def _submit_one(idx, arrival_s, prompt):
                delay = arrival_s - (time.perf_counter() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
                send_time = time.perf_counter()
                first_token_time = None
                n_tokens = 0
                async for output in engine.generate(
                        prompt, params_list[idx], request_id=f"req_{idx}"):
                    if first_token_time is None and output.outputs:
                        if len(output.outputs[0].token_ids) >= 1:
                            first_token_time = time.perf_counter()
                    if output.outputs:
                        n_tokens = len(output.outputs[0].token_ids)
                end_time = time.perf_counter()
                results[idx] = {
                    "request_id": f"req_{idx}",
                    "arrival_s": arrival_s,
                    "send_time_s": send_time - t0,
                    "ttft_ms": ((first_token_time - send_time) * 1000
                                if first_token_time else None),
                    "e2e_ms": (end_time - send_time) * 1000,
                    "n_output_tokens": n_tokens,
                }
                if n_tokens > 1 and first_token_time:
                    results[idx]["tpot_ms"] = (
                        (end_time - first_token_time) * 1000 / (n_tokens - 1))
                else:
                    results[idx]["tpot_ms"] = None

            tasks = [_submit_one(i, arrivals[i], prompts[i])
                     for i in range(len(prompts))]
            await asyncio.gather(*tasks)
            wall_s = time.perf_counter() - t0
            gen_end_epoch_us = time.time() * 1e6

            if args.nsys_bracket:
                _torch.cuda.cudart().cudaProfilerStop()
                print(f"WALL_MS:{wall_s * 1000:.3f}")

            return results, wall_s, gen_start_epoch_us, gen_end_epoch_us

        results, wall_s, gen_start_epoch_us, gen_end_epoch_us = asyncio.run(_warmup_and_run())

        # Save per-request metrics
        metrics_path = Path(trace_base).parent / "serving_metrics.json"
        ordered = [results[i] for i in sorted(results)]
        throughput = len(ordered) / wall_s if wall_s > 0 else 0
        metrics = {
            "wall_s": wall_s,
            "throughput_rps": throughput,
            "qps": args.qps,
            "num_requests": len(ordered),
            "per_request": ordered,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[serving] {len(ordered)} requests in {wall_s:.2f}s "
              f"({throughput:.2f} req/s)")

        # Inference-window meta for the msprof GT parser (parse_msprof_timeline
        # --meta): gen_start/end_epoch_us are Unix-epoch µs on the SAME clock as
        # msprof Task Start Time, so the GT timeline is filtered to exactly the
        # served inference window (load/warmup excluded). Written next to the
        # trace so a msprof-wrapped serving run yields GT + prediction in one go.
        meta_path = Path(trace_base).parent / "meta.json"
        with open(meta_path, "w") as f:
            json.dump({"model": args.model,
                       "tp": args.tensor_parallel_size,
                       "ep": args.expert_parallel_size,
                       "num_requests": len(ordered), "seed": args.seed,
                       "qps": args.qps, "generate_s": wall_s,
                       "gen_start_epoch_us": gen_start_epoch_us,
                       "gen_end_epoch_us": gen_end_epoch_us}, f, indent=2)

        # Shutdown engine
        engine.shutdown()
    finally:
        cleanup_sidecar_pth(project_root=_root)

    # Synthesize trace from operator records (same as _run_plugin)
    import glob
    op_files = sorted(glob.glob(op_path + ".*"))
    if not op_files:
        print("[serving] ERROR: no operator trace files captured", file=sys.stderr)
        return 1
    op_counts = {f: sum(1 for _ in open(f)) for f in op_files}
    op_counts = {f: n for f, n in op_counts.items() if n > 0}
    if not op_counts:
        print("[serving] ERROR: operator trace files are empty", file=sys.stderr)
        return 1
    rank0_candidates = [f for f in op_counts if f.endswith(".rank0")]
    rank0_file = rank0_candidates[0] if rank0_candidates else max(
        op_counts, key=op_counts.get)
    print(f"[serving] operator files: "
          f"{ {Path(f).name: n for f, n in op_counts.items()} }; "
          f"using {Path(rank0_file).name}")
    records = read_operator_records(rank0_file)
    model_cfg = load_model_config(args.model)
    hw = HWSpec(args.hw_config)
    gen = OperatorTraceGenerator(model_cfg, hw,
                                 tp_size=engine_kwargs.get("tensor_parallel_size",
                                                           args.tensor_parallel_size),
                                 pp_size=engine_kwargs.get("pipeline_parallel_size",
                                                           args.pipeline_parallel_size),
                                 ep_size=engine_kwargs.get("expert_parallel_size",
                                                           args.expert_parallel_size))
    trace_entries = gen.records_to_trace_direct(records)
    with open(trace_base, "w") as f:
        json.dump(trace_entries, f)

    n_steps = len({r.step_id for r in records})
    provenance = {
        "trace_output": trace_base,
        "scheduler_sidecar": sched_path,
        "operator_trace": rank0_file,
        "mode": "serving",
        "producer": "recipe-from-real-operators",
        "scheduler": "real-vLLM",
        "operators": "real-ModelRunner",
        "qps": args.qps,
        "num_requests": args.num_requests,
        "model": args.model,
        "hw_config": args.hw_config,
        "prompt_lengths": actual_lens,
        "output_lengths": output_lens if output_lens is not None else [args.max_tokens] * args.num_requests,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "expert_parallel_size": args.expert_parallel_size,
        "engine_kwargs": {k: v for k, v in engine_kwargs.items() if k != "model"},
    }
    with open(trace_base + "_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"[serving] {len(records)} operator records over {n_steps} steps")
    print(f"[serving] {len(trace_entries)} priced kernel entries → {trace_base}")
    print("SUCCESS: serving (Poisson arrival) trace generated!")
    return 0



def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    random.seed(args.seed)

    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    # ── Analytic mode: mirror vLLM scheduling, no GPU/vLLM ──────────
    if args.mode == "analytic":
        return _run_analytic(args)

    # ── Plugin mode: real vLLM + approach-D operator capture ────────
    if args.mode == "plugin":
        if args.qps > 0:
            return _run_serving(args)
        return _run_plugin(args)

    # ── Emulation config setup ──────────────────────────────────
    from src.serve_trace.fake_cuda import load_emulation_config as _load_emulation_config, write_emulation_config_json as _write_emulation_json

    # ── Scheduler sidecar setup ──────────────────────────────────
    global _SIDECAR_PATH
    trace_base_for_sidecar = args.trace_output or os.environ["CUDA_TRACE_OUTPUT"]
    _SIDECAR_PATH = trace_base_for_sidecar + "_scheduler.jsonl"
    os.environ["CUDA_SIDECAR_PATH"] = _SIDECAR_PATH
    if os.path.exists(_SIDECAR_PATH):
        os.remove(_SIDECAR_PATH)
    _install_scheduler_sidecar()
    print(f"[sidecar] scheduler sidecar → {_SIDECAR_PATH}")

    emu_cfg = _load_emulation_config(args.hw_config)
    trace_base = args.trace_output or os.environ["CUDA_TRACE_OUTPUT"]
    if args.trace_output:
        os.environ["CUDA_TRACE_OUTPUT"] = trace_base
    config_json_path = _write_emulation_json(emu_cfg, trace_output=trace_base)
    os.environ["CUDA_EMULATE_CONFIG"] = config_json_path
    print(f"[setup] emulation_config: {args.hw_config} → {config_json_path}")
    print(f"[setup] trace_output: {trace_base}")

    # ── Timeout ─────────────────────────────────────────────────
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    # ── Import vLLM ─────────────────────────────────────────────
    # NOTE: Do NOT call torch.cuda.* before LLM creation — it initializes CUDA
    # and forces vLLM to use 'spawn' instead of 'fork', breaking monkey-patch
    # inheritance by the EngineCore child process.

    # ── MoE routing patch for emulation ───────────────────────────
    # Same patch as run_vllm_emulation.py: deterministic diverse expert routing.
    import vllm.model_executor.layers.fused_moe.fused_moe as _fused_moe_mod
    from src.serve_trace.moe_topk import emulation_fused_topk

    _fused_moe_mod.fused_topk = emulation_fused_topk
    print("[emulation] Patched fused_topk with uniform random MoE routing")

    # ── Build prompts ─────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    prompts, actual_lens, output_lens = load_prompts_and_update_request_count(args, tokenizer)
    print(f"[workload] {args.num_requests} requests, prompt_len target={args.prompt_len}, "
          f"actual min={min(actual_lens)}, max={max(actual_lens)}, avg={sum(actual_lens)//len(actual_lens)}")

    # ── vLLM engine setup ─────────────────────────────────────────
    from vllm import LLM, SamplingParams

    engine_kwargs = dict(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        max_model_len=args.seq_len,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.enable_chunked_prefill:
        engine_kwargs["enable_chunked_prefill"] = True
        if args.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
            print(f"[scheduler] chunked prefill ON, max_num_batched_tokens={args.max_num_batched_tokens}")
        else:
            print("[scheduler] chunked prefill ON, max_num_batched_tokens=default")
    else:
        print("[scheduler] chunked prefill OFF")
    if args.engine_args_json:
        extra_engine_args = json.loads(args.engine_args_json)
        if not isinstance(extra_engine_args, dict):
            raise ValueError("--engine-args-json must decode to a JSON object")
        engine_kwargs.update(extra_engine_args)

    print(f"\nCreating LLM engine (model={args.model})...")
    llm = LLM(**engine_kwargs)
    print("LLM engine created!")

    # ── Trace gating: vLLM V1 architecture note ────────────────────
    # In vLLM V1, the EngineCore child process handles all GPU work.
    print("[trace] Capturing all kernels (inference identified by post-hoc filtering)")

    # ── Sampling params ───────────────────────────────────────────
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,   # Force full decode for consistent step count
        ignore_eos=True,              # Don't stop on EOS (emulation outputs zeros)
    )

    # ── Run inference ─────────────────────────────────────────────
    print(f"\nRunning inference (num_requests={args.num_requests}, "
          f"max_tokens={args.max_tokens}, stagger={args.stagger}s)...")

    # Record inference window timestamps (epoch microseconds, matching C trace timestamps)
    inference_start_us = int(time.time() * 1e6)
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    end_time = time.time()
    inference_end_us = int(time.time() * 1e6)
    total_time = end_time - start_time

    # ── Results ───────────────────────────────────────────────────
    print(f"\n=== Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {args.num_requests / total_time:.2f} req/s")
    for i, out in enumerate(outputs[:3]):
        print(f"  Request {i+1}: {len(out.outputs[0].token_ids)} tokens generated")
    if len(outputs) > 3:
        print(f"  ... and {len(outputs) - 3} more requests")

    # Force EngineCore child process shutdown so its trace destructor runs
    del llm
    import gc
    gc.collect()
    time.sleep(2)

    # ── Trace collection ──────────────────────────────────────────
    import glob
    trace_files = glob.glob(f"{trace_base}.*")
    trace_files = [f for f in trace_files if not f.endswith("_provenance.json")]

    if not trace_files:
        print("\n[WARNING] No PID-suffixed trace files found. "
              "LD_PRELOAD may not be active or trace gating blocked all capture.",
              file=sys.stderr)
        sys.exit(1)

    # Record PID-file statistics before any merge/cleanup
    main_pid = os.getpid()
    pid_file_stats = {}
    for tf in sorted(trace_files):
        fsize = os.path.getsize(tf)
        pid_str = tf.split(".")[-1]
        try:
            pid = int(pid_str)
        except ValueError:
            pid = -1
        role = "main" if pid == main_pid else f"worker_{len([p for p in pid_file_stats if pid_file_stats[p]['role'] != 'main'])}"
        pid_file_stats[pid_str] = {"pid": pid, "size_bytes": fsize, "role": role, "path": tf}
        print(f"  {tf} ({fsize:,} bytes) [PID={pid}, role={role}]")

    print(f"\nTrace files: {len(trace_files)}")

    # Validate: in serving mode, at least one non-main PID should have kernel data
    non_main_files = [s for s in pid_file_stats.values() if s["role"] != "main"]
    empty_workers = [s for s in non_main_files if s["size_bytes"] < 10]

    validation_failed = False
    if not non_main_files:
        print("[ERROR] No non-main PID trace files found. "
              "vLLM EngineCore worker did not produce a trace.",
              file=sys.stderr)
        validation_failed = True
    if empty_workers:
        for ew in empty_workers:
            print(f"[ERROR] Empty worker trace file PID={ew['pid']} ({ew['path']}) — "
                  f"LD_PRELOAD may not have been inherited by this worker process",
                  file=sys.stderr)
        validation_failed = True

    # Merge traces (always keep PID files until provenance is written)
    merge_script = Path(__file__).parent / "merge_traces.py"
    if merge_script.exists():
        import subprocess
        cmd = [sys.executable, str(merge_script), trace_base, "--segment",
               "--keep-pid-files", "--main-pid", str(main_pid)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
    else:
        print(f"[WARNING] merge_traces.py not found at {merge_script}", file=sys.stderr)

    # Filter merged trace to inference-only window
    raw_kernel_count = 0
    inference_kernel_count = 0
    merged_path = Path(trace_base)
    if merged_path.exists():
        with open(merged_path) as f:
            raw_trace = json.load(f)
        raw_kernel_count = len(raw_trace)
        inference_trace = [
            e for e in raw_trace
            if e.get("timestamp_us", 0) >= inference_start_us
            and e.get("timestamp_us", 0) <= inference_end_us
        ]
        inference_kernel_count = len(inference_trace)
        # Write inference-only trace (overwrites the merged raw)
        with open(merged_path, "w") as f:
            json.dump(inference_trace, f)
        print(f"[trace] Inference window: {inference_start_us} → {inference_end_us}")
        print(f"[trace] Raw kernels: {raw_kernel_count}, inference-only: {inference_kernel_count}, "
              f"excluded: {raw_kernel_count - inference_kernel_count}")

    # Write provenance sidecar
    workload_hash = None
    if args.prompts_file:
        import hashlib
        with open(args.prompts_file) as f:
            wl = json.load(f)
        workload_hash = wl.get("hash")
    if workload_hash is None:
        import hashlib
        identity = f"{args.model}|{args.num_requests}|{args.prompt_len}|{args.max_tokens}|{args.seed}"
        workload_hash = hashlib.sha256(identity.encode()).hexdigest()[:16]
    provenance = {
        "capture_command": " ".join(sys.argv),
        "capture_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "hw_config": args.hw_config,
        "num_requests": args.num_requests,
        "prompt_len": args.prompt_len,
        "prompt_lengths": actual_lens,
        "max_tokens": args.max_tokens,
        "seq_len": args.seq_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "stagger": args.stagger,
        "chunked_prefill": args.enable_chunked_prefill,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "total_time_s": total_time,
        "trace_output": trace_base,
        "trace_gating_enabled": False,
        "main_pid": main_pid,
        "prompts_file": args.prompts_file,
        "workload_hash": workload_hash,
        "inference_window_us": [inference_start_us, inference_end_us],
        "raw_kernel_count": raw_kernel_count,
        "inference_kernel_count": inference_kernel_count,
        "warmup_kernel_count_excluded": raw_kernel_count - inference_kernel_count,
        "pid_file_stats": {k: {"pid": v["pid"], "size_bytes": v["size_bytes"], "role": v["role"]}
                          for k, v in pid_file_stats.items()},
        "validation_passed": not validation_failed,
        "scheduler_sidecar": _SIDECAR_PATH,
    }
    sidecar_path = trace_base + "_provenance.json"
    with open(sidecar_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"Provenance sidecar written to: {sidecar_path}")

    if validation_failed:
        print("\n[FATAL] Worker trace validation failed. "
              "PID files preserved for diagnosis.", file=sys.stderr)
        sys.exit(2)

    # Clean up PID files after provenance is recorded
    for tf in trace_files:
        if os.path.exists(tf):
            os.remove(tf)
    print(f"Cleaned up {len(trace_files)} PID-suffixed trace file(s)")

    # Report scheduler sidecar stats
    if _SIDECAR_PATH and os.path.exists(_SIDECAR_PATH):
        with open(_SIDECAR_PATH) as f:
            sidecar_lines = f.readlines()
        print(f"[sidecar] Scheduler sidecar: {len(sidecar_lines)} steps recorded → {_SIDECAR_PATH}")
    else:
        print("[sidecar] WARNING: scheduler sidecar file not produced", file=sys.stderr)

    print("\nSUCCESS: vLLM serving emulation completed!")


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if _SIDECAR_PTH_PATH is not None:
            from src.serve_trace.sidecar_lifecycle import cleanup_sidecar_pth
            cleanup_sidecar_pth(_SIDECAR_PTH_PATH)
