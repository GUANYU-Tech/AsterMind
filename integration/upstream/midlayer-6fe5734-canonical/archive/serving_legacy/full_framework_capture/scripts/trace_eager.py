"""
Run vLLM under CUDA emulation and capture kernel trace.

Usage:
    # From project root with LD_PRELOAD set:
    CUDA_VISIBLE_DEVICES=0 \
    LD_PRELOAD="build/fake_cuda_libs/libcudart.so.12:build/fake_cuda_libs/libcublas.so.12:build/fake_cuda_libs/libcublasLt.so.12" \
    LD_LIBRARY_PATH=build/fake_cuda_libs \
    CUDA_TRACE_OUTPUT=/tmp/vllm_trace.json \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    CUDA_HOME=/usr/local/cuda-12.2 \
    python scripts/serve_trace/trace_eager.py [--model MODEL] [--max-tokens N] [--seq-len N]

Outputs trace to CUDA_TRACE_OUTPUT.<pid> (the EngineCore child process).
After running, use src/serve_trace parse_trace() or manually find the largest .pid file.
"""
import argparse
import glob
import json
import os
import signal
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def timeout_handler(signum, frame):
    print("\n[TIMEOUT] Exiting", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run vLLM under CUDA emulation")
    parser.add_argument("--model", default="/share_data/public_models/Qwen3-0.6B",
                        help="Model path")
    parser.add_argument("--max-tokens", type=int, default=8,
                        help="Max tokens to generate per request")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Max model sequence length")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout in seconds")
    parser.add_argument("--prompt", default="Hello, my name is",
                        help="Input prompt")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--hw-config", default="configs/hardware/nvidia_a800.yaml",
                        help="Path to hw config yaml for emulation_config (default: A800)")
    args = parser.parse_args()

    # Write emulation config JSON and set CUDA_EMULATE_CONFIG env var so the
    # C runtime reads the correct GPU props (total_memory, major/minor, SM count).
    import sys as _sys
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from src.serve_trace.fake_cuda import load_emulation_config as _load_emulation_config, write_emulation_config_json as _write_emulation_json
    from src.kernel.validation.dataset_io import get_emulation_output_dir
    emu_cfg = _load_emulation_config(args.hw_config)
    trace_base = os.environ.get("CUDA_TRACE_OUTPUT")
    if trace_base is None:
        output_dir = get_emulation_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        trace_base = os.path.join(output_dir, "eager_trace.json")
        os.environ["CUDA_TRACE_OUTPUT"] = trace_base
    config_json_path = _write_emulation_json(emu_cfg, trace_output=trace_base)
    os.environ["CUDA_EMULATE_CONFIG"] = config_json_path
    print(f"[P36] emulation_config: {args.hw_config} → {config_json_path}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    import torch
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"Memory: free={free/1024**3:.1f}GiB, total={total/1024**3:.1f}GiB")

    # ── MoE routing patch for emulation ──
    # In emulation, CUDA kernels don't execute, so MoE gating output can become
    # degenerate. Patch fused_topk with deterministic diverse expert routing.
    import vllm.model_executor.layers.fused_moe.fused_moe as _fused_moe_mod
    from src.serve_trace.moe_topk import emulation_fused_topk

    _fused_moe_mod.fused_topk = emulation_fused_topk
    print("[EMULATION] Patched fused_topk with uniform random MoE routing")

    from vllm import LLM, SamplingParams

    print(f"\nCreating LLM engine (model={args.model})...")
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        max_model_len=args.seq_len,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print("LLM engine created!")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,  # Force full decode to match nsys step count
        ignore_eos=True,             # Don't stop on EOS (emulation outputs zeros)
    )
    print(f"\nRunning inference (prompt={repr(args.prompt)}, max_tokens={args.max_tokens})...")
    outputs = llm.generate([args.prompt], sampling_params)
    print(f"Token IDs: {outputs[0].outputs[0].token_ids}")
    print("SUCCESS: vLLM inference completed under CUDA emulation!")

    # CUDA_TRACE_OUTPUT is required: it is set both by the C emulation layer
    # (LD_PRELOAD) and by callers of this script. Missing means the env was
    # not configured for emulation, which is a hard error rather than a path
    # to silently fall back to.
    trace_base = os.environ["CUDA_TRACE_OUTPUT"]
    trace_files = glob.glob(f"{trace_base}.*")
    if trace_files:
        # For TP>1, merge all worker traces into one
        all_traces = []
        for tf in sorted(trace_files, key=os.path.getsize, reverse=True):
            with open(tf) as f:
                all_traces.extend(json.load(f))
        
        print(f"\nTrace files: {len(trace_files)}")
        print(f"Total kernel launches: {len(all_traces)}")
        names = {}
        for e in all_traces:
            n = e["kernel_name"]
            if n not in names:
                names[n] = 0
            names[n] += 1
        print(f"Unique kernels: {len(names)}")
        
        # Write merged trace
        canonical = trace_base
        with open(canonical, "w") as f:
            json.dump(all_traces, f)
        print(f"Merged trace written to: {canonical}")

        # Write provenance sidecar with capture metadata
        import time as _time
        provenance = {
            "capture_command": " ".join(sys.argv),
            "capture_timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "hw_config": args.hw_config,
            "max_tokens": args.max_tokens,
            "seq_len": args.seq_len,
            "tp_size": args.tensor_parallel_size,
            "prompt": args.prompt,
        }
        sidecar_path = trace_base + "_provenance.json"
        with open(sidecar_path, "w") as f:
            json.dump(provenance, f, indent=2)
        print(f"Provenance sidecar written to: {sidecar_path}")

        # Clean up pid files
        for f in trace_files:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
