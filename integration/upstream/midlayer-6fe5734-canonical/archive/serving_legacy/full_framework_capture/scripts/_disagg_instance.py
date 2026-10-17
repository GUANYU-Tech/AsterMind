#!/usr/bin/env python3
"""
Run a single vLLM disaggregated instance (prefill or decode) under CUDA emulation.

This is the per-instance helper for scripts/serve_trace/trace_disagg.py. It configures
the vLLM engine with KV transfer settings (kv_connector, kv_role, kv_rank,
kv_parallel_size, kv_ip, kv_port) and transparently passes all engine args.

Not intended to be called directly — invoked by the disagg orchestrator.
"""
import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

_SIDECAR_PTH_PATH = None


def timeout_handler(signum, frame):
    print("\n[TIMEOUT] Exiting", file=sys.stderr)
    sys.exit(1)


def generate_prompt(length_tokens, tokenizer=None):
    vocab = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "artificial", "intelligence", "machine", "learning", "neural",
        "network", "transformer", "attention", "mechanism", "language",
        "model", "generation", "inference", "serving", "batching",
        "throughput", "latency", "optimization", "accelerator", "GPU",
        "tensor", "parallel", "data", "pipeline", "distributed", "system",
    ]
    num_words = int(length_tokens * 0.75)
    prompt = " ".join(random.choices(vocab, k=num_words))
    if tokenizer is not None:
        for _ in range(10):
            toks = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if abs(len(toks) - length_tokens) <= max(10, length_tokens // 10):
                break
            num_words = int(num_words * length_tokens / max(len(toks), 1))
            prompt = " ".join(random.choices(vocab, k=num_words))
    return prompt


def make_disagg_request_id(role, index, kv_ip, kv_port, peer_kv_port):
    """Generate a shared request ID with both P2P address markers.

    Markers encode ZMQ (kv_port) addresses, not HTTP ports. The connector
    parses these to find the peer's ZMQ endpoint for send/recv.
    """
    base_id = f"req-{index}"
    if role == "prefill":
        prefill_port = kv_port
        decode_port = peer_kv_port
    else:
        prefill_port = peer_kv_port
        decode_port = kv_port
    return (f"___prefill_addr_{kv_ip}:{prefill_port}___"
            f"{base_id}___decode_addr_{kv_ip}:{decode_port}")


def _setup_scheduler_sidecar(trace_base):
    """Set up scheduler sidecar for this instance.

    Sets CUDA_SIDECAR_PATH env var and calls the installer.
    Returns the sidecar file path.
    """
    sidecar_path = f"{trace_base}_scheduler.jsonl"
    if os.path.exists(sidecar_path):
        os.remove(sidecar_path)
    os.environ["CUDA_SIDECAR_PATH"] = sidecar_path
    global _SIDECAR_PTH_PATH
    from scripts.serve_trace.trace_serving import _install_scheduler_sidecar
    _SIDECAR_PTH_PATH = _install_scheduler_sidecar()
    return sidecar_path


def _run_disagg_instance(args, trace_base, sidecar_path, plugin):
    """Run one disagg instance in plugin (approach-D) mode: real vLLM with KV
    transfer, operator capture, then synthesize the kernel trace from the
    captured real operator sequence. Mirrors the real-mode run loop but uses no
    LD_PRELOAD and converts operators->trace at the end.
    """
    import glob
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)["prompts"]
    else:
        prompts = [generate_prompt(args.prompt_len, tokenizer)
                   for _ in range(args.num_requests)]

    engine_args = json.loads(args.engine_args_json)
    kv_role = "kv_producer" if args.role == "prefill" else "kv_consumer"
    kv_rank = 0 if args.role == "prefill" else 1
    kv_transfer_config = {
        "kv_connector": args.kv_connector, "kv_role": kv_role, "kv_rank": kv_rank,
        "kv_parallel_size": 2, "kv_ip": args.kv_ip, "kv_port": args.kv_port,
        "kv_connector_extra_config": {"http_port": args.http_port},
    }
    engine_kwargs = {
        "model": args.model, "dtype": "float16", "gpu_memory_utilization": 0.8,
        "enforce_eager": True, "max_model_len": args.seq_len,
        "trust_remote_code": True, "enable_chunked_prefill": True,
        "kv_transfer_config": kv_transfer_config,
    }
    engine_kwargs.update(engine_args)
    print(f"[{args.role}] plugin mode: KV role={kv_role} rank={kv_rank} "
          f"connector={args.kv_connector}")

    from vllm import LLM, SamplingParams
    llm = LLM(**engine_kwargs)

    def _disagg_run_engine(use_tqdm=False):
        outputs = []
        while llm.llm_engine.has_unfinished_requests():
            for output in llm.llm_engine.step():
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: x.request_id)
    llm._run_engine = _disagg_run_engine

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                     min_tokens=args.max_tokens, ignore_eos=True)
    request_ids = [make_disagg_request_id(args.role, i, args.kv_ip, args.kv_port,
                                          args.peer_kv_port)
                   for i in range(len(prompts))]
    llm.request_counter = iter(request_ids)

    # When this instance is wrapped by `nsys profile --capture-range=cudaProfilerApi`
    # (set by trace_disagg --nsys-role), gate the captured window to ONLY the real
    # inference generate() so warmup/init kernels are excluded. This requires the
    # engine to run in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0, also set by the
    # wrapper) so cudaProfilerStart/Stop reach the process the kernels launch in.
    ground_truth_only = os.environ.get("PD_NSYS_PROFILE") == "1"
    if ground_truth_only:
        import torch
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.synchronize()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    if ground_truth_only:
        import torch
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    print(f"[{args.role}] completed {len(outputs)} requests")
    del llm
    import gc; gc.collect(); time.sleep(2)

    return _finalize_plugin_instance(
        args, trace_base, sidecar_path, kv_transfer_config, engine_args,
        ground_truth_only=ground_truth_only)


def _finalize_plugin_instance(args, trace_base, sidecar_path, kv_transfer_config,
                              engine_args, ground_truth_only,
                              op_trace_glob=None):
    """Post-generate handling for a plugin disagg instance.

    Two distinct, explicit branches:

    - ground_truth_only=True (PD_NSYS_PROFILE=1): the instance is wrapped by
      `nsys profile` to capture a role-matched kernel ground truth, and runs the
      engine in-process so the cudaProfiler range reaches the kernels. In-process
      mode means the operator-capture `.pth` hook (loaded only at child-interpreter
      startup) does NOT install, so there are no operator records and NO synth
      trace is produced here. This is intended: the synth trace for this role comes
      from a SEPARATE non-profiled run on the same workload. Write a
      ground-truth-only provenance and return success without scanning/converting
      operator records.

    - ground_truth_only=False (normal plugin run): convert the captured real
      operator sequence (rank 0) into the synthesized kernel trace; fail loudly
      (return 1) if no operator records were captured.

    `op_trace_glob` overrides the operator-trace glob for testing; defaults to the
    CUDA_OPERATOR_TRACE_PATH env the operator patch writes to.
    """
    if ground_truth_only:
        provenance = {
            "role": args.role, "mode": "plugin-nsys-ground-truth",
            "ground_truth_only": True, "nsys_profile": True,
            "producer": "real-vLLM-on-GPU (nsys kernel capture)",
            "scheduler": "real-vLLM", "kv_transfer_config": kv_transfer_config,
            "model": args.model, "hw_config": args.hw_config,
            "seed": args.seed, "max_tokens": args.max_tokens,
            "prompts_file": args.prompts_file,
            "scheduler_sidecar": sidecar_path,
            "synthesized_trace": None,
        }
        with open(f"{trace_base}_provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)
        print(f"[{args.role}] nsys ground-truth-only capture: no synth trace "
              f"(matched by workload to a separate non-profiled run)")
        print(f"[{args.role}] SUCCESS")
        return 0

    # Normal plugin run: convert captured real operator sequence -> synth trace.
    import glob
    glob_pat = op_trace_glob or (os.environ["CUDA_OPERATOR_TRACE_PATH"] + ".*")
    op_files = {f: sum(1 for _ in open(f)) for f in glob.glob(glob_pat)}
    op_files = {f: n for f, n in op_files.items() if n > 0}
    if not op_files:
        print(f"[{args.role}] ERROR: no operator records captured", file=sys.stderr)
        return 1
    rank0 = next((f for f in op_files if f.endswith(".rank0")),
                 max(op_files, key=op_files.get))
    from src.serve_trace.operator_records import (
        read_operator_records, OperatorTraceGenerator)
    from src.serve_predict.model_config import load_model_config
    from src.kernel.hardware_spec import HWSpec
    records = read_operator_records(rank0)
    gen = OperatorTraceGenerator(load_model_config(args.model),
                                 HWSpec(args.hw_config),
                                 tp_size=engine_args.get("tensor_parallel_size", 1),
                                 pp_size=engine_args.get("pipeline_parallel_size", 1),
                                 ep_size=engine_args.get("expert_parallel_size", 1))
    trace_entries = gen.records_to_trace_direct(records)
    with open(trace_base, "w") as f:
        json.dump(trace_entries, f)
    provenance = {
        "role": args.role, "mode": "plugin",
        "producer": "recipe-from-real-operators", "scheduler": "real-vLLM",
        "operators": "real-ModelRunner", "kv_transfer_config": kv_transfer_config,
        "model": args.model, "hw_config": args.hw_config,
        "trace_output": trace_base, "scheduler_sidecar": sidecar_path,
        "operator_trace": rank0,
    }
    with open(f"{trace_base}_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[{args.role}] {len(records)} op records -> {len(trace_entries)} "
          f"priced kernels -> {trace_base}")
    print(f"[{args.role}] SUCCESS")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run single disagg vLLM instance")
    parser.add_argument("--role", required=True, choices=["prefill", "decode"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-args-json", default="{}")
    parser.add_argument("--kv-connector", required=True)
    parser.add_argument("--kv-port", type=int, default=14579)
    parser.add_argument("--kv-ip", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8000,
                        help="HTTP port for P2pNcclConnector metadata")
    parser.add_argument("--peer-kv-port", type=int, default=14580,
                        help="Peer instance ZMQ kv_port (for request ID address markers)")
    parser.add_argument("--num-requests", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--hw-config", default="configs/hardware/nvidia_a800.yaml")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--mode", choices=["real", "plugin"], default="real",
                        help="real = LD_PRELOAD CUDA emulation (set by parent env); "
                             "plugin = approach-D operator capture on real GPU.")
    args = parser.parse_args()

    random.seed(args.seed)

    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    if args.mode == "plugin":
        # approach-D: real vLLM on GPU, capture real scheduler sidecar + operator
        # sequence; no LD_PRELOAD / no fake CUDA emulation config.
        trace_base = args.trace_output
        os.environ["CUDA_OPERATOR_TRACE_PATH"] = trace_base + "_operators.jsonl"
        sidecar_path = trace_base + "_scheduler.jsonl"
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        os.environ["CUDA_SIDECAR_PATH"] = sidecar_path
        import glob as _glob
        for _f in _glob.glob(os.environ["CUDA_OPERATOR_TRACE_PATH"] + ".*"):
            os.remove(_f)
        from src.serve_trace.sidecar_lifecycle import install_sidecar_pth
        global _SIDECAR_PTH_PATH
        _SIDECAR_PTH_PATH = install_sidecar_pth(project_root=_root)
        # Install hooks directly in this process so forked children inherit them
        # (the .pth file only covers spawn mode).
        from src.serve_trace._sidecar_patch import install_sidecar_patch
        from src.serve_trace._operator_patch import install_operator_patch
        install_sidecar_patch()
        install_operator_patch()
        return _run_disagg_instance(args, trace_base, sidecar_path, plugin=True)

    # Setup emulation config
    from src.serve_trace.fake_cuda import load_emulation_config as _load_emulation_config, write_emulation_config_json as _write_emulation_json

    emu_cfg = _load_emulation_config(args.hw_config)
    trace_base = args.trace_output
    os.environ["CUDA_TRACE_OUTPUT"] = trace_base
    config_json_path = _write_emulation_json(emu_cfg, trace_output=trace_base)
    os.environ["CUDA_EMULATE_CONFIG"] = config_json_path

    # Install scheduler sidecar
    sidecar_path = _setup_scheduler_sidecar(trace_base)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    # MoE routing patch
    import vllm.model_executor.layers.fused_moe.fused_moe as _fused_moe_mod
    from src.serve_trace.moe_topk import emulation_fused_topk

    _fused_moe_mod.fused_topk = emulation_fused_topk

    # Build prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    if args.prompts_file:
        with open(args.prompts_file) as f:
            workload = json.load(f)
        prompts = workload["prompts"]
    else:
        prompts = [generate_prompt(args.prompt_len, tokenizer) for _ in range(args.num_requests)]

    # Parse engine args
    engine_args = json.loads(args.engine_args_json)

    # Build KV transfer config
    kv_role = "kv_producer" if args.role == "prefill" else "kv_consumer"
    kv_rank = 0 if args.role == "prefill" else 1
    kv_transfer_config = {
        "kv_connector": args.kv_connector,
        "kv_role": kv_role,
        "kv_rank": kv_rank,
        "kv_parallel_size": 2,
        "kv_ip": args.kv_ip,
        "kv_port": args.kv_port,
        "kv_connector_extra_config": {"http_port": args.http_port},
    }

    # Build vLLM engine kwargs
    engine_kwargs = {
        "model": args.model,
        "dtype": "float16",
        "gpu_memory_utilization": 0.8,
        "enforce_eager": True,
        "max_model_len": args.seq_len,
        "trust_remote_code": True,
        "enable_chunked_prefill": True,
        "kv_transfer_config": kv_transfer_config,
    }
    # Transparently pass all engine args from JSON
    engine_kwargs.update(engine_args)

    print(f"[{args.role}] KV config: role={kv_role}, rank={kv_rank}, "
          f"connector={args.kv_connector}, port={args.kv_port}")
    print(f"[{args.role}] Engine args: {engine_args}")

    from vllm import LLM, SamplingParams

    print(f"[{args.role}] Creating LLM engine...")
    llm = LLM(**engine_kwargs)
    print(f"[{args.role}] Engine created!")

    # Replace _run_engine to handle non-integer decorated request IDs.
    # vLLM's LLM._run_engine sorts by int(request_id) at line 1629,
    # which crashes with P2P connector-decorated IDs. We replace the
    # method with a minimal loop that sorts by string instead.
    def _disagg_run_engine(use_tqdm=False):
        outputs = []
        while llm.llm_engine.has_unfinished_requests():
            step_outputs = llm.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: x.request_id)
    llm._run_engine = _disagg_run_engine

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Override LLM's request_counter so decorated IDs reach _add_request().
    # vLLM 0.11 LLM.generate() does not accept a request_id parameter;
    # internally it calls str(next(self.request_counter)) per request.
    request_ids = [make_disagg_request_id(args.role, i, args.kv_ip, args.kv_port, args.peer_kv_port)
                   for i in range(len(prompts))]
    llm.request_counter = iter(request_ids)

    inference_start_us = int(time.time() * 1e6)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    inference_end_us = int(time.time() * 1e6)
    total_time = (inference_end_us - inference_start_us) / 1e6

    print(f"[{args.role}] Completed: {len(outputs)} requests in {total_time:.2f}s")

    del llm
    import gc
    gc.collect()
    time.sleep(2)

    # Write provenance
    provenance = {
        "role": args.role,
        "kv_transfer_config": kv_transfer_config,
        "engine_args": engine_args,
        "model": args.model,
        "hw_config": args.hw_config,
        "num_requests": args.num_requests,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "trace_output": trace_base,
        "scheduler_sidecar": sidecar_path,
        "inference_window_us": [inference_start_us, inference_end_us],
        "total_time_s": total_time,
    }
    prov_path = f"{trace_base}_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"[{args.role}] SUCCESS")


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = main()
    finally:
        if _SIDECAR_PTH_PATH is not None:
            from src.serve_trace.sidecar_lifecycle import cleanup_sidecar_pth
            cleanup_sidecar_pth(_SIDECAR_PTH_PATH)
    # Propagate the instance return code so the orchestrator (trace_disagg) sees a
    # real failure instead of an accidental zero. main() returns the plugin-path
    # rc (0/1); the real (LD_PRELOAD) path returns None -> treat as success (0).
    sys.exit(_rc or 0)
