#!/usr/bin/env python3
"""GEMM runner for nsys-based LUT build.

Runs F.linear (cuBLAS GEMM) for each (M,N,K) shape in SHAPES_FILE (JSON list
of [M,N,K]), in order, under a cudaProfilerApi capture range. nsys captures
ONLY the capture range (warmup is outside). The captured GEMM kernels are in
execution order == shape order, so the parser maps kernel[i] <-> shape[i]
directly (no index-pairing, no async — this is a single-stream in-order
program).

Run under nsys:
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force=true -o <out> <python> gemm_runner.py <shapes.json>
"""
import json, sys, torch
import torch.nn.functional as F

shapes = json.load(open(sys.argv[1]))  # [[M,N,K], ...]
dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[sys.argv[2]]
dev = "cuda"


def _run(shapes):
    for M, N, K in shapes:
        x = torch.randn(M, K, dtype=dt, device=dev)
        w = torch.randn(N, K, dtype=dt, device=dev)
        F.linear(x, w)
    torch.cuda.synchronize()


# Warmup (OUTSIDE capture) — cuBLAS lazy-init / JIT
_run(shapes)
torch.cuda.empty_cache()

# Capture range: re-run all shapes; nsys captures only this segment.
torch.cuda.cudart().cudaProfilerStart()
_run(shapes)
torch.cuda.cudart().cudaProfilerStop()
torch.cuda.synchronize()
print(f"DONE {len(shapes)} shapes", file=sys.stderr)
