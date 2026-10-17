#!/usr/bin/env python3
"""CPU-only prebuild of the FlashInfer 0.6.18 JIT module GLM-5.2 needs on SM90.

vLLM 0.30 runs GLM-5.2's FP8 block-128 dense linears (N % 64 == 0,
K % 128 == 0) through ``FlashInferFp8DeepGEMMDynamicBlockScaledKernel``:
M < 32 uses FlashInfer ``fp8_blockscale_gemm_sm90`` (JIT module
``gen_fp8_blockscale_gemm_sm90_module``), M >= 32 uses DeepGEMM.  The
installed FlashInfer has no AOT/cubin package, so the module would otherwise
be compiled with nvcc during server start.  This builds it into the same
cache the server uses (``HOME=/mnt/public/yangxinhao/h200_microbench/home``
-> ``~/.cache/flashinfer/0.6.18.post1/90a/cached_ops``).  nvcc only, no GPU.
The per-shape kernels it JIT-compiles at run time (TensorRT-LLM DeepGEMM
path, ``~/.tensorrt_llm``) still compile on first use; the smoke run warms
them.

Usage (h200-tmp, after any other nvcc prebuild on the host has finished):
    CUDA_VISIBLE_DEVICES= FLASHINFER_CUDA_ARCH_LIST=9.0 \
    HOME=/mnt/public/yangxinhao/h200_microbench/home \
    CUDA_HOME=/mnt/public/yangxinhao/tools/cuda-12.9 \
    PATH=/mnt/public/yangxinhao/tools/cuda-12.9/bin:$PATH MAX_JOBS=8 \
    setsid nohup nice -n 19 /mnt/public/yangxinhao/envs/vllm030/bin/python \
        nvidia_h200_glm_5_2_tp8_prebuild_flashinfer.py > <log> 2>&1 &
"""

import os
import time

if os.environ["CUDA_VISIBLE_DEVICES"] != "":
    raise SystemExit("run with CUDA_VISIBLE_DEVICES= (CPU-only build)")
if os.environ["HOME"] != "/mnt/public/yangxinhao/h200_microbench/home":
    raise SystemExit("HOME must match the server's HOME so the cache is hit")

from flashinfer.jit.gemm.fp8_blockscale import gen_fp8_blockscale_gemm_sm90_module

spec = gen_fp8_blockscale_gemm_sm90_module()
start = time.time()
print("build", spec.name, "->", spec.get_library_path(), flush=True)
spec.build()
print("built", spec.name, spec.is_compiled, round(time.time() - start, 1), "s",
      flush=True)
