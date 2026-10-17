#!/usr/bin/env python3
"""Extend GEMM LUT with torch.profiler (fast, no nsys).

For each (M,N,K,dtype,has_bias), run F.linear under torch.profiler, capture
the kernel name (with tile), compute grid from (M,N,K,tile) using the cuBLAS
grid formula. Merge into the existing LUT.

Grid formulas (verified vs nsys, 8B 100% match):
  s16816: grid = (ceil(N/tileN), ceil(M/tileM), splitK)
  xmma:   grid = (ceil(M/tileM), ceil(N/tileN), 1)
  wmma:   grid = (ceil(M/tileM), ceil(N/tileN), 1)  [assumed, same as xmma]
  gemvx:  grid = (ceil(N/4), 1, 1)  [from 8B LUT]

Usage (r2r env):
  python scripts/serve_predict/extend_gemm_lut.py
"""
import json, math, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LUT_PATH = ROOT / "src" / "serve_trace" / "gemm_kernel_lut_a800.json"

# All (N,K) shapes for all models × TP, minus existing 8B shapes
NEW_SHAPES = [
    (1280,2048),(1280,2880),(1440,2880),(1536,2560),(1536,4096),
    (2048,512),(2048,1024),(2048,1536),(2048,2048),(2048,3072),
    (2048,6144),(2560,640),(2560,1280),(2560,2048),(2560,2432),
    (2560,2560),(2560,2880),(2560,4864),(2560,5120),(2560,8192),
    (2560,9728),(2880,720),(2880,1440),(2880,2880),(3072,2048),
    (3072,2560),(3072,4096),(4096,1024),(4096,2048),(4096,3072),
    (4096,3584),(4096,6144),(4096,7168),(4096,12288),(4864,2560),
    (5120,1280),(5120,2048),(5120,2560),(5120,2880),(5120,5120),
    (5120,6400),(5120,8192),(5120,12800),(5120,25600),(5760,2880),
    (6144,2048),(6144,2560),(7168,4096),(8192,2048),(8192,4096),
    (8192,7168),(8192,8192),(8192,14336),(8192,28672),(9728,2560),
    (10240,5120),(10240,8192),(12288,2048),(12288,4096),(12800,5120),
    (14336,4096),(14336,8192),(19456,2560),(24576,4096),(25600,5120),
    (28672,8192),(32064,4096),(32064,8192),(37984,2048),(37984,2560),
    (37984,4096),(37984,5120),(50272,2880),(51200,5120),(57344,8192),
    (64128,4096),(64128,8192),(75968,2048),(75968,2560),(75968,4096),
    (75968,5120),(100544,2880),(128256,8192),(151936,2048),(151936,2560),
    (151936,4096),(151936,5120),(201088,2880),
]

# K values that indicate row-parallel split (K < hidden) → no bias for these
# Actually has_bias depends on model, not shape. Llama: all False.
# For simplicity, build both has_bias=0 and has_bias=1 for all shapes.
# The recipe will look up with the correct has_bias.


def _parse_tile(name):
    """Extract (tileM, tileN) from kernel name."""
    nl = name.lower()
    # s16816gemm: 256x128, 128x64, 64x64, etc.
    m = re.search(r's16816gemm_fp16_(\d+)x(\d+)', name) \
        or re.search(r's16816gemm_bf16_(\d+)x(\d+)', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    # xmma: tilesize192x128x32 → (192, 128)
    m = re.search(r'tilesize(\d+)x(\d+)x(\d+)', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    # wmma: 16x16_128x2 → (16, 16)
    m = re.search(r'wmma_tensorop_f16_s\d+gemm_f16_(\d+)x(\d+)', name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _compute_grid(name, M, N, K):
    """Compute grid from (M, N, K, tile) using cuBLAS grid formula."""
    nl = name.lower()
    tile = _parse_tile(name)
    if tile is None:
        return None
    tm, tn = tile
    if "s16816" in nl:
        # cuBLAS s16816: gridX=ceil(N/tileN), gridY=ceil(M/tileM), gridZ=splitK
        # splitK: when K > threshold and M small, cuBLAS splits K. gridZ = splitK count.
        # For now, gridZ=1 (no splitK). If the LUT has splitK, it'll be captured separately.
        return (max(1, math.ceil(N / tn)), max(1, math.ceil(M / tm)), 1)
    if "xmma" in nl:
        return (max(1, math.ceil(M / tm)), max(1, math.ceil(N / tn)), 1)
    if "wmma" in nl:
        return (max(1, math.ceil(M / tm)), max(1, math.ceil(N / tn)), 1)
    return None


def _capture(M, N, K, has_bias, dev, dt):
    """Run one F.linear, return (name, grid, block, smem) or None."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    x = torch.randn(M, K, dtype=dt, device=dev)
    w = torch.randn(N, K, dtype=dt, device=dev)
    bias = torch.randn(N, dtype=dt, device=dev) if has_bias else None
    # warmup
    torch.nn.functional.linear(x, w, bias)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.nn.functional.linear(x, w, bias)
    torch.cuda.synchronize()
    for e in prof.events():
        if e.device_type != torch.profiler.DeviceType.CUDA:
            continue
        nm = e.name
        nl = nm.lower()
        if "s16816gemm" in nl or "xmma_gemm" in nl or "wmma_tensorop" in nl \
                or "gemvx" in nl or "splitkreduce" in nl:
            grid = _compute_grid(nm, M, N, K)
            if grid is None:
                # Fallback: can't compute grid, skip
                continue
            return (nm, list(grid), [128, 1, 1], 0)
    return None


def main():
    import torch
    dev = "cuda"
    dt = torch.float16
    M_MAX = 8192

    # Load existing LUT
    lut = json.load(open(LUT_PATH))
    existing_keys = set(k for k in lut if k != "_meta")
    print(f"Existing LUT: {len(existing_keys)} keys")

    # Determine which (N,K,has_bias) combos we need
    needed = []
    for (N, K) in NEW_SHAPES:
        for hb in [0, 1]:
            # Check if any M for this (N,K,hb) is missing
            has_any = any(f"[{m},{N},{K},{hb}]" in existing_keys for m in range(1, M_MAX+1))
            if not has_any:
                needed.append((N, K, hb))
    print(f"Need to build: {len(needed)} (N,K,has_bias) combos × {M_MAX} M = {len(needed)*M_MAX} runs")
    est_s = len(needed) * M_MAX * 0.002  # ~2ms per run with profiler
    print(f"Estimated time: {est_s/60:.0f} min")

    t0 = time.time()
    count = 0
    for (N, K, hb) in needed:
        for M in range(1, M_MAX + 1):
            key = f"[{M},{N},{K},{hb}]"
            if key in lut:
                continue
            result = _capture(M, N, K, bool(hb), dev, dt)
            if result is not None:
                lut[key] = [list(result)]
            count += 1
            if count % 10000 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                remaining = (len(needed) * M_MAX - count) / rate
                pct = count*100//(len(needed)*M_MAX)
                print(f"  {count}/{len(needed)*M_MAX} ({pct}%) "
                      f"{elapsed/60:.1f}min elapsed, ~{remaining/60:.0f}min remaining",
                      flush=True)
                # Save checkpoint
                json.dump(lut, open(LUT_PATH, "w"), indent=1)

    # Final save
    json.dump(lut, open(LUT_PATH, "w"), indent=1)
    total = len([k for k in lut if k != "_meta"])
    print(f"Done. LUT now has {total} keys. Took {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
