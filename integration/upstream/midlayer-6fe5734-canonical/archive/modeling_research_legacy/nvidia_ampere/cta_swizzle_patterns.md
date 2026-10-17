# CTA-to-SM Scheduling and Threadblock Swizzle Patterns for GEMM on Ampere (SM80)

## Research Summary

---

## 1. CTA Swizzle Patterns in CUTLASS/cuBLAS

### Patterns Available in CUTLASS (from `threadblock_swizzle.h`):

| Swizzle | Template | Description |
|---------|----------|-------------|
| **Identity<1>** | `GemmIdentityThreadblockSwizzle<1>` | Column-major: blockIdx.x → M, blockIdx.y → N (no swizzle) |
| **Identity<2>** | `GemmIdentityThreadblockSwizzle<2>` | 2-wide column groups (log_tile=1) |
| **Identity<4>** | `GemmIdentityThreadblockSwizzle<4>` | 4-wide column groups (log_tile=2) |
| **Identity<8>** | `GemmIdentityThreadblockSwizzle<8>` | 8-wide column groups (log_tile=3) |
| **Horizontal** | `GemmHorizontalThreadblockSwizzle` | Row-major: blockIdx.x → N, blockIdx.y → M |
| **StreamK** | `ThreadblockSwizzleStreamK` | Persistent + cohort rasterization (8×4 blocks) |

### The "Identity Swizzle" Algorithm (the key one for SM80 GEMM):

The `GemmIdentityThreadblockSwizzle<N>` template implements a **banded column-major** swizzle:

```
Grid launch: dim3(M_tiles * tile_width, ceil(N_tiles / tile_width), split_k)

get_tile_offset(log_tile):
  m_idx = blockIdx.x >> log_tile          // divide x by tile_width
  n_idx = (blockIdx.y << log_tile) + (blockIdx.x & ((1 << log_tile) - 1))
```

**What this does**: Instead of traversing tiles in pure column-major (all M for one N), it groups `tile_width` consecutive N-columns together. CTAs with consecutive blockIdx.x values map to different N positions within a band of width `tile_width`, then advance in M.

**Adaptive selection**: `get_log_tile()` picks the width automatically:
- N_tiles >= 6 and N >= 8 → log_tile = 3 (8-wide bands)
- N_tiles >= 3 and N >= 4 → log_tile = 2 (4-wide bands)
- N_tiles >= 2 and N >= 2 → log_tile = 1 (2-wide bands)
- Otherwise → log_tile = 0 (pure column-major)

### What cuBLAS actually uses for `ampere_fp16_s16816gemm`:

cuBLAS is closed-source, but based on:
1. CUTLASS being NVIDIA's reference implementation that tracks cuBLAS kernels
2. The default swizzle in CUTLASS SM80 GEMM being `GemmIdentityThreadblockSwizzle<>`  (N=1, auto-adapts)
3. NCU profiling observations showing grid dimensions consistent with this pattern

**cuBLAS likely uses the Identity swizzle (column-major with banding)** — the same pattern as CUTLASS. The specific band width depends on problem dimensions. For typical large GEMM sizes on A100, this would be a 4-wide or 8-wide band pattern.

**No Hilbert, snake, or Z-order curves** exist in CUTLASS. These were theoretical proposals in academic papers but NVIDIA did not implement them — the simpler banded column-major pattern is sufficient.

---

## 2. Hardware GPC Round-Robin CTA Scheduling

### A100 (GA100) Architecture:
- **8 GPCs** (Graphics Processing Clusters), each with **16 SMs** on A100 (128 SMs total; A800 has 108 SMs enabled)
- Each GPC has **8 TPCs** (2 SMs per TPC)

### CTA-to-SM Assignment (Hardware Scheduler):
The CUDA hardware scheduler distributes CTAs (thread blocks) to SMs using a **GPC-based round-robin** mechanism:

1. **Block dispatcher** in the GPU schedules CTAs from the grid in linear order of `blockIdx` (flattened: `blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y`)
2. Distribution is **round-robin across GPCs first**, then across SMs within a GPC
3. The order is approximately: CTA0→GPC0/SM0, CTA1→GPC1/SM0, ..., CTA7→GPC7/SM0, CTA8→GPC0/SM1, etc.
4. **This is NOT guaranteed** — it depends on SM availability and occupancy limits

### Key Insight for Cache Reuse:
- CTAs with **consecutive linear blockIdx** are distributed to **different GPCs** (not same GPC)
- CTAs assigned to the **same GPC** share the **same L2 cache partition** (L2 is physically partitioned across memory controllers associated with GPCs)
- A100 has **40MB L2** cache, partitioned into sectors aligned with HBM channels

### CUTLASS SM90's `get_max_cta_occupancy()` function explicitly models this:
```cpp
constexpr int max_sm_per_gpc = 18;  // For GH100
int min_num_gpc = sm_count / max_sm_per_gpc;
int max_cta_occupancy_per_gpc = max_sm_per_gpc - (max_sm_per_gpc % cluster_size);
```

---

## 3. CTA Traversal Order and L2 Cache Reuse

### The Problem:
For C = A × B where A is M×K and B is K×N:
- **CTAs in the same M-row** (same tile_m, different tile_n) share the **same A panel** (rows of A)
- **CTAs in the same N-column** (same tile_n, different tile_m) share the **same B panel** (columns of B)

### How Swizzle Affects L2 Reuse:

**Pure column-major** (log_tile=0): CTAs 0,1,2,... all have same tile_n but different tile_m. They share B panels but NOT A panels. When M is large, by the time CTA_k finishes with A_row_k, CTA_k+1's A_row is probably not in L2.

**Banded swizzle** (log_tile=2, 4-wide): Consecutive CTAs span a 4-column band:
```
CTA ordering (linearized):
  (m=0,n=0), (m=0,n=1), (m=0,n=2), (m=0,n=3),  ← share A[0] panel!
  (m=1,n=0), (m=1,n=1), (m=1,n=2), (m=1,n=3),  ← share A[1] panel!
  ...
```
Now CTAs that execute close in time (and thus likely land on the same wave of SMs) share A panels within the band. This **dramatically improves A-panel L2 reuse**.

**Raster order heuristic** (SM90+): The scheduler picks `AlongN` vs `AlongM` based on which dimension is larger:
- If `tiles_n > tiles_m` → raster AlongM (the major sweep is along M, so consecutive blocks in a column share B panels)
- If `tiles_m >= tiles_n` → raster AlongN (the major sweep is along N, so consecutive blocks in a row share A panels)

This ensures the sweep goes along the **longer dimension**, maximizing the number of CTAs that share the **panel from the shorter dimension** (which fits better in L2).

### StreamK Cohort Rasterization:
The StreamK swizzle uses **8×4 cohorts** — blocks of 8 rows × 4 columns of tiles. All 32 CTAs in a cohort are assigned consecutive linear indices, so they execute close in time and share both A and B panels within the cohort. The cohort dimensions (8M × 4N) are chosen to balance A-panel vs B-panel sharing.

---

## 4. Documentation and Code for `ampere_fp16_s16816gemm` Kernels

### CUTLASS Source Code:
- **Main swizzle implementation**: `include/cutlass/gemm/threadblock/threadblock_swizzle.h`
- **SM80 default GEMM kernel**: `include/cutlass/gemm/kernel/default_gemm.h` (template param `ThreadblockSwizzle`)
- **Default swizzle**: `GemmIdentityThreadblockSwizzle<>` (N=1, auto-adapts based on problem size)
- **SM90+ persistent scheduler**: `include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp` — implements `log_swizzle_size` + `RasterOrder`

### Key Algorithm (SM90 `get_work_idx_m_and_n`):
```cpp
// Swizzle decomposition:
offset = cluster_id & ((1 << log_swizzle_size) - 1);   // position within swizzle band
extra = cluster_id >> log_swizzle_size;                  // which full band
divmod(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);
cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;
```
This is the generalized form of the SM80 banded swizzle, extended for cluster-aware scheduling.

### cuBLAS Kernel Names:
The naming `ampere_fp16_s16816gemm` refers to:
- `ampere` = SM80 architecture
- `fp16` = half-precision
- `s16816` = mma.sync instruction shape (m16n8k16)
- `gemm` = general matrix multiply

These are cuBLAS internal kernels that implement essentially the same swizzle as CUTLASS Identity<N>.

### Key Papers and Presentations:
1. **"Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU"** (Osama et al., PPoPP 2023) — describes cohort rasterization for L2 reuse
2. **GTC 2020: "Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100"** (S21745) — discusses threadblock swizzle for L2 optimization
3. **CUTLASS documentation**: `media/docs/efficient_gemm.md` — "Threadblock Rasterization" section
4. **"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"** (Jia et al.) — documents GPC round-robin scheduling

---

## 5. Inferring Swizzle from NCU Profiling

### Yes, the swizzle pattern CAN be inferred from NCU data:

#### Method 1: Grid Dimensions
From NCU's "Launch Statistics":
- `Grid Size` shows the launched grid `(X, Y, Z)`
- Compare with expected `(M_tiles, N_tiles, 1)` for no swizzle
- If you see `(M_tiles * tile_width, ceil(N_tiles/tile_width), 1)`, the `tile_width` is the swizzle factor

Example: For M=4096, N=4096 with tile 256×128:
- No swizzle: grid = (16, 32, 1)
- Swizzle<4>: grid = (16*4, 32/4, 1) = (64, 8, 1)

#### Method 2: Source Metrics per CTA
NCU's "Source" page with `--set full --launch-count all` shows per-block L2 hit rates. With proper swizzle, blocks in the same wave should show high L2 hit rates for the shared operand.

#### Method 3: Memory Access Pattern Analysis
Using NCU `--section MemoryWorkloadAnalysis`:
- `l2_tex_hit_rate` will be significantly higher (often 2-3x) with swizzle vs without
- `dram_bytes_read` will be lower with effective swizzle

#### Method 4: Block-to-SM mapping
NCU with `--metrics smsp__block_id` or examining the "Occupancy" section can reveal which blocks run on which SM. Plotting (block_id → SM_id) reveals the round-robin pattern.

#### Method 5: Kernel Name Inspection
For cuBLAS: `ncu --print-kernel-id` shows kernel names like `ampere_fp16_s16816gemm_256x128_ldg8_...`
The suffix sometimes encodes tile configuration but NOT the swizzle directly. The swizzle is internal to the kernel's CTA indexing logic.

#### Method 6: Compare Empirical L2 Traffic
Profile the same GEMM with different M/N ratios. The swizzle adapts to problem shape:
- Very tall (M >> N): expect column-major (blocks share B panels)
- Very wide (N >> M): expect row-major (blocks share A panels)
- Square: expect banded pattern

---

## Summary Table: Swizzle Selection Logic

| Condition | log_tile | Band Width | Traversal Pattern |
|-----------|----------|------------|-------------------|
| min(M_tiles, N_tiles) >= 6 | 3 | 8 | 8-column bands, sweep in M |
| min(M_tiles, N_tiles) >= 3 | 2 | 4 | 4-column bands, sweep in M |
| min(M_tiles, N_tiles) >= 2 | 1 | 2 | 2-column bands, sweep in M |
| Otherwise | 0 | 1 | Pure column-major |

For SM90+ (persistent scheduler), the raster direction is also chosen:
- `tiles_n > tiles_m` → raster AlongM (sweep in M, share B panels)
- `tiles_m >= tiles_n` → raster AlongN (sweep in N, share A panels)
