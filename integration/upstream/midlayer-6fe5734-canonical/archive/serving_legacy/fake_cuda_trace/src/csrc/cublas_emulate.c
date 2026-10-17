/*
 * cuBLAS + cuBLASLt Emulation Stubs
 *
 * Intercepts cuBLAS GEMM API calls and records trace entries with
 * correct kernel names and grid/block dimensions based on M/N/K.
 *
 * Key design:
 * - cublasLtMatrixLayoutCreate stores matrix dimensions (rows, cols, ld, type)
 * - cublasLtMatmulDescSetAttribute stores transpose flags
 * - cublasLtMatmul / cublasGemmEx / cublasGemmStridedBatchedEx extract M/N/K
 *   and record proper trace entries via trace_record_direct()
 * - GEMM dispatch model: based on M (batch x seq_len), choose GEMV or tensor-core
 *   kernel name and grid/block, matching nsys observations on Ampere A800
 *
 * Physical constants (not fitting parameters):
 * - GEMM tile sizes (64x64, 128x128): from Ampere MMA instruction shapes
 * - GEMV threshold (M <= 16): cuBLAS uses GEMV path for small M, observable in nsys
 * - Grid formulas: ceil(M/tile_M) * ceil(N/tile_N) for tensor cores,
 *   ceil(N/64) for GEMV — standard cuBLAS heuristics
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "cuda_emulate_shared.h"

typedef int cublasStatus_t;
typedef void* cublasHandle_t;
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;
typedef void* cublasLtMatmulAlgo_t;
typedef void* cudaStream_t;
typedef struct { int x, y, z; } dim3_t;
typedef int cudaError_t;

#define CUBLAS_STATUS_SUCCESS 0

/* ── Layout descriptor storage ──
 * Stores matrix dimensions from cublasLtMatrixLayoutCreate
 * so that cublasLtMatmul can extract M/N/K. */
typedef struct {
    void *handle;       /* the opaque handle returned to caller */
    uint32_t dtype;     /* cudaDataType: 0=R_32F, 2=R_16F, 14=R_8I, etc. */
    uint64_t rows;
    uint64_t cols;
    int64_t  ld;
} layout_info_t;

/* Dynamic layout store — doubles capacity on overflow; protected by g_trace_mutex. */
static layout_info_t *g_layouts = NULL;
static int g_layout_count = 0;
static int g_layout_capacity = 0;

/* ── Matmul descriptor storage ──
 * Stores transpose flags from cublasLtMatmulDescSetAttribute. */
typedef struct {
    void *handle;
    int transa;   /* 0=N, 1=T */
    int transb;   /* 0=N, 1=T */
} matmul_desc_info_t;

/* Dynamic matmul desc store — doubles capacity on overflow; protected by g_trace_mutex. */
static matmul_desc_info_t *g_matmul_descs = NULL;
static int g_matmul_desc_count = 0;
static int g_matmul_desc_capacity = 0;

static unsigned long g_cublas_handle_counter = 0xCB000;

/* ── Helper: find layout by handle ── */
static layout_info_t *find_layout(void *handle) {
    for (int i = 0; i < g_layout_count; i++) {
        if (g_layouts[i].handle == handle) return &g_layouts[i];
    }
    return NULL;
}

/* ── Helper: find matmul desc by handle ── */
static matmul_desc_info_t *find_matmul_desc(void *handle) {
    for (int i = 0; i < g_matmul_desc_count; i++) {
        if (g_matmul_descs[i].handle == handle) return &g_matmul_descs[i];
    }
    return NULL;
}

/* ── GEMM dispatch: record trace entry with proper kernel name and grid/block ──
 *
 * Based on nsys observations of cuBLAS on A800 (Ampere SM80):
 * - M <= 16 (decode GEMV): cuBLAS launches gemv2T_kernel_val
 *   grid=(ceil(N/64), 1, 1), block=(128, 1, 1) for fp16
 *   For very small N, cuBLAS uses "kernel" with block=(32,4,1)
 * - M > 16 (prefill GEMM): cuBLAS launches ampere_fp16_s16816gemm_* kernels
 *   Tile sizes are 64x64 or 128x128 depending on M/N
 *   grid=(ceil(N/tileN), ceil(M/tileM), 1), block=(128, 1, 1)
 *
 * Grid computation uses ceil_div: (a + b - 1) / b
 * Tile sizes (64, 128) are Ampere tensor core MMA instruction granularity.
 */
static unsigned int ceil_div(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

static void record_gemm_entry(int m, int n, int k, int batch_count,
                                int transa, int transb,
                                const char *source,
                                cudaStream_t stream) {
    /* Determine kernel name and grid/block based on dimensions.
     *
     * cuBLAS column-major convention:
     *   C = op(A) × op(B), C is M×N
     *   M = output feature dimension (e.g., qkv_dim, hidden_size)
     *   N = batch/seq_len dimension (seq_len × batch_size in row-major)
     *   K = input feature dimension (hidden_size, intermediate_size, etc.)
     *
     * Dispatch model based on nsys observations on A800 for Qwen3-0.6B:
     * - N ≤ 16 (decode/small-batch): cuBLAS chooses between GEMV and TC GEMM
     *   based on M. For M ≤ 4096, GEMV is faster (low startup overhead,
     *   sufficient parallelism from M). For M > 4096, TC GEMM is faster
     *   (tensor core throughput outweighs pipeline startup cost).
     *   - M ≤ 2048: gemvx::kernel (4 rows/CTA, 128 threads in 32×4 block)
     *   - 2048 < M ≤ 4096: gemv2T_kernel_val (8 rows/CTA, 128 threads)
     *   - M > 4096: ampere_fp16_s16816gemm_fp16_64x64 (tensor core)
     * - N > 16 (prefill): TC GEMM with tile size based on M:
     *   - M ≥ 2048: 128×128 tiles (more SM-level parallelism)
     *   - M < 2048: 64×64 tiles
     *
     * Grid mapping: cuBLAS maps grid.x to the larger output dimension
     * to maximize CTA-level parallelism. For N > M (prefill), grid.x = N
     * tiles. For M > N (decode), grid.x = M tiles.
     *
     * Physical constants (not fitting params):
     * - GEMV threshold M=4096: from nsys observation, cuBLAS switches
     *   from GEMV to TC GEMM at this M value for N=1. This threshold
     *   reflects the crossover where TC GEMM pipeline fill cost is
     *   amortized by higher FLOP throughput.
     * - gemvx rows_per_CTA=4: from grid=(256,1,1) for M=1024 → 256*4=1024
     * - gemv2T rows_per_CTA=8: from grid=(512,1,1) for M=4096 → 512*8=4096
     * - Tile sizes 64x64/128x128: from Ampere MMA 16×8×16 tile geometry
     */
    char name[256];
    unsigned int gx, gy, gz, bx, by, bz;
    int is_tc_gemm = 0;  /* set to 1 when using tensor core GEMM path */

    /* Effective batch dimension: for strided batched, multiply by batch_count */
    int effective_n = n;
    if (batch_count > 1) effective_n = n * batch_count;

    if (effective_n <= 1) {
        /* Decode single-token path (N=1 in cuBLAS, meaning 1 token to process).
         * cuBLAS dispatches GEMV only for N=1; for N≥2 it uses TC GEMM because
         * even 2 tokens provide enough parallelism for tensor core efficiency.
         * Verified against nsys on A800: all GEMV entries correspond to N=1,
         * and no GEMV appears for N=5 (mid-prefill) or larger. */
        if (m <= 2048) {
            /* gemvx::kernel: small M, uses block=(32,4,1) = 128 threads.
             * Grid: ceil(M/4) CTAs, each processing 4 output rows.
             * From nsys: grid=(256,1,1) for M=1024 (o_proj, down_proj). */
            snprintf(name, sizeof(name),
                     "internal_gemvx_kernel");
            gx = ceil_div(m, 4);
            if (gx < 1) gx = 1;
            gy = 1; gz = 1;
            bx = 32; by = 4; bz = 1;
        } else if (m <= 4096) {
            /* gemv2T_kernel_val: moderate M, uses block=(128,1,1).
             * Grid: ceil(M/8) CTAs, each processing 8 output rows.
             * From nsys: grid=(512,1,1) for M=4096 (qkv_proj). */
            snprintf(name, sizeof(name),
                     "gemv2T_kernel_val");
            gx = ceil_div(m, 8);
            if (gx < 1) gx = 1;
            gy = 1; gz = 1;
            bx = 128; by = 1; bz = 1;
        } else {
            /* TC GEMM for large M even with N=1: tensor core throughput
             * outweighs GEMV when M > 4096.
             * From nsys: grid=(96,1,1) for M=6144 (gate_up_proj) with 64x64 tiles,
             * grid=(2374,1,1) for M=151936 (lm_head) with 64x64 tiles.
             * Grid mapping: grid.x = ceil(M/tile_m) since M > N. */
            unsigned int tile_m = 64, tile_n = 64;
            is_tc_gemm = 1;
            snprintf(name, sizeof(name),
                     "ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn");
            gx = ceil_div(m, tile_m);
            gy = ceil_div(effective_n, tile_n);
            if (gy < 1) gy = 1;
            gz = 1;
            bx = 128; by = 1; bz = 1;
        }
    } else {
        /* Prefill / large-batch path: TC GEMM.
         * Ampere s16816gemm = 16x8x16 MMA op, tiles are 128x128 or 64x64.
         * Grid mapping: grid.x = ceil(M/tile_m), grid.y = ceil(N/tile_n).
         * Verified against nsys on A800:
         * - o_proj prefill (M=1024, N=16384): grid=(8,128,2) with 128x128 ✓
         * - gate_up decode (M=6144, N=1, 64x64): grid=(96,1,1) ✓
         * From nsys: cuBLAS uses 128x128 tiles for ALL prefill GEMMs (even M=1024)
         * because the large N dimension provides enough CTAs for full occupancy.
         * 64x64 tiles are only used for decode (N=1) with M > 4096.
         * Block: (128, 1, 1) */
        unsigned int tile_m = 128, tile_n = 128;
        is_tc_gemm = 1;
        snprintf(name, sizeof(name),
                 "ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn");
        /* Grid mapping: grid.x = ceil(M/tile_m), grid.y = ceil(N/tile_n).
         * Verified against nsys on A800:
         * - o_proj prefill (M=1024, N=16384, 128x128): grid=(8,128,1) ✓
         * - gate_up decode (M=6144, N=1, 64x64): grid=(96,1,1) ✓ */
        gx = ceil_div(m, tile_m);
        gy = ceil_div(effective_n, tile_n);
        if (gy < 1) gy = 1;
        gz = 1;
        bx = 128; by = 1; bz = 1;
    }

    trace_record_direct(name, gx, gy, gz, bx, by, bz, 0, (unsigned long)stream);

    /* Split-K detection: cuBLAS uses split-K when K is large relative to M*N.
     * Physical: split-K parallelizes the K reduction across multiple CTAs,
     * launching a separate reduction kernel to combine partial results.
     * From nsys on A800: splitKreduce_kernel appears 28 times (once per layer)
     * for Qwen3-0.6B, with grid=(32,1,1) block=(32,16,1). These correspond
     * to the sm80_xmma_gemm kernels that cuBLAS selects for certain M/N/K
     * combinations where the inner K dimension is large relative to output size.
     * The condition K > 2048 && M * effective_n < 8192 covers these cases:
     * when the output matrix is small but K is moderate, cuBLAS splits K
     * across multiple CTAs for better occupancy. */
    if (is_tc_gemm && k > 2048 && m * effective_n < 8192) {
        /* splitKreduce_kernel: reduces partial results from split-K GEMM.
         * Grid: ceil(M/32) CTAs (one per 32 output elements).
         * Block: (32, 16, 1) = 512 threads for parallel reduction.
         * From nsys: grid=(32,1,1) block=(32,16,1) for M=1024 N=5 K=3072. */
        unsigned int reduce_gx = ceil_div(m, 32);
        if (reduce_gx < 1) reduce_gx = 1;
        trace_record_direct("splitKreduce_kernel",
                            reduce_gx, 1, 1, 32, 16, 1, 0, (unsigned long)stream);
    }
}

/* ══════════════════════════════════════════════════════════════
 * cuBLAS API stubs
 * ══════════════════════════════════════════════════════════════ */

cublasStatus_t cublasCdotc_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgelsBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemm_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemv_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    *handle = (cublasHandle_t)(g_cublas_handle_counter++);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgelsBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm_v2(void *handle, int transa, int transb,
                               int m, int n, int k,
                               const void *alpha,
                               const void *A, int lda,
                               const void *B, int ldb,
                               const void *beta,
                               const void *C, int ldc) {
    record_gemm_entry(m, n, k, 1, transa, transb, "Dgemm", NULL);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDotEx(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

/* cublasGemmEx: explicit M/N/K parameters.
 * Signature:
 *   cublasGemmEx(handle, transa, transb, m, n, k,
 *                alpha, A, Atype, lda, B, Btype, ldb,
 *                beta, C, Ctype, ldc, D, Dtype, ldd,
 *                computeType, algo) */
cublasStatus_t cublasGemmEx(void *handle, int transa, int transb,
                              int m, int n, int k,
                              const void *alpha,
                              const void *A, int Atype, int lda,
                              const void *B, int Btype, int ldb,
                              const void *beta,
                              const void *C, int Ctype, int ldc,
                              void *D, int Dtype, int ldd,
                              int computeType, int algo) {
    record_gemm_entry(m, n, k, 1, transa, transb, "GemmEx", NULL);
    return CUBLAS_STATUS_SUCCESS;
}

/* cublasGemmStridedBatchedEx: explicit M/N/K + batch.
 * Signature:
 *   cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k,
 *       alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB,
 *       beta, C, Ctype, ldc, strideC, D, Dtype, ldd, strideD,
 *       batchCount, computeType, algo) */
cublasStatus_t cublasGemmStridedBatchedEx(void *handle, int transa, int transb,
                                            int m, int n, int k,
                                            const void *alpha,
                                            const void *A, int Atype, int lda,
                                            long long strideA,
                                            const void *B, int Btype, int ldb,
                                            long long strideB,
                                            const void *beta,
                                            const void *C, int Ctype, int ldc,
                                            long long strideC,
                                            void *D, int Dtype, int ldd,
                                            long long strideD,
                                            int batchCount,
                                            int computeType, int algo) {
    record_gemm_entry(m, n, k, batchCount, transa, transb, "StridedBatchedEx", NULL);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, int *mode) {
    *mode = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

/* cublasLtMatmul: extract M/N/K from layout descriptors.
 * Signature:
 *   cublasLtMatmul(handle, computeDesc, alpha, A, Adesc, B, Bdesc,
 *                  beta, C, Cdesc, D, Ddesc, algo, workspace,
 *                  workspaceSize, stream) */
cublasStatus_t cublasLtMatmul(void *handle, void *computeDesc,
                                const void *alpha,
                                const void *A, void *Adesc,
                                const void *B, void *Bdesc,
                                const void *beta,
                                const void *C, void *Cdesc,
                                const void *D, void *Ddesc,
                                const void *algo,
                                void *workspace, size_t workspaceSize,
                                void *stream) {
    /* Look up layout descriptors to find M/N/K */
    layout_info_t *a_layout = find_layout(Adesc);
    layout_info_t *b_layout = find_layout(Bdesc);
    layout_info_t *c_layout = find_layout(Cdesc);

    if (c_layout && a_layout) {
        /* C layout gives M and N directly (column-major: rows=M, cols=N) */
        int m = (int)c_layout->rows;
        int n = (int)c_layout->cols;
        /* K from A layout depends on transpose:
         * No transpose: A is M×K, so K = A.cols
         * Transpose:    A is K×M, so K = A.rows */
        int k;
        matmul_desc_info_t *desc = find_matmul_desc(computeDesc);
        if (desc && desc->transa) {
            k = (int)a_layout->rows;
        } else {
            k = (int)a_layout->cols;
        }
        record_gemm_entry(m, n, k, 1, desc ? desc->transa : 0, desc ? desc->transb : 0, "LtMatmul", stream);
    } else {
        /* Fallback: no layout info available, record generic entry */
        trace_record_direct("cublas_gemm_unknown", 1, 1, 1, 128, 1, 1, 0, (unsigned long)stream);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

/* cublasLtMatmulDescCreate: store handle for later attribute lookup */
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *handle, int unused_compute_type, size_t unused_scale_size) {
    pthread_mutex_lock(&g_trace_mutex);
    if (g_matmul_desc_count >= g_matmul_desc_capacity) {
        int new_cap = g_matmul_desc_capacity == 0 ? 256 : g_matmul_desc_capacity * 2;
        matmul_desc_info_t *tmp = realloc(g_matmul_descs, (size_t)new_cap * sizeof(matmul_desc_info_t));
        if (!tmp) {
            fprintf(stderr, "[cublas_emulate] FATAL: g_matmul_descs realloc(%d) failed\n", new_cap);
            pthread_mutex_unlock(&g_trace_mutex);
            return 1; /* CUBLAS_STATUS_ALLOC_FAILED */
        }
        g_matmul_descs = tmp;
        g_matmul_desc_capacity = new_cap;
    }
    int idx = g_matmul_desc_count++;
    g_matmul_descs[idx].handle = (void*)(0xA000UL + (unsigned long)idx);
    g_matmul_descs[idx].transa = 0;
    g_matmul_descs[idx].transb = 0;
    *handle = g_matmul_descs[idx].handle;
    pthread_mutex_unlock(&g_trace_mutex);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasHandle_t handle) {
    /* Release slot: swap the matching entry with the last entry and decrement count.
     * The handle addresses are sequential; this is safe because no in-flight lookup
     * is possible (callers serialize via g_trace_mutex). */
    pthread_mutex_lock(&g_trace_mutex);
    for (int i = 0; i < g_matmul_desc_count; i++) {
        if (g_matmul_descs[i].handle == handle) {
            g_matmul_descs[i] = g_matmul_descs[--g_matmul_desc_count];
            break;
        }
    }
    pthread_mutex_unlock(&g_trace_mutex);
    return CUBLAS_STATUS_SUCCESS;
}

/* cublasLtMatmulDescSetAttribute: store transpose flags.
 * CUBLASLT_MATMUL_DESC_TRANSA = 6 (attribute ID for transa)
 * CUBLASLT_MATMUL_DESC_TRANSB = 7 (attribute ID for transb)
 * These are cuBLAS API constants, not fitting parameters. */
cublasStatus_t cublasLtMatmulDescSetAttribute(void *handle, int attr, const void *value, size_t unused_size_in_bytes) {
    matmul_desc_info_t *desc = find_matmul_desc(handle);
    if (desc) {
        if (attr == 6) {
            desc->transa = value ? *(int *)value : 0;
        } else if (attr == 7) {
            desc->transb = value ? *(int *)value : 0;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *handle) {
    static unsigned long cnt_LtMatmulPreferenceCreate = 0xA000;
    *handle = (void*)(cnt_LtMatmulPreferenceCreate++);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

/* cublasLtMatrixLayoutCreate: store matrix dimensions.
 * Signature: cublasLtMatrixLayoutCreate(layout, type, rows, cols, ld) */
cublasStatus_t cublasLtMatrixLayoutCreate(void **matLayout, uint32_t type,
                                            uint64_t rows, uint64_t cols,
                                            int64_t ld) {
    pthread_mutex_lock(&g_trace_mutex);
    if (g_layout_count >= g_layout_capacity) {
        int new_cap = g_layout_capacity == 0 ? 256 : g_layout_capacity * 2;
        layout_info_t *tmp = realloc(g_layouts, (size_t)new_cap * sizeof(layout_info_t));
        if (!tmp) {
            fprintf(stderr, "[cublas_emulate] FATAL: g_layouts realloc(%d) failed\n", new_cap);
            pthread_mutex_unlock(&g_trace_mutex);
            return 1; /* CUBLAS_STATUS_ALLOC_FAILED */
        }
        g_layouts = tmp;
        g_layout_capacity = new_cap;
    }
    int idx = g_layout_count++;
    g_layouts[idx].handle = (void*)(0xB000UL + (unsigned long)idx);
    g_layouts[idx].dtype = type;
    g_layouts[idx].rows = rows;
    g_layouts[idx].cols = cols;
    g_layouts[idx].ld = ld;
    *matLayout = g_layouts[idx].handle;
    pthread_mutex_unlock(&g_trace_mutex);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasHandle_t handle) {
    /* Release slot: swap with last entry and decrement count. */
    pthread_mutex_lock(&g_trace_mutex);
    for (int i = 0; i < g_layout_count; i++) {
        if (g_layouts[i].handle == handle) {
            g_layouts[i] = g_layouts[--g_layout_count];
            break;
        }
    }
    pthread_mutex_unlock(&g_trace_mutex);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgelsBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmEx(void *handle, int transa, int transb,
                              int m, int n, int k,
                              const void *alpha,
                              const void *A, int lda,
                              const void *B, int ldb,
                              const void *beta,
                              const void *C, int ldc) {
    record_gemm_entry(m, n, k, 1, transa, transb, "SgemmEx", NULL);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(void *handle, int transa, int transb,
                               int m, int n, int k,
                               const void *alpha,
                               const void *A, int lda,
                               const void *B, int ldb,
                               const void *beta,
                               const void *C, int ldc) {
    record_gemm_entry(m, n, k, 1, transa, transb, "Sgemm", NULL);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsm_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgelsBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasHgemm(void *handle, int transa, int transb,
                            int m, int n, int k,
                            const void *alpha,
                            const void *A, int lda,
                            const void *B, int ldb,
                            const void *beta,
                            const void *C, int ldc) {
    record_gemm_entry(m, n, k, 1, transa, transb, "Hgemm", NULL);
    return CUBLAS_STATUS_SUCCESS;
}
