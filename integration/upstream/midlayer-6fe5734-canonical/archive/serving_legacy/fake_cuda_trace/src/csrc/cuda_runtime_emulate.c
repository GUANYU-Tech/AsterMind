/*
 * CUDA Runtime API Emulation Layer
 *
 * Intercepts CUDA runtime API calls (libcudart.so) via LD_PRELOAD.
 * This is the primary API used by PyTorch/vLLM for kernel launches.
 *
 * Build and use together with cuda_emulate.c (Driver API):
 *   gcc -O2 -shared -fPIC -o build/libcuda_emulate.so \
 *       src/emulation/cuda_emulate.c src/emulation/cuda_runtime_emulate.c \
 *       -lpthread
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <dlfcn.h>

/* ── CUDA runtime type definitions ── */
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef struct { int x, y, z; } dim3;
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void *userData);

#define cudaSuccess 0

#include "cuda_emulate_shared.h"

/* Per-thread current device tracking (CUDA semantics: per-thread) */
static __thread int g_current_device = 0;

/* Device properties structure (matches CUDA runtime) */
typedef struct {
    char name[256];
    int uuid[16];
    char luid[8];
    unsigned int luidDeviceNodeMask;
    int totalGlobalMem_hi, totalGlobalMem_lo;  /* Use size_t in real code */
    size_t totalGlobalMem;
    int sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int totalConstMem;
    int major;
    int minor;
    int textureAlignment;
    int texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    int surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    int sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    int sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int readMemoryPoolSupported;
    int memoryPoolSupported;
    int memoryPoolExportHandle;
    int gpuDirectRDMASupported;
    int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    int memoryPoolSupportedHandleTypes;
    int deferredMappingCudaArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
} cudaDeviceProp;

/* ── Helper ── */
static double get_timestamp_us_runtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

/* ── Trace gating state (mirror from shared header) ──
 * These are defined in cuda_emulate.c; declare as extern here. */
extern int g_trace_enabled;
extern int g_trace_gating_required;

/* ── CUDA Runtime API implementations
 * ══════════════════════════════════════════════════════════════ */

cudaError_t cudaGetDeviceCount(int *count) {
    /* Derive device count from CUDA_VISIBLE_DEVICES.
     * If set, count comma-separated IDs; otherwise default to 1. */
    const char *vis = getenv("CUDA_VISIBLE_DEVICES");
    if (vis && vis[0] != '\0') {
        int n = 1;
        for (const char *p = vis; *p; p++) {
            if (*p == ',') n++;
        }
        *count = n;
    } else {
        *count = 1;
    }
    return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    *driverVersion = 12080;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    *runtimeVersion = 12080;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    *device = g_current_device;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    g_current_device = device;
    return cudaSuccess;
}

/* Device properties: write directly at correct byte offsets matching CUDA 12.8 struct layout.
 * This avoids maintaining a mirror struct definition that can drift out of sync.
 * Struct size on x86_64 is approximately 984 bytes (from header analysis). */
cudaError_t cudaGetDeviceProperties(void *prop_raw, int device) {
    char *p = (char *)prop_raw;
    /* Zero the full struct (CUDA 12.8: sizeof(cudaDeviceProp) = 1032) */
    memset(p, 0, 1032);
    
    /* Offsets from CUDA 12.8 driver_types.h (x86_64, verified with offsetof) */
    strncpy(p + 0, g_gpu.gpu_name, 255);                          /* name[256] at 0 */
    *(size_t*)(p + 288) = (size_t)g_gpu.total_memory;             /* totalGlobalMem */
    *(size_t*)(p + 296) = (size_t)g_gpu.max_shared_memory_per_block; /* sharedMemPerBlock */
    *(int*)(p + 304) = 65536;                                      /* regsPerBlock */
    *(int*)(p + 308) = g_gpu.warp_size;                           /* warpSize */
    *(size_t*)(p + 312) = 2147483647;                             /* memPitch */
    *(int*)(p + 320) = g_gpu.max_threads_per_block;               /* maxThreadsPerBlock */
    *(int*)(p + 324) = g_gpu.max_block_dim_x;                    /* maxThreadsDim[0] */
    *(int*)(p + 328) = g_gpu.max_block_dim_y;                    /* maxThreadsDim[1] */
    *(int*)(p + 332) = g_gpu.max_block_dim_z;                    /* maxThreadsDim[2] */
    *(int*)(p + 336) = g_gpu.max_grid_dim_x;                     /* maxGridSize[0] */
    *(int*)(p + 340) = g_gpu.max_grid_dim_y;                     /* maxGridSize[1] */
    *(int*)(p + 344) = g_gpu.max_grid_dim_z;                     /* maxGridSize[2] */
    *(int*)(p + 348) = g_gpu.clock_rate;                          /* clockRate */
    *(size_t*)(p + 352) = 65536;                                  /* totalConstMem */
    *(int*)(p + 360) = g_gpu.major;                               /* major */
    *(int*)(p + 364) = g_gpu.minor;                               /* minor */
    *(size_t*)(p + 368) = 512;                                    /* textureAlignment */
    *(size_t*)(p + 376) = 32;                                     /* texturePitchAlignment */
    *(int*)(p + 384) = 1;                                         /* deviceOverlap */
    *(int*)(p + 388) = g_gpu.multi_processor_count;               /* multiProcessorCount */
    *(int*)(p + 400) = 1;                                         /* canMapHostMemory */
    *(int*)(p + 576) = 1;                                         /* concurrentKernels */
    *(int*)(p + 624) = g_gpu.max_threads_per_multiprocessor;      /* maxThreadsPerMultiProcessor */
    *(size_t*)(p + 640) = (size_t)g_gpu.max_shared_memory_per_multiprocessor; /* sharedMemPerMultiprocessor */
    *(int*)(p + 648) = 65536;                                     /* regsPerMultiprocessor */
    
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSetCacheConfig(int config) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    /* CUDA Runtime attribute enum values from driver_types.h (CUDA 12.x).
     * Only the subset actually queried by PyTorch/vLLM/flash_attn is needed. */
    switch (attr) {
        case 1:  *value = g_gpu.max_threads_per_block; break; /* cudaDevAttrMaxThreadsPerBlock */
        case 2:  *value = g_gpu.max_block_dim_x; break;      /* cudaDevAttrMaxBlockDimX */
        case 3:  *value = g_gpu.max_block_dim_y; break;      /* cudaDevAttrMaxBlockDimY */
        case 4:  *value = g_gpu.max_block_dim_z; break;      /* cudaDevAttrMaxBlockDimZ */
        case 5:  *value = g_gpu.max_grid_dim_x; break;       /* cudaDevAttrMaxGridDimX */
        case 6:  *value = g_gpu.max_grid_dim_y; break;       /* cudaDevAttrMaxGridDimY */
        case 7:  *value = g_gpu.max_grid_dim_z; break;       /* cudaDevAttrMaxGridDimZ */
        case 8:  *value = g_gpu.max_shared_memory_per_block; break; /* cudaDevAttrMaxSharedMemoryPerBlock */
        case 9:  *value = 65536; break;                       /* cudaDevAttrTotalConstantMemory */
        case 10: *value = g_gpu.warp_size; break;             /* cudaDevAttrWarpSize */
        case 12: *value = 65536; break;                       /* cudaDevAttrMaxRegistersPerBlock */
        case 13: *value = g_gpu.clock_rate; break;            /* cudaDevAttrClockRate (kHz) */
        case 14: *value = 512; break;                         /* cudaDevAttrTextureAlignment */
        case 16: *value = g_gpu.multi_processor_count; break; /* cudaDevAttrMultiProcessorCount */
        case 17: *value = 0; break;                           /* cudaDevAttrKernelExecTimeout */
        case 18: *value = 0; break;                           /* cudaDevAttrIntegrated */
        case 19: *value = 1; break;                           /* cudaDevAttrCanMapHostMemory */
        case 20: *value = 0; break;                           /* cudaDevAttrComputeMode */
        case 32: *value = 8; break;                           /* cudaDevAttrAsyncEngineCount */
        case 33: *value = 0; break;                           /* cudaDevAttrGpuOverlap (deprecated) */
        case 36: *value = g_gpu.memory_clock_rate; break;     /* cudaDevAttrMemoryClockRate (kHz) */
        case 37: *value = g_gpu.memory_bus_width; break;      /* cudaDevAttrGlobalMemoryBusWidth (bits) */
        case 38: *value = g_gpu.l2_cache_size; break;         /* cudaDevAttrL2CacheSize */
        case 39: *value = g_gpu.max_threads_per_multiprocessor; break; /* cudaDevAttrMaxThreadsPerMultiProcessor */
        case 40: *value = 8; break;                           /* cudaDevAttrAsyncEngineCount */
        case 41: *value = 1; break;                           /* cudaDevAttrUnifiedAddressing */
        case 75: *value = g_gpu.major; break;                 /* cudaDevAttrComputeCapabilityMajor */
        case 76: *value = g_gpu.minor; break;                 /* cudaDevAttrComputeCapabilityMinor */
        case 79: *value = 1; break;                           /* cudaDevAttrGlobalL1CacheSupported */
        case 81: *value = g_gpu.max_shared_memory_per_multiprocessor; break; /* cudaDevAttrMaxSharedMemoryPerMultiprocessor */
        case 82: *value = 65536; break;                       /* cudaDevAttrMaxRegistersPerMultiprocessor */
        case 83: *value = 1; break;                           /* cudaDevAttrManagedMemory */
        case 84: *value = 0; break;                           /* cudaDevAttrIsMultiGpuBoard */
        case 86: *value = 1; break;                           /* cudaDevAttrHostNativeAtomicSupported */
        case 87: *value = 2; break;                           /* cudaDevAttrSingleToDoublePrecisionPerfRatio */
        case 89: *value = 1; break;                           /* cudaDevAttrConcurrentManagedAccess */
        case 90: *value = 1; break;                           /* cudaDevAttrComputePreemptionSupported */
        case 91: *value = 1; break;                           /* cudaDevAttrCanUseHostPointerForRegisteredMem */
        case 95: *value = 1; break;                           /* cudaDevAttrCooperativeLaunch */
        case 97: *value = g_gpu.max_shared_memory_per_block; break; /* cudaDevAttrMaxSharedMemoryPerBlockOptin */
        case 99: *value = 1; break;                           /* cudaDevAttrPageableMemoryAccess */
        case 100: *value = 1; break;                          /* cudaDevAttrPageableMemoryAccessUsesHostPageTables */
        case 101: *value = 1; break;                          /* cudaDevAttrDirectManagedMemAccessFromHost */
        case 106: *value = 1; break;                          /* cudaDevAttrMemoryPoolsSupported */
        default: *value = 0; break;
    }
    return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(unsigned long long *pValue, int limit) {
    *pValue = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceSetLimit(int limit, unsigned long long value) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetSharedMemConfig(int *pConfig) {
    *pConfig = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceSetSharedMemConfig(int config) {
    return cudaSuccess;
}

/* ── Memory management ── */
static unsigned long long g_runtime_next_ptr = 0x200000000ULL;
static size_t g_allocated_bytes = 0;  /* atomic: updated from multiple threads */

/* P37: device allocation accounting for OOM behavior + cudaFree decrement.
 * Real GPU cudaMalloc returns cudaErrorMemoryAllocation (=2) once the
 * configured device memory is exceeded. We track allocations in a small
 * fixed table so cudaFree can reclaim slots; the table is sized for the
 * worst-case vLLM workload (KV cache + activation buffers + workspace)
 * which is well under 16K live allocations on observed traces. */
#define cudaErrorMemoryAllocation 2
#define MAX_DEVICE_ALLOCS 16384
typedef struct {
    void *ptr;     /* NULL means slot is free */
    size_t size;
} dev_alloc_record_t;
static dev_alloc_record_t g_dev_allocs[MAX_DEVICE_ALLOCS];
static pthread_mutex_t g_dev_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

static int _dev_alloc_record_insert(void *ptr, size_t size) {
    pthread_mutex_lock(&g_dev_alloc_mutex);
    int slot = -1;
    for (int i = 0; i < MAX_DEVICE_ALLOCS; i++) {
        if (g_dev_allocs[i].ptr == NULL) {
            g_dev_allocs[i].ptr = ptr;
            g_dev_allocs[i].size = size;
            slot = i;
            break;
        }
    }
    pthread_mutex_unlock(&g_dev_alloc_mutex);
    if (slot < 0) {
        /* Fail loud rather than silently desync: if we can't track this
         * allocation, cudaFree won't decrement g_allocated_bytes and
         * cudaMemGetInfo will report decreasing free memory until the
         * process appears OOM. Same philosophy as g_fat_handles overflow
         * (P53.c). Increase MAX_DEVICE_ALLOCS if this triggers. */
        fprintf(stderr, "[cuda_emulate] FATAL: g_dev_allocs[%d] table full. "
                "cudaFree accounting would desync. Increase MAX_DEVICE_ALLOCS.\n",
                MAX_DEVICE_ALLOCS);
        abort();
    }
    return slot;
}

static size_t _dev_alloc_record_remove(void *ptr) {
    pthread_mutex_lock(&g_dev_alloc_mutex);
    size_t size = 0;
    for (int i = 0; i < MAX_DEVICE_ALLOCS; i++) {
        if (g_dev_allocs[i].ptr == ptr) {
            size = g_dev_allocs[i].size;
            g_dev_allocs[i].ptr = NULL;
            g_dev_allocs[i].size = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_dev_alloc_mutex);
    return size;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    /* OOM check: reject if cumulative live allocations would exceed
     * configured device memory. Behavior matches a real GPU. */
    size_t live = __atomic_load_n(&g_allocated_bytes, __ATOMIC_RELAXED);
    if (g_gpu.total_memory > 0 && live + size > (size_t)g_gpu.total_memory) {
        *devPtr = NULL;
        return cudaErrorMemoryAllocation;
    }
    *devPtr = (void *)g_runtime_next_ptr;
    g_runtime_next_ptr += size;
    g_runtime_next_ptr = (g_runtime_next_ptr + 255) & ~255ULL;
    __atomic_add_fetch(&g_allocated_bytes, size, __ATOMIC_RELAXED);
    _dev_alloc_record_insert(*devPtr, size);
    return cudaSuccess;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    /* cudaMallocHost is pinned host memory on real GPU, separate from device
     * memory budget — keep using calloc and don't count toward g_allocated_bytes. */
    *ptr = calloc(1, size);
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    if (devPtr == NULL) return cudaSuccess;
    size_t freed = _dev_alloc_record_remove(devPtr);
    if (freed > 0) {
        __atomic_sub_fetch(&g_allocated_bytes, freed, __ATOMIC_RELAXED);
    }
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    *total = (size_t)g_gpu.total_memory;
    /* Report actual live allocations (now that cudaFree decrements correctly,
     * this is the real device memory budget remaining). */
    size_t live = __atomic_load_n(&g_allocated_bytes, __ATOMIC_RELAXED);
    if (live > *total) live = *total;  /* defensive */
    *free = *total - live;
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind) {
    /* kind: 0=H2H, 1=H2D, 2=D2H, 3=D2D */
    if (kind == 0) { /* Host-to-Host: real memcpy */
        memcpy(dst, src, count);
    } else if (kind == 2) { /* Device-to-Host: zero-fill host buffer */
        memset(dst, 0, count);
    }
    /* kind==1 (H2D) and kind==3 (D2D): dst is fake device ptr, no-op */
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind, cudaStream_t stream) {
    /* kind: 0=H2H, 1=H2D, 2=D2H, 3=D2D */
    if (kind == 0) {
        memcpy(dst, src, count);
    } else if (kind == 2) {
        memset(dst, 0, count);
    }
    /* kind==1 (H2D) and kind==3 (D2D): dst is fake device ptr, no-op */
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaMemAdvise(const void *devPtr, size_t count, int advice, int device) {
    return cudaSuccess;
}

/* ── Stream management ── */
static unsigned long g_stream_counter = 0x3000;

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    *pStream = (cudaStream_t)(g_stream_counter++);
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
    return cudaStreamCreate(pStream);
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) {
    return cudaStreamCreate(pStream);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority) {
    *priority = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags) {
    *flags = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags) {
    /* Execute callback immediately (no GPU work) */
    callback(stream, cudaSuccess, userData);
    return cudaSuccess;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int mode) {
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, void **pGraph) {
    *pGraph = NULL;
    return cudaSuccess;
}

/* ── Event management ── */
static unsigned long g_event_counter = 0x4000;

cudaError_t cudaEventCreate(cudaEvent_t *pEvent) {
    *pEvent = (cudaEvent_t)(g_event_counter++);
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *pEvent, unsigned int flags) {
    return cudaEventCreate(pEvent);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    /* No-op in emulation. Timestamps are not needed since
     * cudaEventElapsedTime returns a fixed positive value. */
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    /* In emulation mode, kernels are no-ops so real GPU time is 0.
     * But torch._inductor.runtime.benchmarking divides by estimated_timing,
     * so returning 0 causes ZeroDivisionError. Return a small positive value
     * (10us) as synthetic kernel execution time. */
    *ms = 0.01f;
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    return cudaSuccess; /* ready */
}

/* ── Kernel launch (THE KEY INTERCEPTION) ── */
cudaError_t cudaLaunchKernel(const void *func,
                              dim3 gridDim,
                              dim3 blockDim,
                              void **args,
                              size_t sharedMem,
                              cudaStream_t stream) {
    if (!g_trace_enabled) return cudaSuccess;
    /* Record trace entry */
    pthread_mutex_lock(&g_trace_mutex);
    if (g_trace_count < MAX_TRACE_ENTRIES) {
        trace_entry_t *entry = &g_trace_entries[g_trace_count];
        entry->timestamp_us = get_timestamp_us_runtime();
        /* Resolve function pointer to symbol name via dladdr */
        Dl_info info;
        if (dladdr(func, &info) && info.dli_sname) {
            snprintf(entry->kernel_name, sizeof(entry->kernel_name), "%s", info.dli_sname);
        } else if (dladdr(func, &info) && info.dli_fname) {
            /* dladdr found the .so but not the symbol — likely RTLD_LOCAL.
             * Store basename+offset for offline resolution. */
            const char *slash = strrchr(info.dli_fname, '/');
            const char *basename = slash ? slash + 1 : info.dli_fname;
            unsigned long offset = (unsigned long)func - (unsigned long)info.dli_fbase;
            snprintf(entry->kernel_name, sizeof(entry->kernel_name),
                     "so:%s+0x%lx", basename, offset);
        } else {
            snprintf(entry->kernel_name, sizeof(entry->kernel_name), "%p", func);
        }
        entry->grid_x = gridDim.x; entry->grid_y = gridDim.y; entry->grid_z = gridDim.z;
        entry->block_x = blockDim.x; entry->block_y = blockDim.y; entry->block_z = blockDim.z;
        entry->shared_mem = (unsigned int)sharedMem;
        entry->stream = (unsigned long)stream;
        g_trace_count++;
    }
    pthread_mutex_unlock(&g_trace_mutex);
    
    return cudaSuccess;
}

/* ── Other runtime APIs ── */

/* Legacy kernel launch: cudaConfigureCall saves grid/block/sharedMem; cudaLaunch records
 * the trace entry. This path is used by some PyTorch-internal kernels (e.g. NORM, ROTARY,
 * ACTIVATION) that were missing from the emulation trace. */
static dim3 g_pending_grid = {1, 1, 1};
static dim3 g_pending_block = {1, 1, 1};
static size_t g_pending_shared = 0;

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    g_pending_grid = gridDim;
    g_pending_block = blockDim;
    g_pending_shared = sharedMem;
    return cudaSuccess;
}

cudaError_t cudaSetupArgument(void *arg, size_t size, size_t offset) {
    return cudaSuccess;
}

cudaError_t cudaLaunch(const void *func) {
    if (!g_trace_enabled) return cudaSuccess;
    /* Record via the same dladdr path as cudaLaunchKernel. */
    Dl_info info;
    char name[256];
    int ok = dladdr(func, &info);
    if (ok && info.dli_sname) {
        snprintf(name, sizeof(name), "%s", info.dli_sname);
    } else if (ok && info.dli_fname) {
        const char *slash = strrchr(info.dli_fname, '/');
        const char *bn = slash ? slash + 1 : info.dli_fname;
        unsigned long off = (unsigned long)func - (unsigned long)info.dli_fbase;
        snprintf(name, sizeof(name), "so:%s+0x%lx", bn, off);
    } else {
        snprintf(name, sizeof(name), "kernel_%p", func);
    }
    pthread_mutex_lock(&g_trace_mutex);
    if (g_trace_count < MAX_TRACE_ENTRIES) {
        trace_entry_t *entry = &g_trace_entries[g_trace_count];
        entry->timestamp_us = get_timestamp_us_runtime();
        snprintf(entry->kernel_name, sizeof(entry->kernel_name), "%s", name);
        entry->grid_x = g_pending_grid.x;
        entry->grid_y = g_pending_grid.y;
        entry->grid_z = g_pending_grid.z;
        entry->block_x = g_pending_block.x;
        entry->block_y = g_pending_block.y;
        entry->block_z = g_pending_block.z;
        entry->shared_mem = (int)g_pending_shared;
        g_trace_count++;
    }
    pthread_mutex_unlock(&g_trace_mutex);
    return cudaSuccess;
}

cudaError_t cudaBindTexture(size_t *offset, const void *texref, const void *devPtr, const void *desc, size_t size) {
    return cudaSuccess;
}

cudaError_t cudaUnbindTexture(const void *texref) {
    return cudaSuccess;
}

/* ── Direct trace recording (called from libcublas.so.12) ── */
void trace_record_direct(const char *kernel_name,
                           unsigned int gx, unsigned int gy, unsigned int gz,
                           unsigned int bx, unsigned int by, unsigned int bz,
                           unsigned int shared_mem, unsigned long stream) {
    if (!g_trace_enabled) return;
    pthread_mutex_lock(&g_trace_mutex);
    if (g_trace_count < MAX_TRACE_ENTRIES) {
        trace_entry_t *entry = &g_trace_entries[g_trace_count];
        entry->timestamp_us = get_timestamp_us_runtime();
        snprintf(entry->kernel_name, sizeof(entry->kernel_name), "%s", kernel_name);
        entry->grid_x = gx; entry->grid_y = gy; entry->grid_z = gz;
        entry->block_x = bx; entry->block_y = by; entry->block_z = bz;
        entry->shared_mem = shared_mem;
        entry->stream = stream;
        g_trace_count++;
    }
    pthread_mutex_unlock(&g_trace_mutex);
}

cudaError_t cudaGetLastError(void) {
    return cudaSuccess;
}

cudaError_t cudaPeekAtLastError(void) {
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) {
    return "no error";
}

const char* cudaGetErrorName(cudaError_t error) {
    return "cudaSuccess";
}

cudaError_t cudaProfilerStart(void) {
    if (g_trace_gating_required) {
        g_trace_enabled = 1;
        fprintf(stderr, "[cuda_emulate] Trace enabled (cudaProfilerStart)\n");
    }
    return cudaSuccess;
}

cudaError_t cudaProfilerStop(void) {
    if (g_trace_gating_required) {
        g_trace_enabled = 0;
        fprintf(stderr, "[cuda_emulate] Trace disabled (cudaProfilerStop)\n");
    }
    return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void *func, int attribute, int value) {
    return cudaSuccess;
}

cudaError_t cudaFuncSetCacheConfig(const void *func, int config) {
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
    *numBlocks = g_gpu.max_threads_per_multiprocessor / blockSize;
    if (*numBlocks > 32) *numBlocks = 32;
    if (*numBlocks < 1) *numBlocks = 1;
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

cudaError_t cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, const void *func, size_t dynamicSMemSize, int blockSizeLimit) {
    *blockSize = 256;
    *minGridSize = g_gpu.multi_processor_count * 4;
    return cudaSuccess;
}

/* ── IPC ── */
cudaError_t cudaIpcGetMemHandle(void *handle, void *devPtr) {
    memset(handle, 0, 64);
    return cudaSuccess;
}

cudaError_t cudaIpcOpenMemHandle(void **devPtr, const void *handle, unsigned int flags) {
    *devPtr = (void *)g_runtime_next_ptr;
    g_runtime_next_ptr += 4096;
    __atomic_add_fetch(&g_allocated_bytes, 4096, __ATOMIC_RELAXED);
    return cudaSuccess;
}

cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
    return cudaSuccess;
}

cudaError_t cudaIpcGetEventHandle(void *handle, cudaEvent_t event) {
    memset(handle, 0, 64);
    return cudaSuccess;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, const void *handle) {
    return cudaEventCreate(event);
}

/* ── Graph API ── */

/* Per-graph kernel-node list: records kernel nodes added during graph capture.
 * Layout: g_graph_nodes[N] holds a kernel-launch-params snapshot.
 * When cudaGraphLaunch is called, the stored nodes are replayed as trace entries
 * via trace_record_direct (same path as live cudaLaunchKernel).
 * Scope: kernel nodes only (P49). Memcpy / event / conditional nodes are stubs.
 */
#define MAX_GRAPH_NODES 512

typedef struct {
    char func_name[256];
    int  grid_x, grid_y, grid_z;
    int  block_x, block_y, block_z;
    size_t shared_mem;
} graph_node_t;

static graph_node_t g_graph_nodes[MAX_GRAPH_NODES];
static int g_graph_node_count = 0;

cudaError_t cudaGraphCreate(void **pGraph, unsigned int flags) {
    *pGraph = (void *)0x5000;
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(void *graph) {
    return cudaSuccess;
}

/* cudaGraphAddKernelNode: store kernel node for replay.
 * The nodeParams struct starts with func pointer and then {gridDim, blockDim, sharedMem, ...}.
 * Since this is opaque void*, we cast to a minimal params layout that matches
 * the CUDA 12.x cudaKernelNodeParams struct (https://docs.nvidia.com/cuda/cuda-runtime-api/).
 */
typedef struct {
    const void *func;             /* kernel function pointer */
    dim3        gridDim;
    dim3        blockDim;
    unsigned    sharedMemBytes;
    void      **kernelParams;
    void      **extra;
} _kernel_node_params_t;

cudaError_t cudaGraphAddKernelNode(void **pGraphNode, void *graph,
                                    const void *const *pDependencies, size_t numDependencies,
                                    const void *pNodeParams) {
    /* Always store graph node metadata regardless of trace gate state.
     * The trace gate only suppresses emission during cudaGraphLaunch,
     * not the bookkeeping needed for future replay. */
    pthread_mutex_lock(&g_trace_mutex);
    if (g_graph_node_count >= MAX_GRAPH_NODES) {
        fprintf(stderr, "[cuda_emulate] FATAL: g_graph_nodes[%d] overflow in cudaGraphAddKernelNode\n",
                MAX_GRAPH_NODES);
        abort();
    }
    *pGraphNode = (void *)(0x8000UL + (unsigned long)g_graph_node_count);
    if (pNodeParams) {
        const _kernel_node_params_t *p = (const _kernel_node_params_t *)pNodeParams;
        graph_node_t *n = &g_graph_nodes[g_graph_node_count++];
        Dl_info di;
        if (p->func && dladdr(p->func, &di) && di.dli_sname) {
            strncpy(n->func_name, di.dli_sname, sizeof(n->func_name) - 1);
            n->func_name[sizeof(n->func_name) - 1] = '\0';
        } else {
            snprintf(n->func_name, sizeof(n->func_name), "graph_kernel_0x%lx",
                     (unsigned long)(size_t)(const void *)p->func);
        }
        n->grid_x  = (int)p->gridDim.x;
        n->grid_y  = (int)p->gridDim.y;
        n->grid_z  = (int)p->gridDim.z;
        n->block_x = (int)p->blockDim.x;
        n->block_y = (int)p->blockDim.y;
        n->block_z = (int)p->blockDim.z;
        n->shared_mem = p->sharedMemBytes;
    }
    pthread_mutex_unlock(&g_trace_mutex);
    return cudaSuccess;
}

cudaError_t cudaGraphExecKernelNodeSetParams(void *hGraphExec, void *node, const void *pNodeParams) {
    return cudaSuccess;
}

cudaError_t cudaGraphLaunch(void *graphExec, cudaStream_t stream) {
    /* Replay all captured kernel nodes as trace entries. */
    if (!g_trace_enabled) return cudaSuccess;
    pthread_mutex_lock(&g_trace_mutex);
    for (int i = 0; i < g_graph_node_count && g_trace_count < MAX_TRACE_ENTRIES; i++) {
        const graph_node_t *n = &g_graph_nodes[i];
        trace_entry_t *entry = &g_trace_entries[g_trace_count];
        strncpy(entry->kernel_name, n->func_name, 255);
        entry->kernel_name[255] = '\0';
        entry->grid_x  = n->grid_x;
        entry->grid_y  = n->grid_y;
        entry->grid_z  = n->grid_z;
        entry->block_x = n->block_x;
        entry->block_y = n->block_y;
        entry->block_z = n->block_z;
        entry->shared_mem = (int)n->shared_mem;
        g_trace_count++;
    }
    pthread_mutex_unlock(&g_trace_mutex);
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(void *graphExec) {
    pthread_mutex_lock(&g_trace_mutex);
    g_graph_node_count = 0;
    pthread_mutex_unlock(&g_trace_mutex);
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiate(void **pGraphExec, void *graph, void *errNode, char *logBuffer, size_t bufferSize) {
    *pGraphExec = (void *)0x6000;
    return cudaSuccess;
}

/* ── External memory ── */
cudaError_t cudaImportExternalMemory(void **extMem, void *handle) {
    *extMem = (void *)0x7000;
    return cudaSuccess;
}

cudaError_t cudaDestroyExternalMemory(void *extMem) {
    return cudaSuccess;
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void **devPtr, void *extMem, unsigned long long offset, unsigned long long size) {
    *devPtr = (void *)g_runtime_next_ptr;
    g_runtime_next_ptr += size;
    __atomic_add_fetch(&g_allocated_bytes, (size_t)size, __ATOMIC_RELAXED);
    return cudaSuccess;
}

/* ── Additional runtime APIs needed when fully replacing libcudart.so.12 ── */

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
    *canAccessPeer = 1;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetDefaultMemPool(void **memPool, int device) {
    static char fake_pool[64];
    *memPool = fake_pool;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    *leastPriority = 0;
    *greatestPriority = -1;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    strncpy(pciBusId, "0000:00:00.0", len);
    return cudaSuccess;
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
    *ptr = calloc(1, size);
    return cudaSuccess;
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    return cudaSuccess;
}

cudaError_t cudaHostUnregister(void *ptr) {
    return cudaSuccess;
}

cudaError_t cudaMemPoolGetAttribute(void *memPool, int attr, void *value) {
    /* Most pool attributes are 8 bytes (uint64_t / size_t on 64-bit).
     * CUDA memory pool attributes: cudaMemPoolAttrReleaseThreshold (size_t=8),
     * cudaMemPoolAttrReservedMemCurrent/High (size_t=8), etc. */
    memset(value, 0, sizeof(size_t));
    return cudaSuccess;
}

cudaError_t cudaMemPoolSetAttribute(void *memPool, int attr, void *value) {
    return cudaSuccess;
}

cudaError_t cudaMemPoolSetAccess(void *memPool, void *descList, size_t count) {
    return cudaSuccess;
}

cudaError_t cudaMemPoolTrimTo(void *memPool, size_t minBytesToKeep) {
    return cudaSuccess;
}

cudaError_t cudaPointerGetAttributes(void *attributes, const void *ptr) {
    /* cudaPointerAttributes layout (CUDA 12.2, driver_types.h line 1581-1614):
     *   enum cudaMemoryType type  (4 bytes)
     *   int device                (4 bytes)
     *   void *devicePointer       (8 bytes)
     *   void *hostPointer         (8 bytes)
     * = 24 bytes on x86-64. Verified: sizeof(cudaPointerAttributes)=24 via
     * nvcc /tmp/size_test.cu on CUDA 12.2.140 / A800. */
    memset(attributes, 0, 24);
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, int *pCaptureStatus) {
    *pCaptureStatus = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, int *captureStatus_out,
                                         unsigned long long *id_out, void **graph_out,
                                         void **deps_out, size_t *numDeps_out) {
    *captureStatus_out = 0;
    if (id_out) *id_out = 0;
    if (graph_out) *graph_out = NULL;
    if (deps_out) *deps_out = NULL;
    if (numDeps_out) *numDeps_out = 0;
    return cudaSuccess;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode) {
    return cudaSuccess;
}

cudaError_t cudaLaunchKernelExC(void *config, const void *func, void **args) {
    /* cudaLaunchConfig_t structure in CUDA 12.x:
     *   unsigned int gridDimX, gridDimY, gridDimZ;   (offsets 0, 4, 8)
     *   unsigned int blockDimX, blockDimY, blockDimZ; (offsets 12, 16, 20)
     *   size_t dynamicSmemBytes;                      (offset 24)
     *   cudaStream_t stream;                          (offset 32)
     *   cudaLaunchAttribute *attrs;                   (offset 40)
     *   unsigned int numAttrs;                        (offset 48)
     */
    unsigned int *cfg = (unsigned int *)config;
    unsigned int gridX = cfg[0], gridY = cfg[1], gridZ = cfg[2];
    unsigned int blockX = cfg[3], blockY = cfg[4], blockZ = cfg[5];
    unsigned int smem = cfg[6];

    if (!g_trace_enabled) return cudaSuccess;
    pthread_mutex_lock(&g_trace_mutex);
    if (g_trace_count < MAX_TRACE_ENTRIES) {
        trace_entry_t *entry = &g_trace_entries[g_trace_count];
        entry->timestamp_us = get_timestamp_us_runtime();
        Dl_info info;
        if (dladdr(func, &info) && info.dli_sname) {
            snprintf(entry->kernel_name, sizeof(entry->kernel_name), "%s", info.dli_sname);
        } else if (dladdr(func, &info) && info.dli_fname) {
            const char *slash = strrchr(info.dli_fname, '/');
            const char *basename = slash ? slash + 1 : info.dli_fname;
            unsigned long offset = (unsigned long)func - (unsigned long)info.dli_fbase;
            snprintf(entry->kernel_name, sizeof(entry->kernel_name),
                     "so:%s+0x%lx", basename, offset);
        } else {
            snprintf(entry->kernel_name, sizeof(entry->kernel_name), "%p", func);
        }
        entry->grid_x = gridX; entry->grid_y = gridY; entry->grid_z = gridZ;
        entry->block_x = blockX; entry->block_y = blockY; entry->block_z = blockZ;
        entry->shared_mem = smem;
        entry->stream = 0;
        g_trace_count++;
    }
    pthread_mutex_unlock(&g_trace_mutex);
    return cudaSuccess;
}

cudaError_t cudaFuncGetAttributes(void *attr, const void *func) {
    /* cudaFuncAttributes structure (CUDA 12.8, verified offsets):
     *   offset 0:  size_t sharedSizeBytes
     *   offset 8:  size_t constSizeBytes
     *   offset 16: size_t localSizeBytes
     *   offset 24: int maxThreadsPerBlock
     *   offset 28: int numRegs
     *   offset 32: int ptxVersion
     *   offset 36: int binaryVersion
     *   sizeof = 144
     */
    memset(attr, 0, 144);
    char *p = (char *)attr;
    *(int*)(p + 24) = 1024;   /* maxThreadsPerBlock — critical for grid calculation */
    *(int*)(p + 28) = 32;     /* numRegs */
    *(int*)(p + 32) = 80;     /* ptxVersion (8.0) */
    *(int*)(p + 36) = 80;     /* binaryVersion (8.0) */
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties_v2(void *prop, int device) {
    return cudaGetDeviceProperties(prop, device);
}

cudaError_t cudaGraphGetNodes(void *graph, void *nodes, size_t *numNodes) {
    if (numNodes) *numNodes = 0;
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiateWithFlags(void **pGraphExec, void *graph, unsigned long long flags) {
    *pGraphExec = (void *)0x6000;
    return cudaSuccess;
}

cudaError_t cudaGraphDebugDotPrint(void *graph, const char *path, unsigned int flags) {
    return cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void *userData) {
    if (fn) fn(userData);
    return cudaSuccess;
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                               size_t spitch, size_t width, size_t height,
                               int kind, cudaStream_t stream) {
    if (kind == 2 || kind == 3) {
        memset(dst, 0, dpitch * height);
    }
    return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
                                size_t offset, int kind) {
    return cudaSuccess;
}

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
    *pDevice = pHost;  /* unified address space emulation */
    return cudaSuccess;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    return cudaSuccess;
}

/* ── CUDA static registration ──
 * These MUST be provided when fully replacing libcudart.so.12, since
 * libtorch_cuda.so's static constructors call them with @libcudart.so.12 version.
 * They are no-ops: kernel registration is tracked but never executed.
 */

static void *g_fat_handles[8192];
static int g_fat_count = 0;

void** __cudaRegisterFatBinary(void *fatCubin) {
    int idx = __atomic_fetch_add(&g_fat_count, 1, __ATOMIC_RELAXED);
    if (idx >= 8192) {
        /* Fail loud rather than clamp: aliased handles silently break later
         * FatBinary lookups. If overflow occurs, raise the array size with
         * an actual count justified by the scenario. */
        fprintf(stderr, "[cuda_emulate] FATAL: g_fat_handles overflow (%d >= 8192). "
                "Increase the array size or investigate why so many fat binaries "
                "are registered.\n", idx);
        abort();
    }
    g_fat_handles[idx] = fatCubin;
    return &g_fat_handles[idx];
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) { }
void __cudaUnregisterFatBinary(void **fatCubinHandle) { }

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, void *tid, void *bid,
                            void *bDim, void *gDim, int *wSize) { }

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, size_t size,
                       int constant, int global) { }

/* ── Call configuration stack (thread-local) ── */
#define MAX_CALL_CONFIG_DEPTH 32
typedef struct {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    cudaStream_t stream;
} call_config_t;

static __thread call_config_t g_call_config_stack[MAX_CALL_CONFIG_DEPTH];
static __thread int g_call_config_top = 0;

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, cudaStream_t stream) {
    if (g_call_config_top < MAX_CALL_CONFIG_DEPTH) {
        g_call_config_stack[g_call_config_top].grid = gridDim;
        g_call_config_stack[g_call_config_top].block = blockDim;
        g_call_config_stack[g_call_config_top].shared_mem = sharedMem;
        g_call_config_stack[g_call_config_top].stream = stream;
        g_call_config_top++;
    }
    return 0;  /* cudaSuccess */
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                        size_t *sharedMem, cudaStream_t *stream) {
    if (g_call_config_top > 0) {
        g_call_config_top--;
        if (gridDim) *gridDim = g_call_config_stack[g_call_config_top].grid;
        if (blockDim) *blockDim = g_call_config_stack[g_call_config_top].block;
        if (sharedMem) *sharedMem = g_call_config_stack[g_call_config_top].shared_mem;
        if (stream) *stream = g_call_config_stack[g_call_config_top].stream;
    } else {
        if (gridDim) { gridDim->x = 1; gridDim->y = 1; gridDim->z = 1; }
        if (blockDim) { blockDim->x = 1; blockDim->y = 1; blockDim->z = 1; }
        if (sharedMem) *sharedMem = 0;
        if (stream) *stream = NULL;
    }
    return cudaSuccess;
}

void __cudaInitModule(void **fatCubinHandle) { }

/* ── Driver Entry Point (PyTorch uses to get cu* function pointers) ── */
#include <dlfcn.h>

cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, void *driverStatus) {
    if (!symbol || !funcPtr) return cudaSuccess;
    void *ptr = dlsym(RTLD_DEFAULT, symbol);
    *funcPtr = ptr;
    if (driverStatus) *(int*)driverStatus = ptr ? 0 : 1;
    return cudaSuccess;
}

cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, void *driverStatus) {
    return cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);
}
