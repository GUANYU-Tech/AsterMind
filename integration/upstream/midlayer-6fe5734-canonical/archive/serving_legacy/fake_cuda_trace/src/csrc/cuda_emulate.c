/*
 * CUDA Driver API Emulation Layer
 *
 * Intercepts CUDA driver calls via LD_PRELOAD to enable GPU-free
 * inference framework execution. Kernels are NOT executed; their
 * metadata is recorded to a trace file.
 *
 * Usage:
 *   LD_PRELOAD=./build/libcuda_emulate.so \
 *   CUDA_EMULATE_CONFIG=/path/to/config.json \
 *   CUDA_TRACE_OUTPUT=/tmp/cuda_trace.json \
 *   python -m vllm.entrypoints.openai.api_server ...
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o build/libcuda_emulate.so \
 *       src/emulation/cuda_emulate.c -lpthread
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

#include "cuda_emulate_shared.h"

/* ── CUDA driver type definitions ── */
typedef int CUresult;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUevent;
typedef void* CUdeviceptr;
typedef int CUdevice;

/* Extended types (opaque handles / enums) */
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef void* CUgraphNode;
typedef void* CUmemGenericAllocationHandle;
typedef void* CUarray;
typedef void* CUmipmappedArray;
typedef void* CUtexref;
typedef void* CUsurfref;
typedef unsigned long long CUipcEventHandle[2];  /* 16 bytes */
typedef unsigned long long CUipcMemHandle[2];     /* 16 bytes */
typedef int CUmoduleLoadingMode;
typedef int CUfunction_attribute;
typedef int CUfunc_cache;
typedef int CUsharedconfig;
typedef int CUlimit;
enum {
    CU_LIMIT_STACK_SIZE = 0,
    CU_LIMIT_PRINTF_FIFO_SIZE = 1,
    CU_LIMIT_MALLOC_HEAP_SIZE = 2,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
};
typedef void (*CUoccupancyB2DSize)(int);
typedef int CUstreamCaptureMode;
typedef int CUstreamCaptureState;
typedef int CUmemAllocationGranularity_flags;
typedef int CUmemAllocationHandleType;
typedef int CUarray_format;
typedef int CUfilter_mode;
typedef int CUaddress_mode;
typedef int CUpointer_attribute;
typedef int CUdevice_P2P_attribute;

/* Struct stubs for array/memory descriptors */
typedef struct { int Width; int Height; CUarray_format Format; int NumChannels; } CUDA_ARRAY_DESCRIPTOR;
typedef struct { int Width; int Height; int Depth; CUarray_format Format; int NumChannels; unsigned int Flags; } CUDA_ARRAY3D_DESCRIPTOR;
typedef struct { int type; int requestedHandleTypes; } CUmemAllocationProp;
typedef struct { int resType; } CUarrayMapInfo;

#define CUDA_SUCCESS 0

/* ── Global state definitions ── */
trace_entry_t *g_trace_entries;
int g_trace_count;
pthread_mutex_t g_trace_mutex;
gpu_config_t g_gpu;
unsigned long long g_next_ptr = 0x100000000ULL;
int g_gpu_device_count = 1;
char g_trace_output[512];
int g_initialized = 0;

/* ── Trace gating state ── */
int g_trace_enabled = 1;
int g_trace_gating_required = 0;

/* ── Simple JSON config parser (no external deps) ── */
static int parse_int_field(const char *json, const char *key, int default_val) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (pos == NULL) return default_val;
    pos = strstr(pos, ":");
    if (pos == NULL) return default_val;
    pos++;
    while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;
    return atoi(pos);
}

static unsigned long long parse_ull_field(const char *json, const char *key, unsigned long long default_val) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (pos == NULL) return default_val;
    pos = strstr(pos, ":");
    if (pos == NULL) return default_val;
    pos++;
    while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;
    return strtoull(pos, NULL, 10);
}

static void parse_string_field(const char *json, const char *key, char *buf, int buflen, const char *default_val) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (pos == NULL) { strncpy(buf, default_val, buflen); return; }
    pos = strstr(pos, ":");
    if (pos == NULL) { strncpy(buf, default_val, buflen); return; }
    pos++;
    while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;
    if (*pos == '"') {
        pos++;
        int i = 0;
        while (*pos != '"' && *pos != '\0' && i < buflen - 1) {
            buf[i++] = *pos++;
        }
        buf[i] = '\0';
    } else {
        strncpy(buf, default_val, buflen);
    }
}

static void load_config(void) {
    /* Set defaults */
    strncpy(g_gpu.gpu_name, "NVIDIA A800-SXM4-80GB", sizeof(g_gpu.gpu_name));
    g_gpu.total_memory = 85899345920ULL;
    g_gpu.major = 8;
    g_gpu.minor = 0;
    g_gpu.multi_processor_count = 108;
    g_gpu.max_threads_per_block = 1024;
    g_gpu.max_threads_per_multiprocessor = 2048;
    g_gpu.max_block_dim_x = 1024;
    g_gpu.max_block_dim_y = 1024;
    g_gpu.max_block_dim_z = 64;
    g_gpu.max_grid_dim_x = 2147483647;
    g_gpu.max_grid_dim_y = 65535;
    g_gpu.max_grid_dim_z = 65535;
    g_gpu.warp_size = 32;
    g_gpu.clock_rate = 1410000;
    g_gpu.memory_clock_rate = 1593000;
    g_gpu.memory_bus_width = 5120;
    g_gpu.l2_cache_size = 41943040;
    g_gpu.max_shared_memory_per_block = 49152;
    g_gpu.max_shared_memory_per_multiprocessor = 167936;
    
    /* Check env override for trace output */
    const char *trace_env = getenv("CUDA_TRACE_OUTPUT");
    if (trace_env) {
        strncpy(g_trace_output, trace_env, sizeof(g_trace_output) - 1);
    } else {
        strncpy(g_trace_output, "/tmp/cuda_trace.json", sizeof(g_trace_output));
    }

    /* Trace gating: if CUDA_TRACE_REQUIRE_PROFILER=1, trace is disabled
     * until cudaProfilerStart() is called. */
    const char *gating_env = getenv("CUDA_TRACE_REQUIRE_PROFILER");
    if (gating_env && gating_env[0] == '1') {
        g_trace_gating_required = 1;
        g_trace_enabled = 0;
        fprintf(stderr, "[cuda_emulate] Trace gating enabled (CUDA_TRACE_REQUIRE_PROFILER=1)\n");
    }

    /* Load config file */
    const char *config_path = getenv("CUDA_EMULATE_CONFIG");
    if (config_path == NULL) {
        fprintf(stderr, "[cuda_emulate] No config, using A800 defaults\n");
        return;
    }
    
    FILE *f = fopen(config_path, "r");
    if (f == NULL) {
        fprintf(stderr, "[cuda_emulate] Cannot open config %s\n", config_path);
        return;
    }
    
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = (char *)malloc(fsize + 1);
    fread(json, 1, fsize, f);
    json[fsize] = '\0';
    fclose(f);
    
    /* Find "gpu" section */
    const char *gpu_section = strstr(json, "\"gpu\"");
    if (gpu_section == NULL) {
        free(json);
        return;
    }
    
    parse_string_field(gpu_section, "name", g_gpu.gpu_name, sizeof(g_gpu.gpu_name), g_gpu.gpu_name);
    g_gpu.total_memory = parse_ull_field(gpu_section, "total_memory", g_gpu.total_memory);
    g_gpu.major = parse_int_field(gpu_section, "major", g_gpu.major);
    g_gpu.minor = parse_int_field(gpu_section, "minor", g_gpu.minor);
    g_gpu.multi_processor_count = parse_int_field(gpu_section, "multi_processor_count", g_gpu.multi_processor_count);
    g_gpu.max_threads_per_block = parse_int_field(gpu_section, "max_threads_per_block", g_gpu.max_threads_per_block);
    g_gpu.max_threads_per_multiprocessor = parse_int_field(gpu_section, "max_threads_per_multiprocessor", g_gpu.max_threads_per_multiprocessor);
    g_gpu.clock_rate = parse_int_field(gpu_section, "clock_rate", g_gpu.clock_rate);
    g_gpu.memory_clock_rate = parse_int_field(gpu_section, "memory_clock_rate", g_gpu.memory_clock_rate);
    g_gpu.memory_bus_width = parse_int_field(gpu_section, "memory_bus_width", g_gpu.memory_bus_width);
    g_gpu.l2_cache_size = parse_int_field(gpu_section, "l2_cache_size", g_gpu.l2_cache_size);
    
    /* Trace output from config */
    const char *trace_out = strstr(json, "\"trace_output\"");
    if (trace_out) {
        parse_string_field(trace_out, "trace_output", g_trace_output, sizeof(g_trace_output), g_trace_output);
    }
    
    free(json);
    fprintf(stderr, "[cuda_emulate] Config: GPU=%s, SMs=%d, Mem=%llu MB, trace=%s\n",
            g_gpu.gpu_name, g_gpu.multi_processor_count,
            g_gpu.total_memory / (1024*1024), g_trace_output);
}

/* ── Initialization ── */
static void __attribute__((constructor)) cuda_emulate_init(void) {
    if (g_initialized) return;
    fprintf(stderr, "[cuda_emulate] Initializing\n");
    
    g_next_ptr = 0x100000000ULL;
    g_trace_entries = (trace_entry_t *)calloc(MAX_TRACE_ENTRIES, sizeof(trace_entry_t));
    g_trace_count = 0;
    pthread_mutex_init(&g_trace_mutex, NULL);
    g_gpu_device_count = 1;
    
    load_config();
    g_initialized = 1;
}

/* ── Cleanup: write trace ── */
static void __attribute__((destructor)) cuda_emulate_fini(void) {
    if (!g_initialized) return;
    g_initialized = 0;  /* Prevent double-call from multiple .so destructors */
    
    /* Only write trace if this process actually captured kernel launches */
    if (g_trace_count == 0) {
        fprintf(stderr, "[cuda_emulate] Writing 0 trace entries (skipped, no kernels captured)\n");
        return;
    }
    
    /* Append PID to trace output path to avoid multi-process overwrite.
     * The process with the most entries is the one that ran the model. */
    char trace_path[600];
    snprintf(trace_path, sizeof(trace_path), "%s.%d", g_trace_output, (int)getpid());
    
    fprintf(stderr, "[cuda_emulate] Writing %d trace entries to %s\n",
            g_trace_count, trace_path);
    
    FILE *f = fopen(trace_path, "w");
    if (f == NULL) {
        fprintf(stderr, "[cuda_emulate] ERROR: Cannot open trace output\n");
        return;
    }
    
    fprintf(f, "[\n");
    for (int i = 0; i < g_trace_count; i++) {
        trace_entry_t *e = &g_trace_entries[i];
        fprintf(f, "  {\"timestamp_us\":%.1f, \"kernel_name\":\"%s\", "
                   "\"grid\":[%u,%u,%u], \"block\":[%u,%u,%u], "
                   "\"shared_mem\":%u, \"stream\":%lu}%s\n",
                e->timestamp_us, e->kernel_name,
                e->grid_x, e->grid_y, e->grid_z,
                e->block_x, e->block_y, e->block_z,
                e->shared_mem, e->stream,
                (i < g_trace_count - 1) ? "," : "");
    }
    fprintf(f, "]\n");
    fclose(f);
    
    free(g_trace_entries);
    pthread_mutex_destroy(&g_trace_mutex);
}

/* ── Timestamp ── */
static double get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

/* ══════════════════════════════════════════════════════════════
 * CUDA Driver API implementations
 * ══════════════════════════════════════════════════════════════ */

CUresult cuInit(unsigned int flags) {
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    *driverVersion = 12080;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    *count = g_gpu_device_count;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (ordinal >= g_gpu_device_count) return 1;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    strncpy(name, g_gpu.gpu_name, len - 1);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice dev) {
    switch (attrib) {
        case 1:  *value = g_gpu.max_threads_per_block; break;
        case 2:  *value = g_gpu.max_block_dim_x; break;
        case 3:  *value = g_gpu.max_block_dim_y; break;
        case 4:  *value = g_gpu.max_block_dim_z; break;
        case 5:  *value = g_gpu.max_grid_dim_x; break;
        case 6:  *value = g_gpu.max_grid_dim_y; break;
        case 7:  *value = g_gpu.max_grid_dim_z; break;
        case 8:  *value = g_gpu.max_shared_memory_per_block; break;
        case 9:  *value = 65536; break; /* TOTAL_CONSTANT_MEMORY */
        case 10: *value = g_gpu.warp_size; break;
        case 12: *value = 65536; break; /* MAX_REGISTERS_PER_BLOCK */
        case 13: *value = g_gpu.clock_rate; break;
        case 14: *value = 512; break; /* TEXTURE_ALIGNMENT */
        case 17: *value = g_gpu.multi_processor_count; break;
        case 18: *value = 0; break; /* KERNEL_EXEC_TIMEOUT */
        case 20: *value = 0; break; /* INTEGRATED */
        case 21: *value = 1; break; /* CAN_MAP_HOST_MEMORY */
        case 24: *value = g_gpu.major; break;
        case 25: *value = g_gpu.minor; break;
        case 28: *value = g_gpu.max_threads_per_multiprocessor; break;
        case 36: *value = g_gpu.max_shared_memory_per_multiprocessor; break;
        case 37: *value = 65536; break; /* MAX_REGISTERS_PER_MULTIPROCESSOR */
        case 39: *value = g_gpu.memory_clock_rate; break;
        case 40: *value = g_gpu.memory_bus_width; break;
        case 42: *value = g_gpu.l2_cache_size; break;
        case 46: *value = 1; break; /* CONCURRENT_KERNELS */
        case 68: *value = g_gpu.max_shared_memory_per_multiprocessor; break;
        case 75: *value = 1; break; /* COMPUTE_PREEMPTION_SUPPORTED */
        case 87: *value = 1; break; /* PAGEABLE_MEMORY_ACCESS */
        case 88: *value = 1; break; /* PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES */
        case 97: *value = g_gpu.max_shared_memory_per_multiprocessor; break; /* MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: Ampere A800=167936 */
        default: *value = 0; break;
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(unsigned long long *bytes, CUdevice dev) {
    *bytes = g_gpu.total_memory;
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    *pctx = (CUcontext)0x1;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) { return CUDA_SUCCESS; }
CUresult cuCtxSetCurrent(CUcontext ctx) { return CUDA_SUCCESS; }

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    *pctx = (CUcontext)0x1;
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) { return CUDA_SUCCESS; }
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    if (!pvalue) return 1;
    /* Return reasonable defaults for A800 */
    if (limit == CU_LIMIT_STACK_SIZE) *pvalue = 1024;
    else if (limit == CU_LIMIT_PRINTF_FIFO_SIZE) *pvalue = 1048576;
    else if (limit == CU_LIMIT_MALLOC_HEAP_SIZE) *pvalue = 8388608;
    else if (limit == CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH) *pvalue = 4;
    else if (limit == CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT) *pvalue = 2048;
    else *pvalue = 0;
    return CUDA_SUCCESS;
}
CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    (void)limit; (void)value;
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, unsigned long long bytesize) {
    *dptr = (CUdeviceptr)g_next_ptr;
    g_next_ptr += bytesize;
    g_next_ptr = (g_next_ptr + 255) & ~255ULL;
    return CUDA_SUCCESS;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, unsigned long long bytesize, unsigned int flags) {
    return cuMemAlloc(dptr, bytesize);
}

CUresult cuMemFree(CUdeviceptr dptr) { return CUDA_SUCCESS; }
CUresult cuMemFreeHost(void *p) { free(p); return CUDA_SUCCESS; }

CUresult cuMemAllocHost(void **pp, unsigned long long bytesize) {
    *pp = calloc(1, bytesize);
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo(unsigned long long *free, unsigned long long *total) {
    *total = g_gpu.total_memory;
    *free = *total * 80 / 100;
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, unsigned long long n) { return CUDA_SUCCESS; }

CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, unsigned long long n) {
    memset(dst, 0, n);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, unsigned long long n) { return CUDA_SUCCESS; }
CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, unsigned long long n, CUstream s) { return CUDA_SUCCESS; }

CUresult cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, unsigned long long n, CUstream s) {
    memset(dst, 0, n);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8(CUdeviceptr dst, unsigned char uc, unsigned long long n) { return CUDA_SUCCESS; }
CUresult cuMemsetD32(CUdeviceptr dst, unsigned int ui, unsigned long long n) { return CUDA_SUCCESS; }

CUresult cuModuleLoad(CUmodule *module, const char *fname) { *module = (CUmodule)0x2; return CUDA_SUCCESS; }
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    (void)image;
    *module = (CUmodule)0x2;
    return CUDA_SUCCESS;
}
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) { *module = (CUmodule)0x2; return CUDA_SUCCESS; }
CUresult cuModuleUnload(CUmodule hmod) { return CUDA_SUCCESS; }

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    unsigned long hash = 0xDEAD0000UL | (unsigned long)(name[0] & 0xFF);
    *hfunc = (CUfunction)hash;
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void **kernelParams,
                        void **extra) {
    if (!g_trace_enabled) return CUDA_SUCCESS;
    pthread_mutex_lock(&g_trace_mutex);
    if (g_trace_count < MAX_TRACE_ENTRIES) {
        trace_entry_t *entry = &g_trace_entries[g_trace_count];
        entry->timestamp_us = get_timestamp_us();
        /* Resolve function pointer to symbol name via dladdr.
         * Many kernels from dlopen'd .so (flash_attn, vllm ops) have symbols
         * that dladdr can resolve if the .so was loaded with RTLD_GLOBAL or
         * if we search all loaded modules.
         * Single dladdr() call: reuse the result for both the symbol-name and
         * the so:basename+offset fallback path (P53.f cleanup). */
        Dl_info info;
        int dladdr_ok = dladdr(f, &info);
        if (dladdr_ok && info.dli_sname) {
            snprintf(entry->kernel_name, sizeof(entry->kernel_name), "%s", info.dli_sname);
        } else if (dladdr_ok && info.dli_fname) {
            /* dladdr returned the .so but no symbol name — likely loaded with
             * RTLD_LOCAL. Store the .so filename + offset for post-processing.
             * Format: "so:basename+offset" where offset is relative to the
             * library's text section base. */
            const char *slash = strrchr(info.dli_fname, '/');
            const char *basename = slash ? slash + 1 : info.dli_fname;
            unsigned long offset = (unsigned long)f - (unsigned long)info.dli_fbase;
            snprintf(entry->kernel_name, sizeof(entry->kernel_name),
                     "so:%s+0x%lx", basename, offset);
        } else {
            snprintf(entry->kernel_name, sizeof(entry->kernel_name),
                     "kernel_%p", (void*)f);
        }
        entry->grid_x = gridDimX; entry->grid_y = gridDimY; entry->grid_z = gridDimZ;
        entry->block_x = blockDimX; entry->block_y = blockDimY; entry->block_z = blockDimZ;
        entry->shared_mem = sharedMemBytes;
        entry->stream = (unsigned long)hStream;
        g_trace_count++;
    }
    pthread_mutex_unlock(&g_trace_mutex);
    return CUDA_SUCCESS;
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    static unsigned long stream_counter = 0x1000;
    *phStream = (CUstream)(stream_counter++);
    return CUDA_SUCCESS;
}

CUresult cuStreamDestroy(CUstream hStream) { return CUDA_SUCCESS; }
CUresult cuStreamSynchronize(CUstream hStream) { return CUDA_SUCCESS; }

CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    static unsigned long event_counter = 0x2000;
    *phEvent = (CUevent)(event_counter++);
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy(CUevent hEvent) { return CUDA_SUCCESS; }
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) { return CUDA_SUCCESS; }
CUresult cuEventSynchronize(CUevent hEvent) { return CUDA_SUCCESS; }

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    *pMilliseconds = 0.0f;
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(const char **pStr, CUresult error) {
    *pStr = "unknown error";
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(const char **pStr, CUresult error) {
    *pStr = "CUDA_ERROR_UNKNOWN";
    return CUDA_SUCCESS;
}

/* ── Primary context management ── */
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    *pctx = (CUcontext)0x1;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    *flags = 0;
    *active = 1;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    return CUDA_SUCCESS;
}

/* ── Context management (extended) ── */
CUresult cuCtxGetDevice(CUdevice *device) {
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int *flags) {
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    *version = 12080;
    return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
    *pctx = (CUcontext)0x1;
    return CUDA_SUCCESS;
}

/* ── Memory management (extended) ── */
CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    *dptr = (CUdeviceptr)g_next_ptr;
    g_next_ptr += WidthInBytes * Height;
    g_next_ptr = (g_next_ptr + 255) & ~255ULL;
    *pPitch = WidthInBytes;
    return CUDA_SUCCESS;
}

/* cuMemAllocManaged already defined above (with unsigned long long bytesize) */

CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult cuMemHostUnregister(void *p) {
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    *pdptr = (CUdeviceptr)p;
    return CUDA_SUCCESS;
}

CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    *pFlags = 0;
    return CUDA_SUCCESS;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    return CUDA_SUCCESS;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2D(void *pCopy) {
    return CUDA_SUCCESS;
}

CUresult cuMemcpy2DAsync(void *pCopy, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3D(void *pCopy) {
    return CUDA_SUCCESS;
}

CUresult cuMemcpy3DAsync(void *pCopy, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char uc, size_t N, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int ui, size_t N, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) {
    return CUDA_SUCCESS;
}

/* ── Module management (extended) ── */
CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
    *mode = 0; /* EMBEDDED */
    return CUDA_SUCCESS;
}

/* cuModuleGetFunction already defined above */

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    *dptr = (CUdeviceptr)g_next_ptr;
    g_next_ptr += 256;
    *bytes = 256;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetSurfRef(void **pSurfRef, CUmodule hmod, const char *name) {
    *pSurfRef = (void*)0x8;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetTexRef(void **pTexRef, CUmodule hmod, const char *name) {
    *pTexRef = (void*)0x9;
    return CUDA_SUCCESS;
}

/* ── Function attributes ── */
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
    /* Return physically correct values for compiled kernel attributes.
     * These are queried by Triton during kernel launch validation.
     * Enum values from cuda.h:
     *   0=MAX_THREADS_PER_BLOCK, 1=SHARED_SIZE_BYTES, 2=CONST_SIZE_BYTES,
     *   3=LOCAL_SIZE_BYTES, 4=NUM_REGS, 5=PTX_VERSION, 6=BINARY_VERSION,
     *   7=CACHE_MODE_CA, 8=MAX_DYNAMIC_SHARED_SIZE_BYTES,
     *   9=PREFERRED_SHARED_MEMORY_CARVEOUT */
    switch (attrib) {
        case 0:  *pi = g_gpu.max_threads_per_block; break;  /* MAX_THREADS_PER_BLOCK */
        case 1:  *pi = 0; break;  /* SHARED_SIZE_BYTES (static shared, kernel-specific) */
        case 2:  *pi = 64; break; /* CONST_SIZE_BYTES */
        case 3:  *pi = 0; break;  /* LOCAL_SIZE_BYTES (local memory per thread) */
        case 4:  *pi = 32; break; /* NUM_REGS (typical for Triton kernels) */
        case 5:  *pi = 80; break; /* PTX_VERSION (8.0 = SM80) */
        case 6:  *pi = 80; break; /* BINARY_VERSION (SM80 for Ampere) */
        case 7:  *pi = 1; break;  /* CACHE_MODE_CA */
        case 8:  *pi = g_gpu.max_shared_memory_per_block - 0; break;  /* MAX_DYNAMIC_SHARED_SIZE_BYTES */
        case 9:  *pi = 1; break;  /* PREFERRED_SHARED_MEMORY_CARVEOUT */
        default: *pi = 0; break;
    }
    return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    return CUDA_SUCCESS;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
    return CUDA_SUCCESS;
}

/* ── Occupancy ── */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
    *numBlocks = g_gpu.max_threads_per_multiprocessor / blockSize;
    if (*numBlocks > 32) *numBlocks = 32;
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, int dynamicSMemSize, int blockSizeLimit) {
    *blockSize = 256;
    *minGridSize = 1;
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, int dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
    *blockSize = 256;
    *minGridSize = 1;
    return CUDA_SUCCESS;
}

/* ── Stream management (extended) ── */
CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    *pctx = (CUcontext)0x1;
    return CUDA_SUCCESS;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) {
    return CUDA_SUCCESS;
}

CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) {
    return CUDA_SUCCESS;
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
    *phGraph = (CUgraph)0xA;
    return CUDA_SUCCESS;
}

CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureState *pCaptureStatus) {
    *pCaptureStatus = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureState *pCaptureStatus, unsigned long long *id) {
    *pCaptureStatus = 0;
    *id = 0;
    return CUDA_SUCCESS;
}

/* ── Event management (extended) ── */
CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
    return CUDA_SUCCESS;
}

/* ── Graph management ── */
CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
    *phGraph = (CUgraph)0xB;
    return CUDA_SUCCESS;
}

CUresult cuGraphDestroy(CUgraph hGraph) {
    return CUDA_SUCCESS;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
    return CUDA_SUCCESS;
}

CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize) {
    *phGraphExec = (CUgraphExec)0xC;
    return CUDA_SUCCESS;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags) {
    *phGraphExec = (CUgraphExec)0xC;
    return CUDA_SUCCESS;
}

/* ── Virtual memory management ── */
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    *ptr = (CUdeviceptr)g_next_ptr;
    g_next_ptr += size;
    g_next_ptr = (g_next_ptr + 65535) & ~65535ULL;
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    return CUDA_SUCCESS;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags) {
    *handle = (CUmemGenericAllocationHandle)0xD;
    return CUDA_SUCCESS;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    return CUDA_SUCCESS;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    return CUDA_SUCCESS;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    return CUDA_SUCCESS;
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream) {
    return CUDA_SUCCESS;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
    *handle = (CUmemGenericAllocationHandle)0xD;
    return CUDA_SUCCESS;
}

CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option) {
    *granularity = 2097152; /* 2MB */
    return CUDA_SUCCESS;
}

CUresult cuMemGetAllocationInfo(CUmemGenericAllocationHandle handle, CUmemAllocationProp *prop) {
    memset(prop, 0, sizeof(CUmemAllocationProp));
    return CUDA_SUCCESS;
}

CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
    memset(shareableHandle, 0, 64);
    return CUDA_SUCCESS;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *shareableHandle, CUmemAllocationHandleType handleType) {
    *handle = (CUmemGenericAllocationHandle)0xD;
    return CUDA_SUCCESS;
}

/* ── Multicast ── */
CUresult cuMulticastCreate(CUmemGenericAllocationHandle *mcHandle, unsigned int numDevices) {
    *mcHandle = (CUmemGenericAllocationHandle)0xE;
    return CUDA_SUCCESS;
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev) {
    return CUDA_SUCCESS;
}

CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) {
    return CUDA_SUCCESS;
}

CUresult cuMulticastUnbindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdevice dev) {
    return CUDA_SUCCESS;
}

/* ── Array management ── */
CUresult cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    *pHandle = (CUarray)0xF;
    return CUDA_SUCCESS;
}

CUresult cuArrayDestroy(CUarray hArray) {
    return CUDA_SUCCESS;
}

CUresult cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    *pHandle = (CUarray)0xF;
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipDesc, unsigned int numMipLevels) {
    *pHandle = (CUmipmappedArray)0x10;
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    return CUDA_SUCCESS;
}

CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) {
    *pLevelArray = (CUarray)0xF;
    return CUDA_SUCCESS;
}

/* ── Texture/Surface reference ── */
CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
    *ByteOffset = 0;
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
    *phArray = (CUarray)0xF;
    return CUDA_SUCCESS;
}

/* ── Profiler ── */
CUresult cuProfilerStart(void) {
    if (g_trace_gating_required) {
        g_trace_enabled = 1;
        fprintf(stderr, "[cuda_emulate] Trace enabled (cuProfilerStart)\n");
    }
    return CUDA_SUCCESS;
}

CUresult cuProfilerStop(void) {
    if (g_trace_gating_required) {
        g_trace_enabled = 0;
        fprintf(stderr, "[cuda_emulate] Trace disabled (cuProfilerStop)\n");
    }
    return CUDA_SUCCESS;
}

/* ── Device management (extended) ── */
CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    *dev = 0;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    strncpy(pciBusId, "0000:00:00.0", len);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags) {
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2P_attribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
    *value = 1;
    return CUDA_SUCCESS;
}

CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    *canAccessPeer = 1;
    return CUDA_SUCCESS;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    return CUDA_SUCCESS;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    return CUDA_SUCCESS;
}

/* ── Pointer attributes ── */
CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    memset(data, 0, 8);
    return CUDA_SUCCESS;
}

CUresult cuPointerSetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr) {
    return CUDA_SUCCESS;
}

/* ── Ipc ── */
CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
    memset(pHandle, 0, sizeof(CUipcEventHandle));
    return CUDA_SUCCESS;
}

CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
    *phEvent = (CUevent)0x2001;
    return CUDA_SUCCESS;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
    memset(pHandle, 0, sizeof(CUipcMemHandle));
    return CUDA_SUCCESS;
}

CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags) {
    *pdptr = (CUdeviceptr)g_next_ptr;
    g_next_ptr += 4096;
    return CUDA_SUCCESS;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    return CUDA_SUCCESS;
}

/* ── Linker (JIT) ── */
CUresult cuLinkCreate_v2(unsigned int numOptions, void *options, void **optionValues, void **stateOut) {
    static unsigned long fake_link = 0x5001;
    *stateOut = (void*)fake_link++;
    return CUDA_SUCCESS;
}

CUresult cuLinkAddData_v2(void *state, int type, void *data, size_t size, const char *name,
                          unsigned int numOptions, void *options, void **optionValues) {
    return CUDA_SUCCESS;
}

CUresult cuLinkAddFile_v2(void *state, int type, const char *path,
                          unsigned int numOptions, void *options, void **optionValues) {
    return CUDA_SUCCESS;
}

CUresult cuLinkComplete(void *state, void **cubinOut, size_t *sizeOut) {
    static char fake_cubin[8] = {0};
    if (cubinOut) *cubinOut = fake_cubin;
    if (sizeOut) *sizeOut = sizeof(fake_cubin);
    return CUDA_SUCCESS;
}

CUresult cuLinkDestroy(void *state) {
    return CUDA_SUCCESS;
}

/* ── Module extended ── */
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                            void *options, void **optionValues) {
    static unsigned long fake_mod = 0x6001;
    *module = (CUmodule)fake_mod++;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    *dptr = (CUdeviceptr)g_next_ptr;
    g_next_ptr += 256;
    if (bytes) *bytes = 256;
    return CUDA_SUCCESS;
}

/* ── Launch extended ── */
CUresult cuLaunchKernelEx(void *config, CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                          unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                          unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                          void **kernelParams, void **extra) {
    /* Delegate to standard cuLaunchKernel trace recording */
    return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                          blockDimX, blockDimY, blockDimZ,
                          sharedMemBytes, hStream, kernelParams, extra);
}

/* ── Memory copy ── */
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    /* Return zeros — kernel outputs are not executed */
    if (dstHost && ByteCount > 0) {
        memset(dstHost, 0, ByteCount);
    }
    return CUDA_SUCCESS;
}

/* ── TensorMap (Hopper+) ── */
CUresult cuTensorMapEncodeTiled(void *tensorMap, unsigned int tensorDataType,
                                unsigned int tensorRank, void *globalAddress,
                                const long long *globalDim, const long long *globalStrides,
                                const unsigned int *boxDim, const unsigned int *elementStrides,
                                unsigned int interleave, unsigned int swizzle,
                                unsigned int l2Promotion, unsigned int oobFill) {
    if (tensorMap) memset(tensorMap, 0, 128);  /* CUtensorMap is 128 bytes */
    return CUDA_SUCCESS;
}
