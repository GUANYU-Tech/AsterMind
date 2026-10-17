/*
 * Shared types and globals for CUDA emulation layer.
 */

#ifndef CUDA_EMULATE_SHARED_H
#define CUDA_EMULATE_SHARED_H

#include <pthread.h>

/* ── Trace entry ── */
typedef struct {
    double timestamp_us;
    char kernel_name[256];
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
    unsigned int shared_mem;
    unsigned long stream;
} trace_entry_t;

/* ── Config ── */
typedef struct {
    char gpu_name[128];
    unsigned long long total_memory;
    int major, minor;
    int multi_processor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int max_block_dim_x, max_block_dim_y, max_block_dim_z;
    int max_grid_dim_x, max_grid_dim_y, max_grid_dim_z;
    int warp_size;
    int clock_rate;
    int memory_clock_rate;
    int memory_bus_width;
    int l2_cache_size;
    int max_shared_memory_per_block;
    int max_shared_memory_per_multiprocessor;
} gpu_config_t;

/* ── Global state ── */
extern trace_entry_t *g_trace_entries;
extern int g_trace_count;
extern pthread_mutex_t g_trace_mutex;
extern gpu_config_t g_gpu;
extern unsigned long long g_next_ptr;
extern int g_gpu_device_count;
extern char g_trace_output[512];
extern int g_initialized;

/* ── Trace gating ──
 * g_trace_enabled:    master switch; trace entries are only recorded when true.
 * g_trace_gating_required: when true, cudaProfilerStart/Stop toggle the gate.
 * Default: g_trace_enabled=1 (backward compatible), g_trace_gating_required=0.
 * Set CUDA_TRACE_REQUIRE_PROFILER=1 to enable gating mode. */
extern int g_trace_enabled;
extern int g_trace_gating_required;

#define MAX_TRACE_ENTRIES 200000

/* ── Direct trace recording ──
 * Allows cuBLAS stub to record trace entries with custom kernel names,
 * bypassing dladdr resolution. Implemented in libcudart.so.12.
 * Called from libcublas.so.12 via dynamic linking. */
extern void trace_record_direct(const char *kernel_name,
                                  unsigned int gx, unsigned int gy, unsigned int gz,
                                  unsigned int bx, unsigned int by, unsigned int bz,
                                  unsigned int shared_mem, unsigned long stream);

#endif
