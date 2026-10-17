/**
 * NCCL emulation stub library.
 *
 * Intercepts NCCL API calls via LD_LIBRARY_PATH or VLLM_NCCL_SO_PATH,
 * allowing vLLM to run with tensor parallelism under CUDA emulation.
 *
 * Communication operations (allreduce/allgather/etc.) are no-ops that:
 * 1. Record the operation in the trace buffer (same as CUDA kernels)
 * 2. Return immediately without actual data transfer
 *
 * Physical communication latency is predicted offline by the trace predictor
 * using the interconnect bandwidth model from hw_configs YAML.
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o build/fake_cuda_libs/libnccl.so.2 \
 *       src/emulation/nccl_emulate.c -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dlfcn.h>

/* ── NCCL type definitions ── */

typedef int ncclResult_t;
#define ncclSuccess 0
#define ncclUnhandledCudaError 1
#define ncclSystemError 2
#define ncclInternalError 3
#define ncclInvalidArgument 4
#define ncclInvalidUsage 5

typedef void* ncclComm_t;

/* ncclUniqueId: 128-byte opaque identifier */
typedef struct { char internal[128]; } ncclUniqueId;

/* NCCL data types */
typedef enum {
    ncclInt8 = 0, ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2, ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6, ncclHalf = 6,
    ncclFloat32 = 7, ncclFloat = 7,
    ncclFloat64 = 8, ncclDouble = 8,
    ncclBfloat16 = 9,
    ncclNumTypes = 10,
} ncclDataType_t;

/* NCCL reduction operations */
typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
    ncclNumOps = 5,
} ncclRedOp_t;

typedef enum {
    ncclSum_resident = 0,
} ncclRedOp_resident_t;

/* ── Trace recording ── */

/* Import trace_record_direct from libcudart.so.12 */
extern void trace_record_direct(const char *name,
                                unsigned int gx, unsigned int gy, unsigned int gz,
                                unsigned int bx, unsigned int by, unsigned int bz,
                                unsigned int smem, unsigned long stream);

/* Virtual communicator state */
static int g_nccl_nranks = 0;
static int g_nccl_rank = 0;
static ncclComm_t g_nccl_comm = (ncclComm_t)0x10000000;  /* sentinel */

/* Communication counter for trace naming */
static int g_nccl_op_count = 0;
static pthread_mutex_t g_nccl_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ── Shared memory IPC for ncclCommInitRank barrier ── */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

/* Barrier shared memory layout:
 * - int nranks: total number of ranks (set by first rank)
 * - int arrived: count of ranks that have arrived
 * - int generation: barrier generation counter
 * - pid_t pids[64]: PIDs of arrived ranks (for cleanup)
 */
#define NCCL_BARRIER_SHM_NAME "/midlayer_nccl_barrier"
#define NCCL_MAX_RANKS 64

typedef struct {
    int nranks;
    int arrived;
    int generation;
    pid_t pids[NCCL_MAX_RANKS];
} nccl_barrier_shm_t;

static nccl_barrier_shm_t* g_barrier_shm = NULL;
static int g_barrier_fd = -1;

static nccl_barrier_shm_t* nccl_barrier_get(int nranks) {
    if (g_barrier_shm) return g_barrier_shm;

    /* Try to create new shared memory; if exists, open existing */
    int created = 0;
    g_barrier_fd = shm_open(NCCL_BARRIER_SHM_NAME, O_CREAT | O_EXCL | O_RDWR, 0666);
    if (g_barrier_fd >= 0) {
        created = 1;
        ftruncate(g_barrier_fd, sizeof(nccl_barrier_shm_t));
    } else {
        g_barrier_fd = shm_open(NCCL_BARRIER_SHM_NAME, O_RDWR, 0666);
    }
    if (g_barrier_fd < 0) {
        fprintf(stderr, "[nccl_emulate] shm_open failed: %s\n", strerror(errno));
        return NULL;
    }

    g_barrier_shm = (nccl_barrier_shm_t*)mmap(NULL, sizeof(nccl_barrier_shm_t),
                                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                                g_barrier_fd, 0);
    if (g_barrier_shm == MAP_FAILED) {
        fprintf(stderr, "[nccl_emulate] mmap failed: %s\n", strerror(errno));
        g_barrier_shm = NULL;
        close(g_barrier_fd);
        g_barrier_fd = -1;
        return NULL;
    }

    if (created) {
        g_barrier_shm->nranks = nranks;
        g_barrier_shm->arrived = 0;
        g_barrier_shm->generation = 0;
        memset(g_barrier_shm->pids, 0, sizeof(g_barrier_shm->pids));
    }

    return g_barrier_shm;
}

static void nccl_barrier_wait(nccl_barrier_shm_t* shm, int rank) {
    if (!shm) return;

    /* Register arrival */
    shm->pids[rank] = getpid();
    int old_gen = __atomic_load_n(&shm->generation, __ATOMIC_ACQUIRE);
    int n = __sync_add_and_fetch(&shm->arrived, 1);

    if (n >= shm->nranks) {
        /* Last to arrive: reset `arrived` BEFORE publishing the new generation.
         * If we published `generation = g+1` first (RELEASE), a fast rank could
         * observe gen=g+1, immediately re-enter nccl_barrier_wait for gen g+1,
         * increment `arrived` to 1, and have its arrival count erased by our
         * subsequent `arrived = 0` store — deadlocking the next barrier.
         * Resetting `arrived` before the RELEASE store on `generation` ensures
         * no waiter can wake (and thus no waiter can re-increment `arrived`)
         * until the counter has already been cleared. */
        __atomic_store_n(&shm->arrived, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&shm->generation, old_gen + 1, __ATOMIC_RELEASE);
    } else {
        /* Wait for generation to advance (do NOT check arrived count —
         * that creates a race window between reset and generation bump). */
        int timeout = 3000; /* 30 seconds (10ms * 3000) */
        while (__atomic_load_n(&shm->generation, __ATOMIC_ACQUIRE) == old_gen) {
            usleep(10000); /* 10ms poll interval */
            timeout--;
            if (timeout <= 0) {
                fprintf(stderr, "[nccl_emulate] barrier timeout: rank=%d, gen=%d, arrived=%d/%d\n",
                        rank, old_gen, __atomic_load_n(&shm->arrived, __ATOMIC_RELAXED),
                        shm->nranks);
                break;
            }
        }
    }
}

static void nccl_barrier_cleanup(void) {
    if (g_barrier_shm) {
        munmap(g_barrier_shm, sizeof(nccl_barrier_shm_t));
        g_barrier_shm = NULL;
    }
    if (g_barrier_fd >= 0) {
        close(g_barrier_fd);
        g_barrier_fd = -1;
    }
    /* Note: do NOT shm_unlink here — another rank may still be using the shm.
     * The shm segment persists until all processes unmap it. It will be cleaned
     * up on next ncclGetUniqueId call (O_CREAT|O_EXCL succeeds if no process
     * holds an fd), or manually via: rm /dev/shm/midlayer_nccl_barrier */
}

/* ── NCCL API stubs ── */

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (!uniqueId) return ncclInvalidArgument;
    /* Fill with a deterministic pattern so all ranks get the same ID */
    memset(uniqueId->internal, 0xAB, 128);
    /* Encode a shared memory key hint in the first bytes */
    memcpy(uniqueId->internal, NCCL_BARRIER_SHM_NAME, 
           sizeof(NCCL_BARRIER_SHM_NAME) < 128 ? sizeof(NCCL_BARRIER_SHM_NAME) : 128);
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                               ncclUniqueId commId, int rank) {
    if (!comm) return ncclInvalidArgument;
    g_nccl_nranks = nranks;
    g_nccl_rank = rank;
    *comm = (ncclComm_t)(0x10000001ULL + rank);
    fprintf(stderr, "[nccl_emulate] ncclCommInitRank: rank=%d nranks=%d\n",
            rank, nranks);

    /* Use shared memory barrier to synchronize all ranks */
    nccl_barrier_shm_t* shm = nccl_barrier_get(nranks);
    if (shm) {
        nccl_barrier_wait(shm, rank);
        fprintf(stderr, "[nccl_emulate] ncclCommInitRank: rank=%d barrier passed\n", rank);
    }

    return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comm, int nranks, const int* ranklist) {
    if (!comm) return ncclInvalidArgument;
    g_nccl_nranks = nranks;
    g_nccl_rank = ranklist ? ranklist[0] : 0;
    *comm = (ncclComm_t)(0x10000001ULL);
    return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    nccl_barrier_cleanup();
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(ncclComm_t comm, int* devid) {
    if (devid) *devid = 0;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    if (count) *count = g_nccl_nranks;
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    if (rank) *rank = g_nccl_rank;
    return ncclSuccess;
}

ncclResult_t ncclGetErrorString(ncclResult_t result, const char** str) {
    if (!str) return ncclInvalidArgument;
    switch (result) {
        case ncclSuccess: *str = "ncclSuccess"; break;
        case ncclUnhandledCudaError: *str = "ncclUnhandledCudaError"; break;
        case ncclSystemError: *str = "ncclSystemError"; break;
        case ncclInternalError: *str = "ncclInternalError"; break;
        case ncclInvalidArgument: *str = "ncclInvalidArgument"; break;
        case ncclInvalidUsage: *str = "ncclInvalidUsage"; break;
        default: *str = "ncclUnknownError"; break;
    }
    return ncclSuccess;
}

ncclResult_t ncclGetLastError(ncclComm_t comm) {
    return ncclSuccess;
}

/* ── Communication operations ── */

/**
 * Record a communication operation in the trace buffer.
 * Physical latency prediction is done offline by the trace predictor
 * using message size × interconnect bandwidth model.
 */
static void record_nccl_op(const char* op_name, size_t count,
                            ncclDataType_t datatype, int nranks,
                            unsigned long stream) {
    pthread_mutex_lock(&g_nccl_mutex);
    int op_id = g_nccl_op_count++;
    pthread_mutex_unlock(&g_nccl_mutex);

    /* Estimate message size in bytes */
    static const int type_sizes[] = {
        1, 1, 4, 4, 8, 8, 2, 4, 8, 2  /* ncclInt8..ncclBfloat16 */
    };
    int elem_size = (datatype >= 0 && datatype < ncclNumTypes)
                    ? type_sizes[datatype] : 4;
    size_t bytes = count * elem_size;

    /* Record as a single-entry trace with bytes in shared_mem field.
     * Grid dimensions encode nranks and rank for offline analysis:
     *   grid.x = nranks, grid.y = rank, grid.z = op_id
     * Block dimensions encode message size info:
     *   block.x = min(bytes, 0xFFFFFFFF), block.y = bytes >> 32
     */
    char name[128];
    snprintf(name, sizeof(name), "nccl_%s", op_name);

    trace_record_direct(name,
                        (unsigned int)nranks,
                        (unsigned int)g_nccl_rank,
                        (unsigned int)op_id,
                        (unsigned int)(bytes & 0xFFFFFFFF),
                        (unsigned int)(bytes >> 32),
                        0,  /* bz */
                        0,  /* smem */
                        stream);
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, void* stream) {
    record_nccl_op("allreduce", count, datatype, g_nccl_nranks,
                   (unsigned long)stream);
    return ncclSuccess;
}

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int root,
                         ncclComm_t comm, void* stream) {
    record_nccl_op("reduce", count, datatype, g_nccl_nranks,
                   (unsigned long)stream);
    return ncclSuccess;
}

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                            ncclDataType_t datatype, ncclComm_t comm,
                            void* stream) {
    record_nccl_op("allgather", sendcount, datatype, g_nccl_nranks,
                   (unsigned long)stream);
    return ncclSuccess;
}

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                                size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm, void* stream) {
    record_nccl_op("reducescatter", recvcount, datatype, g_nccl_nranks,
                   (unsigned long)stream);
    return ncclSuccess;
}

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int root,
                            ncclComm_t comm, void* stream) {
    record_nccl_op("broadcast", count, datatype, g_nccl_nranks,
                   (unsigned long)stream);
    return ncclSuccess;
}

ncclResult_t ncclSend(const void* sendbuff, size_t count,
                       ncclDataType_t datatype, int peer,
                       ncclComm_t comm, void* stream) {
    record_nccl_op("send", count, datatype, g_nccl_nranks,
                   (unsigned long)stream);
    return ncclSuccess;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count,
                       ncclDataType_t datatype, int peer,
                       ncclComm_t comm, void* stream) {
    record_nccl_op("recv", count, datatype, g_nccl_nranks,
                   (unsigned long)stream);
    return ncclSuccess;
}

/* Group operations */
ncclResult_t ncclGroupStart(void) {
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void) {
    return ncclSuccess;
}

/* NCCL memory registration (used by PyTorch) */
ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
    if (ptr) *ptr = NULL;
    return ncclSuccess;
}

ncclResult_t ncclMemFree(void* ptr) {
    return ncclSuccess;
}

/* NCCL proxy operations (used by vLLM) */
ncclResult_t ncclCommRegister(void* comm, void* buff, size_t size, void** handle) {
    if (handle) *handle = NULL;
    return ncclSuccess;
}

ncclResult_t ncclCommDeregister(void* comm, void* handle) {
    return ncclSuccess;
}

ncclResult_t ncclCommFinalize(void* comm) {
    return ncclSuccess;
}

ncclResult_t ncclCommGetAsyncError(void* comm, int* async_error) {
    if (async_error) *async_error = 0;
    return ncclSuccess;
}

ncclResult_t ncclCommInitRankConfig(void* comm, int nranks, void* cliqueId, int rank, void* config) {
    return ncclSuccess;
}

ncclResult_t ncclCommInitRankScalable(void* comm, int nranks, void* cliqueId, int rank, void* config) {
    return ncclSuccess;
}

ncclResult_t ncclCommSplit(void* parent, int color, int key, void** child, void* config) {
    /* Return a distinct sentinel so destroying child doesn't affect parent */
    if (child) *child = (void*)((unsigned long)parent + 0x1000);
    return ncclSuccess;
}

ncclResult_t ncclCommWindowRegister(void* comm, void* buff, size_t size, void** handle) {
    if (handle) *handle = NULL;
    return ncclSuccess;
}

ncclResult_t ncclCommWindowDeregister(void* comm, void* handle) {
    return ncclSuccess;
}

ncclResult_t ncclGroupSimulateEnd(void) {
    return ncclSuccess;
}

ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, void* comm, void* stream) {
    return ncclSuccess;
}

ncclResult_t ncclRedOpCreatePreMulSum(void* redOp, void* scalar, ncclDataType_t datatype, ncclRedOp_resident_t resident, void* comm) {
    return ncclSuccess;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t redOp, void* comm) {
    return ncclSuccess;
}

/* ── NCCL version info ── */

ncclResult_t ncclGetVersion(int* version) {
    if (version) *version = 21803;  /* NCCL 2.18.3 */
    return ncclSuccess;
}
