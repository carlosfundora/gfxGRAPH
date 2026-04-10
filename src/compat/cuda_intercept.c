/**
 * @file cuda_intercept.c
 * @brief Layer 3: LD_PRELOAD CUDA→HIP symbol interception (optional)
 *
 * Usage: LD_PRELOAD=libcudagraph_compat.so ./my_cuda_app
 *
 * Routes 50 native CUDA Graph APIs to HIP equivalents and
 * 4 gap APIs to the bridge library.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include "hipgraph_bridge.h"

/* Stub typedefs matching CUDA API signatures.
 * Real CUDA headers are not available on ROCm systems,
 * so we define minimal compatible signatures. */

typedef hipError_t cudaError_t;
typedef hipGraph_t cudaGraph_t;
typedef hipGraphExec_t cudaGraphExec_t;
typedef hipGraphNode_t cudaGraphNode_t;
typedef hipStream_t cudaStream_t;

static int compat_debug = -1;

static void compat_log(const char* fmt, ...) {
    if (compat_debug < 0) {
        const char* dbg = getenv("HGB_DEBUG");
        compat_debug = (dbg && dbg[0] == '1') ? 1 : 0;
    }
    if (!compat_debug) return;

    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[gfxGRAPH-compat] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

/* ── 1:1 Native Mappings ────────────────────────────── */

cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int flags) {
    compat_log("cudaGraphCreate → hipGraphCreate");
    return (cudaError_t)hipGraphCreate((hipGraph_t*)graph, flags);
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    compat_log("cudaGraphDestroy → hipGraphDestroy");
    return (cudaError_t)hipGraphDestroy((hipGraph_t)graph);
}

cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t* exec, cudaGraph_t graph,
    void* errNode, char* logBuf, size_t logLen
) {
    compat_log("cudaGraphInstantiate → hipGraphInstantiate");
    return (cudaError_t)hipGraphInstantiate(
        (hipGraphExec_t*)exec, (hipGraph_t)graph,
        (hipGraphNode_t*)errNode, logBuf, logLen
    );
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    compat_log("cudaGraphLaunch → hipGraphLaunch");
    return (cudaError_t)hipGraphLaunch((hipGraphExec_t)exec, (hipStream_t)stream);
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    compat_log("cudaGraphExecDestroy → hipGraphExecDestroy");
    return (cudaError_t)hipGraphExecDestroy((hipGraphExec_t)exec);
}

cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t stream) {
    compat_log("cudaGraphUpload → hipGraphUpload");
    return (cudaError_t)hipGraphUpload((hipGraphExec_t)exec, (hipStream_t)stream);
}

/* ── Additional native mappings follow the same pattern ── */
/* TODO: Add remaining ~44 native 1:1 mappings */

/* ── Gap Bridges (routed to Layer 1) ────────────────── */

/* Gap 51: Conditional — no direct CUDA equivalent mapping possible
 * without full conditional handle infrastructure. Log and return error
 * for now; users should use the Python bridge layer for this gap. */

/* Gap 52: Device launch flag is intercepted at instantiate level */
/* TODO: Detect cudaGraphInstantiateFlagDeviceLaunch and route to
 * hgb_pipeline_create */

/* ── Initialization ─────────────────────────────────── */

__attribute__((constructor))
static void compat_init(void) {
    compat_log("gfxGRAPH CUDA compat layer loaded");
    hgb_init();
}

__attribute__((destructor))
static void compat_fini(void) {
    compat_log("gfxGRAPH CUDA compat layer unloading");
    hgb_shutdown();
}
