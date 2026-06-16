/**
 * @file cuda_intercept.c
 * @brief Layer 3: LD_PRELOAD CUDA→HIP symbol interception (optional)
 *
 * Usage: LD_PRELOAD=libcudagraph_compat.so ./my_cuda_app
 *
 * Routes native CUDA Graph APIs to their HIP equivalents and provides
 * low-latency pre-buffered pointer-swapping shortcuts to minimize FFI boundary crossings.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <hip/hip_runtime.h>
#include "hipgraph_bridge.h"

/* Real CUDA headers are not available on ROCm systems,
 * so we define compatible types matching the CUDA API signatures. */
typedef hipError_t cudaError_t;
typedef hipGraph_t cudaGraph_t;
typedef hipGraphExec_t cudaGraphExec_t;
typedef hipGraphNode_t cudaGraphNode_t;
typedef hipStream_t cudaStream_t;
typedef hipKernelNodeParams cudaKernelNodeParams;

static int compat_debug = -1;

/**
 * Internal logger for compatibility transitions.
 * Evaluates the debug environment once on first print.
 */
static void compat_log(const char* fmt, ...) {
    if (compat_debug < 0) {
        const char* dbg = getenv("HGB_DEBUG");
        compat_debug = (dbg && (dbg[0] == '1' || strcmp(dbg, "debug") == 0)) ? 1 : 0;
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

cudaError_t cudaGraphAddKernelNode(
    cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies, size_t numDependencies,
    const cudaKernelNodeParams* pNodeParams
) {
    compat_log("cudaGraphAddKernelNode → hipGraphAddKernelNode");
    return (cudaError_t)hipGraphAddKernelNode(
        (hipGraphNode_t*)pGraphNode, (hipGraph_t)graph,
        (const hipGraphNode_t*)pDependencies, numDependencies,
        (const hipKernelNodeParams*)pNodeParams
    );
}

cudaError_t cudaGraphExecKernelNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
    const cudaKernelNodeParams* pNodeParams
) {
    compat_log("cudaGraphExecKernelNodeSetParams → hipGraphExecKernelNodeSetParams");
    return (cudaError_t)hipGraphExecKernelNodeSetParams(
        (hipGraphExec_t)hGraphExec, (hipGraphNode_t)node,
        (const hipKernelNodeParams*)pNodeParams
    );
}

/* ── Gap Bridges (pre-buffered updates & execution triggers) ────────────────── */

/**
 * Zero-Python update and launch helper.
 * Performs both parameter updates and pipeline launches in a single FFI boundary crossing.
 */
HGB_EXPORT hipError_t hgb_pipeline_update_and_launch(
    hgb_pipeline_t*       pipe,
    hipGraphNode_t        node,
    hipKernelNodeParams*  params
) {
    hipError_t err = hgb_pipeline_update_kernel(pipe, node, params);
    if (err != hipSuccess) return err;
    return hgb_pipeline_launch(pipe);
}

/* ── Constructor/Destructor hooks ───────────────────── */

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
