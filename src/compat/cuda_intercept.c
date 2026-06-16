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
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "hipgraph_bridge.h"

/* Real CUDA headers are not available on ROCm systems,
 * so we define compatible types matching the CUDA API signatures. */
typedef hipError_t cudaError_t;
typedef hipGraph_t cudaGraph_t;
typedef hipGraphExec_t cudaGraphExec_t;
typedef hipGraphNode_t cudaGraphNode_t;
typedef hipStream_t cudaStream_t;
typedef hipStreamCaptureMode cudaStreamCaptureMode;
typedef hipKernelNodeParams cudaKernelNodeParams;
typedef hipGraphInstantiateParams cudaGraphInstantiateParams;
typedef hipGraphNodeParams cudaGraphNodeParams;

#define HGB_COMPAT_MAX_GRAPHS 1024
#define HGB_COMPAT_MAX_CAPTURES 256

typedef struct {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    unsigned long long flags;
    cudaStream_t upload_stream;
    uint64_t instantiate_seq;
    uint64_t launch_count;
    int in_use;
} compat_graph_entry_t;

typedef struct {
    cudaStream_t stream;
    pthread_t owner;
    cudaStreamCaptureMode mode;
    int in_use;
} compat_capture_entry_t;

static int compat_debug = -1;
static pthread_mutex_t compat_registry_lock = PTHREAD_MUTEX_INITIALIZER;
static compat_graph_entry_t compat_graphs[HGB_COMPAT_MAX_GRAPHS];
static compat_capture_entry_t compat_captures[HGB_COMPAT_MAX_CAPTURES];
static uint64_t compat_next_instantiate_seq = 1;

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

static void compat_registry_register_exec(
    cudaGraph_t graph,
    cudaGraphExec_t exec,
    unsigned long long flags,
    cudaStream_t upload_stream
) {
    if (!exec) return;

    pthread_mutex_lock(&compat_registry_lock);
    int free_index = -1;
    for (int i = 0; i < HGB_COMPAT_MAX_GRAPHS; ++i) {
        if (compat_graphs[i].in_use && compat_graphs[i].exec == exec) {
            compat_graphs[i].graph = graph;
            compat_graphs[i].flags = flags;
            compat_graphs[i].upload_stream = upload_stream;
            pthread_mutex_unlock(&compat_registry_lock);
            return;
        }
        if (!compat_graphs[i].in_use && free_index < 0) {
            free_index = i;
        }
    }

    if (free_index >= 0) {
        compat_graphs[free_index].graph = graph;
        compat_graphs[free_index].exec = exec;
        compat_graphs[free_index].flags = flags;
        compat_graphs[free_index].upload_stream = upload_stream;
        compat_graphs[free_index].instantiate_seq = compat_next_instantiate_seq++;
        compat_graphs[free_index].launch_count = 0;
        compat_graphs[free_index].in_use = 1;
    } else {
        compat_log("registry full; exec %p is untracked", (void*)exec);
    }
    pthread_mutex_unlock(&compat_registry_lock);
}

static void compat_registry_unregister_exec(cudaGraphExec_t exec) {
    if (!exec) return;

    pthread_mutex_lock(&compat_registry_lock);
    for (int i = 0; i < HGB_COMPAT_MAX_GRAPHS; ++i) {
        if (compat_graphs[i].in_use && compat_graphs[i].exec == exec) {
            memset(&compat_graphs[i], 0, sizeof(compat_graphs[i]));
            break;
        }
    }
    pthread_mutex_unlock(&compat_registry_lock);
}

static void compat_registry_note_launch(cudaGraphExec_t exec) {
    if (!exec) return;

    pthread_mutex_lock(&compat_registry_lock);
    for (int i = 0; i < HGB_COMPAT_MAX_GRAPHS; ++i) {
        if (compat_graphs[i].in_use && compat_graphs[i].exec == exec) {
            compat_graphs[i].launch_count++;
            break;
        }
    }
    pthread_mutex_unlock(&compat_registry_lock);
}

static cudaError_t compat_capture_begin(
    cudaStream_t stream,
    cudaStreamCaptureMode mode
) {
    pthread_mutex_lock(&compat_registry_lock);
    int free_index = -1;
    for (int i = 0; i < HGB_COMPAT_MAX_CAPTURES; ++i) {
        if (compat_captures[i].in_use && compat_captures[i].stream == stream) {
            pthread_mutex_unlock(&compat_registry_lock);
            compat_log("capture overlap rejected for stream %p", (void*)stream);
            return (cudaError_t)hipErrorStreamCaptureMerge;
        }
        if (!compat_captures[i].in_use && free_index < 0) {
            free_index = i;
        }
    }
    if (free_index < 0) {
        pthread_mutex_unlock(&compat_registry_lock);
        compat_log("capture registry full");
        return (cudaError_t)hipErrorStreamCaptureUnsupported;
    }

    cudaError_t err = (cudaError_t)hipStreamBeginCapture((hipStream_t)stream, mode);
    if (err == hipSuccess) {
        compat_captures[free_index].stream = stream;
        compat_captures[free_index].owner = pthread_self();
        compat_captures[free_index].mode = mode;
        compat_captures[free_index].in_use = 1;
    }
    pthread_mutex_unlock(&compat_registry_lock);
    return err;
}

static cudaError_t compat_capture_end(cudaStream_t stream, cudaGraph_t* graph) {
    pthread_mutex_lock(&compat_registry_lock);
    int index = -1;
    for (int i = 0; i < HGB_COMPAT_MAX_CAPTURES; ++i) {
        if (compat_captures[i].in_use && compat_captures[i].stream == stream) {
            index = i;
            break;
        }
    }
    if (index < 0) {
        pthread_mutex_unlock(&compat_registry_lock);
        compat_log("capture end without matching begin for stream %p", (void*)stream);
        return (cudaError_t)hipErrorStreamCaptureUnmatched;
    }
    if (!pthread_equal(compat_captures[index].owner, pthread_self())) {
        pthread_mutex_unlock(&compat_registry_lock);
        compat_log("capture end from wrong thread for stream %p", (void*)stream);
        return (cudaError_t)hipErrorStreamCaptureWrongThread;
    }

    cudaError_t err = (cudaError_t)hipStreamEndCapture((hipStream_t)stream, (hipGraph_t*)graph);
    if (err == hipSuccess || err == hipErrorStreamCaptureInvalidated ||
        err == hipErrorStreamCaptureUnmatched) {
        memset(&compat_captures[index], 0, sizeof(compat_captures[index]));
    }
    pthread_mutex_unlock(&compat_registry_lock);
    return err;
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
    cudaError_t err = (cudaError_t)hipGraphInstantiate(
        (hipGraphExec_t*)exec, (hipGraph_t)graph,
        (hipGraphNode_t*)errNode, logBuf, logLen
    );
    if (err == hipSuccess && exec) {
        compat_registry_register_exec(graph, *exec, 0, NULL);
    }
    return err;
}

cudaError_t cudaGraphInstantiateWithFlags(
    cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags
) {
    compat_log("cudaGraphInstantiateWithFlags → hipGraphInstantiateWithFlags flags=%llu", flags);
    cudaError_t err = (cudaError_t)hipGraphInstantiateWithFlags(
        (hipGraphExec_t*)exec, (hipGraph_t)graph, flags
    );
    if (err == hipSuccess && exec) {
        compat_registry_register_exec(graph, *exec, flags, NULL);
    }
    return err;
}

cudaError_t cudaGraphInstantiateWithParams(
    cudaGraphExec_t* exec, cudaGraph_t graph, cudaGraphInstantiateParams* params
) {
    compat_log("cudaGraphInstantiateWithParams → hipGraphInstantiateWithParams");
    cudaError_t err = (cudaError_t)hipGraphInstantiateWithParams(
        (hipGraphExec_t*)exec, (hipGraph_t)graph, (hipGraphInstantiateParams*)params
    );
    if (err == hipSuccess && exec) {
        unsigned long long flags = params ? params->flags : 0;
        cudaStream_t upload_stream = params ? (cudaStream_t)params->uploadStream : NULL;
        compat_registry_register_exec(graph, *exec, flags, upload_stream);
    }
    return err;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    compat_log("cudaGraphLaunch → hipGraphLaunch");
    cudaError_t err = (cudaError_t)hipGraphLaunch((hipGraphExec_t)exec, (hipStream_t)stream);
    if (err == hipSuccess) {
        compat_registry_note_launch(exec);
    }
    return err;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    compat_log("cudaGraphExecDestroy → hipGraphExecDestroy");
    cudaError_t err = (cudaError_t)hipGraphExecDestroy((hipGraphExec_t)exec);
    if (err == hipSuccess) {
        compat_registry_unregister_exec(exec);
    }
    return err;
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

cudaError_t cudaGraphAddChildGraphNode(
    cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies, size_t numDependencies,
    cudaGraph_t childGraph
) {
    compat_log("cudaGraphAddChildGraphNode → hipGraphAddChildGraphNode");
    return (cudaError_t)hipGraphAddChildGraphNode(
        (hipGraphNode_t*)pGraphNode, (hipGraph_t)graph,
        (const hipGraphNode_t*)pDependencies, numDependencies,
        (hipGraph_t)childGraph
    );
}

cudaError_t cudaGraphExecNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
    cudaGraphNodeParams* nodeParams
) {
    compat_log("cudaGraphExecNodeSetParams → hipGraphExecNodeSetParams");
    return (cudaError_t)hipGraphExecNodeSetParams(
        (hipGraphExec_t)hGraphExec, (hipGraphNode_t)node,
        (hipGraphNodeParams*)nodeParams
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

cudaError_t cudaGraphExecChildGraphNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
    cudaGraph_t childGraph
) {
    compat_log("cudaGraphExecChildGraphNodeSetParams → hipGraphExecChildGraphNodeSetParams");
    return (cudaError_t)hipGraphExecChildGraphNodeSetParams(
        (hipGraphExec_t)hGraphExec, (hipGraphNode_t)node, (hipGraph_t)childGraph
    );
}

cudaError_t cudaGraphNodeSetEnabled(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t node, unsigned int isEnabled
) {
    compat_log("cudaGraphNodeSetEnabled → hipGraphNodeSetEnabled");
    return (cudaError_t)hipGraphNodeSetEnabled(
        (hipGraphExec_t)hGraphExec, (hipGraphNode_t)node, isEnabled
    );
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    compat_log("cudaStreamBeginCapture → hipStreamBeginCapture");
    return compat_capture_begin(stream, mode);
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    compat_log("cudaStreamEndCapture → hipStreamEndCapture");
    return compat_capture_end(stream, pGraph);
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
