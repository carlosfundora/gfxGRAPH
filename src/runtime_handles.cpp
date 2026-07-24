/**
 * @file runtime_handles.cpp
 * @brief Opaque C ABI handles for native graph execution.
 */
#include "hipgraph_bridge.h"

#include <cstdlib>
#include <cstring>

struct hgb_pipeline_handle {
    hgb_pipeline_t pipeline;
};

struct hgb_composed_graph_handle {
    hgb_composed_graph_t composed;
};

struct hgb_decode_pool_handle {
    hgb_decode_pool_t pool;
};

extern "C" HGB_EXPORT hipError_t hgb_pipeline_handle_create(
    hipGraph_t              graph,
    hgb_pipeline_handle_t** out
) {
    if (!out) return hipErrorInvalidValue;
    *out = nullptr;

    auto* handle = static_cast<hgb_pipeline_handle_t*>(
        std::calloc(1, sizeof(hgb_pipeline_handle_t))
    );
    if (!handle) return hipErrorOutOfMemory;

    hipError_t err = hgb_pipeline_create(graph, &handle->pipeline);
    if (err != hipSuccess) {
        std::free(handle);
        return err;
    }

    *out = handle;
    return hipSuccess;
}

extern "C" HGB_EXPORT hipError_t hgb_pipeline_handle_launch(
    hgb_pipeline_handle_t* handle
) {
    if (!handle) return hipErrorInvalidValue;
    return hgb_pipeline_launch(&handle->pipeline);
}

extern "C" HGB_EXPORT hipError_t hgb_pipeline_handle_update_kernel(
    hgb_pipeline_handle_t* handle,
    hipGraphNode_t         node,
    hipKernelNodeParams*   params
) {
    if (!handle) return hipErrorInvalidValue;
    return hgb_pipeline_update_kernel(&handle->pipeline, node, params);
}

extern "C" HGB_EXPORT hipError_t hgb_pipeline_handle_update_and_launch(
    hgb_pipeline_handle_t* handle,
    hipGraphNode_t         node,
    hipKernelNodeParams*   params
) {
    if (!handle) return hipErrorInvalidValue;
    return hgb_pipeline_update_and_launch(&handle->pipeline, node, params);
}

extern "C" HGB_EXPORT void hgb_pipeline_handle_destroy(
    hgb_pipeline_handle_t* handle
) {
    if (!handle) return;
    hgb_pipeline_destroy(&handle->pipeline);
    std::free(handle);
}

extern "C" HGB_EXPORT hipError_t hgb_composed_handle_create(
    hipGraph_t*                   sub_graphs,
    int                           count,
    const int*                    deps,
    hgb_composed_graph_handle_t** out
) {
    if (!out) return hipErrorInvalidValue;
    *out = nullptr;

    auto* handle = static_cast<hgb_composed_graph_handle_t*>(
        std::calloc(1, sizeof(hgb_composed_graph_handle_t))
    );
    if (!handle) return hipErrorOutOfMemory;

    hipError_t err = hgb_compose_graphs(sub_graphs, count, deps, &handle->composed);
    if (err != hipSuccess) {
        std::free(handle);
        return err;
    }

    *out = handle;
    return hipSuccess;
}

extern "C" HGB_EXPORT hipError_t hgb_composed_handle_launch(
    hgb_composed_graph_handle_t* handle,
    hipStream_t                  stream
) {
    if (!handle) return hipErrorInvalidValue;
    return hgb_compose_launch(&handle->composed, stream);
}

extern "C" HGB_EXPORT hipError_t hgb_composed_handle_update_child(
    hgb_composed_graph_handle_t* handle,
    int                          child_index,
    hipGraph_t                   new_sub_graph
) {
    if (!handle) return hipErrorInvalidValue;
    return hgb_compose_update_child(&handle->composed, child_index, new_sub_graph);
}

extern "C" HGB_EXPORT void hgb_composed_handle_destroy(
    hgb_composed_graph_handle_t* handle
) {
    if (!handle) return;
    hgb_compose_destroy(&handle->composed);
    std::free(handle);
}

/* ── Capture-safe decode pool (opaque handle) ─────────── */

extern "C" HGB_EXPORT hipError_t hgb_decode_pool_handle_create(
    hgb_decode_capture_fn      fn,
    void*                      ctx,
    const int*                 buckets,
    int                        num_buckets,
    int                        max_num_seqs,
    int                        max_blocks_per_seq,
    hgb_decode_pool_handle_t** out
) {
    if (!out) return hipErrorInvalidValue;
    *out = nullptr;

    auto* handle = static_cast<hgb_decode_pool_handle_t*>(
        std::calloc(1, sizeof(hgb_decode_pool_handle_t))
    );
    if (!handle) return hipErrorOutOfMemory;

    hipError_t err = hgb_decode_pool_create(
        fn, ctx, buckets, num_buckets, max_num_seqs, max_blocks_per_seq, &handle->pool
    );
    if (err != hipSuccess) {
        std::free(handle);
        return err;
    }

    *out = handle;
    return hipSuccess;
}

extern "C" HGB_EXPORT hipError_t hgb_decode_pool_handle_replay(
    hgb_decode_pool_handle_t* handle,
    int                       input_size,
    const int*                h_seq_lens,
    int                       num_seqs,
    const int*                h_block_tables,
    hipStream_t               stream,
    int*                      actual_bucket
) {
    if (!handle) return hipErrorInvalidValue;
    return hgb_decode_pool_replay(
        &handle->pool, input_size, h_seq_lens, num_seqs, h_block_tables, stream, actual_bucket
    );
}

extern "C" HGB_EXPORT void hgb_decode_pool_handle_destroy(
    hgb_decode_pool_handle_t* handle
) {
    if (!handle) return;
    hgb_decode_pool_destroy(&handle->pool);
    std::free(handle);
}
