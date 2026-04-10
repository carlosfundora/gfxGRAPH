/**
 * @file hipgraph_bridge.h
 * @brief gfxGRAPH — CUDA Graph → HIP Graph translation layer for gfx1030
 *
 * Bridges 4 CUDA Graph parity gaps on AMD RDNA2:
 *   Gap 51: Conditional execution (per-branch exec dispatch)
 *   Gap 52: Rapid graph launch pipeline (Upload + event-tracked double-buffer)
 *   Gap 53: Dynamic input shapes (bucketing + param update)
 *   Gap 54: Capture composition (sequential + child graph nodes)
 */

#pragma once
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version & Lifecycle ────────────────────────────── */

#define HGB_VERSION_MAJOR 0
#define HGB_VERSION_MINOR 2
#define HGB_VERSION_PATCH 0

#define HGB_MAX_BRANCHES 32
#define HGB_MAX_BUCKETS  64

typedef struct {
    int major;
    int minor;
    int patch;
    const char* gfx_target;
    const char* rocm_version;
} hgb_version_t;

hgb_version_t hgb_get_version(void);

/**
 * Initialize the bridge. Call once before any other hgb_* function.
 * Validates gfx1030 target and caches device properties.
 */
hipError_t hgb_init(void);

/** Release all cached resources. Safe to call multiple times. */
void hgb_shutdown(void);

/** Check if bridge is initialized. Returns 1 if init'd, 0 otherwise. */
int hgb_is_initialized(void);

/* ── Gap 51: Conditional Execution ──────────────────── */

typedef struct {
    hipGraphExec_t*   branch_execs;
    int               num_branches;
    int               device_id;
} hgb_conditional_t;

/**
 * Create a conditional graph with per-branch exec dispatch.
 * Each branch graph is instantiated independently; launch selects one.
 *
 * @param graphs   Array of sub-graphs, one per branch
 * @param count    Number of branches (1..HGB_MAX_BRANCHES, typically 2 for if/else)
 * @param out      Populated conditional handle
 * @return hipSuccess or error code
 */
hipError_t hgb_conditional_create(
    hipGraph_t*         graphs,
    int                 count,
    hgb_conditional_t*  out
);

/**
 * Select a branch by index and launch on the given stream.
 */
hipError_t hgb_conditional_launch(
    hgb_conditional_t*  cond,
    int                 branch_index,
    hipStream_t         stream
);

/**
 * Select branch from device-side flag (requires host sync).
 * Reads *d_flag, uses as branch index.
 */
hipError_t hgb_conditional_launch_by_flag(
    hgb_conditional_t*  cond,
    int*                d_flag,
    hipStream_t         stream
);

void hgb_conditional_destroy(hgb_conditional_t* cond);

/* ── Gap 52: Rapid Graph Launch Pipeline ────────────── */

typedef enum {
    HGB_EXEC_IDLE      = 0,
    HGB_EXEC_UPLOADED   = 1,
    HGB_EXEC_IN_FLIGHT  = 2,
} hgb_exec_state_t;

typedef struct {
    hipGraphExec_t    exec[2];
    hipStream_t       stream;
    hipEvent_t        event;
    hgb_exec_state_t  state[2];
    int               active;
    int               launched;
    int               device_id;
} hgb_pipeline_t;

/**
 * Create a double-buffered graph launch pipeline.
 * Pre-uploads graph to GPU for ~44μs launch latency.
 */
hipError_t hgb_pipeline_create(
    hipGraph_t       graph,
    hgb_pipeline_t*  out
);

/** Launch active graph and swap buffers. */
hipError_t hgb_pipeline_launch(hgb_pipeline_t* pipe);

/** Update kernel params on inactive buffer (live on next launch). */
hipError_t hgb_pipeline_update_kernel(
    hgb_pipeline_t*       pipe,
    hipGraphNode_t        node,
    hipKernelNodeParams*  params
);

void hgb_pipeline_destroy(hgb_pipeline_t* pipe);

/* ── Gap 53: Dynamic Shape Management ───────────────── */

typedef hipError_t (*hgb_capture_fn)(int size, hipGraph_t* out, void* ctx);

typedef struct {
    int*             bucket_sizes;
    int              num_buckets;
    hipGraphExec_t*  execs;
    hipGraph_t*      graphs;
    void**           static_bufs;
    int              device_id;
} hgb_shape_pool_t;

/**
 * Create a shape-bucketed graph pool.
 *
 * @param graph_fn    Callback that captures a graph for a given size
 * @param ctx         User context passed to graph_fn
 * @param buckets     Sorted array of bucket sizes
 * @param num_buckets Length of buckets array
 * @param out         Populated shape pool
 */
hipError_t hgb_shape_pool_create(
    hgb_capture_fn    graph_fn,
    void*             ctx,
    const int*        buckets,
    int               num_buckets,
    hgb_shape_pool_t* out
);

/**
 * Launch graph for the given input size.
 * Selects smallest bucket >= size. Returns actual bucket used.
 */
hipError_t hgb_shape_pool_launch(
    hgb_shape_pool_t* pool,
    int               input_size,
    hipStream_t       stream,
    int*              actual_bucket
);

/** Update kernel params in-place for a specific bucket (no re-instantiate). */
hipError_t hgb_shape_pool_update_params(
    hgb_shape_pool_t*     pool,
    int                   bucket_index,
    hipGraphNode_t        node,
    hipKernelNodeParams*  params
);

void hgb_shape_pool_destroy(hgb_shape_pool_t* pool);

/* ── Gap 54: Capture Compositor ─────────────────────── */

typedef struct {
    hipGraph_t      composed;
    hipGraphExec_t  exec;
    hipGraphNode_t* child_nodes;
    int             num_children;
    int             device_id;
} hgb_composed_graph_t;

/**
 * Compose multiple sub-graphs into one via child graph nodes.
 *
 * @param sub_graphs  Array of pre-captured sub-graphs
 * @param count       Number of sub-graphs
 * @param deps        Dependency array: deps[i] = parent index, -1 = root
 * @param out         Populated composed graph
 */
hipError_t hgb_compose_graphs(
    hipGraph_t*           sub_graphs,
    int                   count,
    const int*            deps,
    hgb_composed_graph_t* out
);

/** Update a child sub-graph without re-instantiation. */
hipError_t hgb_compose_update_child(
    hgb_composed_graph_t* comp,
    int                   child_index,
    hipGraph_t            new_sub_graph
);

hipError_t hgb_compose_launch(
    hgb_composed_graph_t* comp,
    hipStream_t           stream
);

void hgb_compose_destroy(hgb_composed_graph_t* comp);

/* ── Utilities ──────────────────────────────────────── */

/** Check if running on gfx1030. Returns hipSuccess or hipErrorInvalidDevice. */
hipError_t hgb_validate_gfx1030(void);

/** Enable debug logging (also via env HGB_DEBUG=1). */
void hgb_set_debug(int enabled);

#ifdef __cplusplus
}
#endif
