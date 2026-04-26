/**
 * @file graph_utils.hip
 * @brief Shared utilities for gfxGRAPH bridge kernels
 */
#include "hipgraph_bridge.h"
#include <cstdlib>

extern "C" void hgb_log(const char* fmt, ...);

hipError_t hgb_find_kernel_nodes(
    hipGraph_t       graph,
    hipGraphNode_t** out_nodes,
    size_t*          out_count
) {
    /* Get total node count */
    size_t num_nodes = 0;
    hipError_t err = hipGraphGetNodes(graph, nullptr, &num_nodes);
    if (err != hipSuccess) return err;

    if (num_nodes == 0) {
        *out_nodes = nullptr;
        *out_count = 0;
        return hipSuccess;
    }

    /* Get all nodes */
    hipGraphNode_t* all = (hipGraphNode_t*)malloc(num_nodes * sizeof(hipGraphNode_t));
    if (!all) return hipErrorOutOfMemory;

    err = hipGraphGetNodes(graph, all, &num_nodes);
    if (err != hipSuccess) {
        free(all);
        return err;
    }

    /* Filter to kernel nodes */
    hipGraphNode_t* kernels = (hipGraphNode_t*)malloc(num_nodes * sizeof(hipGraphNode_t));
    if (!kernels) {
        free(all);
        return hipErrorOutOfMemory;
    }

    size_t kernel_count = 0;
    for (size_t i = 0; i < num_nodes; i++) {
        hipGraphNodeType type;
        hipGraphNodeGetType(all[i], &type);
        if (type == hipGraphNodeTypeKernel) {
            kernels[kernel_count++] = all[i];
        }
    }

    free(all);
    *out_nodes = kernels;
    *out_count = kernel_count;
    return hipSuccess;
}
