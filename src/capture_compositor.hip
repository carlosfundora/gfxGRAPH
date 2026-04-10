/**
 * @file capture_compositor.hip
 * @brief Gap 54: Compose sub-graphs via sequential capture + child nodes
 *
 * Strategy: Capture sub-graphs independently, then assemble them as
 * child graph nodes in a parent graph with dependency edges.
 * ~95% of nested capture performance.
 */
#include "hipgraph_bridge.h"
#include <cstdlib>
#include <cstring>

extern void hgb_log(const char* fmt, ...);

hipError_t hgb_compose_graphs(
    hipGraph_t*           sub_graphs,
    int                   count,
    const int*            deps,
    hgb_composed_graph_t* out
) {
    if (!sub_graphs || count <= 0 || !deps || !out)
        return hipErrorInvalidValue;

    memset(out, 0, sizeof(*out));
    out->num_children = count;

    out->child_nodes = (hipGraphNode_t*)calloc(count, sizeof(hipGraphNode_t));
    if (!out->child_nodes) return hipErrorOutOfMemory;

    /* Create parent graph */
    hipError_t err = hipGraphCreate(&out->composed, 0);
    if (err != hipSuccess) {
        hgb_compose_destroy(out);
        return err;
    }

    /* Add each sub-graph as a child node */
    for (int i = 0; i < count; i++) {
        hipGraphNode_t* dep_list = nullptr;
        size_t num_deps = 0;

        /* If this node depends on a prior node, set dependency */
        if (deps[i] >= 0 && deps[i] < i) {
            dep_list = &out->child_nodes[deps[i]];
            num_deps = 1;
        }

        err = hipGraphAddChildGraphNode(
            &out->child_nodes[i],
            out->composed,
            dep_list, num_deps,
            sub_graphs[i]
        );
        if (err != hipSuccess) {
            hgb_log("Failed to add child %d: %d", i, (int)err);
            hgb_compose_destroy(out);
            return err;
        }

        hgb_log("Added child %d (dep=%d)", i, deps[i]);
    }

    /* Instantiate the composed graph */
    err = hipGraphInstantiate(&out->exec, out->composed, nullptr, nullptr, 0);
    if (err != hipSuccess) {
        hgb_log("Failed to instantiate composed graph: %d", (int)err);
        hgb_compose_destroy(out);
        return err;
    }

    hgb_log("Composed graph: %d children", count);
    return hipSuccess;
}

hipError_t hgb_compose_update_child(
    hgb_composed_graph_t* comp,
    int                   child_index,
    hipGraph_t            new_sub_graph
) {
    if (!comp || !comp->exec) return hipErrorInvalidValue;
    if (child_index < 0 || child_index >= comp->num_children)
        return hipErrorInvalidValue;

    return hipGraphExecChildGraphNodeSetParams(
        comp->exec,
        comp->child_nodes[child_index],
        new_sub_graph
    );
}

hipError_t hgb_compose_launch(
    hgb_composed_graph_t* comp,
    hipStream_t           stream
) {
    if (!comp || !comp->exec) return hipErrorInvalidValue;
    return hipGraphLaunch(comp->exec, stream);
}

void hgb_compose_destroy(hgb_composed_graph_t* comp) {
    if (!comp) return;
    if (comp->exec) {
        hipGraphExecDestroy(comp->exec);
        comp->exec = nullptr;
    }
    if (comp->composed) {
        hipGraphDestroy(comp->composed);
        comp->composed = nullptr;
    }
    free(comp->child_nodes);
    comp->child_nodes = nullptr;
    comp->num_children = 0;
}
