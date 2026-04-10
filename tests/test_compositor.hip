/**
 * @file test_compositor.hip
 * @brief Tests for Gap 54: Capture composition
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>

__global__ void add_kernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += val;
}

static hipGraph_t capture_add(float* d_data, float val, int n) {
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipGraph_t graph;
    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_data, val, n);
    hipStreamEndCapture(stream, &graph);

    hipStreamDestroy(stream);
    return graph;
}

int main() {
    printf("=== test_compositor ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    const int N = 256;
    float* d_data;
    hipMalloc(&d_data, N * sizeof(float));
    hipMemset(d_data, 0, N * sizeof(float));

    /* Create 3 sub-graphs: +1.0, +2.0, +3.0 */
    hipGraph_t subs[3];
    subs[0] = capture_add(d_data, 1.0f, N);
    subs[1] = capture_add(d_data, 2.0f, N);
    subs[2] = capture_add(d_data, 3.0f, N);

    /* Compose: linear chain 0 → 1 → 2 */
    int deps[] = {-1, 0, 1};
    hgb_composed_graph_t comp;
    err = hgb_compose_graphs(subs, 3, deps, &comp);
    assert(err == hipSuccess);
    printf("  Composed 3 sub-graphs in chain\n");

    /* Launch */
    hipStream_t stream;
    hipStreamCreate(&stream);

    err = hgb_compose_launch(&comp, stream);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    float h_data[4];
    hipMemcpy(h_data, d_data, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Result: [%.1f, %.1f, %.1f, %.1f] (expected 6.0)\n",
           h_data[0], h_data[1], h_data[2], h_data[3]);
    assert(h_data[0] == 6.0f);  /* 1 + 2 + 3 = 6 */

    /* Test child update: replace sub[2] (+3.0) with +10.0 */
    hipGraph_t new_sub = capture_add(d_data, 10.0f, N);
    err = hgb_compose_update_child(&comp, 2, new_sub);
    assert(err == hipSuccess);
    printf("  Updated child 2: +3.0 → +10.0\n");

    hipMemset(d_data, 0, N * sizeof(float));
    err = hgb_compose_launch(&comp, stream);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    hipMemcpy(h_data, d_data, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Result: [%.1f] (expected 13.0 = 1+2+10)\n", h_data[0]);
    assert(h_data[0] == 13.0f);

    /* Cleanup */
    hgb_compose_destroy(&comp);
    for (int i = 0; i < 3; i++) hipGraphDestroy(subs[i]);
    hipGraphDestroy(new_sub);
    hipFree(d_data);
    hipStreamDestroy(stream);
    hgb_shutdown();

    printf("  PASSED\n");
    return 0;
}
