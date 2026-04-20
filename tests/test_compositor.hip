/**
 * @file test_compositor.hip
 * @brief Tests for Gap 54: Capture composition
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>

__global__ void busy_wait_kernel(unsigned long long cycles) {
    unsigned long long start = clock64();
    while ((clock64() - start) < cycles) {
    }
}

__global__ void add_kernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += val;
}

static hipGraph_t capture_add(
    float* d_data,
    float  val,
    int    n,
    unsigned long long delay_cycles = 0
) {
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipGraph_t graph;
    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    if (delay_cycles > 0) {
        busy_wait_kernel<<<1, 1, 0, stream>>>(delay_cycles);
    }
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
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    unsigned long long delay_cycles =
        props.clockRate > 0 ? static_cast<unsigned long long>(props.clockRate) * 25ULL
                            : 25000000ULL;

    float* d_data;
    hipMalloc(&d_data, N * sizeof(float));
    hipMemset(d_data, 0, N * sizeof(float));

    /* Create 3 sub-graphs: +1.0, +2.0, +3.0 */
    hipGraph_t subs[3];
    subs[0] = capture_add(d_data, 1.0f, N, delay_cycles);
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
    hipGraph_t new_sub_10 = capture_add(d_data, 10.0f, N);
    err = hgb_compose_update_child(&comp, 2, new_sub_10);
    assert(err == hipSuccess);
    printf("  Updated child 2: +3.0 → +10.0\n");

    hipMemset(d_data, 0, N * sizeof(float));
    err = hgb_compose_launch(&comp, stream);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    hipMemcpy(h_data, d_data, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Result: [%.1f] (expected 13.0 = 1+2+10)\n", h_data[0]);
    assert(h_data[0] == 13.0f);

    /* Test in-flight child update: launch old graph, then update while stream is busy */
    hipGraph_t new_sub_20 = capture_add(d_data, 20.0f, N);
    hipMemset(d_data, 0, N * sizeof(float));

    err = hgb_compose_launch(&comp, stream);
    assert(err == hipSuccess);

    err = hgb_compose_update_child(&comp, 2, new_sub_20);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    hipMemcpy(h_data, d_data, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  In-flight update preserved current launch result: [%.1f] (expected 13.0)\n",
           h_data[0]);
    assert(h_data[0] == 13.0f);

    hipMemset(d_data, 0, N * sizeof(float));
    err = hgb_compose_launch(&comp, stream);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    hipMemcpy(h_data, d_data, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Next launch sees updated child: [%.1f] (expected 23.0 = 1+2+20)\n",
           h_data[0]);
    assert(h_data[0] == 23.0f);

    /* Cleanup */
    hgb_compose_destroy(&comp);
    for (int i = 0; i < 3; i++) hipGraphDestroy(subs[i]);
    hipGraphDestroy(new_sub_10);
    hipGraphDestroy(new_sub_20);
    hipFree(d_data);
    hipStreamDestroy(stream);
    hgb_shutdown();

    printf("  PASSED\n");
    return 0;
}
