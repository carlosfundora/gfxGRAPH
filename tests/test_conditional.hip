/**
 * @file test_conditional.hip
 * @brief Tests for Gap 51: Conditional execution bridge
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>

__global__ void fill_kernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

/* Helper: capture a simple fill graph */
static hipError_t capture_fill(float* d_out, float val, int n, hipGraph_t* graph) {
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    fill_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_out, val, n);
    hipStreamEndCapture(stream, graph);

    hipStreamDestroy(stream);
    return hipSuccess;
}

int main() {
    printf("=== test_conditional ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    const int N = 1024;
    float* d_out;
    hipMalloc(&d_out, N * sizeof(float));

    /* Create two branch graphs: fill with 1.0 or 2.0 */
    hipGraph_t graphs[2];
    capture_fill(d_out, 1.0f, N, &graphs[0]);
    capture_fill(d_out, 2.0f, N, &graphs[1]);

    /* Build conditional */
    hgb_conditional_t cond;
    err = hgb_conditional_create(graphs, 2, &cond);
    assert(err == hipSuccess);
    printf("  Created conditional with %d branches\n", cond.num_branches);

    hipStream_t stream;
    hipStreamCreate(&stream);

    /* Test branch 0 */
    err = hgb_conditional_launch(&cond, 0, stream);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    float h_out[4];
    hipMemcpy(h_out, d_out, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Branch 0: [%.1f, %.1f, %.1f, %.1f]\n",
           h_out[0], h_out[1], h_out[2], h_out[3]);
    assert(h_out[0] == 1.0f);

    /* Test branch 1 */
    err = hgb_conditional_launch(&cond, 1, stream);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    hipMemcpy(h_out, d_out, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Branch 1: [%.1f, %.1f, %.1f, %.1f]\n",
           h_out[0], h_out[1], h_out[2], h_out[3]);
    assert(h_out[0] == 2.0f);

    /* Test device flag dispatch */
    int* d_flag;
    hipMalloc(&d_flag, sizeof(int));
    int flag = 0;
    hipMemcpy(d_flag, &flag, sizeof(int), hipMemcpyHostToDevice);

    err = hgb_conditional_launch_by_flag(&cond, d_flag, stream);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    hipMemcpy(h_out, d_out, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Flag=0: [%.1f] (expected 1.0)\n", h_out[0]);
    assert(h_out[0] == 1.0f);

    /* Cleanup */
    hgb_conditional_destroy(&cond);
    hipGraphDestroy(graphs[0]);
    hipGraphDestroy(graphs[1]);
    hipFree(d_out);
    hipFree(d_flag);
    hipStreamDestroy(stream);
    hgb_shutdown();

    printf("  PASSED\n");
    return 0;
}
