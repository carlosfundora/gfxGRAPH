/**
 * @file test_pipeline.hip
 * @brief Tests for Gap 52: Rapid graph launch pipeline
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include <chrono>

__global__ void inc_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

int main() {
    printf("=== test_pipeline ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    const int N = 1024;
    float* d_data;
    hipMalloc(&d_data, N * sizeof(float));
    hipMemset(d_data, 0, N * sizeof(float));

    /* Capture a simple increment graph */
    hipStream_t cap_stream;
    hipStreamCreate(&cap_stream);

    hipGraph_t graph;
    hipStreamBeginCapture(cap_stream, hipStreamCaptureModeGlobal);
    inc_kernel<<<(N + 255) / 256, 256, 0, cap_stream>>>(d_data, N);
    hipStreamEndCapture(cap_stream, &graph);
    hipStreamDestroy(cap_stream);

    /* Create pipeline */
    hgb_pipeline_t pipe;
    err = hgb_pipeline_create(graph, &pipe);
    assert(err == hipSuccess);
    printf("  Pipeline created, active=%d, launched=%d\n", pipe.active, pipe.launched);

    /* Launch 10 times */
    for (int i = 0; i < 10; i++) {
        err = hgb_pipeline_launch(&pipe);
        assert(err == hipSuccess);
    }
    hipDeviceSynchronize();

    float h_data[4];
    hipMemcpy(h_data, d_data, 4 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  After 10 launches: [%.1f, %.1f, %.1f, %.1f] (expected 10.0)\n",
           h_data[0], h_data[1], h_data[2], h_data[3]);
    assert(h_data[0] == 10.0f);

    /* Latency benchmark */
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    const int ITERS = 1000;
    for (int i = 0; i < ITERS; i++) {
        hgb_pipeline_launch(&pipe);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double per_launch = us / ITERS;
    printf("  Latency: %.1f μs/launch (%d iterations)\n", per_launch, ITERS);

    /* Cleanup */
    hgb_pipeline_destroy(&pipe);
    hipGraphDestroy(graph);
    hipFree(d_data);
    hgb_shutdown();

    printf("  PASSED\n");
    return 0;
}
