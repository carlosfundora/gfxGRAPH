/**
 * @file test_shapes.hip
 * @brief Tests for Gap 53: Dynamic shape management
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>

__global__ void scale_kernel(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= factor;
}

static float* g_data;

static hipError_t capture_scale(int size, hipGraph_t* graph, void* ctx) {
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    scale_kernel<<<(size + 255) / 256, 256, 0, stream>>>(g_data, 2.0f, size);
    hipStreamEndCapture(stream, graph);

    hipStreamDestroy(stream);
    return hipSuccess;
}

int main() {
    printf("=== test_shapes ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    const int MAX_SIZE = 64;
    hipMalloc(&g_data, MAX_SIZE * sizeof(float));

    /* Create shape pool with buckets */
    int buckets[] = {1, 4, 8, 16, 32, 64};
    hgb_shape_pool_t pool;
    err = hgb_shape_pool_create(capture_scale, nullptr, buckets, 6, &pool);
    assert(err == hipSuccess);
    printf("  Created pool with %d buckets\n", pool.num_buckets);

    hipStream_t stream;
    hipStreamCreate(&stream);

    /* Test: input size 5 → should pick bucket 8 */
    float h_init[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    hipMemcpy(g_data, h_init, 8 * sizeof(float), hipMemcpyHostToDevice);

    int actual = 0;
    err = hgb_shape_pool_launch(&pool, 5, stream, &actual);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    printf("  Input size 5 → bucket %d\n", actual);
    assert(actual == 8);

    float h_out[8];
    hipMemcpy(h_out, g_data, 8 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Output: [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
           h_out[0], h_out[1], h_out[2], h_out[3], h_out[4]);
    assert(h_out[0] == 2.0f);  /* First 5 elements scaled */

    /* Test: input size 1 → bucket 1 */
    hipMemcpy(g_data, h_init, sizeof(float), hipMemcpyHostToDevice);
    err = hgb_shape_pool_launch(&pool, 1, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 1);
    printf("  Input size 1 → bucket %d ✓\n", actual);

    /* Test: input size 64 → bucket 64 (exact) */
    err = hgb_shape_pool_launch(&pool, 64, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 64);
    printf("  Input size 64 → bucket %d ✓\n", actual);

    /* Test: input too large → error */
    err = hgb_shape_pool_launch(&pool, 100, stream, &actual);
    assert(err != hipSuccess);
    printf("  Input size 100 → rejected ✓\n");

    /* Cleanup */
    hgb_shape_pool_destroy(&pool);
    hipFree(g_data);
    hipStreamDestroy(stream);
    hgb_shutdown();

    printf("  PASSED\n");
    return 0;
}
