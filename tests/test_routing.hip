/**
 * @file test_routing.hip
 * @brief Tests for shape bucket routing logic
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstdlib>

__global__ void dummy_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] + 1.0f;
}

static float* g_data;

static hipError_t capture_dummy(int size, hipGraph_t* graph, void* ctx) {
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    dummy_kernel<<<(size + 255) / 256, 256, 0, stream>>>(g_data, size);
    hipStreamEndCapture(stream, graph);

    hipStreamDestroy(stream);
    return hipSuccess;
}

int main() {
    printf("=== test_routing ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    const int MAX_SIZE = 128;
    hipMalloc(&g_data, MAX_SIZE * sizeof(float));

    /* Create shape pool with buckets */
    int buckets[] = {1, 4, 8, 16, 32, 64};
    hgb_shape_pool_t pool;
    err = hgb_shape_pool_create(capture_dummy, nullptr, buckets, 6, &pool);
    assert(err == hipSuccess);
    printf("  Created pool with %d buckets\n", pool.num_buckets);

    hipStream_t stream;
    hipStreamCreate(&stream);

    int actual = 0;

    /* Test exact matches */
    err = hgb_shape_pool_launch(&pool, 1, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 1);
    printf("  Input size 1 → bucket %d ✓\n", actual);

    err = hgb_shape_pool_launch(&pool, 4, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 4);
    printf("  Input size 4 → bucket %d ✓\n", actual);

    err = hgb_shape_pool_launch(&pool, 16, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 16);
    printf("  Input size 16 → bucket %d ✓\n", actual);

    err = hgb_shape_pool_launch(&pool, 64, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 64);
    printf("  Input size 64 → bucket %d ✓\n", actual);

    /* Test intermediate sizes requiring rounding up */
    err = hgb_shape_pool_launch(&pool, 2, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 4);
    printf("  Input size 2 → bucket %d ✓\n", actual);

    err = hgb_shape_pool_launch(&pool, 5, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 8);
    printf("  Input size 5 → bucket %d ✓\n", actual);

    err = hgb_shape_pool_launch(&pool, 12, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 16);
    printf("  Input size 12 → bucket %d ✓\n", actual);

    err = hgb_shape_pool_launch(&pool, 40, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 64);
    printf("  Input size 40 → bucket %d ✓\n", actual);

    /* Test invalid size bounds */
    err = hgb_shape_pool_launch(&pool, 0, stream, &actual);
    assert(err != hipSuccess);
    printf("  Input size 0 → rejected ✓\n");

    err = hgb_shape_pool_launch(&pool, -5, stream, &actual);
    assert(err != hipSuccess);
    printf("  Input size -5 → rejected ✓\n");

    err = hgb_shape_pool_launch(&pool, 65, stream, &actual);
    assert(err != hipSuccess);
    printf("  Input size 65 → rejected ✓\n");

    /* Cleanup */
    hgb_shape_pool_destroy(&pool);
    hipFree(g_data);
    hipStreamDestroy(stream);
    hgb_shutdown();

    printf("  PASSED\n");
    return 0;
}
