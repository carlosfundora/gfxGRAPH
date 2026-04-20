/**
 * @file test_shapes.hip
 * @brief Tests for Gap 53: Dynamic shape management
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstdlib>

__global__ void scale_kernel(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= factor;
}

__global__ void busy_wait_kernel(unsigned long long cycles) {
    unsigned long long start = clock64();
    while ((clock64() - start) < cycles) {
    }
}

static float* g_data;
static unsigned long long g_delay_cycles;
static int g_delayed_bucket_size = 8;

static hipError_t capture_scale(int size, hipGraph_t* graph, void* ctx) {
    hipStream_t stream;
    hipStreamCreate(&stream);

    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    if (size == g_delayed_bucket_size && g_delay_cycles > 0) {
        busy_wait_kernel<<<1, 1, 0, stream>>>(g_delay_cycles);
    }
    scale_kernel<<<(size + 255) / 256, 256, 0, stream>>>(g_data, 2.0f, size);
    hipStreamEndCapture(stream, graph);

    hipStreamDestroy(stream);
    return hipSuccess;
}

static hipError_t find_last_kernel_node(hipGraph_t graph, hipGraphNode_t* out_node) {
    size_t num_nodes = 0;
    hipError_t err = hipGraphGetNodes(graph, nullptr, &num_nodes);
    if (err != hipSuccess) return err;
    if (num_nodes == 0) return hipErrorInvalidValue;

    hipGraphNode_t* nodes = (hipGraphNode_t*)malloc(num_nodes * sizeof(hipGraphNode_t));
    if (!nodes) return hipErrorOutOfMemory;

    err = hipGraphGetNodes(graph, nodes, &num_nodes);
    if (err != hipSuccess) {
        free(nodes);
        return err;
    }

    bool found = false;
    for (size_t i = 0; i < num_nodes; i++) {
        hipGraphNodeType type;
        err = hipGraphNodeGetType(nodes[i], &type);
        if (err != hipSuccess) {
            free(nodes);
            return err;
        }
        if (type == hipGraphNodeTypeKernel) {
            *out_node = nodes[i];
            found = true;
        }
    }

    free(nodes);
    return found ? hipSuccess : hipErrorInvalidValue;
}

int main() {
    printf("=== test_shapes ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    g_delay_cycles =
        props.clockRate > 0 ? static_cast<unsigned long long>(props.clockRate) * 25ULL
                            : 25000000ULL;

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

    /* Update params for bucket 8 while its prior launch is still in flight */
    const int bucket_index = 2;
    hipGraphNode_t scale_node;
    err = find_last_kernel_node(pool.graphs[bucket_index], &scale_node);
    assert(err == hipSuccess);

    float factor3 = 3.0f;
    int bucket_size = buckets[bucket_index];
    void* kernel_args[] = {&g_data, &factor3, &bucket_size};
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(scale_kernel);
    params.gridDim = dim3((bucket_size + 255) / 256);
    params.blockDim = dim3(256);
    params.sharedMemBytes = 0;
    params.kernelParams = kernel_args;
    params.extra = nullptr;

    hipMemcpy(g_data, h_init, 8 * sizeof(float), hipMemcpyHostToDevice);
    err = hgb_shape_pool_launch(&pool, 5, stream, &actual);
    assert(err == hipSuccess);
    assert(actual == 8);

    err = hgb_shape_pool_update_params(&pool, bucket_index, scale_node, &params);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    hipMemcpy(h_out, g_data, 8 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  In-flight update preserved current launch result: [%.1f] (expected 2.0)\n",
           h_out[0]);
    assert(h_out[0] == 2.0f);

    hipMemcpy(g_data, h_init, 8 * sizeof(float), hipMemcpyHostToDevice);
    err = hgb_shape_pool_launch(&pool, 5, stream, &actual);
    assert(err == hipSuccess);
    hipStreamSynchronize(stream);

    hipMemcpy(h_out, g_data, 8 * sizeof(float), hipMemcpyDeviceToHost);
    printf("  Next launch sees updated params: [%.1f] (expected 3.0)\n", h_out[0]);
    assert(h_out[0] == 3.0f);

    /* Cleanup */
    hgb_shape_pool_destroy(&pool);
    hipFree(g_data);
    hipStreamDestroy(stream);
    hgb_shutdown();

    printf("  PASSED\n");
    return 0;
}
