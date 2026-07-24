/**
 * @file test_runtime_handles.hip
 * @brief Tests opaque native runtime handles for FFI callers.
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdio>

__global__ void add_kernel(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += value;
}

static hipError_t capture_add_graph(float* d_data, int n, float value, hipGraph_t* graph) {
    hipStream_t stream = nullptr;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) return err;

    err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    if (err != hipSuccess) {
        hipStreamDestroy(stream);
        return err;
    }

    add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_data, n, value);

    err = hipStreamEndCapture(stream, graph);
    hipStreamDestroy(stream);
    return err;
}

int main() {
    std::printf("=== test_runtime_handles ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    constexpr int N = 1024;
    float* d_data = nullptr;
    err = hipMalloc(&d_data, N * sizeof(float));
    assert(err == hipSuccess);
    err = hipMemset(d_data, 0, N * sizeof(float));
    assert(err == hipSuccess);

    hipGraph_t pipeline_graph = nullptr;
    err = capture_add_graph(d_data, N, 1.0f, &pipeline_graph);
    assert(err == hipSuccess);

    hgb_pipeline_handle_t* pipeline = nullptr;
    err = hgb_pipeline_handle_create(pipeline_graph, &pipeline);
    assert(err == hipSuccess);
    assert(pipeline != nullptr);

    for (int i = 0; i < 5; ++i) {
        err = hgb_pipeline_handle_launch(pipeline);
        assert(err == hipSuccess);
    }
    err = hipDeviceSynchronize();
    assert(err == hipSuccess);

    float h_data[4] = {};
    err = hipMemcpy(h_data, d_data, sizeof(h_data), hipMemcpyDeviceToHost);
    assert(err == hipSuccess);
    std::printf("  pipeline handle result %.1f (expected 5.0)\n", h_data[0]);
    assert(h_data[0] == 5.0f);

    hgb_pipeline_handle_destroy(pipeline);
    hipGraphDestroy(pipeline_graph);

    err = hipMemset(d_data, 0, N * sizeof(float));
    assert(err == hipSuccess);

    hipGraph_t sub_graphs[2] = {};
    err = capture_add_graph(d_data, N, 2.0f, &sub_graphs[0]);
    assert(err == hipSuccess);
    err = capture_add_graph(d_data, N, 3.0f, &sub_graphs[1]);
    assert(err == hipSuccess);

    int deps[2] = {-1, 0};
    hgb_composed_graph_handle_t* composed = nullptr;
    err = hgb_composed_handle_create(sub_graphs, 2, deps, &composed);
    assert(err == hipSuccess);
    assert(composed != nullptr);

    hipStream_t launch_stream = nullptr;
    err = hipStreamCreate(&launch_stream);
    assert(err == hipSuccess);
    err = hgb_composed_handle_launch(composed, launch_stream);
    assert(err == hipSuccess);
    err = hipStreamSynchronize(launch_stream);
    assert(err == hipSuccess);

    err = hipMemcpy(h_data, d_data, sizeof(h_data), hipMemcpyDeviceToHost);
    assert(err == hipSuccess);
    std::printf("  composed handle result %.1f (expected 5.0)\n", h_data[0]);
    assert(h_data[0] == 5.0f);

    hgb_composed_handle_destroy(composed);
    hipStreamDestroy(launch_stream);
    hipGraphDestroy(sub_graphs[0]);
    hipGraphDestroy(sub_graphs[1]);
    hipFree(d_data);
    hgb_shutdown();

    std::printf("  PASSED\n");
    return 0;
}
