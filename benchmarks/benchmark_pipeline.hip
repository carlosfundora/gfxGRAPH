/**
 * @file benchmark_pipeline.hip
 * @brief Informational launch-latency benchmark for the native pipeline bridge
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>

__global__ void inc_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void clock_probe_kernel(unsigned long long* out) {
    unsigned long long start = clock64();
    float acc = 0.0f;
    for (int i = 0; i < 256; ++i) {
        acc += (float)i * 0.0001f;
    }
    unsigned long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = end - start + (unsigned long long)acc;
    }
}

int main() {
    constexpr int N = 1024;
    constexpr int WARMUP = 100;
    constexpr int ITERS = 10000;

    hipError_t err = hgb_init();
    if (err != hipSuccess) {
        std::fprintf(stderr, "hgb_init failed: %d\n", (int)err);
        return 1;
    }

    float* d_data = nullptr;
    hipMalloc(&d_data, N * sizeof(float));
    hipMemset(d_data, 0, N * sizeof(float));

    hipStream_t cap_stream;
    hipStreamCreate(&cap_stream);

    hipGraph_t graph;
    hipStreamBeginCapture(cap_stream, hipStreamCaptureModeGlobal);
    inc_kernel<<<(N + 255) / 256, 256, 0, cap_stream>>>(d_data, N);
    hipStreamEndCapture(cap_stream, &graph);
    hipStreamDestroy(cap_stream);

    hgb_pipeline_t pipe;
    err = hgb_pipeline_create(graph, &pipe);
    if (err != hipSuccess) {
        std::fprintf(stderr, "hgb_pipeline_create failed: %d\n", (int)err);
        hipGraphDestroy(graph);
        hipFree(d_data);
        hgb_shutdown();
        return 1;
    }

    for (int i = 0; i < WARMUP; i++) {
        err = hgb_pipeline_launch(&pipe);
        if (err != hipSuccess) {
            std::fprintf(stderr, "warmup launch failed: %d\n", (int)err);
            hgb_pipeline_destroy(&pipe);
            hipGraphDestroy(graph);
            hipFree(d_data);
            hgb_shutdown();
            return 1;
        }
    }
    hipDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        err = hgb_pipeline_launch(&pipe);
        if (err != hipSuccess) {
            std::fprintf(stderr, "benchmark launch failed: %d\n", (int)err);
            hgb_pipeline_destroy(&pipe);
            hipGraphDestroy(graph);
            hipFree(d_data);
            hgb_shutdown();
            return 1;
        }
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = total_us / ITERS;

    unsigned long long* d_cycles = nullptr;
    unsigned long long h_cycles = 0;
    hipMalloc(&d_cycles, sizeof(unsigned long long));
    clock_probe_kernel<<<1, 64>>>(d_cycles);
    hipMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), hipMemcpyDeviceToHost);
    hipFree(d_cycles);
    hgb_profiler_record(
        HGB_PROFILE_EVENT_DEVICE_CLOCK_PROBE,
        0,
        (uint64_t)h_cycles,
        256
    );

    hgb_profile_counters_t counters{};
    hgb_profile_sample_t samples[16];
    size_t sample_count = hgb_profiler_snapshot(samples, 16, &counters);

    std::printf("=== benchmark_pipeline ===\n");
    std::printf("Warmup launches: %d\n", WARMUP);
    std::printf("Measured launches: %d\n", ITERS);
    std::printf("Average pipeline launch latency: %.2f us\n", avg_us);
    std::printf("Device clock64 probe cycles: %llu\n", h_cycles);
    std::printf("Profiler samples copied: %zu\n", sample_count);
    std::printf("Profiler events written: %llu\n",
                (unsigned long long)counters.written);
    std::printf("Profiler events dropped: %llu\n",
                (unsigned long long)counters.dropped);

    hgb_pipeline_destroy(&pipe);
    hipGraphDestroy(graph);
    hipFree(d_data);
    hgb_shutdown();
    return 0;
}
