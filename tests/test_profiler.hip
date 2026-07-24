/**
 * @file test_profiler.hip
 * @brief Tests for native profiler ring buffer.
 */
#include "hipgraph_bridge.h"
#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdio>

int main() {
    std::printf("=== test_profiler ===\n");
    hipError_t err = hgb_init();
    assert(err == hipSuccess);

    hgb_profiler_reset();
    uint64_t seq1 = hgb_profiler_record(HGB_PROFILE_EVENT_PIPELINE_CREATE, 10, 1, 2);
    uint64_t seq2 = hgb_profiler_record(HGB_PROFILE_EVENT_PIPELINE_LAUNCH, 20, 3, 4);
    assert(seq1 == 1);
    assert(seq2 == 2);

    hgb_profile_counters_t counters{};
    hgb_profile_sample_t samples[8];
    size_t copied = hgb_profiler_snapshot(samples, 8, &counters);

    assert(copied == 2);
    assert(counters.written == 2);
    assert(counters.capacity == HGB_PROFILER_CAPACITY);
    assert(samples[0].event == HGB_PROFILE_EVENT_PIPELINE_CREATE);
    assert(samples[1].event == HGB_PROFILE_EVENT_PIPELINE_LAUNCH);
    assert(samples[1].duration_ns == 20);

    std::printf("  copied=%zu written=%llu capacity=%llu\n",
                copied,
                (unsigned long long)counters.written,
                (unsigned long long)counters.capacity);

    hgb_shutdown();
    std::printf("  PASSED\n");
    return 0;
}
