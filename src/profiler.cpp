/**
 * @file profiler.cpp
 * @brief Low-overhead native telemetry for gfxGRAPH runtime paths.
 */
#include "hipgraph_bridge.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>

namespace {

struct ProfileSlot {
    std::atomic<uint64_t> published_seq;
    hgb_profile_sample_t sample;
};

std::array<ProfileSlot, HGB_PROFILER_CAPACITY> g_slots{};
std::atomic<uint64_t> g_next_seq{1};
std::atomic<uint64_t> g_dropped{0};

uint32_t current_device_id() {
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device < 0) {
        return UINT32_MAX;
    }
    return static_cast<uint32_t>(device);
}

}  // namespace

extern "C" HGB_EXPORT uint64_t hgb_monotonic_ns(void) {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock::now().time_since_epoch()
        ).count()
    );
}

extern "C" HGB_EXPORT void hgb_profiler_reset(void) {
    for (auto& slot : g_slots) {
        slot.published_seq.store(0, std::memory_order_release);
        std::memset(&slot.sample, 0, sizeof(slot.sample));
    }
    g_next_seq.store(1, std::memory_order_release);
    g_dropped.store(0, std::memory_order_release);
}

extern "C" HGB_EXPORT uint64_t hgb_profiler_record(
    uint32_t event,
    uint64_t duration_ns,
    uint64_t value0,
    uint64_t value1
) {
    const uint64_t seq = g_next_seq.fetch_add(1, std::memory_order_acq_rel);
    const uint64_t index = (seq - 1) % HGB_PROFILER_CAPACITY;

    if (seq > HGB_PROFILER_CAPACITY) {
        g_dropped.fetch_add(1, std::memory_order_relaxed);
    }

    hgb_profile_sample_t sample{};
    sample.seq = seq;
    sample.timestamp_ns = hgb_monotonic_ns();
    sample.duration_ns = duration_ns;
    sample.value0 = value0;
    sample.value1 = value1;
    sample.event = event;
    sample.device_id = current_device_id();
    sample.stream_id = 0;
    sample.flags = 0;

    g_slots[index].sample = sample;
    g_slots[index].published_seq.store(seq, std::memory_order_release);
    return seq;
}

extern "C" HGB_EXPORT size_t hgb_profiler_snapshot(
    hgb_profile_sample_t*   out,
    size_t                  max_samples,
    hgb_profile_counters_t* counters
) {
    const uint64_t next_seq = g_next_seq.load(std::memory_order_acquire);
    const uint64_t written = next_seq > 0 ? next_seq - 1 : 0;
    const uint64_t available =
        written < HGB_PROFILER_CAPACITY ? written : HGB_PROFILER_CAPACITY;
    const uint64_t to_copy =
        max_samples < available ? max_samples : available;
    const uint64_t first_seq = written >= to_copy ? written - to_copy + 1 : 1;

    if (counters) {
        counters->written = written;
        counters->dropped = g_dropped.load(std::memory_order_acquire);
        counters->capacity = HGB_PROFILER_CAPACITY;
    }

    if (!out || max_samples == 0 || to_copy == 0) {
        return 0;
    }

    size_t copied = 0;
    for (uint64_t seq = first_seq; seq <= written; ++seq) {
        const uint64_t index = (seq - 1) % HGB_PROFILER_CAPACITY;
        const uint64_t published =
            g_slots[index].published_seq.load(std::memory_order_acquire);
        if (published == seq) {
            out[copied++] = g_slots[index].sample;
        }
    }
    return copied;
}
