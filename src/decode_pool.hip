/**
 * @file decode_pool.hip
 * @brief Capture-safe paged-decode pool — fixes decode-attention-under-hipGraph garble.
 *
 * The "aiter garbles under gfxGRAPH capture" report is decode-attn-under-graph REPLAY-DATA-UNSAFETY,
 * not JIT-during-capture: a captured decode graph that bakes the per-step sequence length / block
 * table BY VALUE attends the capture-time length on every replay, so output garbles as the sequence
 * grows. (Confirmed on gfx1030: a stale-metadata replay tracks the capture-time answer; refreshing the
 * persistent device buffer in place makes the same captured graph bit-exact at the live length.)
 *
 * This pool owns PERSISTENT device metadata buffers (seq_lens, block_tables) at fixed addresses. Each
 * bucket's captured graph reads those buffers; replay = memcpy(live -> persistent) + hipGraphLaunch.
 * No hipGraphExec*SetParams is needed — the kernel-node pointer args never change, only the buffer
 * CONTENTS, so the captured node reads fresh data each replay. One graph per bucket (bucket-constant
 * max-seq-len) replays correctly for every step. This wires the previously-vestigial
 * hgb_shape_pool_t::static_bufs idea (allocated-but-never-written) the correct way.
 *
 * Mirrors shape_manager.hip's per-object-mutex thread-safety and event-tracked in-flight handling.
 */
#include "hipgraph_bridge.h"
#include <cstdlib>
#include <cstring>

extern "C" void hgb_log(const char* fmt, ...);
extern "C" int hgb_is_initialized(void);

namespace {

struct DecodeMutexGuard {
    pthread_mutex_t* m;
    DecodeMutexGuard(pthread_mutex_t* mtx) : m(mtx) { pthread_mutex_lock(m); }
    ~DecodeMutexGuard() { pthread_mutex_unlock(m); }
};

void decode_destroy_event(hipEvent_t* event) {
    if (event && *event) {
        hipEventDestroy(*event);
        *event = nullptr;
    }
}

}  // namespace

hipError_t hgb_decode_pool_create(
    hgb_decode_capture_fn fn,
    void*                 ctx,
    const int*            buckets,
    int                   num_buckets,
    int                   max_num_seqs,
    int                   max_blocks_per_seq,
    hgb_decode_pool_t*    out
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!fn || !buckets || num_buckets <= 0 || max_num_seqs <= 0 || max_blocks_per_seq <= 0 || !out)
        return hipErrorInvalidValue;

    if (num_buckets > HGB_MAX_BUCKETS) {
        hgb_log("ERROR: decode num_buckets %d exceeds max %d", num_buckets, HGB_MAX_BUCKETS);
        return hipErrorInvalidValue;
    }

    /* Buckets strictly ascending and positive (smallest-bucket-ge-size selection depends on it). */
    for (int i = 0; i < num_buckets; i++) {
        if (buckets[i] <= 0) {
            hgb_log("ERROR: decode bucket[%d] = %d must be positive", i, buckets[i]);
            return hipErrorInvalidValue;
        }
        if (i > 0 && buckets[i] <= buckets[i - 1]) {
            hgb_log("ERROR: decode buckets not ascending: [%d]=%d <= [%d]=%d",
                    i, buckets[i], i - 1, buckets[i - 1]);
            return hipErrorInvalidValue;
        }
    }

    memset(out, 0, sizeof(*out));
    pthread_mutex_init(&out->lock, nullptr);
    out->num_buckets = num_buckets;
    out->max_num_seqs = max_num_seqs;
    out->max_blocks_per_seq = max_blocks_per_seq;
    hipGetDevice(&out->device_id);

    out->bucket_sizes = (int*)malloc(num_buckets * sizeof(int));
    out->graphs       = (hipGraph_t*)calloc(num_buckets, sizeof(hipGraph_t));
    out->execs        = (hipGraphExec_t*)calloc(num_buckets, sizeof(hipGraphExec_t));
    out->events       = (hipEvent_t*)calloc(num_buckets, sizeof(hipEvent_t));
    if (!out->bucket_sizes || !out->graphs || !out->execs || !out->events) {
        hgb_decode_pool_destroy(out);
        return hipErrorOutOfMemory;
    }
    memcpy(out->bucket_sizes, buckets, num_buckets * sizeof(int));

    /* Persistent metadata buffers — captured graphs read these by pointer; replay refreshes contents. */
    if (hipMalloc((void**)&out->d_seq_lens, (size_t)max_num_seqs * sizeof(int)) != hipSuccess) {
        hgb_decode_pool_destroy(out);
        return hipErrorOutOfMemory;
    }
    if (hipMalloc((void**)&out->d_block_tables,
                  (size_t)max_num_seqs * max_blocks_per_seq * sizeof(int)) != hipSuccess) {
        hgb_decode_pool_destroy(out);
        return hipErrorOutOfMemory;
    }
    hipMemset(out->d_seq_lens, 0, (size_t)max_num_seqs * sizeof(int));
    hipMemset(out->d_block_tables, 0, (size_t)max_num_seqs * max_blocks_per_seq * sizeof(int));

    /* Capture + instantiate one graph per bucket, each reading the SAME persistent buffers. */
    for (int i = 0; i < num_buckets; i++) {
        hgb_log("Capturing decode graph for bucket (max_seq_len) %d", buckets[i]);

        hipError_t err = fn(buckets[i], out->d_seq_lens, out->d_block_tables, &out->graphs[i], ctx);
        if (err != hipSuccess) {
            hgb_log("Decode capture failed for bucket %d: %d", buckets[i], (int)err);
            hgb_decode_pool_destroy(out);
            return err;
        }
        err = hipGraphInstantiate(&out->execs[i], out->graphs[i], nullptr, nullptr, 0);
        if (err != hipSuccess) {
            hgb_log("Decode instantiate failed for bucket %d: %d", buckets[i], (int)err);
            hgb_decode_pool_destroy(out);
            return err;
        }
        err = hipEventCreateWithFlags(&out->events[i], hipEventDisableTiming);
        if (err != hipSuccess) {
            hgb_log("Decode event create failed for bucket %d: %d", buckets[i], (int)err);
            hgb_decode_pool_destroy(out);
            return err;
        }
    }

    hgb_log("Decode pool created: %d buckets [%d..%d], max_num_seqs=%d max_blocks_per_seq=%d",
            num_buckets, buckets[0], buckets[num_buckets - 1], max_num_seqs, max_blocks_per_seq);
    return hipSuccess;
}

hipError_t hgb_decode_pool_replay(
    hgb_decode_pool_t* pool,
    int                input_size,
    const int*         h_seq_lens,
    int                num_seqs,
    const int*         h_block_tables,
    hipStream_t        stream,
    int*               actual_bucket
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!pool || !pool->execs || !h_seq_lens || input_size <= 0 || num_seqs <= 0)
        return hipErrorInvalidValue;
    if (num_seqs > pool->max_num_seqs) {
        hgb_log("ERROR: decode replay num_seqs %d exceeds max_num_seqs %d", num_seqs, pool->max_num_seqs);
        return hipErrorInvalidValue;
    }

    DecodeMutexGuard guard(&pool->lock);

    int cur_dev;
    hipGetDevice(&cur_dev);
    if (cur_dev != pool->device_id) {
        hgb_log("ERROR: decode pool on device %d, current is %d", pool->device_id, cur_dev);
        return hipErrorInvalidDevice;
    }

    /* Binary search for smallest bucket >= input_size. */
    int lo = 0, hi = pool->num_buckets - 1, best = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (pool->bucket_sizes[mid] >= input_size) { best = mid; hi = mid - 1; }
        else { lo = mid + 1; }
    }
    if (best < 0) {
        hgb_log("Decode input size %d exceeds largest bucket %d",
                input_size, pool->bucket_sizes[pool->num_buckets - 1]);
        return hipErrorInvalidValue;
    }
    if (actual_bucket) *actual_bucket = pool->bucket_sizes[best];

    /* Refresh the persistent metadata IN PLACE on the launch stream (ordered before the graph). This
     * is the fix: the captured graph reads these device buffers, so fresh contents => correct replay. */
    hipError_t err = hipMemcpyAsync(pool->d_seq_lens, h_seq_lens, (size_t)num_seqs * sizeof(int),
                                    hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) return err;
    if (h_block_tables) {
        err = hipMemcpyAsync(pool->d_block_tables, h_block_tables,
                             (size_t)num_seqs * pool->max_blocks_per_seq * sizeof(int),
                             hipMemcpyHostToDevice, stream);
        if (err != hipSuccess) return err;
    }

    err = hipGraphLaunch(pool->execs[best], stream);
    if (err != hipSuccess) return err;

    /* Record completion so a later replay/destroy can sync before reusing the persistent buffers. */
    if (pool->events && pool->events[best]) {
        hipError_t rec = hipEventRecord(pool->events[best], stream);
        if (rec != hipSuccess) {
            hgb_log("WARNING: decode event record for bucket %d failed: %d", best, (int)rec);
            decode_destroy_event(&pool->events[best]);
        }
    }
    return hipSuccess;
}

void hgb_decode_pool_destroy(hgb_decode_pool_t* pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);

    if (pool->device_id >= 0) {
        int prev_dev;
        hipGetDevice(&prev_dev);
        if (prev_dev != pool->device_id) hipSetDevice(pool->device_id);
        hipDeviceSynchronize();
        if (prev_dev != pool->device_id) hipSetDevice(prev_dev);
    }

    for (int i = 0; i < pool->num_buckets; i++) {
        if (pool->events) decode_destroy_event(&pool->events[i]);
        if (pool->execs && pool->execs[i]) hipGraphExecDestroy(pool->execs[i]);
        if (pool->graphs && pool->graphs[i]) hipGraphDestroy(pool->graphs[i]);
    }
    if (pool->d_seq_lens) hipFree(pool->d_seq_lens);
    if (pool->d_block_tables) hipFree(pool->d_block_tables);

    free(pool->bucket_sizes);
    free(pool->graphs);
    free(pool->execs);
    free(pool->events);
    memset(pool, 0, sizeof(*pool));
    pool->device_id = -1;

    pthread_mutex_unlock(&pool->lock);
    pthread_mutex_destroy(&pool->lock);
}
