/**
 * @file shape_manager.hip
 * @brief Gap 53: Dynamic input shapes via bucketing + param update
 *
 * Strategy: Pre-capture one graph per bucket size. At runtime, select
 * the smallest bucket >= input_size, pad if needed, launch cached graph.
 * ~90-95% of static graph performance.
 *
 * Thread safety: All public functions are protected by a per-object mutex,
 * matching CUDA Graph thread-safety guarantees.
 *
 * Follows the vLLM/SGLang shape-bucketing pattern.
 */
#include "hipgraph_bridge.h"
#include <cstdlib>
#include <cstring>

extern "C" void hgb_log(const char* fmt, ...);
extern "C" int hgb_is_initialized(void);

struct MutexGuard {
    pthread_mutex_t* m;
    MutexGuard(pthread_mutex_t* mtx) : m(mtx) { pthread_mutex_lock(m); }
    ~MutexGuard() { pthread_mutex_unlock(m); }
};

static void destroy_event(hipEvent_t* event) {
    if (event && *event) {
        hipEventDestroy(*event);
        *event = nullptr;
    }
}

static hipError_t ensure_bucket_event(hgb_shape_pool_t* pool, int bucket_index) {
    if (pool->events[bucket_index]) {
        return hipSuccess;
    }
    return hipEventCreateWithFlags(
        &pool->events[bucket_index], hipEventDisableTiming
    );
}

hipError_t hgb_shape_pool_create(
    hgb_capture_fn    graph_fn,
    void*             ctx,
    const int*        buckets,
    int               num_buckets,
    hgb_shape_pool_t* out
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!graph_fn || !buckets || num_buckets <= 0 || !out)
        return hipErrorInvalidValue;

    /* Enforce max bucket limit */
    if (num_buckets > HGB_MAX_BUCKETS) {
        hgb_log("ERROR: num_buckets %d exceeds max %d", num_buckets, HGB_MAX_BUCKETS);
        return hipErrorInvalidValue;
    }

    /* Validate buckets are in strictly ascending order and positive */
    for (int i = 0; i < num_buckets; i++) {
        if (buckets[i] <= 0) {
            hgb_log("ERROR: bucket[%d] = %d must be positive", i, buckets[i]);
            return hipErrorInvalidValue;
        }
        if (i > 0 && buckets[i] <= buckets[i - 1]) {
            hgb_log("ERROR: buckets not ascending: [%d]=%d <= [%d]=%d",
                    i, buckets[i], i - 1, buckets[i - 1]);
            return hipErrorInvalidValue;
        }
    }

    memset(out, 0, sizeof(*out));
    pthread_mutex_init(&out->lock, nullptr);
    out->num_buckets = num_buckets;
    hipGetDevice(&out->device_id);

    /* Allocate arrays */
    out->bucket_sizes = (int*)malloc(num_buckets * sizeof(int));
    out->graphs       = (hipGraph_t*)calloc(num_buckets, sizeof(hipGraph_t));
    out->execs        = (hipGraphExec_t*)calloc(num_buckets, sizeof(hipGraphExec_t));
    out->static_bufs  = (void**)calloc(num_buckets, sizeof(void*));
    out->events       = (hipEvent_t*)calloc(num_buckets, sizeof(hipEvent_t));

    if (!out->bucket_sizes || !out->graphs || !out->execs || !out->static_bufs || !out->events) {
        hgb_shape_pool_destroy(out);
        return hipErrorOutOfMemory;
    }

    memcpy(out->bucket_sizes, buckets, num_buckets * sizeof(int));

    /* Capture and instantiate a graph for each bucket */
    for (int i = 0; i < num_buckets; i++) {
        hgb_log("Capturing graph for bucket size %d", buckets[i]);

        hipError_t err = graph_fn(buckets[i], &out->graphs[i], ctx);
        if (err != hipSuccess) {
            hgb_log("Capture failed for bucket %d: %d", buckets[i], (int)err);
            hgb_shape_pool_destroy(out);
            return err;
        }

        err = hipGraphInstantiate(
            &out->execs[i], out->graphs[i], nullptr, nullptr, 0
        );
        if (err != hipSuccess) {
            hgb_log("Instantiate failed for bucket %d: %d", buckets[i], (int)err);
            hgb_shape_pool_destroy(out);
            return err;
        }

        err = hipEventCreateWithFlags(&out->events[i], hipEventDisableTiming);
        if (err != hipSuccess) {
            hgb_log("Event create failed for bucket %d: %d", buckets[i], (int)err);
            hgb_shape_pool_destroy(out);
            return err;
        }
    }

    hgb_log("Shape pool created: %d buckets [%d..%d]",
            num_buckets, buckets[0], buckets[num_buckets - 1]);
    return hipSuccess;
}

hipError_t hgb_shape_pool_launch(
    hgb_shape_pool_t* pool,
    int               input_size,
    hipStream_t       stream,
    int*              actual_bucket
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!pool || !pool->execs || input_size <= 0) return hipErrorInvalidValue;

    MutexGuard guard(&pool->lock);

    /* Device affinity check */
    int cur_dev;
    hipGetDevice(&cur_dev);
    if (cur_dev != pool->device_id) {
        hgb_log("ERROR: shape pool on device %d, current is %d",
                pool->device_id, cur_dev);
        return hipErrorInvalidDevice;
    }

    /* Binary search for smallest bucket >= input_size */
    int lo = 0, hi = pool->num_buckets - 1, best = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (pool->bucket_sizes[mid] >= input_size) {
            best = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    if (best < 0) {
        hgb_log("Input size %d exceeds largest bucket %d",
                input_size, pool->bucket_sizes[pool->num_buckets - 1]);
        return hipErrorInvalidValue;
    }

    if (actual_bucket) *actual_bucket = pool->bucket_sizes[best];

    hgb_log("Input %d → bucket %d (index %d, padding %d)",
            input_size, pool->bucket_sizes[best], best,
            pool->bucket_sizes[best] - input_size);

    hipError_t err = hipGraphLaunch(pool->execs[best], stream);
    if (err != hipSuccess) return err;

    err = ensure_bucket_event(pool, best);
    if (err != hipSuccess) {
        hgb_log("WARNING: event create for bucket %d failed: %d", best, (int)err);
        return hipSuccess;
    }

    /* Record event so update_params can sync before mutating this exec */
    err = hipEventRecord(pool->events[best], stream);
    if (err != hipSuccess) {
        hgb_log("WARNING: event record for bucket %d failed: %d", best, (int)err);
        destroy_event(&pool->events[best]);
    }

    return hipSuccess;
}

hipError_t hgb_shape_pool_update_params(
    hgb_shape_pool_t*     pool,
    int                   bucket_index,
    hipGraphNode_t        node,
    hipKernelNodeParams*  params
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!pool || !params) return hipErrorInvalidValue;
    if (bucket_index < 0 || bucket_index >= pool->num_buckets)
        return hipErrorInvalidValue;

    MutexGuard guard(&pool->lock);

    /* Device affinity check */
    int cur_dev;
    hipGetDevice(&cur_dev);
    if (cur_dev != pool->device_id) {
        hgb_log("ERROR: shape pool on device %d, current is %d",
                pool->device_id, cur_dev);
        return hipErrorInvalidDevice;
    }

    /* Sync any in-flight launch of this bucket before mutating its exec */
    if (pool->events && pool->events[bucket_index]) {
        hipError_t query_err = hipEventQuery(pool->events[bucket_index]);
        if (query_err == hipErrorNotReady) {
            hipError_t sync_err = hipEventSynchronize(pool->events[bucket_index]);
            if (sync_err != hipSuccess) {
                hgb_log("WARNING: event sync for bucket %d failed: %d",
                        bucket_index, (int)sync_err);
                return sync_err;
            }
        } else if (query_err != hipSuccess) {
            hgb_log("WARNING: event query for bucket %d failed: %d",
                    bucket_index, (int)query_err);
            hipError_t sync_err = hipDeviceSynchronize();
            if (sync_err != hipSuccess) {
                return sync_err;
            }
        }
    } else {
        hgb_log("WARNING: event tracking unavailable for bucket %d, falling back to device sync",
                bucket_index);
        hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            return sync_err;
        }
    }

    return hipGraphExecKernelNodeSetParams(
        pool->execs[bucket_index], node, params
    );
}

void hgb_shape_pool_destroy(hgb_shape_pool_t* pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);

    /* Sync on the owning device before destroying graph resources */
    if (pool->device_id >= 0) {
        int prev_dev;
        hipGetDevice(&prev_dev);
        if (prev_dev != pool->device_id) {
            hipSetDevice(pool->device_id);
        }
        hipDeviceSynchronize();
        if (prev_dev != pool->device_id) {
            hipSetDevice(prev_dev);
        }
    }

    for (int i = 0; i < pool->num_buckets; i++) {
        if (pool->events)
            destroy_event(&pool->events[i]);
        if (pool->execs && pool->execs[i])
            hipGraphExecDestroy(pool->execs[i]);
        if (pool->graphs && pool->graphs[i])
            hipGraphDestroy(pool->graphs[i]);
        if (pool->static_bufs && pool->static_bufs[i])
            hipFree(pool->static_bufs[i]);
    }

    free(pool->bucket_sizes);
    free(pool->graphs);
    free(pool->execs);
    free(pool->static_bufs);
    free(pool->events);
    memset(pool, 0, sizeof(*pool));
    pool->device_id = -1;

    pthread_mutex_unlock(&pool->lock);
    pthread_mutex_destroy(&pool->lock);
}
