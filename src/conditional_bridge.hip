/**
 * @file conditional_bridge.hip
 * @brief Gap 51: Conditional execution via per-branch graph exec dispatch
 *
 * Strategy: Instantiate each branch as a separate hipGraphExec_t.
 * At launch time, dispatch to the selected branch exec directly.
 * This avoids hipGraphNodeSetEnabled on child graph nodes, which is
 * unsupported on HIP 7.2 (only kernel/memset/memcpy nodes supported).
 *
 * Thread safety: All public functions are protected by a per-object mutex,
 * matching CUDA Graph thread-safety guarantees.
 *
 * Destruction: caller must ensure no launches are in-flight on the
 * stream passed to launch(). Destroy calls hipDeviceSynchronize()
 * as a safety net, but this is expensive — prefer explicit sync.
 *
 * Limitation: While-loop semantics are NOT supported (no device-side loop).
 */
#include "hipgraph_bridge.h"
#include <cstdlib>
#include <cstring>

extern void hgb_log(const char* fmt, ...);
extern int hgb_is_initialized(void);

/* RAII lock guard for pthread_mutex_t */
struct MutexGuard {
    pthread_mutex_t* m;
    MutexGuard(pthread_mutex_t* mtx) : m(mtx) { pthread_mutex_lock(m); }
    ~MutexGuard() { pthread_mutex_unlock(m); }
};

static hipError_t check_device(int expected) {
    int cur;
    hipGetDevice(&cur);
    if (cur != expected) {
        hgb_log("ERROR: object on device %d, current is %d", expected, cur);
        return hipErrorInvalidDevice;
    }
    return hipSuccess;
}

hipError_t hgb_conditional_create(
    hipGraph_t*         graphs,
    int                 count,
    hgb_conditional_t*  out
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!graphs || count <= 0 || count > HGB_MAX_BRANCHES || !out)
        return hipErrorInvalidValue;

    memset(out, 0, sizeof(*out));
    pthread_mutex_init(&out->lock, nullptr);
    out->num_branches = count;
    hipGetDevice(&out->device_id);

    out->branch_execs = (hipGraphExec_t*)calloc(count, sizeof(hipGraphExec_t));
    if (!out->branch_execs) {
        hgb_conditional_destroy(out);
        return hipErrorOutOfMemory;
    }

    for (int i = 0; i < count; i++) {
        hipError_t err = hipGraphInstantiate(
            &out->branch_execs[i], graphs[i], nullptr, nullptr, 0
        );
        if (err != hipSuccess) {
            hgb_log("Failed to instantiate branch %d: %d", i, (int)err);
            hgb_conditional_destroy(out);
            return err;
        }

        size_t num_nodes = 0;
        hipGraphGetNodes(graphs[i], nullptr, &num_nodes);
        hgb_log("Branch %d: %zu nodes, exec ready", i, num_nodes);
    }

    hgb_log("Conditional created: %d branches (per-branch exec dispatch)", count);
    return hipSuccess;
}

hipError_t hgb_conditional_launch(
    hgb_conditional_t*  cond,
    int                 branch_index,
    hipStream_t         stream
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!cond || !cond->branch_execs) return hipErrorInvalidValue;
    if (branch_index < 0 || branch_index >= cond->num_branches)
        return hipErrorInvalidValue;

    MutexGuard guard(&cond->lock);

    hipError_t err = check_device(cond->device_id);
    if (err != hipSuccess) return err;

    return hipGraphLaunch(cond->branch_execs[branch_index], stream);
}

hipError_t hgb_conditional_launch_by_flag(
    hgb_conditional_t*  cond,
    int*                d_flag,
    hipStream_t         stream
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!cond || !d_flag) return hipErrorInvalidValue;

    MutexGuard guard(&cond->lock);

    hipError_t err = check_device(cond->device_id);
    if (err != hipSuccess) return err;

    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) return err;

    int flag = 0;
    err = hipMemcpy(&flag, d_flag, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) return err;

    if (flag < 0 || flag >= cond->num_branches) {
        hgb_log("ERROR: branch flag %d out of range [0, %d)",
                flag, cond->num_branches);
        return hipErrorInvalidValue;
    }

    return hipGraphLaunch(cond->branch_execs[flag], stream);
}

void hgb_conditional_destroy(hgb_conditional_t* cond) {
    if (!cond) return;

    pthread_mutex_lock(&cond->lock);

    if (cond->device_id >= 0) {
        int prev_dev;
        hipGetDevice(&prev_dev);
        if (prev_dev != cond->device_id)
            hipSetDevice(cond->device_id);
        hipDeviceSynchronize();
        if (prev_dev != cond->device_id)
            hipSetDevice(prev_dev);
    }

    for (int i = 0; i < cond->num_branches; i++) {
        if (cond->branch_execs && cond->branch_execs[i])
            hipGraphExecDestroy(cond->branch_execs[i]);
    }

    free(cond->branch_execs);
    cond->branch_execs = nullptr;
    cond->num_branches = 0;
    cond->device_id = -1;

    pthread_mutex_unlock(&cond->lock);
    pthread_mutex_destroy(&cond->lock);
}
