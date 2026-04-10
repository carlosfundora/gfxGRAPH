/**
 * @file launch_pipeline.hip
 * @brief Gap 52: Rapid graph launch via pre-upload + serialized double-buffer
 *
 * Strategy: Maintain two hipGraphExec_t from the same graph. The "active"
 * index is the NEXT exec to launch. Updates always target exec[active].
 * Launches are serialized via a single event — since both execs share
 * captured memory pointers, concurrent execution would corrupt data.
 *
 * The double-buffer enables overlapping kernel param updates with
 * execution: update exec[active] while exec[active^1] runs, then launch.
 *
 * Measured ~44μs per launch on gfx1030 (vs device-side launch ~5μs on CDNA).
 * Limitation: Requires host round-trip (no device-side graph trigger).
 */
#include "hipgraph_bridge.h"
#include <cstring>

extern void hgb_log(const char* fmt, ...);
extern int hgb_is_initialized(void);

hipError_t hgb_pipeline_create(
    hipGraph_t       graph,
    hgb_pipeline_t*  out
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!out) return hipErrorInvalidValue;
    memset(out, 0, sizeof(*out));
    hipGetDevice(&out->device_id);

    hipError_t err;

    /* Create two exec instances (double buffer) */
    for (int i = 0; i < 2; i++) {
        err = hipGraphInstantiate(&out->exec[i], graph, nullptr, nullptr, 0);
        if (err != hipSuccess) {
            hgb_pipeline_destroy(out);
            return err;
        }
        out->state[i] = HGB_EXEC_IDLE;
    }

    /* Single stream + single event for serialization */
    err = hipStreamCreate(&out->stream);
    if (err != hipSuccess) {
        hgb_pipeline_destroy(out);
        return err;
    }

    err = hipEventCreate(&out->event);
    if (err != hipSuccess) {
        hgb_pipeline_destroy(out);
        return err;
    }

    /* Pre-upload the first exec */
    err = hipGraphUpload(out->exec[0], out->stream);
    if (err == hipSuccess) {
        out->state[0] = HGB_EXEC_UPLOADED;
    } else {
        hgb_log("WARNING: hipGraphUpload failed (%d), launch still works", (int)err);
    }

    out->active = 0;
    out->launched = 0;
    hgb_log("Pipeline created: serialized double-buffer, event-tracked");
    return hipSuccess;
}

hipError_t hgb_pipeline_launch(hgb_pipeline_t* pipe) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!pipe || !pipe->exec[0]) return hipErrorInvalidValue;

    /* Device affinity check */
    int cur_dev;
    hipGetDevice(&cur_dev);
    if (cur_dev != pipe->device_id) {
        hgb_log("ERROR: pipeline created on device %d, current is %d",
                pipe->device_id, cur_dev);
        return hipErrorInvalidDevice;
    }

    /* Wait for any prior launch to complete (shared memory serialization) */
    if (pipe->launched) {
        hipError_t err = hipEventSynchronize(pipe->event);
        if (err != hipSuccess) return err;
        /* Mark previously-launched exec as idle */
        int prev = pipe->active ^ 1;
        pipe->state[prev] = HGB_EXEC_IDLE;
    }

    int cur = pipe->active;

    /* Re-upload if needed (e.g., after param update) */
    if (pipe->state[cur] == HGB_EXEC_IDLE) {
        hipError_t err = hipGraphUpload(pipe->exec[cur], pipe->stream);
        if (err != hipSuccess) {
            hgb_log("WARNING: re-upload failed (%d)", (int)err);
            /* Continue — launch may still work without pre-upload */
        } else {
            pipe->state[cur] = HGB_EXEC_UPLOADED;
        }
    }

    /* Launch on the single serialized stream */
    hipError_t err = hipGraphLaunch(pipe->exec[cur], pipe->stream);
    if (err != hipSuccess) return err;

    /* Record completion event */
    err = hipEventRecord(pipe->event, pipe->stream);
    if (err != hipSuccess) {
        /* Conservative: sync stream to ensure we don't lose track of in-flight work */
        hipStreamSynchronize(pipe->stream);
        pipe->state[cur] = HGB_EXEC_IDLE;
        return err;
    }

    pipe->state[cur] = HGB_EXEC_IN_FLIGHT;
    pipe->launched = 1;

    /* Swap: active now points to the next-to-launch buffer */
    pipe->active ^= 1;

    return hipSuccess;
}

hipError_t hgb_pipeline_update_kernel(
    hgb_pipeline_t*       pipe,
    hipGraphNode_t        node,
    hipKernelNodeParams*  params
) {
    if (!hgb_is_initialized()) return hipErrorNotInitialized;
    if (!pipe || !params) return hipErrorInvalidValue;

    /*
     * Update exec[active] — the NEXT buffer to be launched.
     * This is safe because:
     * - If nothing launched yet: active=0, both idle, update is safe.
     * - After a launch: active was swapped, so active = the OTHER buffer
     *   which is either IDLE or was previously completed.
     * - The launched buffer (active^1) may be in-flight, but we don't touch it.
     */
    int target = pipe->active;

    hipError_t err = hipGraphExecKernelNodeSetParams(
        pipe->exec[target], node, params
    );
    if (err != hipSuccess) return err;

    /* Mark as needing re-upload before next launch */
    pipe->state[target] = HGB_EXEC_IDLE;
    return hipSuccess;
}

void hgb_pipeline_destroy(hgb_pipeline_t* pipe) {
    if (!pipe) return;

    /* Wait for any in-flight work */
    if (pipe->stream) {
        hipStreamSynchronize(pipe->stream);
    }

    if (pipe->event) {
        hipEventDestroy(pipe->event);
        pipe->event = nullptr;
    }

    for (int i = 0; i < 2; i++) {
        if (pipe->exec[i]) {
            hipGraphExecDestroy(pipe->exec[i]);
            pipe->exec[i] = nullptr;
        }
        pipe->state[i] = HGB_EXEC_IDLE;
    }

    if (pipe->stream) {
        hipStreamDestroy(pipe->stream);
        pipe->stream = nullptr;
    }

    pipe->active = 0;
    pipe->launched = 0;
    pipe->device_id = -1;
}
