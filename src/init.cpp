/**
 * @file init.cpp
 * @brief gfxGRAPH lifecycle: init, shutdown, version, validation
 */
#include "hipgraph_bridge.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_initialized = 0;
static int g_debug = 0;
static hipDeviceProp_t g_dev_props;

void hgb_log(const char* fmt, ...) {
    if (!g_debug) return;
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[gfxGRAPH] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

hgb_version_t hgb_get_version(void) {
    hgb_version_t v;
    v.major = HGB_VERSION_MAJOR;
    v.minor = HGB_VERSION_MINOR;
    v.patch = HGB_VERSION_PATCH;
    v.gfx_target = "gfx1030";
    v.rocm_version = "7.2.0";
    return v;
}

hipError_t hgb_validate_gfx1030(void) {
    hipDeviceProp_t props;
    hipError_t err = hipGetDeviceProperties(&props, 0);
    if (err != hipSuccess) return err;

    /* gcnArchName contains "gfx1030" on RDNA2 */
    if (strstr(props.gcnArchName, "gfx1030") == nullptr) {
        hgb_log("ERROR: expected gfx1030, got %s", props.gcnArchName);
        return hipErrorInvalidDevice;
    }
    return hipSuccess;
}

hipError_t hgb_init(void) {
    if (g_initialized) return hipSuccess;

    /* Check env for debug flag */
    const char* dbg = getenv("HGB_DEBUG");
    if (dbg && dbg[0] == '1') g_debug = 1;

    hgb_log("Initializing gfxGRAPH v%d.%d.%d",
            HGB_VERSION_MAJOR, HGB_VERSION_MINOR, HGB_VERSION_PATCH);

    hipError_t err = hipGetDeviceProperties(&g_dev_props, 0);
    if (err != hipSuccess) return err;

    err = hgb_validate_gfx1030();
    if (err != hipSuccess) {
        hgb_log("WARNING: not gfx1030 — bridge may not work correctly");
        /* Continue anyway — some bridges work on other RDNA */
    }

    hgb_log("GPU: %s (%s)", g_dev_props.name, g_dev_props.gcnArchName);
    hgb_log("Compute units: %d, Clock: %d MHz",
            g_dev_props.multiProcessorCount, g_dev_props.clockRate / 1000);

    g_initialized = 1;
    return hipSuccess;
}

void hgb_shutdown(void) {
    if (!g_initialized) return;
    hgb_log("Shutting down gfxGRAPH");
    g_initialized = 0;
}

void hgb_set_debug(int enabled) {
    g_debug = enabled;
}
