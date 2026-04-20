/**
 * @file init.cpp
 * @brief gfxGRAPH lifecycle: init, shutdown, version, validation
 */
#include "hipgraph_bridge.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <atomic>
#include <mutex>

static std::atomic<int> g_initialized{0};
static std::atomic<int> g_debug{0};
static std::mutex       g_init_mutex;
static hipDeviceProp_t  g_dev_props;
static int              g_device_id = -1;
static int              g_rocm_runtime_version = 0;

extern "C" void hgb_log(const char* fmt, ...) {
    if (!g_debug.load(std::memory_order_relaxed)) return;
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[gfxGRAPH] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

extern "C" HGB_EXPORT int hgb_is_initialized(void) {
    return g_initialized.load(std::memory_order_acquire);
}

extern "C" HGB_EXPORT hgb_version_t hgb_get_version(void) {
    hgb_version_t v;
    v.major = HGB_VERSION_MAJOR;
    v.minor = HGB_VERSION_MINOR;
    v.patch = HGB_VERSION_PATCH;
    v.gfx_target = "gfx1030";

    /* Detect ROCm version at runtime */
    static char rocm_ver_str[32] = "unknown";
    if (g_rocm_runtime_version > 0) {
        int major = g_rocm_runtime_version / 10000000;
        int minor = (g_rocm_runtime_version / 100000) % 100;
        int patch = g_rocm_runtime_version % 100000;
        snprintf(rocm_ver_str, sizeof(rocm_ver_str), "%d.%d.%d", major, minor, patch);
    }
    v.rocm_version = rocm_ver_str;
    return v;
}

extern "C" HGB_EXPORT hipError_t hgb_validate_gfx1030(void) {
    int dev;
    hipError_t err = hipGetDevice(&dev);
    if (err != hipSuccess) return err;

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, dev);
    if (err != hipSuccess) return err;

    if (strstr(props.gcnArchName, "gfx1030") == nullptr) {
        hgb_log("ERROR: expected gfx1030, got %s", props.gcnArchName);
        return hipErrorInvalidDevice;
    }
    return hipSuccess;
}

static hipError_t hgb_init_impl(void) {
    /* Check env for debug flag */
    const char* dbg = getenv("HGB_DEBUG");
    if (dbg && (dbg[0] == '1' || strcmp(dbg, "debug") == 0))
        g_debug.store(1, std::memory_order_relaxed);

    /* Also check GFXGRAPH env for debug mode */
    const char* gfx = getenv("GFXGRAPH");
    if (gfx && strcmp(gfx, "debug") == 0)
        g_debug.store(1, std::memory_order_relaxed);

    hgb_log("Initializing gfxGRAPH v%d.%d.%d",
            HGB_VERSION_MAJOR, HGB_VERSION_MINOR, HGB_VERSION_PATCH);

    /* Get current device (not hardcoded 0) */
    hipError_t err = hipGetDevice(&g_device_id);
    if (err != hipSuccess) return err;

    err = hipGetDeviceProperties(&g_dev_props, g_device_id);
    if (err != hipSuccess) return err;

    /* Runtime ROCm version detection */
    hipRuntimeGetVersion(&g_rocm_runtime_version);
    hgb_log("ROCm runtime version: %d", g_rocm_runtime_version);

    err = hgb_validate_gfx1030();
    if (err != hipSuccess) {
        hgb_log("WARNING: not gfx1030 — bridge may not work correctly");
        /* Continue anyway — some bridges work on other RDNA */
    }

    hgb_log("GPU: %s (%s) on device %d",
            g_dev_props.name, g_dev_props.gcnArchName, g_device_id);
    hgb_log("Compute units: %d, Clock: %d MHz",
            g_dev_props.multiProcessorCount, g_dev_props.clockRate / 1000);

    g_initialized.store(1, std::memory_order_release);
    return hipSuccess;
}

extern "C" HGB_EXPORT hipError_t hgb_init(void) {
    if (g_initialized.load(std::memory_order_acquire)) return hipSuccess;

    std::lock_guard<std::mutex> lock(g_init_mutex);

    /* Double-check after acquiring lock */
    if (g_initialized.load(std::memory_order_acquire)) return hipSuccess;

    return hgb_init_impl();
}

extern "C" HGB_EXPORT void hgb_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (!g_initialized.load(std::memory_order_acquire)) return;
    hgb_log("Shutting down gfxGRAPH");
    g_initialized.store(0, std::memory_order_release);
    g_device_id = -1;
}

extern "C" HGB_EXPORT void hgb_set_debug(int enabled) {
    g_debug.store(enabled, std::memory_order_relaxed);
}
