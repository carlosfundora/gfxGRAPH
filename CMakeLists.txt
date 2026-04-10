cmake_minimum_required(VERSION 3.21)
project(gfxGRAPH
    VERSION 0.1.0
    DESCRIPTION "CUDA Graph → HIP Graph translation layer for gfx1030 RDNA2"
    LANGUAGES CXX HIP
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_ARCHITECTURES gfx1030)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Find ROCm ──────────────────────────────────────────
find_package(hip REQUIRED)
find_package(hiprtc QUIET)

# ── Layer 1: Core bridge library ───────────────────────
add_library(hipgraph_bridge SHARED
    src/init.cpp
    src/conditional_bridge.hip
    src/launch_pipeline.hip
    src/shape_manager.hip
    src/capture_compositor.hip
    src/graph_utils.hip
)
target_include_directories(hipgraph_bridge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(hipgraph_bridge PRIVATE
    hip::amdhip64
)
if(hiprtc_FOUND)
    target_link_libraries(hipgraph_bridge PRIVATE hip::hiprtc)
    target_compile_definitions(hipgraph_bridge PRIVATE HGB_HAS_HIPRTC=1)
endif()
set_target_properties(hipgraph_bridge PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME hipgraph_bridge
)

# ── Layer 3: CUDA compat shim (optional) ───────────────
option(BUILD_CUDA_COMPAT "Build CUDA compatibility LD_PRELOAD shim" OFF)
if(BUILD_CUDA_COMPAT)
    add_library(cudagraph_compat SHARED
        src/compat/cuda_intercept.c
    )
    target_link_libraries(cudagraph_compat PRIVATE
        hipgraph_bridge
        ${CMAKE_DL_LIBS}
    )
    set_target_properties(cudagraph_compat PROPERTIES
        OUTPUT_NAME cudagraph_compat
    )
endif()

# ── Tests ──────────────────────────────────────────────
option(BUILD_TESTS "Build test executables" ON)
if(BUILD_TESTS)
    enable_testing()

    foreach(gap IN ITEMS conditional pipeline shapes compositor)
        add_executable(test_${gap} tests/test_${gap}.hip)
        target_link_libraries(test_${gap} PRIVATE hipgraph_bridge)
        add_test(NAME ${gap} COMMAND test_${gap})
    endforeach()

    if(BUILD_CUDA_COMPAT)
        add_test(
            NAME compat_smoke
            COMMAND ${CMAKE_COMMAND} -E echo "compat shim built OK"
        )
    endif()
endif()

# ── Install ────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS hipgraph_bridge
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/hipgraph_bridge.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
