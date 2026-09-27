# OpenDLSS-NR MetalFX — top-level build.
# Builds the platform-neutral core + CPU reference backend on every OS, plus
# the platform GPU backend (Metal on macOS, D3D12 on Windows, Vulkan on Linux).
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/opendlss-nr info
#
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
cmake_minimum_required(VERSION 3.20)
project(opendlss-nr-metalfx VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(OPENDLSS_BUILD_TESTS "build the test suite" ON)
option(OPENDLSS_BUILD_CLI "build the opendlss-nr CLI" ON)
option(OPENDLSS_BUILD_GPU_BACKENDS "build the platform GPU backend" ON)

# ---------------- core library (always) ----------------
add_library(opendlss_core STATIC
    core/src/geometry.cpp
    core/src/logging.cpp
    core/src/image.cpp
    core/src/model.cpp
    core/src/pipeline.cpp
    core/src/pipeline_jobs.cpp
    core/backends/cpu/cpu_backend.cpp
    core/backends/cpu/neural_graph_cpu.cpp
)
target_include_directories(opendlss_core PUBLIC core/include)
target_compile_definitions(opendlss_core PUBLIC $<$<BOOL:${WIN32}>:NOMINMAX>)

# ---------------- GPU backends ----------------
if(APPLE AND OPENDLSS_BUILD_GPU_BACKENDS)
    enable_language(OBJC OBJCXX)
    add_library(opendlss_metal STATIC
        platforms/macos/src/metal_backend.mm
    )
    target_include_directories(opendlss_metal PUBLIC core/include platforms/macos/include)
    target_compile_definitions(opendlss_metal PUBLIC OPENDLSS_HAS_METAL
        OPENDLSS_METAL_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/platforms/macos/shaders")
    target_link_libraries(opendlss_metal PUBLIC
        "-framework Metal" "-framework MetalFX" "-framework Foundation"
        "-framework AppKit" "-framework MetalKit")
    target_compile_options(opendlss_metal PUBLIC -fobjc-arc)
endif()

if(WIN32 AND OPENDLSS_BUILD_GPU_BACKENDS)
    add_library(opendlss_d3d12 STATIC
        platforms/windows/src/d3d12_backend.cpp
    )
    target_include_directories(opendlss_d3d12 PUBLIC core/include)
    target_compile_definitions(opendlss_d3d12 PUBLIC OPENDLSS_HAS_D3D12 NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(opendlss_d3d12 PUBLIC d3d12 dxgi d3dcompiler)
endif()

if(UNIX AND NOT APPLE AND OPENDLSS_BUILD_GPU_BACKENDS)
    add_library(opendlss_vulkan STATIC
        platforms/linux/src/vulkan_backend.cpp
    )
    target_include_directories(opendlss_vulkan PUBLIC core/include platforms/linux/include)
    target_compile_definitions(opendlss_vulkan PUBLIC OPENDLSS_HAS_VULKAN)
    target_link_libraries(opendlss_vulkan PUBLIC ${CMAKE_DL_LIBS} pthread)

    # compile GLSL -> SPIR-V at build time when glslangValidator is available
    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(GLSLANG_VALIDATOR)
        file(GLOB ODL_COMP_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/platforms/linux/shaders/*.comp)
        set(ODL_SPV_OUTPUTS "")
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders_spv)
        foreach(comp ${ODL_COMP_SHADERS})
            get_filename_component(name ${comp} NAME_WE)
            set(spv ${CMAKE_BINARY_DIR}/shaders_spv/${name}.spv)
            add_custom_command(OUTPUT ${spv}
                COMMAND ${GLSLANG_VALIDATOR} -V -o ${spv} ${comp}
                DEPENDS ${comp}
                COMMENT "glslang ${name}")
            list(APPEND ODL_SPV_OUTPUTS ${spv})
        endforeach()
        add_custom_target(odl_shaders ALL DEPENDS ${ODL_SPV_OUTPUTS})
        add_dependencies(opendlss_vulkan odl_shaders)
    else()
        message(WARNING "glslangValidator not found: Vulkan shaders must be compiled manually")
    endif()
endif()

# ---------------- SDK (C ABI for engines) ----------------
add_library(opendlss_nr_sdk SHARED
    sdk/src/opendlss_nr_sdk.cpp
)
target_include_directories(opendlss_nr_sdk PUBLIC sdk/include)
target_link_libraries(opendlss_nr_sdk PRIVATE opendlss_core)
set_target_properties(opendlss_nr_sdk PROPERTIES OUTPUT_NAME opendlss_nr_sdk)

# ---------------- CLI ----------------
if(OPENDLSS_BUILD_CLI)
    add_executable(opendlss-nr apps/cli/main.cpp third_party/stb_impl.cpp)
    target_link_libraries(opendlss-nr PRIVATE opendlss_core)
    if(TARGET opendlss_metal)
        target_link_libraries(opendlss-nr PRIVATE opendlss_metal)
    endif()
    if(TARGET opendlss_d3d12)
        target_link_libraries(opendlss-nr PRIVATE opendlss_d3d12)
    endif()
    if(TARGET opendlss_vulkan)
        target_link_libraries(opendlss-nr PRIVATE opendlss_vulkan)
        if(GLSLANG_VALIDATOR)
            add_custom_command(TARGET opendlss-nr POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                        ${CMAKE_BINARY_DIR}/shaders_spv $<TARGET_FILE_DIR:opendlss-nr>/shaders_spv)
        endif()
    endif()
endif()

# ---------------- tests ----------------
if(OPENDLSS_BUILD_TESTS)
    enable_testing()
    add_executable(test_core tests/test_core.cpp third_party/stb_impl.cpp)
    target_link_libraries(test_core PRIVATE opendlss_core)
    add_test(NAME core_tests COMMAND test_core)
    set_tests_properties(core_tests PROPERTIES ENVIRONMENT "OPENDLSS_TEST_MODEL=${CMAKE_SOURCE_DIR}/models/demo-nr")
endif()
