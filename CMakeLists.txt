cmake_minimum_required(VERSION 3.24)
project(OpenDLSS-NR-MetalFX VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(OPENDDLSS_WITH_VIDEO "FFmpeg video pipeline" ON)
option(OPENDDLSS_BUILD_TESTS "Unit tests" ON)
option(OPENDDLSS_BUILD_CLI "opendlss command line tool" ON)
option(OPENDDLSS_BUILD_DEMO "macOS realtime demo (Apple only)" ON)
option(OPENDDLSS_WARN_AS_ERROR "Treat warnings as errors" OFF)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Portable core (shared by every target)
# ---------------------------------------------------------------------------
set(NR_COMMON_SOURCES
  src/common/nr_frame.cpp
  src/common/nr_image.cpp
)

# ---------------------------------------------------------------------------
# Apple: Metal + MetalFX backend
# ---------------------------------------------------------------------------
if (APPLE)
  enable_language(OBJC OBJCXX)

  add_library(opendlss_metal STATIC
    src/common/nr_frame_metal.mm
    src/metal/nr_metal_context.mm
    src/metal/nr_metal_model.mm
    src/metal/nr_metal_kernels.mm
    src/metal/nr_metal_graph.mm
    src/metal/nr_metalfx.mm
  )
  target_include_directories(opendlss_metal PUBLIC src src/common)
  target_compile_definitions(opendlss_metal PUBLIC NR_FRAME_BACKEND_METAL)
  set_target_properties(opendlss_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_options(opendlss_metal PRIVATE -fobjc-arc -Wno-unused-function)
  target_link_libraries(opendlss_metal PUBLIC
    "-framework Metal" "-framework MetalKit" "-framework MetalFX" "-framework Foundation"
  )
endif()

# ---------------------------------------------------------------------------
# Windows / Linux: Vulkan backend (the original bit-exact route)
# ---------------------------------------------------------------------------
if (NOT APPLE)
  include(cmake/vulkan-deps.cmake)

  if (EXISTS "${OPENDDLSS_VULKAN_DEPS}/volk/volk.c")
    add_library(volk_impl STATIC "${OPENDDLSS_VULKAN_DEPS}/volk/volk.c")
    set_target_properties(volk_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(volk_impl PRIVATE ${VULKAN_DEP_INCLUDES})
    target_compile_definitions(volk_impl PRIVATE VK_ENABLE_BETA_EXTENSIONS)
    list(APPEND VULKAN_DEP_LIBS volk_impl)
  endif()

  add_library(opendlss_vk STATIC
    src/common/nr_frame_vulkan.cpp
    src/vulkan/vk_context.cpp
    src/vulkan/kernels.cpp
    src/vulkan/nr_graph.cpp
    src/vulkan/nr_model.cpp
    src/vulkan/reference.cpp
    src/vulkan/verify.cpp
  )
  set_target_properties(opendlss_vk PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(opendlss_vk PUBLIC src src/common src/vulkan ${VULKAN_DEP_INCLUDES})
  target_compile_definitions(opendlss_vk PUBLIC NR_FRAME_BACKEND_VULKAN VK_ENABLE_BETA_EXTENSIONS OPENDLSS_INSTALL_SHADERS="${CMAKE_INSTALL_PREFIX}/share/opendlss/shaders")
  target_link_libraries(opendlss_vk PUBLIC ${VULKAN_DEP_LIBS} Threads::Threads)
endif()

# ---------------------------------------------------------------------------
# libopendlss: the portable C API
# ---------------------------------------------------------------------------
add_library(opendlss SHARED src/sdk/opendlss.cpp src/common/nr_cpu_reference.cpp ${NR_COMMON_SOURCES})
target_include_directories(opendlss PUBLIC include src src/common third_party/stb)
set_target_properties(opendlss PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
if (APPLE)
  target_link_libraries(opendlss PUBLIC opendlss_metal)
else()
  target_link_libraries(opendlss PUBLIC opendlss_vk)
endif()
if (OPENDDLSS_WITH_VIDEO)
  find_package(PkgConfig QUIET)
  if (PkgConfig_FOUND)
    pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libswscale libavutil)
  endif()
  if (FFMPEG_FOUND)
    target_compile_definitions(opendlss PUBLIC NR_WITH_VIDEO)
    target_link_libraries(opendlss PUBLIC PkgConfig::FFMPEG)
  else()
    message(STATUS "FFmpeg not found: building without the video pipeline")
  endif()
endif()

# ---------------------------------------------------------------------------
# opendlss CLI
# ---------------------------------------------------------------------------
if (OPENDDLSS_BUILD_CLI)
  add_executable(opendlss-cli src/cli/opendlss_main.cpp src/common/nr_cpu_reference.cpp ${NR_COMMON_SOURCES})
  target_include_directories(opendlss-cli PRIVATE src src/common third_party/stb)
  if (APPLE)
    target_compile_definitions(opendlss-cli PRIVATE NR_FRAME_BACKEND_METAL)
    target_link_libraries(opendlss-cli PRIVATE opendlss_metal)
  else()
    target_compile_definitions(opendlss-cli PRIVATE NR_FRAME_BACKEND_VULKAN)
    target_link_libraries(opendlss-cli PRIVATE opendlss_vk)
  endif()
  if (FFMPEG_FOUND)
    target_compile_definitions(opendlss-cli PRIVATE NR_WITH_VIDEO)
    target_link_libraries(opendlss-cli PRIVATE PkgConfig::FFMPEG)
  endif()
  set_target_properties(opendlss-cli PROPERTIES OUTPUT_NAME opendlss)
endif()

# ---------------------------------------------------------------------------
# dlss5vk: the original Windows/Linux parity + verification tool (unchanged)
# ---------------------------------------------------------------------------
if (NOT APPLE)
  add_executable(dlss5vk src/vulkan/main.cpp)
  target_include_directories(dlss5vk PRIVATE src src/common src/vulkan)
  target_link_libraries(dlss5vk PRIVATE opendlss_vk)
endif()

# ---------------------------------------------------------------------------
# macOS realtime demo
# ---------------------------------------------------------------------------
if (APPLE AND OPENDDLSS_BUILD_DEMO)
  add_executable(demo-game-macos apps/demo-game-macos/main.mm)
  target_include_directories(demo-game-macos PRIVATE src src/common)
  target_compile_options(demo-game-macos PRIVATE -fobjc-arc)
  target_link_libraries(demo-game-macos PRIVATE opendlss_metal
    "-framework AppKit" "-framework MetalKit" "-framework MetalFX")
endif()

# ---------------------------------------------------------------------------
# Tests (portable; run everywhere including CI macOS arm64)
# ---------------------------------------------------------------------------
if (OPENDDLSS_BUILD_TESTS)
  add_executable(opendlss-tests
    tests/test_main.cpp
    tests/test_numeric.cpp
    tests/test_reference.cpp
    tests/test_geometry.cpp
    src/common/nr_cpu_reference.cpp
  )
  target_include_directories(opendlss-tests PRIVATE src src/common)
  if (APPLE)
    target_link_libraries(opendlss-tests PRIVATE opendlss_metal)
  else()
    target_link_libraries(opendlss-tests PRIVATE opendlss_vk)
  endif()
  enable_testing()
  add_test(NAME opendlss-tests COMMAND opendlss-tests)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS opendlss RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
if (OPENDDLSS_BUILD_CLI)
  install(TARGETS opendlss-cli RUNTIME DESTINATION bin)
endif()
install(FILES include/opendlss/opendlss.h DESTINATION include/opendlss)
if (NOT APPLE)
  install(TARGETS dlss5vk RUNTIME DESTINATION bin)
  install(DIRECTORY shaders/vulkan-glsl/ DESTINATION share/opendlss/shaders
          FILES_MATCHING PATTERN "*.spv")
endif()
if (APPLE)
  install(FILES ${CMAKE_BINARY_DIR}/opendlss.metallib DESTINATION share/opendlss OPTIONAL)
endif()
