# cmake/vulkan-deps.cmake — Vulkan dependencies for the Windows/Linux backend.
#
# The original project downloads a portable toolchain (glslang, Vulkan-Headers,
# volk) into tools/ (scripts/windows/fetch_tools.ps1, scripts/common/fetch_tools.sh);
# the build consumes either that tree or a system Vulkan SDK.
#
# Set OPENDDLSS_VULKAN_DEPS to the tools/ directory (default: repo-root/tools).

set(OPENDDLSS_VULKAN_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/tools" CACHE PATH "Portable Vulkan deps root")

set(VULKAN_DEP_INCLUDES "")
set(VULKAN_DEP_LIBS "")

if (EXISTS "${OPENDDLSS_VULKAN_DEPS}/Vulkan-Headers/include/vulkan/vulkan.h")
  list(APPEND VULKAN_DEP_INCLUDES "${OPENDDLSS_VULKAN_DEPS}/Vulkan-Headers/include")
  list(APPEND VULKAN_DEP_INCLUDES "${OPENDDLSS_VULKAN_DEPS}/volk")
  # volk.c compiles against its own volk.h; the include order above covers both.
  if (WIN32)
    list(APPEND VULKAN_DEP_LIBS "ws2_32")   # volk linkage on some toolchains
  else()
    list(APPEND VULKAN_DEP_LIBS "dl")
  endif()
  message(STATUS "Vulkan deps: portable tree at ${OPENDDLASS_VULKAN_DEPS}")
else()
  find_package(Vulkan QUIET)
  if (Vulkan_FOUND)
    list(APPEND VULKAN_DEP_INCLUDES "${Vulkan_INCLUDE_DIRS}")
    if (TARGET Vulkan::volk)
      list(APPEND VULKAN_DEP_INCLUDES "${volk_INCLUDE_DIRS}")
    endif()
    message(STATUS "Vulkan deps: system SDK at ${Vulkan_INCLUDE_DIRS}")
  else()
    message(WARNING
      "No Vulkan headers found. Run scripts/common/fetch_tools.sh (or "
      "scripts/windows/fetch_tools.ps1 on Windows) to fill tools/, or install "
      "the Vulkan SDK. The Vulkan backend will not build.")
  endif()
endif()

# volk: compile the official volk.c (or a generated implementation TU as fallback).
# (volk is compiled by CMakeLists.txt as the `volk_impl` target, with the
# Vulkan-Headers include path ahead of volk's own directory.)
