cmake_minimum_required(VERSION 3.17)

project(tritonopenpplbackend LANGUAGES C CXX)

# --------------------------------------------------------------------------- #

# options

option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

# --------------------------------------------------------------------------- #

# variables

set(TRITON_BUILD_CONTAINER "" CACHE STRING "Triton container to use a base for build")
set(TRITON_BUILD_CONTAINER_VERSION "" CACHE STRING "Triton container version to target")
set(TRITON_BUILD_OPENPPL_VERSION "" CACHE STRING "OpenPPL version to build")
set(TRITON_BUILD_CUDA_VERSION "" CACHE STRING "Version of CUDA install")
set(TRITON_BUILD_CUDA_HOME "" CACHE PATH "Path to CUDA install")
set(TRITON_BUILD_CUDNN_HOME "" CACHE PATH "Path to CUDNN install")
set(TRITON_BUILD_TENSORRT_HOME "" CACHE PATH "Path to TensorRT install")
set(TRITON_ONNX_TENSORRT_REPO_TAG "" CACHE STRING "Tag for onnx-tensorrt repo")
set(TRT_VERSION "" CACHE STRING "TRT version for this build.")
set(TRITON_OPENPPL_LIB_PATHS "" CACHE PATH "Paths to OpenPPL libraries")

set(TRITON_BACKEND_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/common repo")
set(OPENPPL_REPO_TAG "r21.12" CACHE STRING "Tag for openppl repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --------------------------------------------------------------------------- #

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY git@github.com:triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY git@github.com:triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY git@github.com:triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

FetchContent_Declare(
  openppl
  GIT_REPOSITORY git@github.com:openppl-public/ppl.nn.git
  GIT_TAG ${OPENPPL_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(openppl)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_openppl.ldscript libtriton_openppl.ldscript COPYONLY)

add_library(
  triton-openppl-backend SHARED
  src/openppl.cc
  src/openppl_utils.cc
  src/openppl_utils.h
)

add_library(
  TritonOpenPPLBackend::triton-openppl-backend ALIAS triton-openppl-backend
)

target_include_directories(
  triton-openppl-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PPLNN_INCLUDE_DIRECTORIES}
)

target_compile_features(triton-openppl-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-openppl-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-openppl-backend
    PRIVATE TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU
if(${TRITON_ENABLE_OPENPPL_TENSORRT})
  target_compile_definitions(
    triton-openppl-backend
    PRIVATE TRITON_ENABLE_OPENPPL_TENSORRT=1
  )
endif() # TRITON_ENABLE_OPENPPL_TENSORRT


FOREACH(p ${TRITON_OPENPPL_LIB_PATHS})
  target_link_directories(
    triton-openppl-backend
    PRIVATE ${p}
  )
ENDFOREACH(p)

target_link_libraries(
  triton-openppl-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    pplnn_static            # add openppl lib path
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-openppl-backend
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU


if(WIN32)
  set_target_properties(
    triton-openppl-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_openppl
  )
else()
  set_target_properties(
    triton-openppl-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_openppl
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_openppl.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_openppl.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonOpenPPLBackend)

install(
  TARGETS
    triton-openppl-backend
  EXPORT
    triton-openppl-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/openppl
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/openppl
)

install(
  EXPORT
    triton-openppl-backend-targets
  FILE
    TritonOpenPPLBackendTargets.cmake
  NAMESPACE
    TritonOpenPPLBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonOpenPPLBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenPPLBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenPPLBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-openppl-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenPPLBackendTargets.cmake
  NAMESPACE TritonOpenPPLBackend::
)

export(PACKAGE TritonOpenPPLBackend)