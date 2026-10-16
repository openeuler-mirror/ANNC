cmake_minimum_required(VERSION 3.21)

project(annc LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
option(ANNC_ENABLE_CONSTANT_FOLDING "Enable ANNC constant-folding canonicalization feature" OFF)
option(ANNC_ENABLE_COVERAGE "Enable source-based code coverage (clang only)" OFF)
option(ANNC_DEBUG "Build only ANNC-owned code with debug flags (-g3 -O0 -UNDEBUG); third-party build type unchanged" OFF)

# Framework frontends and their addons. A framework switch gates its frontend
# tool, its proto preparation and its addon; an addon switch is only meaningful
# while its framework switch is on. The defaults keep the historical build:
# TensorFlow tools + addons and the ONNX frontend are built, nothing else is
# required. build.sh mirrors these options (--disable-tensorflow,
# --disable-tensorflow-addon, --disable-onnx).
option(ANNC_ENABLE_TENSORFLOW "Build the TensorFlow frontend tools (annc-tf2atir, annc-converter, annc-tf-pipeline)" ON)
option(ANNC_ENABLE_TENSORFLOW_ADDON "Build the TensorFlow addons (annc_optimizer, annc_fused_op); requires ANNC_ENABLE_TENSORFLOW" ${ANNC_ENABLE_TENSORFLOW})
option(ANNC_ENABLE_ONNX "Build the ONNX frontend (annc-onnx2atir) and its protobuf sources" ON)
# The ONNX Runtime EP plugin is opt-in: it is compiled against an external ORT
# SDK that the default build neither needs nor downloads.
option(ANNC_ENABLE_ONNXRUNTIME_EP "Build the ANNC ONNX Runtime execution-provider plugin (addons/onnxruntime); requires ANNC_ENABLE_ONNX and ANNC_ONNXRUNTIME_INCLUDE_DIR" OFF)
set(ANNC_ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "Directory holding onnxruntime_c_api.h and onnxruntime_ep_c_api.h, e.g. <ORT_ROOT>/include/onnxruntime/core/session (required by ANNC_ENABLE_ONNXRUNTIME_EP)")
option(ANNC_EP_NATIVE_POOL "Back the EP plugin's thread pool with ORT's native operator pool, so the scheduling decision is ORT's own (private ABI: verified context layout + ORT build-tree archives; see addons/onnxruntime/annc_ep_native_pool.h); requires ANNC_EP_ORT_BUILD_TREE" OFF)
set(ANNC_EP_ORT_BUILD_TREE "" CACHE PATH "ORT build tree (the Release/ directory of the ORT build that will host the plugin, e.g. <ORT_WS>/build_wheel_fullstack/Release) providing libonnxruntime_common.a and libcpuinfo.a for ANNC_EP_NATIVE_POOL; an install/SDK tree does not carry them")
set(ANNC_EP_NATIVE_POOL_ORT_VERSION "1.26.0" CACHE STRING "ORT release the native-pool context layout and archives were verified against; the adapter refuses the native path in any other host and falls back to the public KernelContext_ParallelFor adapter")
# MLAS GEMM backend for the builtin kernels: a kernel whose host hands over its
# native pool (AnncThreadPool::native_pool; the ORT EP does) may pass it
# straight to MlasGemmBatch.  Opt-in like the EP switch: it needs the ORT
# source tree's mlas.h to compile and the ORT build tree's archives to link
# into the kernel shared libraries the annc driver produces.
option(ANNC_KERNEL_MLAS "Let builtin kernels call MLAS GEMM entries with the host's native pool (requires ANNC_MLAS_INCLUDE_DIR and ANNC_MLAS_BUILD_TREE; pair with ANNC_EP_NATIVE_POOL)" OFF)
set(ANNC_MLAS_INCLUDE_DIR "" CACHE PATH "ORT source tree's onnxruntime/core/mlas/inc directory (holding mlas.h), required by ANNC_KERNEL_MLAS")
set(ANNC_MLAS_BUILD_TREE "" CACHE PATH "ORT build tree (the Release/ directory of the ORT build that will load the kernels) providing libonnxruntime_mlas.a, libkleidiai.a, libonnxruntime_common.a and libcpuinfo.a for ANNC_KERNEL_MLAS")

if(ANNC_ENABLE_TENSORFLOW_ADDON AND NOT ANNC_ENABLE_TENSORFLOW)
  message(FATAL_ERROR
    "ANNC_ENABLE_TENSORFLOW_ADDON=ON requires ANNC_ENABLE_TENSORFLOW=ON: the "
    "TensorFlow addons are built from the TensorFlow frontend toolchain. Set "
    "ANNC_ENABLE_TENSORFLOW_ADDON=OFF or ANNC_ENABLE_TENSORFLOW=ON.")
endif()

if(ANNC_ENABLE_ONNXRUNTIME_EP AND NOT ANNC_ENABLE_ONNX)
  message(FATAL_ERROR
    "ANNC_ENABLE_ONNXRUNTIME_EP=ON requires ANNC_ENABLE_ONNX=ON: the plugin "
    "loads the AOT kernels produced through the ONNX frontend. Set "
    "ANNC_ENABLE_ONNXRUNTIME_EP=OFF or ANNC_ENABLE_ONNX=ON.")
endif()

if(ANNC_ENABLE_ONNXRUNTIME_EP)
  if(ANNC_ONNXRUNTIME_INCLUDE_DIR STREQUAL "")
    message(FATAL_ERROR
      "ANNC_ENABLE_ONNXRUNTIME_EP=ON requires ANNC_ONNXRUNTIME_INCLUDE_DIR to "
      "point at the ORT 1.26.0 session headers, e.g. "
      "<ORT_ROOT>/include/onnxruntime/core/session. The SDK is never downloaded "
      "automatically.")
  endif()
  if(NOT EXISTS "${ANNC_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" OR
     NOT EXISTS "${ANNC_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_ep_c_api.h")
    message(FATAL_ERROR
      "ANNC_ONNXRUNTIME_INCLUDE_DIR='${ANNC_ONNXRUNTIME_INCLUDE_DIR}' does not "
      "contain onnxruntime_c_api.h and onnxruntime_ep_c_api.h. Point it at the "
      "ORT 1.26.0 include/onnxruntime/core/session directory.")
  endif()
endif()

if(ANNC_KERNEL_MLAS)
  if(ANNC_MLAS_INCLUDE_DIR STREQUAL "" OR
     NOT EXISTS "${ANNC_MLAS_INCLUDE_DIR}/mlas.h")
    message(FATAL_ERROR
      "ANNC_KERNEL_MLAS=ON requires ANNC_MLAS_INCLUDE_DIR to point at the ORT "
      "source tree's onnxruntime/core/mlas/inc directory (holding mlas.h).")
  endif()
  set(ANNC_MLAS_ARCHIVES
      "${ANNC_MLAS_BUILD_TREE}/libonnxruntime_mlas.a"
      "${ANNC_MLAS_BUILD_TREE}/_deps/kleidiai-build/libkleidiai.a"
      "${ANNC_MLAS_BUILD_TREE}/libonnxruntime_common.a"
      "${ANNC_MLAS_BUILD_TREE}/_deps/pytorch_cpuinfo-build/libcpuinfo.a")
  foreach(archive IN LISTS ANNC_MLAS_ARCHIVES)
    if(NOT EXISTS "${archive}")
      message(FATAL_ERROR
        "ANNC_KERNEL_MLAS=ON needs ${archive}: point ANNC_MLAS_BUILD_TREE at "
        "the ORT build tree (the Release/ directory, not the install tree) of "
        "the runtime that will load the kernels.")
    endif()
  endforeach()
  # The annc driver bakes these paths into every kernel shared-library link,
  # so the archives need not live in the single -L directory the driver is
  # given (ANNC_KERNEL_LIB_PATH).
  string(REPLACE ";" " " ANNC_MLAS_ARCHIVE_ARGS "${ANNC_MLAS_ARCHIVES}")
endif()
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# set(ABSL_PROPAGATE_CXX_STD ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_warn_duplicate_libraries")

include(FetchContent)

# nlohmann/json v3.11.3 — header-only JSON library (MIT)
FetchContent_Declare(json
  GIT_REPOSITORY https://gitee.com/mirrors/nlohmann-json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
  SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third_party/json
)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/json/CMakeLists.txt")
    message(STATUS "nlohmann/json source already exists, skipping fetch")
    set(json_POPULATED ON CACHE INTERNAL "json populated" FORCE)
    set(json_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/json" CACHE INTERNAL "json source dir" FORCE)
    set(json_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/json-build" CACHE INTERNAL "json binary dir" FORCE)
  else()
    message(STATUS "Fetching nlohmann/json from gitee.com/mirrors/nlohmann-json.git (v3.11.3) ...")
    # TODO: CMake 3.29+ 使用 FetchContent_MakeAvailable 替代 FetchContent_Populate
    FetchContent_Populate(json)
  endif()
endif()
add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR})
# nlohmann_json is referenced by libraries in the MLIR export set
install(TARGETS nlohmann_json EXPORT MLIRTargets)
include(${CMAKE_SOURCE_DIR}/cmake/ThirdPartyKDNN.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/ThirdPartyGemmKernels.cmake)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable assertions")

# Fetch LLVM/MLIR source at configure time

# LLVM 裁剪选项 — 只编译 AArch64 后端 + MLIR
set(LLVM_TARGETS_TO_BUILD "AArch64" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "mlir;clang" CACHE STRING "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_EH ON CACHE BOOL "")
set(LLVM_ENABLE_WERROR OFF CACHE BOOL "")
set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")

FetchContent_Declare(llvm
  GIT_REPOSITORY https://gitee.com/mirrors/LLVM.git
  GIT_TAG        llvmorg-21.1.3
  GIT_SHALLOW    TRUE
  SOURCE_DIR     ${CMAKE_SOURCE_DIR}/third_party/llvm
)
FetchContent_GetProperties(llvm)
if(NOT llvm_POPULATED)
  if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/llvm/llvm/CMakeLists.txt")
    message(STATUS "LLVM source already exists, skipping fetch")
    set(llvm_POPULATED ON CACHE INTERNAL "LLVM populated" FORCE)
    set(llvm_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm" CACHE INTERNAL "LLVM source dir" FORCE)
    set(llvm_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/llvm-build" CACHE INTERNAL "LLVM binary dir" FORCE)
  else()
    message(STATUS "Fetching LLVM/MLIR from gitee.com/mirrors/LLVM.git (llvmorg-21.1.3) ...")
    # TODO: CMake 3.29+ 使用 FetchContent_MakeAvailable 替代 FetchContent_Populate
    FetchContent_Populate(llvm)
  endif()
endif()

# CMake 3.22 FindPython 规避：本工程一次 configure 内混合了多种 Python 组件组合
# （MLIR 的 Development.Module-only、mlir/python 的 Module+Embed、ANNC 顶层的全量
# Development），后一次 find 会改写/清空 _Python_DEVELOPMENT_*_SIGNATURE，导致下一次
# configure 时 MLIR 的 Module-only 检测签名失配，报
# "Could NOT find Python (missing: Development.Module)"。
# 在进入 LLVM 子构建前清空上一轮残留的 FindPython 内部状态，使每次 configure 都从
# 全新发现开始（解释器可执行文件仍由 Python_EXECUTABLE/Python3_EXECUTABLE 缓存锚定）。
get_cmake_property(_annc_py_cache_vars CACHE_VARIABLES)
foreach(_v IN LISTS _annc_py_cache_vars)
  if(_v MATCHES "^_Python3?_")
    unset(${_v} CACHE)
  endif()
endforeach()
unset(_annc_py_cache_vars)

# LLVM monorepo: CMakeLists.txt 入口在 llvm/ 子目录
add_subdirectory(${llvm_SOURCE_DIR}/llvm ${llvm_BINARY_DIR})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

# LLVM/MLIR cmake 模块路径不能通过 add_subdirectory 自动传播到顶层作用域
list(APPEND CMAKE_MODULE_PATH "${llvm_SOURCE_DIR}/llvm/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${llvm_SOURCE_DIR}/mlir/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# LLVM_INCLUDE_DIRS/MLIR_INCLUDE_DIRS 只在安装配置中存在, 构建树中需显式指定
include_directories(${llvm_SOURCE_DIR}/llvm/include)
include_directories(${llvm_BINARY_DIR}/include)
include_directories(${llvm_SOURCE_DIR}/mlir/include)
include_directories(${llvm_BINARY_DIR}/tools/mlir/include)

# 供下游工具的 find_package(MLIR) 使用 (build-tree MLIRConfig.cmake)
set(MLIR_DIR "${CMAKE_BINARY_DIR}/lib/cmake/mlir")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/annc/include)
include_directories(${CMAKE_BINARY_DIR}/annc/include)

include_directories(${json_SOURCE_DIR}/include)

# Keep ANNC-owned debug information relocatable in build/install artifacts.
# This is deliberately added after the LLVM sub-build so third-party LLVM is
# not rebuilt just to remap its own source paths.
add_compile_options(
  "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
  "-fdebug-prefix-map=${CMAKE_SOURCE_DIR}=."
  "-fdebug-prefix-map=${CMAKE_BINARY_DIR}=./build"
)

# Code coverage via gcov (--coverage, portable across gcc/clang). Set AFTER the
# LLVM/MLIR sub-build so LLVM is neither recompiled nor relinked with --coverage;
# the options apply only to directories processed afterwards (annc/, python/,
# frontends/, tests/, and the enabled addons). Both compile and link flags are required: compile flags
# instrument objects (referencing __gcov_* symbols), link flags pull in the gcov
# runtime that resolves them — so any ANNC executable linking an instrumented
# object (e.g. libANNCKernel) must also get the link flag, which the global
# scope guarantees. Toggling coverage recompiles ANNC sources only, not LLVM.
if(ANNC_ENABLE_COVERAGE)
  add_compile_options(--coverage)
  add_link_options(--coverage)
  message(STATUS "ANNC code coverage enabled (--coverage / gcov).")
endif()

# ANNC-scoped debug mode (--annc-debug). Set AFTER the LLVM/MLIR sub-build, for
# the same reason as coverage above: directory-level compile options apply only
# to targets defined afterwards (annc/, python/, frontends/, tensorflow_addons/,
# addons/onnxruntime/, tests/), so third-party LLVM/json compile commands are untouched and ninja does not
# rebuild them. The flags are appended at the end of the compile command, so
# -O0 overrides the Release -O3 and -UNDEBUG re-enables assert() in ANNC code
# even when CMAKE_BUILD_TYPE=Release.
if(ANNC_DEBUG)
  add_compile_options(-g3 -O0 -fno-omit-frame-pointer -UNDEBUG)
  message(STATUS "ANNC debug mode enabled: ANNC-only -g3 -O0 -UNDEBUG; third-party build type unchanged.")
endif()

include(CTest)

add_subdirectory(annc)
add_subdirectory(python)
# TensorFlow configuration is owned by the addon subdirectory alone, so it is
# entered only when the addon is enabled: with the addon off no TensorFlow
# headers or OpenSSL are looked up.
if(ANNC_ENABLE_TENSORFLOW_ADDON)
  add_subdirectory(tensorflow_addons)
endif()
# Framework frontends, registered after the ANNC libraries they link against.
# frontends/CMakeLists.txt applies the per-framework switches.
add_subdirectory(frontends)
# Optional ORT EP plugin; entered only when explicitly enabled (the option
# block above already rejected a missing ONNX parent or ORT SDK path).
if(ANNC_ENABLE_ONNXRUNTIME_EP)
  add_subdirectory(addons/onnxruntime)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
