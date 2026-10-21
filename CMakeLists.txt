cmake_minimum_required(VERSION 3.21)

project(annc LANGUAGES CXX C)

include(GNUInstallDirs)

# Python bindings are off by default: the inference acceleration path (TF
# addons, ORT EP, AOT kernels) is pure C++ and does not load them. Enable with
# -DANNC_ENABLE_PYTHON=ON (build.sh --enable-python); this pulls in pybind11,
# nanobind, FindPython and the python/ subdirectory.
option(ANNC_ENABLE_PYTHON "Build the ANNC Python bindings" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
option(ANNC_ENABLE_CONSTANT_FOLDING "Enable ANNC constant-folding canonicalization feature" OFF)
option(ANNC_ENABLE_COVERAGE "Enable gcov source coverage instrumentation" OFF)
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
set(ANNC_EP_ORT_BUILD_TREE "" CACHE PATH "ORT build tree (the Release/ directory of the ORT build that will host the plugin, e.g. <ORT_WS>/build_wheel_fullstack/Release) providing libonnxruntime_common.a and libcpuinfo.a for the EP's native pool (always used by the EP); an install/SDK tree does not carry them")
set(ANNC_EP_NATIVE_POOL_ORT_VERSION "1.26.0" CACHE STRING "ORT release the native-pool context layout and archives were verified against; the adapter refuses the native path in any other host and falls back to the public KernelContext_ParallelFor adapter")
# MLAS GEMM backend for the builtin kernels: a kernel whose host hands over its
# native pool (AnncThreadPool::native_pool; the ORT EP does) may pass it
# straight to MlasGemmBatch.  It needs the ORT source tree's mlas.h to compile;
# the annc driver resolves the four ORT archives at link time from
# ANNC_EP_ORT_BUILD_TREE.
option(ANNC_KERNEL_MLAS "Let builtin kernels call MLAS GEMM entries with the host's native pool (requires ANNC_MLAS_INCLUDE_DIR; runtime archives resolved via ANNC_EP_ORT_BUILD_TREE)" OFF)
set(ANNC_MLAS_INCLUDE_DIR "" CACHE PATH "ORT source tree's onnxruntime/core/mlas/inc directory (holding mlas.h), required by ANNC_KERNEL_MLAS")
# The ORT EP always uses the MLAS-backed kernel path; enabling the EP enables
# the MLAS backend automatically.  TF-only builds keep it off unless the user
# explicitly turns it on.
if(ANNC_ENABLE_ONNXRUNTIME_EP)
  set(ANNC_KERNEL_MLAS ON CACHE BOOL
      "Let builtin kernels call MLAS GEMM entries with the host's native pool (automatically ON when ANNC_ENABLE_ONNXRUNTIME_EP=ON)" FORCE)
endif()
# The annc driver assembles generated AOT/JIT GEMM kernels against the GEMM-ASM
# headers in <checkout>/Sve and <checkout>/Neon.  Baked in here as the default;
# the runtime ANNC_GEMM_ASM_SOURCE_DIR environment variable overrides it.
set(ANNC_GEMM_ASM_SOURCE_DIR "" CACHE PATH "GEMM-ASM checkout holding Sve/ and Neon/ assembly headers, used by the annc driver when ANNC_GEMM_ASM_SOURCE_DIR is not set at runtime")

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
include(${CMAKE_SOURCE_DIR}/cmake/ThirdPartyKDNN.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/ThirdPartyGemmKernels.cmake)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# ── LLVM/MLIR 来源 ──────────────────────────────────────────────────────────
# LLVM/MLIR 由外部工具链提供，通过 find_package 使用预装工具链与库；ANNC 不再
# 在 configure 期下载或从源码编译 LLVM。**不锁定具体版本**：外部可传入任意与
# ANNC API 兼容的 LLVM/MLIR（本机验证版本为 openEuler llvm-toolset-20 20.1.8）。
# openEuler 24.03 SP3 及以后可用的安装命令：
#   sudo dnf install -y llvm-toolset-20-llvm llvm-toolset-20-llvm-devel \
#     llvm-toolset-20-llvm-libs llvm-toolset-20-llvm-static \
#     llvm-toolset-20-llvm-mlir llvm-toolset-20-llvm-mlir-devel \
#     llvm-toolset-20-llvm-mlir-static llvm-toolset-20-python3-mlir \
#     llvm-toolset-20-clang
#
# 查找提示优先级：显式 -DLLVM_DIR/-DMLIR_DIR（find_package 自身行为）高于
# ANNC_LLVM_BIN 环境变量，最后回退 llvm-config --cmakedir。
set(_annc_llvm_hints "")
set(_annc_mlir_hints "")
if(DEFINED ENV{ANNC_LLVM_BIN} AND NOT "$ENV{ANNC_LLVM_BIN}" STREQUAL "")
  get_filename_component(_annc_llvm_prefix "$ENV{ANNC_LLVM_BIN}/.." ABSOLUTE)
  list(APPEND _annc_llvm_hints
    "${_annc_llvm_prefix}/lib64/cmake/llvm"
    "${_annc_llvm_prefix}/lib/cmake/llvm")
  list(APPEND _annc_mlir_hints
    "${_annc_llvm_prefix}/lib64/cmake/mlir"
    "${_annc_llvm_prefix}/lib/cmake/mlir")
endif()
# LLVM_ROOT / MLIR_ROOT（环境变量或 CMake 变量）可指向安装前缀，也可直接指向
# 含 LLVMConfig.cmake / MLIRConfig.cmake 的目录。
foreach(_annc_root IN ITEMS
    "$ENV{LLVM_ROOT}" "${LLVM_ROOT}" "$ENV{MLIR_ROOT}" "${MLIR_ROOT}")
  if(NOT "${_annc_root}" STREQUAL "")
    list(APPEND _annc_llvm_hints
      "${_annc_root}"
      "${_annc_root}/lib64/cmake/llvm"
      "${_annc_root}/lib/cmake/llvm")
    list(APPEND _annc_mlir_hints
      "${_annc_root}"
      "${_annc_root}/lib64/cmake/mlir"
      "${_annc_root}/lib/cmake/mlir")
  endif()
endforeach()
if(NOT DEFINED LLVM_DIR)
  find_program(ANNC_LLVM_CONFIG NAMES
    llvm-config-22 llvm-config-21 llvm-config-20 llvm-config)
  if(ANNC_LLVM_CONFIG)
    execute_process(COMMAND "${ANNC_LLVM_CONFIG}" --cmakedir
      OUTPUT_VARIABLE _annc_llvm_cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(EXISTS "${_annc_llvm_cmake_dir}/LLVMConfig.cmake")
      list(APPEND _annc_llvm_hints "${_annc_llvm_cmake_dir}")
    endif()
  endif()
endif()
# MLIR 的 CMake 配置目录通常与 LLVM 的同级（<prefix>/lib{,64}/cmake/mlir）。
foreach(_hint IN LISTS _annc_llvm_hints LLVM_DIR)
  if(EXISTS "${_hint}/LLVMConfig.cmake")
    get_filename_component(_annc_llvm_cmake_root "${_hint}" DIRECTORY)
    list(APPEND _annc_mlir_hints "${_annc_llvm_cmake_root}/mlir")
  endif()
endforeach()

find_package(LLVM CONFIG QUIET HINTS ${_annc_llvm_hints})
if(NOT LLVM_FOUND)
  message(FATAL_ERROR
    "LLVM not found (no LLVMConfig.cmake). Install an LLVM/MLIR toolchain "
    "(on openEuler 24.03 the verified packages are):\n"
    "  sudo dnf install -y llvm-toolset-20-llvm llvm-toolset-20-llvm-devel "
    "llvm-toolset-20-llvm-libs llvm-toolset-20-llvm-static "
    "llvm-toolset-20-llvm-mlir llvm-toolset-20-llvm-mlir-devel "
    "llvm-toolset-20-llvm-mlir-static llvm-toolset-20-python3-mlir "
    "llvm-toolset-20-clang\n"
    "or pass -DLLVM_DIR=<directory containing LLVMConfig.cmake>, "
    "-DLLVM_ROOT=<install prefix> or set ANNC_LLVM_BIN. "
    "(LLVM_DIR='${LLVM_DIR}', LLVM_ROOT='${LLVM_ROOT}')")
endif()
# Required form (find_package caches the hit above): keeps the canonical
# "REQUIRED CONFIG" contract while the QUIET probe supplies the package hint.
find_package(LLVM REQUIRED CONFIG HINTS ${_annc_llvm_hints})
find_package(MLIR CONFIG QUIET HINTS ${_annc_mlir_hints})
if(NOT MLIR_FOUND)
  message(FATAL_ERROR
    "MLIR not found (no MLIRConfig.cmake). Install the matching MLIR "
    "development package (e.g. llvm-toolset-20-llvm-mlir-devel and "
    "llvm-toolset-20-python3-mlir for the Python binding sources), or pass "
    "-DMLIR_DIR=<directory containing MLIRConfig.cmake>, "
    "-DMLIR_ROOT=<install prefix> or set ANNC_LLVM_BIN. "
    "(MLIR_DIR='${MLIR_DIR}', MLIR_ROOT='${MLIR_ROOT}')")
endif()
find_package(MLIR REQUIRED CONFIG HINTS ${_annc_mlir_hints})

# 发行版同时提供 libMLIR.so、libMLIR-C.so 与逐 dialect 的静态归档。同一个进程
# 里混用多份 MLIR 实现会出现多份 TypeID/StorageUniquer：libMLIR-C.so 内嵌了一份
# 隐藏可见性的 MLIR，它创建的 MLIRContext 交给 libMLIR.so 的 C++ 代码使用时会
# 命中 ParametricStorageUniquer 空指针断言（Python 绑定实测）。llvm-toolset-20-
# python3-mlir 提供的 libMLIRPythonCAPI.so 是一份自包含、导出符号的 MLIR C API
# 实现，与 libMLIR.so 组合使用已验证（发行版自带的 mlir Python 包即如此构建）。
# 因此：C API 目标统一指向 libMLIRPythonCAPI.so，其余 MLIR 静态目标指向
# libMLIR.so；执行引擎、运行时与测试目标不在 libMLIR.so 中，保持原样。
get_target_property(_annc_mlir_shared MLIR IMPORTED_LOCATION)
foreach(_cfg _RELWITHDEBINFO _RELEASE _DEBUG _MINSIZEREL)
  if(NOT _annc_mlir_shared)
    get_target_property(_annc_mlir_shared MLIR IMPORTED_LOCATION${_cfg})
  endif()
endforeach()
if(NOT _annc_mlir_shared)
  if(ANNC_ENABLE_PYTHON)
    message(FATAL_ERROR
      "Python bindings require a shared libMLIR (the ANNC extension must share "
      "the MLIR implementation with libMLIRPythonCAPI.so). Build the external "
      "LLVM/MLIR with a shared MLIR library, or configure with "
      "-DANNC_ENABLE_PYTHON=OFF.")
  endif()
  message(STATUS "ANNC: no shared MLIR target; using the toolchain's static "
                 "libraries")
endif()

if(ANNC_ENABLE_PYTHON)
  find_file(ANNC_MLIR_PYTHON_CAPI_LIBRARY
    NAMES libMLIRPythonCAPI.so
    HINTS
      "${LLVM_INSTALL_PREFIX}/python_packages/mlir_core/mlir/_mlir_libs"
      "${LLVM_INSTALL_PREFIX}/lib/python_packages/mlir_core/mlir/_mlir_libs"
      "${MLIR_INSTALL_PREFIX}/python_packages/mlir_core/mlir/_mlir_libs")
  if(NOT ANNC_MLIR_PYTHON_CAPI_LIBRARY)
    message(FATAL_ERROR
      "libMLIRPythonCAPI.so not found. Install llvm-toolset-20-python3-mlir "
      "(it provides the MLIR Python C API library) or pass "
      "-DANNC_MLIR_PYTHON_CAPI_LIBRARY=<path>.")
  endif()
  message(STATUS "ANNC: MLIR Python C API: ${ANNC_MLIR_PYTHON_CAPI_LIBRARY}")
endif()

get_property(_annc_imported_targets DIRECTORY PROPERTY IMPORTED_TARGETS)
if(_annc_mlir_shared)
foreach(_annc_tgt IN LISTS _annc_imported_targets)
  if(NOT _annc_tgt MATCHES "^MLIR")
    continue()
  endif()
  get_target_property(_annc_type ${_annc_tgt} TYPE)
  if(NOT _annc_type STREQUAL "STATIC_LIBRARY")
    continue()
  endif()
  # Not part of libMLIR.so / libMLIRPythonCAPI.so.
  if(_annc_tgt MATCHES "^mlir_" OR _annc_tgt MATCHES "^MLIRExecutionEngine" OR
     _annc_tgt MATCHES "Test")
    continue()
  endif()
  if(_annc_tgt MATCHES "^MLIRCAP")
    if(ANNC_ENABLE_PYTHON)
      set(_annc_shared "${ANNC_MLIR_PYTHON_CAPI_LIBRARY}")
    else()
      continue()
    endif()
  else()
    set(_annc_shared "${_annc_mlir_shared}")
  endif()
  set_target_properties(${_annc_tgt} PROPERTIES
    IMPORTED_LOCATION "${_annc_shared}"
    IMPORTED_LOCATION_RELEASE "${_annc_shared}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${_annc_shared}"
    IMPORTED_LOCATION_DEBUG "${_annc_shared}"
    IMPORTED_LOCATION_MINSIZEREL "${_annc_shared}")
endforeach()
message(STATUS "ANNC: linking MLIR via ${_annc_mlir_shared} (shared)")
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
separate_arguments(_annc_llvm_definitions NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${_annc_llvm_definitions})

# Toolchain locations the ANNC driver and pipeline use at build and run time.
if(NOT LLVM_LIBRARY_DIRS)
  message(FATAL_ERROR "LLVM_LIBRARY_DIRS is empty; LLVMConfig.cmake is incomplete.")
endif()
list(GET LLVM_LIBRARY_DIRS 0 ANNC_LLVM_LIB_DIR)
set(ANNC_LLVM_BIN_DIR "${LLVM_TOOLS_BINARY_DIR}")
# Build-time default for tests/FileCheck/run.sh (the script also accepts the
# ANNC_LLVM_BIN environment variable, which takes precedence).
file(WRITE "${CMAKE_BINARY_DIR}/annc_llvm_bin.txt" "${ANNC_LLVM_BIN_DIR}\n")

message(STATUS "ANNC: using LLVM/MLIR ${LLVM_PACKAGE_VERSION} "
               "(version not pinned; external toolchain)")
message(STATUS "ANNC: LLVM_DIR=${LLVM_DIR}")
message(STATUS "ANNC: MLIR_DIR=${MLIR_DIR}")
message(STATUS "ANNC: LLVM tools: ${ANNC_LLVM_BIN_DIR}")
message(STATUS "ANNC: LLVM libraries: ${ANNC_LLVM_LIB_DIR}")

# 绑定所需的 MLIR Python 源码目标由 llvm-toolset-20-python3-mlir 提供
# （<prefix>/src/python），因此不需要内建 LLVM。
if(ANNC_ENABLE_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 REQUIRED)
  include_directories(${pybind11_INCLUDE_DIRS})

  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  find_package(nanobind CONFIG REQUIRED)
endif()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/annc/include)
include_directories(${CMAKE_BINARY_DIR}/annc/include)

include_directories(${json_SOURCE_DIR}/include)

# Keep ANNC-owned debug information relocatable in build/install artifacts.
# LLVM/MLIR are prebuilt system libraries, so these flags apply to ANNC code
# only.
add_compile_options(
  "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
  "-fdebug-prefix-map=${CMAKE_SOURCE_DIR}=."
  "-fdebug-prefix-map=${CMAKE_BINARY_DIR}=./build"
)

# Code coverage via gcov (--coverage, portable across gcc/clang). The options
# apply only to directories processed afterwards (annc/, python/, frontends/,
# tests/, and the enabled addons), so system LLVM/MLIR are never instrumented.
# Both compile and link flags are required: compile flags instrument objects
# (referencing __gcov_* symbols), link flags pull in the gcov runtime that
# resolves them — so any ANNC executable linking an instrumented object (e.g.
# libANNCKernel) must also get the link flag, which the global scope guarantees.
if(ANNC_ENABLE_COVERAGE)
  add_compile_options(--coverage)
  add_link_options(--coverage)
  message(STATUS "ANNC code coverage enabled (--coverage / gcov).")
endif()

# ANNC-scoped debug mode (--annc-debug). Directory-level compile options apply
# only to targets defined afterwards (annc/, python/, frontends/,
# tensorflow_addons/, addons/onnxruntime/, tests/), so third-party json compile
# commands are untouched. The flags are appended at the end of the compile
# command, so -O0 overrides the Release -O3 and -UNDEBUG re-enables assert() in
# ANNC code even when CMAKE_BUILD_TYPE=Release.
if(ANNC_DEBUG)
  add_compile_options(-g3 -O0 -fno-omit-frame-pointer -UNDEBUG)
  message(STATUS "ANNC debug mode enabled: ANNC-only -g3 -O0 -UNDEBUG; third-party build type unchanged.")
endif()

include(CTest)

add_subdirectory(annc)
if(ANNC_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
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
