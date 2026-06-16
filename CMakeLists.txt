cmake_minimum_required(VERSION 3.21)

project(annc LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# set(ABSL_PROPAGATE_CXX_STD ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
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

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable assertions")

# Fetch LLVM/MLIR source at configure time

# LLVM 裁剪选项 — 只编译 AArch64 后端 + MLIR
set(LLVM_TARGETS_TO_BUILD "AArch64" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
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

set(TENSORFLOW_LIBRARY_DIR $ENV{TENSORFLOW_LIBRARY_DIR})
if(NOT TENSORFLOW_LIBRARY_DIR)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
    OUTPUT_VARIABLE TENSORFLOW_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
  OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TENSORFLOW_INCLUDE_DIRS)
  set(TENSORFLOW_INCLUDE_DIRS "${TENSORFLOW_LIBRARY_DIR}/include")
endif()

# Allow finding versioned TF shared libraries when the unversioned .so symlink is absent.
# Scope the suffix change so it does not leak to other find_library calls.
set(_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2)

find_library(TENSORFLOW_FRAMEWORK_LIB
  NAMES tensorflow_framework
  PATHS ${TENSORFLOW_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

find_library(TENSORFLOW_CC_LIB
  NAMES tensorflow_cc
  PATHS ${TENSORFLOW_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES})

if (NOT TENSORFLOW_FRAMEWORK_LIB OR NOT TENSORFLOW_CC_LIB)
  message(FATAL_ERROR "Could not find TensorFlow libraries. Please ensure they are installed.")
endif()

set(TENSORFLOW_LIBS ${TENSORFLOW_FRAMEWORK_LIB} ${TENSORFLOW_CC_LIB})

message(STATUS "TensorFlow include dir: ${TENSORFLOW_INCLUDE_DIRS}")
#message(STATUS "TensorFlow library dir: ${TENSORFLOW_LIBRARY_DIR}")
message(STATUS "TensorFlow libraries: ${TENSORFLOW_LIBS}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/annc/include)
include_directories(${CMAKE_BINARY_DIR}/annc/include)

include_directories(${json_SOURCE_DIR}/include)

add_subdirectory(annc)
add_subdirectory(python)
add_subdirectory(tensorflow_addons)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
