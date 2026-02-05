cmake_minimum_required(VERSION 3.21)

project(kpgemm LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# set(ABSL_PROPAGATE_CXX_STD ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_warn_duplicate_libraries")

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/json)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(LLVM_ENABLE_ASSERTIONS ON)

find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(TENSORFLOW_LIBRARY_DIR $ENV{TENSORFLOW_LIBRARY_DIR})
set(TENSORFLOW_INCLUDE_DIRS "${TENSORFLOW_LIBRARY_DIR}/include")

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

if (NOT TENSORFLOW_FRAMEWORK_LIB OR NOT TENSORFLOW_CC_LIB)
  message(FATAL_ERROR "Could not find TensorFlow libraries. Please ensure they are installed.")
endif()

set(TENSORFLOW_LIBS ${TENSORFLOW_FRAMEWORK_LIB} ${TENSORFLOW_CC_LIB})

message(STATUS "TensorFlow include dir: ${TENSORFLOW_INCLUDE_DIRS}")
#message(STATUS "TensorFlow library dir: ${TENSORFLOW_LIBRARY_DIR}")
message(STATUS "TensorFlow libraries: ${TENSORFLOW_LIBS}")

find_package(MLIR REQUIRED CONFIG)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${TENSORFLOW_INCLUDE_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/kpgemm/include)
include_directories(${CMAKE_SOURCE_DIR}/build/kpgemm/include)

include_directories(${CMAKE_SOURCE_DIR}/third_party/json/include)

add_subdirectory(kpgemm)
add_subdirectory(python)