cmake_minimum_required(VERSION 3.15)
project(annc-tf2atir)

find_package(Protobuf REQUIRED)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

file(GLOB_RECURSE PROTO_SRCS "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code/*.pb.cc")

set(ANNC_TF2ATIR_SOURCES
    annc-tf2atir.cpp
    tf_frontend.cpp
    tf_tensor_resolver.cpp
    tensor_proto_decoder.cpp
    node_info_adapter.cpp
    ${PROTO_SRCS}
)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ABSL_LOG QUIET absl_log)
    pkg_check_modules(ABSL_CHECK_OP QUIET absl_log_internal_check_op)
    pkg_check_modules(ABSL_BASE QUIET absl_base)
endif()

# Some Abseil releases do not export the unsigned-long LogMessage template
# instantiation required by TensorFlow's generated protobuf code. Detect the
# actual linkable API instead of keying this compatibility shim to an OS
# release (the same openEuler release can carry different Abseil packages).
unset(ANNC_ABSL_LOG_MESSAGE_INCLUDE_DIR CACHE)
find_path(ANNC_ABSL_LOG_MESSAGE_INCLUDE_DIR
    absl/log/internal/log_message.h
    HINTS ${ABSL_LOG_INCLUDE_DIRS} ${ABSL_CHECK_OP_INCLUDE_DIRS}
          ${ABSL_BASE_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS})
set(ANNC_TF2ATIR_NEEDS_ABSL_LOG_SHIM OFF)
if(ANNC_ABSL_LOG_MESSAGE_INCLUDE_DIR AND ABSL_LOG_FOUND AND
   ABSL_CHECK_OP_FOUND)
    include(CheckCXXSourceCompiles)
    set(_annc_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(_annc_saved_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
    set(CMAKE_REQUIRED_INCLUDES
        "${ANNC_ABSL_LOG_MESSAGE_INCLUDE_DIR}"
        ${ABSL_LOG_INCLUDE_DIRS}
        ${ABSL_CHECK_OP_INCLUDE_DIRS}
        ${ABSL_BASE_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES
        "${ABSL_LOG_LIBRARIES};${ABSL_CHECK_OP_LIBRARIES}")
    unset(ANNC_ABSL_LOG_OPERATOR_UNSIGNED_LONG_LINKS CACHE)
    check_cxx_source_compiles([=[
#include <absl/log/internal/log_message.h>
static void useUnsignedLong(absl::log_internal::LogMessage &message) {
  message << static_cast<unsigned long>(0);
}
using UseUnsignedLong = void (*)(absl::log_internal::LogMessage &);
volatile UseUnsignedLong useFunction = &useUnsignedLong;
int main() {
  auto *message = static_cast<absl::log_internal::LogMessage *>(nullptr);
  useFunction(*message);
  return 0;
}
]=] ANNC_ABSL_LOG_OPERATOR_UNSIGNED_LONG_LINKS)
    set(CMAKE_REQUIRED_INCLUDES "${_annc_saved_required_includes}")
    set(CMAKE_REQUIRED_LIBRARIES "${_annc_saved_required_libraries}")
    if(NOT ANNC_ABSL_LOG_OPERATOR_UNSIGNED_LONG_LINKS)
        set(ANNC_TF2ATIR_NEEDS_ABSL_LOG_SHIM ON)
    endif()
elseif(ANNC_ABSL_LOG_MESSAGE_INCLUDE_DIR)
    # Without the libraries needed for a link probe, retain the historical
    # shim. This is conservative for installations where Abseil is supplied
    # transitively by Protobuf or TensorFlow rather than pkg-config.
    set(ANNC_TF2ATIR_NEEDS_ABSL_LOG_SHIM ON)
    message(STATUS
        "annc-tf2atir: cannot probe Abseil LogMessage; keeping compatibility shim")
endif()

if(ANNC_TF2ATIR_NEEDS_ABSL_LOG_SHIM)
    list(APPEND ANNC_TF2ATIR_SOURCES absl_log_shim.cpp)
    message(STATUS "annc-tf2atir: enabling Abseil LogMessage compatibility shim")
else()
    message(STATUS "annc-tf2atir: Abseil LogMessage compatibility shim not needed")
endif()

add_executable(annc-tf2atir
    ${ANNC_TF2ATIR_SOURCES}
)

target_include_directories(annc-tf2atir BEFORE PRIVATE
    "${CMAKE_SOURCE_DIR}/annc/include"
    "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code"
)
target_include_directories(annc-tf2atir PRIVATE
    "${MLIR_INCLUDE_DIRS}"
    "${LLVM_INCLUDE_DIRS}"
)

add_definitions(-DABSL_MIN_LOG_LEVEL=4)

target_include_directories(annc-tf2atir SYSTEM PRIVATE
    "${Protobuf_INCLUDE_DIRS}"
)

if(ABSL_LOG_INCLUDE_DIRS OR ABSL_CHECK_OP_INCLUDE_DIRS OR ABSL_BASE_INCLUDE_DIRS)
    target_include_directories(annc-tf2atir SYSTEM PRIVATE
        ${ABSL_LOG_INCLUDE_DIRS}
        ${ABSL_CHECK_OP_INCLUDE_DIRS}
        ${ABSL_BASE_INCLUDE_DIRS}
    )
endif()

target_link_libraries(annc-tf2atir PRIVATE
    ${Protobuf_LIBRARIES}
)

if(ABSL_LOG_FOUND AND ABSL_CHECK_OP_FOUND)
    message(STATUS "Linking abseil libraries for protobuf dependency")
    target_link_libraries(annc-tf2atir PRIVATE
        ${ABSL_LOG_LIBRARIES}
        ${ABSL_CHECK_OP_LIBRARIES}
    )
else()
    message(STATUS "Abseil libraries not found, skipping (may not be required on this system)")
endif()

target_link_libraries(annc-tf2atir PRIVATE
    AtirBuilder AtirDialect AtirPasses
    AtirToAffineConversion
    AtirToLinalgConversion
    MLIRArithDialect
    MLIROptLib
    ${MLIR_LIBS}
    ${llvm_libs}
)
set_target_properties(annc-tf2atir PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
    BUILD_RPATH "${Protobuf_LIB_DIRS}"
)
install(TARGETS annc-tf2atir DESTINATION bin)

if(BUILD_TESTING AND EXISTS "${CMAKE_SOURCE_DIR}/logs/dffm/tf2atir_tf_frontend_test.cpp")
  enable_testing()
  add_executable(annc-tf2atir-frontend-test
      "${CMAKE_SOURCE_DIR}/logs/dffm/tf2atir_tf_frontend_test.cpp"
      tf_frontend.cpp
      tf_tensor_resolver.cpp
      tensor_proto_decoder.cpp
      node_info_adapter.cpp
      ${PROTO_SRCS}
  )
  target_include_directories(annc-tf2atir-frontend-test BEFORE PRIVATE
      "${CMAKE_SOURCE_DIR}/annc/tools/annc-tf2atir"
      "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code"
  )
  target_include_directories(annc-tf2atir-frontend-test SYSTEM PRIVATE
      "${Protobuf_INCLUDE_DIRS}"
  )
  target_link_libraries(annc-tf2atir-frontend-test PRIVATE
      ${Protobuf_LIBRARIES}
      AtirBuilder
      AtirDialect
      MLIRDialect
      MLIRFuncDialect
      MLIRInferTypeOpInterface
  )
  if(ABSL_LOG_FOUND AND ABSL_CHECK_OP_FOUND)
    target_link_libraries(annc-tf2atir-frontend-test PRIVATE
        ${ABSL_LOG_LIBRARIES}
        ${ABSL_CHECK_OP_LIBRARIES}
    )
  endif()
  add_test(NAME annc-tf2atir-frontend-test
      COMMAND annc-tf2atir-frontend-test)
endif()
