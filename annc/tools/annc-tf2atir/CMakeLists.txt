cmake_minimum_required(VERSION 3.15)
project(annc-tf2atir)

find_package(Protobuf REQUIRED)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

file(GLOB_RECURSE PROTO_SRCS "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code/*.pb.cc")

add_executable(annc-tf2atir
    annc-tf2atir.cpp
    standalone_pb_parser.cpp
    absl_log_shim.cpp
    ${PROTO_SRCS}
)

target_include_directories(annc-tf2atir BEFORE PRIVATE
    "${CMAKE_SOURCE_DIR}/annc/include"
    "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code"
)
target_include_directories(annc-tf2atir PRIVATE
    "${MLIR_INCLUDE_DIRS}"
    "${LLVM_INCLUDE_DIRS}"
)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ABSL_LOG QUIET absl_log)
    pkg_check_modules(ABSL_CHECK_OP QUIET absl_log_internal_check_op)
    pkg_check_modules(ABSL_BASE QUIET absl_base)
endif()

add_definitions(-DABSL_MIN_LOG_LEVEL=4)

target_include_directories(annc-tf2atir SYSTEM PRIVATE
    "${Protobuf_INCLUDE_DIRS}"
)

if(ABSL_BASE_FOUND)
    target_include_directories(annc-tf2atir SYSTEM PRIVATE
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
