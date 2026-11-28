# L0 unit tests for the ONNX -> ATIR frontend.
#
# The test binary compiles the frontend translation units directly (the
# frontend is a CLI tool, not a reusable library), so the sources stay in
# frontends/onnx/onnx2atir/; only the test source and its registration live
# under tests/.

find_package(Protobuf REQUIRED)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

file(GLOB_RECURSE ONNX_PROTO_SRCS
    "${CMAKE_SOURCE_DIR}/frontends/onnx/protos/gen_code/*.pb.cc")

add_executable(annc-onnx2atir-frontend-test
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_frontend_test.cpp"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/onnx2atir/onnx_frontend.cpp"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/onnx2atir/onnx_tensor_resolver.cpp"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/onnx2atir/onnx_tensor_decoder.cpp"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/onnx2atir/onnx_node_info_adapter.cpp"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/onnx2atir/onnx_mlir_builder.cpp"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/onnx2atir/onnx_transformers.cpp"
    ${ONNX_PROTO_SRCS}
)
target_include_directories(annc-onnx2atir-frontend-test BEFORE PRIVATE
    "${CMAKE_SOURCE_DIR}/annc/include"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/onnx2atir"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/protos/gen_code"
)
target_include_directories(annc-onnx2atir-frontend-test SYSTEM PRIVATE
    "${Protobuf_INCLUDE_DIRS}"
    "${MLIR_INCLUDE_DIRS}"
    "${LLVM_INCLUDE_DIRS}"
)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ABSL_LOG QUIET absl_log)
    pkg_check_modules(ABSL_CHECK_OP QUIET absl_log_internal_check_op)
    pkg_check_modules(ABSL_BASE QUIET absl_base)
endif()
if(ABSL_LOG_FOUND AND ABSL_CHECK_OP_FOUND)
    target_link_libraries(annc-onnx2atir-frontend-test PRIVATE
        ${ABSL_LOG_LIBRARIES}
        ${ABSL_CHECK_OP_LIBRARIES}
    )
endif()
target_link_libraries(annc-onnx2atir-frontend-test PRIVATE
    ${Protobuf_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    AtirBuilder
    AtirDialect
    MLIRDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRInferTypeOpInterface
    ${MLIR_LIBS}
    ${llvm_libs}
)
gtest_discover_tests(annc-onnx2atir-frontend-test)
