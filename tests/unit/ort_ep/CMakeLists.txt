# L0 unit tests for the ONNX Runtime execution-provider plugin.
#
# Both suites compile production translation units directly (the plugin is a
# shared library resolved by the ORT host, not a library the tests can link
# against symbol-by-symbol), so the sources stay in addons/onnxruntime/;
# only the test sources live here.

find_package(Protobuf REQUIRED)

file(GLOB_RECURSE ONNX_PROTO_SRCS
    "${CMAKE_SOURCE_DIR}/frontends/onnx/protos/gen_code/*.pb.cc")

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ABSL_LOG QUIET absl_log)
    pkg_check_modules(ABSL_CHECK_OP QUIET absl_log_internal_check_op)
    pkg_check_modules(ABSL_BASE QUIET absl_base)
endif()

# ModelSignature (annc_ep_graph_export.cc) is part of the contract this suite
# pins: the disk cache must not serve one model's folded constants to another,
# so the signature tests link the export translation unit and the generated
# ONNX protobuf code it reads models through.
add_executable(annc-ep-compile-test
    "${CMAKE_CURRENT_SOURCE_DIR}/annc_ep_compile_test.cpp"
    "${CMAKE_SOURCE_DIR}/addons/onnxruntime/annc_ep_graph_export.cc"
    "${CMAKE_SOURCE_DIR}/addons/onnxruntime/annc_ep_site.cc"
    ${ONNX_PROTO_SRCS}
)
target_include_directories(annc-ep-compile-test PRIVATE
    "${CMAKE_SOURCE_DIR}/addons/onnxruntime"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/protos/gen_code"
    "${Protobuf_INCLUDE_DIRS}"
    "${CMAKE_SOURCE_DIR}/third_party/json/include"
)
target_include_directories(annc-ep-compile-test SYSTEM PRIVATE
    "${ANNC_ONNXRUNTIME_INCLUDE_DIR}"
)
if(ABSL_LOG_FOUND AND ABSL_CHECK_OP_FOUND)
    target_include_directories(annc-ep-compile-test SYSTEM PRIVATE
        ${ABSL_LOG_INCLUDE_DIRS}
        ${ABSL_CHECK_OP_INCLUDE_DIRS}
        ${ABSL_BASE_INCLUDE_DIRS}
    )
    target_link_libraries(annc-ep-compile-test PRIVATE
        ${ABSL_LOG_LIBRARIES}
        ${ABSL_CHECK_OP_LIBRARIES}
    )
endif()
target_link_libraries(annc-ep-compile-test PRIVATE
    GTest::gtest_main ${Protobuf_LIBRARIES} ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(annc-ep-compile-test PROPERTIES CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(annc-ep-compile-test)

# The execution context's failure bookkeeping, driven through a fake OrtApi
# (no ORT host needed): the fixture suites never provoke a failing ORT call.
add_executable(annc-ep-execution-test
    "${CMAKE_CURRENT_SOURCE_DIR}/annc_ep_execution_test.cpp"
    "${CMAKE_SOURCE_DIR}/addons/onnxruntime/annc_ep_execution.cc"
)
target_include_directories(annc-ep-execution-test SYSTEM PRIVATE
    "${ANNC_ONNXRUNTIME_INCLUDE_DIR}"
)
target_include_directories(annc-ep-execution-test PRIVATE
    "${CMAKE_SOURCE_DIR}/addons/onnxruntime"
    "${CMAKE_SOURCE_DIR}/annc/include"
)
target_link_libraries(annc-ep-execution-test PRIVATE
    GTest::gtest_main ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(annc-ep-execution-test PROPERTIES CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(annc-ep-execution-test PROPERTIES TIMEOUT 60)
