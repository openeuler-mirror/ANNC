# Optional ONNX Runtime execution-provider plugin. Registered from the root
# CMakeLists.txt only while ANNC_ENABLE_ONNXRUNTIME_EP is ON, which also
# requires the ONNX frontend and ANNC_ONNXRUNTIME_INCLUDE_DIR pointing at the
# ORT headers. The library is compiled against the external ORT SDK and is not
# linked into any ANNC target; the loader is the ORT host process itself.
cmake_minimum_required(VERSION 3.15)
include(GNUInstallDirs)

# Building the plugin needs onnxruntime_c_api.h and onnxruntime_ep_c_api.h; the
# root CMakeLists.txt fails the configure when the directory does not hold
# them, so their presence is not re-checked here. File presence does not imply
# a compatible ORT version: compile and host-loading acceptance must use the
# ORT 1.26.0 SDK/runtime.
#
# The runtime-compile mode additionally serializes OrtGraph to ONNX
# (ort_graph_to_proto.h, vendored from the ORT tree) and reads the toolchain's
# site report JSON, so the plugin builds the generated ONNX protobuf code in and
# uses the vendored nlohmann/json. Prefer the static protobuf archive so the
# host process does not gain a new shared-library dependency.
set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf REQUIRED)

file(GLOB_RECURSE ONNX_PROTO_SRCS
    "${CMAKE_SOURCE_DIR}/frontends/onnx/protos/gen_code/*.pb.cc")

add_library(annc_ep SHARED
    annc_ep.cpp
    annc_ep_site.cc
    annc_ep_graph_export.cc
    annc_ep_compile.cc
    annc_ep_execution.cc
    ${ONNX_PROTO_SRCS}
)
target_include_directories(annc_ep SYSTEM PRIVATE
    "${ANNC_ONNXRUNTIME_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/frontends/onnx/protos/gen_code"
)
target_include_directories(annc_ep SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/third_party/json/include"
)
target_include_directories(annc_ep PRIVATE
    "${Protobuf_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # Execution-context ABI and the AnncThreadPool interface the context
    # carries into the kernels. Header-only use: the plugin never links ANNC.
    "${CMAKE_SOURCE_DIR}/annc/include"
)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ABSL_LOG QUIET absl_log)
    pkg_check_modules(ABSL_CHECK_OP QUIET absl_log_internal_check_op)
    pkg_check_modules(ABSL_BASE QUIET absl_base)
endif()
if(ABSL_LOG_FOUND AND ABSL_CHECK_OP_FOUND)
    target_include_directories(annc_ep SYSTEM PRIVATE
        ${ABSL_LOG_INCLUDE_DIRS}
        ${ABSL_CHECK_OP_INCLUDE_DIRS}
        ${ABSL_BASE_INCLUDE_DIRS}
    )
endif()

# dlopen/dlsym for the per-model kernels.
target_link_libraries(annc_ep PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(annc_ep PRIVATE ${Protobuf_LIBRARIES})
if(ABSL_LOG_FOUND AND ABSL_CHECK_OP_FOUND)
    target_link_libraries(annc_ep PRIVATE
        ${ABSL_LOG_LIBRARIES}
        ${ABSL_CHECK_OP_LIBRARIES}
    )
endif()
target_link_libraries(annc_ep PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# Back the EP's thread pool with ORT's native operator pool: the scheduling
# decision (serial versus parallel, block size, real degree of parallelism)
# moves from ANNC's port of Eigen's cost model to ORT's own TryParallelFor,
# through build-tree archives of the *same* ORT build that hosts the plugin.
# The EP always uses this path; there is no build option to disable it.
# Private-ABI coupling; see annc_ep_native_pool.h for the layout pin and the
# fallback conditions.
set(ANNC_EP_ORT_COMMON_ARCHIVE
    "${ANNC_EP_ORT_BUILD_TREE}/libonnxruntime_common.a")
set(ANNC_EP_ORT_CPUINFO_ARCHIVE
    "${ANNC_EP_ORT_BUILD_TREE}/_deps/pytorch_cpuinfo-build/libcpuinfo.a")
foreach(archive IN ITEMS "${ANNC_EP_ORT_COMMON_ARCHIVE}"
                         "${ANNC_EP_ORT_CPUINFO_ARCHIVE}")
  if(NOT EXISTS "${archive}")
    message(FATAL_ERROR
      "Building the ANNC EP needs ${archive}: point ANNC_EP_ORT_BUILD_TREE "
      "at the ORT build tree (not the install tree) of the runtime that "
      "will host the plugin.")
  endif()
endforeach()
target_compile_definitions(annc_ep PRIVATE
    ANNC_EP_NATIVE_POOL=1
    ANNC_EP_NATIVE_POOL_ORT_VERSION="${ANNC_EP_NATIVE_POOL_ORT_VERSION}")
target_link_libraries(annc_ep PRIVATE
    "-Wl,--start-group"
    "${ANNC_EP_ORT_COMMON_ARCHIVE}"
    "${ANNC_EP_ORT_CPUINFO_ARCHIVE}"
    "-Wl,--end-group")
# Keep the archives' ORT symbols out of the plugin's dynamic symbol table:
# the host resolves exactly two entry points from this library
# (onnxruntime/core/session/plugin_ep/ep_library_plugin.cc), and an
# exported copy of ORT's thread pool could interpose on other dynamically
# loaded libraries binding the same names.
target_link_options(annc_ep PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/annc_ep.map")

set_target_properties(annc_ep PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "annc_ep"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  include(GoogleTest)
  # ModelSignature (annc_ep_graph_export.cc) is part of the contract this
  # suite pins: the disk cache must not serve one model's folded constants to
  # another, so the signature tests link the export translation unit and the
  # generated ONNX protobuf code it reads models through.
  add_executable(annc-ep-compile-test
      annc_ep_compile_test.cpp
      annc_ep_graph_export.cc
      annc_ep_site.cc
      ${ONNX_PROTO_SRCS})
  target_include_directories(annc-ep-compile-test PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_SOURCE_DIR}/frontends/onnx/protos/gen_code"
      "${Protobuf_INCLUDE_DIRS}"
      "${CMAKE_SOURCE_DIR}/third_party/json/include")
  target_include_directories(annc-ep-compile-test SYSTEM PRIVATE
      "${ANNC_ONNXRUNTIME_INCLUDE_DIR}")
  if(ABSL_LOG_FOUND AND ABSL_CHECK_OP_FOUND)
    target_include_directories(annc-ep-compile-test SYSTEM PRIVATE
        ${ABSL_LOG_INCLUDE_DIRS}
        ${ABSL_CHECK_OP_INCLUDE_DIRS}
        ${ABSL_BASE_INCLUDE_DIRS})
    target_link_libraries(annc-ep-compile-test PRIVATE
        ${ABSL_LOG_LIBRARIES}
        ${ABSL_CHECK_OP_LIBRARIES})
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
      annc_ep_execution_test.cpp
      annc_ep_execution.cc)
  target_include_directories(annc-ep-execution-test SYSTEM PRIVATE
      "${ANNC_ONNXRUNTIME_INCLUDE_DIR}")
  target_include_directories(annc-ep-execution-test PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_SOURCE_DIR}/annc/include")
  target_link_libraries(annc-ep-execution-test PRIVATE
      GTest::gtest_main ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  set_target_properties(annc-ep-execution-test PROPERTIES CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON)
  gtest_discover_tests(annc-ep-execution-test PROPERTIES TIMEOUT 60)
  foreach(mode IN ITEMS normal missing-symbol missing-library malformed-report concurrent
                   cache-config missing-pipeline pipeline-failure zero-site)
    add_test(NAME AnncEpAotFixture.${mode}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ep_aot_fixture.py
        --ep-library $<TARGET_FILE:annc_ep> --mode ${mode})
    set_tests_properties(AnncEpAotFixture.${mode} PROPERTIES TIMEOUT 60)
  endforeach()
  # annc_execution_v2 sites: the context callbacks, the pool the context
  # carries, and the failure paths of both.
  foreach(mode IN ITEMS normal threads nested concurrent same-session error bad-slot)
    add_test(NAME AnncEpContextFixture.${mode}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ep_context_fixture.py
        --ep-library $<TARGET_FILE:annc_ep> --mode ${mode})
    set_tests_properties(AnncEpContextFixture.${mode} PROPERTIES TIMEOUT 120)
  endforeach()
  add_test(NAME AnncEpFtdmmAotE2e
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/ftdmm_aot_e2e.py
      --ep-library $<TARGET_FILE:annc_ep>)
  set_tests_properties(AnncEpFtdmmAotE2e PROPERTIES TIMEOUT 300)
  # One fused pattern per switch; each is run enabled (claimed) and disabled
  # (the EP drops the site and the CPU EP runs it).
  add_test(NAME AnncEpFusionToggles
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/fusion_toggles_e2e.py
      --ep-library $<TARGET_FILE:annc_ep>)
  set_tests_properties(AnncEpFusionToggles PROPERTIES TIMEOUT 300)
  # The context-ABI FTDMM needs the pipeline and kernel library built from
  # this tree (the legacy e2e can also run against an installed dev pipeline).
  if(TARGET annc-onnx-pipeline)
    set(AnncEpFtdmmContextE2eLlvmLibArg "")
    if(TARGET LLVMSupport)
      list(APPEND AnncEpFtdmmContextE2eLlvmLibArg
           --llvm-lib-dir $<TARGET_FILE_DIR:LLVMSupport>)
    endif()
    set(AnncEpFtdmmContextE2eExtraLibArgs "")
    if(TARGET ANNCGemmSupport)
      list(APPEND AnncEpFtdmmContextE2eExtraLibArgs
           --extra-lib-dir $<TARGET_FILE_DIR:ANNCGemmSupport>)
    endif()
    if(TARGET ANNC::AArch64GemmMicrokernels)
      list(APPEND AnncEpFtdmmContextE2eExtraLibArgs
           --extra-lib-dir $<TARGET_FILE_DIR:ANNC::AArch64GemmMicrokernels>)
    endif()
    add_test(NAME AnncEpFtdmmContextE2e
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ftdmm_context_e2e.py
        --ep-library $<TARGET_FILE:annc_ep>
        --pipeline $<TARGET_FILE:annc-onnx-pipeline>
        --kernel-lib-dir $<TARGET_FILE_DIR:ANNCBuiltinKernels>
        ${AnncEpFtdmmContextE2eLlvmLibArg}
        ${AnncEpFtdmmContextE2eExtraLibArgs})
    set_tests_properties(AnncEpFtdmmContextE2e PROPERTIES TIMEOUT 600)
    # masked_mul: a site whose arguments need name binding and whose two
    # outputs need the report->fused-node permutation.
    add_test(NAME AnncEpMaskedMulContextE2e
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/masked_mul_context_e2e.py
        --ep-library $<TARGET_FILE:annc_ep>
        --pipeline $<TARGET_FILE:annc-onnx-pipeline>
        --kernel-lib-dir $<TARGET_FILE_DIR:ANNCBuiltinKernels>
        ${AnncEpFtdmmContextE2eLlvmLibArg}
        ${AnncEpFtdmmContextE2eExtraLibArgs})
    set_tests_properties(AnncEpMaskedMulContextE2e PROPERTIES TIMEOUT 600)
    # layer_norm: rank-1 gamma/beta force name binding, eps is rank-0.
    add_test(NAME AnncEpLayerNormContextE2e
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/layer_norm_context_e2e.py
        --ep-library $<TARGET_FILE:annc_ep>
        --pipeline $<TARGET_FILE:annc-onnx-pipeline>
        --kernel-lib-dir $<TARGET_FILE_DIR:ANNCBuiltinKernels>
        ${AnncEpFtdmmContextE2eLlvmLibArg}
        ${AnncEpFtdmmContextE2eExtraLibArgs})
    set_tests_properties(AnncEpLayerNormContextE2e PROPERTIES TIMEOUT 600)
    # batched_matmul_add: all three arguments are rank-3 f32 (name binding).
    add_test(NAME AnncEpBatchedMatMulAddContextE2e
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/batched_matmul_add_context_e2e.py
        --ep-library $<TARGET_FILE:annc_ep>
        --pipeline $<TARGET_FILE:annc-onnx-pipeline>
        --kernel-lib-dir $<TARGET_FILE_DIR:ANNCBuiltinKernels>
        ${AnncEpFtdmmContextE2eLlvmLibArg}
        ${AnncEpFtdmmContextE2eExtraLibArgs})
    set_tests_properties(AnncEpBatchedMatMulAddContextE2e PROPERTIES TIMEOUT 600)
  endif()
endif()

# The ORT C++ API resolves OrtGetApiBase from the host process at load time, so
# the plugin must link with undefined symbols allowed -- the same arrangement
# the TensorFlow addon uses when its host provides the TF symbols.
target_link_options(annc_ep PRIVATE
    "-Wl,--allow-shlib-undefined" "-Wl,-z,undefs")
install(TARGETS annc_ep DESTINATION ${CMAKE_INSTALL_LIBDIR})
