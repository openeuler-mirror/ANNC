# ANNC test tree.
#
# Every test-only dependency discovery and every test target registration lives
# here or in a per-suite CMakeLists below this directory.  The production
# subdirectories (annc/, frontends/, addons/, tensorflow_addons/) own no test
# sources and must not call find_package(GTest) or register add_test entries.
#
# The root CMakeLists.txt enters this directory only while BUILD_TESTING is on,
# after ALL production targets have been defined, so the optional suites below
# can gate on target existence (annc-onnx2atir, annc_ep).

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
include(GoogleTest)

# Keep the test interpreter separate from the Python bindings' interpreter.
set(ANNC_TEST_PYTHON "" CACHE FILEPATH
    "Interpreter for CTest-driven Python suites (defaults to the one found by FindPython3)")
if(ANNC_TEST_PYTHON)
  set(Python3_EXECUTABLE "${ANNC_TEST_PYTHON}")
endif()
if(TARGET annc_ep)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
else()
  find_package(Python3 COMPONENTS Interpreter QUIET)
endif()

# L0: C++ unit tests, grouped by the production subsystem they cover.
add_subdirectory(unit/annc)
add_subdirectory(unit/kernels)
if(TARGET annc_tensorflow_support AND TARGET annc-tf-pipeline)
  set(_tf_jit_env
    "ANNC_PIPELINE_PATH=$<TARGET_FILE:annc-tf-pipeline>"
    "ANNC_KERNEL_LIB_PATH=${PROJECT_BINARY_DIR}/lib"
    "ANNC_KEEP_TEMPS=0")
  if(ANNC_GEMM_PACKAGE_AVAILABLE)
    list(APPEND _tf_jit_env "LIBRARY_PATH=${ANNC_GEMM_PACKAGE_DIR}:$ENV{LIBRARY_PATH}")
  endif()
endif()
add_subdirectory(unit/tensorflow)

# ONNX frontend unit test: registered only while ANNC_ENABLE_ONNX=ON, which is
# what defines the annc-onnx2atir target.
if(TARGET annc-onnx2atir)
  add_subdirectory(unit/onnx2atir)
endif()

# annc-converter-v2 unit tests: registered while the V2 route built the core
# library (the core target carries the TF include/link setup).
if(TARGET annc_converter_v2_core)
  add_subdirectory(unit/converter_v2)
endif()

# ONNX Runtime EP tests: registered only while ANNC_ENABLE_ONNXRUNTIME_EP=ON,
# which is what defines the annc_ep target.  The GTest suites live under
# unit/ort_ep; Python fixtures live under integration/ort_ep, and Context
# kernel e2e drivers under e2e/kernel_context. Their CTest registration stays
# in integration/ort_ep.
if(TARGET annc_ep)
  add_subdirectory(unit/ort_ep)
  add_subdirectory(integration/ort_ep)
endif()

if(TARGET annc_fused_op_1_15 AND TARGET annc-tf-pipeline)
  set(_tf_tests "${CMAKE_CURRENT_SOURCE_DIR}/integration/tensorflow")
  if(ANNC_TENSORFLOW_PRELOAD)
    set(_tf_host "${ANNC_TENSORFLOW_PYTHON_EXECUTABLE}" "${_tf_tests}/plugin_load.py")
  else()
    # Run against the same native SDK as the linked plugins. A Python wheel
    # may already embed its own copy of TensorFlow's registration tables.
    add_executable(tf_linked_host "${_tf_tests}/linked_plugin_load.cpp")
    get_target_property(_sdk_standard annc_optimizer_1_15 CXX_STANDARD)
    set_target_properties(tf_linked_host PROPERTIES
      CXX_STANDARD ${_sdk_standard} CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
    target_include_directories(tf_linked_host SYSTEM PRIVATE
      $<TARGET_PROPERTY:annc_optimizer_1_15,INCLUDE_DIRECTORIES>)
    target_compile_definitions(tf_linked_host PRIVATE
      $<TARGET_PROPERTY:annc_optimizer_1_15,COMPILE_DEFINITIONS>)
    target_link_libraries(tf_linked_host PRIVATE
      $<TARGET_PROPERTY:annc_optimizer_1_15,LINK_LIBRARIES>)
    set(_tf_host $<TARGET_FILE:tf_linked_host>)
  endif()
  add_test(NAME TensorFlowPluginLoad COMMAND ${_tf_host}
    $<TARGET_FILE:annc_optimizer_1_15> $<TARGET_FILE:annc_fused_op_1_15> plugin_load.pbtxt)
  set_tests_properties(TensorFlowPluginLoad PROPERTIES
    WORKING_DIRECTORY "${_tf_tests}/Inputs" ENVIRONMENT "${_tf_jit_env}")
endif()

if(ANNC_ENABLE_COVERAGE)
    set(ANNC_COVERAGE_DIR "${CMAKE_BINARY_DIR}/coverage")
    find_program(GCOVR_BIN gcovr)

    # Single quick entry: run ctest under profiling, then gcovr scans the
    # build tree for .gcda and emits a terminal summary + HTML report. The
    # absolute-path --filter scopes the report to ANNC Kernel sources — the
    # only code linked into the ctest binaries. (Global --coverage also
    # instruments other ANNC code linked into production executables that ctest
    # never runs; without this filter those 0%-covered files drown the report.)
    # Search path is limited to the build tree so gcovr doesn't walk into
    # third-party test fixtures.
    if(GCOVR_BIN)
        add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E make_directory ${ANNC_COVERAGE_DIR}
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND ${GCOVR_BIN}
                    --root ${CMAKE_SOURCE_DIR}
                    --filter "${CMAKE_SOURCE_DIR}/annc/(lib|include)/Kernel"
                    --filter "${CMAKE_SOURCE_DIR}/context/"
                    ${CMAKE_BINARY_DIR}
                    --html-details ${ANNC_COVERAGE_DIR}/index.html
                    --print-summary
            COMMAND ${CMAKE_COMMAND} -E echo
                    "Coverage HTML: ${ANNC_COVERAGE_DIR}/index.html"
            DEPENDS kernel_registry_test kernel_priority_resolver_test
                    builtin_shared_spec_test customcall_symbol_resolution_test
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running ctest and generating coverage report"
            VERBATIM
        )
    else()
        message(STATUS "gcovr not found; 'coverage' target disabled. Install with: pip install gcovr")
    endif()
endif()
