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

# L0: C++ unit tests, grouped by the production subsystem they cover.
add_subdirectory(unit/annc)
add_subdirectory(unit/kernels)

# ONNX frontend unit test: registered only while ANNC_ENABLE_ONNX=ON, which is
# what defines the annc-onnx2atir target.
if(TARGET annc-onnx2atir)
  add_subdirectory(unit/onnx2atir)
endif()

# ONNX Runtime EP tests: registered only while ANNC_ENABLE_ONNXRUNTIME_EP=ON,
# which is what defines the annc_ep target.  The GTest suites live under
# unit/ort_ep; the Python fixtures and e2e drivers (CTest add_test entries)
# live under integration/ort_ep.
if(TARGET annc_ep)
  # The EP Python fixtures need onnx + onnxruntime, which the interpreter used
  # for the C++ build does not necessarily provide.  Keep the override local to
  # the test tree so it cannot change the Python bindings' interpreter.
  set(ANNC_TEST_PYTHON "" CACHE FILEPATH
      "Interpreter for CTest-driven Python suites (defaults to the one found by FindPython3)")
  if(ANNC_TEST_PYTHON)
    set(Python3_EXECUTABLE "${ANNC_TEST_PYTHON}")
  endif()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_subdirectory(unit/ort_ep)
  add_subdirectory(integration/ort_ep)
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
