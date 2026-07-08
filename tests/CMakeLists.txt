add_subdirectory(kernels)

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
    # third_party/llvm's committed .gcno/.gcda test fixtures.
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
