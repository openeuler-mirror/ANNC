find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
include(GoogleTest)

add_executable(op_coverage_test
    op_coverage_test.cpp
)
target_include_directories(op_coverage_test PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(op_coverage_test PRIVATE
    GTest::gtest_main
    Threads::Threads
    AtirBuilder
    AtirDialect
    MLIRArithDialect
)
gtest_discover_tests(op_coverage_test)

add_executable(jit_template_fingerprint_test
    jit_template_fingerprint_test.cpp
)
target_include_directories(jit_template_fingerprint_test PRIVATE
    "${PROJECT_SOURCE_DIR}/annc/include"
    "${PROJECT_BINARY_DIR}/annc/include"
)
target_link_libraries(jit_template_fingerprint_test PRIVATE
    GTest::gtest_main
    Threads::Threads
    AtirDialect
    MLIRDialect
    MLIRFuncDialect
    MLIRParser
)
gtest_discover_tests(jit_template_fingerprint_test
    TEST_PREFIX "JitTemplateFingerprint."
)

add_executable(annc_jit_cache_test
    annc_jit_cache_test.cpp
)
target_include_directories(annc_jit_cache_test PRIVATE
    "${PROJECT_SOURCE_DIR}/tensorflow_addons"
)
target_link_libraries(annc_jit_cache_test PRIVATE
    GTest::gtest_main
    Threads::Threads
    annc_jit_cache
)
gtest_discover_tests(annc_jit_cache_test
    TEST_PREFIX "JitCompilationCache."
)
