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

# The JIT cache lives in the TensorFlow addon, so its test is registered only
# while that addon is built (ANNC_ENABLE_TENSORFLOW_ADDON gates
# tensorflow_addons/, which defines the annc_jit_cache target).
if(ANNC_ENABLE_TENSORFLOW_ADDON)
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
endif()

# The mlir_ciface argument dispatch is header-only and independent of
# TensorFlow; keep its regression test available even in TF-disabled builds.
add_executable(ciface_dispatch_test
    ciface_dispatch_test.cpp
)
target_include_directories(ciface_dispatch_test PRIVATE
    "${PROJECT_SOURCE_DIR}/tensorflow_addons"
)
target_link_libraries(ciface_dispatch_test PRIVATE
    GTest::gtest_main
    Threads::Threads
)
gtest_discover_tests(ciface_dispatch_test
    TEST_PREFIX "CifaceDispatch."
)

add_executable(gemm_fusion_level_test
    gemm_fusion_level_test.cpp
)
target_include_directories(gemm_fusion_level_test PRIVATE
    "${PROJECT_SOURCE_DIR}/annc/include"
    "${PROJECT_BINARY_DIR}/annc/include"
)
target_link_libraries(gemm_fusion_level_test PRIVATE
    GTest::gtest_main
    Threads::Threads
    AtirDialect
    MLIRDialect
)
gtest_discover_tests(gemm_fusion_level_test)
