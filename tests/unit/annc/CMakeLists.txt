find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
include(GoogleTest)

add_executable(gemm_config_test gemm_config_test.cpp)
target_include_directories(gemm_config_test PRIVATE
  "${PROJECT_SOURCE_DIR}/annc/include"
)
target_link_libraries(gemm_config_test PRIVATE GTest::gtest_main)
gtest_discover_tests(gemm_config_test TEST_PREFIX "GemmConfig.")

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

# Register this test only for the original TensorFlow addon's cache target.
if(TARGET annc_jit_cache)
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

# Sink component contracts (stablehlo layer only).
if(ANNC_ENABLE_TF2ATIR_V2)
    add_executable(stablehlo_sink_component_test stablehlo_sink_component_test.cpp)
    target_include_directories(stablehlo_sink_component_test PRIVATE
        "${PROJECT_SOURCE_DIR}/annc/include" "${PROJECT_BINARY_DIR}/annc/include")
    target_link_libraries(stablehlo_sink_component_test PRIVATE
        GTest::gtest_main Threads::Threads StablehloToAtirConversion AtirDialect)
    gtest_discover_tests(stablehlo_sink_component_test
        TEST_PREFIX "StablehloSinkComponent.")
    add_executable(stablehlo_sink_pass_test stablehlo_sink_pass_test.cpp)
    target_include_directories(stablehlo_sink_pass_test PRIVATE
        "${PROJECT_SOURCE_DIR}/annc/include" "${PROJECT_BINARY_DIR}/annc/include")
    target_link_libraries(stablehlo_sink_pass_test PRIVATE
        GTest::gtest_main Threads::Threads StablehloToAtirConversion AtirDialect)
    gtest_discover_tests(stablehlo_sink_pass_test
        TEST_PREFIX "StablehloSinkPass.")
endif()
