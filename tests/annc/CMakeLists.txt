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
