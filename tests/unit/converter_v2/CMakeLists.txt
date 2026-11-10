# annc-converter-v2 unit tests (in-tree registration). The standalone build
# registers the same suites inside the addon's CMakeLists.txt instead.
find_package(GTest REQUIRED)
include(GoogleTest)

foreach(_test fusion_meta graphdef_rewriter)
  add_executable(${_test}_test ${_test}_test.cpp)
  target_link_libraries(${_test}_test PRIVATE
    annc_converter_v2_core GTest::gtest_main)
  gtest_discover_tests(${_test}_test TEST_PREFIX "ConverterV2.")
endforeach()
