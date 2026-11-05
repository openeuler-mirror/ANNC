if(TARGET annc-tf2atir)
  add_executable(op_coverage_test
    frontend/op_coverage_test.cpp
    "${PROJECT_SOURCE_DIR}/frontends/tensorflow/tf2atir/tf_mlir_builder.cpp"
    "${PROJECT_SOURCE_DIR}/frontends/tensorflow/tf2atir/tf_transformers.cpp"
  )
  target_include_directories(op_coverage_test PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/frontends/tensorflow/tf2atir")
  target_link_libraries(op_coverage_test PRIVATE
    GTest::gtest_main
    Threads::Threads
    AtirBuilder
    AtirDialect
    MLIRArithDialect
  )
  gtest_discover_tests(op_coverage_test)
endif()
