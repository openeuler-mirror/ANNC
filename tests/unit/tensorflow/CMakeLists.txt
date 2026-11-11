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

if(NOT TARGET annc_tensorflow_support)
  return()
endif()

find_package(OpenSSL REQUIRED COMPONENTS Crypto)
set(_plugin "${PROJECT_SOURCE_DIR}/addons/tensorflow/plugin")
set(_integration "${PROJECT_SOURCE_DIR}/tests/integration/tensorflow")

# Keep C++17 kernel types on the support side of the C callback boundary.
add_library(tf_pool_test_kernel STATIC plugin/threadpool_test_kernel.cpp)
target_link_libraries(tf_pool_test_kernel PRIVATE ANNCThreadPool)
set_target_properties(tf_pool_test_kernel PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_compile_definitions(tf_pool_test_kernel PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

function(add_tf_caller_test target standard abi)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE "${_plugin}")
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD ${standard} CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
  target_compile_definitions(${target} PRIVATE _GLIBCXX_USE_CXX11_ABI=${abi})
endfunction()

foreach(_standard IN ITEMS 14 17)
  # Dispatch only forwards pointers. One caller per language standard suffices.
  if(_standard EQUAL 14)
    set(_dispatch_abi 0)
  else()
    set(_dispatch_abi 1)
  endif()
  set(_dispatch tf_dispatch_cxx${_standard}_abi${_dispatch_abi})
  add_tf_caller_test(${_dispatch} ${_standard} ${_dispatch_abi}
    plugin/ciface_dispatch_test.cpp)
  add_test(NAME ${_dispatch} COMMAND ${_dispatch})

  foreach(_abi IN ITEMS 0 1)
    set(_pool tf_threadpool_cxx${_standard}_abi${_abi})
    add_tf_caller_test(${_pool} ${_standard} ${_abi} plugin/threadpool_adapter_test.cpp)
    target_link_libraries(${_pool} PRIVATE annc_tensorflow_support tf_pool_test_kernel)
    add_test(NAME ${_pool} COMMAND ${_pool})

    if(TARGET annc-tf-pipeline)
      set(_jit tf_jit_cxx${_standard}_abi${_abi})
      add_tf_caller_test(${_jit} ${_standard} ${_abi}
        "${_integration}/jit_c_api_test.cpp" "${_plugin}/jit_cache.cc")
      target_link_libraries(${_jit} PRIVATE annc_tensorflow_support OpenSSL::Crypto Threads::Threads)
      set(_jit_inputs "${_integration}/Inputs/jit_add.mlir")
      set(_jit_env ${_tf_jit_env})
      if(ANNC_GEMM_PACKAGE_AVAILABLE)
        list(APPEND _jit_inputs "${_integration}/Inputs/jit_matmul.mlir")
        list(APPEND _jit_env "ANNC_GEMM=3")
      endif()
      add_test(NAME ${_jit} COMMAND ${_jit} ${_jit_inputs})
      set_tests_properties(${_jit} PROPERTIES ENVIRONMENT "${_jit_env}")
    endif()
  endforeach()
endforeach()
