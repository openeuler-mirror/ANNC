cmake_minimum_required(VERSION 3.15)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(annc_tensorflow_1_15 LANGUAGES CXX)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/TensorFlowPluginConfig.cmake")
annc_resolve_tf1_15(_tf_include _tf_libraries)
get_filename_component(_annc_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
get_filename_component(_plugin "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(Threads REQUIRED)
if(NOT TARGET annc_tensorflow_support)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../support"
                   "${CMAKE_CURRENT_BINARY_DIR}/support")
endif()

add_library(annc_jit_cache_1_15 STATIC "${_plugin}/jit_cache.cc")
target_include_directories(annc_jit_cache_1_15 PUBLIC "${_plugin}")
target_link_libraries(annc_jit_cache_1_15 PUBLIC OpenSSL::Crypto
                      annc_tensorflow_support)

add_library(annc_fused_op_1_15 SHARED
  "${_plugin}/fused_op.cc" "${_plugin}/fused_op_register.cc"
  "${_plugin}/fused_op_jit.cc" "${_plugin}/fused_op_context.cc"
  threadpool_callbacks.cc)
target_link_libraries(annc_fused_op_1_15 PRIVATE
  annc_jit_cache_1_15 Threads::Threads dl)

add_library(annc_optimizer_1_15 SHARED
  "${_plugin}/optimizer.cc" "${_plugin}/optimizer_register.cc")
target_link_libraries(annc_optimizer_1_15 PRIVATE Threads::Threads dl)

foreach(_target annc_jit_cache_1_15 annc_fused_op_1_15 annc_optimizer_1_15)
  set_target_properties(${_target} PROPERTIES
    CXX_STANDARD ${ANNC_TENSORFLOW_CXX_STANDARD}
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(${_target} PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${ANNC_TENSORFLOW_CXX11_ABI})
  target_include_directories(${_target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${_plugin}" "${_annc_root}/annc/include"
    "${_annc_root}/context/include")
endforeach()
set_target_properties(annc_jit_cache_1_15 PROPERTIES
  OUTPUT_NAME annc_jit_cache
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/tensorflow/1.15")

set(_tf_rpath_dirs)
foreach(_library IN LISTS _tf_libraries)
  get_filename_component(_library_dir "${_library}" DIRECTORY)
  list(APPEND _tf_rpath_dirs "${_library_dir}")
endforeach()
list(REMOVE_DUPLICATES _tf_rpath_dirs)
foreach(_name annc_fused_op annc_optimizer)
  set(_target "${_name}_1_15")
  target_include_directories(${_target} SYSTEM PRIVATE "${_tf_include}")
  set_target_properties(${_target} PROPERTIES
    OUTPUT_NAME ${_name}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/tensorflow/1.15")
  if(ANNC_TENSORFLOW_PRELOAD)
    target_link_options(${_target} PRIVATE
      "-Wl,--allow-shlib-undefined" "-Wl,-z,undefs")
  else()
    target_link_libraries(${_target} PRIVATE ${_tf_libraries})
    set_target_properties(${_target} PROPERTIES
      BUILD_RPATH "${_tf_rpath_dirs}" INSTALL_RPATH "${_tf_rpath_dirs}")
  endif()
  install(TARGETS ${_target} LIBRARY DESTINATION lib/tensorflow/1.15)
endforeach()
