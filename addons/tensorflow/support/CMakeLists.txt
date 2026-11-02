cmake_minimum_required(VERSION 3.15)

get_filename_component(_annc_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
find_package(Threads REQUIRED)

add_library(annc_tensorflow_support STATIC
  c_api.cc jit_compiler.cc threadpool_adapter.cc)
set_target_properties(annc_tensorflow_support PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
# This side has its own ABI; SDK settings must not propagate into it.
target_compile_definitions(annc_tensorflow_support PRIVATE
  _GLIBCXX_USE_CXX11_ABI=1)
target_include_directories(annc_tensorflow_support PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${_annc_root}/annc/include"
  "${_annc_root}/context/include")
target_link_libraries(annc_tensorflow_support PRIVATE Threads::Threads dl)
if(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(annc_tensorflow_support PRIVATE nlohmann_json::nlohmann_json)
else()
  target_include_directories(annc_tensorflow_support PRIVATE
    "${_annc_root}/third_party/json/include")
endif()
if(ANNC_ENABLE_KDNN_ADAPTOR)
  target_compile_definitions(annc_tensorflow_support PRIVATE ANNC_ENABLE_KDNN_ADAPTOR)
endif()
if(ANNC_ENABLE_CONSTANT_FOLDING)
  target_compile_definitions(annc_tensorflow_support PRIVATE ANNC_ENABLE_CONSTANT_FOLDING)
endif()
