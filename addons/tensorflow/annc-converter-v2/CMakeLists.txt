# annc-converter-v2: standalone TensorFlow GraphDef rewrite. TF/LLVM process
# isolation: links ONLY the pip TF wheel, never annc's LLVM/MLIR targets.
# Standalone: cmake -S addons/tensorflow/annc-converter-v2 -B <build-dir>
# In-tree:    frontends/CMakeLists.txt under ANNC_ENABLE_TF2ATIR_V2.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(annc-converter-v2 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
  include(CTest)
endif()

# TensorFlow wheel discovery: prefer TENSORFLOW_ROOT, else probe python3 via
# tensorflow.sysconfig (headers and runtime lib from the same wheel).
set(TENSORFLOW_ROOT "" CACHE PATH
  "pip TensorFlow wheel root (site-packages/tensorflow); probed from python3 when empty")
set(_v2_tf_include "")
set(_v2_tf_lib "")
if(NOT "${TENSORFLOW_ROOT}" STREQUAL "")
  set(_v2_tf_include "${TENSORFLOW_ROOT}/include")
  set(_v2_tf_lib "${TENSORFLOW_ROOT}")
else()
  execute_process(
    COMMAND python3 -c
      "import tensorflow as tf; print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib())"
    OUTPUT_VARIABLE _v2_tf_probe
    RESULT_VARIABLE _v2_tf_probe_rc
    ERROR_QUIET)
  if(NOT _v2_tf_probe_rc EQUAL 0)
    message(FATAL_ERROR
      "annc-converter-v2 requires a pip TensorFlow installation: set "
      "TENSORFLOW_ROOT=<site-packages/tensorflow> or make 'import "
      "tensorflow' work under python3.")
  endif()
  string(REPLACE "\n" ";" _v2_tf_lines "${_v2_tf_probe}")
  list(GET _v2_tf_lines 0 _v2_tf_include)
  list(GET _v2_tf_lines 1 _v2_tf_lib)
  string(STRIP "${_v2_tf_include}" _v2_tf_include)
  string(STRIP "${_v2_tf_lib}" _v2_tf_lib)
endif()
if(NOT EXISTS "${_v2_tf_include}/tensorflow/core/framework/graph.pb.h")
  message(FATAL_ERROR
    "TensorFlow wheel headers not found under '${_v2_tf_include}'. Set "
    "TENSORFLOW_ROOT to the pip site-packages/tensorflow directory.")
endif()
file(GLOB _v2_tf_framework "${_v2_tf_lib}/libtensorflow_framework.so*")
list(SORT _v2_tf_framework)
if(_v2_tf_framework)
  list(GET _v2_tf_framework 0 _v2_tf_framework)
endif()
if(NOT _v2_tf_framework)
  message(FATAL_ERROR
    "libtensorflow_framework.so.* not found under '${_v2_tf_lib}'. Set "
    "TENSORFLOW_ROOT to the pip site-packages/tensorflow directory.")
endif()
message(STATUS "annc-converter-v2: TF wheel include: ${_v2_tf_include}")

# nlohmann/json: in-tree target from the repo root; standalone adds
# third_party/json.
if(NOT TARGET nlohmann_json::nlohmann_json)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/json" nlohmann_json)
endif()

# Core library shared by the unit tests and the executable (later commit).
add_library(annc_converter_v2_core STATIC
  fusion_meta.cpp
  graphdef_rewriter.cpp
)
target_include_directories(annc_converter_v2_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_features(annc_converter_v2_core PUBLIC cxx_std_17)
# ANNC_DEBUG's global -UNDEBUG cancels the NDEBUG below; the wheel's protobuf
# headers then reference absl log symbols that libtensorflow_framework hides,
# and linking fails. The two options cannot combine.
if(ANNC_DEBUG)
  message(FATAL_ERROR
    "ANNC_DEBUG=ON is incompatible with annc-converter-v2; build with "
    "ANNC_DEBUG=OFF (standalone builds are unaffected).")
endif()

# NDEBUG is a hard requirement for every TU that includes the wheel's
# protobuf headers (see the wheel discovery note above); PUBLIC so dependents
# inherit it.
target_compile_definitions(annc_converter_v2_core PUBLIC NDEBUG)
target_include_directories(annc_converter_v2_core SYSTEM PUBLIC
  "${_v2_tf_include}"
)
target_link_libraries(annc_converter_v2_core PUBLIC
  "${_v2_tf_framework}"
  nlohmann_json::nlohmann_json
)
set_target_properties(annc_converter_v2_core PROPERTIES
  BUILD_RPATH "${_v2_tf_lib}"
  INSTALL_RPATH "${_v2_tf_lib}"
)

# Unit tests live in the shared test tree; ctest runs the same cases.
if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  add_executable(fusion_meta_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/unit/converter_v2/fusion_meta_test.cpp"
  )
  target_link_libraries(fusion_meta_test PRIVATE
    annc_converter_v2_core
    GTest::gtest_main
  )
  gtest_discover_tests(fusion_meta_test
    TEST_PREFIX "FusionMeta."
  )
  add_executable(graphdef_rewriter_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/unit/converter_v2/graphdef_rewriter_test.cpp"
  )
  target_link_libraries(graphdef_rewriter_test PRIVATE
    annc_converter_v2_core
    GTest::gtest_main
  )
  gtest_discover_tests(graphdef_rewriter_test
    TEST_PREFIX "GraphDefRewriter."
  )
endif()
