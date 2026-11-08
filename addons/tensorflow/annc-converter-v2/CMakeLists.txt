# annc-converter-v2: standalone TensorFlow GraphDef rewrite. TF/LLVM process
# isolation: links ONLY the pip TF wheel, never annc's LLVM/MLIR targets.
# Standalone: cmake -S addons/tensorflow/annc-converter-v2 -B <build-dir>
# In-tree:    frontends/CMakeLists.txt under ANNC_ENABLE_TF_WHEEL_CONVERTER.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(annc-converter-v2 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
  include(CTest)
endif()

# nlohmann/json: in-tree target when configured from the repository root;
# standalone builds add third_party/json.
if(NOT TARGET nlohmann_json::nlohmann_json)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/json" nlohmann_json)
endif()

# Core library shared by the unit tests and the executable (later commit).
add_library(annc_converter_v2_core STATIC
  fusion_meta.cpp
)
target_include_directories(annc_converter_v2_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_features(annc_converter_v2_core PUBLIC cxx_std_17)
target_link_libraries(annc_converter_v2_core PUBLIC
  nlohmann_json::nlohmann_json
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
endif()
