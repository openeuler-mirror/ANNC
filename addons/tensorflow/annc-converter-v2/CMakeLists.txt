# annc-converter-v2: standalone TensorFlow GraphDef rewrite. TF dependency
# resolved exactly like tensorflow_addons (same variables, forwarded by
# build.sh), linking only libtensorflow_framework; never annc's LLVM/MLIR.
# Standalone: cmake -S addons/tensorflow/annc-converter-v2 -B <build-dir>
# In-tree:    frontends/CMakeLists.txt under ANNC_ENABLE_TF2ATIR_V2.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(annc-converter-v2 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
  include(CTest)
endif()

set(ANNC_TENSORFLOW_INCLUDE_DIR "" CACHE PATH
  "TensorFlow header root; empty uses Python TensorFlow")
set(ANNC_TENSORFLOW_LIBRARIES "" CACHE STRING
  "TensorFlow shared libraries; empty uses Python TensorFlow")
set(ANNC_TENSORFLOW_CXX11_ABI "" CACHE STRING
  "Required _GLIBCXX_USE_CXX11_ABI value (0 or 1)")

# Resolution flow of tensorflow_addons/CMakeLists.txt: explicit
# ANNC_TENSORFLOW_* first, else Python tf.sysconfig (headers, compile flags
# and libtensorflow_framework from one source).
function(annc_resolve_tensorflow out_include out_libraries out_compile_options)
  if(NOT ANNC_TENSORFLOW_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR "ANNC_TENSORFLOW_CXX11_ABI must be 0 or 1")
  endif()

  set(_compile_options)
  if(ANNC_TENSORFLOW_INCLUDE_DIR)
    get_filename_component(_include_dir
      "${ANNC_TENSORFLOW_INCLUDE_DIR}" ABSOLUTE)
    set(_include_source "explicit")
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import tensorflow as tf; print(tf.sysconfig.get_include())"
      OUTPUT_VARIABLE _include_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _include_result
      ERROR_VARIABLE _include_error)
    if(NOT _include_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to query Python TensorFlow headers: ${_include_error}")
    endif()
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import tensorflow as tf; print(chr(32).join(tf.sysconfig.get_compile_flags()))"
      OUTPUT_VARIABLE _compile_flags_string
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _compile_flags_result
      ERROR_VARIABLE _compile_flags_error)
    if(NOT _compile_flags_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to query Python TensorFlow compile flags: ${_compile_flags_error}")
    endif()
    separate_arguments(_compile_flags UNIX_COMMAND "${_compile_flags_string}")
    foreach(_flag IN LISTS _compile_flags)
      if(NOT _flag MATCHES "^-I" AND NOT _flag MATCHES "^--std="
         AND NOT _flag MATCHES "_GLIBCXX_USE_CXX11_ABI")
        list(APPEND _compile_options "${_flag}")
      endif()
    endforeach()
    set(_include_source "Python")
  endif()

  if(NOT EXISTS "${_include_dir}/tensorflow")
    message(FATAL_ERROR "TensorFlow headers not found at ${_include_dir}")
  endif()
  message(STATUS
    "annc-converter-v2: TensorFlow headers: ${_include_dir} (source: ${_include_source})")

  if(ANNC_TENSORFLOW_LIBRARIES)
    set(_libraries ${ANNC_TENSORFLOW_LIBRARIES})
    set(_library_source "explicit")
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import tensorflow as tf; print(tf.sysconfig.get_lib())"
      OUTPUT_VARIABLE _library_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _library_dir_result
      ERROR_VARIABLE _library_dir_error)
    if(NOT _library_dir_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to query Python TensorFlow libraries: ${_library_dir_error}")
    endif()
    set(_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2)
    find_library(_tensorflow_framework NAMES tensorflow_framework
      PATHS "${_library_dir}" NO_DEFAULT_PATH)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_saved_suffixes}")
    set(_libraries "${_tensorflow_framework}")
    set(_library_source "Python")
  endif()

  list(LENGTH _libraries _library_count)
  if(NOT _library_count EQUAL 1 OR NOT EXISTS "${_libraries}")
    message(FATAL_ERROR
      "annc-converter-v2 requires exactly one existing libtensorflow_framework "
      "in ANNC_TENSORFLOW_LIBRARIES (or a Python TensorFlow providing it); got: "
      "${_libraries}")
  endif()
  message(STATUS
    "annc-converter-v2: TensorFlow framework: ${_libraries} (source: ${_library_source})")

  set(${out_include} "${_include_dir}" PARENT_SCOPE)
  set(${out_libraries} "${_libraries}" PARENT_SCOPE)
  set(${out_compile_options} "${_compile_options}" PARENT_SCOPE)
endfunction()

annc_resolve_tensorflow(_v2_tf_include _v2_tf_framework _v2_tf_flags)

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
target_include_directories(annc_converter_v2_core SYSTEM PUBLIC
  "${_v2_tf_include}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_features(annc_converter_v2_core PUBLIC cxx_std_17)
target_compile_options(annc_converter_v2_core PUBLIC ${_v2_tf_flags})
target_compile_definitions(annc_converter_v2_core PUBLIC
  _GLIBCXX_USE_CXX11_ABI=${ANNC_TENSORFLOW_CXX11_ABI}
)
# NDEBUG is a hard requirement for every TU that includes the TF headers'
# protobuf output (see the resolution note above); PUBLIC so dependents
# inherit it. ANNC_DEBUG's global -UNDEBUG therefore breaks the link with
# absl symbols the headers reference but libtensorflow_framework hides.
if(ANNC_DEBUG)
  message(FATAL_ERROR
    "ANNC_DEBUG=ON is incompatible with annc-converter-v2; build with "
    "ANNC_DEBUG=OFF (standalone builds are unaffected).")
endif()
target_compile_definitions(annc_converter_v2_core PUBLIC NDEBUG)
target_link_libraries(annc_converter_v2_core PUBLIC
  nlohmann_json::nlohmann_json
  "${_v2_tf_framework}"
)
get_filename_component(_v2_tf_libdir "${_v2_tf_framework}" DIRECTORY)
set_target_properties(annc_converter_v2_core PROPERTIES
  BUILD_RPATH "${_v2_tf_libdir}"
  INSTALL_RPATH "${_v2_tf_libdir}"
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
