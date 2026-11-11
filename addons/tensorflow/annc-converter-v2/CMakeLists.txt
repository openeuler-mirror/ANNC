# annc-converter-v2: TensorFlow GraphDef rewrite as a host-resolved PRELOAD
# plugin. The deployable form is libannc_graph_rewriter.so: its TF/protobuf
# symbols stay undefined and are resolved by the predictor process at load
# time (the tensorflow_addons ANNC_TENSORFLOW_PRELOAD mechanism), so no TF
# library is probed or linked and pip wheels stay out of the deployment.
# In-tree:    frontends/CMakeLists.txt under ANNC_ENABLE_TF2ATIR_V2.
# Standalone: cmake -S addons/tensorflow/annc-converter-v2 -B <build-dir>
#             -DANNC_TENSORFLOW_PRELOAD=ON -DANNC_TENSORFLOW_INCLUDE_DIR=<SDK>

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(annc-converter-v2 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
endif()

set(_v2_standalone FALSE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_v2_standalone TRUE)
endif()

set(ANNC_TENSORFLOW_INCLUDE_DIR "" CACHE PATH
  "Host TensorFlow SDK header root (required by the standalone plugin build)")
set(ANNC_TENSORFLOW_LIBRARIES "" CACHE STRING
  "TensorFlow shared libraries for the in-tree unit tests; empty uses Python TensorFlow")
set(ANNC_TENSORFLOW_CXX11_ABI "" CACHE STRING
  "Required _GLIBCXX_USE_CXX11_ABI value (0 or 1)")

# The PRELOAD switch is defined by tensorflow_addons in-tree (default ON, the
# serving form); standalone builds declare it here. The plugin is only
# produced with an explicit host SDK header root at hand.
if(_v2_standalone AND NOT DEFINED ANNC_TENSORFLOW_PRELOAD)
  option(ANNC_TENSORFLOW_PRELOAD
    "Build annc_graph_rewriter as a host-resolved PRELOAD plugin" OFF)
endif()
set(_v2_plugin OFF)
if(ANNC_TENSORFLOW_PRELOAD
   AND (_v2_standalone OR ANNC_TENSORFLOW_INCLUDE_DIR))
  set(_v2_plugin ON)
endif()
if(_v2_standalone)
  if(NOT ANNC_TENSORFLOW_PRELOAD)
    message(STATUS
      "annc-converter-v2: nothing to build in standalone mode without "
      "ANNC_TENSORFLOW_PRELOAD=ON (the standalone form is the plugin only).")
    return()
  endif()
  if(NOT ANNC_TENSORFLOW_INCLUDE_DIR)
    message(FATAL_ERROR
      "ANNC_TENSORFLOW_PRELOAD=ON requires an explicit ANNC_TENSORFLOW_INCLUDE_DIR "
      "(the host SDK headers); the plugin mode never probes a Python TensorFlow.")
  endif()
endif()

# Resolution flow of tensorflow_addons/CMakeLists.txt. Only the in-tree unit
# tests need a symbol provider (their executables link libtensorflow_framework
# through the core); the plugin itself compiles against headers alone.
function(annc_resolve_tensorflow out_include out_libraries out_compile_options)
  if(NOT ANNC_TENSORFLOW_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR "ANNC_TENSORFLOW_CXX11_ABI must be 0 or 1")
  endif()

  set(_compile_options)
  if(ANNC_TENSORFLOW_INCLUDE_DIR)
    get_filename_component(_include_dir
      "${ANNC_TENSORFLOW_INCLUDE_DIR}" ABSOLUTE)
    set(_include_source "explicit")
    set(_version_header "${_include_dir}/tensorflow/core/public/version.h")
    if(NOT EXISTS "${_version_header}")
      message(FATAL_ERROR
        "TensorFlow version header not found: ${_version_header}")
    endif()
    file(READ "${_version_header}" _version_text)
    if(_version_text MATCHES "#define TF_MAJOR_VERSION ([0-9]+)")
      if(NOT CMAKE_MATCH_1 EQUAL 2)
        message(FATAL_ERROR
          "annc-converter-v2 supports TensorFlow 2.x only; version.h says "
          "major=${CMAKE_MATCH_1}. The TF 1.15 line lives in "
          "tensorflow_addons/plugin/1_15.")
      endif()
    else()
      # Install-form SDK headers wrap version.h without the version macros;
      # the explicit path then relies on the caller providing a 2.x SDK
      # (a 1.15 framework would ship libtensorflow_framework.so.1).
      message(STATUS
        "annc-converter-v2: TF version not readable from the SDK headers; "
        "expecting a TensorFlow 2.x SDK on the explicit path.")
    endif()
  else()
    # Python TensorFlow: headers, compile flags and version all come from
    # one tf.sysconfig source.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import tensorflow as tf; print(tf.__version__)"
      OUTPUT_VARIABLE _tf_version
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _tf_version_result
      ERROR_QUIET)
    if(NOT _tf_version_result EQUAL 0 OR NOT _tf_version MATCHES "^2\.")
      message(FATAL_ERROR
        "annc-converter-v2 supports TensorFlow 2.x only; got "
        "'${_tf_version}'. The TF 1.15 line lives in "
        "tensorflow_addons/plugin/1_15.")
    endif()
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
      if(NOT _flag MATCHES "^-I" AND NOT _flag MATCHES "^--?std="
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

if(_v2_plugin AND NOT _v2_standalone)
  # In-tree plugin build: reuse whatever headers the test-oriented resolution
  # picked (wheel or SDK); the plugin links no library either way.
  annc_resolve_tensorflow(_v2_tf_include _v2_tf_framework _v2_tf_flags)
  message(STATUS
    "annc-converter-v2: plugin headers from the in-tree resolution (${_v2_tf_include})")
elseif(_v2_plugin)
  # Standalone plugin: headers only (same source as the host predictor).
  get_filename_component(_v2_tf_include
    "${ANNC_TENSORFLOW_INCLUDE_DIR}" ABSOLUTE)
  set(_v2_tf_framework "")
  set(_v2_tf_flags "")
  message(STATUS
    "annc-converter-v2: plugin headers: ${_v2_tf_include} (host SDK; no TF library probing)")
else()
  # In-tree without the plugin flag: resolve only for the unit tests.
  annc_resolve_tensorflow(_v2_tf_include _v2_tf_framework _v2_tf_flags)
endif()

# nlohmann/json: in-tree target from the repo root; standalone adds
# third_party/json.
if(NOT TARGET nlohmann_json::nlohmann_json)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/json" nlohmann_json)
endif()

# Compiled once as OBJECT pieces, consumed twice: the in-tree static library
# (which links libtensorflow_framework for the unit-test executables) and the
# PRELOAD plugin (which must not link it).
add_library(annc_converter_v2_core_objs OBJECT
  fusion_meta.cpp
  graphdef_rewriter.cpp
  graph_rewriter_entry.cpp
)
set_target_properties(annc_converter_v2_core_objs PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(annc_converter_v2_core_objs SYSTEM PUBLIC
  "${_v2_tf_include}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(annc_converter_v2_core_objs PUBLIC
  nlohmann_json::nlohmann_json)
target_compile_features(annc_converter_v2_core_objs PUBLIC cxx_std_17)
target_compile_options(annc_converter_v2_core_objs PUBLIC ${_v2_tf_flags})
target_compile_definitions(annc_converter_v2_core_objs PUBLIC
  _GLIBCXX_USE_CXX11_ABI=${ANNC_TENSORFLOW_CXX11_ABI}
  NDEBUG
)
# NDEBUG is a hard requirement for every TU that includes the TF headers'
# protobuf output; ANNC_DEBUG's global -UNDEBUG would re-enable the absl log
# references those headers emit and cannot link against.
if(ANNC_DEBUG)
  message(FATAL_ERROR
    "ANNC_DEBUG=ON is incompatible with annc-converter-v2; build with "
    "ANNC_DEBUG=OFF (standalone builds are unaffected).")
endif()

# Deployable PRELOAD plugin: keeps TF/protobuf symbols undefined (-z undefs
# wins over any earlier -z defs) for the host predictor process to resolve at
# load time.
if(_v2_plugin)
  add_library(annc_graph_rewriter SHARED
    $<TARGET_OBJECTS:annc_converter_v2_core_objs>
  )
  # CMake 3.16 does not propagate usage requirements through
  # $<TARGET_OBJECTS>; restate the includes here.
  target_include_directories(annc_graph_rewriter SYSTEM PRIVATE
    "${_v2_tf_include}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(annc_graph_rewriter PRIVATE
    nlohmann_json::nlohmann_json)
  target_compile_definitions(annc_graph_rewriter PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${ANNC_TENSORFLOW_CXX11_ABI}
    NDEBUG
  )
  target_link_options(annc_graph_rewriter PRIVATE
    "-Wl,--allow-shlib-undefined" "-Wl,-z,undefs")
  install(TARGETS annc_graph_rewriter DESTINATION lib)
endif()

if(_v2_standalone)
  return()
endif()

# In-tree static library wrapper: carries the framework link for the unit
# test executables (the plugin above deliberately does not inherit it).
add_library(annc_converter_v2_core STATIC
  $<TARGET_OBJECTS:annc_converter_v2_core_objs>
)
# CMake 3.16: usage requirements do not flow through $<TARGET_OBJECTS>;
# restate the public surface for the test executables.
target_include_directories(annc_converter_v2_core SYSTEM PUBLIC
  "${_v2_tf_include}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(annc_converter_v2_core PUBLIC
  _GLIBCXX_USE_CXX11_ABI=${ANNC_TENSORFLOW_CXX11_ABI}
  NDEBUG
)
target_compile_options(annc_converter_v2_core PUBLIC ${_v2_tf_flags})
target_link_libraries(annc_converter_v2_core PUBLIC
  nlohmann_json::nlohmann_json
  "${_v2_tf_framework}"
)
get_filename_component(_v2_tf_libdir "${_v2_tf_framework}" DIRECTORY)
set_target_properties(annc_converter_v2_core PROPERTIES
  BUILD_RPATH "${_v2_tf_libdir}"
  INSTALL_RPATH "${_v2_tf_libdir}"
)
