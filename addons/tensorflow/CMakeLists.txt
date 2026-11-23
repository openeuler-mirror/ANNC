set(ANNC_TENSORFLOW_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
  "Python interpreter for the target TensorFlow environment")
set(ANNC_TENSORFLOW_CXX_STANDARD "" CACHE STRING
  "TensorFlow plugin C++ standard; empty uses the target SDK")
set(ANNC_TENSORFLOW_CXX11_ABI "" CACHE STRING
  "TensorFlow plugin libstdc++ ABI; empty uses the target SDK")

# Keep the target TensorFlow interpreter local to the addon subtree.
if(ANNC_TENSORFLOW_PYTHON_EXECUTABLE)
  set(Python3_EXECUTABLE "${ANNC_TENSORFLOW_PYTHON_EXECUTABLE}")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import tensorflow as tf; print(tf.__version__)"
  OUTPUT_VARIABLE _tf_version
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _tf_version_result
  ERROR_VARIABLE _tf_version_error)
if(NOT _tf_version_result EQUAL 0)
  message(FATAL_ERROR "Failed to query TensorFlow version: ${_tf_version_error}")
endif()
message(STATUS "ANNC TensorFlow version: ${_tf_version}")

if(NOT _tf_version MATCHES "^(1\\.15|2\\.20)(\\.|$)")
  message(FATAL_ERROR
    "Unsupported TensorFlow version: ${_tf_version}; supported versions are 1.15 and 2.20.")
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import tensorflow as tf; print(chr(32).join(tf.sysconfig.get_compile_flags()))"
  OUTPUT_VARIABLE _tf_compile_flags_string
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _tf_compile_flags_result
  ERROR_VARIABLE _tf_compile_flags_error)
if(NOT _tf_compile_flags_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to query TensorFlow compile flags: ${_tf_compile_flags_error}")
endif()
separate_arguments(_tf_compile_flags UNIX_COMMAND "${_tf_compile_flags_string}")
set(_tf_cxx11_abi "")
set(_tf_cxx_standard "")
foreach(_flag IN LISTS _tf_compile_flags)
  if(_flag MATCHES "^-D_GLIBCXX_USE_CXX11_ABI=([01])$")
    set(_tf_cxx11_abi "${CMAKE_MATCH_1}")
  elseif(_flag MATCHES "^--?std=(c|gnu)\\+\\+([0-9]+)$")
    set(_tf_cxx_standard "${CMAKE_MATCH_2}")
  endif()
endforeach()

# Cache only explicit settings; detected values belong to the selected SDK.
if("${ANNC_TENSORFLOW_CXX11_ABI}" STREQUAL "")
  if("${_tf_cxx11_abi}" STREQUAL "")
    message(FATAL_ERROR
      "TensorFlow ${_tf_version} does not report its C++11 ABI; "
      "set ANNC_TENSORFLOW_CXX11_ABI explicitly.")
  endif()
  set(ANNC_TENSORFLOW_CXX11_ABI "${_tf_cxx11_abi}")
elseif(NOT "${_tf_cxx11_abi}" STREQUAL "" AND
       NOT "${ANNC_TENSORFLOW_CXX11_ABI}" STREQUAL "${_tf_cxx11_abi}")
  message(FATAL_ERROR
    "ANNC_TENSORFLOW_CXX11_ABI=${ANNC_TENSORFLOW_CXX11_ABI} "
    "conflicts with the target TensorFlow ABI ${_tf_cxx11_abi}.")
endif()

if("${ANNC_TENSORFLOW_CXX_STANDARD}" STREQUAL "")
  if("${_tf_cxx_standard}" STREQUAL "")
    message(FATAL_ERROR
      "TensorFlow ${_tf_version} does not report its C++ standard; "
      "set ANNC_TENSORFLOW_CXX_STANDARD explicitly.")
  endif()
  set(ANNC_TENSORFLOW_CXX_STANDARD "${_tf_cxx_standard}")
elseif(NOT "${_tf_cxx_standard}" STREQUAL "" AND
       NOT "${ANNC_TENSORFLOW_CXX_STANDARD}" STREQUAL "${_tf_cxx_standard}")
  message(FATAL_ERROR
    "ANNC_TENSORFLOW_CXX_STANDARD=${ANNC_TENSORFLOW_CXX_STANDARD} "
    "conflicts with the target TensorFlow C++ standard ${_tf_cxx_standard}.")
endif()
message(STATUS "ANNC TensorFlow CXX11 ABI: ${ANNC_TENSORFLOW_CXX11_ABI}")
message(STATUS "ANNC TensorFlow CXX Standard: ${ANNC_TENSORFLOW_CXX_STANDARD}")

if(_tf_version MATCHES "^1\\.15(\\.|$)")
  if(ANNC_ENABLE_ONNXRUNTIME_EP)
    message(FATAL_ERROR
      "TensorFlow 1.15 with ORT (--all) is not validated; use --enable-tf.")
  endif()
  add_subdirectory(plugin/1_15)
else()
  add_subdirectory("${PROJECT_SOURCE_DIR}/tensorflow_addons"
                   "${PROJECT_BINARY_DIR}/tensorflow_addons")
  # pb2stablehlo: wheel-isolated GraphDef -> StableHLO frontend, the producer
  # half of the V2 route (its artifacts feed the sink-side entry tool). Own
  # MLIR/LLVM headers (ABI-locked to the TF 2.20 wheel), so it deliberately
  # shares nothing with the toolchain setup above. Same switch as the sink
  # side: ANNC_ENABLE_TF2ATIR_V2.
  if(ANNC_ENABLE_TF2ATIR_V2)
    add_subdirectory(transform)
  endif()
endif()
