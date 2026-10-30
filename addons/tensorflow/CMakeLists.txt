set(ANNC_TENSORFLOW_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
  "Python interpreter for the target TensorFlow environment")
set(ANNC_TENSORFLOW_CXX_STANDARD 17 CACHE STRING "TensorFlow plugin C++ standard")
set(ANNC_TENSORFLOW_CXX11_ABI 1 CACHE STRING "TensorFlow plugin libstdc++ ABI")

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

if(_tf_version MATCHES "^1\\.15(\\.|$)")
  if(ANNC_ENABLE_ONNXRUNTIME_EP)
    message(FATAL_ERROR
      "TensorFlow 1.15 with ORT (--all) is not validated; use --enable-tf.")
  endif()
  add_subdirectory(plugin/1_15)
elseif(_tf_version MATCHES "^2\\.20(\\.|$)")
  add_subdirectory("${PROJECT_SOURCE_DIR}/tensorflow_addons"
                   "${PROJECT_BINARY_DIR}/tensorflow_addons")
else()
  message(FATAL_ERROR
    "Unsupported TensorFlow version: ${_tf_version}; supported versions are 1.15 and 2.20.")
endif()
