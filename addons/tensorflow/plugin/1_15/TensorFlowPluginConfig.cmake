# Explicit SDK settings override discovery from the target TensorFlow Python.
set(ANNC_TENSORFLOW_INCLUDE_DIR "" CACHE PATH "TensorFlow 1.15 SDK include directory")
set(ANNC_TENSORFLOW_LIBRARIES "" CACHE STRING
  "TF1.15 libtensorflow_cc and libtensorflow_framework for linked mode")
set(ANNC_TENSORFLOW_CXX_STANDARD "" CACHE STRING "TF1.15 SDK C++ standard (14 or 17)")
set(ANNC_TENSORFLOW_CXX11_ABI "" CACHE STRING "TF1.15 SDK libstdc++ ABI (0 or 1)")
option(ANNC_TENSORFLOW_PRELOAD "Resolve TensorFlow symbols from the host process" ON)

function(annc_resolve_tf1_15 out_include out_libraries)
  # A TF version does not identify the flags used to build its runtime.
  if(NOT ANNC_TENSORFLOW_CXX_STANDARD MATCHES "^(14|17)$")
    message(FATAL_ERROR
      "ANNC_TENSORFLOW_CXX_STANDARD must specify the SDK standard: 14 or 17")
  endif()
  if(NOT ANNC_TENSORFLOW_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR
      "ANNC_TENSORFLOW_CXX11_ABI must specify the SDK ABI: 0 or 1")
  endif()
  if(NOT ANNC_TENSORFLOW_PRELOAD MATCHES "^(ON|OFF)$")
    message(FATAL_ERROR "ANNC_TENSORFLOW_PRELOAD must be ON or OFF")
  endif()

  if(ANNC_TENSORFLOW_PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE "${ANNC_TENSORFLOW_PYTHON_EXECUTABLE}")
  endif()
  if(ANNC_TENSORFLOW_INCLUDE_DIR)
    get_filename_component(_include "${ANNC_TENSORFLOW_INCLUDE_DIR}" ABSOLUTE)
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import tensorflow as tf; print(tf.sysconfig.get_include())"
      OUTPUT_VARIABLE _include
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _include_result
      ERROR_VARIABLE _include_error)
    if(NOT _include_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to query Python TensorFlow headers: ${_include_error}")
    endif()
  endif()
  set(_version_header "${_include}/tensorflow/core/public/version.h")
  if(NOT _include OR NOT EXISTS "${_version_header}" OR
     IS_DIRECTORY "${_version_header}")
    message(FATAL_ERROR
      "ANNC_TENSORFLOW_INCLUDE_DIR must contain tensorflow/core/public/version.h")
  endif()
  file(READ "${_version_header}" _version)
  if(NOT _version MATCHES "(^|\n)#define[ \t]+TF_MAJOR_VERSION[ \t]+1([ \t\r\n]|$)" OR
     NOT _version MATCHES "(^|\n)#define[ \t]+TF_MINOR_VERSION[ \t]+15([ \t\r\n]|$)")
    message(FATAL_ERROR
      "ANNC_TENSORFLOW_INCLUDE_DIR must select TensorFlow 1.15 headers")
  endif()

  set(_libraries)
  if(ANNC_TENSORFLOW_PRELOAD)
    if(ANNC_TENSORFLOW_LIBRARIES)
      message(FATAL_ERROR
        "ANNC_TENSORFLOW_LIBRARIES is invalid with ANNC_TENSORFLOW_PRELOAD=ON")
    endif()
  else()
    set(_libraries ${ANNC_TENSORFLOW_LIBRARIES})
    if(NOT _libraries)
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
      # Cached library paths may belong to a previously selected SDK.
      unset(_tensorflow_cc CACHE)
      unset(_tensorflow_framework CACHE)
      set(_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
      list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.1)
      find_library(_tensorflow_cc NAMES tensorflow_cc
        PATHS "${_library_dir}" NO_DEFAULT_PATH)
      find_library(_tensorflow_framework NAMES tensorflow_framework
        PATHS "${_library_dir}" NO_DEFAULT_PATH)
      set(CMAKE_FIND_LIBRARY_SUFFIXES "${_saved_suffixes}")
      set(_libraries "${_tensorflow_cc}" "${_tensorflow_framework}")
    endif()
    set(_cc)
    set(_framework)
    foreach(_library IN LISTS _libraries)
      get_filename_component(_library "${_library}" ABSOLUTE)
      if(NOT EXISTS "${_library}" OR IS_DIRECTORY "${_library}")
        message(FATAL_ERROR "TF1.15 library file not found: ${_library}")
      endif()
      get_filename_component(_name "${_library}" NAME)
      if(_name MATCHES "^libtensorflow_cc\\.so(\\.1(\\.[0-9]+)*)?$")
        list(APPEND _cc "${_library}")
      elseif(_name MATCHES "^libtensorflow_framework\\.so(\\.1(\\.[0-9]+)*)?$")
        list(APPEND _framework "${_library}")
      else()
        message(FATAL_ERROR "Unexpected TF1.15 library filename: ${_name}")
      endif()
    endforeach()
    list(LENGTH _cc _cc_count)
    list(LENGTH _framework _framework_count)
    if(NOT _cc_count EQUAL 1 OR NOT _framework_count EQUAL 1)
      message(FATAL_ERROR
        "Linked mode requires exactly one libtensorflow_cc and one libtensorflow_framework")
    endif()
    # Paths and filenames do not prove binary compatibility; linking and
    # execution against the selected SDK must verify that separately.
    set(_libraries ${_cc} ${_framework})
  endif()

  set(${out_include} "${_include}" PARENT_SCOPE)
  set(${out_libraries} "${_libraries}" PARENT_SCOPE)
endfunction()
