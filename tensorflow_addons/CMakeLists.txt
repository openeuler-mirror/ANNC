cmake_minimum_required(VERSION 3.15)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(annc_optimizer CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ANNC_TENSORFLOW_PRELOAD
    "Build an LD_PRELOAD plugin whose TensorFlow symbols come from its host" ON)
option(ANNC_WITH_THREADPOOL "Include ThreadPool in annc_fused_op" OFF)
set(ANNC_TENSORFLOW_INCLUDE_DIR "" CACHE PATH
    "TensorFlow header root; empty uses Python TensorFlow")
set(ANNC_TENSORFLOW_LIBRARIES "" CACHE STRING
    "TensorFlow shared libraries; empty uses Python TensorFlow in linked mode")
set(ANNC_TENSORFLOW_CXX11_ABI "" CACHE STRING
    "Required _GLIBCXX_USE_CXX11_ABI value (0 or 1)")

function(annc_resolve_tensorflow out_include out_libraries out_compile_options)
    if(NOT ANNC_TENSORFLOW_CXX11_ABI MATCHES "^[01]$")
        message(FATAL_ERROR
            "ANNC_TENSORFLOW_CXX11_ABI must be explicitly set to 0 or 1")
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
    message(STATUS "ANNC TensorFlow headers: ${_include_dir} (source: ${_include_source})")

    set(_libraries)
    if(ANNC_TENSORFLOW_PRELOAD)
        if(ANNC_TENSORFLOW_LIBRARIES)
            message(FATAL_ERROR
                "ANNC_TENSORFLOW_LIBRARIES is invalid when ANNC_TENSORFLOW_PRELOAD=ON")
        endif()
    elseif(ANNC_TENSORFLOW_LIBRARIES)
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
        find_library(_tensorflow_cc NAMES tensorflow_cc
            PATHS "${_library_dir}" NO_DEFAULT_PATH)
        find_library(_tensorflow_framework NAMES tensorflow_framework
            PATHS "${_library_dir}" NO_DEFAULT_PATH)
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${_saved_suffixes}")
        set(_libraries "${_tensorflow_cc}" "${_tensorflow_framework}")
        set(_library_source "Python")
    endif()

    if(NOT ANNC_TENSORFLOW_PRELOAD)
        list(LENGTH _libraries _library_count)
        if(NOT _library_count EQUAL 2)
            message(FATAL_ERROR
                "Linked mode requires libtensorflow_cc and libtensorflow_framework")
        endif()

        set(_tensorflow_cc_library)
        set(_tensorflow_framework_library)
        foreach(_library IN LISTS _libraries)
            if(NOT EXISTS "${_library}")
                message(FATAL_ERROR "TensorFlow library not found: ${_library}")
            endif()
            get_filename_component(_library_name "${_library}" NAME)
            if(_library_name MATCHES "^libtensorflow_cc\\.so(\\..*)?$")
                if(_tensorflow_cc_library)
                    message(FATAL_ERROR
                        "Linked mode requires exactly one libtensorflow_cc")
                endif()
                set(_tensorflow_cc_library "${_library}")
            elseif(_library_name MATCHES "^libtensorflow_framework\\.so(\\..*)?$")
                if(_tensorflow_framework_library)
                    message(FATAL_ERROR
                        "Linked mode requires exactly one libtensorflow_framework")
                endif()
                set(_tensorflow_framework_library "${_library}")
            else()
                message(FATAL_ERROR
                    "Unexpected TensorFlow library: ${_library_name}")
            endif()
        endforeach()
        if(NOT _tensorflow_cc_library OR NOT _tensorflow_framework_library)
            message(FATAL_ERROR
                "Linked mode requires libtensorflow_cc and libtensorflow_framework")
        endif()
        set(_libraries "${_tensorflow_cc_library}" "${_tensorflow_framework_library}")
        message(STATUS
            "ANNC TensorFlow libraries: ${_libraries} (source: ${_library_source})")
    endif()

    set(${out_include} "${_include_dir}" PARENT_SCOPE)
    set(${out_libraries} "${_libraries}" PARENT_SCOPE)
    set(${out_compile_options} "${_compile_options}" PARENT_SCOPE)
endfunction()

annc_resolve_tensorflow(ANNC_TF_INCLUDE_DIR ANNC_TF_LIBRARIES
                        ANNC_TF_COMPILE_OPTIONS)


# ── Source files ─────────────────────────────────────────────────────────────
set(ANNC_OPTIMIZER_SOURCES
    annc_optimizer.cc
    annc_optimizer_register.cc
)

set(ANNC_FUSED_OP_SOURCES
    annc_fused_op.cc
    annc_fused_op_jit.cc
    annc_fused_op_register.cc
)

# ── Build targets ────────────────────────────────────────────────────────────
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_library(annc_jit_cache STATIC annc_jit_cache.cc)
target_include_directories(annc_jit_cache PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(annc_jit_cache PUBLIC OpenSSL::Crypto PRIVATE dl)

add_library(annc_optimizer SHARED ${ANNC_OPTIMIZER_SOURCES})
add_library(annc_fused_op SHARED ${ANNC_FUSED_OP_SOURCES})
if(ANNC_ENABLE_CONSTANT_FOLDING)
    target_compile_definitions(annc_fused_op PRIVATE
        ANNC_ENABLE_CONSTANT_FOLDING)
endif()
target_link_libraries(annc_fused_op PRIVATE annc_jit_cache)
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(annc_fused_op PRIVATE nlohmann_json::nlohmann_json)
else()
    target_include_directories(annc_fused_op PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/include
    )
endif()

foreach(_target annc_optimizer annc_fused_op)
    target_include_directories(${_target} PRIVATE
        "${ANNC_TF_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../annc/include")
    target_compile_definitions(${_target} PRIVATE
        "_GLIBCXX_USE_CXX11_ABI=${ANNC_TENSORFLOW_CXX11_ABI}")
    target_compile_options(${_target} PRIVATE ${ANNC_TF_COMPILE_OPTIONS})
endforeach()

if(ANNC_TENSORFLOW_PRELOAD)
    target_link_libraries(annc_optimizer PRIVATE dl pthread)
    target_include_directories(annc_fused_op PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/annc")
    foreach(_target annc_optimizer annc_fused_op)
        target_link_options(${_target} PRIVATE
            "-Wl,--allow-shlib-undefined" "-Wl,-z,undefs")
    endforeach()
else()
    set(_annc_tf_rpath_dirs)
    foreach(_library IN LISTS ANNC_TF_LIBRARIES)
        get_filename_component(_library_dir "${_library}" DIRECTORY)
        list(APPEND _annc_tf_rpath_dirs "${_library_dir}")
    endforeach()
    list(REMOVE_DUPLICATES _annc_tf_rpath_dirs)
    foreach(_target annc_optimizer annc_fused_op)
        target_link_libraries(${_target} PRIVATE ${ANNC_TF_LIBRARIES})
        set_target_properties(${_target} PROPERTIES
            BUILD_RPATH "${_annc_tf_rpath_dirs}"
            INSTALL_RPATH "${_annc_tf_rpath_dirs}")
    endforeach()
endif()

install(TARGETS annc_optimizer annc_fused_op
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
