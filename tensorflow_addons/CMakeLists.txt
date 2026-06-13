cmake_minimum_required(VERSION 3.15)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(annc_optimizer CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED TENSORFLOW_INCLUDE_DIRS)
    set(TF_INCLUDE_DIR ${TENSORFLOW_INCLUDE_DIRS})
else()
    execute_process(
        COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
        OUTPUT_VARIABLE TF_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if(DEFINED TENSORFLOW_LIBRARY_DIR)
    set(TF_LIB_DIR ${TENSORFLOW_LIBRARY_DIR})
else()
    execute_process(
        COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
        OUTPUT_VARIABLE TF_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

execute_process(
    COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(TF_COMPILE_FLAGS UNIX_COMMAND "${TF_COMPILE_FLAGS_STRING}")

set(TF_EXTRA_COMPILE_OPTIONS)
foreach(TF_FLAG IN LISTS TF_COMPILE_FLAGS)
    if(TF_FLAG MATCHES "^-I")
        # Include directories are configured explicitly below.
    elseif(TF_FLAG MATCHES "^--std=")
        # CMAKE_CXX_STANDARD is the single source of truth for the C++ standard.
    else()
        list(APPEND TF_EXTRA_COMPILE_OPTIONS ${TF_FLAG})
    endif()
endforeach()

message(STATUS "TensorFlow include directory: ${TF_INCLUDE_DIR}")
message(STATUS "TensorFlow library directory: ${TF_LIB_DIR}")
message(STATUS "TensorFlow compile options: ${TF_EXTRA_COMPILE_OPTIONS}")

if(NOT EXISTS "${TF_INCLUDE_DIR}/tensorflow")
    message(FATAL_ERROR "TensorFlow headers not found at ${TF_INCLUDE_DIR}")
endif()

# Resolve TF libraries, preferring libraries passed from the parent build.
# When built standalone, search with .so.2 suffix support in case the
# unversioned .so symlink is absent.
if(DEFINED TENSORFLOW_LIBS)
    set(TENSORFLOW_LOCAL_LIBS ${TENSORFLOW_LIBS})
else()
    set(_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2)
    find_library(TENSORFLOW_FRAMEWORK_LIB
        NAMES tensorflow_framework
        PATHS ${TF_LIB_DIR}
        NO_DEFAULT_PATH
    )
    find_library(TENSORFLOW_CC_LIB
        NAMES tensorflow_cc
        PATHS ${TF_LIB_DIR}
        NO_DEFAULT_PATH
    )
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIGINAL_CMAKE_FIND_LIBRARY_SUFFIXES})
    if(NOT TENSORFLOW_FRAMEWORK_LIB OR NOT TENSORFLOW_CC_LIB)
        message(FATAL_ERROR "TensorFlow libraries not found in ${TF_LIB_DIR}")
    endif()
    set(TENSORFLOW_LOCAL_LIBS ${TENSORFLOW_FRAMEWORK_LIB} ${TENSORFLOW_CC_LIB})
endif()

add_compile_options(
    ${TF_EXTRA_COMPILE_OPTIONS}
)

include_directories(
    ${TF_INCLUDE_DIR}
    ${TF_INCLUDE_DIR}/external/nsync/public
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../annc/include
)

set(ANNC_OPTIMIZER_SOURCES
    annc_optimizer.cc
    annc_optimizer_register.cc
)

set(ANNC_FUSED_OP_SOURCES
    annc_fused_op.cc
    annc_fused_op_register.cc
)

add_library(annc_optimizer SHARED ${ANNC_OPTIMIZER_SOURCES})
add_library(annc_fused_op SHARED ${ANNC_FUSED_OP_SOURCES})

target_link_directories(annc_optimizer PRIVATE ${TF_LIB_DIR})
target_link_directories(annc_fused_op PRIVATE ${TF_LIB_DIR})

foreach(tgt annc_optimizer annc_fused_op)
    target_link_libraries(${tgt} PRIVATE ${TENSORFLOW_LOCAL_LIBS})
endforeach()
set_target_properties(annc_optimizer annc_fused_op PROPERTIES
    BUILD_RPATH "${TF_LIB_DIR}"
    INSTALL_RPATH "${TF_LIB_DIR}"
)

# When built standalone, preserve the original behavior of keeping the TF
# libraries linked even if no symbols are directly referenced.
if(NOT DEFINED TENSORFLOW_LIBS)
    set_target_properties(annc_optimizer annc_fused_op PROPERTIES
        LINK_FLAGS "-Wl,--no-as-needed"
    )
endif()

install(TARGETS annc_optimizer annc_fused_op
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
