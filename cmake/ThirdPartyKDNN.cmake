option(ANNC_ENABLE_KDNN_ADAPTOR "Build ANNC builtin kernels backed by KDNN" OFF)

set(ANNC_KDNN_SOURCE "LOCAL"
    CACHE STRING "KDNN source: LOCAL uses ANNC_KDNN_DIR; REMOTE fetches from ANNC_KDNN_GIT_REPOSITORY")
set_property(CACHE ANNC_KDNN_SOURCE PROPERTY STRINGS LOCAL REMOTE)

set(ANNC_KDNN_DIR "${CMAKE_SOURCE_DIR}/third_party/KDNN"
    CACHE PATH "Path to a local KDNN tree or installation root")
if(DEFINED KDNN_DIR AND NOT "${KDNN_DIR}" STREQUAL ""
   AND NOT "${KDNN_DIR}" STREQUAL "${ANNC_KDNN_DIR}")
    set(ANNC_KDNN_DIR "${KDNN_DIR}" CACHE PATH "Path to a local KDNN tree or installation root" FORCE)
else()
    set(KDNN_DIR "${ANNC_KDNN_DIR}" CACHE PATH "Alias for ANNC_KDNN_DIR" FORCE)
endif()

set(ANNC_THIRD_PARTY_KDNN_DIR "${CMAKE_SOURCE_DIR}/third_party/kdnn"
    CACHE PATH "Path used when fetching KDNN from git")
set(ANNC_KDNN_GIT_REPOSITORY "https://gitcode.com/boostkit/kdnn.git"
    CACHE STRING "Git repository used to fetch KDNN")
set(ANNC_KDNN_GIT_TAG "ccc8373f15652bbd99215c9812660651ade3d1b1"
    CACHE STRING "Git revision used when fetching KDNN")

if(TARGET_PLATFORM)
    set(_annc_default_kdnn_target_platform "${TARGET_PLATFORM}")
else()
    set(_annc_default_kdnn_target_platform "KP920")
endif()
set(ANNC_KDNN_TARGET_PLATFORM "${_annc_default_kdnn_target_platform}"
    CACHE STRING "TARGET_PLATFORM passed to the KDNN CMake build")
unset(_annc_default_kdnn_target_platform)

if(NOT ANNC_ENABLE_KDNN_ADAPTOR)
    return()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
set(_annc_kdnn_openmp_link_flags "${OpenMP_CXX_LIBRARIES}")
if(NOT _annc_kdnn_openmp_link_flags)
    set(_annc_kdnn_openmp_link_flags "${OpenMP_CXX_FLAGS}")
endif()
string(REPLACE ";" " " _annc_kdnn_openmp_link_flags "${_annc_kdnn_openmp_link_flags}")
set(ANNC_KDNN_OPENMP_LINK_FLAGS "${_annc_kdnn_openmp_link_flags}"
    CACHE STRING "OpenMP link flags required by KDNN" FORCE)

if(TARGET KDNN::kdnn)
    return()
endif()

function(annc_import_kdnn kdnn_root)
    set(_kdnn_include_dir "${kdnn_root}/include")
    set(_kdnn_library "")
    if(EXISTS "${kdnn_root}/lib/libkdnn.so")
        set(_kdnn_library "${kdnn_root}/lib/libkdnn.so")
    elseif(EXISTS "${kdnn_root}/lib/libkdnn.a")
        set(_kdnn_library "${kdnn_root}/lib/libkdnn.a")
    elseif(EXISTS "${kdnn_root}/src/libkdnn.so")
        set(_kdnn_library "${kdnn_root}/src/libkdnn.so")
    elseif(EXISTS "${kdnn_root}/src/libkdnn.a")
        set(_kdnn_library "${kdnn_root}/src/libkdnn.a")
    endif()

    if(NOT EXISTS "${_kdnn_include_dir}/kdnn.hpp"
       OR NOT EXISTS "${_kdnn_include_dir}/kdnn_config.h"
       OR NOT EXISTS "${_kdnn_library}")
        message(FATAL_ERROR
            "ANNC_ENABLE_KDNN_ADAPTOR is ON, but KDNN was not found under "
            "${kdnn_root}. Expected include/kdnn.hpp, include/kdnn_config.h, "
            "and lib/libkdnn.* or src/libkdnn.*.")
    endif()
    if(ANNC_ENABLE_CONSTANT_FOLDING)
        set(_kdnn_gemm_header "${_kdnn_include_dir}/operations/kdnn_gemm.hpp")
        if(NOT EXISTS "${_kdnn_gemm_header}")
            message(FATAL_ERROR
                "ANNC_ENABLE_CONSTANT_FOLDING is ON, but ${_kdnn_gemm_header} "
                "was not found.")
        endif()
        file(STRINGS "${_kdnn_gemm_header}" _kdnn_packed_b_api
             REGEX "RunWithPackedB")
        if(NOT _kdnn_packed_b_api)
            message(FATAL_ERROR
                "ANNC_ENABLE_CONSTANT_FOLDING is ON, but local KDNN at "
                "${kdnn_root} does not provide KDNN::Gemm::RunWithPackedB. "
                "Use a patched local KDNN or set ANNC_KDNN_SOURCE=REMOTE "
                "so ANNC can apply patches/kdnn_rhs_packed.")
        endif()
    endif()

    get_filename_component(_kdnn_lib_dir "${_kdnn_library}" DIRECTORY)
    set(KDNN_INCLUDE_DIR "${_kdnn_include_dir}" CACHE PATH "KDNN include directory" FORCE)
    set(KDNN_LIBRARY "${_kdnn_library}" CACHE FILEPATH "KDNN library" FORCE)
    set(KDNN_LIB_DIR "${_kdnn_lib_dir}" CACHE PATH "KDNN library directory" FORCE)

    add_library(KDNN::kdnn UNKNOWN IMPORTED GLOBAL)
    set_target_properties(KDNN::kdnn PROPERTIES
        IMPORTED_LOCATION "${KDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KDNN_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX
    )
endfunction()

if(ANNC_KDNN_SOURCE STREQUAL "LOCAL")
    annc_import_kdnn("${ANNC_KDNN_DIR}")
    message(STATUS "Using local KDNN from ${ANNC_KDNN_DIR}")
    return()
endif()

if(NOT ANNC_KDNN_SOURCE STREQUAL "REMOTE")
    message(FATAL_ERROR
        "Invalid ANNC_KDNN_SOURCE='${ANNC_KDNN_SOURCE}'. "
        "Expected LOCAL or REMOTE.")
endif()

include(ExternalProject)
set(ANNC_KDNN_SOURCE_DIR "${ANNC_THIRD_PARTY_KDNN_DIR}/src/dnn")

FetchContent_Declare(kdnn
    GIT_REPOSITORY ${ANNC_KDNN_GIT_REPOSITORY}
    GIT_TAG        ${ANNC_KDNN_GIT_TAG}
    GIT_SHALLOW    TRUE
    SOURCE_DIR     ${ANNC_THIRD_PARTY_KDNN_DIR}
)
FetchContent_GetProperties(kdnn)
if(NOT kdnn_POPULATED)
    if(EXISTS "${ANNC_KDNN_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "KDNN source already exists, skipping fetch")
        set(kdnn_POPULATED ON CACHE INTERNAL "kdnn populated" FORCE)
        set(kdnn_SOURCE_DIR "${ANNC_THIRD_PARTY_KDNN_DIR}" CACHE INTERNAL "kdnn source dir" FORCE)
        set(kdnn_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/kdnn-build" CACHE INTERNAL "kdnn binary dir" FORCE)
    else()
        message(STATUS
            "Fetching KDNN from ${ANNC_KDNN_GIT_REPOSITORY} (${ANNC_KDNN_GIT_TAG}) ...")
        FetchContent_Populate(kdnn)
    endif()
endif()

if(NOT EXISTS "${ANNC_KDNN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "ANNC_ENABLE_KDNN_ADAPTOR is ON, but KDNN source was not found at "
        "${ANNC_KDNN_SOURCE_DIR}. Fetch KDNN failed or provide "
        "ANNC_KDNN_DIR and use ANNC_KDNN_SOURCE=LOCAL.")
endif()

set(ANNC_KDNN_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/kdnn")
set(ANNC_KDNN_INSTALL_DIR "${ANNC_KDNN_BINARY_DIR}/install")
set(KDNN_INCLUDE_DIR "${ANNC_KDNN_SOURCE_DIR}/include"
    CACHE PATH "KDNN include directory" FORCE)
set(KDNN_LIBRARY "${ANNC_KDNN_INSTALL_DIR}/lib/libkdnn.so"
    CACHE FILEPATH "KDNN shared library" FORCE)
set(KDNN_LIB_DIR "${ANNC_KDNN_INSTALL_DIR}/lib"
    CACHE PATH "KDNN library directory" FORCE)

set(_annc_kdnn_patch_command "")
if(ANNC_ENABLE_CONSTANT_FOLDING)
    set(ANNC_KDNN_PATCH_FILE
        "${CMAKE_SOURCE_DIR}/patches/kdnn_rhs_packed/0001-kdnn-run-with-packed-b.patch")
    set(_annc_kdnn_patch_command
        ${CMAKE_COMMAND}
            -DANNC_PATCH_FILE=${ANNC_KDNN_PATCH_FILE}
            -DANNC_PATCH_WORKDIR=${ANNC_THIRD_PARTY_KDNN_DIR}
            -P ${CMAKE_SOURCE_DIR}/cmake/ApplyPatchIfNeeded.cmake)
endif()

ExternalProject_Add(annc_third_party_kdnn
    SOURCE_DIR "${ANNC_KDNN_SOURCE_DIR}"
    BINARY_DIR "${ANNC_KDNN_BINARY_DIR}"
    PATCH_COMMAND ${_annc_kdnn_patch_command}
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${ANNC_KDNN_INSTALL_DIR}
        -DTARGET_PLATFORM=${ANNC_KDNN_TARGET_PLATFORM}
    BUILD_BYPRODUCTS "${KDNN_LIBRARY}"
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

add_library(KDNN::kdnn SHARED IMPORTED GLOBAL)
add_dependencies(KDNN::kdnn annc_third_party_kdnn)
set_target_properties(KDNN::kdnn PROPERTIES
    IMPORTED_LOCATION "${KDNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${KDNN_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX
)
