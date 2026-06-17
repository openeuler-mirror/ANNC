option(ANNC_ENABLE_KDNN_ADAPTOR "Build ANNC builtin kernels backed by KDNN" OFF)

set(ANNC_THIRD_PARTY_KDNN_DIR "${CMAKE_SOURCE_DIR}/third_party/kdnn"
    CACHE PATH "Path to the KDNN third-party source tree")
set(ANNC_KDNN_GIT_REPOSITORY "https://gitcode.com/boostkit/kdnn.git"
    CACHE STRING "Git repository used to fetch KDNN when local sources are absent")
set(ANNC_KDNN_GIT_TAG "ccc8373f15652bbd99215c9812660651ade3d1b1"
    CACHE STRING "Git revision used when fetching KDNN")
set(ANNC_BUILD_THIRD_PARTY_KDNN "AUTO"
    CACHE STRING "Build KDNN from third_party/kdnn when ANNC_ENABLE_KDNN_ADAPTOR is ON")
set_property(CACHE ANNC_BUILD_THIRD_PARTY_KDNN PROPERTY STRINGS AUTO ON OFF)

if(TARGET_PLATFORM)
    set(_annc_default_kdnn_target_platform "${TARGET_PLATFORM}")
else()
    set(_annc_default_kdnn_target_platform "KP920")
endif()
set(ANNC_KDNN_TARGET_PLATFORM "${_annc_default_kdnn_target_platform}"
    CACHE STRING "TARGET_PLATFORM passed to the KDNN CMake build")
unset(_annc_default_kdnn_target_platform)
set(ANNC_KDNN_SOURCE_DIR "${ANNC_THIRD_PARTY_KDNN_DIR}/src/dnn")

if(NOT ANNC_ENABLE_KDNN_ADAPTOR)
    return()
endif()

include(ExternalProject)

if(TARGET KDNN::kdnn)
    return()
endif()

if(KDNN_INCLUDE_DIR AND KDNN_LIBRARY)
    add_library(KDNN::kdnn UNKNOWN IMPORTED GLOBAL)
    set_target_properties(KDNN::kdnn PROPERTIES
        IMPORTED_LOCATION "${KDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KDNN_INCLUDE_DIR}"
    )
    return()
endif()

if(ANNC_BUILD_THIRD_PARTY_KDNN STREQUAL "OFF")
    message(FATAL_ERROR
        "ANNC_ENABLE_KDNN_ADAPTOR is ON, but KDNN_INCLUDE_DIR/KDNN_LIBRARY "
        "were not provided and ANNC_BUILD_THIRD_PARTY_KDNN is OFF")
endif()

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
        "KDNN_INCLUDE_DIR and KDNN_LIBRARY.")
endif()

set(ANNC_KDNN_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/kdnn")
set(ANNC_KDNN_INSTALL_DIR "${ANNC_KDNN_BINARY_DIR}/install")
set(KDNN_INCLUDE_DIR "${ANNC_KDNN_SOURCE_DIR}/include"
    CACHE PATH "KDNN include directory" FORCE)
set(KDNN_LIBRARY "${ANNC_KDNN_INSTALL_DIR}/lib/libkdnn.so"
    CACHE FILEPATH "KDNN shared library" FORCE)

set(ANNC_KDNN_PATCH_FILE
    "${CMAKE_SOURCE_DIR}/patches/kdnn_rhs_packed/0001-kdnn-run-with-packed-b.patch")

ExternalProject_Add(annc_third_party_kdnn
    SOURCE_DIR "${ANNC_KDNN_SOURCE_DIR}"
    BINARY_DIR "${ANNC_KDNN_BINARY_DIR}"
    PATCH_COMMAND ${CMAKE_COMMAND}
        -DANNC_PATCH_FILE=${ANNC_KDNN_PATCH_FILE}
        -DANNC_PATCH_WORKDIR=${ANNC_THIRD_PARTY_KDNN_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/ApplyPatchIfNeeded.cmake
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
)
