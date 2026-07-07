option(ANNC_ENABLE_KDNN_ADAPTOR "Build ANNC builtin kernels backed by KDNN" OFF)

set(ANNC_KDNN_SOURCE "LOCAL"
    CACHE STRING "KDNN source: LOCAL uses ANNC_KDNN_DIR; REMOTE fetches from ANNC_KDNN_GIT_REPOSITORY; RELEASE downloads a release zip")
set_property(CACHE ANNC_KDNN_SOURCE PROPERTY STRINGS LOCAL REMOTE RELEASE)

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
set(ANNC_KDNN_GIT_TAG "v3.1.0"
    CACHE STRING "Git revision used when fetching KDNN")

set(ANNC_KDNN_RELEASE_URL
    "https://gitcode.com/boostkit/boostsra/releases/download/v1.2.0/BoostKit-boostcore-kdnn_3.1.0.zip"
    CACHE STRING "URL of the KDNN release zip package")

set(ANNC_KDNN_RELEASE_SHA256
    "61a4b0b55a80ca742b43dde638b7fdd63c7ef36f26f3615e4f1ad008750217a8"
    CACHE STRING "Expected SHA256 of the KDNN release zip package (empty to skip)")

set(ANNC_KDNN_LIB_VARIANT "sve-threadpool"
    CACHE STRING "KDNN library variant to use from the release package")
set_property(CACHE ANNC_KDNN_LIB_VARIANT PROPERTY STRINGS
    sve-threadpool sve-omp sve2-threadpool sve2-omp)

set(ANNC_KDNN_RELEASE_DIR "${CMAKE_SOURCE_DIR}/third_party/kdnn-release"
    CACHE PATH "Directory where the release KDNN package is extracted")

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

if(ANNC_KDNN_SOURCE STREQUAL "RELEASE")
    if(ANNC_ENABLE_CONSTANT_FOLDING)
        message(FATAL_ERROR
            "ANNC_ENABLE_CONSTANT_FOLDING is not supported with "
            "ANNC_KDNN_SOURCE=RELEASE in this version.")
    endif()

    if(NOT ANNC_KDNN_LIB_VARIANT MATCHES "^(sve-threadpool|sve-omp|sve2-threadpool|sve2-omp)$")
        message(FATAL_ERROR
            "Invalid ANNC_KDNN_LIB_VARIANT='${ANNC_KDNN_LIB_VARIANT}'. "
            "Expected one of: sve-threadpool, sve-omp, sve2-threadpool, sve2-omp.")
    endif()

    set(_kdnn_release_root "${ANNC_KDNN_RELEASE_DIR}")
    set(_kdnn_zip_file "${_kdnn_release_root}/BoostKit-boostcore-kdnn_3.1.0.zip")
    set(_kdnn_rpm_file "${_kdnn_release_root}/boostcore-kdnn-3.1.0-1.aarch64.rpm")
    set(_kdnn_extract_root "${_kdnn_release_root}/extract")
    set(_kdnn_stamp_file "${_kdnn_release_root}/.annc_kdnn_release_stamp")
    set(_kdnn_stamp_content "${ANNC_KDNN_RELEASE_URL}\n${ANNC_KDNN_LIB_VARIANT}")

    set(_kdnn_release_cache_valid FALSE)
    if(EXISTS "${_kdnn_release_root}/include/kdnn.hpp"
       AND EXISTS "${_kdnn_release_root}/src/libkdnn.a"
       AND EXISTS "${_kdnn_stamp_file}")
        file(READ "${_kdnn_stamp_file}" _kdnn_existing_stamp)
        if(_kdnn_existing_stamp STREQUAL "${_kdnn_stamp_content}\n")
            set(_kdnn_release_cache_valid TRUE)
        else()
            message(STATUS "KDNN release configuration changed, clearing staged KDNN tree")
            file(REMOVE_RECURSE "${_kdnn_release_root}/include")
            file(REMOVE_RECURSE "${_kdnn_release_root}/src")
            file(REMOVE "${_kdnn_stamp_file}")
        endif()
    endif()

    if(NOT _kdnn_release_cache_valid)
        if(NOT EXISTS "${_kdnn_rpm_file}" AND NOT EXISTS "${_kdnn_zip_file}")
            message(STATUS "Downloading KDNN release from ${ANNC_KDNN_RELEASE_URL} ...")
            set(_kdnn_download_hash_args "")
            if(NOT "${ANNC_KDNN_RELEASE_SHA256}" STREQUAL "")
                list(APPEND _kdnn_download_hash_args EXPECTED_HASH SHA256=${ANNC_KDNN_RELEASE_SHA256})
            endif()
            file(DOWNLOAD "${ANNC_KDNN_RELEASE_URL}" "${_kdnn_zip_file}"
                 SHOW_PROGRESS TIMEOUT 300 ${_kdnn_download_hash_args} STATUS _download_status)
            list(GET _download_status 0 _download_rc)
            if(NOT _download_rc EQUAL 0)
                list(GET _download_status 1 _download_err)
                message(FATAL_ERROR "Failed to download KDNN release: ${_download_err}")
            endif()
        endif()

        if(NOT EXISTS "${_kdnn_rpm_file}")
            find_program(_unzip_cmd unzip REQUIRED)
            execute_process(
                COMMAND ${_unzip_cmd} -o "${_kdnn_zip_file}" -d "${_kdnn_release_root}"
                RESULT_VARIABLE _unzip_rc)
            if(NOT _unzip_rc EQUAL 0)
                message(FATAL_ERROR "Failed to unzip KDNN release package")
            endif()
        endif()

        find_program(_rpm2cpio_cmd rpm2cpio REQUIRED)
        find_program(_cpio_cmd cpio REQUIRED)
        file(MAKE_DIRECTORY "${_kdnn_extract_root}")
        execute_process(
            COMMAND ${_rpm2cpio_cmd} "${_kdnn_rpm_file}"
            COMMAND ${_cpio_cmd} -idm
            WORKING_DIRECTORY "${_kdnn_extract_root}"
            RESULT_VARIABLE _extract_rc)
        if(NOT _extract_rc EQUAL 0)
            message(FATAL_ERROR "Failed to extract KDNN rpm package")
        endif()

        set(_kdnn_variant_dir "lib/sve/threadpool")
        if(ANNC_KDNN_LIB_VARIANT STREQUAL "sve-omp")
            set(_kdnn_variant_dir "lib/sve/omp")
        elseif(ANNC_KDNN_LIB_VARIANT STREQUAL "sve2-threadpool")
            set(_kdnn_variant_dir "lib/sve2/threadpool")
        elseif(ANNC_KDNN_LIB_VARIANT STREQUAL "sve2-omp")
            set(_kdnn_variant_dir "lib/sve2/omp")
        endif()

        set(_kdnn_lib_src
            "${_kdnn_extract_root}/usr/local/kdnn/${_kdnn_variant_dir}/libkdnn.a")
        if(NOT EXISTS "${_kdnn_lib_src}")
            message(FATAL_ERROR "KDNN library variant not found: ${_kdnn_lib_src}")
        endif()

        file(MAKE_DIRECTORY "${_kdnn_release_root}/include")
        file(MAKE_DIRECTORY "${_kdnn_release_root}/src")
        file(COPY "${_kdnn_extract_root}/usr/local/kdnn/include/"
             DESTINATION "${_kdnn_release_root}/include")
        file(COPY "${_kdnn_lib_src}" DESTINATION "${_kdnn_release_root}/src")
        file(WRITE "${_kdnn_stamp_file}" "${_kdnn_stamp_content}\n")
    endif()

    annc_import_kdnn("${_kdnn_release_root}")
    message(STATUS "Using release KDNN from ${_kdnn_release_root}")
    return()
endif()

if(ANNC_KDNN_SOURCE STREQUAL "LOCAL")
    annc_import_kdnn("${ANNC_KDNN_DIR}")
    message(STATUS "Using local KDNN from ${ANNC_KDNN_DIR}")
    return()
endif()

if(NOT ANNC_KDNN_SOURCE STREQUAL "REMOTE")
    message(FATAL_ERROR
        "Invalid ANNC_KDNN_SOURCE='${ANNC_KDNN_SOURCE}'. "
        "Expected LOCAL, REMOTE, or RELEASE.")
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
        -DENABLE_ASAN=off
        -DENABLE_GCOV=off
        -DENABLE_HBM=off
        -DKDNN_CPU_RUNTIME=OMP
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
