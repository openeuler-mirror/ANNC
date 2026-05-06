# ── KDNN integration options ──────────────────────────────────────────
# These are project-level options because both the Kernel layer (builtin
# kernel registration) and the Atir Pattern layer (#ifdef guards) depend
# on them.  Placing them here ensures they are resolved before any
# sub-directory is entered.

set(KDNN_DIR "" CACHE PATH
    "Path to KDNN installation root. Must contain include/kdnn.hpp, include/kdnn_config.h and lib/libkdnn.so or lib/libkdnn.a")

if(KDNN_DIR)
    set(KDNN_INCLUDE_DIR "${KDNN_DIR}/include" CACHE PATH "KDNN include directory" FORCE)
    set(KDNN_LIB_DIR "${KDNN_DIR}/lib" CACHE PATH "KDNN library directory" FORCE)
    # Prefer shared library; fall back to static
    if(EXISTS "${KDNN_DIR}/lib/libkdnn.so")
        set(KDNN_LIBRARY "${KDNN_DIR}/lib/libkdnn.so" CACHE FILEPATH "KDNN library" FORCE)
    elseif(EXISTS "${KDNN_DIR}/lib/libkdnn.a")
        set(KDNN_LIBRARY "${KDNN_DIR}/lib/libkdnn.a" CACHE FILEPATH "KDNN library" FORCE)
    else()
        set(KDNN_LIBRARY "" CACHE FILEPATH "KDNN library" FORCE)
    endif()
endif()

set(_annc_enable_kdnn_default OFF)
if(KDNN_DIR AND EXISTS "${KDNN_INCLUDE_DIR}/kdnn.hpp"
   AND EXISTS "${KDNN_INCLUDE_DIR}/kdnn_config.h"
   AND EXISTS "${KDNN_LIBRARY}")
    set(_annc_enable_kdnn_default ON)
endif()
option(ANNC_ENABLE_KDNN_ADAPTOR "Enable builtin KDNN adaptor kernels" ${_annc_enable_kdnn_default})

if(ANNC_ENABLE_KDNN_ADAPTOR)
    if(NOT KDNN_DIR)
        message(FATAL_ERROR "ANNC_ENABLE_KDNN_ADAPTOR=ON but KDNN_DIR is not set")
    endif()
    if(NOT EXISTS "${KDNN_INCLUDE_DIR}/kdnn.hpp")
        message(FATAL_ERROR "ANNC_ENABLE_KDNN_ADAPTOR=ON but kdnn.hpp was not found under ${KDNN_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${KDNN_INCLUDE_DIR}/kdnn_config.h")
        message(FATAL_ERROR "ANNC_ENABLE_KDNN_ADAPTOR=ON but kdnn_config.h was not found under ${KDNN_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${KDNN_LIBRARY}")
        message(FATAL_ERROR "ANNC_ENABLE_KDNN_ADAPTOR=ON but KDNN library was not found at ${KDNN_LIB_DIR}")
    endif()
    add_definitions(-DANNC_ENABLE_KDNN_ADAPTOR)
    add_definitions(-DKDNN_LIB_DIR="${KDNN_LIB_DIR}")
endif()

# ── Sub-directories ──────────────────────────────────────────────────
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
