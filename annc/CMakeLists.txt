if(ANNC_ENABLE_KDNN_ADAPTOR)
    # LOCAL mode expects a pre-built KDNN tree, so verify it immediately.
    # REMOTE mode builds KDNN as an ExternalProject during the build phase;
    # the generated headers and library do not exist at configure time.
    if(ANNC_KDNN_SOURCE STREQUAL "LOCAL")
        if(NOT EXISTS "${KDNN_INCLUDE_DIR}/kdnn.hpp")
            message(FATAL_ERROR "ANNC_ENABLE_KDNN_ADAPTOR=ON and ANNC_KDNN_SOURCE=LOCAL but kdnn.hpp was not found under ${KDNN_INCLUDE_DIR}")
        endif()
        if(NOT EXISTS "${KDNN_INCLUDE_DIR}/kdnn_config.h")
            message(FATAL_ERROR "ANNC_ENABLE_KDNN_ADAPTOR=ON and ANNC_KDNN_SOURCE=LOCAL but kdnn_config.h was not found under ${KDNN_INCLUDE_DIR}")
        endif()
        if(NOT EXISTS "${KDNN_LIBRARY}")
            message(FATAL_ERROR "ANNC_ENABLE_KDNN_ADAPTOR=ON and ANNC_KDNN_SOURCE=LOCAL but KDNN library was not found at ${KDNN_LIB_DIR}")
        endif()
    endif()
    add_definitions(-DANNC_ENABLE_KDNN_ADAPTOR)
    add_definitions(-DKDNN_LIB_DIR="${KDNN_LIB_DIR}")
    add_definitions(-DANNC_KDNN_OPENMP_LINK_FLAGS="${ANNC_KDNN_OPENMP_LINK_FLAGS}")
endif()

if(ANNC_ENABLE_CONSTANT_FOLDING)
    add_definitions(-DANNC_ENABLE_CONSTANT_FOLDING)
endif()

# ── Sub-directories ──────────────────────────────────────────────────
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
