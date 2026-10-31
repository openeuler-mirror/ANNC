# Remove cache entries from the pre-atomic-package interface so an existing
# build directory cannot retain an independently selectable archive/codegen
# configuration after reconfiguration.
unset(ANNC_AARCH64_GEMM_KERNEL_LIBRARY CACHE)
unset(ANNC_GEMM_KERNELS_PACKAGE_DIR CACHE)
unset(ANNC_REQUIRE_GEMM_CODEGEN CACHE)

set(ANNC_GEMM_PACKAGE_DIR
    "${CMAKE_SOURCE_DIR}/third_party/annc-gemm-kernels"
    CACHE PATH "Directory containing the complete GEMM-ASM binary package")
option(ANNC_REQUIRE_GEMM_PACKAGE
    "Fail configuration if the complete GEMM-ASM package is unavailable" OFF)

# GEMM-ASM is delivered as one atomic package. The archive and codegen
# targets are separate consumers of that package, but they must never be
# selected from independently versioned locations.
set(ANNC_GEMM_PACKAGE_AVAILABLE OFF)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _annc_gemm_processor)
set(_annc_gemm_package_config
    "${ANNC_GEMM_PACKAGE_DIR}/cmake/AnncGemmKernelsConfig.cmake")
if(_annc_gemm_processor MATCHES "^(aarch64|arm64)$" AND
   EXISTS "${_annc_gemm_package_config}")
  set(AnncGemmKernels_DIR "${ANNC_GEMM_PACKAGE_DIR}/cmake")
  find_package(AnncGemmKernels CONFIG REQUIRED
    PATHS "${ANNC_GEMM_PACKAGE_DIR}/cmake" NO_DEFAULT_PATH)
  if(NOT AnncGemmKernels_API_VERSION EQUAL 1 OR
     NOT AnncGemmKernels_KERNEL_ABI_VERSION EQUAL 1 OR
     NOT TARGET ANNC::GemmCodegen OR
     NOT TARGET ANNC::AArch64GemmMicrokernels)
    message(FATAL_ERROR
      "Incompatible GEMM-ASM package: API 1 and kernel ABI 1 are required")
  endif()
  set(ANNC_GEMM_PACKAGE_AVAILABLE ON)
  install(DIRECTORY "${ANNC_GEMM_PACKAGE_DIR}/"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/annc-gemm-kernels")
  list(APPEND CMAKE_INSTALL_RPATH
    "$ORIGIN/../lib"
    "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
    "$ORIGIN/annc-gemm-kernels"
    "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/annc-gemm-kernels")
  list(REMOVE_DUPLICATES CMAKE_INSTALL_RPATH)
  message(STATUS "GEMM-ASM package: ${AnncGemmKernels_VERSION} (${AnncGemmKernels_BUILD_ID})")
else()
  message(STATUS
    "Complete AArch64 GEMM-ASM package unavailable; regular ANNC targets remain enabled")
endif()

if(NOT ANNC_GEMM_PACKAGE_AVAILABLE)
  if(ANNC_REQUIRE_GEMM_PACKAGE)
    message(FATAL_ERROR
      "Complete GEMM-ASM package required at ANNC_GEMM_PACKAGE_DIR")
  endif()
  message(STATUS
    "GEMM-ASM package unavailable; GEMM microkernel and epilogue requests will report an error")
endif()

unset(_annc_gemm_processor)
unset(_annc_gemm_package_config)
