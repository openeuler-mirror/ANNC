set(ANNC_AARCH64_GEMM_KERNEL_LIBRARY
    "${CMAKE_SOURCE_DIR}/third_party/annc-gemm-kernels/libannc_gemm_microkernels.a"
    CACHE FILEPATH "Path to the optional AArch64 GEMM microkernel archive")

set(ANNC_AARCH64_GEMM_KERNELS_AVAILABLE OFF)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _annc_gemm_processor)
if(_annc_gemm_processor MATCHES "^(aarch64|arm64)$" AND
   EXISTS "${ANNC_AARCH64_GEMM_KERNEL_LIBRARY}")
  add_library(ANNC::AArch64GemmMicrokernels STATIC IMPORTED GLOBAL)
  set_target_properties(ANNC::AArch64GemmMicrokernels PROPERTIES
      IMPORTED_LOCATION "${ANNC_AARCH64_GEMM_KERNEL_LIBRARY}")
  set(ANNC_AARCH64_GEMM_KERNELS_AVAILABLE ON)
  install(FILES "${ANNC_AARCH64_GEMM_KERNEL_LIBRARY}"
          DESTINATION lib
          RENAME libannc_gemm_microkernels.a)
  message(STATUS "AArch64 GEMM microkernels: ${ANNC_AARCH64_GEMM_KERNEL_LIBRARY}")
else()
  message(STATUS
      "AArch64 GEMM microkernel archive unavailable; regular ANNC targets remain enabled")
endif()
unset(_annc_gemm_processor)
