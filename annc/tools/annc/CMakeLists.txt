# ANNC - MLIR to Executable Compiler Driver
cmake_minimum_required(VERSION 3.16)
project(annc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 
add_executable(annc annc.cpp)

target_compile_definitions(annc PRIVATE
    ANNC_KERNEL_LIB_PATH="${CMAKE_INSTALL_PREFIX}/lib"
    ANNC_GEMM_ASM_SOURCE_DIR="${ANNC_GEMM_ASM_SOURCE_DIR}"
)
if(ANNC_AARCH64_GEMM_KERNELS_AVAILABLE)
    target_compile_definitions(annc PRIVATE
        ANNC_AARCH64_GEMM_KERNELS_AVAILABLE=1)
else()
    target_compile_definitions(annc PRIVATE
        ANNC_AARCH64_GEMM_KERNELS_AVAILABLE=0)
endif()

# MLAS archives for kernels built with ANNC_KERNEL_MLAS: baked as absolute
# paths into the link commands for kernel shared libraries (and test
# executables), so the single -L directory the driver gets for the ANNC and
# LLVM archives is untouched.
if(ANNC_KERNEL_MLAS)
    target_compile_definitions(annc PRIVATE
        ANNC_MLAS_ARCHIVE_ARGS="${ANNC_MLAS_ARCHIVE_ARGS}")
endif()

# bin
install(TARGETS annc DESTINATION bin)
