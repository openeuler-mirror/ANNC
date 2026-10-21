# ANNC - MLIR to Executable Compiler Driver
cmake_minimum_required(VERSION 3.16)
project(annc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 
add_executable(annc annc.cpp)

target_compile_definitions(annc PRIVATE
    ANNC_KERNEL_LIB_PATH="${CMAKE_INSTALL_FULL_LIBDIR}"
    ANNC_GEMM_ASM_SOURCE_DIR="${ANNC_GEMM_ASM_SOURCE_DIR}"
    # LLVM 20 toolchain baked in at configure time; the ANNC_LLVM_BIN
    # environment variable overrides it at run time.
    ANNC_LLVM_BIN_DEFAULT="${ANNC_LLVM_BIN_DIR}"
    ANNC_LLVM_LIB_DIR_DEFAULT="${ANNC_LLVM_LIB_DIR}"
)
if(ANNC_AARCH64_GEMM_KERNELS_AVAILABLE)
    target_compile_definitions(annc PRIVATE
        ANNC_AARCH64_GEMM_KERNELS_AVAILABLE=1)
else()
    target_compile_definitions(annc PRIVATE
        ANNC_AARCH64_GEMM_KERNELS_AVAILABLE=0)
endif()

# MLAS archives for kernels built with ANNC_KERNEL_MLAS: the driver resolves
# them at link time from ANNC_EP_ORT_BUILD_TREE.
if(ANNC_KERNEL_MLAS)
    target_compile_definitions(annc PRIVATE ANNC_KERNEL_MLAS=1)
endif()

# bin
install(TARGETS annc DESTINATION bin)
