cmake_minimum_required(VERSION 3.15)
project(annc-onnx-pipeline)

# Orchestrates the ONNX -> AOT kernel pipeline for the ONNX Runtime EP's
# runtime-compile mode. It runs the sibling tools as
# subprocesses and parses the fused ATIR module in-process to emit the site
# report, so it links the ATIR dialect and the fusion metadata library -- but
# not protobuf, which only the ONNX frontend needs.
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

add_executable(annc-onnx-pipeline
    annc-onnx-pipeline.cpp
)

target_include_directories(annc-onnx-pipeline PRIVATE
    "${CMAKE_SOURCE_DIR}/annc/include"
    "${MLIR_INCLUDE_DIRS}"
    "${LLVM_INCLUDE_DIRS}"
)

# Keep in sync with the kdnn outlining pattern compiled into annc-opt: the
# asm invocation must carry --enable-kdnn in the same build.
if(ANNC_ENABLE_KDNN_ADAPTOR)
    target_compile_definitions(annc-onnx-pipeline PRIVATE ANNC_ENABLE_KDNN_ADAPTOR)
endif()

target_link_libraries(annc-onnx-pipeline PRIVATE
    AtirDialect
    ANNCFusionMetadata
    MLIRParser
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRDialect
    ${MLIR_LIBS}
    ${llvm_libs}
)

# The annc driver links the generated kernels against the archives in
# <prefix>/<libdir> (ANNC_KERNEL_LIB_PATH) and the LLVM support archives in the
# LLVM 20 lib directory. Bake both in as build-time defaults; the runtime
# ANNC_LLVM_BIN overrides the toolchain location.
target_compile_definitions(annc-onnx-pipeline PRIVATE
    ANNC_KERNEL_LIB_PATH="${CMAKE_INSTALL_FULL_LIBDIR}"
    ANNC_LLVM_BIN_DEFAULT="${ANNC_LLVM_BIN_DIR}"
    ANNC_LLVM_LIB_DIR_DEFAULT="${ANNC_LLVM_LIB_DIR}"
)
set_target_properties(annc-onnx-pipeline PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)
install(TARGETS annc-onnx-pipeline DESTINATION bin)
