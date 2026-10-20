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
# <prefix>/lib (ANNC_KERNEL_LIB_PATH); keep the tool relocatable with the
# same relative layout the other tools use.
set_target_properties(annc-onnx-pipeline PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
)
install(TARGETS annc-onnx-pipeline DESTINATION bin)
