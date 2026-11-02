# This addon owns only the ONNX-to-StableHLO producer pipeline.
if(NOT EXISTS "${ANNC_ONNX_MLIR_SOURCE_DIR}/src/Dialect/ONNX/ONNXDialect.hpp")
  message(FATAL_ERROR
    "ONNX-MLIR source is missing. Initialize the submodule with: "
    "git submodule update --init --recursive third-party/onnx-mlir")
endif()
if(NOT EXISTS "${ANNC_ONNX_MLIR_BUILD_DIR}/Release/bin/onnx-mlir" OR
   NOT EXISTS "${ANNC_ONNX_MLIR_BUILD_DIR}/Release/bin/onnx-mlir-opt")
  message(FATAL_ERROR
    "Build the pinned ONNX-MLIR submodule first and set "
    "ANNC_ONNX_MLIR_BUILD_DIR to its build tree")
endif()

set(ANNC_STABLEHLO_SOURCE_DIR
    "${ANNC_ONNX_MLIR_SOURCE_DIR}/third_party/stablehlo" CACHE PATH
    "StableHLO submodule paired with ONNX-MLIR")
set(ANNC_STABLEHLO_BUILD_DIR
    "${ANNC_ONNX_MLIR_BUILD_DIR}/third_party/stablehlo" CACHE PATH
    "StableHLO build tree paired with ONNX-MLIR")
set(ANNC_STABLEHLO_LIB_DIR
    "${ANNC_ONNX_MLIR_BUILD_DIR}/Release/lib" CACHE PATH
    "ONNX-MLIR and StableHLO library directory")

add_subdirectory(pipeline)

add_custom_target(annc_onnx_stablehlo_addon
  DEPENDS annc-onnx-framework-opt
  COMMENT "Building the ONNX-to-StableHLO producer pipeline")
