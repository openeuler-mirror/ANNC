# This addon owns only the ONNX-to-StableHLO producer pipeline.
if(NOT EXISTS "${ANNC_ONNX_MLIR_SOURCE_DIR}/src/Dialect/ONNX/ONNXDialect.hpp")
  message(FATAL_ERROR
    "ONNX-MLIR source is missing. Initialize the submodule with: "
    "git submodule update --init --recursive third-party/onnx-mlir")
endif()
get_filename_component(_onnx_mlir_config_dir "${ANNC_ONNX_MLIR_LIBRARY_DIR}" DIRECTORY)
if(NOT EXISTS "${_onnx_mlir_config_dir}/bin/onnx-mlir" OR
   NOT EXISTS "${_onnx_mlir_config_dir}/bin/onnx-mlir-opt")
  message(FATAL_ERROR
    "ONNX-MLIR tools not found in ${_onnx_mlir_config_dir}/bin. "
    "Build the pinned ONNX-MLIR submodule first or set ANNC_ONNX_MLIR_BUILD_DIR.")
endif()

set(ANNC_STABLEHLO_SOURCE_DIR
    "${ANNC_ONNX_MLIR_SOURCE_DIR}/third_party/stablehlo" CACHE PATH
    "StableHLO submodule paired with ONNX-MLIR")
set(ANNC_STABLEHLO_BUILD_DIR
    "${ANNC_ONNX_MLIR_BUILD_DIR}/third_party/stablehlo" CACHE PATH
    "StableHLO build tree paired with ONNX-MLIR")

add_subdirectory(pipeline)

add_custom_target(annc_onnx_stablehlo_addon
  DEPENDS annc-onnx2stablehlo annc-onnx-framework-opt
  COMMENT "Building the ONNX-to-StableHLO producer pipeline")
