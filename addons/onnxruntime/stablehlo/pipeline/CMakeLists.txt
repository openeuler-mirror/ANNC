cmake_minimum_required(VERSION 3.15)
project(annc_onnx_stablehlo_pipeline LANGUAGES CXX)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

set(ANNC_STABLEHLO_SOURCE_DIR
    "${ANNC_ONNX_MLIR_SOURCE_DIR}/third_party/stablehlo" CACHE PATH
    "StableHLO source paired with ONNX-MLIR")
set(ANNC_STABLEHLO_BUILD_DIR
    "${ANNC_ONNX_MLIR_BUILD_DIR}/third_party/stablehlo" CACHE PATH
    "StableHLO build tree paired with ONNX-MLIR")

add_executable(annc-onnx-framework-opt
  annc-onnx-framework-opt.cpp
  ../passes/preprocessing/onnx_slice_preprocess.cpp
  ../passes/preprocessing/onnx_dynamic_tile_preprocess.cpp
  ../passes/preprocessing/onnx_string_to_framework_custom_call.cpp
  ../passes/postprocessing/framework_custom_call_to_stablehlo.cpp
  ../passes/postprocessing/framework_shape_legalize.cpp
  )
target_include_directories(annc-onnx-framework-opt PRIVATE
  "${ANNC_ONNX_MLIR_SOURCE_DIR}"
  "${ANNC_ONNX_MLIR_SOURCE_DIR}/include"
  "${ANNC_ONNX_MLIR_BUILD_DIR}"
  "${ANNC_STABLEHLO_SOURCE_DIR}"
  "${ANNC_STABLEHLO_BUILD_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../passes/preprocessing"
  ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_directories(annc-onnx-framework-opt PRIVATE
  "${ANNC_ONNX_MLIR_BUILD_DIR}/Release/lib")
target_link_libraries(annc-onnx-framework-opt PRIVATE
  -Wl,--start-group
  MLIROptLib MLIRPass MLIRTransforms MLIRIR MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRTensorDialect MLIRShapeDialect
  MLIRAffineUtils MLIRMathDialect MLIRSCFDialect MLIRVectorDialect
  MLIRLLVMIRTransforms MLIRQuantDialect
  StablehloOps StablehloBase StablehloTypeInference StablehloAssemblyFormat
  StablehloBroadcastUtils ChloOps VhloOps VhloTypes
  OMCompilerOptions OMAccelerator
  OMONNXOps OMONNXElementsAttr OMShapeHelperOpInterface
  OMShapeInferenceOpInterface OMResultTypeInferenceOpInterface
  OMHasOnnxSubgraphOpInterface OMSpecializedKernelOpInterface
  OMMlirDialects OMMlirUtilities OMSmallFPConversion OMDiagnostic
  -Wl,--end-group ${MLIR_LIBS} ${llvm_libs})
install(TARGETS annc-onnx-framework-opt DESTINATION bin)
