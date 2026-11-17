cmake_minimum_required(VERSION 3.15)
project(annc_onnx_stablehlo_pipeline LANGUAGES CXX)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Reuse the ONNX library already built with the pinned ONNX-MLIR toolchain.
# Its exported targets carry matching protobuf headers and link dependencies.
find_package(Protobuf CONFIG QUIET)
find_package(absl CONFIG QUIET)
find_package(utf8_range CONFIG QUIET)
find_package(ONNX REQUIRED CONFIG
  PATHS "${ANNC_ONNX_MLIR_BUILD_DIR}/third_party/onnx" NO_DEFAULT_PATH)

set(ANNC_STABLEHLO_SOURCE_DIR
    "${ANNC_ONNX_MLIR_SOURCE_DIR}/third_party/stablehlo" CACHE PATH
    "StableHLO source paired with ONNX-MLIR")
set(ANNC_STABLEHLO_BUILD_DIR
    "${ANNC_ONNX_MLIR_BUILD_DIR}/third_party/stablehlo" CACHE PATH
    "StableHLO build tree paired with ONNX-MLIR")

add_executable(annc-onnx2stablehlo annc-onnx2stablehlo.cpp)
target_include_directories(annc-onnx2stablehlo PRIVATE ${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(_driver_llvm Support)
target_link_libraries(annc-onnx2stablehlo PRIVATE ${_driver_llvm} nlohmann_json::nlohmann_json)
target_compile_definitions(annc-onnx2stablehlo PRIVATE
  ANNC_LLVM_VERSION="${LLVM_PACKAGE_VERSION}")

# Build ATIR with this same paired MLIR, never link a system-MLIR build.
if(NOT TARGET AtirDialect)
  message(FATAL_ERROR "Configure the StableHLO addon from the ANNC root to build its ATIR dependency with the paired MLIR")
endif()

add_executable(annc-onnx-framework-opt
  annc-onnx-framework-opt.cpp
  ../passes/preprocessing/onnx_slice_preprocess.cpp
  ../passes/preprocessing/onnx_dynamic_tile_preprocess.cpp
  ../passes/preprocessing/onnx_string_to_framework_custom_call.cpp
  ../passes/postprocessing/onnx_external_to_stablehlo.cpp
  ../passes/postprocessing/framework_custom_call_to_stablehlo.cpp
  ../passes/postprocessing/framework_shape_legalize.cpp
  ../passes/postprocessing/verify_framework_stablehlo.cpp)
target_sources(annc-onnx-framework-opt PRIVATE
  ../passes/postprocessing/onnx_source_finalize.cpp)
target_include_directories(annc-onnx-framework-opt PRIVATE
  "${CMAKE_SOURCE_DIR}/annc/include"
  "${CMAKE_BINARY_DIR}/annc/include"
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
  AtirDialect nlohmann_json::nlohmann_json
  -Wl,--start-group
  MLIROptLib MLIRPass MLIRTransforms MLIRIR MLIRSupport
  MLIRFuncDialect MLIRArithDialect MLIRTensorDialect MLIRShapeDialect
  MLIRAffineUtils MLIRMathDialect MLIRSCFDialect MLIRVectorDialect
  MLIRLLVMIRTransforms MLIRQuantDialect
  StablehloOps StablehloBase StablehloTypeInference StablehloAssemblyFormat
  StablehloBroadcastUtils ChloOps VhloOps VhloTypes
  OMBuilder OMCompilerOptions OMAccelerator
  OMONNXOps OMONNXElementsAttr OMShapeHelperOpInterface
  OMShapeInferenceOpInterface OMResultTypeInferenceOpInterface
  OMHasOnnxSubgraphOpInterface OMSpecializedKernelOpInterface
  OMMlirDialects OMMlirUtilities OMSmallFPConversion OMDiagnostic
  -Wl,--end-group ONNX::onnx ${MLIR_LIBS} ${llvm_libs})
install(TARGETS annc-onnx2stablehlo annc-onnx-framework-opt DESTINATION bin)
