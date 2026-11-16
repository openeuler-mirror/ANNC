# The consumer must use the same MLIR and StableHLO build as the producer.
set(_stablehlo_source "${ANNC_ONNX_MLIR_SOURCE_DIR}/third_party/stablehlo")
set(_stablehlo_build "${ANNC_ONNX_MLIR_BUILD_DIR}/third_party/stablehlo")
if(NOT EXISTS "${_stablehlo_source}/stablehlo/dialect/StablehloOps.h" OR
   NOT EXISTS "${_stablehlo_build}/stablehlo/dialect/StablehloOps.h.inc")
  message(FATAL_ERROR "StableHLO frontend requires the pinned StableHLO source and build; set ANNC_ONNX_MLIR_SOURCE_DIR and ANNC_ONNX_MLIR_BUILD_DIR")
endif()
add_llvm_executable(annc-stablehlo-opt PARTIAL_SOURCES_INTENDED
  annc-stablehlo-opt.cpp Partition.cpp HloGraph.cpp
  ../onnx/converter/BindStablehlo.cpp
  Patterns/BMMAdd.cpp Patterns/LayerNorm.cpp Patterns/MaskedMul.cpp
  Patterns/Tensordot.cpp Patterns/Elementwise.cpp Patterns/ShapeProof.cpp)
target_include_directories(annc-stablehlo-opt PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${_stablehlo_source}" "${_stablehlo_build}")
target_link_directories(annc-stablehlo-opt PRIVATE
  "${ANNC_ONNX_MLIR_BUILD_DIR}/Release/lib")
target_link_libraries(annc-stablehlo-opt PRIVATE
  AtirDialect ANNCFusionMetadata nlohmann_json::nlohmann_json
  MLIROptLib MLIRFuncDialect MLIRLLVMDialect
  MLIRArithDialect MLIRTensorDialect MLIRShapeDialect MLIRQuantDialect
  StablehloOps StablehloBase StablehloTypeInference StablehloAssemblyFormat
  StablehloBroadcastUtils ChloOps VhloOps VhloTypes)
llvm_update_compile_flags(annc-stablehlo-opt)
install(TARGETS annc-stablehlo-opt DESTINATION bin)
