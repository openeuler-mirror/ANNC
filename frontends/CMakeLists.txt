# Framework frontend registry. Each frontend lives in its own subdirectory and
# is registered only when its framework switch is enabled; the switches are
# defined in the root CMakeLists.txt. Future frontends are appended here rather
# than to annc/tools/.
if(ANNC_ENABLE_TENSORFLOW)
  add_subdirectory(tensorflow)
endif()

if(ANNC_ENABLE_ONNX)
  add_subdirectory(onnx/onnx2atir)
  # Pipeline orchestrator for the ONNX Runtime EP's runtime-compile mode.
  add_subdirectory(onnx/onnx-pipeline)
endif()

# TF stablehlo V2 route (opt-in, ANNC_ENABLE_TF2ATIR_V2): annc-converter-v2
# is the pip-TF-wheel based GraphDef rewrite; it is process-isolated from
# ANNC's LLVM/MLIR stack (no annc target is linked).
if(ANNC_ENABLE_TF2ATIR_V2)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../addons/tensorflow/annc-converter-v2
    annc-converter-v2)
endif()
