# Framework frontend registry. Each frontend lives in its own subdirectory and
# is registered only when its framework switch is enabled; the switches are
# defined in the root CMakeLists.txt. Future frontends are appended here rather
# than to annc/tools/.
if(ANNC_ENABLE_ONNX)
  add_subdirectory(onnx/onnx2atir)
  # Pipeline orchestrator for the ONNX Runtime EP's runtime-compile mode.
  add_subdirectory(onnx/onnx-pipeline)
endif()
