add_subdirectory(annc-opt)
add_subdirectory(annc-verify)
add_subdirectory(annc-asm)
add_subdirectory(annc)
add_subdirectory(annc-model-kernel-export)
add_dependencies(annc annc-model-kernel-export)
add_subdirectory(annc-fusion-metadata)
# TensorFlow frontend tools, gated by ANNC_ENABLE_TENSORFLOW (see the root
# CMakeLists.txt option block). With the switch off their TensorFlow protobuf
# sources are neither generated nor required.
if(ANNC_ENABLE_TENSORFLOW)
  add_subdirectory(annc-converter)
  add_subdirectory(annc-tf2atir)
  add_subdirectory(annc-tf-pipeline)
endif()
