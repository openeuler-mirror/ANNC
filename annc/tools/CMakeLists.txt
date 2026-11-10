add_subdirectory(annc-opt)
add_subdirectory(annc-verify)
add_subdirectory(annc-asm)
add_subdirectory(annc)
add_subdirectory(annc-gemm-epilogue-kernel-compile)
add_dependencies(annc annc-gemm-epilogue-kernel-compile)
add_subdirectory(annc-fusion-metadata)
# V2 route tools: StablehloToAtirConversion is only registered under the
# same switch (annc/lib/Conversion/CMakeLists.txt).
if(ANNC_ENABLE_TF2ATIR_V2)
  add_subdirectory(annc-tf2atir-v2)
endif()
add_subdirectory(annc-partition)
