set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AtirConversion)
mlir_tablegen(AtirConversionPasses.capi.h.inc -gen-pass-capi-header --prefix AtirConversion)
mlir_tablegen(AtirConversionPasses.capi.cpp.inc -gen-pass-capi-impl --prefix AtirConversion)
add_public_tablegen_target(AtirMlirConversionPassIncGen)

# Sink-layer pass defs (tblgen): opt-in under the unified V2-route switch.
if(ANNC_ENABLE_TF2ATIR_V2)
  add_subdirectory(StablehloToAtir)
endif()
