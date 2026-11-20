# annc-stablehlo-sink: the V2-route entry tool (consumes the frontend stablehlo
# artifact and runs the residue sink). Registered only under
# ANNC_ENABLE_TF2ATIR_V2=ON, together with StablehloToAtirConversion.
set(LLVM_LINK_COMPONENTS
  Core
  Support
)
set(LIBS
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRShapeDialect
  MLIRTensorDialect

  AtirDialect
  AtirBuilder
  StablehloRegister
  StablehloToAtirConversion
  ANNCHelper
)
add_llvm_executable(annc-stablehlo-sink
  annc-stablehlo-sink.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(annc-stablehlo-sink PRIVATE ${LIBS})
llvm_update_compile_flags(annc-stablehlo-sink)
mlir_check_all_link_libraries(annc-stablehlo-sink)
install(TARGETS annc-stablehlo-sink DESTINATION bin)
