# annc-tf2atir-v2: the V2-route entry tool (consumes the frontend stablehlo
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
add_llvm_executable(annc-tf2atir-v2
  annc-tf2atir-v2.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(annc-tf2atir-v2 PRIVATE ${LIBS})
llvm_update_compile_flags(annc-tf2atir-v2)
mlir_check_all_link_libraries(annc-tf2atir-v2)
install(TARGETS annc-tf2atir-v2 DESTINATION bin)
