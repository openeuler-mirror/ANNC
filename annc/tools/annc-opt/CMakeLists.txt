get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
)
set(LIBS
  MLIROptLib
  MLIRLLVMDialect
  MLIRParser
  ${dialect_libs}
  ${conversion_libs}

  AtirDialect
  AtirPasses
  ANNCHelper
  AtirToLinalgConversion
  ContextConversion
)
add_llvm_executable(annc-opt
  annc-opt.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(annc-opt PRIVATE ${LIBS})
# V2-route parsing needs the stablehlo and tf_type compat dialects; follows
# the global ANNC_ENABLE_TF2ATIR_V2 switch (default OFF).
if(ANNC_ENABLE_TF2ATIR_V2)
  target_compile_definitions(annc-opt PRIVATE ANNC_ENABLE_TF2ATIR_V2=1)
  target_link_libraries(annc-opt PRIVATE
    StablehloRegister
    StablehloToAtirConversion)
endif()
llvm_update_compile_flags(annc-opt)
mlir_check_all_link_libraries(annc-opt)
install(TARGETS annc-opt DESTINATION bin)
