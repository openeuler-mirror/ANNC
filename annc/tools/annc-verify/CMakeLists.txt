get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
)
set(LIBS
  MLIROptLib
  MLIRParser
  ${dialect_libs}
  ${conversion_libs}

  AtirDialect
  AtirBuilder
  AtirPasses
  AtirOpVerify
)
add_llvm_executable(annc-verify
  annc-verify.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(annc-verify PRIVATE ${LIBS}
  ${TENSORFLOW_LIBS}
)
llvm_update_compile_flags(annc-verify)
mlir_check_all_link_libraries(annc-verify)
install(TARGETS annc-verify DESTINATION bin)