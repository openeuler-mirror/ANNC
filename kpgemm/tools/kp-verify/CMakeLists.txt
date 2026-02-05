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

  PimpDialect
  PimpBuilder
  PimpPasses
  PimpOpVerify
)
add_llvm_executable(kp-verify
  kp-verify.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(kp-verify PRIVATE ${LIBS}
  ${TENSORFLOW_LIBS}
)
llvm_update_compile_flags(kp-verify)
mlir_check_all_link_libraries(kp-verify)
install(TARGETS kp-verify DESTINATION bin)