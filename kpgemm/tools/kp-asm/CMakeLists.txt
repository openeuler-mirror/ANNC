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
  PimpToAffineConversion
  PimpToLinalgConversion
)
add_llvm_executable(kp-asm
  kp-asm.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(kp-asm PRIVATE ${LIBS})
llvm_update_compile_flags(kp-asm)
mlir_check_all_link_libraries(kp-asm)
install(TARGETS kp-asm DESTINATION bin)