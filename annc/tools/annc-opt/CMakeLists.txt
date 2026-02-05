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
)
add_llvm_executable(kp-opt
  kp-opt.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(kp-opt PRIVATE ${LIBS})
llvm_update_compile_flags(kp-opt)
mlir_check_all_link_libraries(kp-opt)
install(TARGETS kp-opt DESTINATION bin)