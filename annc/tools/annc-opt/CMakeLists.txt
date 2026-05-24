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
  AtirPasses
  ANNCHelper
  AtirToAffineConversion
  AtirToLinalgConversion
)
add_llvm_executable(annc-opt
  annc-opt.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(annc-opt PRIVATE ${LIBS})
llvm_update_compile_flags(annc-opt)
mlir_check_all_link_libraries(annc-opt)
install(TARGETS annc-opt DESTINATION bin)
