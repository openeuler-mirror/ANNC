set(LLVM_LINK_COMPONENTS
  Support
)

set(LIBS
  MLIRParser
  MLIRFuncDialect
  MLIRLLVMDialect
  AtirDialect
  MLIRDialect
)

add_llvm_executable(annc-tf-pipeline
  annc-tf-pipeline.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(annc-tf-pipeline PRIVATE ${LIBS})
llvm_update_compile_flags(annc-tf-pipeline)
mlir_check_all_link_libraries(annc-tf-pipeline)
install(TARGETS annc-tf-pipeline DESTINATION bin)
