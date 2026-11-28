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
if(ANNC_ENABLE_KDNN_ADAPTOR)
  target_compile_definitions(annc-tf-pipeline PRIVATE
    ANNC_ENABLE_KDNN_ADAPTOR
    KDNN_LIB_DIR="${KDNN_LIB_DIR}"
    ANNC_KDNN_OPENMP_LINK_FLAGS="${ANNC_KDNN_OPENMP_LINK_FLAGS}")
endif()
if(ANNC_ENABLE_CONSTANT_FOLDING)
  target_compile_definitions(annc-tf-pipeline PRIVATE
    ANNC_ENABLE_CONSTANT_FOLDING)
endif()

target_link_libraries(annc-tf-pipeline PRIVATE ${LIBS})
llvm_update_compile_flags(annc-tf-pipeline)
mlir_check_all_link_libraries(annc-tf-pipeline)
install(TARGETS annc-tf-pipeline DESTINATION bin)
