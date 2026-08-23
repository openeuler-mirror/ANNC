set(LLVM_LINK_COMPONENTS
  Support
)

set(LIBS
  MLIRParser
  MLIRFuncDialect
  MLIRLLVMDialect
  AtirDialect
  MLIRDialect
  ANNCFusionMetadata
)

add_llvm_executable(annc-fusion-metadata
  annc-fusion-metadata.cpp
  DEPENDS
  ${LIBS}
)

target_link_libraries(annc-fusion-metadata PRIVATE
  ${LIBS}
)

llvm_update_compile_flags(annc-fusion-metadata)
mlir_check_all_link_libraries(annc-fusion-metadata)
install(TARGETS annc-fusion-metadata DESTINATION bin)
