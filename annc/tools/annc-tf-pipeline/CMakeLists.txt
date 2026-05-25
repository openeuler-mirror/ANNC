set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(annc-tf-pipeline
  annc-tf-pipeline.cpp
)
target_link_libraries(annc-tf-pipeline PRIVATE LLVMSupport)
llvm_update_compile_flags(annc-tf-pipeline)
install(TARGETS annc-tf-pipeline DESTINATION bin)
