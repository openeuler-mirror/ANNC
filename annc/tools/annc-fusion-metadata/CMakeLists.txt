set(LLVM_LINK_COMPONENTS
  Support
)

set(LIBS
  MLIRParser
  MLIRFuncDialect
  AtirDialect
  MLIRDialect
)

add_llvm_executable(annc-fusion-metadata
  annc-fusion-metadata.cpp
  DEPENDS
  ${LIBS}
)

target_link_libraries(annc-fusion-metadata PRIVATE
  ${LIBS}
  nlohmann_json::nlohmann_json
)

llvm_update_compile_flags(annc-fusion-metadata)
mlir_check_all_link_libraries(annc-fusion-metadata)
install(TARGETS annc-fusion-metadata DESTINATION bin)
