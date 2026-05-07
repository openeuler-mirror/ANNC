get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(tf_json_parser_LIBS
  MLIROptLib
  MLIRParser
  TFAdaptor
)

add_llvm_executable(tf-adaptor
    tf-adaptor.cpp
    DEPENDS
    ${tf_json_parser_LIBS}
)

target_link_libraries(tf-adaptor PRIVATE ${TENSORFLOW_LIBS}
    ${tf_json_parser_LIBS}

    nlohmann_json::nlohmann_json
)

target_include_directories(tf-adaptor PRIVATE
  ${TENSORFLOW_INCLUDE_DIRS}
)

llvm_update_compile_flags(tf-adaptor)
mlir_check_all_link_libraries(tf-adaptor)
install(TARGETS tf-adaptor DESTINATION bin)


