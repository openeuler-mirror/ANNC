# annc-converter: Atir MLIR → TF GraphDef
cmake_minimum_required(VERSION 3.16)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  MLIROptLib
  MLIRParser
  ${dialect_libs}
  ${conversion_libs}

  AtirDialect
  AtirBuilder
  TFAdaptor
  ANNCHelper
)

add_llvm_executable(annc-converter
  annc-converter.cpp
  DEPENDS
  ${LIBS}
)

target_link_libraries(annc-converter PRIVATE
  ${LIBS}
  ${TENSORFLOW_LIBS}
  nlohmann_json::nlohmann_json
)
target_link_libraries(annc-converter PUBLIC ${TENSORFLOW_LIBRARY_DIR}/libtensorflow_framework.so.2)

target_include_directories(annc-converter PRIVATE
  ${TENSORFLOW_INCLUDE_DIRS}
)

llvm_update_compile_flags(annc-converter)
mlir_check_all_link_libraries(annc-converter)
install(TARGETS annc-converter DESTINATION bin)
