set(LLVM_OPTIONAL_SOURCES
  Helper.cpp
  Tensor.cpp
)

set(helper_
  Helper.cpp)

add_mlir_library(ANNCHelper
  ${helper_}

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRBytecodeWriter
  MLIRBytecodeReader
)

add_subdirectory(Dialect)
add_subdirectory(Builder)
add_subdirectory(CAPI)
add_subdirectory(Adaptor)
add_subdirectory(Conversion)
