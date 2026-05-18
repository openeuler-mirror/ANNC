set(LLVM_OPTIONAL_SOURCES
  Dialects.cpp
  Passes.cpp
)

add_mlir_public_c_api_library(AtirCAPIDialect
  Dialects.cpp

  LINK_LIBS PUBLIC
  AtirBuilder
  AtirDialect
)

add_mlir_public_c_api_library(AtirCAPIPasses
  Passes.cpp

  LINK_LIBS PUBLIC
  AtirDialect
  AtirBuilder
  AtirToAffineConversion
  AtirToLinalgConversion
)