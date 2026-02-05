set(LLVM_OPTIONAL_SOURCES
  Dialects.cpp
  Passes.cpp
)

add_mlir_public_c_api_library(PimpCAPIDialect
  Dialects.cpp

  LINK_LIBS PUBLIC
  PimpBuilder
  PimpDialect
)

add_mlir_public_c_api_library(PimpCAPIPasses
  Passes.cpp

  LINK_LIBS PUBLIC
  PimpDialect
  PimpPasses
  PimpBuilder
  PimpToAffineConversion
  PimpToLinalgConversion
)