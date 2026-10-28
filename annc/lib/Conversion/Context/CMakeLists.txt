add_mlir_library(ContextConversion
  LowerContextKernelEntries.cpp

  DEPENDS
  AtirMlirConversionPassIncGen

  LINK_LIBS PUBLIC
  CommonConversion
)
