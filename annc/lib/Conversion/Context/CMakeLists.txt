add_mlir_library(ContextConversion
  LowerContextKernelEntries.cpp
  ConvertContextKernelABI.cpp

  DEPENDS
  AtirMlirConversionPassIncGen

  LINK_LIBS PUBLIC
  CommonConversion
  MLIRLLVMCommonConversion
)
