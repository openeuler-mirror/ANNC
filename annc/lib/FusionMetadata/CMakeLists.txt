add_mlir_library(ANNCFusionMetadata
  FusionMetadata.cpp
  FusionMetadataJson.cpp

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  nlohmann_json::nlohmann_json
)
