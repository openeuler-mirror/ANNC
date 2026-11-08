# Root files are framework-agnostic; tf/ is the TF adapter (mapping table +
# wheel type-compat). New frameworks add a subdirectory.
file(GLOB _srcs CONFIGURE_DEPENDS *.cpp tf/*.cpp)
add_mlir_library(StablehloToAtirConversion ${_srcs}
  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/annc/include
  LINK_LIBS PUBLIC
  MLIRIR MLIRSupport MLIRTransforms MLIRArithDialect MLIRFuncDialect
  StablehloRegister AtirDialect ANNCHelper nlohmann_json::nlohmann_json
)
