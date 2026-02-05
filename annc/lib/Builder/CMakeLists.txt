file(GLOB _srcs *.cpp)

add_mlir_library(AtirBuilder
    ${_srcs}
  
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    ANNCHelper
    AtirDialect
)