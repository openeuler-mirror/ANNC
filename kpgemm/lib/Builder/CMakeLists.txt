file(GLOB _srcs *.cpp)

add_mlir_library(PimpBuilder
    ${_srcs}
  
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    KPGEMMHelper
    PimpDialect
)