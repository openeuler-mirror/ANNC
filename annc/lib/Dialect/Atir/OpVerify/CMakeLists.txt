file(GLOB _srcs *.cpp)

add_mlir_library(AtirOpVerify
    ${_srcs}
    
    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/annc/include
    ${CMAKE_SOURCE_DIR}/build/annc/include
  
    DEPENDS
    AtirTableGen
    MLIRAtirOpVerifyIncGen
  
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms

    AtirShapeInferIntfc
    AtirInterpretIntfc
)