file(GLOB _srcs *.cpp Patterns/*.cpp Patterns/CustomPatterns/*.cpp)

add_mlir_library(AtirPasses
    ${_srcs}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/annc/include
    ${CMAKE_SOURCE_DIR}/build/annc/include

    DEPENDS
    AtirTableGen
    MLIRAtirPassIncGen
  
    LINK_LIBS PUBLIC
    AtirDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    MLIRDialect
    
    AtirShapeInferIntfc
    AtirInterpretIntfc
)
