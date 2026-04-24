file(GLOB _srcs *.cpp)

add_mlir_library(PimpPasses
    ${_srcs}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/kpgemm/include
    ${CMAKE_SOURCE_DIR}/build/kpgemm/include

    DEPENDS
    PimpTableGen
    MLIRPimpPassIncGen
  
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    
    PimpShapeInferIntfc
    PimpInterpretIntfc
)