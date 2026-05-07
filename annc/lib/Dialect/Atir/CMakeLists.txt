file(GLOB _srcs *.cpp)

add_subdirectory(Interfaces)
add_subdirectory(Passes)
add_subdirectory(OpVerify)

add_mlir_library(AtirDialect
    ${_srcs}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/annc/include
    ${CMAKE_SOURCE_DIR}/build/annc/include

    DEPENDS
    AtirTableGen
  
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