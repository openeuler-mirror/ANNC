file(GLOB _srcs *.cpp)

add_subdirectory(Interfaces)
add_subdirectory(Passes)
add_subdirectory(OpVerify)

add_mlir_library(PimpDialect
    ${_srcs}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_SOURCE_DIR}/kpgemm/include
    ${CMAKE_SOURCE_DIR}/build/kpgemm/include

    DEPENDS
    PimpTableGen
  
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