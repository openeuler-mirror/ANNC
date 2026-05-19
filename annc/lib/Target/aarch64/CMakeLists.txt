file(GLOB _srcs *.cpp)

add_mlir_library(AArch64Codegen
        ${_srcs}

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/annc/include
        ${CMAKE_SOURCE_DIR}/build/annc/include

        DEPENDS
        AArch64CodeGenPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRParser
        MLIRRewrite
        MLIRTransforms
        MLIRAffineDialect
        MLIRArithDialect
        MLIRMemRefDialect
        MLIRFuncDialect
        MLIRBufferizationTransforms
        MLIRLinalgDialect
        MLIRLinalgTransforms
)