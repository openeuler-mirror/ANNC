file(GLOB _srcs *.cpp)

set(_gemm_srcs
        Gemm/GemmPlan.cpp
)

add_mlir_library(AArch64Codegen
        ${_srcs}
        ${_gemm_srcs}

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