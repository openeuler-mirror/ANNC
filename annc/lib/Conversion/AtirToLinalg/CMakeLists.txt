file(GLOB _srcs *.cpp)

add_mlir_library(AtirToLinalgConversion
        ${_srcs}

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/annc/include
        ${CMAKE_SOURCE_DIR}/build/annc/include

        DEPENDS
        AtirMlirConversionPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRParser
        MLIRRewrite
        MLIRTransforms
        MLIRLinalgDialect
        MLIRArithDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        CommonConversion
        ANNCKernel
)
