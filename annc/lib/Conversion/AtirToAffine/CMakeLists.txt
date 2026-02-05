file(GLOB _srcs *.cpp)

add_mlir_library(AtirToAffineConversion
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
        MLIRAffineDialect
        MLIRArithDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        CommonConversion
)