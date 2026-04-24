file(GLOB _srcs *.cpp)

add_mlir_library(PimpToAffineConversion
        ${_srcs}

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/kpgemm/include
        ${CMAKE_SOURCE_DIR}/build/kpgemm/include

        DEPENDS
        PimpMlirConversionPassIncGen

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