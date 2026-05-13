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

target_include_directories(AtirOpVerify PUBLIC ${TENSORFLOW_INCLUDE_DIRS})
target_link_libraries(AtirOpVerify PUBLIC ${TENSORFLOW_LIBRARY_DIR}/libtensorflow_framework.so.2)