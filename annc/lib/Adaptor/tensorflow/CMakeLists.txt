file(GLOB _srcs *.cpp)

add_mlir_library(TFAdaptor
   TFSavedModelParser.cpp
   ${_srcs}
   LINK_LIBS PUBLIC
        ANNCHelper
        MLIRIR
        MLIRSupport
        MLIRPass
        nlohmann_json::nlohmann_json
)

target_link_libraries(TFAdaptor PUBLIC
    ${TENSORFLOW_FRAMEWORK_LIB}
    ${TENSORFLOW_CC_LIB}
)

target_include_directories(TFAdaptor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)