# annc-converter: TensorFlow GraphDef rewrite
cmake_minimum_required(VERSION 3.16)

add_executable(annc-converter
  ANNCFusedNodeBuilder.cpp
  annc-converter.cpp
  GraphDefRewriter.cpp
)

target_link_libraries(annc-converter PRIVATE
  ANNCFusionMetadata
  ${TENSORFLOW_LIBS}
)
target_include_directories(annc-converter PRIVATE
  ${TENSORFLOW_INCLUDE_DIRS}
)

install(TARGETS annc-converter DESTINATION bin)
