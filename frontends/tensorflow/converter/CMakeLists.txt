# annc-converter: TensorFlow GraphDef rewrite
cmake_minimum_required(VERSION 3.16)

find_package(Protobuf REQUIRED)

file(GLOB PROTO_SRCS
  "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code/tensorflow/core/framework/*.pb.cc"
)

add_executable(annc-converter
  ANNCFusedNodeBuilder.cpp
  annc-converter.cpp
  GraphDefRewriter.cpp
  ${PROTO_SRCS}
)

target_include_directories(annc-converter BEFORE PRIVATE
  "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code"
)
target_include_directories(annc-converter SYSTEM PRIVATE
  "${Protobuf_INCLUDE_DIRS}"
)

target_link_libraries(annc-converter PRIVATE
  ANNCFusionMetadata
  ${Protobuf_LIBRARIES}
)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(ANNC_CONVERTER_ABSL_LOG QUIET absl_log)
  pkg_check_modules(ANNC_CONVERTER_ABSL_CHECK_OP QUIET
    absl_log_internal_check_op)
endif()

if(ANNC_CONVERTER_ABSL_LOG_FOUND AND ANNC_CONVERTER_ABSL_CHECK_OP_FOUND)
  target_link_libraries(annc-converter PRIVATE
    ${ANNC_CONVERTER_ABSL_LOG_LIBRARIES}
    ${ANNC_CONVERTER_ABSL_CHECK_OP_LIBRARIES}
  )
endif()

set_target_properties(annc-converter PROPERTIES
  BUILD_RPATH "${Protobuf_LIB_DIRS}"
)

install(TARGETS annc-converter DESTINATION bin)
