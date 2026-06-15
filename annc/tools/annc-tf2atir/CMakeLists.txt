cmake_minimum_required(VERSION 3.15)
project(annc-tf2atir)

find_package(Protobuf REQUIRED)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# 自动收集所有生成的 .pb.cc 文件
file(GLOB_RECURSE PROTO_SRCS "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code/*.pb.cc")

add_executable(annc-tf2atir
    annc-tf2atir.cpp
    standalone_pb_parser.cpp
    absl_log_shim.cpp
    ${PROTO_SRCS}
)

target_include_directories(annc-tf2atir BEFORE PRIVATE
    "${CMAKE_SOURCE_DIR}/annc/include"
    "${CMAKE_SOURCE_DIR}/tf_protos_minimal/gen_code"
    "${TENSORFLOW_INCLUDE_DIRS}"
    "${Protobuf_INCLUDE_DIRS}"
)
target_include_directories(annc-tf2atir PRIVATE
    "${MLIR_INCLUDE_DIRS}"
    "${LLVM_INCLUDE_DIRS}"
)

# 自动检测是否需要链接系统 abseil 库
# openEuler 24.03 的 protobuf 依赖 abseil，而 22.03 不需要
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ABSL_LOG QUIET absl_log)
    pkg_check_modules(ABSL_CHECK_OP QUIET absl_log_internal_check_op)
endif()

add_definitions(-DABSL_MIN_LOG_LEVEL=4)

target_link_libraries(annc-tf2atir PRIVATE
    ${TENSORFLOW_FRAMEWORK_LIB}
)

# 条件性链接 abseil 库（如果检测到）
if(ABSL_LOG_FOUND AND ABSL_CHECK_OP_FOUND)
    message(STATUS "Linking abseil libraries for protobuf dependency")
    target_link_libraries(annc-tf2atir PRIVATE
        ${ABSL_LOG_LIBRARIES}
        ${ABSL_CHECK_OP_LIBRARIES}
    )
else()
    message(STATUS "Abseil libraries not found, skipping (may not be required on this system)")
endif()

# 链接 ANNC 和 MLIR/LLVM 库
target_link_libraries(annc-tf2atir PRIVATE
    AtirBuilder AtirDialect AtirPasses
    AtirToAffineConversion
    AtirToLinalgConversion
    MLIRArithDialect
    MLIROptLib
    ${MLIR_LIBS}
    ${llvm_libs}
)
install(TARGETS annc-tf2atir DESTINATION bin)