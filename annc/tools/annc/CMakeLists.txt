# ANNC - MLIR to Executable Compiler Driver
cmake_minimum_required(VERSION 3.16)
project(annc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 
add_executable(annc annc.cpp)

target_compile_definitions(annc PRIVATE
    ANNC_KERNEL_LIB_PATH="${CMAKE_INSTALL_PREFIX}/lib"
)

# bin
install(TARGETS annc DESTINATION bin)