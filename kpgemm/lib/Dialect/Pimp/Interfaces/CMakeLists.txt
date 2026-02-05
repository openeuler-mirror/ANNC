set(LLVM_OPTIONAL_SOURCES
  ShapeInfer.cpp
  Interpret.cpp
)

add_mlir_interface_library(ShapeInfer)
add_mlir_interface_library(Interpret)