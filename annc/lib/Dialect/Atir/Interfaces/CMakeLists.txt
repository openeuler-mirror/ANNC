set(LLVM_OPTIONAL_SOURCES
  ShapeInfer.cpp
  Interpret.cpp
  Interpret/*.cpp
)

add_mlir_interface_library(ShapeInfer)

# Collect all .cpp files in Interpret/ directory
file(GLOB INTERPRET_SOURCES Interpret/*.cpp)

# Create Interpret library with all split implementation files
# These are compiled as separate translation units (not included)
add_mlir_library(AtirInterpretIntfc
  Interpret.cpp
  ${INTERPRET_SOURCES}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRInterpretIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  ANNCHelper
)