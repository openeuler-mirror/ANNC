#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "Builder/Builder.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Conversion/Passes.h"
#include "Target/Passes.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"


using namespace mlir;

int main(int argc, char **argv) {
  atir::registerAllAtirPasses();
  atir::registerAtirConversionPasses();
  annc::registerAllTargetPasses();

  DialectRegistry registry;
  registry.insert<
          mlir::func::FuncDialect,
          mlir::arith::ArithDialect,
          mlir::linalg::LinalgDialect,
          mlir::memref::MemRefDialect,
          mlir::affine::AffineDialect,
          mlir::bufferization::BufferizationDialect,
          mlir::vector::VectorDialect,
          mlir::scf::SCFDialect,
          mlir::ub::UBDialect,
          atir::AtirDialect
          >();
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowsUnregisteredDialects();

  // mlir::OpPrintingFlags flags;
  // flags.elideLargeElementsAttrs(0);
  // root->print(llvm::errs(), flags);
  // annc-asm input.bin --xxx_pass
  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "ANNC Code Generation\n", registry));
}
