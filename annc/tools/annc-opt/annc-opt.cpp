#include "Conversion/Passes.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#ifdef ANNC_ENABLE_TF2ATIR_V2
#include "Conversion/StablehloToAtir/StablehloSink.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"
#endif

using namespace mlir;

int main(int argc, char **argv) {
  atir::registerAllAtirPasses();
  atir::registerAtirConversionPasses();

  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
#ifdef ANNC_ENABLE_TF2ATIR_V2
                  // V2 artifacts carry stablehlo ops, shape scaffolding and
                  // tf_type compat types; all three must parse here.
                  mlir::arith::ArithDialect,
                  mlir::shape::ShapeDialect,
                  mlir::tensor::TensorDialect,
#endif
                  atir::AtirDialect
#ifdef ANNC_ENABLE_TF2ATIR_V2
                  ,
                  mlir::stablehlo::StablehloDialect
#endif
                  >();
#ifdef ANNC_ENABLE_TF2ATIR_V2
  annc::tfcompat::registerTfTypeCompatDialect(registry);
#endif

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowsUnregisteredDialects();

  // std::ifstream f(argv[1]);
  // nlohmann::json jgraph = nlohmann::json::parse(f);
  // f.close();

  // mlir::OpPrintingFlags flags;
  // flags.elideLargeElementsAttrs(0);
  // root->print(llvm::errs(), flags);

  // annc::outputBinary(root, "input.bin");

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "ANNC Code Generation\n", registry));
}
