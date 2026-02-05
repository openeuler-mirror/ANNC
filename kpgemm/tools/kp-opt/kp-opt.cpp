#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "Builder/Builder.h"
#include "Dialect/Pimp/Passes/Passes.h"
#include "Conversion/Passes.h"
#include "Dialect/Pimp/PimpOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  pimp::registerAllPimpPasses();
  pimp::registerPimpConversionPasses();

  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, pimp::PimpDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowsUnregisteredDialects();

  std::ifstream f(argv[1]);
  nlohmann::json jgraph = nlohmann::json::parse(f);
  f.close();
  
  auto builder = std::make_shared<kpgemm::KPGEMMBuilder>(&context);
  auto root = builder->buildModule("demo", jgraph);

  // mlir::OpPrintingFlags flags;
  // flags.elideLargeElementsAttrs(0);
  // root->print(llvm::errs(), flags);

  kpgemm::outputBinary(root, "input.bin");

  argv[1] = "input.bin";

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "KPGEMM Code Generation\n", registry));
}