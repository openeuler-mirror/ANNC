#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "Builder/Builder.h"
#include "Dialect/Pimp/OpVerify/OpVerify.h"
#include "Dialect/Pimp/PimpOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

/*
./kp-opt input.model --passes (input.bin, output.bin)
./kp-asm output.bin --passes (lib.so)
./kp-verify output.bin --pimp-op-verify="kpGenLibPath=..."
*/

int main(int argc, char **argv) {
  pimp::registerPimpOpVerifyPass();

  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, pimp::PimpDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowsUnregisteredDialects();

  if (argc < 3) {
    std::cerr << ": ./kp-verify output.bin --pimp-op-verify=\"kpGenLibPath=...\" or --pimp-op-verify=\"llmGenLibPath=...\"" << std::endl;
    return 1;
  }

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "KP Operation Verification\n", registry));
}