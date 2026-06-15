#include "standalone_pb_parser.h"
#include "Builder/Builder.h"
#include "Helper.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Conversion/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
    llvm::cl::ResetAllOptionOccurrences();
    
    if (argc < 2) {
        llvm::errs() << "Usage: " << argv[0] << " <model_path> [mlir-opt options]\n";
        return 1;
    }

    // Extract --batch_size before MlirOptMain consumes argv
    int64_t batch_size = 2;
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoll(argv[i + 1]);
            ++i;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = static_cast<int>(filtered_argv.size());

    std::string model_path = filtered_argv[1];

    DialectRegistry registry;
    registry.insert<func::FuncDialect, atir::AtirDialect, arith::ArithDialect>();
    atir::registerAllAtirPasses();
    atir::registerAtirConversionPasses();

    StandalonePbParser parser(model_path, batch_size);
    if (!parser.parse()) {
        llvm::errs() << "Error: Failed to parse PB file.\n";
        return 1;
    }
    if (!parser.isValid()) {
        llvm::errs() << "Error: Parser reported invalid state after parsing.\n";
        return 1;
    }
    
    const std::vector<annc::NodeInfo>& nodes = parser.getNodes();

    MLIRContext context(registry);
    context.loadAllAvailableDialects();
    context.allowsUnregisteredDialects();

    auto builder = std::make_shared<annc::ANNCBuilder>(&context);
    auto module = builder->buildModule("main", nodes);
    if (!module) {
        llvm::errs() << "Error: Failed to build MLIR module from nodes.\n";
        return 1;
    }

    std::string temp_bin = "temp_output.bin";
    annc::outputBinary(module, temp_bin);
    filtered_argv[1] = const_cast<char*>(temp_bin.c_str());

    int result = mlir::asMainReturnCode(
        mlir::MlirOptMain(filtered_argc, filtered_argv.data(), "ANNC Direct TF-to-ATIR Bridge (No JSON)\n", registry)
    );

    (void)llvm::sys::fs::remove(temp_bin);
    return result;
}