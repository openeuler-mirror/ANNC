#include "standalone_pb_parser.h"
#include "Builder/Builder.h"
#include "Helper.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

// 严格禁用冲突项
#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
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
    
    std::string model_path = argv[1];

    DialectRegistry registry;
    registry.insert<func::FuncDialect, atir::AtirDialect, arith::ArithDialect>();
    atir::registerAllAtirPasses();
    atir::registerAtirConversionPasses();

    // 使用隔离解析器直接解析 PB 文件为 NodeInfo 列表
    StandalonePbParser parser(model_path);
    if (!parser.parse()) {
        llvm::errs() << "Error: Failed to parse PB file.\n";
        return 1;
    }
    if (!parser.isValid()) {
        llvm::errs() << "Error: Parser reported invalid state after parsing.\n";
        return 1;
    }
    
    // 直接获取解析后的节点列表（无需JSON转换）
    const std::vector<annc::NodeInfo>& nodes = parser.getNodes();

    // 初始化 MLIR 环境
    MLIRContext context(registry);
    context.loadAllAvailableDialects();
    context.allowsUnregisteredDialects();

    // 直接使用 NodeInfo 列表构建 MLIR Module
    auto builder = std::make_shared<annc::ANNCBuilder>(&context);
    auto module = builder->buildModule("main", nodes);
    if (!module) {
        llvm::errs() << "Error: Failed to build MLIR module from nodes.\n";
        return 1;
    }

    // 输出到临时二进制文件供 mlir-opt 处理
    std::string temp_bin = "temp_output.bin";
    annc::outputBinary(module, temp_bin);
    argv[1] = const_cast<char*>(temp_bin.c_str());

    int result = mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "ANNC Direct TF-to-ATIR Bridge (No JSON)\n", registry)
    );

    (void)llvm::sys::fs::remove(temp_bin);
    return result;
}