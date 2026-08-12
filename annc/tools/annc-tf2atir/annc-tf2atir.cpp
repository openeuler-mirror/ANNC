#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Builder/Builder.h"
#include "Helper.h"
#include "node_info_adapter.h"
#include "tf_frontend.h"
#include "tf_tensor_resolver.h"

#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1

#include "Conversion/Passes.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::cl::ResetAllOptionOccurrences();

  if (argc < 2) {
    llvm::errs() << "Usage: " << argv[0]
                 << " <model_path> [--batch_size N] "
                    "[--output_tensor <name>]... "
                    "[mlir-opt options]\n";
    return 1;
  }

  // Extract --batch_size before MlirOptMain consumes argv
  // -1 means "not specified": keep dynamic shapes as-is
  int64_t batch_size = -1;
  std::vector<std::string> output_tensors;
  std::vector<char *> filtered_argv;
  filtered_argv.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--batch_size" && i + 1 < argc) {
      batch_size = std::stoll(argv[i + 1]);
      ++i;
    } else if (std::string(argv[i]) == "--output_tensor") {
      if (i + 1 >= argc || std::string(argv[i + 1]).empty() ||
          std::string(argv[i + 1]).front() == '-') {
        llvm::errs() << "Error: --output_tensor requires a non-empty value.\n";
        return 1;
      }
      output_tensors.emplace_back(argv[i + 1]);
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

  annc::tf2atir::LoadedTfModel loaded_model;
  annc::tf2atir::TfModelLoader model_loader;
  std::string error;
  if (!model_loader.load(model_path, output_tensors, loaded_model, error)) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }
  annc::tf2atir::TfGraph graph;
  annc::tf2atir::TfGraphParser graph_parser;
  if (!graph_parser.parse(*loaded_model.graph_def, loaded_model.output_names,
                          graph, error)) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }
  annc::tf2atir::ResolvedTfGraph resolved_graph;
  annc::tf2atir::TfTensorResolver tensor_resolver;
  if (!tensor_resolver.resolve(
          graph, resolved_graph, error,
          annc::tf2atir::ShapeOverridePolicy{batch_size})) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }
  std::vector<annc::NodeInfo> nodes;
  annc::tf2atir::NodeInfoAdapter adapter;
  if (!adapter.adapt(resolved_graph, nodes, error)) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowsUnregisteredDialects();

  // The MLIR diagnostic engine drops Warning/Note diagnostics when no
  // handler is registered; only Errors reach stderr by default. The builder
  // relies on emitWarning for unmapped-attribute reporting, so surface
  // warnings here. Errors keep the default formatting (return failure()).
  context.getDiagEngine().registerHandler([](Diagnostic& diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) return failure();
    llvm::errs() << diag.getLocation() << ": "
                 << (diag.getSeverity() == DiagnosticSeverity::Warning
                         ? "warning: "
                         : "note: ")
                 << diag << '\n';
    return success();
  });

  auto builder = std::make_shared<annc::ANNCBuilder>(&context);
  auto module = builder->buildModule("main", nodes);
  if (!module) {
    llvm::errs() << "Error: Failed to build MLIR module from nodes.\n";
    return 1;
  }

  std::string temp_bin = "temp_output.bin";
  annc::outputBinary(module, temp_bin);
  filtered_argv[1] = const_cast<char *>(temp_bin.c_str());

  int result = mlir::asMainReturnCode(
      mlir::MlirOptMain(filtered_argc, filtered_argv.data(),
                        "ANNC Direct TF-to-ATIR Bridge (No JSON)\n", registry));

  (void)llvm::sys::fs::remove(temp_bin);
  return result;
}
