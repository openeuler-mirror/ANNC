#include <map>
#include <string>
#include <system_error>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"

namespace {
constexpr llvm::StringLiteral kGeneratedAttr =
    "annc.aarch64.generated_microkernel";

mlir::LogicalResult exportKernels(mlir::ModuleOp module,
                                  llvm::StringRef outputDir) {
  std::map<std::string, std::string> bySymbol;
  bool invalid = false;
  module.walk([&](mlir::Operation *op) {
    auto attr = op->getAttrOfType<mlir::DictionaryAttr>(kGeneratedAttr);
    if (!attr) return;
    auto symbol = attr.getAs<mlir::StringAttr>("symbol");
    auto assembly = attr.getAs<mlir::StringAttr>("asm");
    if (!symbol || !assembly) {
      invalid = true;
      return;
    }
    auto [it, inserted] =
        bySymbol.emplace(symbol.getValue().str(), assembly.getValue().str());
    if (!inserted) invalid |= it->second != assembly.getValue();
  });
  if (invalid) return mlir::failure();
  if (bySymbol.empty()) return mlir::success();

  llvm::SmallString<256> directory(outputDir);
  llvm::sys::path::append(directory, "model_generated_kernels");
  if (llvm::sys::fs::create_directories(directory)) return mlir::failure();
  std::map<std::string, std::string> sourceByAssembly;
  for (const auto &entry : bySymbol) {
    auto [source, inserted] =
        sourceByAssembly.emplace(entry.second, entry.first + ".S");
    if (!inserted) continue;
    llvm::SmallString<256> path(directory);
    llvm::sys::path::append(path, source->second);
    std::error_code error;
    llvm::raw_fd_ostream assembly(path, error, llvm::sys::fs::OF_Text);
    if (error) return mlir::failure();
    assembly << source->first;
    if (source->first.empty() || source->first.back() != '\n') assembly << '\n';
  }

  llvm::SmallString<256> path(directory);
  llvm::sys::path::append(path, "manifest.json");
  std::error_code error;
  llvm::raw_fd_ostream manifest(path, error, llvm::sys::fs::OF_Text);
  if (error) return mlir::failure();
  manifest << "[\n";
  size_t index = 0;
  for (const auto &entry : bySymbol) {
    const std::string &file = sourceByAssembly.find(entry.second)->second;
    manifest << "  {\"symbol\": \"" << entry.first << "\", \"file\": \"" << file
             << "\"}" << (++index == bySymbol.size() ? "\n" : ",\n");
  }
  manifest << "]\n";
  return mlir::success();
}
}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                   llvm::cl::desc("<input.mlir>"),
                                   llvm::cl::init("-"));
  llvm::cl::opt<std::string> outputDir(
      "output-dir", llvm::cl::desc("Artifact directory"), llvm::cl::init("."));
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "ANNC model AOT kernel export\n");
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
  if (!module || mlir::failed(exportKernels(*module, outputDir))) {
    llvm::errs() << "failed to export model kernels\n";
    return 1;
  }
  return 0;
}
