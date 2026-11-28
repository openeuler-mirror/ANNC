#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>

#include "Target/aarch64/Gemm/GemmCodegenAdapter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
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
namespace fs = std::filesystem;
namespace codegen = annc::aarch64::gemm::codegen;
constexpr llvm::StringLiteral kGeneratedAttr =
    "annc.aarch64.generated_microkernel";

struct Request {
  std::string normalized, build;
  std::set<std::string> symbols;
};

bool identifier(llvm::StringRef value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
  });
}

mlir::LogicalResult compileKernels(mlir::ModuleOp module,
                                  const fs::path &directory) {
  std::map<std::string, Request> requests;
  std::map<std::string, std::string> symbols;
  bool invalid = false;
  module.walk([&](mlir::Operation *op) {
    auto raw = op->getAttr(kGeneratedAttr);
    if (!raw) return;
    auto attr = mlir::dyn_cast<mlir::DictionaryAttr>(raw);
    if (attr && attr.get("asm")) {
      op->emitError(
          "legacy generated microkernel assembly metadata; re-run lowering "
          "with the current codegen package");
      invalid = true;
      return;
    }
    auto string = [&](llvm::StringRef name) {
      return attr ? attr.getAs<mlir::StringAttr>(name) : mlir::StringAttr{};
    };
    auto version =
        attr ? attr.getAs<mlir::IntegerAttr>("version") : mlir::IntegerAttr{};
    auto symbol = string("symbol"), normalized = string("normalized_request"),
         artifact = string("artifact_key"), build = string("build_id");
    if (!attr || attr.size() != 5 || !version || version.getInt() != 1 ||
        !symbol || !identifier(symbol.getValue()) || !normalized || !artifact ||
        artifact.getValue().size() > 128 || !identifier(artifact.getValue()) ||
        !build || build.getValue().empty()) {
      op->emitError(
          "invalid versioned generated microkernel request; re-run lowering");
      invalid = true;
      return;
    }
    std::string key = artifact.getValue().str();
    auto [entry, inserted] = requests.emplace(
        key, Request{normalized.getValue().str(), build.getValue().str(), {}});
    auto [existing, newSymbol] = symbols.emplace(symbol.getValue().str(), key);
    if ((!inserted && (entry->second.normalized != normalized.getValue() ||
                       entry->second.build != build.getValue())) ||
        (!newSymbol && existing->second != key)) {
      op->emitError(
          "conflicting generated microkernel entry or artifact identity");
      invalid = true;
      return;
    }
    entry->second.symbols.insert(symbol.getValue().str());
  });
  if (invalid) return mlir::failure();
  std::error_code error;
  fs::create_directories(directory.parent_path(), error);
  if (error)
    return module.emitError("cannot create generated epilogue kernel parent directory: ")
           << error.message();
  if (!fs::create_directory(directory, error))
    return module.emitError("generated epilogue kernel directory must be new: ")
           << (error ? error.message() : directory.string());
  struct Cleanup {
    fs::path path;
    bool keep = false;
    ~Cleanup() {
      if (keep) return;
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }
  } cleanup{directory};
  for (const auto &[key, request] : requests) {
    codegen::CompiledObject object;
    std::string diagnostic;
    if (!codegen::compileSerializedRequest(request.normalized, request.build,
                                           object, diagnostic))
      return module.emitError("generated microkernel compilation failed: ")
             << diagnostic;
    if (object.artifactKey != key || !object.dependencySymbols.empty())
      return module.emitError(
          "codegen object identity or dependency contract mismatch");
    std::set<std::string> exported(object.definedSymbols.begin(),
                                   object.definedSymbols.end());
    if (!std::includes(exported.begin(), exported.end(),
                       request.symbols.begin(), request.symbols.end()))
      return module.emitError("prepared entry is absent from generated object");
    std::string file = key + ".o";
    std::ofstream stream(directory / file, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(object.bytes.data()),
                 object.bytes.size());
    stream.close();
    if (!stream) return module.emitError("cannot write generated object");
  }
  cleanup.keep = true;
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
                                    "ANNC GEMM epilogue kernel object compiler\n");
  fs::path directory =
      fs::path(outputDir.getValue()) / "model_generated_kernels";
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
  if (!module || mlir::failed(compileKernels(*module, directory))) {
    llvm::errs() << "failed to compile GEMM epilogue kernels\n";
    return 1;
  }
  return 0;
}
