// annc-tf2atir-v2 — new-path entry, coexisting with legacy annc-tf2atir.
// Consumes a frontend artifact (stablehlo + tf residue .mlir): the sink maps
// residues through the shared OpSpec table into atir (unmapped ones land in
// atir.unsupported), keeps other dialects, and prints [residue-sink] stats.
// The sink itself is frontend-agnostic; this tool is the TF entry.
#include <memory>
#include <string>

#include "Conversion/StablehloToAtir/StablehloSink.h"
#include "Dialect/Atir/AtirOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {

bool hasArtifactExtension(const std::string& path) {
  return llvm::StringRef(path).ends_with(".mlir");
}

int runSink(const std::string& artifactPath, const std::string& outPath,
            DialectRegistry& registry) {
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects();
  context.getDiagEngine().registerHandler([](Diagnostic& diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error) return failure();
    llvm::errs() << diag.getLocation() << ": "
                 << (diag.getSeverity() == DiagnosticSeverity::Warning
                         ? "warning: "
                         : "note: ")
                 << diag << '\n';
    return success();
  });

  auto fileOrErr =
      llvm::MemoryBuffer::getFile(artifactPath);
  if (std::error_code err = fileOrErr.getError()) {
    llvm::errs() << "Error: cannot open artifact '" << artifactPath
                 << "': " << err.message() << "\n";
    return 1;
  }
  ParserConfig parserConfig(&context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      fileOrErr.get()->getBuffer(), parserConfig, artifactPath);
  if (!module) {
    llvm::errs() << "Error: failed to parse artifact '" << artifactPath
                 << "'\n";
    return 1;
  }
  if (failed((*module).verify())) {
    llvm::errs() << "Error: artifact failed verification (see diagnostics "
                    "above); refusing to sink\n";
    return 1;
  }

  PassManager pm(&context, "builtin.module");
  pm.addPass(annc::createStablehloSink());
  pm.enableVerifier(false);  // final verify below, report diagnostics once
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "Error: residue-sink failed (see diagnostics above)\n";
    return 1;
  }
  if (failed((*module).verify())) {
    llvm::errs() << "Error: sunk module failed verification (see diagnostics "
                    "above)\n";
    return 1;
  }

  OpPrintingFlags printFlags;
  printFlags.enableDebugInfo();
  if (!outPath.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream out(outPath, ec, llvm::sys::fs::CD_CreateAlways);
    if (ec) {
      llvm::errs() << "Error: cannot open output '" << outPath
                   << "': " << ec.message() << "\n";
      return 1;
    }
    (*module).print(out, printFlags);
  } else {
    (*module).print(llvm::outs(), printFlags);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  llvm::cl::ResetAllOptionOccurrences();

  llvm::cl::opt<std::string> inputPath(
      llvm::cl::Positional, llvm::cl::desc("<artifact.mlir>"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> outPath("o", llvm::cl::desc("output file"),
                                     llvm::cl::value_desc("path"),
                                     llvm::cl::init(""));
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "ANNC frontend-artifact -> ATIR sink layer (v2)\n");

  DialectRegistry registry;
  registry.insert<func::FuncDialect, atir::AtirDialect, arith::ArithDialect,
                  shape::ShapeDialect, tensor::TensorDialect,
                  stablehlo::StablehloDialect>();
  annc::tfcompat::registerTfTypeCompatDialect(registry);

  if (!hasArtifactExtension(inputPath)) {
    llvm::errs() << "Error: expected a frontend artifact (.mlir), "
                    "got '" << inputPath << "'\n";
    return 1;
  }
  return runSink(inputPath, outPath, registry);
}
