#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "FusionMetadata/FusionMetadata.h"
#include "FusionMetadata/FusionMetadataJson.h"
#ifdef ANNC_ENABLE_TF2ATIR_V2
#include "Conversion/StablehloToAtir/StablehloSink.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"
#endif
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

static int reportError(StringRef prefix, llvm::Error error) {
  llvm::errs() << prefix << llvm::toString(std::move(error)) << "\n";
  return 1;
}

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<fused atir .mlir>"),
      llvm::cl::Required);
  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output metadata JSON"),
      llvm::cl::value_desc("filename"), llvm::cl::Required);

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "annc-fusion-metadata: extract ANNCFused metadata\n");

  DialectRegistry registry;
  registry.insert<func::FuncDialect, LLVM::LLVMDialect, arith::ArithDialect,
#if ANNC_ENABLE_TF2ATIR_V2
                  // V2 fused artifacts keep the mixed stablehlo/shape/
                  // arith/tensor carriers plus tf_type compat types.
                  mlir::shape::ShapeDialect, mlir::tensor::TensorDialect,
                  mlir::stablehlo::StablehloDialect,
#endif
                  atir::AtirDialect>();
#if ANNC_ENABLE_TF2ATIR_V2
  annc::tfcompat::registerTfTypeCompatDialect(registry);
#endif
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::errs() << "[annc-fusion-metadata] Error: failed to parse ATIR: "
                 << inputFilename << "\n";
    return 1;
  }

  auto fusions = annc::fusion::extractFusionInfos(*module);
  if (!fusions) {
    return reportError("[annc-fusion-metadata] Error: ",
                       fusions.takeError());
  }

  if (auto err =
          annc::fusion::writeFusionMetadataJson(*fusions, outputFilename)) {
    return reportError("[annc-fusion-metadata] Error: ", std::move(err));
  }
  return 0;
}
