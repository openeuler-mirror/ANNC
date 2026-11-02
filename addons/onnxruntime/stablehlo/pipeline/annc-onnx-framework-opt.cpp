#include "onnx_preprocess.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace onnx_mlir;

void registerOnnxStringToFrameworkCustomCallPass();
void registerAnncFrameworkDialect(DialectRegistry &registry);
void registerFrameworkVerifyPass();
void registerFrameworkShapePass();
void registerFrameworkCustomCallToStablehloPass();

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  removeUnrelatedOptions({&OnnxMlirCommonOptions, &OnnxMlirOptOptions});

  registerOnnxSlicePreprocessPass();
  registerOnnxDynamicTilePreprocessPass();
  registerOnnxStringToFrameworkCustomCallPass();
  registerFrameworkVerifyPass();
  registerFrameworkShapePass();
  registerFrameworkCustomCallToStablehloPass();
  registerTransformsPasses();

  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                  shape::ShapeDialect, tensor::TensorDialect,
                  stablehlo::StablehloDialect, ONNXDialect>();
  registerAnncFrameworkDialect(registry);

  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  if (!llvm::cl::ParseCommandLineOptions(
          argc, argv, "ANNC ONNX framework passes\n", &llvm::errs()))
    return 1;

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << "Failed to open input: " << errorMessage << "\n";
    return 1;
  }
  auto output = openOutputFile(outputBaseName, &errorMessage);
  if (!output) {
    llvm::errs() << "Failed to open output: " << errorMessage << "\n";
    return 1;
  }

  MlirOptMainConfig config;
  config.setPassPipelineParser(passPipeline)
      .splitInputFile(split_input_file ? kDefaultSplitMarker : "")
      .verifyDiagnostics(verify_diagnostics)
      .verifyPasses(verify_passes)
      .allowUnregisteredDialects(allowUnregisteredDialects)
      .emitBytecode(false)
      .useExplicitModule(false);
  if (failed(MlirOptMain(output->os(), std::move(input), registry, config)))
    return 1;

  output->keep();
  return 0;
}
