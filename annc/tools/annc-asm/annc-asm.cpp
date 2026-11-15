#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "Dialect/Atir/Passes/Passes.h"
#include "Conversion/Passes.h"
#include "Target/Passes.h"
#include "Dialect/Atir/AtirOps.h"
#ifdef ANNC_ENABLE_TF2ATIR_V2
#include "Conversion/StablehloToAtir/StablehloSink.h"
#include "stablehlo/dialect/StablehloOps.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef ANNC_ENABLE_TF2ATIR_V2
#include "stablehlo/transforms/Passes.h"
#endif

using namespace mlir;

int main(int argc, char **argv) {
  atir::registerAllAtirPasses();
  atir::registerAtirConversionPasses();
  annc::registerAllTargetPasses();
#ifdef ANNC_ENABLE_TF2ATIR_V2
  mlir::stablehlo::registerShapeLegalizeToStablehloPass();
  mlir::registerReconcileUnrealizedCastsPass();
#endif
#ifdef ANNC_ASM_ENABLE_SHAPE_LEGALIZE
  // Register the upstream pass under its native option name. ANNC owns the
  // pipeline boundary, while StableHLO owns the Shape-to-StableHLO rewrite.
  mlir::stablehlo::registerShapeLegalizeToStablehloPass();
  mlir::registerReconcileUnrealizedCastsPass();
#endif

  DialectRegistry registry;
  registry.insert<
          mlir::func::FuncDialect,
          mlir::LLVM::LLVMDialect,
          mlir::arith::ArithDialect,
          mlir::linalg::LinalgDialect,
          mlir::memref::MemRefDialect,
          mlir::affine::AffineDialect,
          mlir::bufferization::BufferizationDialect,
          mlir::vector::VectorDialect,
          mlir::scf::SCFDialect,
          mlir::tensor::TensorDialect,
          mlir::ub::UBDialect,
          mlir::shape::ShapeDialect,
          atir::AtirDialect
#if ANNC_ENABLE_TF2ATIR_V2
          , mlir::stablehlo::StablehloDialect
#endif
          >();
#if ANNC_ENABLE_TF2ATIR_V2
  // V2 sink artifacts may carry tf_type compat types (resource etc.); the
  // compat dialect must be registered for parsing.
  annc::tfcompat::registerTfTypeCompatDialect(registry);
#endif
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowsUnregisteredDialects();

  // mlir::OpPrintingFlags flags;
  // flags.elideLargeElementsAttrs(0);
  // root->print(llvm::errs(), flags);
  // annc-asm input.bin --xxx_pass
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ANNC Code Generation\n", registry));
}
