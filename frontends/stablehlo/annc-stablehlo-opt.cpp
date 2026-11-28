#include "Dialect/Atir/AtirOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/StablehloOps.h"

void registerStablehloPartitionPasses();
void registerStablehloOnnxBindingPass();

int main(int argc, char **argv) {
  registerStablehloPartitionPasses();
  registerStablehloOnnxBindingPass();
  mlir::DialectRegistry registry;
  registry
      .insert<atir::AtirDialect, mlir::arith::ArithDialect,
              mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
              mlir::quant::QuantDialect, mlir::shape::ShapeDialect,
              mlir::tensor::TensorDialect, mlir::stablehlo::StablehloDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ANNC StableHLO frontend\n", registry));
}
