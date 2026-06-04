#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"

using namespace mlir;
using namespace atir;

struct MatmulAddReluToCustomCallRewrite : public CustomFusionPatternBase<MatMulOp> {
  MatmulAddReluToCustomCallRewrite(MLIRContext* context, PatternBenefit benefit = 8)
      : CustomFusionPatternBase<MatMulOp>(context, benefit){}
  mlir::LogicalResult matchFusion(
      MatMulOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {


    fusedOps.push_back(anchor);
    if (!anchor->hasOneUse()) {
      return failure();
    }
    auto add = llvm::dyn_cast<AddOp>(*anchor->getUsers().begin());
    if (!add) {
      return failure();
    }
    fusedOps.push_back(add);
    if (!add->hasOneUse()) {
      return failure();
    }
    auto relu = llvm::dyn_cast<ReluOp>(*add->getUsers().begin());
    if (!relu) {
      return failure();
    }
    fusedOps.push_back(relu);
    return success();
  }

  std::string getKernelName(
      MatMulOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override{
    return "my_matmul_add_relu";
  }
};

struct MatmulAddToCustomCallRewrite : public CustomFusionPatternBase<MatMulOp> {
  MatmulAddToCustomCallRewrite(MLIRContext* context, PatternBenefit benefit = 7)
      : CustomFusionPatternBase<MatMulOp>(context, benefit){}
  mlir::LogicalResult matchFusion(
      MatMulOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {


    fusedOps.push_back(anchor);
    if (!anchor->hasOneUse()) {
      return failure();
    }
    auto add = llvm::dyn_cast<AddOp>(*anchor->getUsers().begin());
    if (!add) {
      return failure();
    }
    fusedOps.push_back(add);
    return success();
  }

  std::string getKernelName(
      MatMulOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override{
    return "my_matmul_add";
  }
};

struct MatmulToCustomCallRewrite : public CustomFusionPatternBase<MatMulOp> {
  MatmulToCustomCallRewrite(MLIRContext* context, PatternBenefit benefit = 6)
      : CustomFusionPatternBase<MatMulOp>(context, benefit){}
  mlir::LogicalResult matchFusion(
      MatMulOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    fusedOps.push_back(anchor);
    return success();
  }

  std::string getKernelName(
      MatMulOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override{
    return "MatMul";
  }
};

REGISTER_CUSTOM_PATTERN(MatmulAddReluToCustomCallRewrite);
REGISTER_CUSTOM_PATTERN(MatmulAddToCustomCallRewrite);
REGISTER_CUSTOM_PATTERN(MatmulToCustomCallRewrite);

// ---------------------------------------------------------------------------
// Patterns for ops that only exist in external kernel libraries.
// These are guarded by the corresponding compile-time macro so that the
// pattern code is only compiled when the library is available.
// At runtime, CustomFusionPatternBase::matchAndRewrite automatically checks
// hasAnyAvailableKernel() — if the kernel isn't available (e.g. kdnn is
// compiled in but --enable-kdnn is not passed), the pattern returns failure
// and the op falls through to its default lowering.
//
// Example:
//   #ifdef ANNC_ENABLE_KDNN_ADAPTOR
//   struct KdnnBatchMatmulRewrite
//       : public CustomFusionPatternBase<BatchMatmulOp> { ... };
//   REGISTER_CUSTOM_PATTERN(KdnnBatchMatmulRewrite)
//   #endif
// ---------------------------------------------------------------------------
