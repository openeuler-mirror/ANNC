#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"

using namespace mlir;
using namespace atir;

struct MatmulToCustomCallRewrite : public CustomFusionPatternBase<MatMulOp> {
  MatmulToCustomCallRewrite(MLIRContext* context, PatternBenefit benefit = 8)
      : CustomFusionPatternBase<MatMulOp>(context, benefit){}

  mlir::LogicalResult matchFusion(
      MatMulOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    fusedOps.push_back(anchor);
    if (!anchor->hasOneUse()) {
      return success();
    }

    auto add = llvm::dyn_cast<AddOp>(*anchor->getUsers().begin());
    if (!add) {
      return success();
    }
    fusedOps.push_back(add);

    if (!add->hasOneUse()) {
      return success();
    }
    auto relu = llvm::dyn_cast<ReluOp>(*add->getUsers().begin());
    if (relu) {
      fusedOps.push_back(relu);
    }
    return success();
  }

  std::string getCustomOpName(
      MatMulOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override{
    if (fusedOps.size() == 3) {
      return "MatMulAddRelu";
    }
    if (fusedOps.size() == 2) {
      return "MatMulAdd";
    }
    return "MatMul";
  }

  // Schema args follow the base class's collected input order:
  //   MatMul-only:   [C, lhs, rhs]
  //   MatMulAdd*:    [C, lhs, rhs, bias]
  // (bias comes from AddOp's second operand, collected after MatMul's)
  CustomOpSchema getCustomOpSchema(
      MatMulOp anchor,
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    if (fusedOps.size() == 3) {
      return CustomOpSchema::get("MatMulAddRelu")
          .TypeVar("T")
          .MemRefArg("output", 2, "T")
          .MemRefArg("lhs", 2, "T")
          .MemRefArg("rhs", 2, "T")
          .MemRefArg("bias", 1, "T");
    }
    if (fusedOps.size() == 2) {
      return CustomOpSchema::get("MatMulAdd")
          .TypeVar("T")
          .MemRefArg("output", 2, "T")
          .MemRefArg("lhs", 2, "T")
          .MemRefArg("rhs", 2, "T")
          .MemRefArg("bias", 1, "T");
    }
    return CustomOpSchema::get("MatMul")
        .TypeVar("T")
        .MemRefArg("output", 2, "T")
        .MemRefArg("lhs", 2, "T")
        .MemRefArg("rhs", 2, "T");
  }
};

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
