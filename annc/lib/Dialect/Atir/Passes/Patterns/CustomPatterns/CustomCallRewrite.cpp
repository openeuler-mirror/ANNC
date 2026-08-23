#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"

using namespace mlir;
using namespace atir;

namespace {

StringRef getFusionAbi(Operation *op) {
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc) return "mlir_ciface";
  auto metadata = parentFunc->getAttrOfType<DictionaryAttr>("fusion.metadata");
  if (!metadata) return "mlir_ciface";
  auto abi = dyn_cast_or_null<StringAttr>(metadata.get("abi"));
  return abi ? abi.getValue() : "mlir_ciface";
}

template <typename OpT>
OpT findUniqueUser(Value value, bool &ambiguous) {
  OpT found = nullptr;
  ambiguous = false;
  for (Operation *user : value.getUsers()) {
    auto candidate = dyn_cast<OpT>(user);
    if (!candidate) continue;
    if (found) {
      ambiguous = true;
      return nullptr;
    }
    found = candidate;
  }
  return found;
}

bool hasAddAndReluBoundary(ArrayRef<Operation *> fusedOps) {
  if (fusedOps.size() != 3) return false;
  auto add = dyn_cast<AddOp>(fusedOps[1]);
  auto relu = dyn_cast<ReluOp>(fusedOps[2]);
  if (!add || !relu) return false;
  SmallVector<Value> outputs = collectEscapingResults(fusedOps);
  return outputs.size() == 2 && outputs[0] == add.getResult() &&
         outputs[1] == relu.getResult();
}

bool isExecutionV2BuiltinCompatible(MatMulOp matmul, AddOp add, ReluOp relu) {
  if (!matmul || !add || !relu) return false;
  if (matmul.getWithBias() || matmul.getDoRelu() ||
      matmul.getRightTranspose() || matmul.getLeftTranspose() ||
      matmul.getOutputTranspose() || matmul.getMStart() || matmul.getNStart() ||
      matmul.getKStart() || matmul.getMSize() || matmul.getNSize() ||
      matmul.getKSize() || matmul->getAttr("rhs_format"))
    return false;
  if (add.getDoRelu() || add.getScalar()) return false;
  return relu.getReluLimit().convertToFloat() == -1.0f;
}

}  // namespace

struct MatmulToCustomCallRewrite : public CustomFusionPatternBase<MatMulOp> {
  MatmulToCustomCallRewrite(MLIRContext *context,
                            const CustomOpTypeFilter &customOpFilter,
                            PatternBenefit benefit = 8)
      : CustomFusionPatternBase<MatMulOp>(context, customOpFilter, benefit) {}

  mlir::LogicalResult matchFusion(
      MatMulOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    if (anchor->hasAttr("annc.fusion_materialized")) return failure();
    fusedOps.push_back(anchor);
    if (getFusionAbi(anchor) == "annc_execution_v2") {
      bool ambiguous = false;
      AddOp add = findUniqueUser<AddOp>(anchor.getResult(), ambiguous);
      if (ambiguous) return failure();
      if (!add) return success();

      ReluOp relu = findUniqueUser<ReluOp>(add.getResult(), ambiguous);
      if (ambiguous) return failure();
      if (relu && isExecutionV2BuiltinCompatible(anchor, add, relu)) {
        fusedOps.push_back(add);
        fusedOps.push_back(relu);
      }
      return success();
    }

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
      llvm::ArrayRef<mlir::Operation *> fusedOps) const override {
    if (getFusionAbi(anchor) == "annc_execution_v2" &&
        hasAddAndReluBoundary(fusedOps)) {
      return "MatMulAddReluWithAddOutput";
    }
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
    if (getFusionAbi(anchor) == "annc_execution_v2") {
      if (hasAddAndReluBoundary(fusedOps)) {
        return CustomOpSchema::get("MatMulAddReluWithAddOutput")
            .TypeVar("T")
            .MemRefArg("lhs", 2, "T")
            .MemRefArg("rhs", 2, "T")
            .MemRefArg("bias", 1, "T")
            .Result("add", 2, "T")
            .Result("relu", 2, "T");
      }
      if (fusedOps.size() == 3) {
        return CustomOpSchema::get("MatMulAddRelu")
            .TypeVar("T")
            .MemRefArg("lhs", 2, "T")
            .MemRefArg("rhs", 2, "T")
            .MemRefArg("bias", 1, "T")
            .Result("relu", 2, "T");
      }
      if (fusedOps.size() == 2) {
        return CustomOpSchema::get("MatMulAdd")
            .TypeVar("T")
            .MemRefArg("lhs", 2, "T")
            .MemRefArg("rhs", 2, "T")
            .MemRefArg("bias", 1, "T")
            .Result("add", 2, "T");
      }
      return CustomOpSchema::get("MatMul")
          .TypeVar("T")
          .MemRefArg("lhs", 2, "T")
          .MemRefArg("rhs", 2, "T")
          .Result("output", 2, "T");
    }

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
