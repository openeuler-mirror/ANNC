#include <string>

#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"
#include "Dialect/Atir/Passes/Patterns/CustomFusionPatternBase.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistryMacros.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir;

namespace {

constexpr llvm::StringRef kPatternName = "kp_gather";

bool isGatherKernelFunc(func::FuncOp func) {
  if (!func || !func->hasAttr("annc.kernel")) return false;
  auto metadata = func->getAttrOfType<DictionaryAttr>("fusion.metadata");
  if (!metadata) return false;
  auto pattern = dyn_cast_or_null<StringAttr>(metadata.get("fusion.pattern"));
  return pattern && pattern.getValue() == kPatternName;
}

template <typename AnchorOp>
struct KpEmbeddingGatherV2RewriteImpl
    : public CustomFusionPatternBase<AnchorOp> {
  using Base = CustomFusionPatternBase<AnchorOp>;
  using Base::Base;

  mlir::LogicalResult matchFusion(
      AnchorOp anchor,
      llvm::SmallVectorImpl<mlir::Operation *> &fusedOps) const override {
    if (!atir::fusionv2::GatherLike::match(anchor.getOperation()))
      return mlir::failure();
    auto func = anchor->template getParentOfType<func::FuncOp>();
    if (!isGatherKernelFunc(func)) return mlir::failure();
    if (func.getNumArguments() != 3) return mlir::failure();
    auto dataTy = dyn_cast<RankedTensorType>(func.getArgument(0).getType());
    auto keysTy = dyn_cast<RankedTensorType>(func.getArgument(1).getType());
    auto beginTy = dyn_cast<RankedTensorType>(func.getArgument(2).getType());
    if (!dataTy || dataTy.getRank() != 2 || !dataTy.getElementType().isF32())
      return mlir::failure();
    if (!keysTy || keysTy.getRank() != 2 ||
        !keysTy.getElementType().isSignlessInteger(64))
      return mlir::failure();
    if (!beginTy || beginTy.getRank() != 1 || beginTy.getShape()[0] != 2 ||
        !beginTy.getElementType().isInteger(32))
      return mlir::failure();
    auto returnOp =
        dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 3) return mlir::failure();
    llvm::SmallPtrSet<Operation *, 32> ordered;
    for (Value out : returnOp.getOperands()) {
      Operation *def = out.getDefiningOp();
      if (def && def->getBlock() == &func.getBody().front() &&
          ordered.insert(def).second)
        fusedOps.push_back(def);
    }
    for (Operation &op : func.getBody().front().getOperations()) {
      if (isa<func::ReturnOp>(op)) continue;
      if (ordered.insert(&op).second) fusedOps.push_back(&op);
    }
    return fusedOps.empty() ? mlir::failure() : mlir::success();
  }

  void augmentFusedSet(
      AnchorOp, llvm::ArrayRef<mlir::Operation *> fusedOps,
      llvm::SmallDenseSet<mlir::Operation *> &fusedSet) const override {
    for (mlir::Operation *op : fusedOps)
      op->walk([&](mlir::Operation *inner) {
        if (inner != op) fusedSet.insert(inner);
      });
  }

  std::string getCustomOpName(
      AnchorOp, llvm::ArrayRef<mlir::Operation *>) const override {
    return "KPFusedGather";
  }

  CustomOpSchema getCustomOpSchema(
      AnchorOp, llvm::ArrayRef<mlir::Operation *>) const override {
    return *getContextKernelSchema("KPFusedGather");
  }

  void getOrderedBoundaryInputs(
      AnchorOp anchorOp, llvm::ArrayRef<mlir::Operation *>,
      llvm::SmallVectorImpl<mlir::Value> &inputs) const override {
    auto func = anchorOp->template getParentOfType<func::FuncOp>();
    if (!func) return;
    for (Value arg : func.getArguments()) inputs.push_back(arg);
  }
};

using KpEmbeddingGatherV2TorchIndexSelectRewrite =
    KpEmbeddingGatherV2RewriteImpl<stablehlo::TorchIndexSelectOp>;
using KpEmbeddingGatherV2ResidueRewrite =
    KpEmbeddingGatherV2RewriteImpl<atir::UnsupportedOp>;

// Spelled-out dual-anchor registration; inert under the V1 pass (matches
// only ops inside op-fusion-v2 outlined kernels, absent from V1 graphs).
REGISTER_CUSTOM_PATTERN(KpEmbeddingGatherV2TorchIndexSelectRewrite);
REGISTER_CUSTOM_PATTERN(KpEmbeddingGatherV2ResidueRewrite);

}  // namespace
