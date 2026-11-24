#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/Patterns.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Support/LLVMCompat.h"
#include "Support/Log.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace atir {

namespace {

// A do-region is inlinable when every non-terminator op is pure: after the
// hoist these ops become host-level members and must not duplicate effects.
bool isPureDoRegion(Region &region) {
  for (Block &block : region)
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>()) continue;
      if (!mlir::isMemoryEffectFree(&op) &&
          !op.hasTrait<OpTrait::AlwaysSpeculatableImplTrait>())
        return false;
    }
  return true;
}

// Inline pure shape.assuming hosts: hoists the do-region chain so members no
// longer straddle a region boundary (outlining would strand region values
// while host-side stays still reference them).
void inlineShapeAssumings(func::FuncOp f) {
  SmallVector<shape::AssumingOp> assumings;
  f.walk([&](shape::AssumingOp op) { assumings.push_back(op); });
  // Walk order is pre-order (parents first); reversing peels inner hosts
  // before their outer containers are hoisted.
  for (shape::AssumingOp assuming : llvm::reverse(assumings)) {
    if (!assuming->getBlock()) continue;  // erased with an outer host
    if (!isPureDoRegion(assuming.getDoRegion())) continue;
    Block *body = assuming.getBody();
    auto yield = dyn_cast<shape::AssumingYieldOp>(body->getTerminator());
    if (!yield) continue;  // malformed; leave for the verifier
    // Hoist the body chain (sans terminator) in front of the host; values
    // captured from above stay dominated since they dominate the host.
    assuming->getBlock()->getOperations().splice(
        assuming->getIterator(), body->getOperations(), body->begin(),
        yield->getIterator());
    for (auto [result, yielded] :
         llvm::zip(assuming.getResults(), yield->getOperands()))
      result.replaceAllUsesWith(yielded);
    assuming->erase();
  }
}

class AtirInlineShapeAssumingsPass
    : public PassWrapper<AtirInlineShapeAssumingsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AtirInlineShapeAssumingsPass)
  llvm::StringRef getArgument() const override {
    return "atir-inline-shape-assumings";
  }
  llvm::StringRef getDescription() const override {
    return "inline pure shape.assuming hosts ahead of V2 fusion (form "
           "normalization to the V1 carrier)";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      if (fn->hasAttr("fusion.metadata")) continue;
      inlineShapeAssumings(fn);
    }
  }
};

// Drop subgraphs that only compute shapes: after an embedding value is
// re-pointed at a kernel output, the lookup/mask/shape chains are value-dead;
// dangling shape operands re-point at shape_of(the data input).
void dropShapeOnlySubgraphs(func::FuncOp f) {
  if (f.empty()) return;
  Block &entry = f.getBody().front();
  llvm::SmallDenseSet<Operation *, 64> dropSet;
  auto shapeOperandIndex = [](Operation *op) -> int {
    if (isa<stablehlo::DynamicReshapeOp>(op)) return 1;
    if (isa<stablehlo::DynamicBroadcastInDimOp>(op)) return 1;
    return -1;
  };
  auto deletable = [](Operation *op) {
    if (mlir::isMemoryEffectFree(op)) return true;
    if (op->hasTrait<OpTrait::AlwaysSpeculatableImplTrait>()) return true;
    if (op->getName().getDialectNamespace() != "atir" ||
        op->hasTrait<OpTrait::IsTerminator>())
      return false;
    // Residues may wrap side-effecting TF ops; only the shape/mask wrapper
    // list the matcher trusts is droppable (fill and other pure carriers
    // stay unconditional).
    if (auto residue = dyn_cast<atir::UnsupportedOp>(op)) {
      static const std::set<llvm::StringRef> kDroppable = {
          "Where",  "Select",  "NotEqual", "ZerosLike", "Tile",
          "Reshape", "Cast",   "Slice",    "StridedSlice", "Pack",
          "ConcatV2", "Prod",  "ExpandDims", "Squeeze", "Fill"};
      return kDroppable.count(residue.getOpType()) != 0;
    }
    return true;
  };
  auto allUsesDroppedOrShapeOperand = [&](Operation *op) {
    for (Value r : op->getResults())
      for (OpOperand &use : r.getUses()) {
        Operation *u = use.getOwner();
        if (dropSet.contains(u)) continue;
        int idx = shapeOperandIndex(u);
        if (idx >= 0 && static_cast<int>(use.getOperandNumber()) == idx)
          continue;
        return false;
      }
    return true;
  };
  llvm::SmallDenseSet<Operation *, 32> protectedOps;
  bool keepGoing = true;
  while (keepGoing) {
    keepGoing = false;
    dropSet.clear();
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : entry) {
        if (isa<func::ReturnOp>(op) || dropSet.contains(&op)) continue;
        if (protectedOps.contains(&op)) continue;
        if (!deletable(&op)) continue;
        if (allUsesDroppedOrShapeOperand(&op)) {
          dropSet.insert(&op);
          changed = true;
        }
      }
    }
    // Validate substitutions: a shape_of(data) rewrite only preserves rank
    // when the data rank equals the width of the shape vector it replaces
    // (reinterpret-style reshapes). Anything else protects its producer.
    bool protectedAny = false;
    for (Operation &op : entry) {
      if (dropSet.contains(&op)) continue;
      int idx = shapeOperandIndex(&op);
      if (idx < 0) continue;
      Value shapeVal = op.getOperand(idx);
      Operation *def = shapeVal.getDefiningOp();
      if (!def || !dropSet.contains(def)) continue;
      auto shapeTy = dyn_cast<RankedTensorType>(shapeVal.getType());
      auto dataTy = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
      bool match = shapeTy && dataTy && shapeTy.getRank() == 1 &&
                   shapeTy.isDynamicDim(0) == false &&
                   shapeTy.getDimSize(0) == dataTy.getRank();
      if (match) {
        // A reinterpret contract is a shape_of-rooted description chain; a
        // constant target shape is a real reshape and must survive.
        Operation *cur = def;
        bool shapeOfRooted = false;
        for (int hop = 0; cur != nullptr && hop < 16; ++hop) {
          if (isa<shape::ShapeOfOp>(cur)) {
            shapeOfRooted = true;
            break;
          }
          bool advanced = false;
          for (Value o : cur->getOperands())
            if (Operation *d = o.getDefiningOp()) {
              StringRef ns = d->getName().getDialectNamespace();
              if (ns == "shape" || ns == "arith") {
                cur = d;
                advanced = true;
                break;
              }
            }
          if (!advanced) break;
        }
        if (shapeOfRooted) continue;
        // fall through: protect the producer closure
      }
      // Protect the whole producer closure so the next round leaves it be.
      SmallVector<Operation *, 16> stack{def};
      while (!stack.empty()) {
        Operation *cur = stack.pop_back_val();
        if (!protectedOps.insert(cur).second) continue;
        protectedAny = true;
        for (Value o : cur->getOperands())
          if (Operation *d = o.getDefiningOp())
            if (dropSet.contains(d)) stack.push_back(d);
      }
    }
    if (protectedAny) keepGoing = true;
  }
  if (dropSet.empty()) return;
  // Re-point shape operands that lost their producer before erasing.
  OpBuilder builder(f.getContext());
  for (Operation &op : llvm::make_early_inc_range(entry)) {
    if (dropSet.contains(&op)) continue;
    int idx = shapeOperandIndex(&op);
    if (idx < 0) continue;
    Value shapeVal = op.getOperand(idx);
    Operation *def = shapeVal.getDefiningOp();
    if (!def || !dropSet.contains(def)) continue;
    builder.setInsertionPoint(&op);
    auto shapeOf =
        builder.create<shape::ShapeOfOp>(op.getLoc(), op.getOperand(0));
    op.setOperand(idx, shapeOf.getResult());
  }
  SmallVector<Operation *> ops(dropSet.begin(), dropSet.end());
  llvm::sort(ops, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  for (Operation *op : llvm::reverse(ops)) op->erase();
}

class AtirDropShapeOnlySubgraphsPass
    : public PassWrapper<AtirDropShapeOnlySubgraphsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AtirDropShapeOnlySubgraphsPass)
  llvm::StringRef getArgument() const override {
    return "atir-drop-shape-only-subgraphs";
  }
  llvm::StringRef getDescription() const override {
    return "drop value-dead shape-only subgraphs after V2 fusion";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      if (fn->hasAttr("fusion.metadata")) continue;
      dropShapeOnlySubgraphs(fn);
    }
  }
};

class AtirOpFusionV2Pass
    : public PassWrapper<AtirOpFusionV2Pass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AtirOpFusionV2Pass)
  llvm::StringRef getArgument() const override { return "atir-op-fusion-v2"; }
  llvm::StringRef getDescription() const override {
    return "operator fusion over the stablehlo+atir_v2 mixed carrier (v2)";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp main = module.lookupSymbol<func::FuncOp>("main");
    if (!main) {
      ANNC_LOG_DEBUG("op-fusion-v2") << "no @main; nothing to probe\n";
      return;
    }

    // Cascade ports live exactly one runOnOperation; the probes run
    // main-first so a later site's boundary reads this run's entries.
    fusionv2::CallPortTable portTable;
    RewritePatternSet patterns(&getContext());
    populateAtirOpFusionV2Patterns(patterns, &portTable);
    // (hash also rides in this set; the dedicated round above consumes those
    // sites first, so the extra table here is a no-op at match time.)

    // tf.If passthrough branch funcs (_functionalize_if_*_branch) hold
    // sites too (presort measurement); kernel funcs are skipped - their
    // members are already absorbed.
    SmallVector<func::FuncOp> probes;
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      if (fn->hasAttr("fusion.metadata")) continue;
      probes.push_back(fn);
    }
    llvm::stable_sort(probes, [](func::FuncOp a, func::FuncOp b) {
      return (a.getName() == "main") > (b.getName() == "main");
    });
    // Hash sites run first and own their columns whole; later sites then match
    // the settled graph and cannot strip hash-domain chains back to TF.
    RewritePatternSet hashPatterns(&getContext());
    addDnnEmbeddingHashBucketV2Pattern(hashPatterns, &portTable);
    FrozenRewritePatternSet frozenHash(std::move(hashPatterns));
    for (func::FuncOp fn : probes) {
      // No folding here either: the hash outline rebuilds shared shape
      // scaffolding, and folding it away would erase the anchors the
      // weights-side / linear segment sites match on.
      if (failed(applyPatternsGreedily(fn, frozenHash))) {
        signalPassFailure();
        return;
      }
    }
    // Sparse and foundational patterns share one greedy round; a settled-
    // graph second round was tried and rolled back (folding an outlined
    // graph rewrote shape scaffolding the sites matched on).
    FrozenRewritePatternSet frozen(std::move(patterns));
    for (func::FuncOp fn : probes) {
      // No folding in the settled round: folding an already-outlined graph
      // rewrites shape scaffolding the sites were built against.
      if (failed(applyPatternsGreedily(fn, frozen))) {
        if (std::getenv("ANNC_DUMP_ON_FAIL") != nullptr) {
          llvm::errs() << "===== op-fusion-v2 failed on " << fn.getName()
                       << " =====\n";
          fn.print(llvm::errs());
          llvm::errs() << "\n";
        }
        signalPassFailure();
        return;
      }
    }

    int kernels = 0;
    for (func::FuncOp fn : module.getOps<func::FuncOp>())
      if (fn->hasAttr("fusion.metadata")) ++kernels;
    ANNC_LOG_INFO("op-fusion-v2")
        << "fusion kernels in module: " << kernels << "\n";
  }
};
}  // namespace

void populateAtirOpFusionV2Patterns(RewritePatternSet &patterns,
                                    fusionv2::CallPortTable *portTable) {
  addKpGatherV2Pattern(patterns, portTable);
  addActionIdGatherV2Pattern(patterns, portTable);
  addDnnEmbeddingHashBucketV2Pattern(patterns, portTable);
  addEmbeddingPaddingV2Pattern(patterns, portTable);
  addEmbeddingPaddingFastV2Pattern(patterns, portTable);
  addSparseDynamicStitchV2Pattern(patterns, portTable);
  addSparseSegmentReduceV2Pattern(patterns, portTable);
  addSparseReshapeV2Pattern(patterns, portTable);
  addSparseSelectV2Pattern(patterns, portTable);
  addTargetBehaviorInteractionV2Pattern(patterns, portTable);
  addDotGemmV2Pattern(patterns);
}

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionV2Pass() {
  return std::make_unique<AtirOpFusionV2Pass>();
}

void registerAtirOpFusionV2Pass() {
  PassRegistration<AtirOpFusionV2Pass>();
  PassRegistration<AtirInlineShapeAssumingsPass>();
  PassRegistration<AtirDropShapeOnlySubgraphsPass>();
}

}  // namespace atir
