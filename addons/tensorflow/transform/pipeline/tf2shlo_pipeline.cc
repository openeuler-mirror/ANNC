// The legalize orchestration: the wheel's odml passes interleaved with
// the wrap-layer passes (pb2::), the Smuggle guard, and signature sync.
// This TU is the last of the link closure; the pb2stablehlo executable
// links it with every object of the transform.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "common/pb2_common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

namespace pb2 {
std::unique_ptr<mlir::Pass> createTfRankInferPass();
std::unique_ptr<mlir::Pass> createStringTransportPass();
std::unique_ptr<mlir::Pass> createStringUnwrapPass();
std::unique_ptr<mlir::Pass> createUnrankWrapPass();
std::unique_ptr<mlir::Pass> createStridedSliceDynPass();
}  // namespace pb2

namespace mlir {
class Pass;

std::unique_ptr<Pass> createSymbolDCEPass();
std::unique_ptr<Pass> createCanonicalizerPass();

namespace odml {
std::unique_ptr<Pass> CreateLegalizeTFXlaCallModuleToStablehloPass();
void AddLegalizeTFToStablehloPasses(OpPassManager& pm,
                                    bool skip_quantization_ops,
                                    bool skip_resize,
                                    bool skip_partitioned_calls);
}  // namespace odml
namespace TF {
std::unique_ptr<Pass> CreateDropWhileShapeInvariantPass();
std::unique_ptr<Pass> CreateTFFunctionalControlFlowToRegions();
}  // namespace TF
namespace tf_executor {
std::unique_ptr<Pass> CreateTFExecutorGraphPruningPass(
    llvm::ArrayRef<std::string> devices);
std::unique_ptr<Pass> CreateTFExecutorIslandCoarseningPass();
}  // namespace tf_executor
namespace TFDevice {
std::unique_ptr<Pass> CreateDecomposeResourceOpsPass();
}  // namespace TFDevice
namespace tf_saved_model {
std::unique_ptr<Pass> CreateOptimizeGlobalTensorsPass();
std::unique_ptr<Pass> CreateFreezeGlobalTensorsPass(bool allow_mutable);
}  // namespace tf_saved_model
namespace quant {
namespace stablehlo {
std::unique_ptr<Pass> CreateConvertTFQuantOpsToMHLOPass();
}  // namespace stablehlo
}  // namespace quant
}  // namespace mlir

namespace shlo_pipeline {

// func.func signature sync (pipeline tail). Definitions only: call sites
// are not touched -- odml's passes keep callee types consistent while the
// tail only repairs signatures that the wrap-layer type edits loosened.
struct SyncFuncSignaturePass
    : public mlir::PassWrapper<SyncFuncSignaturePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SyncFuncSignaturePass)
  llvm::StringRef getArgument() const override { return "sync-func-signature"; }
  llvm::StringRef getDescription() const override {
    return "sync func.func signature with actual entry-block types";
  }
  void runOnOperation() override {
    llvm::SmallVector<mlir::Type, 8> argTys;
    llvm::SmallVector<mlir::Type, 4> resTys;
    getOperation()->walk([&](mlir::Operation* fn) {
      if (fn->getName().getStringRef() != "func.func") return;
      if (fn->getNumRegions() == 0 || fn->getRegion(0).empty()) return;
      auto& block = fn->getRegion(0).front();
      argTys.clear();
      resTys.clear();
      for (mlir::Value a : block.getArguments()) argTys.push_back(a.getType());
      for (auto& op : block.getOperations())
        if (op.getName().getStringRef() == "func.return")
          for (mlir::Value r : op.getOperands()) resTys.push_back(r.getType());
      auto ft = mlir::FunctionType::get(fn->getContext(), argTys, resTys);
      fn->setAttr("function_type", mlir::TypeAttr::get(ft));
    });
  }
};

namespace {
}  // namespace

namespace {

// True if every operand/result element type is stablehlo-representable.
bool HasStablehloPayload(mlir::Operation* op) {
  auto ok = [](mlir::Type t) {
    auto tt = llvm::dyn_cast<mlir::TensorType>(t);
    if (!tt) return false;
    mlir::Type et = tt.getElementType();
    return llvm::isa_and_nonnull<mlir::FloatType, mlir::IntegerType,
                                 mlir::ComplexType>(et);
  };
  return llvm::all_of(op->getOperandTypes(), ok) &&
         llvm::all_of(op->getResultTypes(), ok);
}

// True when a dot-family op has a rank<2 operand.
bool DotFamilyRankFloorViolated(mlir::Operation* op) {
  static const std::set<std::string> kDotFamily = {
      "MatMul", "BatchMatMulV2", "Einsum"};
  llvm::StringRef name = op->getName().getStringRef();
  if (!name.starts_with("tf.") ||
      !kDotFamily.count(name.drop_front(3).str()))
    return false;
  for (mlir::Value v : op->getOperands()) {
    auto t = llvm::dyn_cast<mlir::TensorType>(v.getType());
    if (t && t.hasRank() && t.getRank() < 2) return true;
  }
  return false;
}

// Guard rail: registered ops with unknown-shape inputs segfault the
// legalization patterns (tf.GatherNd with unranked params, bisected).
// Such ops go into stablehlo.custom_call carriers instead; the backend
// dispatches by call_target_name and owns the shape decisions.
struct SmuggleUnrankedOpsPass
    : public mlir::PassWrapper<SmuggleUnrankedOpsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SmuggleUnrankedOpsPass)

  llvm::StringRef getArgument() const override {
    return "smuggle-unranked-ops";
  }
  llvm::StringRef getDescription() const override {
    return "wrap rank-sensitive tf ops with unranked operands into "
           "stablehlo.custom_call";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* ctx = &getContext();
    mlir::OpBuilder b(ctx);
    auto isUnranked = [](mlir::Value v) {
      auto t = llvm::dyn_cast<mlir::TensorType>(v.getType());
      return t != nullptr && !t.hasRank();
    };
    llvm::SmallVector<std::pair<mlir::Operation*, const char*>> targets;
    module.walk([&](mlir::Operation* op) {
      llvm::StringRef name = op->getName().getStringRef();
      if (!name.starts_with("tf.") || name == "tf.residue")
        return;
      if (!g_opspecs.count(name.drop_front(3).str())) return;
      if (!HasStablehloPayload(op)) return;
      // Unranked results are caught too (operand-only checks leaked
      // mixed operand-ranked/result-unranked states into legalize;
      // annc-cvr root cause, now closed).
      bool resultUnranked = false;
      for (mlir::Type t : op->getResultTypes()) {
        auto tt = llvm::dyn_cast<mlir::TensorType>(t);
        if (tt && !tt.hasRank()) {
          resultUnranked = true;
          break;
        }
      }
      // Mul: mixed operand element types crash the legalize template
      // instantiation (same annc-cvr investigation as the resultUnranked
      // case below; corpus showed the mismatch only on Mul).
      if (name == "tf.Mul") {
        mlir::Type elemMulTy = nullptr;
        bool elemMismatch = false;
        for (mlir::Type t : op->getOperandTypes()) {
          auto tt = llvm::dyn_cast<mlir::TensorType>(t);
          if (!tt || (elemMulTy && tt.getElementType() != elemMulTy)) {
            elemMismatch = true;
            break;
          }
          if (elemMulTy == nullptr) elemMulTy = tt.getElementType();
        }
        if (elemMismatch) {
          targets.push_back({op, "elem_type_mismatch"});
        } else if (resultUnranked || llvm::any_of(op->getOperands(), isUnranked)) {
          targets.push_back({op, "unranked"});
        }
        return;
      }
      // Gather/GatherV2 fold into the generic branch: their extra
      // "!rt || !rt.hasRank()" condition is exactly resultUnranked, and a
      // non-tensor result cannot appear on a payload-checked op.
      if (llvm::any_of(op->getOperands(), isUnranked) || resultUnranked) {
        targets.push_back({op, "unranked"});
        return;
      }
      if (DotFamilyRankFloorViolated(op))
        targets.push_back({op, "rank_floor"});
    });

    for (auto it = targets.rbegin(); it != targets.rend(); ++it) {
      mlir::Operation* op = it->first;
      b.setInsertionPoint(op);
      mlir::OperationState st(op->getLoc(),
                              mlir::OperationName("stablehlo.custom_call", ctx));
      st.operands.append(op->operand_begin(), op->operand_end());
      st.addTypes(op->getResultTypes());
      st.addAttribute("call_target_name",
                      b.getStringAttr(op->getName().getStringRef()));
      st.addAttribute("residue_reason", b.getStringAttr(it->second));
      CopyNonReservedAttrs(st, op);
      mlir::Operation* call = b.create(st);
      op->replaceAllUsesWith(call);
      op->erase();
    }
  }
};

}

// tf->stablehlo legalization pipeline: xla-call legalize, executor cleanup,
// canonicalize, resource/global-tensor handling, the Smuggle guard, then
// the main legalization.
void AddTFToStablehloPasses(mlir::OpPassManager& pm) {
  pm.addPass(mlir::odml::CreateLegalizeTFXlaCallModuleToStablehloPass());

  pm.nest("func.func")
      .addPass(mlir::TF::CreateDropWhileShapeInvariantPass());
  pm.nest("func.func")
      .addPass(mlir::tf_executor::CreateTFExecutorGraphPruningPass({}));
  pm.nest("func.func")
      .addPass(mlir::tf_executor::CreateTFExecutorIslandCoarseningPass());
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());

  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.nest("func.func")
      .addPass(mlir::TFDevice::CreateDecomposeResourceOpsPass());

  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  pm.addPass(
      mlir::tf_saved_model::CreateFreezeGlobalTensorsPass(
          /*allow_mutable_tensors=*/true));

  pm.addPass(mlir::createCanonicalizerPass());
  pm.nest("func.func")
      .addPass(
          mlir::quant::stablehlo::CreateConvertTFQuantOpsToMHLOPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Kill-switches are process-wide: read once on first pipeline build.
  static const bool no_rank_infer =
      std::getenv("ANNC_NO_TF_RANK_INFER") != nullptr;
  if (!no_rank_infer) pm.addPass(pb2::createTfRankInferPass());

  static const bool no_unrank_wrap =
      std::getenv("ANNC_NO_UNRANK_WRAP") != nullptr;
  if (!no_unrank_wrap) pm.addPass(pb2::createUnrankWrapPass());
  // Constant-bounds StridedSlice on dynamic inputs: tf2xla's hasStaticShape
  // gate strands every ranked-dynamic instance; translate them here instead.
  static const bool no_ss_dyn =
      std::getenv("ANNC_NO_STRIDED_SLICE_DYN") != nullptr;
  if (!no_ss_dyn) pm.addPass(pb2::createStridedSliceDynPass());
  static const bool no_string_transport =
      std::getenv("ANNC_NO_STRING_TRANSPORT") != nullptr;
  if (!no_string_transport) pm.addPass(pb2::createStringTransportPass());

  pm.addPass(std::make_unique<SmuggleUnrankedOpsPass>());

  mlir::odml::AddLegalizeTFToStablehloPasses(
      pm, /*skip_quantization_ops=*/false, /*skip_resize=*/false,
      /*skip_partitioned_calls=*/false);
  static const bool no_string_unwrap =
      std::getenv("ANNC_NO_STRING_UNWRAP") != nullptr;
  if (!no_string_unwrap) pm.addPass(pb2::createStringUnwrapPass());
  pm.addPass(std::make_unique<SyncFuncSignaturePass>());
}


}  // namespace shlo_pipeline
