#include "common/pb2_common.h"

// Unrank wrapping: shape-agnostic ops with unranked instances get an
// annc.getDim/setDim sequence-view sandwich so the official legalize
// handles them as ranked-dynamic. Unary/same-shape ops only: TF's
// broadcasting binaries would flatten mismatched shapes into wrong math,
// those stay residue. Runs before the carrier rename (tf.residue form).

#include <cstdint>
#include <set>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace pb2 {
namespace {

using llvm::StringRef;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;

bool inUnrankAllowlist(const std::string& cls) {
  static const std::set<std::string> kAllow = {
      "Identity",
      "StopGradient",
      "GuaranteeConst",
      "Abs",
      "Neg",
      "Exp",
      "Sigmoid",
      "Tanh",
      "Relu",
      "Relu6",
      "Log",
      "Sqrt",
      "Rsqrt",
      "Square",
      "Sign",
      "Round",
      "Ceil",
      "Floor",
      "Sin",
      "Cos",
      "Tan",
      "LogicalNot",
      "Cast",
      "ZerosLike",
  };
  return kAllow.count(cls) != 0;
}

class UnrankWrap {
 public:
  explicit UnrankWrap(OpBuilder& b) : b_(b) {}

  std::string rewrite(Operation* op, const std::string& cls);

  std::string rewriteShape(Operation* op);

  std::string rewritePack(Operation* op);

  int encoders() const { return encoders_; }

 private:
  mlir::Operation* emitGetDim(mlir::Location loc, Value v, Type flatTy) {
    auto shapeTy =
        RankedTensorType::get({mlir::ShapedType::kDynamic}, b_.getI64Type());
    OperationState st(loc,
                      mlir::OperationName("tf.residue", b_.getContext()));
    st.addOperands(v);
    st.addTypes(flatTy);
    st.addTypes(shapeTy);
    st.addAttribute("call_target_name", b_.getStringAttr(kGetDim));
    mlir::Operation* enc = b_.create(st);
    ++encoders_;
    return enc;
  }

  mlir::Operation* emitSetDim(mlir::Location loc, Value v,
                              mlir::Type elemTy) {
    return emitBoundary(loc, v, kSetDim, mlir::UnrankedTensorType::get(elemTy));
  }

  mlir::Operation* emitBoundary(mlir::Location loc, Value v, StringRef target,
                                Type result) {
    OperationState st(loc, mlir::OperationName("tf.residue", b_.getContext()));
    st.addOperands(v);
    st.addTypes(result);
    st.addAttribute("call_target_name", b_.getStringAttr(target));
    return b_.create(st);
  }

  OpBuilder& b_;
  int encoders_ = 0;
};

std::string UnrankWrap::rewrite(Operation* op, const std::string& cls) {
  if (cls == "Shape") {
    return rewriteShape(op);
  }
  if (cls == "Pack") {
    return rewritePack(op);
  }
  // Fill: no specialization -- its unranked shape is the RESULT (dims and
  // value are always ranked), and this pass dispatches on unranked
  // operands, so a Fill rewrite is unreachable by construction.

  if (op->getNumResults() != 1 || op->getNumOperands() == 0) {
    return "shape(numResults/Operands)";
  }

  bool anyUnranked = false;
  mlir::Type elemTy = nullptr;
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto tt = llvm::dyn_cast<mlir::TensorType>(op->getOperand(i).getType());
    if (!tt) {
      return "operand-not-tensor";
    }
    if (!tt.hasRank()) {
      anyUnranked = true;
    }
    if (elemTy == nullptr) {
      elemTy = tt.getElementType();
    } else if (tt.getElementType() != elemTy) {
      return "elem-mismatch";
    }
  }
  if (!anyUnranked) {
    return "all-ranked";
  }
  auto resTy = llvm::dyn_cast<mlir::TensorType>(op->getResult(0).getType());
  if (!resTy) {
    return "result-not-tensor";
  }

  b_.setInsertionPoint(op);
  auto flatResTy = RankedTensorType::get({mlir::ShapedType::kDynamic},
                                         resTy.getElementType());

  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    Value v = op->getOperand(i);
    auto tt = llvm::cast<mlir::TensorType>(v.getType());
    Type flatTy =
        tt.hasRank()
            ? RankedTensorType::get({tt.getNumElements()}, tt.getElementType())
            : RankedTensorType::get({mlir::ShapedType::kDynamic},
                                    tt.getElementType());
    mlir::Operation* enc = emitGetDim(op->getLoc(), v, flatTy);
    op->setOperand(i, enc->getResult(0));
  }

  op->getResult(0).setType(flatResTy);

  b_.setInsertionPointAfter(op);
  mlir::Operation* dec =
      emitSetDim(op->getLoc(), op->getResult(0), resTy.getElementType());
  op->getResult(0).replaceUsesWithIf(
      dec->getResult(0),
      [&](mlir::OpOperand& use) { return use.getOwner() != dec; });
  return "";
}

std::string UnrankWrap::rewriteShape(Operation* op) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
    return "shape(arity)";
  }
  if (!llvm::dyn_cast<mlir::TensorType>(op->getOperand(0).getType())) {
    return "operand-not-tensor";
  }
  auto resTy = llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());
  if (!resTy || !(resTy.getElementType().isSignlessInteger(32) ||
                  resTy.getElementType().isSignlessInteger(64)))
    return "shape-elem";

  b_.setInsertionPoint(op);
  mlir::Operation* enc = emitGetDim(
      op->getLoc(), op->getOperand(0),
      RankedTensorType::get({mlir::ShapedType::kDynamic}, b_.getI64Type()));
  Value out = enc->getResult(1);
  if (resTy.getElementType() != b_.getI64Type()) {
    OperationState st(
        mlir::UnknownLoc::get(b_.getContext()),
        mlir::OperationName("stablehlo.convert", b_.getContext()));
    st.addOperands(out);
    st.addTypes(RankedTensorType::get({mlir::ShapedType::kDynamic},
                                      resTy.getElementType()));
    out = b_.create(st)->getResult(0);
  }
  op->getResult(0).replaceAllUsesWith(out);
  op->erase();
  return "";
}

std::string UnrankWrap::rewritePack(Operation* op) {
  auto axis = op->getAttrOfType<mlir::IntegerAttr>("axis");
  if (axis && axis.getInt() != 0) {
    return "pack-axis";
  }
  if (op->getNumOperands() == 0) {
    return "shape(numOperands)";
  }
  mlir::Type elemTy = nullptr;
  for (Value v : op->getOperands()) {
    auto tt = llvm::dyn_cast<mlir::TensorType>(v.getType());
    if (!tt) {
      return "operand-not-tensor";
    }
    if (elemTy == nullptr) {
      elemTy = tt.getElementType();
    } else if (tt.getElementType() != elemTy) {
      return "elem-mismatch";
    }
  }
  if (!llvm::dyn_cast<mlir::TensorType>(op->getResult(0).getType())) {
    return "result-not-tensor";
  }

  b_.setInsertionPoint(op);
  llvm::SmallVector<Value, 4> flats;
  for (Value v : op->getOperands()) {
    auto tt = llvm::cast<mlir::TensorType>(v.getType());
    Type flatTy =
        tt.hasRank()
            ? RankedTensorType::get({tt.getNumElements()}, tt.getElementType())
            : RankedTensorType::get({mlir::ShapedType::kDynamic},
                                    tt.getElementType());
    flats.push_back(emitGetDim(op->getLoc(), v, flatTy)->getResult(0));
  }
  OperationState cst(
      mlir::UnknownLoc::get(b_.getContext()),
      mlir::OperationName("stablehlo.concatenate", b_.getContext()));
  cst.addOperands(flats);
  cst.addTypes(RankedTensorType::get({mlir::ShapedType::kDynamic}, elemTy));
  cst.addAttribute("dimension", b_.getI64IntegerAttr(0));
  Value concat = b_.create(cst)->getResult(0);

  b_.setInsertionPointAfter(op);
  mlir::Operation* dec = emitSetDim(op->getLoc(), concat, elemTy);
  op->getResult(0).replaceAllUsesWith(dec->getResult(0));
  op->erase();
  return "";
}

struct EncodeUnrankWrapPass
    : public mlir::PassWrapper<EncodeUnrankWrapPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EncodeUnrankWrapPass)

  llvm::StringRef getArgument() const override { return "unrank-wrap"; }
  llvm::StringRef getDescription() const override {
    return "wrap unranked shape-agnostic ops with annc.getDim/setDim sequence "
           "views so the official legalize handles them as ranked-dynamic";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    OpBuilder b(&getContext());
    UnrankWrap w(b);

    int wrapped = 0, skipped = 0, notInList = 0;
    std::map<std::string, int> skipWhy;
    for (int round = 0; round < 16; ++round) {
      llvm::SmallVector<std::pair<Operation*, std::string>, 8> targets;
      int roundNotInList = 0;
      module.walk([&](Operation* op) {
        llvm::StringRef nm = op->getName().getStringRef();
        std::string cls;
        if (nm == "tf.residue" || nm == "annc.residue") {
          auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
          if (!ct || !ct.getValue().starts_with("tf.")) {
            return;
          }
          cls = ct.getValue().drop_front(3).str();
        } else if (nm.starts_with("tf.")) {
          cls = nm.drop_front(3).str();
        } else {
          return;
        }
        bool anyUnranked = false;
        bool hasString = false;
        for (Type t : op->getOperandTypes()) {
          if (auto tt = llvm::dyn_cast<mlir::TensorType>(t)) {
            if (!tt.hasRank()) {
              anyUnranked = true;
            }
            if (llvm::isa<mlir::tf_type::StringType>(tt.getElementType())) {
              hasString = true;
            }
          }
        }
        for (Type t : op->getResultTypes()) {
          if (auto tt = llvm::dyn_cast<mlir::TensorType>(t);
              tt && llvm::isa<mlir::tf_type::StringType>(tt.getElementType())) {
            hasString = true;
          }
        }
        if (!anyUnranked || hasString) {
          return;
        }
        if (cls != "Shape" && cls != "Pack" && cls != "Fill" &&
            !inUnrankAllowlist(cls)) {
          ++roundNotInList;
          return;
        }
        targets.push_back({op, cls});
      });
      notInList = roundNotInList;
      if (targets.empty()) {
        break;
      }

      int roundWrapped = 0;
      for (auto& [op, cls] : targets) {
        std::string why = w.rewrite(op, cls);
        if (why.empty()) {
          ++roundWrapped;
          continue;
        }
        ++skipped;  // accumulated across rounds (comparable to `wrapped`)
        ++skipWhy[cls + ":" + why];
      }
      wrapped += roundWrapped;
      if (round == 15 && !targets.empty())
        llvm::errs() << "[unrank-wrap] hit round cap with "
                     << targets.size() << " targets left\n";
    }
    // Skip reasons stay visible: they are the debugging entry point for
    // template coverage decisions ("not attempted" vs "K dynamic skip").
    if (skipped || notInList) {
      llvm::errs() << "[unrank-wrap] wrapped=" << wrapped
                   << " skipped=" << skipped << " notInList=" << notInList
                   << "\n";
      for (const auto& kv : skipWhy)
        llvm::errs() << "[unrank-wrap-skip] " << kv.first << ": " << kv.second
                     << "\n";
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createUnrankWrapPass() {
  return std::make_unique<EncodeUnrankWrapPass>();
}

}  // namespace pb2
