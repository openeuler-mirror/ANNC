#include "common/pb2_common.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace pb2 {
namespace {

using llvm::StringRef;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::RankedTensorType;
using mlir::Value;

bool isEncodeStringCall(Operation* op) {
  auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
  return (op->getName().getStringRef() == "tf.residue" ||
          op->getName().getStringRef() == "annc.residue") && ct &&
         ct.getValue() == kEncodeString;
}

bool isDecodeStringCall(Operation* op) {
  auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
  return (op->getName().getStringRef() == "tf.residue" ||
          op->getName().getStringRef() == "annc.residue") && ct &&
         ct.getValue() == kDecodeString;
}

bool isGetDimCall(Operation* op) {
  auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
  return (op->getName().getStringRef() == "tf.residue" ||
          op->getName().getStringRef() == "annc.residue") && ct &&
         ct.getValue() == kGetDim;
}

bool isSetDimCall(Operation* op) {
  auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
  return (op->getName().getStringRef() == "tf.residue" ||
          op->getName().getStringRef() == "annc.residue") && ct &&
         ct.getValue() == kSetDim;
}

bool isTfFormCarrier(Operation* op) {
  llvm::StringRef nm = op->getName().getStringRef();
  if (nm == "tf.residue" || nm == "annc.residue") {
    return true;
  }
  if (nm.starts_with("tf.")) {
    return true;
  }
  if (nm == "stablehlo.custom_call") {
    auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
    return ct && ct.getValue().starts_with("tf.");
  }
  return false;
}

struct EncodeStringUnwrapPass
    : public mlir::PassWrapper<EncodeStringUnwrapPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EncodeStringUnwrapPass)

  llvm::StringRef getArgument() const override { return "string-unwrap"; }
  llvm::StringRef getDescription() const override {
    return "roll back string sandwiches whose op2 failed to legalize: restore "
           "original string operands/results and drop encode/decode boundaries";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<Operation*, 8> decodes;
    module.walk([&](Operation* op) {
      if (isDecodeStringCall(op)) {
        decodes.push_back(op);
      }
    });

    int kept = 0, rolledBack = 0, unrankRolled = 0;
    for (Operation* dec : decodes) {
      if (dec->getBlock() == nullptr) {
        continue;
      }
      Value packedOut = dec->getOperand(0);
      Operation* op2 = packedOut.getDefiningOp();
      if (op2 == nullptr || !isTfFormCarrier(op2)) {
        ++kept;
        continue;
      }
      if (failed(rollBack(dec, op2, packedOut))) {
        continue;
      }
      ++rolledBack;
    }

    llvm::SmallVector<Operation*, 8> setDims;
    module.walk([&](Operation* op) {
      if (isSetDimCall(op)) {
        setDims.push_back(op);
      }
    });
    for (Operation* sd : setDims) {
      if (sd->getBlock() == nullptr) {
        continue;
      }
      Operation* up = sd->getOperand(0).getDefiningOp();
      if (up == nullptr) {
        continue;
      }
      const bool viaDecode = isDecodeStringCall(up);
      Operation* op2 = viaDecode ? up->getOperand(0).getDefiningOp() : up;
      if (op2 == nullptr || !isTfFormCarrier(op2)) {
        continue;
      }
      llvm::SmallVector<Operation*, 8> bounds;
      llvm::DenseMap<Value, Value> toOrig;
      llvm::DenseSet<Value> visited;
      for (Value o : op2->getOperands()) {
        collectLineage(o, bounds, toOrig, visited, 1);
      }
      if (bounds.empty()) {
        continue;
      }
      for (unsigned i = 0; i < op2->getNumOperands(); ++i) {
        auto it = toOrig.find(op2->getOperand(i));
        if (it != toOrig.end()) {
          op2->setOperand(i, it->second);
        }
      }
      op2->getResult(0).setType(sd->getResult(0).getType());
      sd->getResult(0).replaceAllUsesWith(op2->getResult(0));
      sd->erase();
      if (viaDecode) {
        up->erase();
      }
      for (Operation* bnd : bounds) {
        if (bnd->use_empty()) {
          bnd->erase();
        }
      }
      ++unrankRolled;
    }
    if (kept + rolledBack + unrankRolled > 0)
      llvm::errs() << "[string-unwrap] kept=" << kept
                   << " rolledBack=" << rolledBack
                   << " unrankRolled=" << unrankRolled << "\n";
  }

 private:
  void collectLineage(Value packed, llvm::SmallVectorImpl<Operation*>& bounds,
                      llvm::DenseMap<Value, Value>& toOrig,
                      llvm::DenseSet<Value>& visited, int depth = 0) {
    if (depth > 64 || visited.count(packed)) {
      return;
    }
    visited.insert(packed);
    Operation* def = packed.getDefiningOp();
    if (def == nullptr) {
      return;
    }
    if (isEncodeStringCall(def) || isGetDimCall(def)) {
      bounds.push_back(def);
      Value src = def->getOperand(0);
      Operation* srcDef = src.getDefiningOp();
      while (srcDef && (isEncodeStringCall(srcDef) || isGetDimCall(srcDef))) {
        bounds.push_back(srcDef);
        src = srcDef->getOperand(0);
        srcDef = src.getDefiningOp();
      }
      toOrig[packed] = src;
      return;
    }
    if (isDecodeStringCall(def) || isSetDimCall(def)) {
      return;
    }
    if (isTfFormCarrier(def)) {
      return;
    }
    if (toOrig.count(packed)) {
      return;
    }
    for (Value o : def->getOperands()) {
      collectLineage(o, bounds, toOrig, visited, depth + 1);
    }
    for (Value o : def->getOperands()) {
      auto it = toOrig.find(o);
      if (it != toOrig.end()) {
        toOrig[packed] = it->second;
        break;
      }
    }
  }

  mlir::LogicalResult rollBack(Operation* dec, Operation* op2,
                               Value packedOut) {
    llvm::SmallVector<Operation*, 8> bounds;
    llvm::DenseMap<Value, Value> toOrig;
    llvm::DenseSet<Value> visited;
    for (Value o : op2->getOperands()) {
      collectLineage(o, bounds, toOrig, visited, 1);
    }
    if (bounds.empty()) {
      return mlir::failure();
    }

    for (unsigned i = 0; i < op2->getNumOperands(); ++i) {
      auto it = toOrig.find(op2->getOperand(i));
      if (it != toOrig.end()) {
        op2->setOperand(i, it->second);
      }
    }

    op2->getResult(0).setType(dec->getResult(0).getType());

    dec->getResult(0).replaceAllUsesWith(op2->getResult(0));

    dec->erase();
    llvm::SmallVector<Operation*, 8> maybeDead;
    for (Operation* enc : bounds) {
      if (!enc->use_empty()) {
        continue;
      }
      Operation* in = enc->getOperand(0).getDefiningOp();
      enc->erase();
      if (in && isGetDimCall(in) && in->use_empty()) {
        maybeDead.push_back(in);
      }
    }
    for (Operation* d : maybeDead) {
      if (d->use_empty()) {
        d->erase();
      }
    }
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createStringUnwrapPass() {
  return std::make_unique<EncodeStringUnwrapPass>();
}

}  // namespace pb2
