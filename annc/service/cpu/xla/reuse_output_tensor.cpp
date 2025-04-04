// xla/mlir_hlo/gml_st/transforms/cpu_tiling/reuse_output_tensor.cc
#include <memory>
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace gml_st {
namespace {

constexpr llvm::StringRef kElementwiseLabel = "__elementwise_label__";  

#define GEN_PASS_DEF_REUSEOUTPUTTENSORPASS
#include "gml_st/transforms/passes.h.inc"

struct ReuseOutputTensorPass
    : public impl::ReuseOutputTensorPassBase<ReuseOutputTensorPass> {

  bool hasLabel(Operation *op, StringRef name) { return op->hasAttr(name); }

  // Check if the input of the fusion op is only read before the fusion op
  // is executed. If it is read after the fusion op is executed, or if it is
  // written to before the fusion op is executed, then it is not safe to use
  // as output of the fusion.
  bool isSafeToReuse1(gml_st::FusionOp op, Value fusionInput) {
    auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    bool safeToReuse = true;

    for (auto &inputUse : fusionInput.getUses()) {
      Operation* inputUser = inputUse.getOwner();

      if (op == inputUser) {
        // This is the fusion op itself, skip this comparaison.
        safeToReuse = true;
      }
      else if (inputUser->isBeforeInBlock(op)) {
        // User is before the fusion op, check if this is a read or a write op.
        if (auto effect = dyn_cast<MemoryEffectOpInterface>(inputUser)) {
          safeToReuse = !effect.getEffectOnValue<MemoryEffects::Write>(inputUse.get()).has_value();
        }
        else if (auto fusionOpUser = dyn_cast<FusionOp>(inputUser)) {
          // FusionOp do not implement this interface, manually check if the use is an input or an output.
          for (auto userOut : dstStyleOp.getDpsInits())
            if (userOut == inputUse.get()) safeToReuse = false;
        }
        else {
          // We dont know if the user will write or not, so we assume it does (conservative choice).
          safeToReuse = false;
        }
      }
      else {
        // User is after the fusion op so its not safe to reuse
        safeToReuse = false;
      }

      if (!safeToReuse) break;
    }

    return safeToReuse;
  }

  // If there are fusion ops that read from a SSA value that is being
  // considered as a reuse candidate (like in this example, where the
  // candidate is %o1):
  //
  // %0 = tensor.empty()
  // %o1 = gml.st_fusion(in=%1, inits=%2)
  // %o2 = gml.st_fusion(in=%o1, inits=%0)
  // %o3 = gml.st_fusion(in=%o1, inits=%0)
  //
  // The bufferization creates a new buffer for the output of %o2 and
  // %o3, because they should be stored in 2 separate buffers. However,
  // if we reuse the input buffer, like this:
  //
  // %o2 = gml.st_fusion(in=%o1, inits=%o1)
  // %o3 = gml.st_fusion(in=%o1, inits=%o1)
  //
  // then the bufferization does not reuse the buffer (not sure why),
  // so we must avoid reusing in that kind of situations.
  bool isSafeToReuse2(gml_st::FusionOp op, Value fusionInput) {
    auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    bool safeToReuse = true;

    for (auto &inputUse : fusionInput.getUses()) {
      Operation* inputUser = inputUse.getOwner();

      if (op != inputUser && !inputUser->isBeforeInBlock(op)) {
        if (auto fusionOpUser = dyn_cast<FusionOp>(inputUser)) {
          for (auto userOut : dstStyleOp.getDpsInputOperands())
            if (userOut->get() == fusionInput) safeToReuse = false;
        }
      }

      if (!safeToReuse) break;
    }
    return safeToReuse;
  }

  // Returns the index of the input tensor of the fusion op that is safe to reuse as
  // output tensor. The criteria to determine if its safe to do so is:
  // 1. The fusion op has only 1 output
  // 2. There is exactly one input tensor in the fusion op with the same shape as
  //    the output.
  // 3. The input tensor candidate for replacement is not a function argument with the
  //    bufferization.writable = false
  // 4. The input tensor candidate for replacement is not used after the fusion op, and
  //    in the case that is used before the fusion op, it is only read.
  int getTensorIndexToReuse(gml_st::FusionOp op) {
    auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op.getOperation());
    if (!dstStyleOp) return -1;

    auto outputs = dstStyleOp.getDpsInitsMutable();
    if (outputs.size() != 1) return -1;

    Type outputTy = outputs[0].get().getType();
    int index = -1;
    int i = 0;
    for (auto use : dstStyleOp.getDpsInputOperands()) {
      Type inputTy = use->get().getType();
      Value fusionInput = use->get();

      if (outputTy == inputTy) {
        bool bufferizationWritable = true;
        // Check if the input of the fusion op is a function argument with the
        // bufferization.writable = false attr, in which case we cannot reuse as
        // output buffer.
        mlir::func::FuncOp funcOp = op->getParentOfType<mlir::func::FuncOp>();
        BlockArgument bbArg = dyn_cast<BlockArgument>(use->get());
        if (bbArg) {
          if (BoolAttr writable = funcOp.getArgAttrOfType<BoolAttr>(
            bbArg.getArgNumber(), BufferizationDialect::kWritableAttrName)) {
              bufferizationWritable = writable.getValue();
          }
        }

        // We can use 2 different heuristics here to determine when it is safe
        // to reuse.
        bool safeToReuse = isSafeToReuse2(op, fusionInput);

        if (bufferizationWritable && safeToReuse)
          return i;
      }

      i++;
    }

    return index;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    Builder builder(context);    

    getOperation()->walk([&](gml_st::FusionOp fusionOp) {      
      if (hasLabel(fusionOp, kElementwiseLabel)) {
        Location loc = fusionOp->getLoc();
        int reuse_idx = getTensorIndexToReuse(fusionOp);

        if (reuse_idx != -1) {
          rewriter.setInsertionPoint(fusionOp.getOperation());

          // Generate new fusion op and steal body.
          SmallVector<Value> operands;
          for (unsigned int i = 0; i < fusionOp->getNumOperands() - 1; i++)
            operands.push_back(fusionOp->getOperand(i));
          operands.push_back(fusionOp->getOperand(reuse_idx));

          TypeRange funcResultTypes = fusionOp.getResultTypes();
          auto newFusionOp = rewriter.create<gml_st::FusionOp>(
            loc, funcResultTypes, operands, fusionOp->getAttrs());
          newFusionOp.getRegion().takeBody(fusionOp.getRegion());

          fusionOp.getOperation()->getResult(0).replaceAllUsesWith(newFusionOp->getResult(0));
          fusionOp.getOperation()->erase();
        }
      }
    });          
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createReuseOutputTensorPass() {
  return std::make_unique<gml_st::ReuseOutputTensorPass>();
}

}  // namespace gml_st
}  // namespace mlir
