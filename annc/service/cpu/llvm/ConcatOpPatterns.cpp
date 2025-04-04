//===- ConcatOpPatterns.cpp - Patterns related to tensor.concat lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tensor {
#define GEN_PASS_DEF_DECOMPOSETENSORCONCAT
#define GEN_PASS_DEF_CONCATREMOVAL
#include "mlir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace tensor
} // namespace mlir

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Decompose `tensor.concat` into `tensor.empty` and a chain of slice inserts.
///
/// %concat = tensor.concat dim(1) %0, %1 :
///         (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
///
/// Becomes
///
/// %empty = tensor.empty() : tensor<2x7xf32>
/// %insert0 = tensor.insert_slice %0 into %empty[0, 0][2, 3][1, 1]
/// %concat = tensor.insert_slice %1 into %insert0[0, 3][2, 4][1, 1]
struct DecomposeTensorConcatOp : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    Location loc = concatOp.getLoc();
    FailureOr<Value> dest =
        tensor::getOrCreateDestination(rewriter, loc, concatOp->getResult(0));
    if (failed(dest))
      return failure();

    auto empty = dest->getDefiningOp<tensor::EmptyOp>();
    if (!empty)
      return failure();

    int64_t dim = concatOp.getDim();
    Value dimValue = rewriter.createOrFold<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(dim));

    int64_t rank = concatOp.getResultType().getRank();
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));

    // Compute the partial sums for the slice offsets.
    AffineExpr sum = rewriter.getAffineDimExpr(0);
    SmallVector<AffineExpr> partialSums = {sum};
    SmallVector<OpFoldResult> offsetStrides = {rewriter.getIndexAttr(0)};
    for (auto [idx, input] :
         llvm::enumerate(concatOp.getInputs().drop_back())) {
      sum = sum + rewriter.getAffineDimExpr(idx + 1);
      partialSums.push_back(sum);
      offsetStrides.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, input, dimValue));
    }
    auto partialSumMap = AffineMap::get(concatOp.getInputs().size(), 0,
                                        partialSums, rewriter.getContext());
    SmallVector<OpFoldResult> dimOffsets =
        affine::makeComposedFoldedMultiResultAffineApply(
            rewriter, loc, partialSumMap, offsetStrides);

    // Construct the chain of insert_slice ops into the destination.
    Value result = *dest;
    for (auto [input, offset] :
         llvm::zip_equal(concatOp.getInputs(), dimOffsets)) {
      SmallVector<OpFoldResult> sizes =
          tensor::getMixedSizes(rewriter, loc, input);
      offsets[dim] = offset;
      result = rewriter.createOrFold<tensor::InsertSliceOp>(
          loc, input, result, offsets, sizes, strides);
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        concatOp, concatOp.getResultType(), result);
    return success();
  }
};

struct ConcatRemoval : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern<ConcatOp>::OpRewritePattern;
 
  // TODO: This is very verbose. This is important in the initial stages but once this
  // transformation is mature we should probably use VLOG for this.
  static FailureOr<std::pair<Operation*, RankedTensorType>>
  isOperandErasable(Value input) {
    Operation* op = input.getDefiningOp();
    if (!op) {
      llvm::errs() << "[concat-removal] concat input is not coming from an operation\n";
      return failure();
    }
 
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
    if (!dpsOp) {
      llvm::errs() << "[concat-removal] concat input is not DPS\n";
      return failure();
    }
 
    if (dpsOp.getNumDpsInits() != 1) {
      llvm::errs() << "[concat-removal] concat input DPS does not have 1 init argument\n";
      return failure();
    }
 
    auto emptyTensorOp = dyn_cast<tensor::EmptyOp>(dpsOp.getDpsInits()[0].getDefiningOp());
    if (!emptyTensorOp) {
      llvm::errs() << "[concat-removal] concat input does not write into an empty op\n";
      return failure();
    }
 
    RankedTensorType type = dyn_cast<RankedTensorType>(emptyTensorOp.getType());
    if (!type) {
      llvm::errs() << "[concat-removal] concat input writes into an unranked empty op\n";
      return failure();
    }
 
    return std::make_pair(op, type);
  }

  /// Given a `concatOp` as input, it fills in `erasableOperands` and `tensorShapes` vectors,
  /// representing the list of erasableOperands and the result type of each operation, respectively.
  ///
  /// A concat operand is considered erasable if both:
  /// 1. isOperandErasable returns true
  /// 2. It is consecutive (in the operand list) with another erasable operand.
  ///
  /// When a "gap" is detected (i.e., an erasable operand is followed by an unerasable operand),
  /// the function returns. In other words, it only captures the first sequence of consecutive
  /// erasable operands in `concatOp`.
  static void
  getOperandsToErase(SmallVector<Value> inputs, SmallVector<Operation*> &erasableOperands, SmallVector<bool> & erasable) {    
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      FailureOr<std::pair<Operation*, RankedTensorType>> maybeOp = isOperandErasable(input);
      if (!failed(maybeOp)) {
        // The input is erasable, add it to the list.   
        erasableOperands.push_back((*maybeOp).first);        
        erasable.push_back(true);
      } else {    
        erasableOperands.push_back(nullptr);        
        erasable.push_back(false);
      }
    }        
  }

  static int64_t
  computeConcatOffset(ConcatOp concatOp, Value operand) {
    auto operands = concatOp.getInputs();

    // 1. Get the index of the operand within the concat op.
    int32_t idx = -1;
    for (auto [i, input] : llvm::enumerate(operands)) {
      if (operand == input) {
        idx = i;
        break;
      }
    }
    assert(idx >= 0);

    // 2. Get the offset by accumulating all offsets before the operand.
    int64_t off = 0;
    for (int i = 0; i < idx; i++) {
      off += cast<RankedTensorType>(operands[i].getType()).getShape()[concatOp.getDim()];
    }
    return off;
  }
 
  LogicalResult matchAndRewrite(ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    Location loc = concatOp.getLoc();
    // This should never happen by construction
    assert(concatOp->getNumResults() == 1);
    assert(concatOp->getResultTypes().size() == 1);

    // TODO: Discard dynamic tensor inputs.
 
    // The concat result is used by more than one op; this case is not supported, skip.
    if (llvm::to_vector(concatOp->getResult(0).getUsers()).size() != 1) {
      llvm::errs() << "[concat-removal] concat result has more than one use\n";
      return failure();
    }

    SmallVector<Value> sortedInputs = concatOp.getInputs();
    std::sort(sortedInputs.begin(), sortedInputs.end(), [&](Value v1, Value v2) {
        Operation *op1 = v1.getDefiningOp();
        Operation *op2 = v2.getDefiningOp();
        if (!op1) return true;
        if (!op2) return false;

        return op1->isBeforeInBlock(op2);
      }
    );

    for (auto sss : sortedInputs) {
      llvm::errs() << "--> ";
      sss.dump();
    }
 
    // Get all operands to be erased.
    SmallVector<Operation*> erasableOperands;    
    SmallVector<bool> erasable;
    getOperandsToErase(sortedInputs, erasableOperands, erasable);
    assert(erasableOperands.size() == erasable.size());
 
    // Compute the size of the new tensor
    int64_t dim = concatOp.getDim();
    RankedTensorType concatResultTy = cast<RankedTensorType>(concatOp.getResultType());
    std::vector<int64_t> newShape = concatResultTy.getShape();
 
    // IR rewriting starts now, begining with the new tensor creation.
    rewriter.setInsertionPointToStart(concatOp->getBlock());
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(loc, newShape, concatResultTy.getElementType());
 
    // Replace the inits of every operation with an extract_slice of the new tensor and insert the
    // output of every operation back to the big tensor (insert_slice)
    SmallVector<Value> lastResult;
    lastResult.push_back(emptyTensor);       
 
    for (auto [idx, operand] : llvm::enumerate(sortedInputs)) {
      
      auto shape = cast<RankedTensorType>(operand.getType()).getShape();

      SmallVector<OpFoldResult> offsets;
      for (int i = 0; i < cast<RankedTensorType>(operand.getType()).getRank(); i++) {
        if (i == dim) {
          offsets.push_back(rewriter.getIndexAttr(computeConcatOffset(concatOp, operand)));
        }
        else
          offsets.push_back(rewriter.getIndexAttr(0));
      }
      SmallVector<OpFoldResult> sizes;
      for (auto s : shape)
        sizes.push_back(rewriter.getIndexAttr(s));
      SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));

      if (erasable[idx]) {
        Operation* op = erasableOperands[idx];
        // 1. Create the extract_slice.
        // rewriter.setInsertionPoint(op);
        auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(loc, lastResult.back(), offsets, sizes, strides);
  
        // 2. Use the extract_slice as the output operand of the op.
        auto dpsOp = cast<DestinationStyleOpInterface>(op);
        dpsOp.getDpsInitsMutable()[0].set(extractSliceOp);      
  
        // 3. Create the insert_slice.      
        rewriter.setInsertionPointAfter(op);
        auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(loc, dpsOp->getResult(0), lastResult.back(), offsets, sizes, strides);
        lastResult.push_back(insertSliceOp);
      }
      else {
        // 3. Create the insert_slice.      
        // lastResult contains either tensor.empty or tensor.insert_slice so we can use getDefiningOp safely.
        if (operand.getDefiningOp())
          rewriter.setInsertionPointAfter(operand.getDefiningOp());
        else
          rewriter.setInsertionPointAfter(lastResult.back().getDefiningOp());
        auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(loc, operand, lastResult.back(), offsets, sizes, strides);
        lastResult.push_back(insertSliceOp);
      }
    }


    rewriter.replaceOp(concatOp, lastResult.back());
    return success();
  }
};

} // namespace

void mlir::tensor::populateDecomposeTensorConcatPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DecomposeTensorConcatOp>(patterns.getContext());
}

void mlir::tensor::populateConcatRemovalPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConcatRemoval>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct DecomposeTensorConcatPass final
    : public tensor::impl::DecomposeTensorConcatBase<
          DecomposeTensorConcatPass> {
  void runOnOperation() override;
};

struct ConcatRemovalPass final
    : public tensor::impl::ConcatRemovalBase<
          ConcatRemovalPass> {
  void runOnOperation() override;
};

} // namespace

void DecomposeTensorConcatPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tensor::populateDecomposeTensorConcatPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void ConcatRemovalPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tensor::populateConcatRemovalPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> tensor::createDecomposeTensorConcatPass() {
  return std::make_unique<DecomposeTensorConcatPass>();
}

std::unique_ptr<Pass> tensor::createConcatRemovalPass() {
  return std::make_unique<ConcatRemovalPass>();
}
