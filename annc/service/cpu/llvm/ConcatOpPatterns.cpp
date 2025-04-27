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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "concat-patterns"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
// Left here intentionally to make review easier in case someone wants to see
// the debug prints, just uncomment it. I'll remove this later.
// #define LLVM_DEBUG(x) (x)

namespace mlir {
namespace tensor {
#define GEN_PASS_DEF_DECOMPOSETENSORCONCAT
#define GEN_PASS_DEF_CONCATREMOVAL
#define GEN_PASS_DEF_SIMPLIFYTENSORCONCAT
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

struct SimplifyTensorConcatOp : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  /// Struct to store the merge opportunities. Each opportunity is represented
  /// with 3 values. For example, given the following IR:
  ///
  /// %arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>
  /// %0 = tensor.extract_slice %arg0[0, 0][8, 4][1, 1]: <8x8xf32> to <8x4xf32>
  /// %1 = tensor.extract_slice %arg0[0, 4][8, 4][1, 1]: <8x8xf32> to <8x4xf32>
  /// %2 = tensor.extract_slice %arg1[0, 0][8, 4][1, 1]: <8x8xf32> to <8x4xf32>
  /// %3 = tensor.extract_slice %arg1[0, 4][8, 4][1, 1]: <8x8xf32> to <8x4xf32>
  /// %res = tensor.concat dim(1) %0, %1, %2, %3 :
  ///   (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) ->
  ///    tensor<8x16xf32>
  ///
  /// The tensor.concat inputs could be reduced from 4 to 2 like this:
  ///
  /// %res = tensor.concat dim(1) %arg0, %arg1 : (...)
  ///
  /// since both (%0, %1) and (%2, %3) could be merged in a single op, e.g., for
  /// %0, %1 (which would be canonicalized easily afterwards)::
  ///
  /// %0 = tensor.extract_slice %arg0[0, 0][8, 8][1, 1]: <8x8xf32> to <8x8xf32>
  ///
  /// This merge would be represented in the struct as follows:
  /// - src:  Contains ExtractSlice ops source operands. In this example it
  ///         would store the value of %arg0.
  /// - dst:  Contains a vector of ExtractSlice ops that can be merged together
  ///         and which read from the source operand contained in src. In this
  ///         example it would store the  values of %0 and %1.
  /// - dims: Contains the dimension where the merge can be performed. In this
  ///         example the merge can happen in the 2nd dimension, so it would
  ///         store a 1.
  ///
  /// A more complex merge may happen when the merge dimension (which can be
  /// identified by looking at the offsets of the extract_slice ops and focusing
  /// on the dimension which value changes) is different to the concat dimension
  /// (i.e., the dim operand). In the example above this would happen if the
  /// concat dimension is 0 instead of 1. In such case, the merge dimension
  /// would be 1, but the concat dimension would be 0. These cases are also
  /// supported, but requires reshape operations between the merged
  /// extract_slice and the concat op.
  struct MergeOpportunities {
    SmallVector<Value> src;
    SmallVector<SmallVector<tensor::ExtractSliceOp>> dst;
    llvm::SmallVector<int64_t> dims;
  };

  /// Checks if, for dimension `dim`, data extracted from `ext1` and `ext2` is
  /// contiguous.
  static bool contiguousExtractSlicesInternal(tensor::ExtractSliceOp ext1,
                                              tensor::ExtractSliceOp ext2,
                                              int dim) {
    // Only extract ops with unit stride can be contiguous.
    for (int64_t stride : ext1.getStaticStrides())
      if (stride != 1) {
        LLVM_DEBUG(DBGS() << "contiguousExtractSlices: Non unit strides\n");
        return false;
      }
    for (int64_t stride : ext2.getStaticStrides())
      if (stride != 1) {
        LLVM_DEBUG(DBGS() << "contiguousExtractSlices: Non unit strides\n");
        return false;
      }

    // We assume the output of ext2 goes after the output of ext1, so compare
    // the sum of the offset and output shape of ext1 against the offset of ext2
    return (ext1.getStaticOffsets()[dim] + ext1.getStaticSizes()[dim]) ==
           ext2.getStaticOffsets()[dim];
  }

  /// Given two input extract slice ops, it returns a boolean value indicating
  /// wether it can be determined that their outputs are contiguous in memory.
  ///
  /// For example, these are contiguous:
  /// %0 = tensor.extract_slice %arg0[0][4][1]: tensor<8xf32> to tensor<4xf32>
  /// %1 = tensor.extract_slice %arg0[4][4][1]: tensor<8xf32> to tensor<4xf32>
  ///
  /// whereas these are not:
  /// %0 = tensor.extract_slice %arg0[0][4][1]: tensor<8xf32> to tensor<4xf32>
  /// %1 = tensor.extract_slice %arg0[2][4][1]: tensor<8xf32> to tensor<4xf32>
  static FailureOr<llvm::SmallVector<bool>>
  contiguousExtractSlices(tensor::ExtractSliceOp ext1,
                          tensor::ExtractSliceOp ext2) {
    // Both must extract from the same source to be contiguous
    if (ext1.getSource() != ext2.getSource()) {
      LLVM_DEBUG(
          DBGS() << "contiguousExtractSlices: extract source are different\n");
      return failure();
    }

    // Only extract ops with static sizes and offsets can be statically be
    // determined to be contiguous.
    if (ShapedType::isDynamicShape(ext1.getStaticSizes()) ||
        ShapedType::isDynamicShape(ext1.getStaticOffsets()) ||
        ShapedType::isDynamicShape(ext1.getStaticStrides())) {
      LLVM_DEBUG(DBGS() << "contiguousExtractSlices: Shape is dynamic\n");
      return failure();
    }
    if (ShapedType::isDynamicShape(ext2.getStaticSizes()) ||
        ShapedType::isDynamicShape(ext2.getStaticOffsets()) ||
        ShapedType::isDynamicShape(ext2.getStaticStrides())) {
      LLVM_DEBUG(DBGS() << "contiguousExtractSlices: Shape is dynamic\n");
      return failure();
    }

    // Only extract ops with ranked tensor types can be determined to be
    // contiguous.
    if (!isa<RankedTensorType>(ext1.getSource().getType())) {
      LLVM_DEBUG(DBGS() << "input type must be ranked tensor");
      return failure();
    }
    if (!isa<RankedTensorType>(ext2.getSource().getType())) {
      LLVM_DEBUG(DBGS() << "input type must be ranked tensor");
      return failure();
    }

    // Check that both extract ops have the same source operand rank
    auto srcType1 = cast<RankedTensorType>(ext1.getSource().getType());
    auto srcType2 = cast<RankedTensorType>(ext2.getSource().getType());
    int64_t srcRank1 = srcType1.getRank();
    int64_t srcRank2 = srcType2.getRank();

    if (srcRank1 != srcRank2) {
      LLVM_DEBUG(
          DBGS()
          << "ExtractSlice ops with different source type cannot be merged\n");
      return failure();
    }

    llvm::SmallVector<bool> ret;

    for (int dim = 0; dim < srcRank1; dim++)
      ret.push_back(contiguousExtractSlicesInternal(ext1, ext2, dim));

    return ret;
  }

  /// Returns the ExtractSliceOp that defines the value or failure if the value
  /// is not defined by a ExtractSliceOp.
  static FailureOr<tensor::ExtractSliceOp> getExtractSliceOp(Value val) {
    Operation *op = val.getDefiningOp();
    if (op == nullptr)
      return failure();
    tensor::ExtractSliceOp extractOp = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!extractOp)
      return failure();
    return extractOp;
  }

  static void
  addConcatVals(llvm::SmallVector<tensor::ExtractSliceOp> &concatVals,
                Value srcVal, int dim, MergeOpportunities &m) {
    // Only consider merge if there are at least two values to merge.
    if (concatVals.size() > 1) {
      m.src.push_back(srcVal);
      m.dst.push_back(concatVals);
      m.dims.push_back(dim);
    }
    concatVals.clear();
  }

  /// Analyze the IR, compute the merge opportunities, and store them in the
  /// MergeOpportunities struct.
  static void computeMergeOpportunities(ConcatOp concatOp,
                                        MergeOpportunities &merge) {
    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<llvm::SmallVector<int32_t>> indexes;
    auto concatInputs = concatOp.getInputs();
    uint32_t highest_idx = 0;

    // 1. Analyze merge opportunities based on the order of the operands of the
    // concat op. Only consecutive extract slices with the same source are valid
    // candidates for merging.
    for (Value val : concatInputs) {
      FailureOr<tensor::ExtractSliceOp> maybeExtractOp = getExtractSliceOp(val);

      if (!failed(maybeExtractOp)) {
        tensor::ExtractSliceOp extractOp = *maybeExtractOp;
        int32_t idx = highest_idx;
        Value extractSrc = extractOp.getSource();

        // Search for the value in the vector of inputs. If its found, compute
        // its index in the vector. Otherwise, insert it into the vector.
        auto it = std::find(inputs.begin(), inputs.end(), extractSrc);
        if (it != inputs.end()) {
          idx = it - inputs.begin();
        } else {
          highest_idx++;
          inputs.push_back(extractSrc);
        }

        if (!indexes.empty() && indexes.back().back() == idx)
          indexes.back().push_back(idx);
        else
          indexes.push_back({idx});
      } else {
        // The current input operand of concat is not an extract slice, insert
        // -1 to signify it cannot be merged with others extract slice ops.
        indexes.push_back({-1});
      }
    }

    /// 2. Filter previously computed opportunities by contiguous data. Only
    /// contiguous extract slices are valid candidates for merging. Also skips
    /// inputs which is index is -1 (which signify that the operand cannot be
    /// merged with others).
    uint32_t concatInputIdx = 0;
    for (auto vec : indexes) {
      int32_t idx = vec[0];

      if (idx == -1) {
        concatInputIdx++;
        continue;
      }

      Value srcVal = inputs[idx];
      llvm::SmallVector<tensor::ExtractSliceOp> concatVals;
      int contiguousDim = -1;

      for (size_t i = 0; i < vec.size(); i++) {
        FailureOr<tensor::ExtractSliceOp> maybeExtractOp =
            getExtractSliceOp(concatInputs[concatInputIdx]);
        assert(!failed(maybeExtractOp) && "expected to get a extract slice op");
        tensor::ExtractSliceOp extractOp = *maybeExtractOp;

        if (concatVals.empty()) {
          concatVals.push_back(extractOp);
        } else {
          FailureOr<llvm::SmallVector<bool>> maybeContiguousVec =
              contiguousExtractSlices(concatVals.back(), extractOp);
          llvm::SmallVector<bool> contiguousVec = *maybeContiguousVec;

          // Merge is only valid if consecutive extract slice ops are contiguous
          if (!failed(maybeContiguousVec) &&
              llvm::any_of(contiguousVec, [&](bool c) { return c; })) {

            LLVM_DEBUG(DBGS() << "Found contiguous extract ops:\n > ");
            LLVM_DEBUG(concatVals.back().dump());
            LLVM_DEBUG(llvm::dbgs() << "and\n > ");
            LLVM_DEBUG(extractOp.dump());
            LLVM_DEBUG(llvm::dbgs() << "\n");
            concatVals.push_back(extractOp);

            // TODO: Check the case where multiple dims can be merged together?
            // Is that legal?
            for (size_t dim = 0; dim < contiguousVec.size(); dim++)
              if (contiguousVec[dim])
                contiguousDim = dim;
          } else {
            // If extract are not contiguous, reset the vector of contiguous
            // values. Before that we also have to add the contiguous values
            // to the main vector if neccesary
            addConcatVals(concatVals, srcVal, contiguousDim, merge);

            // Only push the extractOp if is not the last element of the list
            if (i < vec.size() - 1)
              concatVals.push_back(extractOp);
          }
        }

        concatInputIdx++;
      }

      addConcatVals(concatVals, srcVal, contiguousDim, merge);
    }
  }

  /// Computes the reassociation indices for the reduce and expand shape ops.
  /// The pivot represents the dimension where the reassociacion is divided.
  /// The direction can be either 1 or -1. The former indicates that the
  /// reassociation must happen from left to right and the latter that it
  /// must happen from right to left.
  static void computeReassocIndices(Type dtype, SmallVector<int64_t> shp,
                                    int dim, int mergeDim,
                                    SmallVector<ReassociationIndices> &reassoc,
                                    RankedTensorType &outTy) {
    int pivot = mergeDim;
    int direction = dim > mergeDim ? -1 : 1;
    SmallVector<int64_t> outShape;

    for (int i = 0; i < (int)shp.size(); i++) {
      if (i != pivot) {
        outShape.push_back(shp[i]);

        if (i + direction == pivot) {
          if (direction == 1)
            reassoc.push_back({i, i + 1});
          else
            reassoc.push_back({i - 1, i});
        } else {
          reassoc.push_back({i});
        }
      }
    }

    outTy = RankedTensorType::get(outShape, dtype);
  }

  /// Given the triple of merge opportunities and indices, rewrite the IR and
  /// merge the insert_slice ops.
  static LogicalResult rewriteConcatSlices(ConcatOp concatOp,
                                           PatternRewriter &rewriter,
                                           MergeOpportunities m) {
    // If there is nothing to merge, return early.
    if (m.src.empty()) {
      return failure();
    }

    int64_t dim = concatOp.getDim();
    bool merged = false;

    for (auto [src, dst, mergeDim] : llvm::zip_equal(m.src, m.dst, m.dims)) {
      tensor::ExtractSliceOp sliceOp = dst[0];
      // Compute the offset, sizes and strides for the merged ExtractSliceOp.
      // Use the first sliceOp as reference for both offsets and strides, then
      // compute the output shape by reducing the sizes of all slice ops present
      // in the merge.
      ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
      ArrayRef<int64_t> oldSizes = sliceOp.getStaticSizes();
      ArrayRef<int64_t> strides = sliceOp.getStaticStrides();
      SmallVector<int64_t> outputShape;

      for (int64_t i = 0; i < (int64_t)oldSizes.size(); i++) {
        if (i == mergeDim) {
          int s = 0;
          for (auto oldSlice : dst)
            s += oldSlice.getStaticSizes()[i];
          outputShape.push_back(s);
        } else {
          outputShape.push_back(oldSizes[i]);
        }
      }
      RankedTensorType newSliceType =
          mlir::RankedTensorType::get(outputShape, rewriter.getF32Type());

      // Create the new slice op which merges other slice ops.
      auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
          sliceOp.getLoc(), newSliceType, src, ValueRange({}), ValueRange({}),
          ValueRange({}), rewriter.getDenseI64ArrayAttr(offsets),
          rewriter.getDenseI64ArrayAttr(outputShape),
          rewriter.getDenseI64ArrayAttr(strides));

      // Create a vector with the new values.
      SmallVector<Value> newConcatOperands;
      SmallVector<Value> dstVals;
      for (auto op : dst)
        dstVals.push_back(op->getResult(0));

      Value mergedOp;
      if (dim == mergeDim) {
        mergedOp = newSliceOp;
      } else {
        // The shape of the new extract slice op is incompatible with what the
        // concat expects, so we need to reshape the output tensor. For
        // efficiency reasons we'll use a combination of collapse/reshape rather
        // than a tensor reshape.

        // First of all, compute the expected shape of the merged tensors.
        SmallVector<int64_t> mergedShape(
            sliceOp.getResult().getType().getShape());
        for (auto op : dst)
          if (op != sliceOp)
            mergedShape[dim] += op.getResult().getType().getShape()[dim];

        // TODO: This is a bit of a hack...any better way to do this e.g.,
        // without a loop?
        SmallVector<APInt> mergedShapeAPInt;
        for (int64_t val : mergedShape)
          mergedShapeAPInt.push_back(APInt(/*width=*/32, val));

        // Now reshape the new slice output into the expected output of the
        // merged tensors. Use CollapseShapeOp/ExpandShapeOp to perform the
        // reshape.
        Type dtype =
            cast<TensorType>(sliceOp.getResult().getType()).getElementType();
        SmallVector<ReassociationIndices> reassocIndices;
        RankedTensorType collapsedOutputTy;
        computeReassocIndices(dtype, mergedShape, dim, mergeDim, reassocIndices,
                              collapsedOutputTy);

        Value collapsed = rewriter.create<tensor::CollapseShapeOp>(
            sliceOp.getLoc(), collapsedOutputTy, newSliceOp, reassocIndices);

        RankedTensorType expandOutputTy =
            RankedTensorType::get(mergedShape, dtype);
        mergedOp = rewriter.create<tensor::ExpandShapeOp>(
            sliceOp.getLoc(), expandOutputTy, collapsed, reassocIndices);
      }

      // Replace the old slice ops with the new one we just created.
      for (auto operand : concatOp->getOperands()) {
        // If the current operand is equal to the first element of the slices to
        // merge, then insert the new slice op instead. If, on the contrary, the
        // operand is not present in the slices to merge, we need to preserve it
        // in the operands list, so insert it.
        if (std::find(dstVals.begin(), dstVals.end(), operand) !=
            dstVals.end()) {
          if (operand == dstVals[0])
            newConcatOperands.push_back(mergedOp);
        } else
          newConcatOperands.push_back(operand);
      }
      concatOp->setOperands(newConcatOperands);
      merged = true;
    }

    return merged ? success() : failure();
  }

  static void debugMergeOpportunities(MergeOpportunities m) {
    if (m.src.empty())
      LLVM_DEBUG(DBGS() << "Merge opportunities: (empty)\n\n");
    else
      LLVM_DEBUG(DBGS() << "Merge opportunities:\n");

    for (auto [src, dst, dim] : llvm::zip_equal(m.src, m.dst, m.dims)) {
      LLVM_DEBUG(DBGS() << "[DIMENSION " << dim << "] ");
      LLVM_DEBUG(src.dump());
      for (auto dstIt : dst) {
        LLVM_DEBUG(DBGS() << "==> ");
        LLVM_DEBUG(dstIt.dump());
      }
      LLVM_DEBUG(llvm::dbgs() << "\n\n");
    }
  }

  LogicalResult matchAndRewrite(ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    MergeOpportunities merge;

    // 1. Analyze the IR and search for merging opportunities.
    computeMergeOpportunities(concatOp, merge);

    // 2. (If LLVM_DEBUG is enabled) debug merge opportunities
    debugMergeOpportunities(merge);

    // 3. Perform the merging
    return rewriteConcatSlices(concatOp, rewriter, merge);
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

void mlir::tensor::populateSimplifyTensorConcatPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SimplifyTensorConcatOp>(patterns.getContext());
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

struct SimplifyTensorConcatPass final
    : public tensor::impl::SimplifyTensorConcatBase<
          SimplifyTensorConcatPass> {
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

void SimplifyTensorConcatPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tensor::populateSimplifyTensorConcatPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> tensor::createDecomposeTensorConcatPass() {
  return std::make_unique<DecomposeTensorConcatPass>();
}

std::unique_ptr<Pass> tensor::createConcatRemovalPass() {
  return std::make_unique<ConcatRemovalPass>();
}

std::unique_ptr<Pass> tensor::createSimplifyTensorConcatPass() {
  return std::make_unique<SimplifyTensorConcatPass>();
}
