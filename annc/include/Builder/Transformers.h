#ifndef ANNC_TRANSFORMERS_H
#define ANNC_TRANSFORMERS_H

#include "Builder/OpSpec.h"

namespace annc {

// Shared helper: decodes an integer Const node's raw bytes into int64 values
// (handles int32/int64 dtypes). Returns false if the node is missing, not an
// integer constant, or has no data.
bool decodeIntConstValues(const NodeInfo* cnode, std::vector<int64_t>& out);

// Reads an optional int64-typed TF attribute, defaulting to `def`.
int64_t getI64AttrOr(const NodeInfo& node, llvm::StringRef name, int64_t def);

// Builds a private atir.constant carrying integer cache data, used for
// shape/axis/segment-count fallbacks. `isI64` selects a 1-D int64 tensor
// (shape/permutation) versus a 0-D int32 scalar (axis/num_segments).
mlir::Value buildPrivateIntConst(OpContext& ctx, llvm::StringRef name,
                                 llvm::ArrayRef<int64_t> values, bool isI64);

// Builds a private 0-D f32 atir.constant (e.g. a PadV1 zero pad value).
mlir::Value buildPrivateF32Const(OpContext& ctx, llvm::StringRef name,
                                 float value);

// Non-1:1 op transformers. Each builds exactly the ATIR op for its TF op(s),
// using the typed ODS builders, and binds all produced values via the context.
// Every failure path reports a diagnostic and returns failure().

// Matrix / shape semantics.
mlir::LogicalResult transformMatMul(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                    llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformBatchMatMul(const NodeInfo&,
                                         llvm::ArrayRef<mlir::Type>,
                                         llvm::ArrayRef<mlir::Value>,
                                         OpContext&);
mlir::LogicalResult transformReshape(const NodeInfo&,
                                     llvm::ArrayRef<mlir::Type>,
                                     llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformSqueeze(const NodeInfo&,
                                     llvm::ArrayRef<mlir::Type>,
                                     llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformTranspose(const NodeInfo&,
                                       llvm::ArrayRef<mlir::Type>,
                                       llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformExpandDims(const NodeInfo&,
                                        llvm::ArrayRef<mlir::Type>,
                                        llvm::ArrayRef<mlir::Value>,
                                        OpContext&);
mlir::LogicalResult transformBroadcast(const NodeInfo&,
                                       llvm::ArrayRef<mlir::Type>,
                                       llvm::ArrayRef<mlir::Value>, OpContext&);

// Index / slicing semantics.
mlir::LogicalResult transformConcatV2(const NodeInfo&,
                                      llvm::ArrayRef<mlir::Type>,
                                      llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformPack(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                  llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformGather(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                    llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformStridedSlice(const NodeInfo&,
                                          llvm::ArrayRef<mlir::Type>,
                                          llvm::ArrayRef<mlir::Value>,
                                          OpContext&);
mlir::LogicalResult transformSplit(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                   llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformPad(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                 llvm::ArrayRef<mlir::Value>, OpContext&);

// Sparse / segment semantics.
mlir::LogicalResult transformSparseToDense(const NodeInfo&,
                                           llvm::ArrayRef<mlir::Type>,
                                           llvm::ArrayRef<mlir::Value>,
                                           OpContext&);
mlir::LogicalResult transformSparseTensorDenseMatMul(const NodeInfo&,
                                                     llvm::ArrayRef<mlir::Type>,
                                                     llvm::ArrayRef<mlir::Value>,
                                                     OpContext&);
mlir::LogicalResult transformSparseReshape(const NodeInfo&,
                                           llvm::ArrayRef<mlir::Type>,
                                           llvm::ArrayRef<mlir::Value>,
                                           OpContext&);
mlir::LogicalResult transformSparseFillEmptyRows(const NodeInfo&,
                                                 llvm::ArrayRef<mlir::Type>,
                                                 llvm::ArrayRef<mlir::Value>,
                                                 OpContext&);
mlir::LogicalResult transformSparseSegmentSum(const NodeInfo&,
                                              llvm::ArrayRef<mlir::Type>,
                                              llvm::ArrayRef<mlir::Value>,
                                              OpContext&);
mlir::LogicalResult transformSparseSegmentMin(const NodeInfo&,
                                              llvm::ArrayRef<mlir::Type>,
                                              llvm::ArrayRef<mlir::Value>,
                                              OpContext&);
mlir::LogicalResult transformSparseSegmentMean(const NodeInfo&,
                                               llvm::ArrayRef<mlir::Type>,
                                               llvm::ArrayRef<mlir::Value>,
                                               OpContext&);

// Reduction / control-flow / misc semantics.
mlir::LogicalResult transformSum(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                 llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformProd(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                  llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformMerge(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                   llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformDynamicPartition(const NodeInfo&,
                                              llvm::ArrayRef<mlir::Type>,
                                              llvm::ArrayRef<mlir::Value>,
                                              OpContext&);
mlir::LogicalResult transformParallelDynamicStitch(const NodeInfo&,
                                                   llvm::ArrayRef<mlir::Type>,
                                                   llvm::ArrayRef<mlir::Value>,
                                                   OpContext&);
mlir::LogicalResult transformResourceGather(const NodeInfo&,
                                            llvm::ArrayRef<mlir::Type>,
                                            llvm::ArrayRef<mlir::Value>,
                                            OpContext&);
mlir::LogicalResult transformStringToHashBucketFast(const NodeInfo&,
                                                    llvm::ArrayRef<mlir::Type>,
                                                    llvm::ArrayRef<mlir::Value>,
                                                    OpContext&);
mlir::LogicalResult transformStaticRegexReplace(const NodeInfo&,
                                                llvm::ArrayRef<mlir::Type>,
                                                llvm::ArrayRef<mlir::Value>,
                                                OpContext&);
mlir::LogicalResult transformStringSplit(const NodeInfo&,
                                         llvm::ArrayRef<mlir::Type>,
                                         llvm::ArrayRef<mlir::Value>,
                                         OpContext&);
mlir::LogicalResult transformUnique(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                    llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformTopK(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                  llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformSquare(const NodeInfo&, llvm::ArrayRef<mlir::Type>,
                                    llvm::ArrayRef<mlir::Value>, OpContext&);
mlir::LogicalResult transformSquaredDifference(const NodeInfo&,
                                               llvm::ArrayRef<mlir::Type>,
                                               llvm::ArrayRef<mlir::Value>,
                                               OpContext&);
mlir::LogicalResult transformVariable(const NodeInfo&,
                                      llvm::ArrayRef<mlir::Type>,
                                      llvm::ArrayRef<mlir::Value>, OpContext&);

}  // namespace annc

#endif  // ANNC_TRANSFORMERS_H
