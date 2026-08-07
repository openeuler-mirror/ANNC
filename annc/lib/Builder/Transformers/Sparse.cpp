// Transformers for sparse and segment semantics.

#include "Builder/MLIROpBuilder.h"
#include "Builder/Transformers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
namespace annc {

// Builds a fallback num_segments scalar when the graph does not provide one.
// Shared by the SparseSegment* family.
static FailureOr<Value> getNumSegments(const NodeInfo& node, OpContext& ctx,
                                       ArrayRef<Value> ins) {
  if (ins.size() >= 4 && ins[3]) return ins[3];
  return buildPrivateIntConst(ctx, node.name + "/num_segments", {0}, false);
}

// atir.SparseToDense: sparse_indices, output_shape, sparse_values,
// default_value.
LogicalResult transformSparseToDense(const NodeInfo& node, ArrayRef<Type> outs,
                                     ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 4) return ctx.emitError(node, "requires 4 inputs");
  Location loc = ctx.loc(node.name);
  bool validate = true;
  if (!node.getAttr("validate_indices", validate)) {
    if (int64_t v; node.getAttr("validate_indices", v)) validate = (v != 0);
  }
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::SparseToDenseOp>(
      loc, outs[0], outputBuffer.getResult(), ins[0], ins[1], ins[2], ins[3],
      b.getBoolAttr(validate));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.SparseTensorDenseMatMul: sparse A (indices/values/dense_shape) x dense B.
LogicalResult transformSparseTensorDenseMatMul(const NodeInfo& node,
                                               ArrayRef<Type> outs,
                                               ArrayRef<Value> ins,
                                               OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 4 || outs.empty())
    return ctx.emitError(node,
                         "requires indices, values, dense_shape and dense inputs");
  Location loc = ctx.loc(node.name);
  bool adjointA = false;
  bool adjointB = false;
  if (!node.getAttr("adjoint_a", adjointA)) {
    if (int64_t v; node.getAttr("adjoint_a", v)) adjointA = (v != 0);
  }
  if (!node.getAttr("adjoint_b", adjointB)) {
    if (int64_t v; node.getAttr("adjoint_b", v)) adjointB = (v != 0);
  }
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::SparseTensorDenseMatMulOp>(
      loc, outs[0], outputBuffer.getResult(), ins[0], ins[1], ins[2], ins[3],
      b.getBoolAttr(adjointA), b.getBoolAttr(adjointB));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.SparseReshape: produces (output_indices, output_shape).
LogicalResult transformSparseReshape(const NodeInfo& node, ArrayRef<Type> outs,
                                     ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 3)
    return ctx.emitError(node, "requires indices, input_shape and new_shape");
  Location loc = ctx.loc(node.name);
  auto op = b.create<atir::SparseReshapeOp>(loc, outs, ins[0], ins[1], ins[2]);
  if (node.outputs.size() >= 1)
    ctx.bindResult(node.outputs[0].name, op.getOutputIndices());
  if (node.outputs.size() >= 2)
    ctx.bindResult(node.outputs[1].name, op.getOutputShape());
  return success();
}

// atir.SparseFillEmptyRows: four outputs (indices, values, indicator, rev_map).
LogicalResult transformSparseFillEmptyRows(const NodeInfo& node,
                                           ArrayRef<Type> outs,
                                           ArrayRef<Value> ins,
                                           OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 4) return ctx.emitError(node, "requires 4 inputs");
  Location loc = ctx.loc(node.name);
  auto op = b.create<atir::SparseFillEmptyRowsOp>(loc, outs, ins[0], ins[1],
                                                  ins[2], ins[3]);
  if (node.outputs.size() >= 1)
    ctx.bindResult(node.outputs[0].name, op.getOutputIndices());
  if (node.outputs.size() >= 2)
    ctx.bindResult(node.outputs[1].name, op.getOutputValues());
  if (node.outputs.size() >= 3)
    ctx.bindResult(node.outputs[2].name, op.getEmptyRowIndicator());
  if (node.outputs.size() >= 4)
    ctx.bindResult(node.outputs[3].name, op.getReverseIndexMap());
  return success();
}

LogicalResult transformSparseSegmentSum(const NodeInfo& node,
                                        ArrayRef<Type> outs,
                                        ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 3)
    return ctx.emitError(node, "requires input, indices and segment_ids");
  Location loc = ctx.loc(node.name);
  auto numSegments = getNumSegments(node, ctx, ins);
  if (failed(numSegments)) return failure();
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op =
      b.create<atir::SparseSegmentSumOp>(loc, outs[0], outputBuffer.getResult(),
                                         ins[0], ins[1], ins[2], *numSegments);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

LogicalResult transformSparseSegmentMin(const NodeInfo& node,
                                        ArrayRef<Type> outs,
                                        ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 3)
    return ctx.emitError(node, "requires input, indices and segment_ids");
  Location loc = ctx.loc(node.name);
  auto numSegments = getNumSegments(node, ctx, ins);
  if (failed(numSegments)) return failure();
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op =
      b.create<atir::SparseSegmentMinOp>(loc, outs[0], outputBuffer.getResult(),
                                         ins[0], ins[1], ins[2], *numSegments);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

LogicalResult transformSparseSegmentMean(const NodeInfo& node,
                                         ArrayRef<Type> outs,
                                         ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 3)
    return ctx.emitError(node, "requires input, indices and segment_ids");
  Location loc = ctx.loc(node.name);
  auto numSegments = getNumSegments(node, ctx, ins);
  if (failed(numSegments)) return failure();
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::SparseSegmentMeanOp>(
      loc, outs[0], outputBuffer.getResult(), ins[0], ins[1], ins[2],
      *numSegments);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.ResourceGather: the resource handle plus a list of indices.
LogicalResult transformResourceGather(const NodeInfo& node, ArrayRef<Type> outs,
                                      ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2 || outs.empty())
    return ctx.emitError(node, "requires resource and indices");
  Location loc = ctx.loc(node.name);
  Value resource = ins[0];
  SmallVector<Value, 4> indices(ins.begin() + 1, ins.end());
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::ResourceGatherOp>(
      loc, outs[0], outputBuffer.getResult(), resource, indices,
      b.getBoolAttr(false), b.getI64IntegerAttr(0));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

}  // namespace annc
