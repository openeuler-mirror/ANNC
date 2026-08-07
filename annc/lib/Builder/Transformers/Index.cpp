// Transformers for indexing / slicing semantics.

#include "Builder/MLIROpBuilder.h"
#include "Builder/Transformers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
namespace annc {

// Reads an optional int64-typed TF attribute, defaulting to `def`. Shared by
// all transformers (declared in Transformers.h).
int64_t getI64AttrOr(const NodeInfo& node, StringRef name, int64_t def) {
  int64_t v = 0;
  return node.getAttr(name.str(), v) ? v : def;
}

// atir.ConcatV2: the last input is the axis tensor; the rest are values.
LogicalResult transformConcatV2(const NodeInfo& node, ArrayRef<Type> outs,
                                ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2)
    return ctx.emitError(node,
                         "requires at least two inputs (values and axis)");
  Location loc = ctx.loc(node.name);
  SmallVector<Value> values(ins.begin(), ins.end() - 1);
  Value axis = ins.back();
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::ConcatV2Op>(loc, outs[0], outputBuffer.getResult(),
                                       values, axis);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Pack: axis attr of the TF op is currently fixed to 0 (kept for
// compatibility; a transformer exists so the axis can be lifted later).
LogicalResult transformPack(const NodeInfo& node, ArrayRef<Type> outs,
                            ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.empty()) return ctx.emitError(node, "requires at least one input");
  Location loc = ctx.loc(node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::PackOp>(loc, outs[0], outputBuffer.getResult(), ins,
                                   b.getI64IntegerAttr(0));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Gather: params, indices, axis. Legacy Gather (V1) takes only
// (params, indices) and fixes axis at 0; GatherV2 carries the axis as a third
// input tensor. The input count tells the two apart.
LogicalResult transformGather(const NodeInfo& node, ArrayRef<Type> outs,
                              ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2)
    return ctx.emitError(node, "requires params and indices inputs");
  Location loc = ctx.loc(node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  int32_t batchDims = static_cast<int32_t>(getI64AttrOr(node, "batch_dims", 0));
  Value axis = (ins.size() >= 3)
                   ? ins[2]
                   : buildPrivateIntConst(ctx, node.name + "/axis", {0}, false);
  auto op =
      b.create<atir::GatherOp>(loc, outs[0], outputBuffer.getResult(), ins[0],
                               ins[1], axis, b.getI32IntegerAttr(batchDims));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.StridedSlice: the five mask bitfields map 1:1 to TF attributes.
LogicalResult transformStridedSlice(const NodeInfo& node, ArrayRef<Type> outs,
                                    ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 4)
    return ctx.emitError(node, "requires input, begin, end and strides");
  Location loc = ctx.loc(node.name);
  int32_t beginMask = static_cast<int32_t>(getI64AttrOr(node, "begin_mask", 0));
  int32_t endMask = static_cast<int32_t>(getI64AttrOr(node, "end_mask", 0));
  int32_t ellipsisMask =
      static_cast<int32_t>(getI64AttrOr(node, "ellipsis_mask", 0));
  int32_t newAxisMask =
      static_cast<int32_t>(getI64AttrOr(node, "new_axis_mask", 0));
  int32_t shrinkAxisMask =
      static_cast<int32_t>(getI64AttrOr(node, "shrink_axis_mask", 0));
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::StridedSliceOp>(
      loc, outs[0], outputBuffer.getResult(), ins[0], ins[1], ins[2], ins[3],
      b.getI32IntegerAttr(beginMask), b.getI32IntegerAttr(endMask),
      b.getI32IntegerAttr(ellipsisMask), b.getI32IntegerAttr(newAxisMask),
      b.getI32IntegerAttr(shrinkAxisMask));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Split: split_dim is the first input, value the second; num_split comes
// from the TF attr. Each result is bound to its own output name.
LogicalResult transformSplit(const NodeInfo& node, ArrayRef<Type> outs,
                             ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2)
    return ctx.emitError(node, "requires split_dim and value inputs");
  Location loc = ctx.loc(node.name);
  int64_t numSplit = 2;
  if (int64_t v; node.getAttr("num_split", v)) numSplit = v;
  auto op = b.create<atir::SplitOp>(loc, outs, ins[0], ins[1],
                                    b.getI64IntegerAttr(numSplit));
  for (size_t i = 0; i < node.outputs.size(); ++i)
    ctx.bindResult(node.outputs[i].name,
                   op.getResult(static_cast<unsigned>(i)));
  return success();
}

// atir.Pad: PadV1 has no constant_values input, so a zero pad value must be
// synthesized; PadV2 carries it as the third input.
LogicalResult transformPad(const NodeInfo& node, ArrayRef<Type> outs,
                           ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2) return ctx.emitError(node, "requires input and paddings");
  Location loc = ctx.loc(node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  Value value = (ins.size() >= 3)
                    ? ins[2]
                    : buildPrivateF32Const(ctx, node.name + "/pad_value", 0.0f);
  auto op = b.create<atir::PadOp>(loc, outs[0], outputBuffer.getResult(),
                                  ins[0], ins[1], value);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

}  // namespace annc
