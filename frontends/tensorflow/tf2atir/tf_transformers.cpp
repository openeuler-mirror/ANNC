#include "tf_transformers.h"

#include "Builder/MLIROpBuilder.h"
#include "Builder/Transformers.h"

using namespace mlir;

namespace annc::tf2atir {

// atir.Transpose: static permutation from a "perm" attr or the constant input,
// with the permutation tensor kept as an operand for runtime interpretation.
LogicalResult transformTranspose(const NodeInfo& node, ArrayRef<Type> outs,
                                 ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.empty() || outs.empty())
    return ctx.emitError(node, "requires at least one input");
  Location loc = ctx.loc(node.name);

  SmallVector<int64_t> perm;
  if (auto it = node.attrs.find("perm"); it != node.attrs.end()) {
    if (auto* vec = std::get_if<std::vector<int64_t>>(&it->second))
      perm.assign(vec->begin(), vec->end());
  }
  if (perm.empty() && node.inputs.size() >= 2) {
    std::vector<int64_t> decoded;
    if (const NodeInfo* cnode = ctx.findNode(node.inputs[1]);
        cnode && decodeIntConstValues(cnode, decoded))
      perm.assign(decoded.begin(), decoded.end());
  }
  if (perm.empty() && !node.outputs.empty()) {
    int64_t rank = static_cast<int64_t>(node.outputs[0].shape.size());
    for (int64_t i = rank - 1; i >= 0; --i) perm.push_back(i);
  }
  if (perm.empty()) return ctx.emitError(node, "cannot determine permutation");

  SmallVector<Attribute> permAttrs;
  for (int64_t p : perm) permAttrs.push_back(b.getI64IntegerAttr(p));
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  Value permValue = (ins.size() >= 2) ? ins[1] : ins[0];
  auto op =
      b.create<atir::TransposeOp>(loc, outs[0], outputBuffer.getResult(),
                                  ins[0], permValue, b.getArrayAttr(permAttrs));
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

}  // namespace annc::tf2atir
