// Transformers for matrix and shape semantics. Each function ports the
// behaviour of the old per-op builder handlers into the declarative framework.

#include "Builder/MLIROpBuilder.h"
#include "Builder/Transformers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
namespace annc {

// atir.MatMul: output = lhs x rhs (+ optional bias), with fusion attributes
// decoded from the TF graph.
LogicalResult transformMatMul(const NodeInfo& node, ArrayRef<Type> outs,
                              ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  Location loc = ctx.loc(node.name);
  Value lhs = ins[0];
  Value rhs = ins[1];
  Type outputTensorType = outs[0];
  Value C = b.create<atir::BufferOp>(loc, outputTensorType);
  Value bias = (ins.size() >= 3) ? ins[2] : Value();
  bool hasBias = (ins.size() >= 3);
  // TF transpose_a/transpose_b map to ATIR left_transpose/right_transpose.
  // TF serializes these as bool; parse both bool and int64 forms.
  auto readFlag = [&](const char* name) {
    bool b = false;
    if (node.getAttr(name, b)) return b;
    int64_t v = 0;
    if (node.getAttr(name, v)) return v != 0;
    return false;
  };
  bool transposeA = readFlag("transpose_a");
  bool transposeB = readFlag("transpose_b");

  auto matmul = b.create<atir::MatMulOp>(
      loc, outs[0], C, lhs, rhs, hasBias ? bias : Value{},
      b.getBoolAttr(hasBias), b.getBoolAttr(transposeB),
      b.getBoolAttr(transposeA), b.getBoolAttr(false), b.getBoolAttr(false),
      b.getF32FloatAttr(-1.0f), IntegerAttr(), IntegerAttr(), IntegerAttr(),
      IntegerAttr(), IntegerAttr(), IntegerAttr(), [&]() -> StringAttr {
        if (std::string s; node.getAttr("rhs_format", s)) {
          return b.getStringAttr(s);
        }
        return StringAttr();
      }());
  ctx.bindResult(node.outputs[0].name, matmul.getResult());
  return success();
}

// atir.BatchMatMul: batch matmul with TF adj_x/adj_y (V1) or adjoint_a/b (V2)
// mapped to the ATIR transposeA/transposeB attrs. TF serializes these as bool;
// parse both bool and int64 forms.
LogicalResult transformBatchMatMul(const NodeInfo& node, ArrayRef<Type> outs,
                                   ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2 || outs.empty())
    return ctx.emitError(node, "requires two inputs");
  Location loc = ctx.loc(node.name);
  auto readFlag = [&](const char* name) {
    bool flag = false;
    if (node.getAttr(name, flag)) return flag;
    int64_t v = 0;
    if (node.getAttr(name, v)) return v != 0;
    return false;
  };
  bool transposeA = readFlag("adj_x") || readFlag("adjoint_a");
  bool transposeB = readFlag("adj_y") || readFlag("adjoint_b");
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::BatchMatMulOp>(
      loc, outs[0], outputBuffer.getResult(), ins[0], ins[1],
      b.getBoolAttr(transposeA), b.getBoolAttr(transposeB));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Reshape: target shape comes from the upstream SSA value when present,
// otherwise from a private constant built from the static output shape.
LogicalResult transformReshape(const NodeInfo& node, ArrayRef<Type> outs,
                               ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  Location loc = ctx.loc(node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);

  Value targetShapeValue = (ins.size() >= 2 && ins[1]) ? ins[1] : Value();
  if (!targetShapeValue) {
    std::vector<int64_t> shape =
        node.outputs.empty() ? std::vector<int64_t>{} : node.outputs[0].shape;
    targetShapeValue =
        buildPrivateIntConst(ctx, node.name + "/targetShape", shape, true);
  }
  auto op = b.create<atir::ReshapeOp>(loc, outs[0], outputBuffer.getResult(),
                                      ins[0], targetShapeValue);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Reshape (Squeeze reuses Reshape): target shape from static output shape.
LogicalResult transformSqueeze(const NodeInfo& node, ArrayRef<Type> outs,
                               ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  Location loc = ctx.loc(node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  std::vector<int64_t> shape =
      node.outputs.empty() ? std::vector<int64_t>{} : node.outputs[0].shape;
  Value targetShapeValue =
      buildPrivateIntConst(ctx, node.name + "/targetShape", shape, true);
  auto op = b.create<atir::ReshapeOp>(loc, outs[0], outputBuffer.getResult(),
                                      ins[0], targetShapeValue);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

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

// atir.ExpandDims: axis from the upstream SSA value, else from the "axis"
// attr, else inferred from the input/output shapes.
LogicalResult transformExpandDims(const NodeInfo& node, ArrayRef<Type> outs,
                                  ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  Location loc = ctx.loc(node.name);
  auto inputTensorType = dyn_cast_or_null<atir::TensorType>(ins[0].getType());
  auto outputTensorType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputTensorType);

  Value axisValue = (ins.size() >= 2 && ins[1]) ? ins[1] : Value();
  if (!axisValue) {
    int32_t axis = 0;
    bool hasAxis = false;
    if (int64_t v; node.getAttr("axis", v)) {
      axis = static_cast<int32_t>(v);
      hasAxis = true;
    } else if (double d; node.getAttr("axis", d)) {
      axis = static_cast<int32_t>(d);
      hasAxis = true;
    }
    if (!hasAxis && inputTensorType && outputTensorType) {
      auto inputShape = inputTensorType.getShape();
      auto outputShape = outputTensorType.getShape();
      int64_t rank = static_cast<int64_t>(inputShape.size());
      if (rank + 1 == static_cast<int64_t>(outputShape.size())) {
        for (int64_t candidate = 0; candidate <= rank; ++candidate) {
          if (outputShape[candidate] != 1) continue;
          bool ok = true;
          for (int64_t outIdx = 0; outIdx < rank + 1; ++outIdx) {
            if (outIdx == candidate) continue;
            int64_t origIdx = outIdx < candidate ? outIdx : outIdx - 1;
            if (outputShape[outIdx] != inputShape[origIdx]) {
              ok = false;
              break;
            }
          }
          if (ok) {
            axis = static_cast<int32_t>(candidate);
            break;
          }
        }
      }
    }
    axisValue = buildPrivateIntConst(ctx, node.name + "/axis", {axis}, false);
  }
  auto op = b.create<atir::ExpandDimsOp>(loc, outs[0], outputBuffer.getResult(),
                                         ins[0], axisValue);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.broadcast: broadcast sizes taken from the static output shape.
LogicalResult transformBroadcast(const NodeInfo& node, ArrayRef<Type> outs,
                                 ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.empty() || outs.empty())
    return ctx.emitError(node, "requires at least one input");
  Location loc = ctx.loc(node.name);
  llvm::SmallVector<int64_t> broadcastSizes;
  if (!node.outputs.empty())
    broadcastSizes.assign(node.outputs[0].shape.begin(),
                          node.outputs[0].shape.end());
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::BroadcastOp>(loc, outs[0], outputBuffer.getResult(),
                                        ins[0],
                                        b.getDenseI64ArrayAttr(broadcastSizes));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

}  // namespace annc
