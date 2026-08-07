// Transformers for reductions, control flow and misc semantics, plus the
// shared private-constant helper.

#include "Builder/MLIROpBuilder.h"
#include "Builder/Transformers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
namespace annc {

mlir::Value buildPrivateIntConst(OpContext& ctx, llvm::StringRef name,
                                 llvm::ArrayRef<int64_t> values, bool isI64) {
  auto& b = ctx.builder();
  auto loc = ctx.loc(name);
  if (isI64) {
    std::vector<int64_t> shape{static_cast<int64_t>(values.size())};
    auto elems = DenseElementsAttr::get(
        RankedTensorType::get(shape, b.getI64Type()), values);
    atir::TensorType t =
        atir::TensorType::get(shape, b.getI64Type(), b.getStringAttr(name), {},
                              {}, {}, {}, {}, {}, {}, elems);
    return b
        .create<atir::ConstantOp>(loc, t, b.getStringAttr(name),
                                  b.getStringAttr("private"))
        .getResult();
  }
  // 0-D int32 scalar (axis / num_segments).
  std::vector<int32_t> scalar(values.begin(), values.end());
  auto elems = DenseElementsAttr::get(RankedTensorType::get({}, b.getI32Type()),
                                      ArrayRef<int32_t>(scalar));
  atir::TensorType t =
      atir::TensorType::get({}, b.getI32Type(), b.getStringAttr(name), {}, {},
                            {}, {}, {}, {}, {}, elems);
  return b
      .create<atir::ConstantOp>(loc, t, b.getStringAttr(name),
                                b.getStringAttr("private"))
      .getResult();
}

mlir::Value buildPrivateF32Const(OpContext& ctx, llvm::StringRef name,
                                 float value) {
  auto& b = ctx.builder();
  auto loc = ctx.loc(name);
  auto elems =
      DenseElementsAttr::get(RankedTensorType::get({}, b.getF32Type()), value);
  atir::TensorType t =
      atir::TensorType::get({}, b.getF32Type(), b.getStringAttr(name), {}, {},
                            {}, {}, {}, {}, {}, elems);
  return b
      .create<atir::ConstantOp>(loc, t, b.getStringAttr(name),
                                b.getStringAttr("private"))
      .getResult();
}

// atir.Sum: indices from the upstream value, else an empty private constant.
LogicalResult transformSum(const NodeInfo& node, ArrayRef<Type> outs,
                           ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.empty()) return ctx.emitError(node, "requires at least one input");
  Location loc = ctx.loc(node.name);
  Value input = ins[0];
  Value indices = Value{};
  if (ins.size() >= 2) {
    indices = ins[1];
  } else {
    std::vector<int64_t> indicesShape = {0};
    auto indicesTensorType = atir::TensorType::get(
        indicesShape, b.getI32Type(), b.getStringAttr("sum_indices"),
        mlir::Attribute());
    auto indicesConst = b.create<atir::ConstantOp>(
        loc, indicesTensorType, b.getStringAttr("sum_indices"),
        b.getStringAttr("private"));
    indices = indicesConst.getResult();
  }
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::SumOp>(loc, outs[0], outputBuffer.getResult(), input,
                                  indices, b.getBoolAttr(false));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Prod: like Sum, with keep_dims from the TF attrs.
LogicalResult transformProd(const NodeInfo& node, ArrayRef<Type> outs,
                            ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 1) return ctx.emitError(node, "requires at least one input");
  Location loc = ctx.loc(node.name);
  Value input = ins[0];
  Value indices = Value{};
  if (ins.size() >= 2) {
    indices = ins[1];
  } else {
    std::vector<int64_t> indicesShape = {0};
    auto rankedType = RankedTensorType::get(indicesShape, b.getI32Type());
    auto emptyAttr =
        DenseElementsAttr::get(rankedType, llvm::ArrayRef<int32_t>{});
    auto indicesTensorType = atir::TensorType::get(
        indicesShape, b.getI32Type(), b.getStringAttr("prod_indices"),
        Attribute(), ArrayAttr(), StringAttr(), atir::MemTypeAttr(),
        IntegerAttr(), atir::TilingAttr(), atir::TilingAttr(), emptyAttr);
    auto indicesConst = b.create<atir::ConstantOp>(
        loc, indicesTensorType, b.getStringAttr("prod_indices"),
        b.getStringAttr("private"));
    indices = indicesConst.getResult();
  }
  bool keepDims = false;
  if (node.getAttr("keep_dims", keepDims)) {
  } else if (int64_t v; node.getAttr("keepdims", v)) {
    keepDims = (v != 0);
  }
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::ProdOp>(loc, outs[0], outputBuffer.getResult(),
                                   input, indices, b.getBoolAttr(keepDims));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Merge: two results (output value, value index).
LogicalResult transformMerge(const NodeInfo& node, ArrayRef<Type> outs,
                             ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.empty() || outs.size() < 2)
    return ctx.emitError(node, "requires inputs and two output types");
  Location loc = ctx.loc(node.name);
  auto mergeOp = b.create<atir::MergeOp>(loc, TypeRange{outs[0], outs[1]}, ins);
  if (node.outputs.size() >= 2) {
    ctx.bindResult(node.outputs[0].name, mergeOp.getOutput());
    ctx.bindResult(node.outputs[1].name, mergeOp.getValueIndex());
  } else if (node.outputs.size() == 1) {
    ctx.bindResult(node.outputs[0].name, mergeOp.getOutput());
  }
  return success();
}

// atir.DynamicPartition: splits data by partition ids.
LogicalResult transformDynamicPartition(const NodeInfo& node,
                                        ArrayRef<Type> outs,
                                        ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2)
    return ctx.emitError(node, "requires data and partitions");
  Location loc = ctx.loc(node.name);
  Value data = ins[0];
  Value partitions = ins[1];
  int32_t numPartitions = static_cast<int32_t>(outs.size());
  auto partitionOp = b.create<atir::DynamicPartitionOp>(
      loc, outs, data, partitions, b.getI32IntegerAttr(numPartitions));
  for (size_t i = 0; i < node.outputs.size() && i < outs.size(); ++i) {
    Value outVal = partitionOp.getOutputs()[i];
    ctx.bindResult(node.outputs[i].name, outVal);
    if (i == 0)
      ctx.bindResult(node.name, outVal);
    else
      ctx.bindResult(node.name + ":" + std::to_string(i), outVal);
  }
  return success();
}

// atir.ParallelDynamicStitch: interleaves indices and data halves.
LogicalResult transformParallelDynamicStitch(const NodeInfo& node,
                                             ArrayRef<Type> outs,
                                             ArrayRef<Value> ins,
                                             OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2) return ctx.emitError(node, "requires indices and data");
  Location loc = ctx.loc(node.name);
  size_t half = ins.size() / 2;
  SmallVector<Value, 4> indices(ins.begin(), ins.begin() + half);
  SmallVector<Value, 4> data(ins.begin() + half, ins.end());
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::ParallelDynamicStitchOp>(
      loc, outs[0], outputBuffer.getResult(), indices, data);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.StringToHashBucketFast: num_buckets from the TF attr (default 100).
LogicalResult transformStringToHashBucketFast(const NodeInfo& node,
                                              ArrayRef<Type> outs,
                                              ArrayRef<Value> ins,
                                              OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 1 || outs.empty())
    return ctx.emitError(node, "requires one input");
  Location loc = ctx.loc(node.name);
  int64_t numBuckets = 100;
  if (int64_t v; node.getAttr("num_buckets", v)) numBuckets = v;
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::StringToHashBucketFastOp>(
      loc, outs[0], outputBuffer.getResult(), ins[0],
      b.getI64IntegerAttr(numBuckets));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Unique: two results (unique values, indices).
LogicalResult transformUnique(const NodeInfo& node, ArrayRef<Type> outs,
                              ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 1) return ctx.emitError(node, "requires one input");
  Location loc = ctx.loc(node.name);
  auto op = b.create<atir::UniqueOp>(loc, outs[0], outs[1], ins[0]);
  if (node.outputs.size() >= 2) {
    ctx.bindResult(node.outputs[0].name, op.getY());
    ctx.bindResult(node.outputs[1].name, op.getIdx());
  } else if (node.outputs.size() == 1) {
    ctx.bindResult(node.outputs[0].name, op.getY());
  }
  return success();
}

// atir.TopK: two results (values, indices); sorted flag from the TF attr.
LogicalResult transformTopK(const NodeInfo& node, ArrayRef<Type> outs,
                            ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2) return ctx.emitError(node, "requires input and k");
  Location loc = ctx.loc(node.name);
  bool sorted = true;
  if (!node.getAttr("sorted", sorted)) {
    if (int64_t v; node.getAttr("sorted", v)) sorted = (v != 0);
  }
  auto op = b.create<atir::TopKOp>(loc, outs[0], outs[1], ins[0], ins[1],
                                   b.getBoolAttr(sorted));
  if (node.outputs.size() >= 2) {
    ctx.bindResult(node.outputs[0].name, op.getValues());
    ctx.bindResult(node.outputs[1].name, op.getIndices());
  } else if (node.outputs.size() == 1) {
    ctx.bindResult(node.outputs[0].name, op.getValues());
  }
  return success();
}

// atir.Mul: Square is x * x.
LogicalResult transformSquare(const NodeInfo& node, ArrayRef<Type> outs,
                              ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  Location loc = ctx.loc(node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::MulOp>(loc, outs[0], outputBuffer.getResult(),
                                  ins[0], ins[0]);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Mul(atir.Sub(a, b)): SquaredDifference is (a - b) * (a - b).
LogicalResult transformSquaredDifference(const NodeInfo& node,
                                         ArrayRef<Type> outs,
                                         ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 2) return ctx.emitError(node, "requires two inputs");
  Location loc = ctx.loc(node.name);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto subBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto subOp = b.create<atir::SubOp>(loc, outs[0], subBuffer.getResult(),
                                     ins[0], ins[1]);
  auto outBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto mulOp = b.create<atir::MulOp>(loc, outs[0], outBuffer.getResult(),
                                     subOp.getResult(), subOp.getResult());
  ctx.bindResult(node.outputs[0].name, mulOp.getResult());
  return success();
}

// atir.variable: a named, publicly visible variable.
LogicalResult transformVariable(const NodeInfo& node, ArrayRef<Type> outs,
                                ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (outs.empty()) return ctx.emitError(node, "requires one output type");
  Location loc = ctx.loc(node.name);
  auto op = b.create<atir::VariableOp>(loc, outs[0], b.getStringAttr(node.name),
                                       b.getStringAttr("public"));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

}  // namespace annc
