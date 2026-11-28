// Transformers for the ONNX side of the shared builder engine.
//
// Only the ONNX ops whose ATIR form cannot be reached by the generic
// table-driven path live here: ops that carry a value in an *attribute* where
// the ATIR op wants an *operand* (Transpose.perm, Reduce*.axes, Gather.axis),
// ops whose ATIR form is a small composition (Not, Slice), and ops that need a
// private constant with a shape the frontend does not know (none: Reciprocal
// and Sqrt are dedicated ATIR ops precisely to avoid that).
//
// Every transformer reuses the helpers exported by annc/include/Builder/
// Transformers.h so that private constants are built exactly the way the TF
// transformers build them.

#include <string>
#include <vector>

#include "Builder/MLIROpBuilder.h"
#include "Builder/Transformers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "onnx_mlir_builder.h"

using namespace mlir;
namespace annc::onnx2atir {

namespace {

// 0-D bool constant, carried the way the builder carries DType::Bool: an i32
// tensor with encoding "bool".
Value buildPrivateBoolConst(OpContext& ctx, llvm::StringRef name, bool value) {
  auto& b = ctx.builder();
  auto loc = ctx.loc(name);
  auto elems = DenseElementsAttr::get(
      RankedTensorType::get({}, b.getI32Type()),
      ArrayRef<int32_t>({value ? 1 : 0}));
  atir::TensorType type =
      atir::TensorType::get({}, b.getI32Type(), b.getStringAttr(name),
                            b.getStringAttr("bool"), {}, {}, {}, {}, {}, {},
                            elems);
  return b
      .create<atir::ConstantOp>(loc, type, b.getStringAttr(name),
                                b.getStringAttr("private"))
      .getResult();
}

// ONNX `keepdims` defaults to 1 (TF's keep_dims defaults to 0), so an absent
// attribute must not fall through to the ATIR default.
bool readKeepDims(const NodeInfo& node) {
  if (int64_t v; node.getAttr("keepdims", v)) return v != 0;
  if (bool b; node.getAttr("keepdims", b)) return b;
  return true;
}

// Reads an ONNX int-list attribute, accepting the single-int spelling too.
bool readIntAttr(const NodeInfo& node, const char* name,
                 std::vector<int64_t>& out) {
  auto it = node.attrs.find(name);
  if (it == node.attrs.end()) return false;
  if (auto* vec = std::get_if<std::vector<int64_t>>(&it->second)) {
    out.assign(vec->begin(), vec->end());
    return true;
  }
  if (auto* scalar = std::get_if<int64_t>(&it->second)) {
    out.assign(1, *scalar);
    return true;
  }
  return false;
}

}  // namespace

// atir.Transpose: ONNX carries the permutation in a `perm` attribute, while
// atir.Transpose exposes it both as an operand (runtime value) and as the
// `permutation` attribute (build-time fallback). Materialising the attribute
// into a private constant keeps both consistent -- without it the perm operand
// would fall back to the data tensor itself.
LogicalResult onnxTransformTranspose(const NodeInfo& node, ArrayRef<Type> outs,
                                     ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.empty() || outs.empty())
    return ctx.emitError(node, "Transpose requires one input and one output");
  Location loc = ctx.loc(node.name);

  std::vector<int64_t> perm;
  readIntAttr(node, "perm", perm);
  if (perm.empty() && !node.outputs.empty() && node.outputs[0].rankKnown) {
    for (int64_t i = static_cast<int64_t>(node.outputs[0].shape.size()) - 1;
         i >= 0; --i)
      perm.push_back(i);
  }
  if (perm.empty())
    return ctx.emitError(node, "cannot determine the permutation");

  SmallVector<Attribute> permAttrs;
  for (int64_t p : perm) permAttrs.push_back(b.getI64IntegerAttr(p));
  Value permValue =
      buildPrivateIntConst(ctx, node.name + "/perm", perm, /*isI64=*/true);

  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::TransposeOp>(loc, outs[0], outputBuffer.getResult(),
                                        ins[0], permValue,
                                        b.getArrayAttr(permAttrs));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Sum / atir.Prod / atir.ReduceMean all take a reduction-axes *operand*.
// In opset 13 only ReduceSum supplies axes as an operand; ReduceMean and
// ReduceProd still declare it as an attribute, and GlobalAveragePool has no
// axes at all (it reduces every spatial dimension). This transformer unifies
// the three shapes onto the operand form.
LogicalResult onnxTransformReduce(const NodeInfo& node, ArrayRef<Type> outs,
                                  ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.empty() || outs.empty())
    return ctx.emitError(node, "reduction requires at least one input");
  Location loc = ctx.loc(node.name);

  const bool keepDims = readKeepDims(node);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);

  if (node.op_type == "GlobalAveragePool") {
    // ONNX global pooling averages over every spatial dimension and keeps the
    // [N, C, 1, 1] shape; that is exactly a keepdims reduction with all axes
    // from 2 onwards.
    std::vector<int64_t> axes;
    if (auto inputType = dyn_cast_or_null<atir::TensorType>(ins[0].getType()))
      for (int64_t axis = 2; axis < static_cast<int64_t>(inputType.getShape().size());
           ++axis)
        axes.push_back(axis);
    Value indices =
        buildPrivateIntConst(ctx, node.name + "/axes", axes, /*isI64=*/true);
    auto op = b.create<atir::ReduceMeanOp>(loc, outs[0],
                                           outputBuffer.getResult(), ins[0],
                                           indices, b.getBoolAttr(true));
    ctx.bindResult(node.outputs[0].name, op.getResult());
    return success();
  }

  Value indices;
  if (node.op_type == "ReduceSum" && ins.size() >= 2) {
    indices = ins[1];
  } else {
    // An empty axes list means "reduce every dimension", which is how the
    // FusedTensordotMatMul shape chain uses ReduceProd.
    std::vector<int64_t> axes;
    readIntAttr(node, "axes", axes);
    indices =
        buildPrivateIntConst(ctx, node.name + "/axes", axes, /*isI64=*/true);
  }

  const BoolAttr keepDimsAttr = b.getBoolAttr(keepDims);
  if (node.op_type == "ReduceMean") {
    auto op = b.create<atir::ReduceMeanOp>(loc, outs[0], outputBuffer.getResult(),
                                           ins[0], indices, keepDimsAttr);
    ctx.bindResult(node.outputs[0].name, op.getResult());
  } else if (node.op_type == "ReduceProd") {
    auto op = b.create<atir::ProdOp>(loc, outs[0], outputBuffer.getResult(),
                                     ins[0], indices, keepDimsAttr);
    ctx.bindResult(node.outputs[0].name, op.getResult());
  } else {
    auto op = b.create<atir::SumOp>(loc, outs[0], outputBuffer.getResult(),
                                    ins[0], indices, keepDimsAttr);
    ctx.bindResult(node.outputs[0].name, op.getResult());
  }
  return success();
}

// atir.Gather takes the axis as an operand; ONNX Gather carries it in the
// `axis` attribute (it never became an operand -- unlike TF's GatherV2, which
// is why the TF transformer reads ins[2] and this one does not).
LogicalResult onnxTransformGather(const NodeInfo& node, ArrayRef<Type> outs,
                                  ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() != 2 || outs.empty())
    return ctx.emitError(node, "Gather requires data and indices");
  Location loc = ctx.loc(node.name);

  // Absent attribute means the ONNX default axis 0.
  int64_t axis = 0;
  node.getAttr("axis", axis);
  Value axisValue = buildPrivateIntConst(ctx, node.name + "/axis", {axis},
                                         /*isI64=*/false);
  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  auto op = b.create<atir::GatherOp>(loc, outs[0], outputBuffer.getResult(),
                                     ins[0], ins[1], axisValue,
                                     b.getI32IntegerAttr(0));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// ONNX `Not` is boolean negation. ATIR has no dedicated op, but Compare
// broadcasts (`ResultsBroadcastableShape`), so `x != true` is an exact and
// shape-independent replacement -- unlike a materialized boolean tensor, which
// could not be sized for the rank-unknown tensors these models contain.
LogicalResult onnxTransformNot(const NodeInfo& node, ArrayRef<Type> outs,
                               ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() != 1 || outs.empty())
    return ctx.emitError(node, "Not requires one input and one output");
  Location loc = ctx.loc(node.name);
  Value truth = buildPrivateBoolConst(ctx, node.name + "/true", true);
  auto op = b.create<atir::CompareOp>(loc, outs[0], ins[0], truth,
                                      b.getStringAttr("NE"),
                                      b.getBoolAttr(false));
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

// atir.Slice is TF-shaped: one `begin` and one `size` entry per *input*
// dimension, stride 1. ONNX supplies starts/ends plus optional axes/steps for
// the sliced dimensions only, with clamping and negative-index rules, so the
// conversion is done by the adapter (which holds the constant values and the
// resolved shapes) and handed over as two private attributes. All this
// transformer does is materialize those as the operands atir.Slice wants.
LogicalResult onnxTransformSlice(const NodeInfo& node, ArrayRef<Type> outs,
                                 ArrayRef<Value> ins, OpContext& ctx) {
  auto& b = ctx.builder();
  if (ins.size() < 1 || outs.empty())
    return ctx.emitError(node, "Slice requires a data input");
  Location loc = ctx.loc(node.name);

  std::vector<int64_t> begin;
  std::vector<int64_t> size;
  if (!readIntAttr(node, kSliceBeginAttr.str().c_str(), begin) ||
      !readIntAttr(node, kSliceSizeAttr.str().c_str(), size))
    return ctx.emitError(node,
                         "Slice is missing its normalized begin/size window");
  if (begin.size() != size.size())
    return ctx.emitError(node, "Slice begin/size length mismatch");

  auto outputType = dyn_cast_or_null<atir::TensorType>(outs[0]);
  auto outputBuffer = b.create<atir::BufferOp>(loc, outputType);
  Value beginValue =
      buildPrivateIntConst(ctx, node.name + "/begin", begin, /*isI64=*/true);
  Value sizeValue =
      buildPrivateIntConst(ctx, node.name + "/size", size, /*isI64=*/true);
  auto op = b.create<atir::SliceOp>(loc, outs[0], outputBuffer.getResult(),
                                    ins[0], beginValue, sizeValue);
  ctx.bindResult(node.outputs[0].name, op.getResult());
  return success();
}

}  // namespace annc::onnx2atir
