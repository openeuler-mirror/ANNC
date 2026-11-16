#include "Dialect/Atir/AtirOps.h"
#include "framework_fallback.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXTypes.hpp"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {

void collectSourceIds(Location loc, SmallVectorImpl<StringRef> &ids);
ArrayAttr sourceIdsFromOp(PatternRewriter &rewriter, Operation *op);
void copySourceIds(PatternRewriter &rewriter, Operation *source,
                   Operation *target);
void ensureSourceNameLoc(Operation *op);

Operation *createOp(PatternRewriter &rewriter, StringRef name, Location loc,
                    TypeRange resultTypes, ValueRange operands = {},
                    ArrayRef<NamedAttribute> attributes = {}) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(attributes);
  SmallVector<StringRef> sourceIds;
  collectSourceIds(loc, sourceIds);
  llvm::sort(sourceIds);
  sourceIds.erase(std::unique(sourceIds.begin(), sourceIds.end()),
                  sourceIds.end());
  if (!sourceIds.empty()) {
    SmallVector<Attribute> idAttrs;
    for (StringRef id : sourceIds)
      idAttrs.push_back(rewriter.getStringAttr(id));
    state.addAttribute("source_ids", rewriter.getArrayAttr(idAttrs));
  }
  return rewriter.create(state);
}

DenseI64ArrayAttr i64Array(MLIRContext *ctx, ArrayRef<int64_t> values) {
  return DenseI64ArrayAttr::get(ctx, values);
}

SmallVector<int64_t> getI64Array(Operation *op, StringRef name) {
  SmallVector<int64_t> result;
  if (auto dense = op->getAttrOfType<DenseI64ArrayAttr>(name))
    result.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
  else if (auto array = op->getAttrOfType<ArrayAttr>(name))
    for (Attribute attr : array)
      result.push_back(cast<IntegerAttr>(attr).getValue().getSExtValue());
  return result;
}

Value scalarConstant(PatternRewriter &rewriter, Location loc, Type elementType,
                     int64_t value) {
  auto type = RankedTensorType::get({}, elementType);
  ElementsAttr attr;
  if (auto integer = dyn_cast<IntegerType>(elementType))
    attr = DenseElementsAttr::get(type, APInt(integer.getWidth(), value));
  else if (auto floating = dyn_cast<FloatType>(elementType))
    attr = DenseElementsAttr::get(type, APFloat(floating.getFloatSemantics(),
                                                value == 0 ? "0.0" : "1.0"));
  else
    return {};
  return createOp(rewriter, "stablehlo.constant", loc, type, {},
                  {rewriter.getNamedAttr("value", attr)})
      ->getResult(0);
}

Value shapeOf(PatternRewriter &rewriter, Location loc, Value value,
              int64_t rank) {
  auto type = RankedTensorType::get({rank}, rewriter.getIndexType());
  return createOp(rewriter, "shape.shape_of", loc, type, value)->getResult(0);
}

Value dynamicBroadcast(PatternRewriter &rewriter, Location loc, Value value,
                       Value shape, RankedTensorType resultType,
                       ArrayRef<int64_t> dimensions) {
  return createOp(rewriter, "stablehlo.dynamic_broadcast_in_dim", loc,
                  resultType, {value, shape},
                  {rewriter.getNamedAttr(
                      "broadcast_dimensions",
                      i64Array(rewriter.getContext(), dimensions))})
      ->getResult(0);
}

class SimpleUnaryPattern : public RewritePattern {
 public:
  SimpleUnaryPattern(StringRef sourceName, StringRef targetName,
                     MLIRContext *ctx)
      : RewritePattern(sourceName, 1, ctx), targetName(targetName) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ensureSourceNameLoc(op);
    Operation *replacement = createOp(rewriter, targetName, op->getLoc(),
                                      op->getResultTypes(), op->getOperands());
    copySourceIds(rewriter, op, replacement);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }

 private:
  StringRef targetName;
};

class ReciprocalPattern : public RewritePattern {
 public:
  explicit ReciprocalPattern(MLIRContext *ctx)
      : RewritePattern("onnx.Reciprocal", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ensureSourceNameLoc(op);
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!resultType || !inputType) return failure();
    Value one =
        scalarConstant(rewriter, op->getLoc(), resultType.getElementType(), 1);
    Value shape = shapeOf(rewriter, op->getLoc(), op->getOperand(0),
                          resultType.getRank());
    SmallVector<int64_t> dims;
    Value broadcast =
        dynamicBroadcast(rewriter, op->getLoc(), one, shape, resultType, dims);
    Operation *divide = createOp(rewriter, "stablehlo.divide", op->getLoc(),
                                 resultType, {broadcast, op->getOperand(0)});
    copySourceIds(rewriter, op, divide);
    rewriter.replaceOp(op, divide->getResults());
    return success();
  }
};

class ReduceProdPattern : public RewritePattern {
 public:
  explicit ReduceProdPattern(MLIRContext *ctx)
      : RewritePattern("onnx.ReduceProdV13", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ensureSourceNameLoc(op);
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType) return failure();
    SmallVector<int64_t> axes = getI64Array(op, "axes");
    auto keepAttr = op->getAttrOfType<IntegerAttr>("keepdims");
    if (!keepAttr || axes.empty()) return failure();
    bool keep = keepAttr.getValue().getSExtValue() != 0;
    SmallVector<int64_t> normalized;
    for (int64_t axis : axes)
      normalized.push_back(axis < 0 ? axis + inputType.getRank() : axis);
    SmallVector<int64_t> reducedShape;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      bool reduced = llvm::is_contained(normalized, i);
      if (!reduced) reducedShape.push_back(inputType.getDimSize(i));
    }
    auto reduceType =
        RankedTensorType::get(reducedShape, inputType.getElementType());
    Value init =
        scalarConstant(rewriter, op->getLoc(), inputType.getElementType(), 1);
    OperationState state(op->getLoc(), "stablehlo.reduce");
    state.addOperands({op->getOperand(0), init});
    state.addTypes(reduceType);
    state.addAttribute("dimensions",
                       i64Array(rewriter.getContext(), normalized));
    state.addRegion();
    Operation *reduce = rewriter.create(state);
    copySourceIds(rewriter, op, reduce);
    Region &region = reduce->getRegion(0);
    Block *body = new Block();
    region.push_back(body);
    Type scalarType = RankedTensorType::get({}, inputType.getElementType());
    Value lhs = body->addArgument(scalarType, op->getLoc());
    Value rhs = body->addArgument(scalarType, op->getLoc());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(body);
    Operation *mul = createOp(rewriter, "stablehlo.multiply", op->getLoc(),
                              scalarType, {lhs, rhs});
    copySourceIds(rewriter, op, mul);
    Operation *returnOp = createOp(rewriter, "stablehlo.return", op->getLoc(),
                                   TypeRange{}, mul->getResult(0));
    copySourceIds(rewriter, op, returnOp);
    rewriter.setInsertionPointAfter(reduce);
    Value result = reduce->getResult(0);
    if (result.getType() != resultType)
      result = createOp(rewriter, "stablehlo.reshape", op->getLoc(), resultType,
                        result)
                   ->getResult(0);
    if (Operation *resultOp = result.getDefiningOp())
      copySourceIds(rewriter, op, resultOp);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ModPattern : public RewritePattern {
 public:
  explicit ModPattern(MLIRContext *ctx) : RewritePattern("onnx.Mod", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ensureSourceNameLoc(op);
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto lhsType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    if (!resultType || !lhsType || !rhsType) return failure();
    Location loc = op->getLoc();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value shape = createOp(
                      rewriter, "shape.broadcast", loc,
                      RankedTensorType::get({resultType.getRank()},
                                            rewriter.getIndexType()),
                      {shapeOf(rewriter, loc, lhs, lhsType.getRank()),
                       shapeOf(rewriter, loc, rhs, rhsType.getRank())})
                      ->getResult(0);
    auto broadcast = [&](Value value, RankedTensorType type) {
      if (type == resultType && resultType.hasStaticShape()) return value;
      SmallVector<int64_t> dims;
      for (int64_t i = 0; i < type.getRank(); ++i)
        dims.push_back(resultType.getRank() - type.getRank() + i);
      return dynamicBroadcast(rewriter, loc, value, shape, resultType, dims);
    };
    lhs = broadcast(lhs, lhsType);
    rhs = broadcast(rhs, rhsType);
    Value remainder =
        createOp(rewriter, "stablehlo.remainder", loc, resultType, {lhs, rhs})
            ->getResult(0);
    copySourceIds(rewriter, op, remainder.getDefiningOp());
    auto fmod = op->getAttrOfType<IntegerAttr>("fmod");
    if (fmod && fmod.getValue().getSExtValue() != 0) {
      rewriter.replaceOp(op, remainder);
      return success();
    }
    Type elementType = resultType.getElementType();
    Value zero = scalarConstant(rewriter, loc, elementType, 0);
    zero = dynamicBroadcast(rewriter, loc, zero, shape, resultType, {});
    auto boolType =
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    auto compare = [&](Value a, Value b, stablehlo::ComparisonDirection dir) {
      return createOp(
                 rewriter, "stablehlo.compare", loc, boolType, {a, b},
                 {rewriter.getNamedAttr("comparison_direction",
                                        stablehlo::ComparisonDirectionAttr::get(
                                            rewriter.getContext(), dir))})
          ->getResult(0);
    };
    Value lhsNegative = compare(lhs, zero, stablehlo::ComparisonDirection::LT);
    Value rhsNegative = compare(rhs, zero, stablehlo::ComparisonDirection::LT);
    Value signDiff = createOp(rewriter, "stablehlo.xor", loc, boolType,
                              {lhsNegative, rhsNegative})
                         ->getResult(0);
    Value nonzero =
        compare(remainder, zero, stablehlo::ComparisonDirection::NE);
    Value adjust =
        createOp(rewriter, "stablehlo.and", loc, boolType, {signDiff, nonzero})
            ->getResult(0);
    Value added =
        createOp(rewriter, "stablehlo.add", loc, resultType, {remainder, rhs})
            ->getResult(0);
    Value result = createOp(rewriter, "stablehlo.select", loc, resultType,
                            {adjust, added, remainder})
                       ->getResult(0);
    copySourceIds(rewriter, op, result.getDefiningOp());
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConcatShapeTransposePattern : public RewritePattern {
 public:
  explicit ConcatShapeTransposePattern(MLIRContext *ctx)
      : RewritePattern("onnx.ConcatShapeTranspose", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ensureSourceNameLoc(op);
    auto first = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto shapeType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto transposedType =
        dyn_cast<RankedTensorType>(op->getResult(1).getType());
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (!first || !shapeType || !transposedType || !axisAttr) return failure();
    int64_t rank = first.getRank();
    int64_t axis = axisAttr.getValue().getSExtValue();
    if (axis < 0) axis += rank;
    SmallVector<int64_t> concatShape(first.getShape().begin(),
                                     first.getShape().end());
    for (Value operand : op->getOperands().drop_front()) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || type.getRank() != rank) return failure();
      if (concatShape[axis] != ShapedType::kDynamic &&
          type.getDimSize(axis) != ShapedType::kDynamic)
        concatShape[axis] += type.getDimSize(axis);
      else
        concatShape[axis] = ShapedType::kDynamic;
    }
    auto concatType =
        RankedTensorType::get(concatShape, first.getElementType());
    SmallVector<NamedAttribute> concatAttrs = {
        rewriter.getNamedAttr("dimension", rewriter.getI64IntegerAttr(axis))};
    Value concat = createOp(rewriter, "stablehlo.concatenate", op->getLoc(),
                            concatType, op->getOperands(), concatAttrs)
                       ->getResult(0);
    copySourceIds(rewriter, op, concat.getDefiningOp());

    int64_t start =
        op->getAttrOfType<IntegerAttr>("start")
            ? op->getAttrOfType<IntegerAttr>("start").getValue().getSExtValue()
            : 0;
    int64_t end =
        op->getAttrOfType<IntegerAttr>("end")
            ? op->getAttrOfType<IntegerAttr>("end").getValue().getSExtValue()
            : rank;
    if (start < 0) start += rank;
    if (end < 0) end += rank;
    SmallVector<Value> shapeValues;
    for (int64_t dim = start; dim < end; ++dim) {
      Value index =
          createOp(rewriter, "arith.constant", op->getLoc(),
                   rewriter.getIndexType(), {},
                   {rewriter.getNamedAttr("value", rewriter.getIndexAttr(dim))})
              ->getResult(0);
      Value extent = createOp(rewriter, "tensor.dim", op->getLoc(),
                              rewriter.getIndexType(), {concat, index})
                         ->getResult(0);
      shapeValues.push_back(createOp(rewriter, "arith.index_cast", op->getLoc(),
                                     rewriter.getI64Type(), extent)
                                ->getResult(0));
    }
    if (static_cast<int64_t>(shapeValues.size()) != shapeType.getDimSize(0))
      return failure();
    Value shape = createOp(rewriter, "tensor.from_elements", op->getLoc(),
                           shapeType, shapeValues)
                      ->getResult(0);
    copySourceIds(rewriter, op, shape.getDefiningOp());
    auto perm = getI64Array(op, "perm");
    Value transposed = concat;
    if (!perm.empty())
      transposed =
          createOp(rewriter, "stablehlo.transpose", op->getLoc(),
                   transposedType, concat,
                   {rewriter.getNamedAttr(
                       "permutation", i64Array(rewriter.getContext(), perm))})
              ->getResult(0);
    else if (transposed.getType() != transposedType)
      transposed = createOp(rewriter, "stablehlo.reshape", op->getLoc(),
                            transposedType, transposed)
                       ->getResult(0);
    if (Operation *resultOp = transposed.getDefiningOp())
      copySourceIds(rewriter, op, resultOp);
    rewriter.replaceOp(op, {shape, transposed});
    return success();
  }
};

class EntryPointPattern : public RewritePattern {
 public:
  explicit EntryPointPattern(MLIRContext *ctx)
      : RewritePattern("onnx.EntryPoint", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};


void collectSourceIds(Location loc, SmallVectorImpl<StringRef> &ids) {
  if (auto named = dyn_cast<NameLoc>(loc)) {
    StringRef name = named.getName().getValue();
    if (name.contains("/op")) ids.push_back(name);
    collectSourceIds(named.getChildLoc(), ids);
  } else if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fused.getLocations()) collectSourceIds(child, ids);
  } else if (auto callsite = dyn_cast<CallSiteLoc>(loc)) {
    collectSourceIds(callsite.getCallee(), ids);
    collectSourceIds(callsite.getCaller(), ids);
  }
}


ArrayAttr sourceIdsFromOp(PatternRewriter &rewriter, Operation *op) {
  if (auto attr = op->getAttrOfType<ArrayAttr>("source_ids")) return attr;
  if (auto id = op->getAttrOfType<StringAttr>("onnx_source_id"))
    return rewriter.getArrayAttr({id});
  SmallVector<StringRef> ids;
  collectSourceIds(op->getLoc(), ids);
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  SmallVector<Attribute> attrs;
  for (StringRef id : ids) attrs.push_back(rewriter.getStringAttr(id));
  return rewriter.getArrayAttr(attrs);
}

void copySourceIds(PatternRewriter &rewriter, Operation *source,
                   Operation *target) {
  ArrayAttr ids = sourceIdsFromOp(rewriter, source);
  if (!ids.empty()) target->setAttr("source_ids", ids);
}

void ensureSourceNameLoc(Operation *op) {
  SmallVector<StringRef> ids;
  collectSourceIds(op->getLoc(), ids);
  if (!ids.empty()) return;
  if (auto id = op->getAttrOfType<StringAttr>("onnx_source_id"))
    op->setLoc(NameLoc::get(id, op->getLoc()));
}

void materializeNestedProvenance(ModuleOp module) {
  // Materialize direct locations first. Operation::walk visits nested
  // operations before their enclosing operation in post-order, so inherited
  // provenance must be resolved in a separate second walk.
  module.walk([&](Operation *op) {
    if (op == module || isa<func::FuncOp>(op) ||
        op->hasTrait<OpTrait::IsTerminator>() ||
        op->getName().getDialectNamespace() == "onnx" ||
        (op->getAttrOfType<ArrayAttr>("source_ids") &&
         !op->getAttrOfType<ArrayAttr>("source_ids").empty()))
      return;
    SmallVector<StringRef> ids;
    collectSourceIds(op->getLoc(), ids);
    if (!ids.empty()) {
      llvm::sort(ids);
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      SmallVector<Attribute> attrs;
      for (StringRef id : ids)
        attrs.push_back(StringAttr::get(module.getContext(), id));
      op->setAttr("source_ids", ArrayAttr::get(module.getContext(), attrs));
    }
  });

  module.walk([&](Operation *op) {
    if (op == module || isa<func::FuncOp>(op) ||
        op->hasTrait<OpTrait::IsTerminator>() ||
        op->getName().getDialectNamespace() == "onnx" ||
        (op->getAttrOfType<ArrayAttr>("source_ids") &&
         !op->getAttrOfType<ArrayAttr>("source_ids").empty()))
      return;
    for (Region *region = op->getParentRegion(); region;
         region = region->getParentRegion()) {
      Operation *parent = region->getParentOp();
      if (!parent) continue;
      if (auto parentIds = parent->getAttrOfType<ArrayAttr>("source_ids")) {
        if (!parentIds.empty()) {
          op->setAttr("source_ids", parentIds);
          return;
        }
      }
    }

    StringRef name = op->getName().getStringRef();
    if (name == "arith.constant" || name == "shape.const_shape" ||
        name == "stablehlo.constant") {
      op->setAttr("onnx.provenance_kind",
                  StringAttr::get(module.getContext(), "synthetic"));
      op->setAttr(
          "onnx.provenance_reason",
          StringAttr::get(
              module.getContext(),
              "lowering helper constant has no unique ONNX source node"));
    }
  });
}



class DynamicTilePattern : public RewritePattern {
 public:
  explicit DynamicTilePattern(MLIRContext *ctx)
      : RewritePattern("annc_preprocess.dynamic_tile", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input = op->getOperand(0), repeats = op->getOperand(1);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto repeatsType = dyn_cast<RankedTensorType>(repeats.getType());
    auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !repeatsType || !outputType ||
        repeatsType.getRank() != 1 ||
        repeatsType.getDimSize(0) != inputType.getRank() ||
        !repeatsType.getElementType().isInteger(64) ||
        outputType.getRank() != inputType.getRank())
      return rewriter.notifyMatchFailure(op,
                                         "invalid dynamic Tile marker types");

    Location loc = op->getLoc();
    int64_t rank = inputType.getRank();
    auto oneI64 = RankedTensorType::get({1}, rewriter.getI64Type());
    auto create = [&](StringRef name, Type type, ValueRange operands,
                      ArrayRef<NamedAttribute> attrs = {}) {
      OperationState state(loc, name);
      state.addTypes(type);
      state.addOperands(operands);
      state.addAttributes(attrs);
      if (auto ids = op->getAttr("source_ids"))
        state.addAttribute("source_ids", ids);
      return rewriter.create(state)->getResult(0);
    };
    auto slice = [&](Value vector, int64_t i) {
      return create(
          "stablehlo.slice", oneI64, vector,
          {rewriter.getNamedAttr("start_indices",
                                 rewriter.getDenseI64ArrayAttr({i})),
           rewriter.getNamedAttr("limit_indices",
                                 rewriter.getDenseI64ArrayAttr({i + 1})),
           rewriter.getNamedAttr("strides",
                                 rewriter.getDenseI64ArrayAttr({1}))});
    };
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, input);
    Value extents = rewriter.create<shape::ToExtentTensorOp>(
        loc, shape::getExtentTensorType(rewriter.getContext(), rank), shape);
    auto extentI64Type = RankedTensorType::get({rank}, rewriter.getI64Type());
    Value extentsI64 =
        rewriter.create<arith::IndexCastOp>(loc, extentI64Type, extents);
    SmallVector<Value> expandedShape, finalShape;
    SmallVector<int64_t> broadcastDims;
    for (int64_t i = 0; i < rank; ++i) {
      Value dim =
          inputType.isDynamicDim(i)
              ? slice(extentsI64, i)
              : create("stablehlo.constant", oneI64, {},
                       {rewriter.getNamedAttr(
                           "value", DenseIntElementsAttr::get(
                                        oneI64, {inputType.getDimSize(i)}))});
      Value repeat = slice(repeats, i);
      expandedShape.push_back(repeat);
      expandedShape.push_back(dim);
      broadcastDims.push_back(2 * i + 1);
      finalShape.push_back(create("stablehlo.multiply", oneI64, {repeat, dim}));
    }
    auto concat = [&](ValueRange values) {
      auto type = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                        rewriter.getI64Type());
      return create(
          "stablehlo.concatenate", type, values,
          {rewriter.getNamedAttr("dimension", rewriter.getI64IntegerAttr(0))});
    };
    Value expanded = concat(expandedShape);
    SmallVector<int64_t> expandedDims(2 * rank, ShapedType::kDynamic);
    auto broadcastType =
        RankedTensorType::get(expandedDims, inputType.getElementType());
    Value broadcast = create(
        "stablehlo.dynamic_broadcast_in_dim", broadcastType, {input, expanded},
        {rewriter.getNamedAttr("broadcast_dimensions",
                               rewriter.getDenseI64ArrayAttr(broadcastDims))});
    Value finalShapeTensor = concat(finalShape);
    Value result = create("stablehlo.dynamic_reshape", outputType,
                          {broadcast, finalShapeTensor});
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Framework-owned operations keep their original attributes and tensor edges.
// They are never decomposed into numerical operations that ANNC might absorb.
class FrameworkFallbackPattern : public RewritePattern {
 public:
  explicit FrameworkFallbackPattern(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    auto function = op->getAttrOfType<StringAttr>("function_name");
    if (!domain || !function ||
        !annc::isSupportedOnnxFallback(domain.getValue(), function.getValue(),
                                       op->getNumOperands(),
                                       op->getNumResults()))
      return rewriter.notifyMatchFailure(op,
                                         "unsupported ONNX framework fallback");
    ensureSourceNameLoc(op);
    OperationState state(op->getLoc(), atir::OpaqueOp::getOperationName());
    state.addOperands(op->getOperands());
    state.addTypes(op->getResultTypes());
    state.addAttribute(
        "opType", rewriter.getStringAttr(
                      (domain.getValue() + "::" + function.getValue()).str()));
    state.addAttribute("metadata", rewriter.getDictionaryAttr(op->getAttrs()));
    Operation *fallback = rewriter.create(state);
    copySourceIds(rewriter, op, fallback);
    if (!fallback->hasAttr("source_ids"))
      if (auto node = op->getAttrOfType<StringAttr>("onnx_node_name"))
        fallback->setAttr("source_ids", rewriter.getArrayAttr({node}));
    rewriter.replaceOp(op, fallback->getResults());
    return success();
  }
};

class ExternalLoweringPass
    : public PassWrapper<ExternalLoweringPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExternalLoweringPass)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<atir::AtirDialect, arith::ArithDialect, shape::ShapeDialect,
                    stablehlo::StablehloDialect>();
  }
  StringRef getArgument() const final { return "onnx-external-to-stablehlo"; }
  StringRef getDescription() const final {
    return "Lower residual ONNX operations to StableHLO outside onnx-mlir";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimpleUnaryPattern>("onnx.Not", "stablehlo.not",
                                     &getContext());
    patterns.add<SimpleUnaryPattern>("onnx.Or", "stablehlo.or", &getContext());
    patterns.add<ReciprocalPattern, ReduceProdPattern, ModPattern,
                 ConcatShapeTransposePattern, EntryPointPattern,
                 DynamicTilePattern>(&getContext());
    patterns.add<FrameworkFallbackPattern>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
    materializeNestedProvenance(getOperation());
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "onnx" ||
          op->getName().getStringRef() == "annc_preprocess.dynamic_tile") {
        op->emitError("external lowering left an ONNX operation");
        signalPassFailure();
      }
    });
  }
};

static PassRegistration<ExternalLoweringPass> pass;

}  // namespace
