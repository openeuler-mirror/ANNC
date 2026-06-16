#include "Conversion/Common/CustomizeCallLowering.h"

#include "Dialect/Atir/CustomOpSchema.h"
#include "Kernel/KernelPriorityResolver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "iostream"

using namespace mlir;

namespace atir {
namespace {

struct SchemaArg {
  StringRef name;
  StringRef kind;
  int64_t rank = -1;
  StringRef typeVar;
};

std::optional<ArrayAttr> getArrayAttr(DictionaryAttr metadata, StringRef name) {
  if (!metadata) {
    return std::nullopt;
  }
  if (auto attr = dyn_cast_or_null<ArrayAttr>(metadata.get(name))) {
    return attr;
  }
  return std::nullopt;
}

std::optional<SmallVector<SchemaArg>>
readSchemaArgs(DictionaryAttr metadata) {
  auto names = getArrayAttr(metadata, "custom.arg_names");
  auto kinds = getArrayAttr(metadata, "custom.arg_kinds");
  auto ranks = getArrayAttr(metadata, "custom.arg_ranks");
  auto typeVars = getArrayAttr(metadata, "custom.arg_type_vars");
  if (!names || !kinds || !ranks || !typeVars) {
    return std::nullopt;
  }
  if (names->size() != kinds->size() || kinds->size() != ranks->size() ||
      kinds->size() != typeVars->size()) {
    return std::nullopt;
  }

  SmallVector<SchemaArg> args;
  args.reserve(kinds->size());
  for (size_t i = 0; i < kinds->size(); ++i) {
    auto name = dyn_cast<StringAttr>((*names)[i]);
    auto kind = dyn_cast<StringAttr>((*kinds)[i]);
    auto rank = dyn_cast<IntegerAttr>((*ranks)[i]);
    auto typeVar = dyn_cast<StringAttr>((*typeVars)[i]);
    if (!name || !kind || !rank || !typeVar) {
      return std::nullopt;
    }
    args.push_back(
        SchemaArg{name.getValue(), kind.getValue(), rank.getInt(),
                  typeVar.getValue()});
  }
  return args;
}

std::optional<int64_t> readI64AttrArg(DictionaryAttr metadata, StringRef name) {
  std::string attrName = "custom.attr.";
  attrName += name.str();
  auto attr = dyn_cast_or_null<IntegerAttr>(metadata.get(attrName));
  if (!attr) {
    return std::nullopt;
  }
  return attr.getInt();
}

std::optional<std::pair<ArrayRef<int64_t>, Type>>
getShapeAndElementType(Type type) {
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return std::make_pair(memrefType.getShape(), memrefType.getElementType());
  }
  if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
    return std::make_pair(rankedTensorType.getShape(),
                          rankedTensorType.getElementType());
  }
  if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
    return std::make_pair(tensorType.getShape(), tensorType.getElementType());
  }
  return std::nullopt;
}

MemRefType getCanonicalMemRefType(MLIRContext *ctx, Type type, int64_t rank) {
  auto shapeAndElement = getShapeAndElementType(type);
  if (!shapeAndElement || static_cast<int64_t>(shapeAndElement->first.size()) != rank) {
    return {};
  }

  SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  SmallVector<int64_t> strides(rank, ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
  return MemRefType::get(shape, shapeAndElement->second, layout);
}

bool isStringMemRefArg(StringRef opType, const SchemaArg &schemaArg) {
  return opType == "KPFusedDnnEmbeddingWithHashBucket" &&
         schemaArg.name == "input" && schemaArg.typeVar.empty();
}

MemRefType getCanonicalAbiMemRefType(MLIRContext *ctx, Type type,
                                     const SchemaArg &schemaArg,
                                     StringRef opType) {
  auto canonicalType = getCanonicalMemRefType(ctx, type, schemaArg.rank);
  if (!canonicalType || !isStringMemRefArg(opType, schemaArg)) {
    return canonicalType;
  }

  SmallVector<int64_t> shape(schemaArg.rank, ShapedType::kDynamic);
  SmallVector<int64_t> strides(schemaArg.rank, ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
  return MemRefType::get(shape, IntegerType::get(ctx, 64), layout);
}

std::optional<annc::kernels::TypeConstraintInfo>
bindTypeVar(StringRef typeVar, Type type) {
  auto constraint = inferTypeConstraint(typeVar, type);
  if (!constraint || constraint->cpp_type_name.empty()) {
    return std::nullopt;
  }
  return constraint;
}

LogicalResult bindOrCheckTypeVar(
    CustomizeOp op, SmallVectorImpl<annc::kernels::TypeConstraintInfo> &bindings,
    StringRef typeVar, Type type) {
  if (typeVar.empty()) {
    return success();
  }
  auto constraint = bindTypeVar(typeVar, type);
  if (!constraint) {
    return op.emitError() << "unsupported element type for type variable '"
                          << typeVar << "'";
  }
  auto existing = llvm::find_if(
      bindings, [&](const annc::kernels::TypeConstraintInfo &info) {
        return info.name == constraint->name;
      });
  if (existing == bindings.end()) {
    bindings.push_back(*constraint);
    return success();
  }
  if (existing->cpp_type_name != constraint->cpp_type_name) {
    return op.emitError() << "type variable '" << typeVar
                          << "' is bound to both "
                          << existing->cpp_type_name << " and "
                          << constraint->cpp_type_name;
  }
  return success();
}

} // namespace

void lowerCustomizeOpToFuncCall(PatternRewriter &rewriter,
                                CustomizeOpAdaptor adaptor,
                                CustomizeOp op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto metadata = op->getAttrOfType<DictionaryAttr>("metadata");
  auto schemaArgs = readSchemaArgs(metadata);
  if (!schemaArgs) {
    op.emitError() << "CustomizeOp '" << op.getOpType()
                   << "' is missing custom op schema metadata";
    return;
  }

  SmallVector<annc::kernels::TypeConstraintInfo> typeBindings;
  SmallVector<Value> callOperands;
  SmallVector<Type> callOperandTypes;
  SmallVector<Value> outputOperands;

  auto operands = adaptor.getOperands();
  size_t operandIndex = 0;
  for (const SchemaArg &schemaArg : *schemaArgs) {
    if (schemaArg.kind == "i64_attr") {
      auto value = readI64AttrArg(metadata, schemaArg.name);
      if (!value) {
        op.emitError() << "CustomizeOp '" << op.getOpType()
                       << "' is missing i64 metadata argument '"
                       << schemaArg.name << "'";
        return;
      }
      auto constant = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), *value, 64);
      callOperands.push_back(constant.getResult());
      callOperandTypes.push_back(rewriter.getI64Type());
      continue;
    }

    if (schemaArg.kind != "memref") {
      op.emitError() << "CustomizeOp '" << op.getOpType()
                     << "' argument " << schemaArg.name
                     << " uses unsupported ABI kind '" << schemaArg.kind << "'";
      return;
    }
    if (operandIndex >= operands.size()) {
      op.emitError() << "CustomizeOp '" << op.getOpType()
                     << "' has fewer operands than memref schema arguments";
      return;
    }
    Value operand = operands[operandIndex];
    if (schemaArg.rank < 0) {
      op.emitError() << "dynamic-rank CustomizeOp ABI is not supported yet for op '"
                     << op.getOpType() << "'";
      return;
    }

    auto canonicalType = getCanonicalAbiMemRefType(
        rewriter.getContext(), operand.getType(), schemaArg, op.getOpType());
    if (!canonicalType) {
      op.emitError() << "CustomizeOp '" << op.getOpType() << "' argument "
                     << schemaArg.name << " expects rank " << schemaArg.rank
                     << " memref, got " << operand.getType();
      return;
    }
    if (failed(bindOrCheckTypeVar(op, typeBindings, schemaArg.typeVar,
                                  operand.getType()))) {
      return;
    }

    Value callOperand = operand;
    if (operand.getType() != canonicalType) {
      callOperand = rewriter.create<memref::CastOp>(op.getLoc(), canonicalType, operand);
    }
    callOperands.push_back(callOperand);
    callOperandTypes.push_back(canonicalType);
    if (schemaArg.name.str() == "output") {
      outputOperands.push_back(operand);
    }
    ++operandIndex;
  }

  if (operandIndex != operands.size()) {
    op.emitError() << "CustomizeOp '" << op.getOpType()
                   << "' has more operands than memref schema arguments";
    return;
  }

  annc::kernels::KernelResolveRequest req;
  req.op_type = op.getOpType().str();
  req.type_constraints.assign(typeBindings.begin(), typeBindings.end());
  if (auto rhsFormat = op->getAttrOfType<StringAttr>("rhs_format")) {
    req.rhs_format = rhsFormat.getValue().str();
  }
  auto attr = module->getAttrOfType<BoolAttr>("annc.enable_kdnn");
  bool enableKdnn = attr && attr.getValue();
  auto kernelInfo = annc::kernels::resolveBestKernelInfo(req, enableKdnn);
  if (!kernelInfo.has_value()) {
    op.emitError() << "failed to resolve kernel symbol for CustomizeOp '"
                   << op.getOpType() << "'";
    return;
  }

  // if (outputOperands.size() != op->getNumResults()) {
  //   op.emitError() << "CustomizeOp '" << op.getOpType()
  //                  << "' expects one schema output memref per result, got "
  //                  << outputOperands.size() << " output operands for "
  //                  << op->getNumResults() << " results";
  //   return;
  // }

  auto funcType = rewriter.getFunctionType(callOperandTypes, TypeRange{});

  PatternRewriter::InsertionGuard guard(rewriter);

  auto funcOp = module.lookupSymbol<func::FuncOp>(kernelInfo->symbol_name);
  if (!funcOp) {
    rewriter.setInsertionPointToStart(module.getBody());
    funcOp = rewriter.create<func::FuncOp>(op.getLoc(), kernelInfo->symbol_name,
                                           funcType);
    funcOp.setPrivate();
    funcOp->setAttr("llvm.emit_c_interface",
                    UnitAttr::get(rewriter.getContext()));
  } else if (funcOp.getFunctionType() != funcType) {
    op.emitError() << "resolved kernel callee '" << kernelInfo->symbol_name
                   << "' already exists with mismatched function type";
    return;
  }

  rewriter.setInsertionPoint(op);

  rewriter.create<func::CallOp>(
      op.getLoc(), kernelInfo->symbol_name, TypeRange{}, callOperands);

  SmallVector<Value> replacements;
  for (auto [result, output] : llvm::zip(op->getResults(), outputOperands)) {
    (void)result;
    replacements.push_back(output);
  }
  rewriter.replaceOp(op, replacements);
}

} // namespace atir
