#include "Conversion/Common/CustomizeCallLowering.h"

#include "Dialect/Atir/CustomOpSchema.h"
#include "Kernel/KernelPriorityResolver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace atir {
namespace {

struct SchemaArg {
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
readSchemaArgs(DictionaryAttr metadata, size_t operandCount) {
  auto kinds = getArrayAttr(metadata, "custom.arg_kinds");
  auto ranks = getArrayAttr(metadata, "custom.arg_ranks");
  auto typeVars = getArrayAttr(metadata, "custom.arg_type_vars");
  if (!kinds || !ranks || !typeVars) {
    return std::nullopt;
  }
  if (kinds->size() != ranks->size() || kinds->size() != typeVars->size() ||
      kinds->size() != operandCount) {
    return std::nullopt;
  }

  SmallVector<SchemaArg> args;
  args.reserve(kinds->size());
  for (size_t i = 0; i < kinds->size(); ++i) {
    auto kind = dyn_cast<StringAttr>((*kinds)[i]);
    auto rank = dyn_cast<IntegerAttr>((*ranks)[i]);
    auto typeVar = dyn_cast<StringAttr>((*typeVars)[i]);
    if (!kind || !rank || !typeVar) {
      return std::nullopt;
    }
    args.push_back(SchemaArg{kind.getValue(), rank.getInt(), typeVar.getValue()});
  }
  return args;
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

MemRefType getConcreteMemRefType(Type type) {
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return memrefType;
  }
  auto shapeAndElement = getShapeAndElementType(type);
  if (!shapeAndElement) {
    return {};
  }
  return MemRefType::get(shapeAndElement->first, shapeAndElement->second);
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
  auto schemaArgs = readSchemaArgs(metadata, adaptor.getOperands().size());
  if (!schemaArgs) {
    op.emitError() << "CustomizeOp '" << op.getOpType()
                   << "' is missing custom op schema metadata";
    return;
  }

  SmallVector<annc::kernels::TypeConstraintInfo> typeBindings;
  SmallVector<Value> callOperands;
  SmallVector<Type> callOperandTypes;

  auto operands = adaptor.getOperands();
  for (size_t index = 0; index < operands.size(); ++index) {
    Value operand = operands[index];
    const SchemaArg &schemaArg = (*schemaArgs)[index];
    if (schemaArg.kind != "memref") {
      op.emitError() << "CustomizeOp '" << op.getOpType()
                     << "' argument " << index
                     << " uses unsupported ABI kind '" << schemaArg.kind << "'";
      return;
    }
    if (schemaArg.rank < 0) {
      op.emitError() << "dynamic-rank CustomizeOp ABI is not supported yet for op '"
                     << op.getOpType() << "'";
      return;
    }

    auto canonicalType =
        getCanonicalMemRefType(rewriter.getContext(), operand.getType(), schemaArg.rank);
    if (!canonicalType) {
      op.emitError() << "CustomizeOp '" << op.getOpType() << "' argument "
                     << index << " expects rank " << schemaArg.rank
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
  }

  annc::kernels::KernelResolveRequest req;
  req.op_type = op.getOpType().str();
  req.type_constraints.assign(typeBindings.begin(), typeBindings.end());
  auto attr = module->getAttrOfType<BoolAttr>("annc.enable_kdnn");
  bool enableKdnn = attr && attr.getValue();
  auto kernelInfo = annc::kernels::resolveBestKernelInfo(req, enableKdnn);
  if (!kernelInfo.has_value()) {
    op.emitError() << "failed to resolve kernel symbol for CustomizeOp '"
                   << op.getOpType() << "'";
    return;
  }

  SmallVector<Type> concreteResultTypes;
  SmallVector<Type> callResultTypes;
  for (Type resultType : op->getResultTypes()) {
    auto concreteType = getConcreteMemRefType(resultType);
    if (!concreteType) {
      op.emitError() << "unsupported CustomizeOp result type " << resultType;
      return;
    }
    concreteResultTypes.push_back(concreteType);
    callResultTypes.push_back(
        getCanonicalMemRefType(rewriter.getContext(), concreteType,
                               concreteType.getRank()));
  }

  auto funcType = rewriter.getFunctionType(callOperandTypes, callResultTypes);

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

  auto callOp = rewriter.create<func::CallOp>(
      op.getLoc(), kernelInfo->symbol_name, callResultTypes, callOperands);

  SmallVector<Value> replacements;
  for (auto [result, concreteType] :
       llvm::zip(callOp.getResults(), concreteResultTypes)) {
    if (result.getType() == concreteType) {
      replacements.push_back(result);
      continue;
    }
    replacements.push_back(
        rewriter.create<memref::CastOp>(op.getLoc(), concreteType, result));
  }
  rewriter.replaceOp(op, replacements);
}

} // namespace atir
