#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/AtirOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace atir {
namespace {

ArrayAttr makeStringArray(MLIRContext *ctx, ArrayRef<std::string> values) {
  SmallVector<Attribute> attrs;
  for (const auto &value : values) {
    attrs.push_back(StringAttr::get(ctx, value));
  }
  return ArrayAttr::get(ctx, attrs);
}

ArrayAttr makeI64Array(MLIRContext *ctx, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  Builder builder(ctx);
  for (int64_t value : values) {
    attrs.push_back(builder.getI64IntegerAttr(value));
  }
  return ArrayAttr::get(ctx, attrs);
}

std::string kindToString(CustomOpSchema::ArgKind kind) {
  switch (kind) {
  case CustomOpSchema::ArgKind::MemRef:
    return "memref";
  case CustomOpSchema::ArgKind::Scalar:
    return "scalar";
  case CustomOpSchema::ArgKind::I64Attr:
    return "i64_attr";
  case CustomOpSchema::ArgKind::Opaque:
    return "opaque";
  case CustomOpSchema::ArgKind::Custom:
    return "custom";
  }
  return "custom";
}

std::string cppTypeNameFromElementType(Type type) {
  if (type.isF32()) {
    return "float";
  }
  if (type.isF64()) {
    return "double";
  }
  if (type.isInteger(1)) {
    return "bool";
  }
  if (type.isInteger(8)) {
    return "int8_t";
  }
  if (type.isInteger(16)) {
    return "int16_t";
  }
  if (type.isInteger(32)) {
    return "int32_t";
  }
  if (type.isInteger(64)) {
    return "int64_t";
  }
  return "";
}

std::optional<ArrayAttr> getArrayAttr(DictionaryAttr metadata,
                                      StringRef name) {
  if (!metadata) {
    return std::nullopt;
  }
  if (auto attr = dyn_cast_or_null<ArrayAttr>(metadata.get(name))) {
    return attr;
  }
  return std::nullopt;
}

} // namespace

DictionaryAttr CustomOpSchema::toMetadata(MLIRContext *ctx) const {
  SmallVector<NamedAttribute> attrs;
  Builder builder(ctx);
  SmallVector<std::string> argNames;
  SmallVector<std::string> argKinds;
  SmallVector<int64_t> argRanks;
  SmallVector<std::string> argTypeVars;

  for (const Arg &arg : args_) {
    argNames.push_back(arg.name);
    argKinds.push_back(kindToString(arg.kind));
    argRanks.push_back(arg.rank);
    argTypeVars.push_back(arg.typeVar);
    if (arg.kind == ArgKind::I64Attr) {
      std::string attrName = "custom.attr." + arg.name;
      attrs.push_back(builder.getNamedAttr(
          attrName, builder.getI64IntegerAttr(arg.i64Value)));
    }
  }

  attrs.push_back(builder.getNamedAttr("custom.op_name",
                                       builder.getStringAttr(name_)));
  attrs.push_back(builder.getNamedAttr("custom.type_vars",
                                       makeStringArray(ctx, typeVars_)));
  attrs.push_back(builder.getNamedAttr("custom.arg_names",
                                       makeStringArray(ctx, argNames)));
  attrs.push_back(builder.getNamedAttr("custom.arg_kinds",
                                       makeStringArray(ctx, argKinds)));
  attrs.push_back(builder.getNamedAttr("custom.arg_ranks",
                                       makeI64Array(ctx, argRanks)));
  attrs.push_back(builder.getNamedAttr("custom.arg_type_vars",
                                       makeStringArray(ctx, argTypeVars)));
  return DictionaryAttr::get(ctx, attrs);
}

std::optional<annc::kernels::TypeConstraintInfo>
inferTypeConstraint(StringRef typeVar, Type type) {
  if (typeVar.empty()) {
    return std::nullopt;
  }

  Type elementType = type;
  if (auto atirTensorType = dyn_cast<atir::TensorType>(type)) {
    elementType = atirTensorType.getElementType();
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    elementType = shapedType.getElementType();
  }

  std::string cppType = cppTypeNameFromElementType(elementType);
  if (cppType.empty()) {
    return std::nullopt;
  }

  return annc::kernels::TypeConstraintInfo{typeVar.str(), cppType};
}

std::vector<annc::kernels::TypeConstraintInfo>
inferTypeConstraintsFromSchema(DictionaryAttr metadata, ValueRange operands) {
  std::vector<annc::kernels::TypeConstraintInfo> constraints;
  auto kindAttrs = getArrayAttr(metadata, "custom.arg_kinds");
  auto typeVarAttrs = getArrayAttr(metadata, "custom.arg_type_vars");
  if (!kindAttrs || !typeVarAttrs || kindAttrs->size() != typeVarAttrs->size()) {
    return constraints;
  }

  size_t count = std::min<size_t>(operands.size(), typeVarAttrs->size());
  for (size_t i = 0; i < count; ++i) {
    auto kind = dyn_cast<StringAttr>((*kindAttrs)[i]);
    auto typeVar = dyn_cast<StringAttr>((*typeVarAttrs)[i]);
    if (!kind || !typeVar || kind.getValue() != "memref") {
      continue;
    }

    auto constraint = inferTypeConstraint(typeVar.getValue(), operands[i].getType());
    if (!constraint) {
      continue;
    }

    auto existing = llvm::find_if(
        constraints, [&](const annc::kernels::TypeConstraintInfo &info) {
          return info.name == constraint->name;
        });
    if (existing == constraints.end()) {
      constraints.push_back(*constraint);
    }
  }

  return constraints;
}

} // namespace atir
