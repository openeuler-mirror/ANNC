#include "Conversion/Common/CustomizeCallLowering.h"
#include "Conversion/Common/ContextInputOverrides.h"

#include "Dialect/Atir/CustomOpSchema.h"
#include "Kernel/KernelPriorityResolver.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace atir {
namespace {

struct SchemaArg {
  StringRef name;
  StringRef kind;
  int64_t rank = -1;
  StringRef typeVar;
};

struct SchemaResult {
  StringRef name;
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

std::optional<SmallVector<SchemaArg>> readSchemaArgs(DictionaryAttr metadata) {
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
    args.push_back(SchemaArg{name.getValue(), kind.getValue(), rank.getInt(),
                             typeVar.getValue()});
  }
  return args;
}

std::optional<SmallVector<SchemaResult>> readSchemaResults(
    DictionaryAttr metadata) {
  auto names = getArrayAttr(metadata, "custom.result_names");
  auto ranks = getArrayAttr(metadata, "custom.result_ranks");
  auto typeVars = getArrayAttr(metadata, "custom.result_type_vars");
  if (!names || !ranks || !typeVars || names->size() != ranks->size() ||
      ranks->size() != typeVars->size()) {
    return std::nullopt;
  }

  SmallVector<SchemaResult> results;
  results.reserve(names->size());
  for (size_t i = 0; i < names->size(); ++i) {
    auto name = dyn_cast<StringAttr>((*names)[i]);
    auto rank = dyn_cast<IntegerAttr>((*ranks)[i]);
    auto typeVar = dyn_cast<StringAttr>((*typeVars)[i]);
    if (!name || !rank || !typeVar) return std::nullopt;
    results.push_back(
        SchemaResult{name.getValue(), rank.getInt(), typeVar.getValue()});
  }
  return results;
}

std::optional<int64_t> readI64AttrAt(DictionaryAttr metadata, size_t index) {
  auto values = getArrayAttr(metadata, "custom.attr_values");
  if (!values || index >= values->size()) return std::nullopt;
  auto value = dyn_cast<IntegerAttr>((*values)[index]);
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

std::optional<std::pair<ArrayRef<int64_t>, Type>> getShapeAndElementType(
    Type type) {
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
  if (!shapeAndElement ||
      static_cast<int64_t>(shapeAndElement->first.size()) != rank) {
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

Value createCustomizeContext(PatternRewriter &rewriter, CustomizeOp op,
                             Value context) {
  auto operands = op.getOperands();
  SmallVector<DenseElementsAttr> overrides(operands.size());
  std::optional<size_t> effectiveWeightSlot;
  for (auto [index, operand] : llvm::enumerate(operands)) {
    Operation *constant = operand.getDefiningOp();
    if (!constant || (!isa<ConstantOp>(constant) &&
                      !constant->hasTrait<OpTrait::ConstantLike>()))
      continue;
    DenseElementsAttr payload;
    if (!matchPattern(operand, m_Constant(&payload))) {
      // Legacy ATIR/ONNX entries still share this context ABI lowering.
      if (auto tensorType = dyn_cast<atir::TensorType>(operand.getType()))
        payload = dyn_cast_or_null<DenseElementsAttr>(tensorType.getCacheData());
    }
    if (!payload || !payload.getType().hasStaticShape() ||
        !payload.getElementType().isF32()) {
      op.emitError("context constant inputs must be dense f32 tensors");
      return {};
    }
    overrides[index] = payload;
    if (constant->hasAttr("annc.effective_weight"))
      effectiveWeightSlot = index;
  }
  if (llvm::none_of(overrides, [](DenseElementsAttr value) {
        return static_cast<bool>(value);
      }))
    return context;

  auto parent = op->getParentOfType<func::FuncOp>();
  std::string prefix = parent.getName().str();
  Value address = createContextInputOverrides(
      rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(), prefix, context,
      overrides);
  if (effectiveWeightSlot) {
    parent->setAttr(
        "annc.effective_weight_global",
        rewriter.getStringAttr(prefix + "_context_input_" +
                               std::to_string(*effectiveWeightSlot) + "_data"));
  }
  return address;
}

std::optional<annc::kernels::TypeConstraintInfo> bindTypeVar(StringRef typeVar,
                                                             Type type) {
  auto constraint = inferTypeConstraint(typeVar, type);
  if (!constraint || constraint->cpp_type_name.empty()) {
    return std::nullopt;
  }
  return constraint;
}

LogicalResult bindOrCheckTypeVar(
    CustomizeOp op,
    SmallVectorImpl<annc::kernels::TypeConstraintInfo> &bindings,
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
                          << "' is bound to both " << existing->cpp_type_name
                          << " and " << constraint->cpp_type_name;
  }
  return success();
}

std::optional<int64_t> boundaryInputSlot(Value value, Block &entry) {
  while (Operation *op = value.getDefiningOp()) {
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(op)) {
      if (cast.getInputs().size() != 1) return std::nullopt;
      value = cast.getInputs().front();
    } else if (auto tensor = dyn_cast<bufferization::ToTensorOp>(op)) {
      value = tensor.getOperand();
    } else if (isa<ConstantOp>(op) ||
               op->hasTrait<OpTrait::ConstantLike>()) {
      return -1;
    } else if (auto accessor = dyn_cast<func::CallOp>(op)) {
      if (accessor.getCallee() != "annc_context_get_input_memref" ||
          accessor.getNumOperands() != 2 ||
          accessor.getOperand(0) != entry.getArgument(0))
        return std::nullopt;
      auto slot = accessor.getOperand(1).getDefiningOp<LLVM::ConstantOp>();
      if (slot)
        if (auto index = dyn_cast<IntegerAttr>(slot.getValue()))
          return index.getInt();
      return std::nullopt;
    } else if (isa<LLVM::LoadOp>(op)) {
      value = op->getOperand(0);
    } else {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace

LogicalResult lowerCustomizeOpToContextCall(PatternRewriter &rewriter,
                                           CustomizeOp op) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!module || !parentFunc) {
    op.emitError() << "CustomizeOp must be nested in a function and module";
    return failure();
  }

  auto fusionMetadata =
      parentFunc->getAttrOfType<DictionaryAttr>("fusion.metadata");
  auto abi = fusionMetadata
                 ? fusionMetadata.getAs<StringAttr>("abi")
                 : StringAttr{};
  if (!abi || abi.getValue() != "annc_context_v3") {
    op.emitError("CustomizeOp lowering requires annc_context_v3 ABI");
    return failure();
  }
  if (parentFunc.getNumArguments() != 1 ||
      !isa<LLVM::LLVMPointerType>(parentFunc.getArgument(0).getType())) {
    op.emitError("context ABI conversion must provide one !llvm.ptr argument");
    return failure();
  }
  for (Operation &operation : parentFunc.front()) {
    if (&operation == op.getOperation() ||
        operation.hasTrait<OpTrait::ConstantLike>() ||
        isa<func::ReturnOp, ConstantOp, LLVM::ConstantOp, LLVM::LoadOp,
            UnrealizedConversionCastOp, bufferization::ToTensorOp>(operation) ||
        (isa<BufferOp>(operation) && operation.use_empty()))
      continue;
    if (auto call = dyn_cast<func::CallOp>(operation);
        call && call.getCallee() == "annc_context_get_input_memref")
      continue;
    op.emitError("context Customize entry must contain only input unpacking, "
                 "constants, one CustomizeOp, and return");
    return failure();
  }
  auto returnOp = dyn_cast<func::ReturnOp>(op->getBlock()->getTerminator());
  if (parentFunc.getBody().getBlocks().size() != 1 || !returnOp ||
      returnOp.getNumOperands() != op.getNumResults() ||
      !llvm::equal(returnOp.getOperands(), op.getResults()) ||
      !llvm::all_of(op->getResults(), [](Value result) {
        return llvm::all_of(result.getUses(), [](OpOperand &use) {
          return isa<func::ReturnOp>(use.getOwner());
        });
      })) {
    op.emitError("context Customize result must directly feed the entry return");
    return failure();
  }
  for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
    auto slot = boundaryInputSlot(operand, parentFunc.front());
    if (!slot || (*slot >= 0 && *slot != static_cast<int64_t>(index))) {
      op.emitError("context Customize operands must match entry input slots "
                   "or constants");
      return failure();
    }
  }

  auto metadata = op->getAttrOfType<DictionaryAttr>("metadata");
  auto schemaArgs = readSchemaArgs(metadata);
  if (!schemaArgs) {
    op.emitError() << "CustomizeOp '" << op.getOpType()
                   << "' is missing custom op schema metadata";
    return failure();
  }

  SmallVector<annc::kernels::TypeConstraintInfo> typeBindings;
  auto operands = op.getOperands();

  size_t operandIndex = 0;
  size_t attributeIndex = 0;
  for (const SchemaArg &schemaArg : *schemaArgs) {
    if (schemaArg.kind == "i64_attr") {
      auto value = readI64AttrAt(metadata, attributeIndex++);
      if (!value) {
        op.emitError() << "CustomizeOp '" << op.getOpType()
                       << "' is missing i64 metadata argument '"
                       << schemaArg.name << "'";
        return failure();
      }
      (void)value;
      continue;
    }

    if (schemaArg.kind != "memref") {
      op.emitError() << "CustomizeOp '" << op.getOpType() << "' argument "
                     << schemaArg.name << " uses unsupported ABI kind '"
                     << schemaArg.kind << "'";
      return failure();
    }
    if (operandIndex >= operands.size()) {
      op.emitError() << "CustomizeOp '" << op.getOpType()
                     << "' has fewer operands than memref schema arguments";
      return failure();
    }
    Value operand = operands[operandIndex];
    if (schemaArg.rank < 0) {
      op.emitError()
          << "dynamic-rank CustomizeOp ABI is not supported yet for op '"
          << op.getOpType() << "'";
      return failure();
    }

    auto canonicalType = getCanonicalAbiMemRefType(
        rewriter.getContext(), operand.getType(), schemaArg, op.getOpType());
    if (!canonicalType) {
      op.emitError() << "CustomizeOp '" << op.getOpType() << "' argument "
                     << schemaArg.name << " expects rank " << schemaArg.rank
                     << " memref, got " << operand.getType();
      return failure();
    }
    if (failed(bindOrCheckTypeVar(op, typeBindings, schemaArg.typeVar,
                                  operand.getType()))) {
      return failure();
    }

    (void)canonicalType;
    ++operandIndex;
  }

  if (operandIndex != operands.size()) {
    op.emitError() << "CustomizeOp '" << op.getOpType()
                   << "' has more operands than memref schema arguments";
    return failure();
  }

  auto schemaResults = readSchemaResults(metadata);
  if (!schemaResults || schemaResults->size() != op.getNumResults()) {
    op.emitError() << "CustomizeOp '" << op.getOpType()
                   << "' result schema does not match its results";
    return failure();
  }
  for (auto [result, schemaResult] : llvm::zip(op->getResults(), *schemaResults)) {
    auto shapeAndElement = getShapeAndElementType(result.getType());
    if (!shapeAndElement || schemaResult.rank < 0 ||
        static_cast<int64_t>(shapeAndElement->first.size()) != schemaResult.rank) {
      op.emitError() << "CustomizeOp '" << op.getOpType() << "' result "
                     << schemaResult.name << " expects rank " << schemaResult.rank
                     << ", got " << result.getType();
      return failure();
    }
    if (failed(bindOrCheckTypeVar(op, typeBindings, schemaResult.typeVar,
                                  result.getType())))
      return failure();
  }

  annc::kernels::KernelResolveRequest req;
  req.op_type = op.getOpType().str();
  req.type_constraints.assign(typeBindings.begin(), typeBindings.end());
  req.abi = "annc_context_v3";
  if (auto rhsFormat = op->getAttrOfType<StringAttr>("rhs_format")) {
    req.rhs_format = rhsFormat.getValue().str();
  }
  auto attr = module->getAttrOfType<BoolAttr>("annc.enable_kdnn");
  bool enableKdnn = attr && attr.getValue();
  auto kernelInfo = annc::kernels::resolveBestKernelInfo(req, enableKdnn);
  if (!kernelInfo.has_value()) {
    op.emitError() << "failed to resolve kernel symbol for CustomizeOp '"
                   << op.getOpType() << "'";
    return failure();
  }

  auto statusType = rewriter.getI32Type();
  Value context = createCustomizeContext(rewriter, op, parentFunc.getArgument(0));
  if (!context) return failure();
  auto funcType = rewriter.getFunctionType(TypeRange{context.getType()},
                                            TypeRange{statusType});

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
    return failure();
  }

  rewriter.setInsertionPoint(op);

  auto callOp =
      rewriter.create<func::CallOp>(op.getLoc(), kernelInfo->symbol_name,
                                    TypeRange{statusType}, ValueRange{context});
  Value status = callOp.getResult(0);
  returnOp->setOperands(status);
  parentFunc.setType(rewriter.getFunctionType(
      parentFunc.getFunctionType().getInputs(), TypeRange{statusType}));
  rewriter.eraseOp(op);

  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> operations;
    parentFunc.walk([&](Operation *operation) { operations.push_back(operation); });
    for (Operation *operation : llvm::reverse(operations)) {
      auto inputCall = dyn_cast<func::CallOp>(operation);
      bool unusedInputAccessor =
          inputCall && inputCall->use_empty() &&
          inputCall.getCallee() == "annc_context_get_input_memref";
      if (operation == parentFunc.getOperation() ||
          (!unusedInputAccessor && !isOpTriviallyDead(operation)))
        continue;
      rewriter.eraseOp(operation);
      changed = true;
    }
  } while (changed);
  auto inputAccessor =
      module.lookupSymbol<func::FuncOp>("annc_context_get_input_memref");
  if (inputAccessor && inputAccessor.isDeclaration() &&
      SymbolTable::symbolKnownUseEmpty(inputAccessor, module))
    rewriter.eraseOp(inputAccessor);
  return success();
}

}  // namespace atir
