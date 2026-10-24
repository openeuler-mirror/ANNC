#include "Conversion/Common/ContextInputOverrides.h"
#include "Conversion/Passes.h"
#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Kernel/KernelPriorityResolver.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace atir {
namespace {

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

LogicalResult validateContextBoundaryMetadata(CustomizeOp custom,
                                              DictionaryAttr metadata,
                                              size_t numInputs,
                                              size_t numOutputs) {
  if (Attribute order = metadata.get("kernel_arg_order")) {
    auto array = dyn_cast<ArrayAttr>(order);
    if (!array || !array.empty())
      return custom.emitError(
          "context metadata must not reorder kernel arguments");
  }
  // Minimal lowering fixtures have no persisted boundary records. In that case
  // the exact CustomizeOp schema and SSA boundary remain the source of truth.
  if (!metadata.get("args") && !metadata.get("outputs")) return success();
  auto inputs = dyn_cast_or_null<ArrayAttr>(metadata.get("args"));
  auto outputs = dyn_cast_or_null<ArrayAttr>(metadata.get("outputs"));
  if (!inputs || !outputs || inputs.size() != numInputs ||
      outputs.size() != numOutputs)
    return custom.emitError("context fusion metadata does not match boundary");
  auto validate = [&](Attribute attr, Value value,
                      bool output) -> LogicalResult {
    auto record = dyn_cast<DictionaryAttr>(attr);
    if (!record)
      return custom.emitError(
          "context fusion boundary record must be a dictionary");
    auto rank = dyn_cast_or_null<IntegerAttr>(record.get("rank"));
    auto dtype = dyn_cast_or_null<StringAttr>(record.get("dtype"));
    auto shape = dyn_cast_or_null<ArrayAttr>(record.get("shape"));
    auto role = dyn_cast_or_null<StringAttr>(record.get("role"));
    auto tfName = dyn_cast_or_null<StringAttr>(record.get("tf_name"));
    auto actual = getShapeAndElementType(value.getType());
    if (!actual || !rank || !dtype || !shape || !role || !tfName ||
        tfName.empty() ||
        rank.getInt() != static_cast<int64_t>(actual->first.size()) ||
        shape.size() != actual->first.size())
      return custom.emitError(
          "context fusion metadata has incompatible rank or shape");
    if ((output && role.getValue() != "output") ||
        (!output && role.getValue() != "constant" &&
         role.getValue() != "fixed" && role.getValue() != "dynamic"))
      return custom.emitError(
          "context fusion metadata has invalid boundary role");
    std::string actualDtype;
    llvm::raw_string_ostream stream(actualDtype);
    actual->second.print(stream);
    if (auto tensor = dyn_cast<atir::TensorType>(value.getType())) {
      if (auto encoding = dyn_cast_or_null<StringAttr>(tensor.getEncoding());
          encoding && !encoding.empty())
        actualDtype = encoding.str();
    }
    if (dtype.getValue() != actualDtype)
      return custom.emitError(
          "context fusion metadata dtype does not match boundary");
    for (auto [attr, dim] : llvm::zip(shape, actual->first)) {
      auto declared = dyn_cast<IntegerAttr>(attr);
      if (!declared || (declared.getInt() >= 0 && !ShapedType::isDynamic(dim) &&
                        declared.getInt() != dim))
        return custom.emitError(
            "context fusion metadata shape does not match boundary");
    }
    return success();
  };
  for (auto [attr, value] : llvm::zip_equal(inputs, custom.getOperands()))
    if (failed(validate(attr, value, false))) return failure();
  for (auto [attr, value] : llvm::zip_equal(outputs, custom.getResults()))
    if (failed(validate(attr, value, true))) return failure();
  return success();
}

LogicalResult lowerContextKernelEntries(ModuleOp module) {
  // Forward the same boundary context to one builtin. Reject
  // compositions and reordered boundaries instead of silently reinterpreting
  // slots. Keeping this rewrite outside dialect conversion also avoids changing
  // a function signature while its argument materializations are being tracked.
  SmallVector<func::FuncOp> entries;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto metadata = function->getAttrOfType<DictionaryAttr>("fusion.metadata");
    auto abi = metadata ? dyn_cast_or_null<StringAttr>(metadata.get("abi"))
                        : StringAttr{};
    if (abi && abi.getValue() != "mlir_ciface" &&
        abi.getValue() != "annc_context_v3" &&
        abi.getValue() != "annc_execution_v2")
      return function.emitError()
             << "unsupported fusion ABI: " << abi.getValue();
    if (abi && abi.getValue() == "annc_context_v3") entries.push_back(function);
  }
  for (auto function : entries) {
    if (!function->hasAttr("annc.kernel") ||
        !llvm::hasSingleElement(function.getBody()))
      return function.emitError("context entry must be a single-block kernel");
    CustomizeOp custom;
    for (Operation &operation : function.front()) {
      if (auto candidate = dyn_cast<CustomizeOp>(operation)) {
        if (custom)
          return function.emitError(
              "context entry requires exactly one builtin");
        custom = candidate;
      } else if (!isa<func::ReturnOp>(operation) &&
                 !isa<ConstantOp>(operation) &&
                 !(isa<BufferOp>(operation) && operation.use_empty())) {
        return function.emitError(
            "context entry contains unsupported operations");
      }
    }
    auto boundaryReturn =
        dyn_cast<func::ReturnOp>(function.front().getTerminator());
    if (!custom || !boundaryReturn ||
        custom.getNumOperands() != function.getNumArguments() ||
        !llvm::equal(custom.getResults(), boundaryReturn.getOperands()))
      return function.emitError(
          "context entry requires builtin inputs and outputs in boundary "
          "order");

    // A pre-affine fold may replace a boundary initializer in the same slot.
    // Keep the external binding intact and override only this call's view.
    SmallVector<DenseElementsAttr> inputOverrides(custom.getNumOperands());
    std::optional<size_t> effectiveWeightSlot;
    for (auto [index, operand] : llvm::enumerate(custom.getOperands())) {
      if (operand == function.getArgument(index)) continue;
      auto constant = operand.getDefiningOp<ConstantOp>();
      auto tensor = dyn_cast<atir::TensorType>(operand.getType());
      auto payload =
          tensor ? dyn_cast_or_null<DenseElementsAttr>(tensor.getCacheData())
                 : DenseElementsAttr{};
      auto boundary =
          getShapeAndElementType(function.getArgument(index).getType());
      auto actual = getShapeAndElementType(operand.getType());
      if (!constant || !payload || !payload.getElementType().isF32() ||
          !payload.getType().hasStaticShape() || !boundary || !actual ||
          boundary->second != actual->second ||
          boundary->first != actual->first ||
          payload.getType().getShape() != actual->first ||
          !llvm::all_of(operand.getUses(), [&](OpOperand &use) {
            return use.getOwner() == custom.getOperation();
          }))
        return custom.emitError(
            "context inputs must follow boundary order or be dense f32 "
            "constants");
      inputOverrides[index] = payload;
      if (constant->hasAttr("annc.effective_weight"))
        effectiveWeightSlot = index;
    }

    auto schema = custom->getAttrOfType<DictionaryAttr>("metadata");
    SmallVector<int64_t> attributes;
    if (auto values = schema ? schema.getAs<ArrayAttr>("custom.attr_values")
                             : ArrayAttr{})
      for (auto attr : values) {
        auto value = dyn_cast<IntegerAttr>(attr);
        if (!value) return custom.emitError("invalid context attribute");
        attributes.push_back(value.getInt());
      }
    auto contextSchema = getContextKernelSchema(custom.getOpType(), attributes);
    auto expectedSchema = contextSchema
                              ? contextSchema->toMetadata(module.getContext())
                              : DictionaryAttr{};
    if (!expectedSchema || schema != expectedSchema)
      return custom.emitError(
          "context entry requires the exact shared builtin schema");
    SmallVector<CustomOpSchema::Arg> tensorArgs;
    SmallVector<int64_t> attributeValues;
    for (const auto &arg : contextSchema->args()) {
      if (arg.kind == CustomOpSchema::ArgKind::MemRef)
        tensorArgs.push_back(arg);
      else if (arg.kind == CustomOpSchema::ArgKind::I64Attr)
        attributeValues.push_back(arg.i64Value);
      else
        return custom.emitError("unsupported context argument kind");
    }
    auto metadata = function->getAttrOfType<DictionaryAttr>("fusion.metadata");
    auto declaredAttrs =
        dyn_cast_or_null<ArrayAttr>(metadata.get("context_i64_attrs"));
    if ((!attributeValues.empty() && !declaredAttrs) ||
        (declaredAttrs && declaredAttrs.size() != attributeValues.size()))
      return custom.emitError(
          "context attribute metadata does not match schema");
    for (size_t i = 0; i < attributeValues.size(); ++i) {
      auto value = dyn_cast<IntegerAttr>(declaredAttrs[i]);
      if (!value || value.getInt() != attributeValues[i])
        return custom.emitError(
            "context attribute value does not match schema");
    }
    auto results = contextSchema->results();
    if (tensorArgs.size() != custom.getNumOperands() ||
        results.size() != custom.getNumResults())
      return custom.emitError("context builtin schema does not match boundary");
    for (auto [value, arg] : llvm::zip(custom.getOperands(), tensorArgs)) {
      auto type = getShapeAndElementType(value.getType());
      if (!type || arg.rank < 0 ||
          static_cast<int64_t>(type->first.size()) != arg.rank)
        return custom.emitError(
            "context builtin has incompatible input schema");
    }
    for (auto [value, result] : llvm::zip(custom.getResults(), results)) {
      auto type = getShapeAndElementType(value.getType());
      if (!type || result.rank < 0 ||
          static_cast<int64_t>(type->first.size()) != result.rank)
        return custom.emitError(
            "context builtin has incompatible result schema");
    }
    if (failed(validateContextBoundaryMetadata(
            custom, function->getAttrOfType<DictionaryAttr>("fusion.metadata"),
            tensorArgs.size(), results.size())))
      return failure();
    auto bindings = inferTypeConstraintsFromSchema(
        schema, custom->getOperandTypes(), custom->getResultTypes());
    if (!bindings)
      return custom.emitError() << llvm::toString(bindings.takeError());
    annc::kernels::KernelResolveRequest request;
    request.op_type = custom.getOpType().str();
    request.abi = "annc_context_v3";
    if (auto rhsFormat = custom->getAttrOfType<StringAttr>("rhs_format"))
      request.rhs_format = rhsFormat.str();
    request.type_constraints = std::move(*bindings);
    auto enableKdnn = module->getAttrOfType<BoolAttr>("annc.enable_kdnn");
    auto kernel = annc::kernels::resolveBestKernelInfo(
        request, enableKdnn && enableKdnn.getValue());
    if (!kernel)
      return custom.emitError("no compatible context builtin registered");

    OpBuilder builder(module.getContext());
    auto pointer = LLVM::LLVMPointerType::get(module.getContext());
    auto status = builder.getI32Type();
    auto type = builder.getFunctionType(TypeRange{pointer}, TypeRange{status});
    auto callee = module.lookupSymbol<func::FuncOp>(kernel->symbol_name);
    if (callee && callee.getFunctionType() != type)
      return custom.emitError(
          "context builtin symbol has incompatible signature");
    if (!callee) {
      builder.setInsertionPointToStart(module.getBody());
      callee = builder.create<func::FuncOp>(custom.getLoc(),
                                            kernel->symbol_name, type);
      callee.setPrivate();
      callee->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    }
    // Context entries are exported artifacts, never called from the model IR.
    auto uses = SymbolTable::getSymbolUses(function, module);
    if (!uses || !uses->empty())
      return function.emitError(
          "context entry must not have model-level callers");
    Location loc = custom.getLoc();
    function.getBody().dropAllReferences();
    function.getBody().getBlocks().clear();
    function.setFunctionType(type);
    function->removeAttr("arg_attrs");
    function->removeAttr("res_attrs");
    Block *entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    Value callContext =
        createContextInputOverrides(builder, loc, module, function.getSymName(),
                                    entry->getArgument(0), inputOverrides);
    if (effectiveWeightSlot)
      function->setAttr("annc.effective_weight_global",
                        builder.getStringAttr(
                            function.getSymName().str() + "_context_input_" +
                            std::to_string(*effectiveWeightSlot) + "_data"));
    auto call = builder.create<func::CallOp>(
        loc, kernel->symbol_name, TypeRange{status}, ValueRange{callContext});
    builder.create<func::ReturnOp>(loc, call.getResults());
  }
  return success();
}

class LowerContextKernelEntriesPass
    : public LowerContextKernelEntriesBase<LowerContextKernelEntriesPass> {
 public:
  void runOnOperation() override {
    if (failed(lowerContextKernelEntries(getOperation()))) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createLowerContextKernelEntriesPass() {
  return std::make_unique<LowerContextKernelEntriesPass>();
}

}  // namespace atir
