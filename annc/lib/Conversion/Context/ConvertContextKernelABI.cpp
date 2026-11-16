#include "Conversion/Passes.h"
#include "Support/ContextABISwitch.h"
#include "Dialect/Atir/AtirOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace atir {
namespace {

MemRefType getBoundaryMemRefType(Type type) {
  ArrayRef<int64_t> shape;
  Type elementType;
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (!shaped.hasRank()) return {};
    shape = shaped.getShape();
    elementType = shaped.getElementType();
  } else if (auto tensor = dyn_cast<atir::TensorType>(type)) {
    if (!tensor.hasKnownRank()) return {};
    shape = tensor.getShape();
    elementType = tensor.getElementType();
  } else {
    return {};
  }
  return MemRefType::get(shape, elementType);
}

Value loadContextMemRef(OpBuilder &builder, Location loc, Value context,
                        int64_t slot, MemRefType memrefType,
                        LLVMTypeConverter &converter) {
  Value slotIndex = builder.create<LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(slot));
  Value descriptorPointer = builder.create<func::CallOp>(
      loc, "annc_context_get_input_memref",
      TypeRange{LLVM::LLVMPointerType::get(builder.getContext())},
      ValueRange{context, slotIndex}).getResult(0);
  Type llvmDescriptorType = converter.convertType(memrefType);
  Value descriptor =
      builder.create<LLVM::LoadOp>(loc, llvmDescriptorType, descriptorPointer);
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{memrefType},
                                          ValueRange{descriptor})
      .getResult(0);
}

LogicalResult convertEntry(func::FuncOp function,
                           LLVMTypeConverter &converter) {
  auto metadata = function->getAttrOfType<DictionaryAttr>("fusion.metadata");
  auto abi = metadata ? metadata.getAs<StringAttr>("abi") : StringAttr{};
  if (!abi || abi.getValue() != "annc_context_v3") return success();
  if (!function->hasAttr("annc.kernel") || function.getBody().empty())
    return function.emitError("context ABI entry must have a function body");

  Block &entry = function.front();
  SmallVector<BlockArgument> oldArguments(entry.getArguments());
  auto oldResults = function.getFunctionType().getResults();
  auto ptr = LLVM::LLVMPointerType::get(function.getContext());
  Location loc = function.getLoc();
  entry.insertArgument(0u, ptr, loc);
  Value context = entry.getArgument(0);

  OpBuilder builder = OpBuilder::atBlockBegin(&entry);
  if (!oldArguments.empty()) {
    ModuleOp module = function->getParentOfType<ModuleOp>();
    auto declaration =
        module.lookupSymbol<func::FuncOp>("annc_context_get_input_memref");
    auto signature = FunctionType::get(function.getContext(),
                                       {ptr, builder.getI32Type()}, {ptr});
    if (declaration && declaration.getFunctionType() != signature)
      return function.emitError("input accessor declaration has wrong type");
    if (!declaration) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto accessor = builder.create<func::FuncOp>(
          loc, "annc_context_get_input_memref", signature);
      accessor.setPrivate();
    }
  }
  for (auto [index, argument] : llvm::enumerate(oldArguments)) {
    MemRefType memrefType =
        getBoundaryMemRefType(argument.getType());
    if (!memrefType)
      return function.emitError("context ABI arguments must be ranked tensors");
    Value memref = loadContextMemRef(builder, loc, context, index,
                                     memrefType, converter);
    auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                            memrefType.getElementType());
    Value tensor = builder.create<bufferization::ToTensorOp>(
        loc, tensorType, memref, true, true);
    Value replacement =
        builder
            .create<UnrealizedConversionCastOp>(
                loc, TypeRange{argument.getType()}, ValueRange{tensor})
            .getResult(0);
    argument.replaceAllUsesWith(replacement);
  }

  function.setType(FunctionType::get(function.getContext(), {ptr}, oldResults));
  for (int64_t i = static_cast<int64_t>(oldArguments.size()) - 1; i >= 0; --i)
    entry.eraseArgument(i + 1);
  function->removeAttr("arg_attrs");
  return success();
}

class ConvertContextKernelABIPass
    : public ConvertContextKernelABIBase<ConvertContextKernelABIPass> {
 public:
  void runOnOperation() override {
    if (!annc::loweringContextABIEnabled()) return;
    LLVMTypeConverter converter(&getContext());
    for (auto function : getOperation().getOps<func::FuncOp>())
      if (failed(convertEntry(function, converter))) {
        signalPassFailure();
        return;
      }
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertContextKernelABIPass() {
  return std::make_unique<ConvertContextKernelABIPass>();
}

}  // namespace atir
