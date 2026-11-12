#include "Support/LLVMCompat.h"
#include "Support/ContextABISwitch.h"
#include "Target/aarch64/Passes.h"
#include "Context/KernelContext.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace annc {
namespace {

FailureOr<Value> loadContextOutput(OpBuilder &builder, Location loc,
                                   ModuleOp module, Value context,
                                   unsigned slot, memref::AllocOp allocation,
                                   LLVMTypeConverter &converter) {
  auto type = allocation.getType();
  Type elementType = type.getElementType();
  if (!elementType.isF32() && !elementType.isInteger(32) &&
      !elementType.isInteger(64))
    return failure();
  if (type.getRank() > ANNC_CONTEXT_MAX_RANK) return failure();

  auto ptr = LLVM::LLVMPointerType::get(builder.getContext());
  auto i32 = builder.getI32Type();
  auto i64 = builder.getI64Type();
  auto signature = FunctionType::get(builder.getContext(),
                                     {ptr, i32, i32, ptr}, {ptr});
  auto declaration = module.lookupSymbol<func::FuncOp>(
      "annc_context_alloc_output_memref");
  if (declaration && declaration.getFunctionType() != signature)
    return failure();
  if (!declaration) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto accessor = builder.create<func::FuncOp>(
        loc, "annc_context_alloc_output_memref", signature);
    accessor.setPrivate();
  }
  auto constant = [&](int64_t value, Type integerType) {
    return builder.create<LLVM::ConstantOp>(
        loc, integerType, builder.getIntegerAttr(integerType, value));
  };
  Value dims = builder.create<LLVM::AllocaOp>(
      loc, ptr, i64, constant(std::max<int64_t>(type.getRank(), 1), i64), 0);
  unsigned dynamic = 0;
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    Value size;
    if (type.isDynamicDim(axis))
      size = builder.create<arith::IndexCastOp>(
          loc, i64, allocation.getDynamicSizes()[dynamic++]);
    else
      size = constant(type.getDimSize(axis), i64);
    Value address = builder.create<LLVM::GEPOp>(
        loc, ptr, i64, dims, ValueRange{constant(axis, i64)});
    builder.create<LLVM::StoreOp>(loc, size, address);
  }
  Value descriptorPointer = builder.create<func::CallOp>(
      loc, "annc_context_alloc_output_memref", TypeRange{ptr},
      ValueRange{context, constant(slot, i32), constant(type.getRank(), i32),
                 dims}).getResult(0);
  Value descriptor = builder.create<LLVM::LoadOp>(
      loc, converter.convertType(type), descriptorPointer);
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{type},
                                          ValueRange{descriptor})
      .getResult(0);
}

Value returnedAllocation(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  return value.getDefiningOp<memref::AllocOp>() ? value : Value{};
}

LogicalResult replaceAllocations(func::FuncOp func,
                                 LLVMTypeConverter &converter) {
  auto metadata = func->getAttrOfType<DictionaryAttr>("fusion.metadata");
  auto abi = metadata ? metadata.getAs<StringAttr>("abi") : StringAttr{};
  bool isContextKernel = abi && abi.getValue() == "annc_context_v3" &&
                         func->hasAttr("annc.kernel");
  if (!isContextKernel || func.getBody().empty()) return success();
  if (!llvm::all_of(func.getFunctionType().getResults(),
                    [](Type type) { return isa<MemRefType>(type); }))
    return success();

  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp op) { returns.push_back(op); });
  SmallVector<memref::AllocOp> allocations;
  func.walk([&](memref::AllocOp op) { allocations.push_back(op); });
  if (allocations.empty() && func.getFunctionType().getNumResults() == 0)
    return success();

  SmallVector<std::pair<memref::AllocOp, unsigned>> returned;
  for (auto ret : returns) {
    if (ret.getNumOperands() != func.getFunctionType().getNumResults())
      return ret.emitError("return/result count mismatch");
    for (auto [slot, result] : llvm::enumerate(ret.getOperands())) {
      if (auto allocation = returnedAllocation(result)) {
        auto op = allocation.getDefiningOp<memref::AllocOp>();
        auto previous = llvm::find_if(
            returned, [&](const auto &entry) { return entry.first == op; });
        if (previous == returned.end())
          returned.emplace_back(op, slot);
        else if (previous->second != slot)
          return ret.emitError(
              "one allocation cannot bind multiple output slots");
      } else {
        return ret.emitError(
            "context result must originate from an allocation");
      }
    }
  }
  if (!returned.empty() &&
      (func.getNumArguments() != 1 ||
       !isa<LLVM::LLVMPointerType>(func.getArgument(0).getType())))
    return func.emitError(
        "context output binding requires one context pointer");

  for (auto alloc : allocations) {
    OpBuilder builder(alloc);
    Value replacement;
    auto output = llvm::find_if(
        returned, [&](const auto &entry) { return entry.first == alloc; });
    if (output != returned.end()) {
      FailureOr<Value> acquired = loadContextOutput(
          builder, alloc.getLoc(), func->getParentOfType<ModuleOp>(),
          func.getArgument(0), output->second, alloc,
          converter);
      if (failed(acquired))
        return alloc.emitError(
            "context output requires a supported element type and rank");
      replacement = *acquired;
      if (replacement.getType() != alloc.getType())
        replacement = builder.create<memref::CastOp>(
            alloc.getLoc(), alloc.getType(), replacement);
    } else {
      replacement = builder.create<memref::AllocaOp>(
          alloc.getLoc(), alloc.getType(), alloc.getDynamicSizes(),
          alloc.getAlignmentAttr());
    }
    alloc.replaceAllUsesWith(replacement);
    alloc.erase();
  }
  for (auto ret : returns) ret->setOperands(ValueRange{});
  func.setType(FunctionType::get(func.getContext(),
                                 func.getFunctionType().getInputs(), {}));
  return success();
}

class KPGemmOneShotBufferize
    : public KPGemmOneShotBufferizeBase<KPGemmOneShotBufferize> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bufferization::OneShotBufferizationOptions options;
    if (failed(annc::runOneShotBufferize(module, options)))
      return signalPassFailure();
    if (!annc::loweringContextABIEnabled()) return;
    LLVMTypeConverter converter(&getContext());
    for (auto func : module.getOps<func::FuncOp>()) {
      if (failed(replaceAllocations(func, converter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createKPGemmOneShotBufferize() {
  return std::make_unique<KPGemmOneShotBufferize>();
}

}  // namespace annc
