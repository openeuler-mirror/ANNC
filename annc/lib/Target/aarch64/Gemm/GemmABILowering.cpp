#include "GemmPlan.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"

namespace annc {
namespace {

LLVM::LLVMFuncOp getOrCreateAssemblyDeclaration(ModuleOp module,
                                                StringRef symbol,
                                                ArrayRef<Type> inputs,
                                                Type result) {
  if (auto declaration = module.lookupSymbol<LLVM::LLVMFuncOp>(symbol))
    return declaration;
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto type = LLVM::LLVMFunctionType::get(result, inputs);
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), symbol, type);
}

FailureOr<Value> getRankedMemRef(Value value) {
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  if (!llvm::isa<MemRefType>(value.getType())) return failure();
  return value;
}

FailureOr<Value> createElementPointer(OpBuilder &builder, Location loc,
                                      Value opaqueMemRef, Value elementOffset) {
  FailureOr<Value> memref = getRankedMemRef(opaqueMemRef);
  if (failed(memref)) return failure();

  Value aligned =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, *memref);
  Value elementBytes = builder.create<arith::ConstantIndexOp>(loc, 4);
  Value byteOffset =
      builder.create<arith::MulIOp>(loc, elementOffset, elementBytes);
  Value address = builder.create<arith::AddIOp>(loc, aligned, byteOffset);
  Value addressI64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), address);
  return builder
      .create<LLVM::IntToPtrOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()), addressI64)
      .getResult();
}

Value toI32(OpBuilder &builder, Location loc, Value index) {
  return builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), index);
}

Value toI64(OpBuilder &builder, Location loc, Value index) {
  return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
}

LogicalResult lowerPackBCall(ModuleOp module, func::CallOp call) {
  if (call.getNumOperands() != 6)
    return call.emitOpError("has an invalid PackB leaf signature");
  OpBuilder builder(call);
  Location loc = call.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  FailureOr<Value> rhs = createElementPointer(builder, loc, call.getOperand(0),
                                              call.getOperand(2));
  FailureOr<Value> packed =
      createElementPointer(builder, loc, call.getOperand(1), zero);
  if (failed(rhs) || failed(packed))
    return call.emitOpError("cannot materialize PackB raw pointers");

  auto symbol =
      call->getAttrOfType<StringAttr>(aarch64::gemm::kAsmSymbolAttrName);
  if (!symbol)
    return call.emitOpError("has no statically selected PackB symbol");

  Type ptr = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  LLVM::LLVMFuncOp declaration = getOrCreateAssemblyDeclaration(
      module, symbol.getValue(), ArrayRef<Type>{ptr, ptr, i32, i32, i32},
      LLVM::LLVMVoidType::get(module.getContext()));
  builder.create<LLVM::CallOp>(
      loc, declaration,
      ValueRange{*rhs, *packed, toI32(builder, loc, call.getOperand(3)),
                 toI32(builder, loc, call.getOperand(4)),
                 toI32(builder, loc, call.getOperand(5))});
  call.erase();
  return success();
}

LogicalResult lowerMicrokernelCall(ModuleOp module, func::CallOp call) {
  if (call.getNumOperands() != 10)
    return call.emitOpError("has an invalid microkernel leaf signature");
  if (!aarch64::gemm::hasStage(call, aarch64::gemm::kMicrokernelLoweredStage)) {
    return call.emitOpError("was not selected by microkernel lowering");
  }

  OpBuilder builder(call);
  Location loc = call.getLoc();
  FailureOr<Value> lhs = createElementPointer(builder, loc, call.getOperand(0),
                                              call.getOperand(3));
  FailureOr<Value> packed = createElementPointer(
      builder, loc, call.getOperand(1), call.getOperand(4));
  FailureOr<Value> out = createElementPointer(builder, loc, call.getOperand(2),
                                              call.getOperand(5));
  if (failed(lhs) || failed(packed) || failed(out))
    return call.emitOpError("cannot materialize microkernel raw pointers");

  auto symbol =
      call->getAttrOfType<StringAttr>(aarch64::gemm::kAsmSymbolAttrName);
  if (!symbol)
    return call.emitOpError("has no statically selected microkernel symbol");

  SmallVector<Value> arguments = {*lhs, *packed, *out};
  for (unsigned index = 6; index < call.getNumOperands(); ++index)
    arguments.push_back(toI32(builder, loc, call.getOperand(index)));

  Type ptr = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  SmallVector<Type> inputTypes = {ptr, ptr, ptr, i32, i32, i32, i32};
  LLVM::LLVMFuncOp declaration = getOrCreateAssemblyDeclaration(
      module, symbol.getValue(), inputTypes,
      LLVM::LLVMVoidType::get(module.getContext()));
  builder.create<LLVM::CallOp>(loc, declaration, arguments);
  call.erase();
  return success();
}

LogicalResult lowerSvePackedBHelperCall(ModuleOp module, func::CallOp call) {
  if (call.getNumOperands() != 2 || call.getNumResults() != 1 ||
      !call.getResult(0).getType().isIndex()) {
    return call.emitOpError("has an invalid SVE packed-B helper signature");
  }

  OpBuilder builder(call);
  Location loc = call.getLoc();
  Type i64 = builder.getI64Type();
  if (auto declaration = module.lookupSymbol<func::FuncOp>(call.getCallee())) {
    if (!declaration.isDeclaration()) {
      return call.emitOpError(
          "conflicts with an SVE packed-B helper definition");
    }
    declaration.erase();
  }
  LLVM::LLVMFuncOp declaration = getOrCreateAssemblyDeclaration(
      module, call.getCallee(), ArrayRef<Type>{i64, i64}, i64);
  LLVM::CallOp helperCall = builder.create<LLVM::CallOp>(
      loc, declaration,
      ValueRange{toI64(builder, loc, call.getOperand(0)),
                 toI64(builder, loc, call.getOperand(1))});
  Value result = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                    helperCall.getResult());
  call.getResult(0).replaceAllUsesWith(result);
  call.erase();
  return success();
}

bool isSvePackedBHelper(StringRef callee) {
  return callee == aarch64::gemm::kSvePackedBElementsAsmSymbol ||
         callee == aarch64::gemm::kSvePackedBOffsetAsmSymbol;
}

class AArch64GemmABILowering
    : public AArch64GemmABILoweringBase<AArch64GemmABILowering> {
 public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::CallOp> calls;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() == aarch64::gemm::kPackBLeafName ||
          call.getCallee() == aarch64::gemm::kMicrokernelLeafName ||
          isSvePackedBHelper(call.getCallee())) {
        calls.push_back(call);
      }
    });
    for (func::CallOp call : calls) {
      LogicalResult result =
          call.getCallee() == aarch64::gemm::kPackBLeafName
              ? lowerPackBCall(module, call)
          : call.getCallee() == aarch64::gemm::kMicrokernelLeafName
              ? lowerMicrokernelCall(module, call)
              : lowerSvePackedBHelperCall(module, call);
      if (failed(result)) {
        signalPassFailure();
        return;
      }
    }

    for (StringRef name :
         {aarch64::gemm::kPackBLeafName, aarch64::gemm::kMicrokernelLeafName,
          aarch64::gemm::kSvePackedBElementsAsmSymbol,
          aarch64::gemm::kSvePackedBOffsetAsmSymbol}) {
      if (auto declaration = module.lookupSymbol<func::FuncOp>(name);
          declaration && declaration.isDeclaration()) {
        declaration.erase();
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAArch64GemmABILowering() {
  return std::make_unique<AArch64GemmABILowering>();
}

}  // namespace annc
