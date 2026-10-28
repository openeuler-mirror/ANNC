#include "Conversion/Common/ContextInputOverrides.h"

#include "Context/KernelContext.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace atir {
Value createContextInputOverrides(OpBuilder &builder, Location loc,
                                 ModuleOp module, StringRef symbolPrefix,
                                 Value context,
                                 ArrayRef<DenseElementsAttr> overrides) {
  if (llvm::none_of(overrides, [](auto value) { return bool(value); }))
    return context;

  auto ptr = LLVM::LLVMPointerType::get(builder.getContext());
  auto i32 = builder.getI32Type();
  auto i64 = builder.getI64Type();
  // Mirror AnncTensorView (viewType) and AnncKernelContext (contextType) from
  // context/include/Context/KernelContext.h. Keep these field types/order and
  // the field indices below in sync with that header, including changes to
  // packing/alignment. LLVM's target data layout determines the target layout.
  auto viewType = LLVM::LLVMStructType::getLiteral(
      builder.getContext(), {ptr, ptr, i64, ptr, ptr, i32, i32, i32, i32});
  auto contextType = LLVM::LLVMStructType::getLiteral(
      builder.getContext(), {i32, i32, ptr, i32, ptr, i32, ptr, i32, ptr});
  auto integer = [&](int64_t value, Type type) -> Value {
    return builder.create<LLVM::ConstantOp>(
        loc, type, builder.getIntegerAttr(type, value));
  };
  auto global = [&](StringRef suffix, DenseElementsAttr values) -> Value {
    std::string name = (symbolPrefix + suffix).str();
    auto flat = values.reshape(RankedTensorType::get(
        {values.getNumElements()}, values.getElementType()));
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      builder.create<LLVM::GlobalOp>(
          loc, LLVM::LLVMArrayType::get(values.getElementType(),
                                      values.getNumElements()),
          /*isConstant=*/true, LLVM::Linkage::Private, name, flat);
    }
    return builder.create<LLVM::AddressOfOp>(loc, ptr, name);
  };
  Value original = builder.create<LLVM::LoadOp>(loc, contextType, context);
  Value inputs = builder.create<LLVM::ExtractValueOp>(
      loc, original, ArrayRef<int64_t>{2});
  Value localInputs = builder.create<LLVM::AllocaOp>(
      loc, ptr, viewType, integer(overrides.size(), i64), 0);
  for (auto [slot, payload] : llvm::enumerate(overrides)) {
    Value slotIndex = integer(slot, i64);
    Value dest = builder.create<LLVM::GEPOp>(
        loc, ptr, viewType, localInputs, ValueRange{slotIndex});
    Value view;
    if (!payload) {
      Value source = builder.create<LLVM::GEPOp>(
          loc, ptr, viewType, inputs, ValueRange{slotIndex});
      view = builder.create<LLVM::LoadOp>(loc, viewType, source);
    } else {
      const std::string stem = "_context_input_" + std::to_string(slot);
      Value data = global(stem + "_data", payload);
      auto shape = payload.getType().getShape();
      SmallVector<int64_t> strides(shape.size());
      int64_t stride = 1;
      for (size_t axis = shape.size(); axis > 0; --axis) {
        strides[axis - 1] = stride;
        stride *= shape[axis - 1];
      }
      auto shapeType = RankedTensorType::get(
          {static_cast<int64_t>(shape.size())}, i64);
      Value sizes =
          global(stem + "_sizes", DenseIntElementsAttr::get(shapeType, shape));
      Value steps =
          global(stem + "_strides", DenseIntElementsAttr::get(shapeType, strides));
      SmallVector<Value> fields = {
          data, data, integer(0, i64), sizes, steps,
          integer(ANNC_ELEMENT_TYPE_F32, i32), integer(shape.size(), i32),
          integer(ANNC_LAYOUT_STRIDED, i32),
          integer(ANNC_TENSOR_BOUND | ANNC_TENSOR_READ_ONLY, i32)};
      view = builder.create<LLVM::UndefOp>(loc, viewType);
      for (auto [index, field] : llvm::enumerate(fields))
        view = builder.create<LLVM::InsertValueOp>(
            loc, view, field, ArrayRef<int64_t>{static_cast<int64_t>(index)});
    }
    builder.create<LLVM::StoreOp>(loc, view, dest);
  }
  Value local = builder.create<LLVM::InsertValueOp>(
      loc, original, localInputs, ArrayRef<int64_t>{2});
  Value address = builder.create<LLVM::AllocaOp>(
      loc, ptr, contextType, integer(1, i64), 0);
  builder.create<LLVM::StoreOp>(loc, local, address);
  return address;
}
}  // namespace atir
