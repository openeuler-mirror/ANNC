#include "Dialect/Atir/OpVerify/GenericMemRef.h"
#include "Dialect/Atir/AtirOps.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>

namespace atir {

GenericMemRef::DataBuffer GenericMemRef::DataBuffer::allocate(size_t bytes, size_t alignment) {
  DataBuffer buf;
  buf.size = bytes;
  if (bytes > 0 && posix_memalign(&buf.ptr, alignment, bytes) != 0) {
    buf.ptr = nullptr;
    buf.size = 0;
  }
  return buf;
}

size_t getElementBytes(mlir::Type type) {
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth() / 8;
  }
  llvm::errs() << "[Error] Unsupported element type for byte calculation: " << type << "\n";
  return 0;
}

std::optional<GenericMemRef> createGenericMemRef(mlir::DenseElementsAttr attr) {
  auto tensorType = llvm::cast<mlir::RankedTensorType>(attr.getType());
  auto shape = tensorType.getShape();
  int rank = shape.size();
  
  size_t elem_bytes = getElementBytes(tensorType.getElementType());
  if (elem_bytes == 0) {
    return std::nullopt;
  }
  size_t num_elements = 1;
  for (int dim : shape) {
    num_elements *= dim;
  }
  size_t total_bytes = num_elements * elem_bytes;

  auto data_buf = GenericMemRef::DataBuffer::allocate(total_bytes);
  if (!data_buf.ptr && total_bytes > 0) {
    return std::nullopt;
  }
  auto raw = attr.getRawData();
  if (attr.isSplat()) { // Broadcast value: all elements are identical
    for (size_t i = 0; i < num_elements; ++i) {
      std::memcpy(static_cast<char*>(data_buf.ptr) + i * elem_bytes, raw.data(), elem_bytes);
    }
  } else {
    std::memcpy(data_buf.ptr, raw.data(), total_bytes);
  }

  std::vector<int64_t> desc;
  uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(data_buf.ptr);
  desc.push_back(static_cast<int64_t>(ptr_addr));  // Data pointer
  desc.push_back(static_cast<int64_t>(ptr_addr));  // Aligned pointer
  desc.push_back(0);                     // Offset
  
  for (int i = 0; i < rank; ++i) {
    desc.push_back(shape[i]);
  }
  
  // Calculate and append strides
  std::vector<int64_t> strides(rank);
  int64_t stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  for (int i = 0; i < rank; ++i) {
    desc.push_back(strides[i]);
  }

  return GenericMemRef{std::move(data_buf), std::move(desc), elem_bytes};
}

mlir::DenseElementsAttr extractResultToAttr(const GenericMemRef& memRef, 
                                            mlir::RankedTensorType type) {
  size_t expected_bytes = type.getNumElements() * type.getElementTypeBitWidth() / 8;
  assert(expected_bytes == memRef.data_buf.size && "MemRef buffer size mismatch!");
  
  llvm::ArrayRef<char> raw_data(
      reinterpret_cast<const char*>(memRef.data_buf.ptr), 
      memRef.data_buf.size
  );
  return mlir::DenseElementsAttr::getFromRawBuffer(type, raw_data);
}

} // namespace atir
