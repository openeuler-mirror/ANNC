#ifndef ANNC_DIALECT_ATIR_OPVERIFY_GENERICMEMREF_H
#define ANNC_DIALECT_ATIR_OPVERIFY_GENERICMEMREF_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <optional>
#include <vector>

namespace atir {

/// GenericMemRef structure with RAII management
/// MemRef descriptor compatible with MLIR C Interface
struct GenericMemRef {
  /// RAII-managed memory buffer
  struct DataBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    
    ~DataBuffer() {
      if (ptr) free(ptr);
    }
    
    DataBuffer() = default;
    DataBuffer(const DataBuffer&) = delete;
    DataBuffer& operator=(const DataBuffer&) = delete;
    DataBuffer(DataBuffer&& o) noexcept : ptr(o.ptr), size(o.size) {
      o.ptr = nullptr; o.size = 0;
    }
    DataBuffer& operator=(DataBuffer&& o) noexcept {
      std::swap(ptr, o.ptr); std::swap(size, o.size); return *this;
    }
    
    /// Allocate aligned memory
    static DataBuffer allocate(size_t bytes, size_t alignment = 64);
  };
  
  DataBuffer data_buf;              ///< Data buffer
  std::vector<int64_t> descriptor;  ///< MemRef descriptor (pointer, offset, shape, strides)
  size_t element_bytes;             ///< Element size in bytes
  
  GenericMemRef() = delete;
  GenericMemRef(DataBuffer&& buf, std::vector<int64_t>&& desc, size_t elemBytes)
      : data_buf(std::move(buf)), descriptor(std::move(desc)), element_bytes(elemBytes) {}
};

size_t getElementBytes(mlir::Type type);

std::optional<GenericMemRef> createGenericMemRef(mlir::DenseElementsAttr attr);

mlir::DenseElementsAttr extractResultToAttr(const GenericMemRef& memRef, 
                                            mlir::RankedTensorType type);

} // namespace atir

#endif // ANNC_DIALECT_ATIR_OPVERIFY_GENERICMEMREF_H
