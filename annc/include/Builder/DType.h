#ifndef ANNC_DTYPE_H
#define ANNC_DTYPE_H

#include <optional>

#include "llvm/ADT/StringRef.h"

namespace annc {

// TF/ATIR element type, enum-encoded so constant and tensor type construction
// stay in sync (replaces stringly-typed dtype plumbing).
enum class DType {
  F32,
  F64,
  F16,
  BF16,
  I64,
  I32,
  I16,
  I8,
  U8,
  U16,
  U32,
  U64,
  Bool,
  String,
  Complex64,
  Complex128,
};

// Canonical name ("float32", "int64", ...) and its parser (nullopt if unknown).
llvm::StringRef toString(DType dtype);
std::optional<DType> parseDType(llvm::StringRef name);

}  // namespace annc

#endif  // ANNC_DTYPE_H
