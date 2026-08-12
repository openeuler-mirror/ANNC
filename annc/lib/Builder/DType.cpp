#include "Builder/DType.h"

#include "llvm/Support/ErrorHandling.h"

namespace annc {

llvm::StringRef toString(DType dtype) {
  switch (dtype) {
    case DType::F32:
      return "float32";
    case DType::F64:
      return "float64";
    case DType::F16:
      return "float16";
    case DType::BF16:
      return "bfloat16";
    case DType::I64:
      return "int64";
    case DType::I32:
      return "int32";
    case DType::I16:
      return "int16";
    case DType::I8:
      return "int8";
    case DType::U8:
      return "uint8";
    case DType::U16:
      return "uint16";
    case DType::U32:
      return "uint32";
    case DType::U64:
      return "uint64";
    case DType::Bool:
      return "bool";
    case DType::String:
      return "string";
    case DType::Complex64:
      return "complex64";
    case DType::Complex128:
      return "complex128";
  }
  llvm_unreachable("unknown DType");
}

std::optional<DType> parseDType(llvm::StringRef name) {
  if (name == "float32") return DType::F32;
  if (name == "float64") return DType::F64;
  if (name == "float16") return DType::F16;
  if (name == "bfloat16") return DType::BF16;
  if (name == "int64") return DType::I64;
  if (name == "int32") return DType::I32;
  if (name == "int16") return DType::I16;
  if (name == "int8") return DType::I8;
  if (name == "uint8") return DType::U8;
  if (name == "uint16") return DType::U16;
  if (name == "uint32") return DType::U32;
  if (name == "uint64") return DType::U64;
  if (name == "bool") return DType::Bool;
  if (name == "string") return DType::String;
  if (name == "complex64") return DType::Complex64;
  if (name == "complex128") return DType::Complex128;
  return std::nullopt;
}

}  // namespace annc
