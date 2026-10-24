#ifndef ANNC_DIALECT_ATIR_BUILTINKERNELSCHEMA_H
#define ANNC_DIALECT_ATIR_BUILTINKERNELSCHEMA_H

#include "Dialect/Atir/CustomOpSchema.h"
#include "llvm/ADT/ArrayRef.h"

namespace atir {

// Shared TF/ONNX context schema lookup. Attributes follow the schema's I64Attr
// declaration order; callers do not need to interpret individual attributes.
std::optional<CustomOpSchema> getContextKernelSchema(
    llvm::StringRef opName, llvm::ArrayRef<int64_t> attributes = {});

}  // namespace atir

#endif  // ANNC_DIALECT_ATIR_BUILTINKERNELSCHEMA_H
