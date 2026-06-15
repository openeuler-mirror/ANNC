#ifndef ANNC_DIALECT_ATIR_CUSTOMOPSCHEMA_H
#define ANNC_DIALECT_ATIR_CUSTOMOPSCHEMA_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "Kernel/KernelRegistry.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace atir {

class CustomOpSchema {
public:
  enum class ArgKind {
    MemRef,
    Scalar,
    Opaque,
    Custom,
  };

  struct Arg {
    std::string name;
    ArgKind kind = ArgKind::MemRef;
    int64_t rank = -1;
    std::string typeVar;
  };

  static CustomOpSchema get(llvm::StringRef name) {
    CustomOpSchema schema;
    schema.name_ = name.str();
    return schema;
  }

  CustomOpSchema &TypeVar(llvm::StringRef name) {
    typeVars_.push_back(name.str());
    return *this;
  }

  CustomOpSchema &MemRefArg(llvm::StringRef name, int64_t rank,
                            llvm::StringRef typeVar) {
    args_.push_back(Arg{name.str(), ArgKind::MemRef, rank, typeVar.str()});
    return *this;
  }

  mlir::DictionaryAttr toMetadata(mlir::MLIRContext *ctx) const;

private:
  std::string name_;
  std::vector<std::string> typeVars_;
  std::vector<Arg> args_;
};

std::optional<annc::kernels::TypeConstraintInfo>
inferTypeConstraint(llvm::StringRef typeVar, mlir::Type type);

std::vector<annc::kernels::TypeConstraintInfo>
inferTypeConstraintsFromSchema(mlir::DictionaryAttr metadata,
                               mlir::ValueRange operands);

} // namespace atir

#endif // ANNC_DIALECT_ATIR_CUSTOMOPSCHEMA_H
