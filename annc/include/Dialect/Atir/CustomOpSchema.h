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
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace atir {

class CustomOpSchema {
public:
  enum class ArgKind {
    MemRef,
    Scalar,
    I64Attr,
    Opaque,
    Custom,
  };

  struct Arg {
    std::string name;
    ArgKind kind = ArgKind::MemRef;
    int64_t rank = -1;
    std::string typeVar;
    int64_t i64Value = 0;
  };

  struct ResultSpec {
    std::string name;
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

  CustomOpSchema &I64AttrArg(llvm::StringRef name, int64_t value) {
    Arg arg;
    arg.name = name.str();
    arg.kind = ArgKind::I64Attr;
    arg.i64Value = value;
    args_.push_back(arg);
    return *this;
  }

  CustomOpSchema &Result(llvm::StringRef name, int64_t rank,
                         llvm::StringRef typeVar) {
    results_.push_back(ResultSpec{name.str(), rank, typeVar.str()});
    return *this;
  }

  llvm::ArrayRef<ResultSpec> results() const { return results_; }

  mlir::DictionaryAttr toMetadata(mlir::MLIRContext *ctx) const;

private:
  std::string name_;
  std::vector<std::string> typeVars_;
  std::vector<Arg> args_;
  std::vector<ResultSpec> results_;
};

std::optional<annc::kernels::TypeConstraintInfo>
inferTypeConstraint(llvm::StringRef typeVar, mlir::Type type);

llvm::Expected<std::vector<annc::kernels::TypeConstraintInfo>>
inferTypeConstraintsFromSchema(mlir::DictionaryAttr metadata,
                               mlir::TypeRange operandTypes,
                               mlir::TypeRange resultTypes);

} // namespace atir

#endif // ANNC_DIALECT_ATIR_CUSTOMOPSCHEMA_H
