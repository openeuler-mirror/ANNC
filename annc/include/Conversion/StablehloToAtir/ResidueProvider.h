#ifndef ANNC_CONVERSION_STABLEHLO_TO_ATIR_RESIDUE_PROVIDER_H
#define ANNC_CONVERSION_STABLEHLO_TO_ATIR_RESIDUE_PROVIDER_H

#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class MLIRContext;
}  // namespace mlir

namespace annc {

// A frontend residue in a frontend-agnostic view (`opType` without dialect
// prefix; `sourceAttrs` = declared attrs, F1: decoded backend_config).
struct ResidueView {
  std::string opType;
  mlir::DictionaryAttr sourceAttrs;
};

// Adaptation seam: sink = frontend-agnostic (driver, fallback, bridge, stats);
// provider = frontend-specific. New frontend = implement this interface.
class ResidueProvider {
 public:
  virtual ~ResidueProvider() = default;

  virtual bool isResidue(mlir::Operation* op) const = 0;

  virtual std::optional<ResidueView> match(mlir::Operation* op) const = 0;

  virtual bool hasMapping(const ResidueView& view,
                          mlir::MLIRContext* ctx) const {
    (void)view;
    (void)ctx;
    return false;
  }

  virtual mlir::Type stringElementType(mlir::Type elementType,
                                       mlir::MLIRContext* ctx) const {
    (void)elementType;
    (void)ctx;
    return {};
  }

  virtual mlir::LogicalResult emitConstant(
      mlir::RewriterBase& rewriter, mlir::Operation* op,
      const ResidueView& view, mlir::ArrayRef<mlir::Type> resultTypes,
      llvm::SmallVectorImpl<mlir::Value>& results) const {
    (void)rewriter;
    (void)op;
    (void)view;
    (void)resultTypes;
    (void)results;
    return mlir::failure();
  }

  virtual mlir::LogicalResult emitMapped(
      mlir::RewriterBase& rewriter, mlir::Operation* op,
      const ResidueView& view, mlir::ArrayRef<mlir::Value> operands,
      mlir::ArrayRef<mlir::Type> resultTypes,
      llvm::SmallVectorImpl<mlir::Value>& results) const = 0;
};

// TF provider: F1 (`stablehlo.custom_call`) and F2 (unregistered op declaring
// `call_target_name` — the renamed `annc.residue` carrier's form).
std::unique_ptr<ResidueProvider> createTfResidueProvider();

}  // namespace annc

#endif  // ANNC_CONVERSION_STABLEHLO_TO_ATIR_RESIDUE_PROVIDER_H
