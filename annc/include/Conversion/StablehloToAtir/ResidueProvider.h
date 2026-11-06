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
// prefix; `sourceAttrs` = declared attrs, F2: decoded backend_config).
struct ResidueView {
  std::string opType;
  mlir::DictionaryAttr sourceAttrs;
};

// Frontend adaptation seam: the sink owns the driver, fallback carrier and
// stats; a provider owns recognition, the op table and emission.
class ResidueProvider {
 public:
  virtual ~ResidueProvider() = default;

  // True when `op` is a residue this provider must sink.
  virtual bool isResidue(mlir::Operation* op) const = 0;

  // Normalizes `op`; nullopt when it is not a residue.
  virtual std::optional<ResidueView> match(mlir::Operation* op) const = 0;

  // True when `view.opType` is mappable (classifies fallback reasons only).
  virtual bool hasMapping(const ResidueView& view,
                          mlir::MLIRContext* ctx) const {
    (void)view;
    (void)ctx;
    return false;
  }

  // Element type for a string tensor element, or null when unmodeled.
  virtual mlir::Type stringElementType(mlir::Type elementType,
                                       mlir::MLIRContext* ctx) const {
    (void)elementType;
    (void)ctx;
    return {};
  }

  // Emits a constant residue; failure leaves it to the fallback carrier.
  virtual mlir::LogicalResult emitConstant(
      mlir::ConversionPatternRewriter& rewriter, mlir::Operation* op,
      const ResidueView& view, mlir::ArrayRef<mlir::Type> resultTypes,
      llvm::SmallVectorImpl<mlir::Value>& results) const {
    (void)rewriter;
    (void)op;
    (void)view;
    (void)resultTypes;
    (void)results;
    return mlir::failure();
  }

  // Emits the mapped form; failure rolls back to the fallback carrier.
  virtual mlir::LogicalResult emitMapped(
      mlir::ConversionPatternRewriter& rewriter, mlir::Operation* op,
      const ResidueView& view, mlir::ArrayRef<mlir::Value> operands,
      mlir::ArrayRef<mlir::Type> resultTypes,
      llvm::SmallVectorImpl<mlir::Value>& results) const = 0;
};

// TF provider: F1 unregistered `tf.*`, F2 stablehlo.custom_call, F3 declared
// `call_target_name` ops.
std::unique_ptr<ResidueProvider> createTfResidueProvider();

}  // namespace annc

#endif  // ANNC_CONVERSION_STABLEHLO_TO_ATIR_RESIDUE_PROVIDER_H
