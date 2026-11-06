// V2OpMatcher — convention-driven residue -> atir.*_v2 matcher. The provider
// map is the only source of truth (a new V2 op needs an ODS definition and a
// map entry); absence falls back to `atir.unsupported`. emitV2Op assembles
// attributes per ODS-declared name, preserves unconsumed attributes, verifies
// locally and erases on mismatch (fallback carrier).
#ifndef ANNC_CONVERSION_STABLEHLOTOATIR_V2OPMATCHER_H
#define ANNC_CONVERSION_STABLEHLOTOATIR_V2OPMATCHER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>
#include <utility>

namespace mlir {
class MLIRContext;
}  // namespace mlir

namespace annc {

// Source-name override for one ODS-declared target attribute, e.g.
//   {"transposeA", {"adj_x", "adjoint_a"}}
struct V2AttrAlias {
  llvm::StringRef declared;                  // target's ODS attr name
  llvm::ArrayRef<llvm::StringRef> sources;   // candidate source names, in order
};

// Resolves the registered `atir.*_v2` target; nullopt when unregistered.
std::optional<mlir::RegisteredOperationName> resolveV2Target(
    mlir::MLIRContext* ctx, llvm::StringRef sourceOpType,
    llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> opAliases = {});

// Emits `target`; appends results on success, erases and fails on mismatch.
mlir::LogicalResult emitV2Op(
    mlir::ConversionPatternRewriter& rewriter, mlir::Location loc,
    mlir::RegisteredOperationName target, mlir::ValueRange operands,
    mlir::ArrayRef<mlir::Type> resultTypes, mlir::DictionaryAttr source,
    llvm::ArrayRef<V2AttrAlias> attrAliases,
    llvm::ArrayRef<mlir::NamedAttribute> derived,
    llvm::SmallVectorImpl<mlir::Value>& results);

}  // namespace annc

#endif  // ANNC_CONVERSION_STABLEHLOTOATIR_V2OPMATCHER_H
