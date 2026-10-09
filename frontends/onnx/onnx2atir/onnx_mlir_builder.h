#ifndef ANNC_ONNX2ATIR_ONNX_MLIR_BUILDER_H
#define ANNC_ONNX2ATIR_ONNX_MLIR_BUILDER_H

#include "Builder/MLIROpBuilder.h"

namespace annc::onnx2atir {

// The ONNX op table, exported so the coverage report and the adapter can ask
// whether a given op has an ATIR target without duplicating the list.
const OpSpec* lookupOnnxOpSpec(llvm::StringRef opType);

// MLIRBuilder with the ONNX spec table substituted for the TF one. Node
// validation, op construction, attribute application, metadata and diagnostics
// are all inherited unchanged; only two seams differ:
//   * lookupSpec resolves against the ONNX table (ONNX op names collide with
//     TF op names that mean something else);
//   * shouldLowerToOpaque honours the adapter's per-node fallback decision, so
//     a node the adapter could not faithfully normalize lands in atir.opaque
//     instead of being silently mistranslated.
// The TF path keeps the base implementations and is therefore unaffected.
class OnnxMLIRBuilder : public MLIRBuilder {
 public:
  using MLIRBuilder::MLIRBuilder;

 protected:
  const OpSpec* lookupSpec(llvm::StringRef opType) const override;
  bool shouldLowerToOpaque(const NodeInfo& node) const override;
};

// Attributes understood by the adapter's node normalization. All four are
// stored in NodeInfo::attrs with a leading underscore, the marker the builder
// treats as frontend-internal: they steer construction but never reach the
// emitted metadata (see MLIRBuilder::attachMetadata).
inline constexpr llvm::StringLiteral kFallbackAttr = "_onnx_fallback";
inline constexpr llvm::StringLiteral kFallbackReasonAttr =
    "_onnx_fallback_reason";
// Normalized atir.Slice window, computed by the adapter from the ONNX
// starts/ends/axes/steps operands and their static shapes.
inline constexpr llvm::StringLiteral kSliceBeginAttr = "_onnx_slice_begin";
inline constexpr llvm::StringLiteral kSliceSizeAttr = "_onnx_slice_size";

}  // namespace annc::onnx2atir

#endif  // ANNC_ONNX2ATIR_ONNX_MLIR_BUILDER_H
