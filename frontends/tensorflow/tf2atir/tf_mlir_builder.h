#ifndef ANNC_TF2ATIR_TF_MLIR_BUILDER_H
#define ANNC_TF2ATIR_TF_MLIR_BUILDER_H

#include "Builder/MLIROpBuilder.h"

namespace annc::tf2atir {

class TfMLIRBuilder : public MLIRBuilder {
 public:
  using MLIRBuilder::MLIRBuilder;
  static bool isSupportedOp(llvm::StringRef opType);
  static llvm::ArrayRef<OpSpec> getAllOpSpecs();

 protected:
  const OpSpec* lookupSpec(llvm::StringRef opType) const override;
  std::optional<mlir::LogicalResult> buildFrontendNode(
      const NodeInfo& node) override;

 private:
  mlir::LogicalResult buildControlFlowMirrorNode(const NodeInfo& node);
};

}  // namespace annc::tf2atir
#endif
