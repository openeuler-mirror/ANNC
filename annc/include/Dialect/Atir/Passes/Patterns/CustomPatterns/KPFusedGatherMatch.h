#ifndef ANNC_ATIR_KP_FUSED_GATHER_MATCH_H
#define ANNC_ATIR_KP_FUSED_GATHER_MATCH_H

#include "Dialect/Atir/AtirOps.h"
#include "llvm/ADT/SmallVector.h"

namespace atir {

struct KPFusedGatherMatch {
  GatherOp outerGather;
  GatherOp innerGather;
  UniqueOp firstUnique;
  UniqueOp secondUnique;
  StridedSliceOp slice;
  Value data;
  Value keys;
  Value begin;
  llvm::SmallVector<Value, 3> boundaryOutputs;
};

mlir::FailureOr<KPFusedGatherMatch> matchKPFusedGather(GatherOp outerGather);

}  // namespace atir

#endif
