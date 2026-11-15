
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"

namespace atir {
void registerAllAtirPasses() {
    atir::registerAtirIdentityCanonicalizePass();
    atir::registerAtirGemmEpilogueFusionPass();
    atir::registerAtirFoldBatchNormPass();
    atir::registerAtirRankInferencePass();
    atir::registerAtirOpFusionPass();
    atir::registerAtirConfigFusionPass();
    atir::registerAnncFastCodegenPass();
    atir::registerAtirPruneFuncPass();
    atir::registerAnncSelectKernelPass();
    atir::registerAtirSpecializeShapesPass();
}
}  // namespace atir
