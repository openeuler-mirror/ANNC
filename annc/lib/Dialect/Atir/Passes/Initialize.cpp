
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
    atir::registerAtirFastCodegenPass();
    atir::registerAtirPruneFuncPass();
    atir::registerAtirSelectKernelPass();
    atir::registerAtirSpecializeShapesPass();
}
}  // namespace atir
