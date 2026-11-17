
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"

namespace atir {
void registerAllAtirPasses() {
    atir::registerAtirIdentityCanonicalizePass();
    atir::registerAtirGemmEpilogueFusionPass();
    atir::registerAtirFoldBatchNormPass();
    atir::registerAtirRankInferencePass();
    atir::registerAtirOpFusionPass();
#ifdef ANNC_ENABLE_TF2ATIR_V2
    atir::registerAtirOpFusionV2Pass();
#endif
    atir::registerAtirConfigFusionPass();
    atir::registerAnncFastCodegenPass();
    atir::registerAtirPruneFuncPass();
    atir::registerAnncSelectKernelPass();
    atir::registerAtirSpecializeShapesPass();
#ifdef ANNC_ENABLE_TF2ATIR_V2
    atir::registerAnncSpecializeShapesPass();
    atir::registerAnncGemmEpilogueCommitPass();
#endif
}
}  // namespace atir
