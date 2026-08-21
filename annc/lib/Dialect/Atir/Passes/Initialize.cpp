
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"

namespace atir {
void registerAllAtirPasses() {
    atir::registerAtirDistributePass();
    atir::registerAtirTilingPass();
    atir::registerAtirCanonicalizePass();
    atir::registerAtirIdentityCanonicalizePass();
    atir::registerAtirGemmEpilogueFusionPass();
    atir::registerAtirOpFusionPass();
    atir::registerAtirFastCodegenPass();
    atir::registerAtirBlockFusionPass();
    atir::registerAtirUnrollPass();
    atir::registerAtirEltwiseFusionPass();
    atir::registerAtirPruneFuncPass();
    atir::registerAtirSelectKernelPass();
    atir::registerAtirSpecializeShapesPass();
    atir::registerAtirSelectLoweringStrategyPass();
}
}  // namespace atir
