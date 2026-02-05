
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"

namespace atir {
void registerAllAtirPasses() {
    atir::registerAtirDistributePass();
    atir::registerAtirTilingPass();
    atir::registerAtirCanonicalizePass();
    atir::registerAtirOpFusionPass();
    atir::registerAtirFastCodegenPass();
    atir::registerAtirBlockFusionPass();
    atir::registerAtirUnrollPass();
    atir::registerAtirEltwiseFusionPass();
    atir::registerAtirLLMCodeGenPass();
    atir::registerAtirPruneFuncPass();
}
}  // namespace atir