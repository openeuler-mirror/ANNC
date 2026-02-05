
#include "Dialect/Pimp/PimpOps.h"
#include "Dialect/Pimp/Passes/Passes.h"

namespace pimp {
void registerAllPimpPasses() {
    pimp::registerPimpDistributePass();
    pimp::registerPimpTilingPass();
    pimp::registerPimpCanonicalizePass();
    pimp::registerPimpOpFusionPass();
    pimp::registerPimpFastCodegenPass();
    pimp::registerPimpBlockFusionPass();
    pimp::registerPimpUnrollPass();
    pimp::registerPimpEltwiseFusionPass();
    pimp::registerPimpLLMCodeGenPass();
    pimp::registerPimpPruneFuncPass();
}
}  // namespace pimp