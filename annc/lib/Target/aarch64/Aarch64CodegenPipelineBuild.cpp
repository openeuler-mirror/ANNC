#include "Conversion/Passes.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
namespace annc
{
    namespace
    {
    struct AArch64CodegenPipelineOptions
        : public PassPipelineOptions<AArch64CodegenPipelineOptions>
    {
        PassOptions::Option<std::string> configPath{
            *this, "config-path",
            llvm::cl::desc("Path to the external AArch64 GEMM tuning configuration")};
        PassOptions::Option<std::string> packedCPath{
            *this, "packed-c",
            llvm::cl::desc("Path for the generated packed RHS C source")};
        PassOptions::Option<int64_t> intraThreadCount{
            *this, "intra-thread-count",
            llvm::cl::desc("Number of intra-op threads available to the GEMM "
                           "plan (1 = serial execution)"),
            llvm::cl::init(1)};
    };

    void buildAArch64CodegenPipelineImpl(OpPassManager& passManager,
                                         StringRef configPath,
                                         StringRef packedCPath,
                                         int64_t intraThreadCount)
    {
        passManager.addPass(atir::createAtirGemmEpilogueFusionPass());
        passManager.addPass(atir::createConvertAtirToLinalg());
        passManager.addPass(createKPGemmOneShotBufferize());
        passManager.addPass(createAArch64ResolveGemmPlan());
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
        const bool enablePrepack = !packedCPath.empty();
#else
        const bool enablePrepack = false;
#endif
        passManager.addPass(createAArch64SelectGemmStrategy(configPath,
                                                           enablePrepack,
                                                           intraThreadCount));
        passManager.addPass(createAArch64AutotuneGemmPlan());
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
        if (enablePrepack)
            passManager.addPass(createAArch64GemmPrepackRhs(packedCPath));
#endif
        passManager.addPass(createAArch64FinalizeGemmPlan());
        passManager.addPass(createAArch64GemmCacheBlocking());
        passManager.addPass(createAArch64GemmKernelTiling());
        passManager.addPass(createAArch64GemmLeafMaterialization());
        passManager.addPass(createAArch64GemmEpilogueLowering());
        passManager.addPass(createConvertLinalgToLoopsPass());
        passManager.addPass(createAArch64GemmMicrokernelLowering());
        passManager.addPass(createAArch64GemmABILowering());
        passManager.addPass(createAArch64VerifyGemmSchedule());
    }
    } // namespace

    void buildAArch64CodegenPipeline(OpPassManager& passManager)
    {
        buildAArch64CodegenPipelineImpl(passManager, {}, {}, 1);
    }

    void registerAArch64CodegenPipeline()
    {
        static PassPipelineRegistration<AArch64CodegenPipelineOptions> registration(
            "annc-aarch64-gemm-pipeline", "ANNC normal AArch64 GEMM lowering",
            [](OpPassManager &passManager,
               const AArch64CodegenPipelineOptions &options) {
                buildAArch64CodegenPipelineImpl(passManager, options.configPath,
                                                 options.packedCPath,
                                                 options.intraThreadCount);
            });
    }
} //namespace annc
