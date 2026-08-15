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
    };

    void buildAArch64CodegenPipelineImpl(OpPassManager& passManager,
                                         StringRef configPath)
    {
        passManager.addPass(atir::createAtirGemmEpilogueFusionPass());
        passManager.addPass(atir::createConvertAtirToLinalg());
        passManager.addPass(createKPGemmOneShotBufferize());
        passManager.addPass(createAArch64ResolveGemmPlan());
        passManager.addPass(createAArch64SelectGemmStrategy(configPath));
        passManager.addPass(createAArch64AutotuneGemmPlan());
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
        buildAArch64CodegenPipelineImpl(passManager, {});
    }

    void registerAArch64CodegenPipeline()
    {
        static PassPipelineRegistration<AArch64CodegenPipelineOptions> registration(
            "annc-aarch64-gemm-pipeline", "ANNC normal AArch64 GEMM lowering",
            [](OpPassManager &passManager,
               const AArch64CodegenPipelineOptions &options) {
                buildAArch64CodegenPipelineImpl(passManager,
                                                 options.configPath);
            });
    }
} //namespace annc
