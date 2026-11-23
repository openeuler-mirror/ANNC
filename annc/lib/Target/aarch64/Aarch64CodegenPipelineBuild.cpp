#include "Conversion/Passes.h"
#include "Support/ContextABISwitch.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
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
        PassOptions::Option<std::string> jitInputsPath{
            *this, "jit-inputs",
            llvm::cl::desc("JSON array of immutable JIT argument indices "
                           "for runtime packing")};
    };

    class AArch64PipelineFeatureGuard
        : public PassWrapper<AArch64PipelineFeatureGuard,
                             OperationPass<ModuleOp>> {
     public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
            AArch64PipelineFeatureGuard)

        void runOnOperation() override {
#if defined(ANNC_ENABLE_TF2ATIR_V2)
            getOperation().emitError(
                "AArch64 GEMM pipeline requires both "
                "ANNC_ENABLE_TF2ATIR_V2 and "
                "ANNC_ENABLE_LOWERING_CONTEXT_ABI; this build has V2 enabled "
                "but the context ABI route is disabled");
#else
            getOperation().emitError(
                "AArch64 GEMM pipeline requires both "
                "ANNC_ENABLE_TF2ATIR_V2 and "
                "ANNC_ENABLE_LOWERING_CONTEXT_ABI; the context ABI route is "
                "enabled but this build has V2 disabled");
#endif
            signalPassFailure();
        }
    };

    void buildAArch64CodegenPipelineImpl(OpPassManager& passManager,
                                         StringRef configPath,
                                         StringRef packedCPath,
                                         int64_t intraThreadCount,
                                         StringRef jitInputsPath = {})
    {
#if defined(ANNC_ENABLE_TF2ATIR_V2)
        if (!annc::loweringContextABIEnabled()) {
            passManager.addPass(
                std::make_unique<AArch64PipelineFeatureGuard>());
            return;
        }

        passManager.addPass(atir::createAnncGemmEpilogueCommitPass());
        passManager.addPass(atir::createConvertContextKernelABIPass());
        passManager.addPass(atir::createConvertAtirV2ToLinalg());
#else
        if (annc::loweringContextABIEnabled()) {
            passManager.addPass(
                std::make_unique<AArch64PipelineFeatureGuard>());
            return;
        }

        passManager.addPass(createAArch64AtirGemmEpilogueCommit());
        passManager.addPass(atir::createLowerContextKernelEntriesPass());
        passManager.addPass(atir::createConvertAtirToLinalg());
#endif
        passManager.addPass(createKPGemmOneShotBufferize());
        if (annc::loweringContextABIEnabled()) {
            bufferization::BufferDeallocationPipelineOptions options;
            bufferization::buildBufferDeallocationPipeline(passManager,
                                                            options);
        }
        passManager.addPass(createAArch64ExtractGemmProblem());
        const bool enablePrepack = !packedCPath.empty();
        passManager.addPass(createAArch64SelectGemmStrategy(configPath,
                                                           enablePrepack,
                                                           intraThreadCount));
        passManager.addPass(createAArch64GemmThreadPlanning());
        if (enablePrepack)
            passManager.addPass(createAArch64GemmPrepackRhs(
                packedCPath, jitInputsPath));
        passManager.addPass(createAArch64GemmThreadTiling());
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
                buildAArch64CodegenPipelineImpl(passManager,
                                               options.configPath,
                                               options.packedCPath,
                                               options.intraThreadCount,
                                               options.jitInputsPath);
            });
    }
} //namespace annc
