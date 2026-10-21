#include "Support/LLVMCompat.h"
#include "Target/aarch64/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

namespace annc
{
    class KPGemmOneShotBufferize : public KPGemmOneShotBufferizeBase<KPGemmOneShotBufferize>
    {
        using Base::Base;

        void runOnOperation() override
        {
            ModuleOp op = getOperation();

            bufferization::OneShotBufferizationOptions options;
            if (failed(annc::runOneShotBufferize(op, options))) {
              return signalPassFailure();
            }
        }
    };

    std::unique_ptr<mlir::Pass> createKPGemmOneShotBufferize()
    {
        return std::make_unique<KPGemmOneShotBufferize>();
    }

}
