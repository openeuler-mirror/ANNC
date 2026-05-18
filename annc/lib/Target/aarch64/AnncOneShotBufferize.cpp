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
            bufferization::BufferizationState bufferizationState;
            if (failed(runOneShotBufferize(op, options, bufferizationState))) {
              return signalPassFailure();
            }
        }
    };

    std::unique_ptr<mlir::Pass> createKPGemmOneShotBufferize()
    {
        return std::make_unique<KPGemmOneShotBufferize>();
    }

}
