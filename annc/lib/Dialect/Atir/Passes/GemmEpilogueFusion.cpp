#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "Dialect/Atir/Passes/Passes.h"

namespace atir {
namespace {

class AtirGemmEpilogueFusion
    : public AtirGemmEpilogueFusionBase<AtirGemmEpilogueFusion> {
 public:
  using Base::Base;

  void runOnOperation() override {
    getOperation()->walk([&](MatMulOp matmul) {
      if (matmul->hasAttr(kGemmEpilogueCandidateAttr)) return;
      auto program = discoverOrderedEpilogue(matmul);
      if (failed(program)) return;
      Builder builder(matmul.getContext());
      matmul->setAttr(kGemmEpilogueCandidateAttr,
                      buildEpilogueCandidateAttr(builder, *program));
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAtirGemmEpilogueFusionPass() {
  return std::make_unique<AtirGemmEpilogueFusion>();
}

}  // namespace atir
