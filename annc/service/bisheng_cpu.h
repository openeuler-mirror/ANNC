#ifndef XLA_SERVICE_BISHENG_CPU_H_
#define XLA_SERVICE_BISHENG_CPU_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {
class ReduceCombine : public HloModulePass {
public:
  ~ReduceCombine() override = default;
  absl::string_view name() const override { return "reduce-combine"; }

  using HloPassInterface::Run;
  StatusOr<bool>
  Run(HloModule *module,
      const absl::flat_hash_set<absl::string_view> &execution_threads) override;
};

class EmbeddingSimplify : public HloModulePass {
public:
  ~EmbeddingSimplify() override = default;
  absl::string_view name() const override { return "embedding-simplify"; }

  using HloPassInterface::Run;
  StatusOr<bool>
  Run(HloModule *module,
      const absl::flat_hash_set<absl::string_view> &execution_threads) override;
};

class ConcatPartition : public HloModulePass {
public:
  ~ConcatPartition() override = default;
  absl::string_view name() const override { return "concat-partition"; }

  using HloPassInterface::Run;
  StatusOr<bool>
  Run(HloModule *module,
      const absl::flat_hash_set<absl::string_view> &execution_threads) override;

  HloInstruction *CreateEmbeddingv2(HloInstruction *input, int change_dim,
                                    int newSliceSize, HloComputation *comp);
};

#define ADD_GRAPH_OPT_PASSES()                                                 \
  pipeline.AddPass<ConcatPartition>();                                         \
  pipeline.AddPass<EmbeddingSimplify>();                                       \
  pipeline.AddPass<ReduceCombine>();                                           \
  pipeline.AddPass<HloDCE>();

} // namespace xla

#endif // XLA_SERVICE_BISHENG_CPU_H_
