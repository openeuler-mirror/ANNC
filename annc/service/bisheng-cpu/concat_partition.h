#ifndef XLA_SERVICE_CONCAT_PARTITION_H_
#define XLA_SERVICE_CONCAT_PARTITION_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

class ConcatPartition : public HloModulePass {
 public:
  ~ConcatPartition() override = default;
  absl::string_view name() const override { return "concat-partition"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  HloInstruction* CreateEmbeddingv2(HloInstruction* input, int change_dim,
                                    int newSliceSize, HloComputation* comp);

 private:
  HloInstruction* mul = nullptr;
  HloInstruction* broadcast = nullptr;
  HloInstruction* gather = nullptr;
  HloInstruction* select = nullptr;
  HloInstruction* convert = nullptr;
  HloInstruction* andOp = nullptr;
  HloInstruction* comp_0 = nullptr;
  HloInstruction* comp_1 = nullptr;
  HloInstruction* broadcast_1 = nullptr;
  HloInstruction* constant_1 = nullptr;
  HloInstruction* broadcast_0 = nullptr;
  HloInstruction* constant_0 = nullptr;
  HloInstruction* convert_1 = nullptr;
  HloInstruction* concat = nullptr;
};
}  // namespace xla

#endif  // XLA_SERVICE_CONCAT_PARTITION_H_
