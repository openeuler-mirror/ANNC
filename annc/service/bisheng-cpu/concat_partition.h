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
  StatusOr<bool>
  Run(HloModule *module,
      const absl::flat_hash_set<absl::string_view> &execution_threads) override;

  HloInstruction *CreateEmbeddingv2(HloInstruction *input, int change_dim,
                                    int newSliceSize, HloComputation *comp);

private:
  HloInstruction *hlo;
  HloInstruction *mul;
  HloInstruction *broadcast;
  HloInstruction *gather;
  HloInstruction *select;
  HloInstruction *convert;
  HloInstruction *andOp;
  HloInstruction *comp_0;
  HloInstruction *comp_1;
  HloInstruction *broadcast_1;
  HloInstruction *constant_1;
  HloInstruction *broadcast_0;
  HloInstruction *constant_0;
  HloInstruction *convert_1;
  HloInstruction *concat;
};
} // namespace xla

#endif // XLA_SERVICE_CONCAT_PARTITION_H_
