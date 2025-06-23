#ifndef XLA_SERVIVE_EMBEDDING_SIMPLIFY_H_
#define XLA_SERVIVE_EMBEDDING_SIMPLIFY_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

class EmbeddingSimplify : public HloModulePass {
public:
  ~EmbeddingSimplify() override = default;
  absl::string_view name() const override { return "embedding-simplify"; }

  using HloPassInterface::Run;
  StatusOr<bool>
  Run(HloModule *module,
      const absl::flat_hash_set<absl::string_view> &execution_threads) override;

private:
  HloInstruction *hlo;
  HloInstruction *broadcast;
  HloInstruction *gather;
  HloInstruction *mul;
  HloInstruction *select;
  HloInstruction *and0;
  HloInstruction *and1;
  HloInstruction *comp0;
  HloInstruction *comp1;
  HloInstruction *constant0;
  HloInstruction *constant1;
};
} // namespace xla

#endif // XLA_SERVIVE_EMBEDDING_SIMPLIFY_H_
