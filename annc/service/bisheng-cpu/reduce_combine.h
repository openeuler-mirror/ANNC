#ifndef XLA_SERVICE_REDUCE_COMBINE_H_
#define XLA_SERVICE_REDUCE_COMBINE_H_

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
} // namespace xla

#endif // XLA_SERVICE_REDUCE_COMBINE_H_
