#ifndef AI_COMPILER_HLO_UTIL_H_
#define AI_COMPILER_HLO_UTIL_H_

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/literal_util.h"

namespace xla {

#define IS_VALID(condition) \
  if (!(condition)) return false;

static Literal get_param_info(const HloInstruction* arg_param) {
  const Shape& shape = arg_param->shape();
  std::vector<int64> info(shape.dimensions().begin(),
                            shape.dimensions().end());
  info.insert(info.end(), shape.layout().minor_to_major().begin(),
              shape.layout().minor_to_major().end());
  return LiteralUtil::CreateR1<int64>(info);
}

static void add_extra_operands(HloInstruction* fusion,
                               std::vector<HloInstruction*>& operands) {
  // insert constant operations here is not safe
  // create dummy root instruction outside for safety
  HloInstruction* root_instr = fusion->parent()->root_instruction();
  for (const HloInstruction* arg_param : fusion->fused_parameters()) {
    HloInstruction* constant = fusion->parent()->AddInstruction(
        HloInstruction::CreateConstant(get_param_info(arg_param)));
    operands.push_back(constant);
  }
}

}  // namespace xla

#endif  // AI_COMPILER_HLO_UTIL_H_
