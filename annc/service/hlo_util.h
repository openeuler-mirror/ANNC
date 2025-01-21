#ifndef AI_COMPILER_HLO_UTIL_H_
#define AI_COMPILER_HLO_UTIL_H_

#include <vector>

#include "xla/hlo/ir/hlo_module.h"

namespace xla {

static Literal get_param_info(const HloInstruction* arg_param) {
  const Shape& shape = arg_param->shape();
  std::vector<int64_t> info(shape.dimensions().begin(),
                            shape.dimensions().end());
  for (const HloInstruction* user : arg_param->users()) {
    if (user->opcode() == HloOpcode::kDot) {
      auto dot_dim_numbers = user->dot_dimension_numbers();
      int64_t user_index = user->operand_index(arg_param);
      if (user_index == 0) {
        // lhs matrix
        info.push_back(dot_dim_numbers.lhs_contracting_dimensions(0));
      } else {
        // rhs matrix
        if (info.size() == 1) { info.push_back(1); }
        info.push_back(dot_dim_numbers.rhs_contracting_dimensions(0));
      }
    }
  }
  return LiteralUtil::CreateR1<int64_t>(info);
}

}  // namespace xla

#endif  // AI_COMPILER_HLO_UTIL_H_