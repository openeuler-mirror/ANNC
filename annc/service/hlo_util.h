#ifndef AI_COMPILER_HLO_UTIL_H_
#define AI_COMPILER_HLO_UTIL_H_

#include <vector>

#include "xla/hlo/ir/hlo_module.h"

namespace xla {

static Literal get_matmul_info(HloInstruction* instr) {
  const Shape& lhs_shape = instr->operand(0)->shape();
  const Shape& rhs_shape = instr->operand(1)->shape();

  auto dot_dim_numbers = instr->dot_dimension_numbers();
  int64_t lhs_ctr_dim = dot_dim_numbers.lhs_contracting_dimensions(0);
  int64_t rhs_ctr_dim = dot_dim_numbers.rhs_contracting_dimensions(0);

  std::vector<int64_t> info(6);

  info[0] = lhs_shape.rank() == 3 ? lhs_shape.dimensions(0) : 1;  // batch size
  if (lhs_ctr_dim == lhs_shape.rank() - 1) {
    info[1] = lhs_shape.dimensions(lhs_shape.rank() - 2);  // M
    info[2] = lhs_shape.dimensions(lhs_shape.rank() - 1);  // K
  } else {
    info[1] = lhs_shape.dimensions(lhs_shape.rank() - 1);  // M
    info[2] = lhs_shape.dimensions(lhs_shape.rank() - 2);  // K
  }
  if (rhs_ctr_dim == rhs_shape.rank() - 2) {
    info[3] = rhs_shape.dimensions(rhs_shape.rank() - 1);  // N
  } else {
    info[3] = rhs_shape.dimensions(rhs_shape.rank() - 2);  // N
  }
  info[4] = lhs_ctr_dim;  // lhs_contracting_dimension, k axis
  info[5] = rhs_ctr_dim;  // rhs_contracting_dimension, k axis

  return LiteralUtil::CreateR1<int64_t>(info);
}

static Literal get_op_info(HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kDot:
      // {batch, m, k, n, lhs_contracting_dimension, rhs_contracting_dimension}
      return get_matmul_info(instr);
    default:
      break;
  }
  return LiteralUtil::CreateR1<int64_t>({});
}

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
        info.push_back(dot_dim_numbers.rhs_contracting_dimensions(0));
      }
    }
  }
  return LiteralUtil::CreateR1<int64_t>(info);
}

}  // namespace xla

#endif  // AI_COMPILER_HLO_UTIL_H_