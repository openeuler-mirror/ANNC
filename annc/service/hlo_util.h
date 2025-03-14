#ifndef AI_COMPILER_HLO_UTIL_H_
#define AI_COMPILER_HLO_UTIL_H_

#include <vector>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"

namespace xla {

#define IS_VALID(condition) if (!(condition)) return false;

static Literal get_param_info(const HloInstruction* arg_param) {
  const Shape& shape = arg_param->shape();
  std::vector<int64_t> info(shape.dimensions().begin(),
                            shape.dimensions().end());
  info.insert(info.end(), shape.layout().minor_to_major().begin(),
              shape.layout().minor_to_major().end());
  return LiteralUtil::CreateR1<int64_t>(info);
}

}  // namespace xla

#endif  // AI_COMPILER_HLO_UTIL_H_