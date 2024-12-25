#ifndef AI_COMPILER_KDNN_REWRITER_H_
#define AI_COMPILER_KDNN_REWRITER_H_

#include <string>
#include <vector>
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

void test_kdnn_rewriter(HloModule* module);

}  // namespace xla

#endif  // AI_COMPILER_KDNN_REWRITER_H_