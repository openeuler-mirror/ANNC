#include "kdnn_rewriter.h"

namespace xla {

void test_kdnn_rewriter(HloModule* module) {
  printf("%s\n", module->ToString().c_str());
}

}  // namespace xla