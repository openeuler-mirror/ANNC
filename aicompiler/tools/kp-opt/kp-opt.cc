#include "aicompiler/service/cpu/kdnn_rewriter.h"
#include "xla/service/hlo_parser.h"

#include <memory>
#include <iostream>

using namespace xla;

int main(int argc, char** argv) {

  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 3}
}
)";

  absl::StatusOr<std::unique_ptr<xla::HloModule>> 
    m_ = ParseAndReturnUnverifiedModule(hlo_text);

  test_kdnn_rewriter(m_->get());

  return 0;
}