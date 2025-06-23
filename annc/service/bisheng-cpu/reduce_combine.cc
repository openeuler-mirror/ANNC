#include "annc/service/bisheng-cpu/reduce_combine.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"
#include <vector>

namespace xla {
namespace match = ::xla::match;

bool IsAddReducer(const HloInstruction *reduce) {
  return Match(reduce->to_apply()->root_instruction(), match::Add());
}

StatusOr<bool> ReduceCombine::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (auto *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto *instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kReduceWindow)
        continue;
      HloInstruction *reduce_window = instr;
      if (reduce_window->user_count() != 1)
        continue;
      HloInstruction *user = reduce_window->users()[0];
      if (user->opcode() != HloOpcode::kReduce)
        continue;

      HloInstruction *reduce = user;
      if (!IsAddReducer(reduce) || !IsAddReducer(reduce_window))
        continue;

      const Window &window = reduce_window->window();
      const Shape &input_shape = reduce_window->operand(0)->shape();
      std::vector<int64_t> reduced_dims;
      bool can_merge = true;

      // Check window parameters
      const auto &reduceOp_dims = reduce->dimensions();
      for (int64_t i = 0; i < window.dimensions_size(); ++i) {
        const WindowDimension &wd = window.dimensions(i);
        int64_t input_dim = input_shape.dimensions(i);
        if (wd.stride() != wd.size() || wd.padding_high() != 0 ||
            wd.padding_low() != 0 || (input_dim % wd.size()) != 0) {
          can_merge = false;
          break;
        }
        if (wd.size() == 1) {
          continue;
        } else if (std::find(reduceOp_dims.begin(), reduceOp_dims.end(), i) !=
                   reduceOp_dims.end()) {
          reduced_dims.push_back(i);
        } else {
          can_merge = false;
          break;
        }
      }
      if (!can_merge)
        continue;
      VLOG(2) << "Before ReduceCombine pass : " << computation->ToString();

      // Merge reduce dimensions
      std::unordered_set<int64_t> merged_dims(reduced_dims.begin(),
                                              reduced_dims.end());
      for (int64_t dim : reduce->dimensions()) {
        merged_dims.insert(dim);
      }
      std::vector<int64_t> merged_dims_vec(merged_dims.begin(),
                                           merged_dims.end());
      std::sort(merged_dims_vec.begin(), merged_dims_vec.end());

      // Verify output shape
      HloInstruction *operand = reduce_window->mutable_operand(0);
      Shape expected_shape =
          ShapeUtil::DeleteDimensions(merged_dims_vec, operand->shape());
      if (expected_shape != reduce->shape())
        continue;

      // Create a new Reduce to replace old patterns
      HloComputation *reducer = reduce->to_apply();
      HloInstruction *new_reduce =
          computation->AddInstruction(HloInstruction::CreateReduce(
              reduce->shape(), operand, reduce->mutable_operand(1),
              merged_dims_vec, reducer));
      TF_RETURN_IF_ERROR(reduce->ReplaceAllUsesWith(new_reduce));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(reduce));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(reduce_window));
      changed = true;
      VLOG(2) << "After ReduceCombine pass : " << computation->ToString();
    }
  }
  return changed;
}
} // namespace xla
