#include "annc/service/bisheng-cpu/embedding_simplify.h"

#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"

#include "xla/status_macros.h"
#include "xla/types.h"

namespace xla {
namespace match = ::xla::match;

static Shape replaceDim(Shape shape, int64_t dim, int64_t new_size) {
  shape.mutable_dimensions()[dim] = new_size;
  return ShapeUtil::MakeShape(shape.element_type(), shape.mutable_dimensions());
}

StatusOr<bool> EmbeddingSimplify::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  LOG(INFO) << "Running EmbeddingSimplify pass";
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto *inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kMultiply)
        continue;
      auto and_pattern = match::And(
          &and0,
          match::Compare(&comp0, match::Op(),
                         match::Broadcast(match::Constant(&constant0))),
          match::Compare(&comp1, match::Op(),
                         match::Broadcast(match::Constant(&constant1))));
      auto select_pattern =
          match::Select(&select, and_pattern, match::Op(), match::Broadcast());
      auto gather_pattern =
          match::Gather(&gather, match::Parameter(), select_pattern);
      // Match embeddingv2 graph
      bool isMatched = Match(
          inst, match::Multiply(
                    &mul,
                    match::Broadcast(&broadcast,
                                     match::Convert(match::And(
                                         &and1, match::Op(), match::Op()))),
                    gather_pattern));
      if (!isMatched || and0 != and1)
        continue;

      // ------------------ Run Optimization -------------------
      HloInstruction *gather_operand0 = gather->mutable_operand(0);
      auto gather_operand0_shape = gather_operand0->shape();
      auto gather_operand0_dim0 = gather_operand0_shape.mutable_dimensions()[0];
      auto new_gather_operand0_dim0 = gather_operand0_dim0 + 1;
      if (gather_operand0_shape.dimensions_size() != 2) {
        VLOG(2) << "gather operand 0 dimensions isze is not 2";
        continue;
      }

      const Literal &literal0 = constant0->literal();
      if (literal0.shape().rank() > 1 || literal0.element_count() != 1 ||
          !literal0.shape().IsInteger())
        continue;
      const Literal &literal1 = constant1->literal();
      if (literal1.shape().rank() > 1 || literal1.element_count() != 1 ||
          !literal1.shape().IsInteger())
        continue;

      int64_t value0 = *literal0.GetFirstInteger();
      int64_t value1 = *literal1.GetFirstInteger();
      if (!(value0 == 0 &&
            comp0->comparison_direction() == ComparisonDirection::kGe &&
            value1 == gather_operand0_dim0 &&
            comp1->comparison_direction() == ComparisonDirection::kLt) &&
          !(value1 == 0 &&
            comp1->comparison_direction() == ComparisonDirection::kGe &&
            value0 == gather_operand0_dim0 &&
            comp0->comparison_direction() == ComparisonDirection::kLt))
        continue;
      VLOG(2) << "Embedding pass before opt: " << computation->ToString();
      HloInstruction *zero = computation->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
      Shape dummy_slice_shape = replaceDim(gather_operand0_shape, 0, 1);
      HloInstruction *zero_broadcast = computation->AddInstruction(
          HloInstruction::CreateBroadcast({dummy_slice_shape}, zero, {}));
      Shape new_operand0_shape =
          replaceDim(gather_operand0_shape, 0, new_gather_operand0_dim0);
      HloInstruction *newOperand0 =
          computation->AddInstruction(HloInstruction::CreateConcatenate(
              new_operand0_shape, {gather_operand0, zero_broadcast}, 0));

      HloInstruction *select_operand2 = select->mutable_operand(2);
      HloInstruction *old_constant = select_operand2->mutable_operand(0);

      auto old_constant_shape = old_constant->shape();
      auto literal = Literal::CreateFromShape(old_constant_shape);
      if (old_constant_shape.element_type() != PrimitiveType::S32)
        continue;
      auto literal_data = literal.data<int32_t>();
      if (literal_data.size() != 1)
        continue;
      literal_data[0] = gather_operand0_dim0;

      HloInstruction *new_constant_1 = computation->AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
      HloInstruction *new_select_operand2 =
          computation->AddInstruction(select_operand2->CloneWithNewOperands(
              select_operand2->shape(), {new_constant_1}));

      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(select_operand2,
                                                         new_select_operand2));
      HloInstruction *new_gather = computation->AddInstruction(
          gather->CloneWithNewOperands(gather->shape(), {newOperand0, select}));
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(mul, new_gather));
      changed = true;
      VLOG(2) << "Embedding pass after opt: " << computation->ToString();
    }
  }
  return changed;
}

} // namespace xla
