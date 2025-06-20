#include "annc/service/bisheng-cpu/concat_partition.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include <vector>

namespace xla {
namespace match = ::xla::match;

Shape replaceDim(Shape shape, int64_t dim, int64_t new_size) {
  shape.mutable_dimensions()[dim] = new_size;
  return ShapeUtil::MakeShape(shape.element_type(), shape.mutable_dimensions());
}

bool checkShapeConsistent(Shape shape1, Shape shape2) {
  int shape_size =
      std::min(shape1.dimensions().size(), shape2.dimensions().size());
  for (int i = 0; i < shape_size; i++) {
    if (shape1.dimensions()[i] != shape2.dimensions()[i])
      return false;
  }
  return true;
}

HloInstruction *ConcatPartition::CreateEmbeddingv2(HloInstruction *input,
                                                   int change_dim,
                                                   int newSliceSize,
                                                   HloComputation *comp) {
  HloInstruction *param_0 = input;
  HloInstruction *param_1 = broadcast_0->mutable_operand(0);
  HloInstruction *param_2 = broadcast_1->mutable_operand(0);
  HloInstruction *param_3 = gather->mutable_operand(0);
  HloInstruction *param_4 = select->mutable_operand(2)->mutable_operand(0);

  // Create operators to new shape
  auto new_convert_1_shape =
      replaceDim(convert_1->shape(), change_dim, newSliceSize);
  HloInstruction *new_convert_1 = comp->AddInstruction(
      convert_1->CloneWithNewOperands(new_convert_1_shape, {param_0}));
  // Create new broadcast_0, broadcast_1
  HloInstruction *new_broadcast_0 = comp->AddInstruction(
      broadcast_0->CloneWithNewOperands(new_convert_1_shape, {param_1}));
  HloInstruction *new_broadcast_1 = comp->AddInstruction(
      broadcast_1->CloneWithNewOperands(new_convert_1_shape, {param_2}));
  // Create new comp_0, comp_1
  HloInstruction *new_comp_0 =
      comp->AddInstruction(comp_0->CloneWithNewOperands(
          replaceDim(comp_0->shape(), change_dim, newSliceSize),
          {new_convert_1, new_broadcast_0}));
  HloInstruction *new_comp_1 =
      comp->AddInstruction(comp_1->CloneWithNewOperands(
          replaceDim(comp_1->shape(), change_dim, newSliceSize),
          {new_convert_1, new_broadcast_1}));
  // Create new and
  HloInstruction *new_and = comp->AddInstruction(andOp->CloneWithNewOperands(
      new_comp_0->shape(), {new_comp_0, new_comp_1}));
  // Create new convert and broadcast
  HloInstruction *new_convert =
      comp->AddInstruction(convert->CloneWithNewOperands(
          replaceDim(convert->shape(), change_dim, newSliceSize), {new_and}));
  HloInstruction *new_broadcast =
      comp->AddInstruction(broadcast->CloneWithNewOperands(
          replaceDim(broadcast->shape(), change_dim, newSliceSize),
          {new_convert}));
  // Create new select
  HloInstruction *select_operand2 = select->mutable_operand(2);
  HloInstruction *new_select_operand2 = comp->AddInstruction(
      select_operand2->CloneWithNewOperands(new_convert_1_shape, {param_4}));
  HloInstruction *new_select =
      comp->AddInstruction(select->CloneWithNewOperands(
          new_convert_1_shape, {new_and, new_convert_1, new_select_operand2}));
  // Create new gather
  HloInstruction *new_gather =
      comp->AddInstruction(gather->CloneWithNewOperands(
          replaceDim(gather->shape(), change_dim, newSliceSize),
          {param_3, new_select}));
  // Create new mul
  HloInstruction *new_mul = comp->AddInstruction(mul->CloneWithNewOperands(
      new_gather->shape(), {new_broadcast, new_gather}));
  return new_mul;
}

StatusOr<bool> ConcatPartition::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  LOG(INFO) << "Running ConcatPartition pass";
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto *inst : computation->MakeInstructionPostOrder()) {
      // ---------------------Mathc taget pattern ---------------
      if (inst->opcode() != HloOpcode::kMultiply)
        continue;
      std::vector<HloInstruction *> front_slices, back_slices;
      auto convert_1_pattern =
          match::Convert(&convert_1, match::Concatenate(&concat));

      auto and_pattern = match::And(
          &andOp,
          match::Compare(&comp_0, convert_1_pattern, match::Broadcast()),
          match::Compare(&comp_1, convert_1_pattern, match::Broadcast()));

      auto select_pattern =
          match::Select(&select, and_pattern, convert_1_pattern,
                        match::Broadcast(match::Conatant()));
      auto gather_pattern =
          match::Gather(&gather, match::Parameter(), select_pattern);
      // Match target patterns with concat + embedding + split graph
      bool isMatched = Match(
          inst, match::Multiply(
                    &mul,
                    match::Broadcast(&broadcast,
                                     match::Convert(&convert, and_pattern)),
                    gather_pattern));
      if (!isMatched)
        continue;
      broadcast_0 = comp_0->mutable_operand(1);
      broadcast_1 = comp_1->mutable_operand(1);

      for (HloInstruction *user : mul->users()) {
        if (user->opcode() == HloOpcode::kSlice) {
          back_slices.push_back(user);
        } else {
          isMatched = false;
          break;
        }
      }
      const auto &front_operands = concat->mutable_operands();
      for (HloInstruction *operand : front_operands) {
        front_slices.push_back(operand);
      }
      int slice_size = front_slices.size();
      // Check that front slices are consistent with back slices
      if (!isMatched || slice_size < 2 || slice_size != back_slices.size())
        continue;

      std::vector<HloInstruction *> newSlices;
      std::vector<int> concatIndex;
      std::vector<int> noConcatIndex;
      // back_slices has slice with two dimensions of shape (? x 1) and other
      // different shape of (? x ?)
      for (int i = 0; i < slice_size; i++) {
        if (checkShapeConsistent(front_slices[i]->shape(),
                                 back_slices[i]->shape())) {
          isMatched |= true;
          if (front_slices[i]->shape().dimensions()[1] != 1)
            noConcatIndex.push_back(i);
          else {
            concatIndex.push_back(i);
            newSlices.push_back(front_slices[i]);
          }
        }
      }
      if (noConcatIndex.empty())
        continue;
      VLOG(2) << "ConcatPartition pass before opt: " << computation->ToString();
      // -------------------- Run rewrite optimization ---------------------
      // Step1: Create new concatenate operator
      int change_dim = concat->concatenate_dimension();
      int newSliceSize = newSlices.size();
      Shape newConcatShape =
          replaceDim(concat->shape(), change_dim, newSliceSize);
      HloInstruction *newConcat = computation->AddInstruction(
          concat->CloneWithNewOperands(newConcatShape, newSlices));
      HloInstruction *newMul =
          CreateEmbeddingV2(newConcat, change_dim, newSliceSize, computation);
      // Step2: Replace back_slice of shape (? x 1) with embedding operator
      for (int i = 0; i < concatIndex.size(); i++) {
        int index = concatIndex[i];
        HloInstruction *slice = back_slices[index];
        std::vector<int64_t> new_starts = slice->slice_starts();
        std::vector<int64_t> new_limits = slice->slice_limits();
        new_limits[change_dim] =
            new_limits[change_dim] - new_starts[change_dim] + i;
        new_starts[change_dim] = i;
        HloInstruction *newSlice = computation->AddInstruction(
            HloInstruction::CreateSlice(slice->shape(), newMul, new_starts,
                                        new_limits, slice->slice_strides()));
        TR_RETURN_IF_ERROR(computation->ReplaceInstruction(slice, newSlice));
      }

      // Step3: Process other slices with other different shape (? x ?)
      for (int i = 0; i < noConcatIndex.size(); i++) {
        int index = noConcatIndex[i];
        HloInstruction *slice = front_slices[index];
        int slice_size = slice->shape().dimensions()[change_dim];
        HloInstruction *newMul =
            CreateEmbeddingv2(slice, change_dim, slice_size, computation);
        HloInstruction *backSlice = back_slices[index];
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(backSlice, newMul));
      }
      changed = true;
      VLOG(2) << "ConcatePartition pass after opt: " << computation->ToString();
    }
  }
  return changed;
}

} // namespace xla
