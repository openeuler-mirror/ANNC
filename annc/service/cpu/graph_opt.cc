#include "annc/service/kdnn_util.h"
#include "kdnn_rewriter.h"

namespace xla {
namespace cpu {

bool fused_sparse_embedding2(HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kMultiply) return false;
  std::vector<HloInstruction*> fused_instrs = {instr};
  if (instr->mutable_operand(0)->opcode() != HloOpcode::kBroadcast &&
      instr->mutable_operand(1)->opcode() != HloOpcode::kParameter)
      return false;
  HloInstruction* broadcast = instr->mutable_operand(0);
  HloInstruction* parameter = instr->mutable_operand(1);
  fused_instrs.push_back(broadcast);
  fused_instrs.push_back(parameter);
  if (broadcast->mutable_operand(0)->opcode() != HloOpcode::kConvert)
    return false;
  HloInstruction* convert = broadcast->mutable_operand(0);
  fused_instrs.push_back(convert);
  if (convert->mutable_operand(0)->opcode() != HloOpcode::kAnd)
    return false;
  HloInstruction* and_op = convert->mutable_operand(0);
  fused_instrs.push_back(and_op);
  if (and_op->mutable_operand(0)->opcode() != HloOpcode::kCompare &&
      and_op->mutable_operand(1)->opcode() != HloOpcode::kCompare)
      return false;
  HloInstruction* comp_0 = and_op->mutable_operand(0);
  HloInstruction* comp_1 = and_op->mutable_operand(1);
  fused_instrs.push_back(comp_0);
  fused_instrs.push_back(comp_1);
  if (comp_1->mutable_operand(0)->opcode() != HloOpcode::kConvert ||
      comp_1->mutable_operand(1)->opcode() != HloOpcode::kBroadcast)
    return false;
  HloInstruction* broadcast_1 = comp_1->mutable_operand(1);
  fused_instrs.push_back(broadcast_1);
  if (broadcast_1->mutable_operand(0)->opcode() != HloOpcode::kConstant)
    return false;
  HloInstruction* constant_1 = broadcast_1->mutable_operand(0);
  fused_instrs.push_back(constant_1);
  HloInstruction* broadcast_0 = comp_0->mutable_operand(1);
  fused_instrs.push_back(broadcast_0);
  if (broadcast_0->mutable_operand(0)->opcode() != HloOpcode::kConstant)
    return false;
  HloInstruction* constant_0 = broadcast_0->mutable_operand(0);
  fused_instrs.push_back(constant_0);

  HloComputation* parent = instr->parent();
  HloInstruction* fusion = parent->CreateFusionInstruction(
    fused_instrs, HloInstruction::FusionKind::kLoop);
  return true;
}

void register_sparse_embedding2(std::vector<KDnnRewriter>& rewriters,
                                RewriterType rewrite_type, int benefit = 1) {
  // matmul+add+relu+matmul+add
  RewritePattern pattern("__sparse_embedding2", HloOpcode::kDot);
  pattern.custom_rewriter = fused_sparse_embedding2;

  auto rewriter = KDnnRewriter(benefit, pattern, rewrite_type);
  rewriters.push_back(rewriter);
}

void register_graph_opt_rewriters(std::vector<KDnnRewriter>& rewriters) {
#if defined(ANNC_ENABLED_GRAPH_OPT)
  register_sparse_embedding2(rewriters, RewriterType::FUSED_OPERATION, 3);
#endif
  std::sort(rewriters.begin(), rewriters.end(), compare_rewriter);
}

}  // namespace cpu
}  // namespace xla
