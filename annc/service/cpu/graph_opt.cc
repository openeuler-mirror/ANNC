#include "annc/service/hlo_util.h"
#include "kdnn_rewriter.h"

namespace xla {
namespace cpu {

bool fused_sparse_embedding2(HloInstruction* mul) {
  if (mul->opcode() != HloOpcode::kMultiply) return false;
  IS_VALID(mul->operand(0)->opcode() == HloOpcode::kBroadcast &&
           mul->operand(1)->opcode() == HloOpcode::kGather)
  HloInstruction* broadcast = mul->mutable_operand(0);
  HloInstruction* gather = mul->mutable_operand(1);
  IS_VALID(gather->operand(1)->opcode() == HloOpcode::kSelect)
  HloInstruction* select = gather->mutable_operand(1);
  IS_VALID(broadcast->operand(0)->opcode() == HloOpcode::kConvert)
  HloInstruction* convert = broadcast->mutable_operand(0);
  IS_VALID(convert->operand(0)->opcode() == HloOpcode::kAnd)
  HloInstruction* and_op = convert->mutable_operand(0);
  IS_VALID(and_op->operand(0)->opcode() == HloOpcode::kCompare &&
           and_op->operand(1)->opcode() == HloOpcode::kCompare)
  HloInstruction* comp_0 = and_op->mutable_operand(0);
  HloInstruction* comp_1 = and_op->mutable_operand(1);
  IS_VALID(comp_1->operand(0)->opcode() == HloOpcode::kConvert &&
           comp_1->operand(1)->opcode() == HloOpcode::kBroadcast)
  HloInstruction* broadcast_1 = comp_1->mutable_operand(1);
  IS_VALID(broadcast_1->operand(0)->opcode() == HloOpcode::kConstant)
  HloInstruction* constant_1 = broadcast_1->mutable_operand(0);
  HloInstruction* broadcast_0 = comp_0->mutable_operand(1);
  IS_VALID(broadcast_0->operand(0)->opcode() == HloOpcode::kConstant)
  HloInstruction* constant_0 = broadcast_0->mutable_operand(0);

  std::vector<HloInstruction*> fused_instrs = {
      mul, broadcast, gather, select, convert, and_op, comp_0, 
      comp_1, broadcast_1, constant_1, broadcast_0, constant_0};

  HloComputation* parent = mul->parent();
  HloInstruction* fusion = parent->CreateFusionInstruction(
      fused_instrs, HloInstruction::FusionKind::kLoop);
  return true;
}

void register_sparse_embedding2(std::vector<KDnnRewriter>& rewriters,
                                RewriterType rewrite_type, int benefit = 1) {
  RewritePattern pattern("sparse_embedding2", HloOpcode::kMultiply);
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
