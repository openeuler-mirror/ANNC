#include "kdnn_rewriter.h"

#include "annc/service/hlo_util.h"

namespace xla {
namespace cpu {

bool match_op_pattern(HloInstruction* instr, RewritePattern& pattern,
                      std::vector<HloInstruction*>& fused_instrs) {
  if (!pattern.valid) return true;
  if (instr->opcode() != pattern.opcode) return false;

  bool should_rewrite = true;
  if (instr->operand_count() < pattern.dims.size()) return false;
  for (size_t i = 0; i < pattern.dims.size(); i++) {
    const Shape& shape = instr->operand(i)->shape();
    should_rewrite &= LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
    should_rewrite &= (shape.rank() <= pattern.dims[i]);
    should_rewrite &= (shape.element_type() == pattern.dtypes[i]);
    if (!should_rewrite) return false;
  }
  auto& users = instr->users();
  for (size_t i = 0; i < pattern.next_patterns.size(); i++) {
    if (!match_op_pattern(users[i], pattern.next_patterns[i], fused_instrs))
      return false;
  }
  fused_instrs.push_back(instr);
  return should_rewrite;
}

void add_extra_operands(HloInstruction* fusion,
                        std::vector<HloInstruction*>& operands) {
  // insert constant operations here is not safe
  // create dummy root instruction outside for safety
  HloInstruction* root_instr = fusion->parent()->root_instruction();
  for (const HloInstruction* arg_param : fusion->fused_parameters()) {
    HloInstruction* constant = root_instr->AddInstruction(
        HloInstruction::CreateConstant(get_param_info(arg_param)));
    operands.push_back(constant);
  }
}

bool KDnnRewriter::execute(HloInstruction* instr) {
  if (instr->HasControlDependencies()) return false;

  std::vector<HloInstruction*> fused_instrs;
  if (!match_op_pattern(instr, pattern_, fused_instrs)) return false;

  HloComputation* parent = instr->parent();
  HloInstruction* fusion = parent->CreateFusionInstruction(
      fused_instrs, HloInstruction::FusionKind::kInput);

  if (type_ == RewriterType::DNN_CUSTOM_CALL) {
    std::vector<HloInstruction*> operands(fusion->operands().begin(),
                                          fusion->operands().end());
    add_extra_operands(fusion, operands);
    HloInstruction* custom_call =
        fusion->AddInstruction(HloInstruction::CreateCustomCall(
            fusion->shape(), operands, pattern_.name));
    return parent->ReplaceInstruction(fusion, custom_call, false).ok();
  }
  return true;
}

bool compare_rewriter(const KDnnRewriter& a, const KDnnRewriter& b) {
  return a.benefit() > b.benefit();
}

#define RUN_KDNN_FUSION_PASS(pass_class, rewriter_visiter)               \
  StatusOr<bool> pass_class::Run(                                        \
      HloModule* module,                                                 \
      const absl::flat_hash_set<absl::string_view>& execution_threads) { \
    rewriter_visiter visitor;                                            \
    return visitor.RunOnModule(module, execution_threads);               \
  }

RUN_KDNN_FUSION_PASS(KDnnFusionBeforeHloLayoutAssign,
                     KDnnBeforeHloLayoutAssignRewriterVisitor)

RUN_KDNN_FUSION_PASS(KDnnFusionAfterHloLayoutAssign,
                     KDnnAfterHloLayoutAssignRewriterVisitor)

RUN_KDNN_FUSION_PASS(KDnnFusionAfterRunBackend,
                     KDnnAfterRunBackendRewriterVisitor)

}  // namespace cpu
}  // namespace xla