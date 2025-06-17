#include "annc/service/hlo_util.h"
#include "kdnn_rewriter.h"
#include <omp.h>

namespace xla {
namespace cpu {

bool fused_sparse_embedding2(HloInstruction* mul) {
  IS_VALID(mul->opcode() == HloOpcode::kMultiply)
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
      mul,    broadcast, gather,      select,     convert,     and_op,
      comp_0, comp_1,    broadcast_1, constant_1, broadcast_0, constant_0};

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

bool fused_pooling(HloInstruction* select) {
  IS_VALID(select->opcode() == HloOpcode::kSelect)
  IS_VALID(select->operand(0)->opcode() == HloOpcode::kBroadcast &&
           select->operand(1)->opcode() == HloOpcode::kBroadcast &&
           select->operand(2)->opcode() == HloOpcode::kDivide)
  HloInstruction* broadcast_0 = select->mutable_operand(0);
  HloInstruction* divide = select->mutable_operand(2);
  IS_VALID(divide->operand(0)->opcode() == HloOpcode::kReshape &&
           divide->operand(1)->opcode() == HloOpcode::kBroadcast)
  HloInstruction* reshape = divide->mutable_operand(0);
  HloInstruction* broadcast_1 = divide->mutable_operand(1);
  IS_VALID(reshape->operand(0)->opcode() == HloOpcode::kReduce)
  HloInstruction* reduce = reshape->mutable_operand(0);
  IS_VALID(reduce->operand(0)->opcode() == HloOpcode::kSelect)
  HloInstruction* select_1 = reduce->mutable_operand(0);
  IS_VALID(select_1->operand(0)->opcode() == HloOpcode::kBroadcast &&
           select_1->operand(1)->opcode() == HloOpcode::kSlice &&
           select_1->operand(2)->opcode() == HloOpcode::kBroadcast)
  HloInstruction* broadcast_2 = select_1->mutable_operand(0);
  HloInstruction* slice = select_1->mutable_operand(1);
  IS_VALID(broadcast_2->operand(0)->opcode() == HloOpcode::kCompare)
  HloInstruction* compare_0 = broadcast_2->mutable_operand(0);
  IS_VALID(broadcast_0->operand(0)->opcode() == HloOpcode::kCompare)
  HloInstruction* compare_2 = broadcast_0->mutable_operand(0);
  IS_VALID(compare_2->operand(0)->opcode() == HloOpcode::kReduce)
  HloInstruction* reduce_1 = compare_2->mutable_operand(0);
  IS_VALID(reduce_1->operand(0)->opcode() == HloOpcode::kConvert)
  HloInstruction* convert = reduce_1->mutable_operand(0);
  IS_VALID(convert->operand(0)->opcode() == HloOpcode::kCompare)
  std::vector<HloInstruction*> fused_instrs = {
      select,      divide,      broadcast_1, reshape,  reduce,  select_1, slice,
      broadcast_2, broadcast_0, compare_2,   reduce_1, convert, compare_0};

  HloComputation* parent = select->parent();
  HloInstruction* fusion = parent->CreateFusionInstruction(
      fused_instrs, HloInstruction::FusionKind::kInput);

  HloInstruction* root_instr = fusion->parent()->root_instruction();
  HloInstruction* slice_starts =
      root_instr->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int64_t>(slice->slice_starts())));
  HloInstruction* slice_limits =
      root_instr->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int64_t>(slice->slice_limits())));
  HloInstruction* shape_info_0 = root_instr->AddInstruction(
      HloInstruction::CreateConstant(get_param_info(fusion->operand(5))));
  HloInstruction* shape_info_1 = root_instr->AddInstruction(
      HloInstruction::CreateConstant(get_param_info(fusion->operand(3))));
  std::vector<HloInstruction*> operands = {fusion->mutable_operand(5),
                                           fusion->mutable_operand(3),
                                           slice_starts,
                                           slice_limits,
                                           shape_info_0,
                                           shape_info_1};
  HloInstruction* custom_call = fusion->AddInstruction(
      HloInstruction::CreateCustomCall(fusion->shape(), operands, "__pooling"));
  return parent->ReplaceInstruction(fusion, custom_call, false).ok();
}

bool fused_softmax(HloInstruction* dot) {
  IS_VALID(dot->opcode() == HloOpcode::kDot)
  IS_VALID(dot->users()[0]->opcode() == HloOpcode::kMultiply)
  HloInstruction* mul = dot->users()[0];
  IS_VALID(mul->users()[0]->opcode() == HloOpcode::kAdd)
  HloInstruction* add = mul->users()[0];
  IS_VALID(add->users().size() == 2 &&
           add->users()[0]->opcode() == HloOpcode::kReduceWindow &&
           add->users()[1]->opcode() == HloOpcode::kSubtract)
  HloInstruction* reduce_window = add->users()[0];
  HloInstruction* subtract = add->users()[1];
  IS_VALID(reduce_window->users()[0]->opcode() == HloOpcode::kReduce)
  HloInstruction* reduce = reduce_window->users()[0];
  IS_VALID(reduce->users()[0]->opcode() == HloOpcode::kBroadcast)
  HloInstruction* broadcast = reduce->users()[0];
  IS_VALID(broadcast->users()[0]->opcode() == HloOpcode::kSubtract)
  IS_VALID(subtract->users()[0]->opcode() == HloOpcode::kExp)
  HloInstruction* exponential = subtract->users()[0];
  IS_VALID(exponential->users().size() == 2 &&
           exponential->users()[0]->opcode() == HloOpcode::kReduceWindow &&
           exponential->users()[1]->opcode() == HloOpcode::kDivide)
  HloInstruction* reduce_window_1 = exponential->users()[0];
  HloInstruction* divide = exponential->users()[1];
  IS_VALID(reduce_window_1->users()[0]->opcode() == HloOpcode::kReduce)
  HloInstruction* reduce_1 = reduce_window_1->users()[0];
  IS_VALID(reduce_1->users()[0]->opcode() == HloOpcode::kBroadcast)
  HloInstruction* broadcast_1 = reduce_1->users()[0];

  std::vector<HloInstruction*> fused_instrs = {
    divide, broadcast_1, reduce_1, reduce_window_1, exponential,
    subtract, broadcast, reduce, reduce_window, add, mul};
  
  HloComputation* parent = dot->parent();
  HloInstruction* fusion = parent->CreateFusionInstruction(
    fused_instrs, HloInstruction::FusionKind::kInput);
  HloInstruction* root_instr = fusion->parent()->root_instruction();
  HloInstruction* shape_info = root_instr->AddInstruction(
    HloInstruction::CreateConstant(get_param_info(fusion->operand(2)))
  );
  std::vector<HloInstruction*> operands = {fusion->mutable_operand(2),
                                           fusion->mutable_operand(3),
                                           fusion->mutable_operand(4),
                                           shape_info};
  HloInstruction* custom_call = fusion->AddInstruction(
    HloInstruction::CreateCustomCall(fusion->shape(), operands, "__softmax"));
  return parent->ReplaceInstruction(fusion, custom_call, false).ok();
}

void register_softmax(std::vector<KDnnRewriter>& rewriters,
                      RewriterType rewrite_type, int benefit = 1) {
  RewritePattern pattern("softmax", HloOpcode::kDot);
  pattern.custom_rewriter = fused_softmax;
  auto rewriter = KDnnRewriter(benefit, pattern, rewrite_type);
  rewriters.push_back(rewriter);
}

void register_pooling(std::vector<KDnnRewriter>& rewriters,
                      RewriterType rewrite_type, int benefit = 1) {
  RewritePattern pattern("pooling", HloOpcode::kSelect);
  pattern.custom_rewriter = fused_pooling;

  auto rewriter = KDnnRewriter(benefit, pattern, rewrite_type);
  rewriters.push_back(rewriter);
}

void register_graph_opt_rewriters(std::vector<KDnnRewriter>& rewriters) {
#if defined(ANNC_ENABLED_GRAPH_OPT)
  register_sparse_embedding2(rewriters, RewriterType::FUSED_OPERATION, 3);
  register_softmax(rewriters, RewriterType::FUSED_OPERATION, 3);
  // register_pooling(rewriters, RewriterType::FUSED_OPERATION, 3);
#endif
  std::sort(rewriters.begin(), rewriters.end(), compare_rewriter);
}

void __softmax(void* out, void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* mul_param_0 = reinterpret_cast<const float*>(in[1]);
  const float* mul_param_1 = reinterpret_cast<const float*>(in[2]);
  const float* add_param = reinterpret_cast<const float*>(in[0]);
  const uint64_t* shape = reinterpret_cast<const uint64_t*>(in[3]);
  
  uint64_t batches = shape[0];
  uint64_t heads = shape[1];
  uint64_t m = shape[2];
  uint64_t n = shape[3];

  const uint64_t total_rows = batches * heads * m;
  const uint64_t row_size = n;

  #pragma omp parallel for schedule(guided, 64)
  for (uint64_t row_idx = 0; row_idx < total_rows; ++row_idx) {
    // Calculate the position of the current row in the tensor.
    const uint64_t batch_idx = row_idx / (heads * m);
    const uint64_t head_idx = (row_idx % (heads * m)) / m;
    const uint64_t m_idx = row_idx % m;

    // Calculate the starting offset of the current row 
    const uint64_t base_offset = batch_idx * heads * m * n + 
                                 head_idx * m * n +
                                 m_idx * n;

    const float* mul0_row = mul_param_0 + base_offset;
    const float* mul1_row = mul_param_1 + base_offset;
    const float* add_row = add_param + base_offset;
    float* out_row = out_buf + base_offset;

    float max_val = -std::numeric_limits<float>::infinity();

    float temp_vals[row_size];

    for (uint64_t i = 0; i < row_size; ++i) {
      temp_vals[i] = mul0_row[i] * mul1_row[i] + add_row[i];
      if (temp_vals[i] > max_val) {
        max_val = temp_vals[i];
      }
    }

    float exp_sum = 0.0f;
    for (uint64_t i = 0; i < row_size; ++i) {
      float exp_val = std::exp(temp_vals[i] - max_val);
      out_row[i] = exp_val;
      exp_sum += exp_val;
    }

    exp_sum += 1e-8f;
    
    float inv_exp_sum = 1.0f / exp_sum;

    for (uint64_t i = 0; i < row_size; ++i) {
      out_row[i] *= inv_exp_sum;
    }
  }
}

void __pooling(void* out, void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const int64_t* arg = reinterpret_cast<const int64_t*>(in[0]);
  const float* mul_out = reinterpret_cast<const float*>(in[1]);

  const int64_t* slice_starts = reinterpret_cast<const int64_t*>(in[2]);
  const int64_t* slice_limits = reinterpret_cast<const int64_t*>(in[3]);
  const int64_t* arg_shape = reinterpret_cast<const int64_t*>(in[4]);
  const int64_t* mul_shape = reinterpret_cast<const int64_t*>(in[5]);

  int64_t m = arg_shape[0];  // 128
  int64_t k = arg_shape[1];  // 30
  int64_t n = mul_shape[2];  // 16

  bool cond_0[m][k];
  bool cond_1[m];
  float red_1[m] = {0.0f};
  for (int i = 0; i < m; i++) {
    for (int64_t j = 0; j < k; j++) {
      cond_0[i][j] = arg[i * k + j] > 0;
      red_1[i] += (float)(cond_0[i][j]);
    }
    cond_1[i] = (red_1[i] == 0);
  }

  float sel_1[m][n][k] = {0.0f};  // 128, 16, 30
  for (int i = slice_starts[0]; i < slice_limits[0]; i++) {
    for (int j = slice_starts[1]; j < slice_limits[1]; j++) {
      for (int p = slice_starts[2]; p < slice_limits[2]; p++) {
        if (cond_0[i][j]) {
          sel_1[i][p][j] = mul_out[i * k * n + j * n + p];
        }
      }
    }
  }

  for (int i = 0; i < m; i ++) {
    for (int j = 0; j < n; j ++) {
      float sum = 0.0f;
      for (int p = 0; p < k; p++) {
        sum += sel_1[i][j][p];
      }
      out_buf[i * n + j] = cond_1[i] ? 0 : sum / red_1[i];
    }
  }
}

}  // namespace cpu
}  // namespace xla
