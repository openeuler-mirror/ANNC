#ifndef AI_COMPILER_KDNN_REWRITER_H_
#define AI_COMPILER_KDNN_REWRITER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/pattern_matcher.h"

namespace xla {
namespace cpu {
static std::string g_kdnn_fusion_befor_hlo_layout_assign =
    "kdnn-fusion-before-hlo-layout-assign";
static std::string g_kdnn_fusion_after_hlo_layout_assign =
    "kdnn-fusion-aftr-hlo-layout-assign";
static std::string g_kdnn_fusion_after_run_backend =
    "kdnn-fusion-aftr-run-backend";

typedef bool (*CustomRewriter)(HloInstruction*);

struct RewritePattern {
  RewritePattern(std::string name, HloOpcode opcode, bool valid = true)
      : name(name), opcode(opcode), valid(valid) {}
  std::string name;
  HloOpcode opcode;
  bool valid;
  CustomRewriter custom_rewriter{nullptr};
  std::vector<PrimitiveType> dtypes{};
  std::vector<int64_t> dims{};
  std::vector<std::vector<int64_t>> dim_range{};
  std::vector<int64_t> layouts{};
  std::vector<RewritePattern> next_patterns{};
};

enum class RewriterType {
  FUSED_OPERATION,
  DNN_CUSTOM_CALL,
};

/* kdnn rewriters */
class KDnnRewriter {
 public:
  KDnnRewriter(int benefit, const RewritePattern& pattern, RewriterType type)
      : benefit_(benefit), pattern_(pattern), type_(type){};
  virtual ~KDnnRewriter() = default;

  bool execute(HloInstruction* instr);

  int benefit() const { return benefit_; }

 private:
  int benefit_{0};
  RewriterType type_;
  RewritePattern pattern_;
};

bool compare_rewriter(const KDnnRewriter& a, const KDnnRewriter& b);

/* register rewriters */
void register_gemm_rewriters(std::vector<KDnnRewriter>& rewriters);
void register_reduce_rewriters(std::vector<KDnnRewriter>& rewriters);
void register_graph_opt_rewriters(std::vector<KDnnRewriter>& rewriters);

/* rewriter vistors */
#define CREATE_KDNN_REWRITER(rewriter_name)                     \
  class rewriter_name : public DfsHloRewriteVisitor {           \
   public:                                                      \
    Status HandleDot(HloInstruction* instr) override {          \
      std::vector<KDnnRewriter> rewriters;                      \
      register_graph_opt_rewriters(rewriters);                  \
      register_gemm_rewriters(rewriters);                       \
      for (auto& rewriter : rewriters) {                        \
        if (rewriter.execute(instr)) return OkStatus();         \
      }                                                         \
      return OkStatus();                                        \
    }                                                           \
    Status HandleReduceWindow(HloInstruction* instr) override { \
      std::vector<KDnnRewriter> rewriters;                      \
      register_reduce_rewriters(rewriters);                     \
      for (auto& rewriter : rewriters) {                        \
        if (rewriter.execute(instr)) return OkStatus();         \
      }                                                         \
      return OkStatus();                                        \
    }                                                           \
    Status HandleMultiply(HloInstruction* instr) override {     \
      std::vector<KDnnRewriter> rewriters;                      \
      register_graph_opt_rewriters(rewriters);                  \
      for (auto& rewriter : rewriters) {                        \
        if (rewriter.execute(instr)) return OkStatus();         \
      }                                                         \
      return OkStatus();                                        \
    }                                                           \
    Status HandleSelect(HloInstruction* instr) override {       \
      std::vector<KDnnRewriter> rewriters;                      \
      register_graph_opt_rewriters(rewriters);                  \
      for (auto& rewriter : rewriters) {                        \
        if (rewriter.execute(instr)) return OkStatus();         \
      }                                                         \
      return OkStatus();                                        \
    }                                                           \
  };

CREATE_KDNN_REWRITER(KDnnBeforeHloLayoutAssignRewriterVisitor);
CREATE_KDNN_REWRITER(KDnnAfterHloLayoutAssignRewriterVisitor);
CREATE_KDNN_REWRITER(KDnnAfterRunBackendRewriterVisitor);

/* kdnn fusion passes */
#define CREATE_KDNN_FUSION_PASS(pass_class, pass_name)               \
  class pass_class : public HloModulePass {                          \
   public:                                                           \
    absl::string_view name() const override { return pass_name; }    \
    using HloPassInterface::Run;                                     \
    StatusOr<bool> Run(HloModule* module,                            \
                       const absl::flat_hash_set<absl::string_view>& \
                           execution_threads) override;              \
  };

CREATE_KDNN_FUSION_PASS(KDnnFusionBeforeHloLayoutAssign,
                        g_kdnn_fusion_befor_hlo_layout_assign);
CREATE_KDNN_FUSION_PASS(KDnnFusionAfterHloLayoutAssign,
                        g_kdnn_fusion_after_hlo_layout_assign);
CREATE_KDNN_FUSION_PASS(KDnnFusionAfterRunBackend,
                        g_kdnn_fusion_after_run_backend);

/* insert passes during hlo optimization */
#define ADD_PASSES_BEFORE_HLO_LAYOUT_ASSIGN(pipeline) \
  pipeline.AddPass<KDnnFusionBeforeHloLayoutAssign>();

#define ADD_PASSES_AFTER_HLO_LAYOUT_ASSIGN(pipeline) \
  pipeline.AddPass<KDnnFusionAfterHloLayoutAssign>();

#define ADD_PASSES_AFTER_RUN_BACKEND(pipeline) \
  pipeline.AddPass<KDnnFusionAfterRunBackend>();

/* register kernel function */
void __matmul(void* out, const void** in);
void __batch_matmul(void* out, const void** in);
void __matmul_add(void* out, const void** in);
void __matmul_add_relu(void* out, const void** in);
void __reduce_mean(void* out, const void** in);
void __pooling(void* out, void** in);

#define REGISTER_ALL_GEMM_KERNELS()                       \
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__matmul);          \
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__batch_matmul);    \
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__matmul_add);      \
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__matmul_add_relu); \
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__reduce_mean);     \
  XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(__pooling);

}  // namespace cpu
}  // namespace xla

#endif  // AI_COMPILER_KDNN_REWRITER_H_
