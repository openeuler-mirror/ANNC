#ifndef ANNC_TF_GRAPH_OPT_H_
#define ANNC_TF_GRAPH_OPT_H_
#include <unordered_map>

#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"

using namespace tensorflow;
using namespace tensorflow::grappler;

namespace annc {
#define CHECK_NODE_OK(x) \
  if (!(x)) {            \
    return false;        \
  }

static const std::string fusion_appendix = "/kp_fused";

void update_node_indexes(const GraphDef* graph,
                         std::unordered_map<std::string, int>& node_indexes);

class PatternRewriter {
 public:
  PatternRewriter() {}
  virtual ~PatternRewriter() = default;

  virtual bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) = 0;

  virtual std::string name() const { return "PatternRewriter"; };

  const NodeDef* get_node(const std::string& name);
  NodeDef* get_mutable_node(const std::string& name);

  NodeDef* get_operand(const NodeDef* node, std::string op_type);

  const NodeDef* get_user(const NodeDef* node, int index,
                          const std::string& op_type);

  void replace_all_users_with(const NodeDef* old_node, int old_index,
                              const NodeDef* new_node, int new_index,
                              GraphDef* graph);

  GraphDef* graph_;
  std::unordered_map<std::string, int>* indexes_;
};

class GraphOptimizer {
 public:
  GraphOptimizer(GraphDef* graph) : graph_(graph) {}
  virtual ~GraphOptimizer() = default;

  void register_rewriter(std::unique_ptr<PatternRewriter> rewriter);

  void optimize();

 private:
  GraphDef* graph_;
  std::unordered_map<std::string, int> node_indexes_;
  std::vector<std::unique_ptr<PatternRewriter>> rewriters_;
};

void run_graph_optimization(GraphDef* graph);
}  // namespace annc
#endif  // ANNC_TF_GRAPH_OPT_H_
