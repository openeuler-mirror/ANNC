#ifndef ANNC_TF_MODEL_OPTIMIZER_H_
#define ANNC_TF_MODEL_OPTIMIZER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

using namespace tensorflow;
using namespace tensorflow::grappler;

namespace annc {
#define CHECK_NODE_OK(x) \
  if (!(x)) {            \
    return false;        \
  }

static std::unordered_map<std::string, Tensor> g_variable_tensors;
void save_variables_to_proto_file(const size_t& hash);
void load_variables_from_proto_file(const size_t& hash);

class ConstantFoldingRewritter {
 public:
  ConstantFoldingRewritter() {}
  virtual ~ConstantFoldingRewritter() = default;

  virtual bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes, Session* session) = 0;

  bool is_variable(const NodeDef* node) {
    if (IsReadVariableOp(*node) || IsConstant(*node)) return true;
    if (IsIdentity(*node)) {
      const NodeDef& input = graph_->node(indexes_->at(node->input(0)));
      if (IsVariable(input)) return true;
    }
    return false;
  }

  bool check_unordered_operands(const NodeDef* node,
                                std::vector<std::string> op_types,
                                std::vector<NodeDef**> operands);
  const NodeDef* get_operand(
      const std::string& name, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes);
  NodeDef* get_mutable_operand(
      const std::string& name, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes);

  const NodeDef* get_node(std::string name);
  NodeDef* get_mutable_node(std::string name);

  void set_variable_tensor(const NodeDef* node, Tensor& t);

  Tensor get_tensor(const NodeDef* node, Session* session);
  Status get_tensors(const std::vector<const NodeDef*>& vars,
                     std::vector<Tensor>& tensors, Session* session);

  void fold_matmuladd_batchnorm(NodeDef* matmul, NodeDef* biasadd,
                                std::vector<Tensor>& tensors);

  void replace_all_users_with(const NodeDef* old_node, int old_index,
                              const NodeDef* new_node, int new_index,
                              GraphDef* graph) {
    std::string old_name = old_node->name();
    if (old_index) old_name = old_name + ":" + std::to_string(old_index);
    std::string new_name = new_node->name();
    if (new_index) new_name = new_name + ":" + std::to_string(new_index);
    for (int i = 0; i < graph->node_size(); ++i) {
      NodeDef* node = graph->mutable_node(i);
      for (int j = 0; j < node->input_size(); ++j) {
        if (node->input(j) == old_name) {
          VLOG(1) << "-- replace input: " << old_name << " -> " << new_name
                  << "\n";
          node->set_input(j, new_name);
        }
      }
    }
  }

  GraphDef* graph_;
  std::unordered_map<std::string, int>* indexes_;
};

void run_annc_constant_folding(GraphDef* graph, Session* session);
}  // namespace annc

#endif  // ANNC_TF_MODEL_OPTIMIZER_H_
