#ifndef ANNC_TF_GRAPH_OPT_H_
#define ANNC_TF_GRAPH_OPT_H_
#include <type_traits>
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

  bool check_input_dims(NodeDef* op, const std::string& output_name,
                        int dim_size) {
    if (op->attr().count("_output_shapes")) {
      int pos_index = 0;
      size_t pos = output_name.find_last_of(':');
      if (pos != std::string::npos) {
        pos_index = std::stoi(output_name.substr(pos + 1));
      }
      const TensorShapeProto& shape =
          op->attr().at("_output_shapes").list().shape(pos_index);
      if (shape.dim_size() == dim_size) return true;
    } else if (op->attr().count("shape")) {
      const TensorShapeProto& shape = op->attr().at("shape").shape();
      if (shape.dim_size() == dim_size) return true;
    }
    return false;
  }

  bool check_const_shape(NodeDef* op, std::vector<int> dims) {
    if (!((IsConstant(*op) || IsHostConstant(*op)) &&
          HasNodeAttr(*op, "value")))
      return false;

    TensorProto* tensor = (*op->mutable_attr())["value"].mutable_tensor();
    const auto& shape = tensor->tensor_shape();
    if (shape.dim_size() != static_cast<int>(dims.size())) return false;
    for (int i = 0; i < shape.dim_size(); ++i) {
      if (shape.dim(i).size() != dims[i]) return false;
    }
    return true;
  }

  template <typename T>
  bool check_const_value(NodeDef* op, std::vector<T> cmp) {
    if (!((IsConstant(*op) || IsHostConstant(*op)) &&
          HasNodeAttr(*op, "value")))
      return false;

    TensorProto* tensor = (*op->mutable_attr())["value"].mutable_tensor();
    const auto& shape = tensor->tensor_shape();
    int dim_size = 1;
    for (int i = 0; i < shape.dim_size(); ++i) {
      dim_size *= shape.dim(i).size();
    }
    if (dim_size < static_cast<int>(cmp.size())) return false;

    if (std::is_same<T, float>::value) {
      const float* data = tensor->mutable_float_val()->data();
      if (data == nullptr)
        data = reinterpret_cast<const float*>(tensor->tensor_content().data());
      if (data == nullptr) return false;
      for (int i = 0; i < static_cast<int>(cmp.size()); ++i) {
        if (std::fabs(data[i] - cmp[i]) >= 1e-5f) return false;
      }
    } else if (std::is_same<T, int>::value) {
      const int* data = tensor->mutable_int_val()->data();
      if (data == nullptr)
        data = reinterpret_cast<const int*>(tensor->tensor_content().data());
      if (data == nullptr) return false;
      for (int i = 0; i < static_cast<int>(cmp.size()); ++i) {
        if (data[i] != cmp[i]) return false;
      }
    } else if (std::is_same<T, int64_t>::value) {
      const int64_t* data = tensor->mutable_int64_val()->data();
      if (data == nullptr)
        data =
            reinterpret_cast<const int64_t*>(tensor->tensor_content().data());
      if (data == nullptr) return false;
      for (int i = 0; i < static_cast<int>(cmp.size()); ++i) {
        if (data[i] != cmp[i]) return false;
      }
    } else {
      // data type do not support
      return false;
    }
    return true;
  }

  bool check_int_attr(const NodeDef* op, std::string name, int value) {
    if (HasNodeAttr(*op, name)) {
      AttrValue attr = op->attr().at(name);
      if (attr.value_case() == AttrValue::kI && attr.i() == value) return true;
    }
    return false;
  }

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
