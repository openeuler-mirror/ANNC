#include "graph_opt.h"
#include "gflags/gflags.h"

DECLARE_bool(annc);
DECLARE_bool(annc_fusion);
DECLARE_bool(annc_fused_dyn_stitch);
DECLARE_bool(annc_fused_seg_reduce);
DECLARE_bool(annc_fused_emd_padding);
DECLARE_bool(annc_fused_emd_padding_fast);
DECLARE_bool(annc_fused_sparse_select);
DECLARE_bool(annc_fused_gather);
DECLARE_bool(annc_fused_sparse_reshape);
DECLARE_bool(annc_fused_emd_actionid_gather);
DECLARE_bool(annc_fused_seg_reduce_nozero);
DECLARE_bool(annc_fused_matmul);

using namespace tensorflow;
using namespace tensorflow::grappler;

namespace annc {
void update_node_indexes(const GraphDef* graph,
                         std::unordered_map<std::string, int>& node_indexes) {
  for (int i = 0; i < graph->node_size(); ++i) {
    node_indexes[graph->node(i).name()] = i;
  }
}

void GraphOptimizer::register_rewriter(
    std::unique_ptr<PatternRewriter> rewriter) {
  rewriters_.push_back(std::move(rewriter));
}

void GraphOptimizer::optimize() {
  update_node_indexes(graph_, node_indexes_);
  int node_index = 0;
  const int node_size = graph_->node_size();
  while (node_index < node_size) {
    const NodeDef& node = graph_->node(node_index);
    const std::string& node_name = node.name();
    for (auto& rewriter : rewriters_) {
      if (rewriter->match_and_rewrite(&node, graph_, node_indexes_)) {
        update_node_indexes(graph_, node_indexes_);
        const std::string new_node_name = node_name + fusion_appendix;
        node_index = node_indexes_.at(new_node_name);
        break;
      }
    }
    node_index++;
  }
}

std::string get_node_name(const std::string& name) {
  size_t colon_pos = name.find(':');
  std::string node_name = name;
  if (colon_pos != std::string::npos) {
    node_name = name.substr(0, colon_pos);
  }
  return node_name;
}

void set_fusedop_attributes(NodeDef* fused,
                            const absl::Span<const absl::string_view> fused_ops,
                            int num_args = 1, float epsilon = 0.0) {
  auto* attr = fused->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
  SetAttrValue(epsilon, &(*attr)["epsilon"]);  // required only for BatchNorm
}

const NodeDef* PatternRewriter::get_node(const std::string& name) {
  const std::string node_name = get_node_name(name);
  const int node_index = indexes_->at(node_name);
  return &graph_->node(node_index);
}

NodeDef* PatternRewriter::get_mutable_node(const std::string& name) {
  const std::string node_name = get_node_name(name);
  if (indexes_->find(node_name) == indexes_->end()) return nullptr;
  const int node_index = indexes_->at(node_name);
  return graph_->mutable_node(node_index);
}

NodeDef* PatternRewriter::get_operand(const NodeDef* node,
                                      std::string op_type) {
  for (int i = 0; i < node->input_size(); ++i) {
    NodeDef* operand = get_mutable_node(node->input(i));
    if (operand != nullptr && operand->op() == op_type) return operand;
  }
  return nullptr;
}

const NodeDef* PatternRewriter::get_user(const NodeDef* node, int index,
                                         const std::string& op_type) {
  std::string node_name = node->name();
  if (index) std::string node_name = node_name + ":" + std::to_string(index);
  for (int i = 0; i < graph_->node_size(); ++i) {
    const NodeDef* node = graph_->mutable_node(i);
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == node_name && node->op() == op_type) {
        return node;
      }
    }
  }
  return nullptr;
}

void PatternRewriter::replace_all_users_with(const NodeDef* old_node,
                                             int old_index,
                                             const NodeDef* new_node,
                                             int new_index, GraphDef* graph) {
  std::string old_name = old_node->name();
  if (old_index) old_name = old_name + ":" + std::to_string(old_index);
  std::string new_name = new_node->name();
  if (new_index) new_name = new_name + ":" + std::to_string(new_index);
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    for (int j = 0; j < node->input_size(); ++j) {
      if (node->input(j) == old_name) {
        node->set_input(j, new_name);
      }
    }
  }
}

class KPFusedSparseDynamicStitchRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedSparseDynamicStitch"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(node->op() == "ParallelDynamicStitch" &&
                  node->input_size() % 2 == 0)
    int num_inputs = node->input_size();
    // left branch
    const NodeDef* partition = get_node(node->input(0));
    CHECK_NODE_OK(partition->op() == "DynamicPartition" &&
                  partition->input_size() == 2)
    const NodeDef* range = get_node(partition->input(0));
    CHECK_NODE_OK(range->op() == "Range" && range->input_size() == 3)
    const NodeDef* size = get_node(range->input(1));
    CHECK_NODE_OK(IsSize(*size) && size->input_size() == 1)
    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(size->input(0));
    // right branch
    for (int i = num_inputs / 2; i < num_inputs; ++i) {
      const NodeDef* gather = get_node(node->input(i));
      CHECK_NODE_OK(gather->op() == "GatherV2" && gather->input_size() == 3)
      const NodeDef* identity = get_node(gather->input(0));
      CHECK_NODE_OK(identity->op() == "Identity" && identity->input_size() == 1)
      fused_node->add_input(gather->input(0));
    }
    (*fused_node->mutable_attr())["N"].set_i(num_inputs / 2);
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(node, 0, fused_node, 0, graph);
    return true;
  }
};

class KPFusedSparseSegmentReduceRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedSparseSegmentReduce"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsStridedSlice(*node) && node->input_size() == 4)
    const NodeDef* shape = get_node(node->input(0));
    CHECK_NODE_OK(IsShape(*shape) && shape->input_size() == 1)
    const NodeDef* ss_reduce = get_node(shape->input(0));
    CHECK_NODE_OK(ss_reduce->input_size() == 3)
    AttrValue combiner;
    if (ss_reduce->op() == "SparseSegmentMean") {
      combiner.set_i(1);
    } else if (ss_reduce->op() == "SparseSegmentSum") {
      combiner.set_i(0);
    } else {
      return false;
    }
    const NodeDef* cast = get_node(ss_reduce->input(2));
    CHECK_NODE_OK(IsCast(*cast) && cast->input_size() == 1)
    const NodeDef* strided_slice = get_node(cast->input(0));
    CHECK_NODE_OK(IsStridedSlice(*strided_slice))

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(ss_reduce->input(0));
    fused_node->add_input(ss_reduce->input(1));
    fused_node->add_input(strided_slice->input(0));
    fused_node->add_input(strided_slice->input(1));
    fused_node->add_input(strided_slice->input(2));
    fused_node->add_input(strided_slice->input(3));
    AddNodeAttr("combiner", combiner, fused_node);

    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(ss_reduce, 0, fused_node, 0, graph);
    replace_all_users_with(node, 0, fused_node, 1, graph);
    return true;
  }
};

class KPFusedSparseSegmentReduceNonzeroRewriter : public PatternRewriter {
 public:
  std::string name() const override {
    return "KPFusedSparseSegmentReduceNonzero";
  }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(node->op() == "GatherND" &&
                  node->input_size() == 2)  // output:2
    const NodeDef* ss_reduce = get_node(node->input(0));
    CHECK_NODE_OK(ss_reduce->input_size() == 3)
    AttrValue combiner;
    if (ss_reduce->op() == "SparseSegmentMean") {
      combiner.set_i(1);
    } else if (ss_reduce->op() == "SparseSegmentSum") {
      combiner.set_i(0);
    } else {
      return false;
    }
    const NodeDef* where = get_node(node->input(1));
    CHECK_NODE_OK(where->op() == "Where" && where->input_size() == 1)
    const NodeDef* cast = get_user(where, 0, "Cast");
    CHECK_NODE_OK(cast != nullptr)  // output: 1
    const NodeDef* notequal = get_node(where->input(0));
    CHECK_NODE_OK(IsNotEqual(*notequal) && notequal->input_size() == 2);
    const NodeDef* zerolike = get_node(notequal->input(1));
    CHECK_NODE_OK(IsZerosLike(*zerolike) && zerolike->input_size() == 1)
    const NodeDef* cast_1 = get_node(ss_reduce->input(2));
    CHECK_NODE_OK(IsCast(*cast_1) && cast_1->input_size() == 1)
    const NodeDef* strided_slice = get_node(cast->input(0));
    CHECK_NODE_OK(IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)
    const NodeDef* shape = get_user(ss_reduce, 0, "Shape");
    CHECK_NODE_OK(shape != nullptr)
    const NodeDef* cast_2 = get_user(shape, 0, "Cast");  // output: 0
    CHECK_NODE_OK(cast_2 != nullptr)

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(ss_reduce->input(0));
    fused_node->add_input(ss_reduce->input(1));
    fused_node->add_input(strided_slice->input(0));
    fused_node->add_input(strided_slice->input(1));
    fused_node->add_input(strided_slice->input(2));
    fused_node->add_input(strided_slice->input(3));
    AddNodeAttr("combiner", combiner, fused_node);

    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name() << "\n";
    replace_all_users_with(cast_2, 0, fused_node, 0, graph);
    replace_all_users_with(cast, 0, fused_node, 1, graph);
    replace_all_users_with(node, 0, fused_node, 2, graph);
    return true;
  }
};

class KPFusedEmbeddingPaddingRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedEmbeddingPadding"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsReshape(*node))
    const NodeDef* user = get_user(node, 0, "ConcatV2");
    CHECK_NODE_OK(user != nullptr)
    const NodeDef* concat = get_node(node->input(0));
    CHECK_NODE_OK(IsConcat(*concat) && concat->input_size() == 3)
    const NodeDef* fill = get_node(concat->input(1));
    CHECK_NODE_OK(IsFill(*fill) && fill->input_size() == 2)
    const NodeDef* pack = get_node(fill->input(0));
    CHECK_NODE_OK(IsPack(*pack) && pack->input_size() == 2)
    const NodeDef* sub = get_node(pack->input(0));
    CHECK_NODE_OK(IsSub(*sub) && sub->input_size() == 2)
    const NodeDef* strided_slice = get_node(sub->input(0));
    CHECK_NODE_OK(IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)
    const NodeDef* cast = get_node(strided_slice->input(0));
    CHECK_NODE_OK(IsCast(*cast))

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(cast->input(0));
    fused_node->add_input(concat->input(0));
    fused_node->add_input(sub->input(1));
    fused_node->add_input(node->input(1));

    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(sub, 0, fused_node, 0, graph);
    replace_all_users_with(node, 0, fused_node, 1, graph);
    return true;
  }
};

class KPFusedEmbeddingPaddingFastRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedEmbeddingPaddingFast"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsStridedSlice(*node) && node->input_size() == 4)
    const NodeDef* shape = get_node(node->input(0));
    CHECK_NODE_OK(IsShape(*shape) && shape->input_size() == 1)
    const NodeDef* reshape = get_node(shape->input(0));
    CHECK_NODE_OK(IsReshape(*reshape) && reshape->input_size() == 2)
    const NodeDef* concat = get_node(reshape->input(0));
    CHECK_NODE_OK(IsConcat(*concat) && concat->input_size() == 3)
    const NodeDef* fill = get_node(concat->input(1));
    CHECK_NODE_OK(IsFill(*fill) && fill->input_size() == 2)
    const NodeDef* pack = get_node(fill->input(0));
    CHECK_NODE_OK(IsPack(*pack) && pack->input_size() == 2)
    const NodeDef* sub = get_node(pack->input(0));
    CHECK_NODE_OK(IsSub(*sub) && sub->input_size() == 2)
    const NodeDef* strided_slice = get_node(sub->input(0));
    CHECK_NODE_OK(IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)
    const NodeDef* cast = get_node(strided_slice->input(0));
    CHECK_NODE_OK(IsCast(*cast))

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(cast->input(0));
    fused_node->add_input(concat->input(0));
    fused_node->add_input(sub->input(1));
    fused_node->add_input(reshape->input(1));

    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(sub, 0, fused_node, 0, graph);
    replace_all_users_with(node, 0, fused_node, 1, graph);
    return true;
  }
};

class KPFusedSparseSelectRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedSparseSelect"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsConcat(*node) && node->input_size() == 3)
    const NodeDef* select_0 = get_node(node->input(0));
    CHECK_NODE_OK(IsSelect(*select_0) && select_0->input_size() == 3)
    const NodeDef* select_1 = get_node(select_0->input(2));
    CHECK_NODE_OK(IsSelect(*select_1) && select_1->input_size() == 3)
    const NodeDef* cast = get_node(select_1->input(2));
    CHECK_NODE_OK(IsCast(*cast) && cast->input_size() == 1)
    const NodeDef* greater = get_node(cast->input(0));
    CHECK_NODE_OK(IsGreater(*greater) && greater->input_size() == 2)
    const NodeDef* reshape = get_node(greater->input(0));
    CHECK_NODE_OK(IsReshape(*reshape) && reshape->input_size() == 2)
    const NodeDef* equal = get_node(select_0->input(0));
    CHECK_NODE_OK(IsEqual(*equal) && equal->input_size() == 2)
    NodeDef* reshape_1 = get_operand(equal, "Reshape");
    CHECK_NODE_OK(reshape_1 != nullptr && IsReshape(*reshape_1))

    // right branch
    const NodeDef* select_2 = get_node(node->input(1));
    CHECK_NODE_OK(IsSelect(*select_2) && select_2->input_size() == 3)
    const NodeDef* equal_1 = get_node(select_2->input(0));
    CHECK_NODE_OK(IsEqual(*equal_1) && equal_1->input_size() == 2)
    const NodeDef* reshape_2 = get_operand(equal_1, "Reshape");
    CHECK_NODE_OK(reshape_2 != nullptr && IsReshape(*reshape_2))

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(reshape->input(0));
    fused_node->add_input(reshape_1->input(0));
    fused_node->add_input(reshape_2->input(0));
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(reshape, 0, fused_node, 0, graph);
    replace_all_users_with(select_0, 0, fused_node, 1, graph);
    replace_all_users_with(node, 0, fused_node, 2, graph);
    return true;
  }
};

class KPFusedGatherRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedGather"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(node->op() == "GatherV2" && node->input_size() == 3)
    const NodeDef* gather = get_node(node->input(0));
    CHECK_NODE_OK(gather->op() == "GatherV2" &&
                  gather->input_size() == 3)           // input:0
    const NodeDef* unique = get_node(node->input(1));  // output:1
    CHECK_NODE_OK(unique->op() == "Unique" && unique->input_size() == 1)
    const NodeDef* unique_1 = get_node(unique->input(0));
    CHECK_NODE_OK(unique_1->op() == "Unique" && unique_1->input_size() == 1)
    const NodeDef* strided_slice = get_node(unique_1->input(0));
    CHECK_NODE_OK(IsStridedSlice(*strided_slice))  // input:1 2

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(gather->input(0));
    fused_node->add_input(strided_slice->input(0));
    fused_node->add_input(strided_slice->input(1));
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(unique_1, 0, fused_node, 0, graph);
    replace_all_users_with(unique, 0, fused_node, 1, graph);
    replace_all_users_with(node, 0, fused_node, 2, graph);
    return true;
  }
};

class KPFusedSparseReshapeRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedSparseReshape"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(node->op() == "SparseReshape" && node->input_size() == 3)
    const NodeDef* concat = get_node(node->input(0));
    CHECK_NODE_OK(IsConcat(*concat) && concat->input_size() == 3)
    const NodeDef* reshape = get_node(concat->input(1));
    CHECK_NODE_OK(IsReshape(*reshape) && reshape->input_size() == 2)
    const NodeDef* strided_slice = get_node(reshape->input(0));
    CHECK_NODE_OK(IsStridedSlice(*strided_slice) &&
                  strided_slice->input_size() == 4)
    const NodeDef* cast = get_node(node->input(1));
    CHECK_NODE_OK(IsCast(*cast) && cast->input_size() == 1)
    const NodeDef* pack = get_node(cast->input(0));
    CHECK_NODE_OK(IsPack(*pack) && pack->input_size() == 2)
    const NodeDef* strided_slice_1 = get_node(pack->input(0));
    CHECK_NODE_OK(IsStridedSlice(*strided_slice_1) &&
                  strided_slice_1->input_size() == 4)
    const NodeDef* shape = get_node(strided_slice_1->input(0));
    CHECK_NODE_OK(IsShape(*shape) && shape->input_size() == 1)

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(shape->input(0));
    fused_node->add_input(strided_slice->input(1));
    fused_node->add_input(node->input(2));
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(node, 0, fused_node, 0, graph);
    replace_all_users_with(node, 1, fused_node, 1, graph);
    return true;
  }
};

class KPFusedEmbeddingActionIdGatherRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "KPFusedEmbeddingActionIdGather"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsConcat(*node) && node->input_size() == 3)
    const NodeDef* reshape = get_node(node->input(0));
    CHECK_NODE_OK(IsReshape(*reshape) && reshape->input_size() == 2)
    const NodeDef* gather = get_node(reshape->input(0));
    CHECK_NODE_OK(gather->op() == "GatherV2" && gather->input_size() == 3)
    const NodeDef* gather_1 = get_node(gather->input(0));
    CHECK_NODE_OK(gather_1->op() == "GatherV2" && gather_1->input_size() == 3)
    const NodeDef* fill = get_node(node->input(1));
    CHECK_NODE_OK(IsFill(*fill) && fill->input_size() == 2)
    const NodeDef* pack = get_node(fill->input(0));
    CHECK_NODE_OK(IsPack(*pack) && pack->input_size() == 2)

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(gather_1->input(1));
    fused_node->add_input(gather_1->input(0));
    fused_node->add_input(gather->input(1));
    fused_node->add_input(pack->input(0));
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(node, 0, fused_node, 0, graph);
    return true;
  }
};

class KPFusedMatMulRewriter : public PatternRewriter {
 public:
  std::string name() const override { return "_FusedMatMul"; }

  bool match_and_rewrite(
      const NodeDef* node, GraphDef* graph,
      std::unordered_map<std::string, int>& node_indexes) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsRelu(*node) && node->input_size() == 1)
    const NodeDef* add_1 = get_operand(node, "AddV2");
    CHECK_NODE_OK(add_1 != nullptr);
    const NodeDef* bias_weight = get_operand(add_1, "Const");
    CHECK_NODE_OK(bias_weight != nullptr);
    const NodeDef* matmul = get_operand(add_1, "MatMul");
    CHECK_NODE_OK(matmul != nullptr);

    auto nodes = graph->mutable_node();
    NodeDef* fused_node = nodes->Add();
    fused_node->set_name(node->name() + fusion_appendix);
    fused_node->set_op(name());
    fused_node->set_device(node->device());
    fused_node->add_input(matmul->input(0));
    fused_node->add_input(matmul->input(1));
    fused_node->add_input(bias_weight->name());
    auto* attr = fused_node->mutable_attr();
    auto& src_attr = matmul->attr();

    (*attr)["T"] = src_attr.at("T");
    (*attr)["transpose_a"] = src_attr.at("transpose_a");
    (*attr)["transpose_b"] = src_attr.at("transpose_b");

    set_fusedop_attributes(fused_node, {"BiasAdd", "Relu"}, 1);

    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);

    VLOG(0) << "-- Add node: [" << fused_node->op() << "] "
            << fused_node->name();
    replace_all_users_with(node, 0, fused_node, 0, graph);
    return true;
  }
};

bool enabled_aarch64_rewriters() {
  unsigned arch_id;
  __asm__("mrs %0, MIDR_EL1" : "=r"(arch_id));
  int implementer = (arch_id >> 24) & 0xFF;
  int part = (arch_id >> 4) & 0xFFF;
  bool flag = false;
  if (implementer == 0x48) {
    if (part == 0xd01 || part == 0xd02 || part == 0xd03) flag = true;
  }
  return flag;
}

void run_graph_optimization(GraphDef* graph) {
  GraphOptimizer optimizer(graph);
  if (enabled_aarch64_rewriters()) {
    bool enable_all = FLAGS_annc || FLAGS_annc_fusion;
    VLOG(0) << "FLAGS_annc: " << FLAGS_annc;

    // default enable all rewriters
    if (enable_all || FLAGS_annc_fused_dyn_stitch)
      optimizer.register_rewriter(
          std::make_unique<KPFusedSparseDynamicStitchRewriter>());
    if (enable_all || FLAGS_annc_fused_seg_reduce)
      optimizer.register_rewriter(
          std::make_unique<KPFusedSparseSegmentReduceRewriter>());
    if (enable_all || FLAGS_annc_fused_emd_padding_fast)
      optimizer.register_rewriter(
          std::make_unique<KPFusedEmbeddingPaddingFastRewriter>());
    if (enable_all || FLAGS_annc_fused_emd_padding)
      optimizer.register_rewriter(
          std::make_unique<KPFusedEmbeddingPaddingRewriter>());
    if (enable_all || FLAGS_annc_fused_sparse_select)
      optimizer.register_rewriter(
          std::make_unique<KPFusedSparseSelectRewriter>());
    if (enable_all || FLAGS_annc_fused_gather)
      optimizer.register_rewriter(std::make_unique<KPFusedGatherRewriter>());
    if (enable_all || FLAGS_annc_fused_sparse_reshape)
      optimizer.register_rewriter(
          std::make_unique<KPFusedSparseReshapeRewriter>());
    if (FLAGS_annc_fused_emd_actionid_gather)  // default disbaled
      optimizer.register_rewriter(
          std::make_unique<KPFusedEmbeddingActionIdGatherRewriter>());
    if (FLAGS_annc_fused_seg_reduce_nozero)  // default disbaled
      optimizer.register_rewriter(
          std::make_unique<KPFusedSparseSegmentReduceNonzeroRewriter>());
    if (FLAGS_annc_fused_matmul)  // default disbaled
      optimizer.register_rewriter(std::make_unique<KPFusedMatMulRewriter>());
  }
  optimizer.optimize();
}
}  // namespace annc
