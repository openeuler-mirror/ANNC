#include "constant_folding.h"
#include "gflags/gflags.h"

#include <sys/stat.h>
#include <unistd.h>

using namespace tensorflow;
using namespace tensorflow::grappler;

DECLARE_int32(annc_cf_matmul_batchnorm);
DECLARE_bool(annc_cf_relu);
DECLARE_string(annc_cf_dump);
DECLARE_bool(annc_cf_dump_text);

namespace annc {
std::string get_cache_dir() {
  char buffer[4096];
  if (getcwd(buffer, sizeof(buffer)) == nullptr) {
    return "";
  }
  std::string current_path(buffer);
  std::string cache_dir = current_path + "/.annc_cache";
  return cache_dir;
}

void load_variables_from_proto_file(const size_t& hash) {
  std::string cache_dir = get_cache_dir();
  const std::string temp_file_path =
      cache_dir + "/variables_" + std::to_string(hash);
  tensorflow::BundleReader reader(tensorflow::Env::Default(), temp_file_path);
  if (!reader.status().ok()) {
    return;
  }
  std::vector<std::string> tensor_names;
  reader.Seek(kHeaderEntryKey);
  reader.Next();
  while (reader.Valid()) {
    tensor_names.push_back(reader.key().data());
    reader.Next();
  }
  for (const auto& name : tensor_names) {
    tensorflow::Tensor tensor;
    if (!reader.Lookup(name, &tensor).ok()) {
      continue;
    }
    g_variable_tensors[name] = tensor;
  }
}

void save_variables_to_proto_file(const size_t& hash) {
  load_variables_from_proto_file(hash);
  std::string cache_dir = get_cache_dir();
  mkdir(cache_dir.c_str(), 0755);

  const std::string temp_file_path =
      cache_dir + "/variables_" + std::to_string(hash);

  tensorflow::BundleWriter writer(tensorflow::Env::Default(), temp_file_path);

  for (const auto& pair : g_variable_tensors) {
    const std::string& name = pair.first;
    const tensorflow::Tensor& tensor = pair.second;

    tensorflow::Status status = writer.Add(name, tensor);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to add tensor to bundle: " << status.ToString();
    }
  }

  tensorflow::Status status = writer.Finish();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to write bundle: " << status.ToString();
  }
}

const NodeDef* ConstantFoldingRewritter::get_operand(
    const std::string& name, GraphDef* graph,
    std::unordered_map<std::string, int>& node_indexes) {
  size_t colon_pos = name.find(':');
  std::string node_name = name;
  if (colon_pos != std::string::npos) {
    node_name = name.substr(0, colon_pos);
  }
  const int node_index = node_indexes.at(node_name);
  return &graph->node(node_index);
}

NodeDef* ConstantFoldingRewritter::get_mutable_operand(
    const std::string& name, GraphDef* graph,
    std::unordered_map<std::string, int>& node_indexes) {
  size_t colon_pos = name.find(':');
  std::string node_name = name;
  if (colon_pos != std::string::npos) {
    node_name = name.substr(0, colon_pos);
  }
  const int node_index = node_indexes.at(node_name);
  return graph->mutable_node(node_index);
}

const NodeDef* ConstantFoldingRewritter::get_node(std::string name) {
  size_t colon_pos = name.find(':');
  std::string node_name = name;
  if (colon_pos != std::string::npos) {
    node_name = name.substr(0, colon_pos);
  }
  if (indexes_->find(node_name) == indexes_->end()) return nullptr;
  return &(graph_->node(indexes_->at(node_name)));
}

NodeDef* ConstantFoldingRewritter::get_mutable_node(std::string name) {
  size_t colon_pos = name.find(':');
  std::string node_name = name;
  if (colon_pos != std::string::npos) {
    node_name = name.substr(0, colon_pos);
  }
  if (indexes_->find(node_name) == indexes_->end()) return nullptr;
  return graph_->mutable_node(indexes_->at(node_name));
}

bool ConstantFoldingRewritter::check_unordered_operands(
    const NodeDef* node, std::vector<std::string> op_types,
    std::vector<NodeDef**> operands) {
  for (int i = 0; i < node->input_size(); ++i) {
    NodeDef* operand = get_mutable_node(node->input(i));
    for (size_t j = 0; j < op_types.size(); ++j) {
      if (operand->op() == op_types[j] ||
          (op_types[j] == "variable" && is_variable(operand))) {
        (*operands[j]) = operand;
        break;
      }
    }
  }
  for (NodeDef** operand : operands) {
    if (operand == nullptr || *operand == nullptr) return false;
  }
  return true;
}

void ConstantFoldingRewritter::set_variable_tensor(const NodeDef* node,
                                                   Tensor& t) {
  if (node->input_size() != 1) return;
  const NodeDef* operand = get_node(node->input(0));
  if (operand != nullptr && is_variable(operand)) {
    g_variable_tensors[operand->name()] = t;
    set_variable_tensor(operand, t);
  }
}

size_t compute_vars_hash(const std::vector<const NodeDef*>& vars) {
  size_t hash = 0;
  std::hash<int64_t> hasher;
  for (const NodeDef* var : vars) {
    if (!var->attr().count("_output_shapes")) continue;
    const auto& shape_attr = var->attr().at("_output_shapes");
    const TensorShapeProto& shape = shape_attr.list().shape(0);
    for (const auto& dim : shape.dim()) {
      hash ^= hasher(dim.size()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
  }
  return hash;
}

Tensor ConstantFoldingRewritter::get_tensor(const NodeDef* node,
                                            Session* session) {
  const std::vector<const NodeDef*>& vars = {node};
  size_t hash = compute_vars_hash(vars);
  if (session == nullptr) {
    load_variables_from_proto_file(hash);
    return g_variable_tensors[node->name()];
  } else {
    std::vector<Tensor> tensors;
    session->Run({}, {node->name()}, {}, &tensors);
    g_variable_tensors[node->name()] = tensors[0];
    save_variables_to_proto_file(hash);
    return tensors[0];
  }
}

Status ConstantFoldingRewritter::get_tensors(
    const std::vector<const NodeDef*>& vars, std::vector<Tensor>& tensors,
    Session* session) {
  size_t hash = compute_vars_hash(vars);
  if (session == nullptr) {
    load_variables_from_proto_file(hash);
    for (const NodeDef* var : vars) {
      if (g_variable_tensors.find(var->name()) == g_variable_tensors.end()) {
        VLOG(0) << "[ANNC.ModelOptimizer] " << var->name()
                << " not found in global variables";
        if (var->op() != "Const" || (!var->attr().count("value")))
          return errors::NotFound(var->name());
        const auto& tensor_proto = var->attr().at("value").tensor();
        Tensor tensor;
        if (!tensor.FromProto(tensor_proto)) {
          VLOG(0) << "Cannot parse weight tensor proto: " << var->name();
          return errors::Internal("Cannot parse weight tensor proto: ",
                                  var->name());
        }
        g_variable_tensors[var->name()] = tensor;
      }
      tensors.push_back(g_variable_tensors[var->name()]);
    }
  } else {
    std::vector<std::string> var_names;
    for (size_t i = 0; i < vars.size(); ++i) {
      var_names.push_back(vars[i]->name());
    }
    TF_RETURN_IF_ERROR(session->Run({}, var_names, {}, &tensors));
    for (size_t i = 0; i < vars.size(); ++i) {
      g_variable_tensors[vars[i]->name()] = tensors[i];
      set_variable_tensor(vars[i], tensors[i]);
    }
    if (stage_ == LoaderDumpCache) save_variables_to_proto_file(hash);
  }
  return Status::OK();
}

void ConstantFoldingRewritter::fold_matmuladd_batchnorm(
    NodeDef* matmul, NodeDef* biasadd, std::vector<Tensor>& tensors) {
  const Tensor& offset = tensors[0];
  const Tensor& mean = tensors[1];
  const Tensor& scale = tensors[2];
  const Tensor& variance = tensors[3];
  const Tensor& bias = tensors[4];
  const Tensor& weight = tensors[5];
  const float epsilon = tensors[6].scalar<float>()();

  const int64_t rows = weight.dim_size(0);
  const int64_t cols = weight.dim_size(1);

  auto weight_data = weight.matrix<float>().data();
  auto bias_data = bias.flat<float>().data();
  auto scale_data = scale.flat<float>().data();
  auto offset_data = offset.flat<float>().data();
  auto mean_data = mean.flat<float>().data();
  auto variance_data = variance.flat<float>().data();

  Tensor fused_weight(DT_FLOAT, TensorShape({rows, cols}));
  Tensor fused_bias(DT_FLOAT, TensorShape({cols}));
  auto fused_weight_matrix = fused_weight.matrix<float>();
  auto fused_bias_flat = fused_bias.flat<float>();

  for (int i = 0; i < cols; ++i) {
    // scale_factor =  scale / sqrt(variance + epsilon)
    const float scale_factor =
        scale_data[i] / std::sqrt(variance_data[i] + epsilon);
    // b` = scale_factor *(b - mean) + offset
    fused_bias_flat(i) =
        (bias_data[i] - mean_data[i]) * scale_factor + offset_data[i];
    for (int j = 0; j < rows; ++j) {
      // w` = scale_factor * w
      fused_weight_matrix(j, i) = weight_data[j * cols + i] * scale_factor;
    }
  }

  auto nodes = graph_->mutable_node();
  NodeDef* fused_weight_node = nodes->Add();
  fused_weight_node->set_op("Const");
  fused_weight_node->set_name(matmul->name() + "/fused_weight");
  fused_weight_node->set_device(matmul->device());
  (*fused_weight_node->mutable_attr())["dtype"].set_type(DT_FLOAT);
  fused_weight.AsProtoTensorContent(
      (*fused_weight_node->mutable_attr())["value"].mutable_tensor());
  nodes->SwapElements(indexes_->at(matmul->input(1)), nodes->size() - 1);
  matmul->set_input(1, fused_weight_node->name());

  NodeDef* fused_bias_node = nodes->Add();
  fused_bias_node->set_op("Const");
  fused_bias_node->set_name(biasadd->name() + "/fused_bias");
  fused_bias_node->set_device(biasadd->device());
  (*fused_bias_node->mutable_attr())["dtype"].set_type(DT_FLOAT);
  fused_bias.AsProtoTensorContent(
      (*fused_bias_node->mutable_attr())["value"].mutable_tensor());

  int index = 1;
  if (is_variable(get_node(biasadd->input(0)))) index = 0;
  nodes->SwapElements(indexes_->at(biasadd->input(index)), nodes->size() - 1);
  biasadd->set_input(index, fused_bias_node->name());
}

class KPFusedMatMulBiasAddBNRewriter : public ConstantFoldingRewritter {
 public:
  KPFusedMatMulBiasAddBNRewriter(OptStage stage) { stage_ = stage; }

  bool match_and_rewrite(const NodeDef* node, GraphDef* graph,
                         std::unordered_map<std::string, int>& node_indexes,
                         Session* session) override {
    graph_ = graph;
    indexes_ = &node_indexes;
    CHECK_NODE_OK(IsAdd(*node) && node->input_size() == 2)
    NodeDef *mul{nullptr}, *sub{nullptr};
    CHECK_NODE_OK(check_unordered_operands(node, {"Mul", "Sub"}, {&mul, &sub}))
    CHECK_NODE_OK(IsMul(*mul) && mul->input_size() == 2)
    NodeDef* biasadd = get_mutable_operand(mul->input(0), graph, node_indexes);
    const NodeDef* mul_1 = get_operand(mul->input(1), graph, node_indexes);
    CHECK_NODE_OK(IsAdd(*biasadd) && biasadd->input_size() == 2)
    NodeDef *matmul{nullptr}, *bias{nullptr};
    CHECK_NODE_OK(check_unordered_operands(biasadd, {"MatMul", "variable"},
                                           {&matmul, &bias}))
    CHECK_NODE_OK(matmul->op() == "MatMul")
    CHECK_NODE_OK(IsSub(*sub) && sub->input_size() == 2)
    const NodeDef* offset = get_operand(sub->input(0), graph, node_indexes);
    const NodeDef* mul_2 = get_operand(sub->input(1), graph, node_indexes);
    CHECK_NODE_OK(IsMul(*mul_2) && mul_2->input_size() == 2)
    NodeDef* mean;
    CHECK_NODE_OK(check_unordered_operands(mul_2, {"variable"}, {&mean}))
    CHECK_NODE_OK(IsMul(*mul_1) && mul_1->input_size() == 2)
    NodeDef *rsqrt{nullptr}, *scale{nullptr};
    CHECK_NODE_OK(check_unordered_operands(mul_1, {"Rsqrt", "variable"},
                                           {&rsqrt, &scale}))
    CHECK_NODE_OK(IsRsqrt(*rsqrt) && rsqrt->input_size() == 1)
    const NodeDef* add_3 = get_operand(rsqrt->input(0), graph, node_indexes);
    CHECK_NODE_OK(IsAdd(*add_3) && add_3->input_size() == 2)
    NodeDef *variance{nullptr}, *epsilon{nullptr};
    CHECK_NODE_OK(check_unordered_operands(add_3, {"Const", "variable"},
                                           {&epsilon, &variance}))
    CHECK_NODE_OK(is_variable(offset) && is_variable(mean) &&
                  is_variable(scale) && is_variable(variance))
    const NodeDef* weight = get_operand(matmul->input(1), graph, node_indexes);
    CHECK_NODE_OK(is_variable(weight) && is_variable(bias))

    const std::vector<const NodeDef*> vars = {offset, mean,   scale,  variance,
                                              bias,   weight, epsilon};

    std::vector<Tensor> tensors;
    CHECK_NODE_OK(get_tensors(vars, tensors, session).ok());
    const NodeDef* lhs = get_operand(matmul->input(0), graph, node_indexes);
    int lhs_dim1 = -1;
    if (!lhs->attr().count("_output_shapes")) return false;
    std::string lhs_name = matmul->input(0);
    int pos_index = 0;
    size_t lhs_in_pos = lhs_name.find(':');
    if (lhs_in_pos != std::string::npos) {
      pos_index = std::stoi(lhs_name.substr(lhs_in_pos + 1));
    }
    const TensorShapeProto& lhs_shape =
        lhs->attr().at("_output_shapes").list().shape(pos_index);
    if (lhs_shape.dim_size() > 1) {
      lhs_dim1 = lhs_shape.dim(1).size();
    }
    if (tensors[5].dim_size(0) != lhs_dim1) {
      VLOG(0) << "ERROR: [" << matmul->name() << "]" << tensors[5].dim_size(0)
              << " != " << lhs_dim1;
      return false;
    }

    fold_matmuladd_batchnorm(matmul, biasadd, tensors);
    mul->set_input(0, mul->input(1));

    replace_all_users_with(node, 0, biasadd, 0, graph);
    VLOG(0) << "-- matched <matmul+biasadd+bn>:" << matmul->name()
            << ", session is [" << session << "]";
    return true;
  }
};

class KPFusedReluRewriter : public ConstantFoldingRewritter {
 public:
  KPFusedReluRewriter(OptStage stage) { stage_ = stage; }

  bool match_and_rewrite(const NodeDef* node, GraphDef* graph,
                         std::unordered_map<std::string, int>& node_indexes,
                         Session* session) override {
    CHECK_NODE_OK(IsRealDiv(*node) && node->input_size() == 2)
    NodeDef* add = get_mutable_operand(node->input(0), graph, node_indexes);
    NodeDef* vdiv = get_mutable_operand(node->input(1), graph, node_indexes);
    CHECK_NODE_OK(IsAdd(*add) && add->input_size() == 2)
    CHECK_NODE_OK((IsConstant(*vdiv) || IsHostConstant(*vdiv)) &&
                  HasNodeAttr(*vdiv, "value"))
    TensorProto* vdiv_tensor =
        (*vdiv->mutable_attr())["value"].mutable_tensor();
    float vidv_value = vdiv_tensor->mutable_float_val()->data()[0];
    CHECK_NODE_OK(std::fabs(vidv_value - 2.0f) <= 1e-5f)

    const NodeDef* add_1 = get_operand(add->input(0), graph, node_indexes);
    const NodeDef* abs = get_operand(add->input(1), graph, node_indexes);
    CHECK_NODE_OK(abs->op() == "Abs" && abs->input_size() == 1)
    CHECK_NODE_OK(IsAdd(*add_1) && add_1->input_size() == 2)
    CHECK_NODE_OK(get_operand(abs->input(0), graph, node_indexes) == add_1)

    auto nodes = graph->mutable_node();
    NodeDef* relu = nodes->Add();
    relu->set_name(node->name() + "/relu");
    relu->set_op("Relu");
    relu->set_device(node->device());
    relu->add_input(abs->input(0));
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);
    add->set_input(0, abs->name());
    add->set_input(1, abs->name());

    replace_all_users_with(node, 0, relu, 0, graph);
    VLOG(0) << "-- matched relu: " << node->name();
    return true;
  }
};

class KPFusedRelu2Rewriter : public ConstantFoldingRewritter {
 public:
  KPFusedRelu2Rewriter(OptStage stage) { stage_ = stage; }

  bool match_and_rewrite(const NodeDef* node, GraphDef* graph,
                         std::unordered_map<std::string, int>& node_indexes,
                         Session* session) override {
    indexes_ = &node_indexes;
    graph_ = graph;
    CHECK_NODE_OK(IsMul(*node) && node->input_size() == 2)
    NodeDef *vmul{nullptr}, *add{nullptr};
    CHECK_NODE_OK(
        check_unordered_operands(node, {"Const", "AddV2"}, {&vmul, &add}) ||
        check_unordered_operands(node, {"Const", "Add"}, {&vmul, &add}))
    CHECK_NODE_OK(HasNodeAttr(*vmul, "value"))
    TensorProto* vmul_tensor =
        (*vmul->mutable_attr())["value"].mutable_tensor();
    const float* data =
        reinterpret_cast<const float*>(vmul_tensor->tensor_content().data());
    CHECK_NODE_OK(std::fabs(data[0] - 0.5f) <= 1e-5f)

    NodeDef *abs{nullptr}, *add_1{nullptr};
    CHECK_NODE_OK(
        check_unordered_operands(add, {"Abs", "AddV2"}, {&abs, &add_1}) ||
        check_unordered_operands(add, {"Abs", "Add"}, {&abs, &add_1}))
    CHECK_NODE_OK(get_operand(abs->input(0), graph, node_indexes) == add_1)

    auto nodes = graph->mutable_node();
    NodeDef* relu = nodes->Add();
    relu->set_name(node->name() + "/relu");
    relu->set_op("Relu");
    relu->set_device(node->device());
    relu->add_input(abs->input(0));
    nodes->SwapElements(node_indexes.at(node->name()), nodes->size() - 1);
    add->set_input(0, abs->name());
    add->set_input(1, abs->name());

    replace_all_users_with(node, 0, relu, 0, graph);
    VLOG(0) << "-- matched relu2: " << node->name();
    return true;
  }
};

bool enabled_aarch64_cf_rewriters() {
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

Status dump_saved_model(const MetaGraphDef& meta_graph_def,
                        const string& export_dir, bool dump_as_text = false) {
  SavedModel saved_model_proto;
  saved_model_proto.mutable_meta_graphs()->Add()->CopyFrom(meta_graph_def);
  mkdir(export_dir.c_str(), 0755);
  const std::string saved_model_pb_path =
      tensorflow::io::JoinPath(export_dir, "saved_model.pb");
  if (dump_as_text) {
    TF_CHECK_OK(tensorflow::WriteTextProto(
        tensorflow::Env::Default(), absl::StrCat(saved_model_pb_path, "txt"),
        saved_model_proto));
  } else {
    TF_CHECK_OK(tensorflow::WriteBinaryProto(
        tensorflow::Env::Default(), saved_model_pb_path, saved_model_proto));
  }
  LOG(INFO) << "Model saved: " << saved_model_pb_path;
  return Status::OK();
}

void run_annc_constant_folding(GraphDef* graph, Session* session) {
  std::vector<std::unique_ptr<ConstantFoldingRewritter>> rewriters;

  if (enabled_aarch64_cf_rewriters()) {
    // default disable all rewriters
    if (FLAGS_annc_cf_matmul_batchnorm) {
      if (FLAGS_annc_cf_matmul_batchnorm == 1 && session != nullptr) {
        rewriters.push_back(std::make_unique<KPFusedMatMulBiasAddBNRewriter>(
            LoaderNoDumpCache));
      } else if (FLAGS_annc_cf_matmul_batchnorm == 2) {
        OptStage stage = (session != nullptr) ? LoaderDumpCache : Remapper;
        rewriters.push_back(
            std::make_unique<KPFusedMatMulBiasAddBNRewriter>(stage));
      }
    }
    if (FLAGS_annc_cf_relu) {
      OptStage stage = (session != nullptr) ? LoaderNoDumpCache : Remapper;
      rewriters.push_back(std::make_unique<KPFusedReluRewriter>(stage));
      rewriters.push_back(std::make_unique<KPFusedRelu2Rewriter>(stage));
    }
  }

  while (true) {
    int node_size = graph->node_size();
    std::unordered_map<std::string, int> node_indexes;
    for (int i = 0; i < node_size; ++i) {
      node_indexes[graph->node(i).name()] = i;
    }
    bool is_matched = false;
    for (int i = 0; i < node_size; ++i) {
      for (auto& rewriter : rewriters) {
        is_matched = rewriter->match_and_rewrite(graph->mutable_node(i), graph,
                                                 node_indexes, session);
        if (is_matched) break;
      }
      if (is_matched) break;
    }
    if (!is_matched) break;
  }
}

void run_annc_constant_folding(MetaGraphDef& meta_graph_def, Session* session) {
  run_annc_constant_folding(meta_graph_def.mutable_graph_def(), session);
  if (!FLAGS_annc_cf_dump.empty()) {
    const std::string export_dir = FLAGS_annc_cf_dump;
    dump_saved_model(meta_graph_def, export_dir, FLAGS_annc_cf_dump_text);
  }
}
}  // namespace annc
