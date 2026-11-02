#include "tf_frontend.h"

#include <google/protobuf/text_format.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <filesystem>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_set>

namespace annc::tf2atir {
namespace {

bool parseNonNegativeIndex(const std::string& text, int& index) {
  if (text.empty()) return false;
  for (char c : text) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  }
  try {
    std::size_t consumed = 0;
    const long value = std::stol(text, &consumed);
    if (consumed != text.size() || value > static_cast<long>(INT_MAX))
      return false;
    index = static_cast<int>(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool hasSuffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

bool readTextProto(const std::string& path, google::protobuf::Message& message,
                   std::string& error) {
  std::ifstream input(path);
  if (!input) {
    error = "cannot open protobuf text file: " + path;
    return false;
  }
  std::ostringstream content;
  content << input.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(content.str(), &message)) {
    error = "cannot parse protobuf text file: " + path;
    return false;
  }
  return true;
}

bool readBinaryProto(const std::string& path,
                     google::protobuf::Message& message, std::string& error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "cannot open protobuf file: " + path;
    return false;
  }
  if (!message.ParseFromIstream(&input)) {
    error = "cannot parse protobuf file: " + path;
    return false;
  }
  return true;
}

}  // namespace

std::string TensorRef::canonicalName() const {
  return node + ":" + std::to_string(output_index);
}

std::size_t TensorRefHash::operator()(const TensorRef& ref) const noexcept {
  const std::size_t node_hash = std::hash<std::string>{}(ref.node);
  const std::size_t index_hash = std::hash<int>{}(ref.output_index);
  return node_hash ^ (index_hash + static_cast<std::size_t>(0x9e3779b9) +
                      (node_hash << 6) + (node_hash >> 2));
}

const TfNode* TfGraph::findNode(const std::string& name) const {
  auto it = std::find_if(nodes.begin(), nodes.end(),
                         [&](const TfNode& node) { return node.name == name; });
  return it == nodes.end() ? nullptr : &*it;
}

bool TfModelLoader::load(const std::string& model_path,
                         const std::vector<std::string>& explicit_output_names,
                         LoadedTfModel& result, std::string& error) const {
  result = LoadedTfModel{};
  error.clear();
  namespace fs = std::filesystem;
  fs::path path(model_path);
  if (fs::is_directory(path)) {
    const fs::path binary = path / "saved_model.pb";
    const fs::path text = path / "saved_model.pbtxt";
    if (fs::is_regular_file(binary))
      path = binary;
    else if (fs::is_regular_file(text))
      path = text;
    else {
      error =
          "SavedModel directory has no saved_model.pb or saved_model.pbtxt: " +
          model_path;
      return false;
    }
  }
  if (!fs::is_regular_file(path)) {
    error = "model path is neither a GraphDef nor a SavedModel: " + model_path;
    return false;
  }

  const std::string filename = path.filename().string();
  const bool is_text = hasSuffix(filename, ".pbtxt");
  const bool is_saved_model =
      filename == "saved_model.pb" || filename == "saved_model.pbtxt";
  if (!is_saved_model) {
    if (explicit_output_names.empty()) {
      error = "GraphDef input requires at least one --output_tensor";
      return false;
    }
    result.owned_graph_def = std::make_unique<tensorflow::GraphDef>();
    const bool loaded =
        is_text
            ? readTextProto(path.string(), *result.owned_graph_def, error)
            : readBinaryProto(path.string(), *result.owned_graph_def, error);
    if (!loaded) return false;
    result.graph_def = result.owned_graph_def.get();
    result.output_names = explicit_output_names;
    return true;
  }

  result.saved_model = std::make_unique<tensorflow::SavedModel>();
  const bool loaded =
      is_text ? readTextProto(path.string(), *result.saved_model, error)
              : readBinaryProto(path.string(), *result.saved_model, error);
  if (!loaded) return false;
  if (result.saved_model->meta_graphs_size() == 0) {
    error = "SavedModel contains no MetaGraphDef: " + model_path;
    return false;
  }
  const auto& signatures = result.saved_model->meta_graphs(0).signature_def();
  const auto signature = signatures.find("serving_default");
  if (signature == signatures.end() || signature->second.outputs().empty()) {
    error = "SavedModel requires serving_default SignatureDef outputs";
    return false;
  }
  result.graph_def = &result.saved_model->meta_graphs(0).graph_def();
  for (const auto& entry : signature->second.outputs())
    result.output_names.push_back(entry.second.name());
  return true;
}

bool TfGraphParser::parseTensorName(const std::string& value, TensorRef& result,
                                    std::string& error) {
  if (value.empty()) {
    error = "tensor name is empty";
    return false;
  }
  if (value.front() == '^') {
    error = "control input is not a data tensor: " + value;
    return false;
  }

  const std::size_t colon = value.rfind(':');
  if (colon == std::string::npos) {
    result = TensorRef{value, 0};
    return true;
  }
  if (colon == 0 || colon + 1 == value.size()) {
    error = "malformed tensor name: " + value;
    return false;
  }
  int output_index = 0;
  if (!parseNonNegativeIndex(value.substr(colon + 1), output_index)) {
    error = "tensor output index must be a non-negative integer: " + value;
    return false;
  }
  result = TensorRef{value.substr(0, colon), output_index};
  return true;
}

bool TfGraphParser::parseDataInput(const std::string& value, TensorRef& result,
                                   std::string& error) {
  if (!value.empty() && value.front() == '^') return true;
  return parseTensorName(value, result, error);
}

bool TfGraphParser::parse(const tensorflow::GraphDef& graph,
                          const std::vector<std::string>& output_names,
                          TfGraph& result, std::string& error) const {
  result = TfGraph{};
  error.clear();
  if (output_names.empty()) {
    error = "no graph outputs were provided";
    return false;
  }

  std::unordered_map<std::string, const tensorflow::NodeDef*> node_map;
  node_map.reserve(static_cast<std::size_t>(graph.node_size()));
  for (const auto& node : graph.node()) {
    if (node.name().empty()) {
      error = "GraphDef contains a node with an empty name";
      return false;
    }
    if (!node_map.emplace(node.name(), &node).second) {
      error = "GraphDef contains duplicate node name: " + node.name();
      return false;
    }
  }

  std::vector<TensorRef> roots;
  for (const std::string& output_name : output_names) {
    TensorRef output;
    if (!parseTensorName(output_name, output, error)) return false;
    if (!node_map.count(output.node)) {
      error = "output tensor '" + output_name + "' does not exist in GraphDef";
      return false;
    }
    roots.push_back(output);
  }

  std::unordered_set<std::string> reachable;
  std::queue<std::string> worklist;
  for (const TensorRef& root : roots) {
    if (reachable.insert(root.node).second) worklist.push(root.node);
  }
  while (!worklist.empty()) {
    const std::string current = worklist.front();
    worklist.pop();
    const tensorflow::NodeDef* node = node_map.at(current);
    for (const std::string& input : node->input()) {
      if (!input.empty() && input.front() == '^') {
        const std::string control_name = input.substr(1);
        if (!node_map.count(control_name)) {
          error = "node '" + current +
                  "' references missing control input: " + control_name;
          return false;
        }
        if (reachable.insert(control_name).second) worklist.push(control_name);
        continue;
      }
      TensorRef ref;
      if (!parseDataInput(input, ref, error)) return false;
      if (!node_map.count(ref.node)) {
        error = "node '" + current + "' references missing input tensor '" +
                (ref.output_index == 0 ? ref.node : ref.canonicalName()) + "'";
        return false;
      }
      if (reachable.insert(ref.node).second) worklist.push(ref.node);
    }
  }

  std::unordered_map<std::string, TfNode> parsed;
  parsed.reserve(reachable.size());
  for (const auto& node : graph.node()) {
    if (!reachable.count(node.name())) continue;
    TfNode parsed_node;
    parsed_node.name = node.name();
    parsed_node.op = node.op();
    parsed_node.source = &node;
    for (const std::string& input : node.input()) {
      if (!input.empty() && input.front() == '^') {
        parsed_node.control_inputs.push_back(input.substr(1));
        continue;
      }
      TensorRef ref;
      if (!parseDataInput(input, ref, error)) return false;
      parsed_node.inputs.push_back(ref);
    }
    parsed.emplace(parsed_node.name, std::move(parsed_node));
  }
  auto recordOutput = [&](const TensorRef& ref) {
    std::vector<TensorRef>& outputs = parsed.at(ref.node).outputs;
    const bool already_recorded =
        std::any_of(outputs.begin(), outputs.end(),
                    [&](const TensorRef& output) { return output == ref; });
    if (!already_recorded) outputs.push_back(ref);
  };
  for (const TensorRef& root : roots) recordOutput(root);
  for (const auto& entry : parsed) {
    for (const TensorRef& input : entry.second.inputs) recordOutput(input);
  }
  for (auto& entry : parsed) {
    std::sort(entry.second.outputs.begin(), entry.second.outputs.end(),
              [](const TensorRef& lhs, const TensorRef& rhs) {
                return lhs.output_index < rhs.output_index;
              });
  }

  std::unordered_map<std::string, std::size_t> indegree;
  std::unordered_map<std::string, std::vector<std::string>> consumers;
  for (const auto& node : graph.node()) {
    if (reachable.count(node.name())) indegree[node.name()] = 0;
  }
  for (const auto& node : graph.node()) {
    if (!reachable.count(node.name())) continue;
    const TfNode& parsedNode = parsed.at(node.name());
    for (const TensorRef& input : parsedNode.inputs) {
      const TfNode& producer = parsed.at(input.node);
      const bool loopBackedge =
          (parsedNode.op == "Merge" || parsedNode.op == "RefMerge") &&
          (producer.op == "NextIteration" ||
           producer.op == "RefNextIteration");
      if (loopBackedge) continue;
      ++indegree[node.name()];
      consumers[input.node].push_back(node.name());
    }
    for (const std::string& control : parsedNode.control_inputs) {
      ++indegree[node.name()];
      consumers[control].push_back(node.name());
    }
  }

  std::queue<std::string> ready;
  for (const auto& node : graph.node()) {
    if (reachable.count(node.name()) && indegree[node.name()] == 0)
      ready.push(node.name());
  }
  while (!ready.empty()) {
    const std::string name = ready.front();
    ready.pop();
    result.nodes.push_back(std::move(parsed.at(name)));
    for (const std::string& consumer : consumers[name]) {
      if (--indegree[consumer] == 0) ready.push(consumer);
    }
  }
  if (result.nodes.size() != reachable.size()) {
    result = TfGraph{};
    error = "reachable GraphDef contains a data dependency cycle";
    return false;
  }
  for (const TfNode& node : result.nodes) {
    if (node.op == "Placeholder" || node.op == "PlaceholderV2")
      result.inputs.push_back(TensorRef{node.name, 0});
  }
  result.outputs = std::move(roots);
  return true;
}

}  // namespace annc::tf2atir
