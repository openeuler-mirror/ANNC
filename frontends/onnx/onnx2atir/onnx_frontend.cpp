#include "onnx_frontend.h"

#include <fstream>
#include <queue>
#include <unordered_set>

namespace annc::onnx2atir {

const onnx::TensorProto* OnnxGraph::findInitializer(
    const std::string& name) const {
  auto it = initializers.find(name);
  return it == initializers.end() ? nullptr : it->second;
}

bool OnnxModelLoader::load(const std::string& model_path,
                           LoadedOnnxModel& result, std::string& error) const {
  error.clear();
  std::ifstream input(model_path, std::ios::binary);
  if (!input) {
    error = "cannot open model file '" + model_path + "'";
    return false;
  }
  if (!result.model.ParseFromIstream(&input)) {
    error = "failed to parse '" + model_path +
            "' as an ONNX ModelProto (not a valid ONNX model, or the file is "
            "truncated)";
    return false;
  }
  if (!result.model.has_graph()) {
    error = "model '" + model_path + "' carries no graph";
    return false;
  }
  if (result.model.functions_size() > 0) {
    error = "model '" + model_path + "' declares " +
            std::to_string(result.model.functions_size()) +
            " local function(s); function bodies are outside the imported "
            "graph and are not supported";
    return false;
  }
  result.opset_version = 0;
  result.opset_imports.clear();
  for (const onnx::OperatorSetIdProto& entry : result.model.opset_import()) {
    const std::string domain = entry.domain().empty() ? "ai.onnx" : entry.domain();
    result.opset_imports.emplace_back(domain, entry.version());
    if (domain == "ai.onnx") result.opset_version = entry.version();
  }
  return true;
}

namespace {

// A node whose attribute carries a subgraph cannot be represented without
// region support (the subgraph operands have no landing place), and importing
// only part of it would break the node-conservation accounting. The parser
// rejects the whole model instead. This covers If/Loop/Scan and any future op
// that grows a GRAPH-typed attribute. Sparse-tensor attributes are not
// subgraphs: they are a payload form this frontend cannot decode, and the
// adapter reports them as a per-node fallback instead of rejecting the model.
const onnx::AttributeProto* findSubgraphAttribute(const onnx::NodeProto& node) {
  for (const onnx::AttributeProto& attr : node.attribute()) {
    if (attr.has_g() || attr.graphs_size() > 0) return &attr;
  }
  return nullptr;
}

}  // namespace

bool OnnxGraphParser::parse(const onnx::GraphProto& graph, OnnxGraph& result,
                            std::string& error) const {
  error.clear();
  result = OnnxGraph();

  for (const onnx::TensorProto& init : graph.initializer()) {
    if (init.name().empty()) {
      error = "graph carries an unnamed initializer";
      return false;
    }
    // A repeated name would put two values behind one tensor name: the module
    // would carry two constants under it while the conservation accounting
    // counts one. ONNX requires unique tensor names, so refuse instead.
    if (!result.initializers.emplace(init.name(), &init).second) {
      error = "graph declares initializer '" + init.name() + "' twice";
      return false;
    }
    result.initializer_names.push_back(init.name());
  }
  for (const onnx::SparseTensorProto& sparse : graph.sparse_initializer()) {
    // Sparse initializers are recorded but not decoded; the adapter reports
    // them as explicit fallbacks rather than dropping them silently.
    if (!sparse.values().name().empty())
      result.sparse_initializers.push_back(sparse.values().name());
  }

  for (const onnx::ValueInfoProto& value : graph.input())
    result.graph_input_info[value.name()] = &value;

  // Declared tensor types: graph input/output/value_info all contribute. The
  // four measured models leave value_info empty, so these are a fallback the
  // resolver consults only when inference cannot decide.
  for (const onnx::ValueInfoProto& value : graph.input())
    result.declared_types[value.name()] = &value;
  for (const onnx::ValueInfoProto& value : graph.output())
    result.declared_types[value.name()] = &value;
  for (const onnx::ValueInfoProto& value : graph.value_info())
    result.declared_types[value.name()] = &value;

  // A name listed as a graph input *and* as an initializer is a constant, not
  // a runtime input. This frontend keeps the initializer reading.
  for (const onnx::ValueInfoProto& value : graph.input()) {
    if (result.initializers.count(value.name())) continue;
    result.inputs.push_back(value.name());
  }
  for (const onnx::ValueInfoProto& value : graph.output())
    result.outputs.push_back(value.name());

  // Node names must be unique because they key provenance and diagnostics.
  std::unordered_set<std::string> seenNames;
  for (int i = 0; i < graph.node_size(); ++i) {
    const onnx::NodeProto& proto = graph.node(i);
    // Structural validation first: the diagnostics and the synthesized names
    // below read proto.output(0), which is out of bounds on a zero-output
    // node. An empty op_type is refused outright as well -- falling back to
    // the first output's name would forge the provenance trail with a tensor
    // name that is not an operator at all.
    if (proto.output_size() == 0) {
      error = "node '" +
              (proto.name().empty() ? "onnx_node_" + std::to_string(i)
                                    : proto.name()) +
              "' has no output";
      return false;
    }
    if (proto.op_type().empty()) {
      error =
          "node '" + (proto.name().empty() ? proto.output(0) : proto.name()) +
          "' has an empty op_type";
      return false;
    }
    if (const onnx::AttributeProto* attr = findSubgraphAttribute(proto)) {
      error = "control flow is not supported: node '" +
              (proto.name().empty() ? proto.output(0) : proto.name()) + "' (" +
              proto.op_type() + ") carries subgraph attribute '" +
              attr->name() + "'";
      return false;
    }
    OnnxNode node;
    node.index = i;
    node.source = &proto;
    node.name = proto.name().empty() ? proto.output(0) : proto.name();
    if (!seenNames.insert(node.name).second) {
      // Duplicate names would make provenance ambiguous, so the later node is
      // renamed. The original is not recoverable from the module -- the
      // disambiguated name is what both the metadata and the provenance row
      // carry.
      node.name += "__onnx_idx" + std::to_string(i);
      seenNames.insert(node.name);
    }
    node.op_type = proto.op_type();
    node.domain = proto.domain();
    for (const std::string& input : proto.input()) {
      if (input.empty()) continue;  // omitted optional input
      node.inputs.push_back(input);
    }
    for (const std::string& output : proto.output()) node.outputs.push_back(output);
    result.nodes.push_back(std::move(node));
  }

  // Topological sort (Kahn). The ONNX spec requires nodes in topological
  // order, but a malformed model must not turn into a mis-ordered import.
  std::unordered_map<std::string, int> producer;
  for (int i = 0; i < static_cast<int>(result.nodes.size()); ++i)
    for (const std::string& out : result.nodes[i].outputs) producer[out] = i;

  std::vector<int> indegree(result.nodes.size(), 0);
  std::vector<std::vector<int>> successors(result.nodes.size());
  for (int i = 0; i < static_cast<int>(result.nodes.size()); ++i) {
    for (const std::string& in : result.nodes[i].inputs) {
      auto it = producer.find(in);
      if (it == producer.end()) continue;  // graph input or initializer
      if (it->second == i) {
        // A node consuming its own output is not a legal ONNX value chain.
        // Skipping the edge would let the sort accept it (a self-loop never
        // reaches a nonzero indegree) and the failure would only surface later
        // as a misleading "unknown input tensor" in the builder.
        error = "node '" + result.nodes[i].name +
                "' consumes its own output '" + in + "'";
        return false;
      }
      successors[it->second].push_back(i);
      ++indegree[i];
    }
  }
  std::vector<int> ordered;
  ordered.reserve(result.nodes.size());
  std::queue<int> ready;
  for (int i = 0; i < static_cast<int>(indegree.size()); ++i)
    if (indegree[i] == 0) ready.push(i);
  while (!ready.empty()) {
    const int current = ready.front();
    ready.pop();
    ordered.push_back(current);
    for (int next : successors[current])
      if (--indegree[next] == 0) ready.push(next);
  }
  if (ordered.size() != result.nodes.size()) {
    error =
        "graph contains a dependency cycle; only directed acyclic graphs are "
        "supported";
    return false;
  }
  std::vector<OnnxNode> sorted;
  sorted.reserve(ordered.size());
  for (int index : ordered) sorted.push_back(std::move(result.nodes[index]));
  result.nodes = std::move(sorted);
  return true;
}

}  // namespace annc::onnx2atir
