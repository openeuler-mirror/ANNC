#include "node_info_adapter.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "tensor_proto_decoder.h"

namespace annc::tf2atir {
namespace {

bool findLinearBranch(
    const std::string& start,
    const std::unordered_map<std::string, std::vector<std::size_t>>& users,
    const std::vector<annc::NodeInfo>& nodes, std::size_t& mergeIndex,
    std::string& end, std::vector<annc::NodeInfo>& body,
    std::unordered_set<std::string>& consumed, std::string& error) {
  std::string value = start;
  std::unordered_set<std::string> seen;
  while (true) {
    if (!seen.insert(value).second) {
      error = "Switch branch contains a cycle at '" + value + "'";
      return false;
    }
    auto usersIt = users.find(value);
    if (usersIt == users.end() || usersIt->second.empty()) {
      error = "Switch branch from '" + start + "' does not reach a Merge";
      return false;
    }
    if (usersIt->second.size() != 1) {
      error = "Switch branch from '" + start +
              "' has fan-out; only linear branches are currently supported";
      return false;
    }
    const annc::NodeInfo& user = nodes[usersIt->second.front()];
    if (user.op_type == "Merge" || user.op_type == "RefMerge") {
      mergeIndex = usersIt->second.front();
      end = value;
      return true;
    }
    if (consumed.count(user.name)) {
      error = "Switch branches overlap at node '" + user.name + "'";
      return false;
    }
    if (user.outputs.size() != 1) {
      error = "Switch branch node '" + user.name +
              "' must have exactly one output";
      return false;
    }
    body.push_back(user);
    consumed.insert(user.name);
    value = user.outputs.front().name;
  }
}

bool buildSwitchDiamonds(std::vector<annc::NodeInfo>& nodes,
                         std::string& error) {
  std::unordered_map<std::string, std::vector<std::size_t>> users;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    for (const std::string& input : nodes[i].inputs) {
      auto& inputUsers = users[input];
      if (inputUsers.empty() || inputUsers.back() != i)
        inputUsers.push_back(i);
    }
  }

  std::unordered_map<std::size_t, annc::NodeInfo> replacements;
  std::unordered_set<std::string> consumedNodes;
  for (std::size_t switchIndex = 0; switchIndex < nodes.size(); ++switchIndex) {
    const annc::NodeInfo& sw = nodes[switchIndex];
    if (sw.op_type != "Switch" && sw.op_type != "RefSwitch") continue;
    // Best-effort reconstruction: a Switch that cannot be rebuilt as a clean
    // linear diamond is left untouched.  Its raw "Switch" node then reaches
    // the builder's opaque fallback (no OpSpec row), which is exactly how
    // these graphs converted before the diamond reconstruction existed.
    // Skipping must not fail the conversion: real serving graphs (e.g. the
    // presort family) contain dead Switch outputs, fan-out branches, and
    // other shapes that are not reconstructible diamonds.
    if (sw.inputs.size() != 2 || sw.outputs.size() != 2) {
      error = "Switch node '" + sw.name +
              "' must have exactly two inputs and two outputs";
      continue;
    }
    const std::string& falseAlias = sw.outputs[0].name;
    const std::string& trueAlias = sw.outputs[1].name;
    if (falseAlias == trueAlias) {
      error = "Switch node '" + sw.name + "' has identical branch outputs";
      continue;
    }

    annc::NodeInfo structured;
    std::unordered_set<std::string> consumed{sw.name};
    std::size_t falseMergeIndex = 0;
    std::size_t trueMergeIndex = 0;
    std::string falseEnd, trueEnd;
    if (!findLinearBranch(falseAlias, users, nodes, falseMergeIndex, falseEnd,
                          structured.switch_false_nodes, consumed, error) ||
        !findLinearBranch(trueAlias, users, nodes, trueMergeIndex, trueEnd,
                          structured.switch_true_nodes, consumed, error))
      continue;
    if (falseMergeIndex != trueMergeIndex) {
      error = "Switch node '" + sw.name +
              "' branches reach different Merge nodes";
      continue;
    }
    if (falseEnd == trueEnd) {
      error = "Switch node '" + sw.name +
              "' branches converge before the Merge node";
      continue;
    }
    const std::size_t mergeIndex = falseMergeIndex;
    const annc::NodeInfo& merge = nodes[mergeIndex];
    if (merge.inputs.size() != 2 || merge.outputs.size() < 2) {
      error = "Merge node '" + merge.name +
              "' must have two inputs and at least two outputs";
      continue;
    }
    const bool falseFirst =
        merge.inputs[0] == falseEnd && merge.inputs[1] == trueEnd;
    const bool trueFirst =
        merge.inputs[0] == trueEnd && merge.inputs[1] == falseEnd;
    if (!falseFirst && !trueFirst) {
      error = "Merge node '" + merge.name +
              "' does not join the reconstructed Switch branches";
      continue;
    }
    std::unordered_set<std::string> falseValues{falseAlias};
    std::unordered_set<std::string> trueValues{trueAlias};
    for (const annc::NodeInfo& branchNode : structured.switch_false_nodes)
      for (const annc::OutputInfo& output : branchNode.outputs)
        falseValues.insert(output.name);
    for (const annc::NodeInfo& branchNode : structured.switch_true_nodes)
      for (const annc::OutputInfo& output : branchNode.outputs)
        trueValues.insert(output.name);
    auto hasCrossBranchInput = [](const std::vector<annc::NodeInfo>& body,
                                  const std::unordered_set<std::string>& other,
                                  std::string& offendingNode) {
      for (const annc::NodeInfo& branchNode : body)
        for (const std::string& input : branchNode.inputs)
          if (other.count(input)) {
            offendingNode = branchNode.name;
            return true;
          }
      return false;
    };
    std::string offendingNode;
    if (hasCrossBranchInput(structured.switch_false_nodes, trueValues,
                            offendingNode) ||
        hasCrossBranchInput(structured.switch_true_nodes, falseValues,
                            offendingNode)) {
      error = "Switch branch node '" + offendingNode +
              "' depends on a value from the other branch";
      continue;
    }
    // Keep the structured op at the Merge position. All external values used
    // inside either branch are guaranteed to dominate that point in the
    // topologically sorted graph, but need not dominate the original Switch.
    consumed.insert(merge.name);
    bool overlapsExisting = false;
    for (const std::string& name : consumed) {
      if (consumedNodes.count(name)) {
        error = "overlapping Switch diamonds are not supported (node '" +
                name + "')";
        overlapsExisting = true;
        break;
      }
    }
    if (overlapsExisting) continue;
    consumedNodes.insert(consumed.begin(), consumed.end());

    structured.name = merge.name;
    structured.op_type = "ANNCStructuredSwitch";
    structured.inputs = {sw.inputs[0], sw.inputs[1]};
    structured.outputs = merge.outputs;
    structured.switch_data_inputs = {sw.inputs[0]};
    structured.switch_predicate_input = sw.inputs[1];
    structured.switch_false_aliases = {falseAlias};
    structured.switch_true_aliases = {trueAlias};
    structured.switch_false_yield = falseEnd;
    structured.switch_true_yield = trueEnd;
    structured.switch_false_index = falseFirst ? 0 : 1;
    structured.switch_true_index = falseFirst ? 1 : 0;
    if (replacements.find(mergeIndex) != replacements.end()) {
      error = "multiple Switch nodes reconstruct the same Merge node '" +
              merge.name + "'";
      continue;
    }
    replacements.emplace(mergeIndex, std::move(structured));
  }

  if (replacements.empty()) return true;
  std::vector<annc::NodeInfo> rewritten;
  rewritten.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    auto replacement = replacements.find(i);
    if (replacement != replacements.end())
      rewritten.push_back(std::move(replacement->second));
    if (!consumedNodes.count(nodes[i].name))
      rewritten.push_back(std::move(nodes[i]));
  }
  nodes = std::move(rewritten);
  return true;
}

annc::OutputInfo makeOutput(const TensorRef& ref,
                            const TensorDescriptor& descriptor) {
  return annc::OutputInfo{
      ref.output_index == 0 ? ref.node : ref.canonicalName(), descriptor.dtype,
      descriptor.shape, descriptor.rank_known};
}

void copyComputeAttrs(const tensorflow::NodeDef& node, annc::NodeInfo& info) {
  for (const auto& entry : node.attr()) {
    const std::string& name = entry.first;
    const tensorflow::AttrValue& value = entry.second;
    switch (value.value_case()) {
      case tensorflow::AttrValue::kI:
        info.attrs[name] = static_cast<int64_t>(value.i());
        break;
      case tensorflow::AttrValue::kF:
        info.attrs[name] = static_cast<double>(value.f());
        break;
      case tensorflow::AttrValue::kB:
        info.attrs[name] = value.b();
        break;
      case tensorflow::AttrValue::kS:
        info.attrs[name] = value.s();
        break;
      case tensorflow::AttrValue::kList: {
        const auto& list = value.list();
        if (list.i_size() > 0) {
          std::vector<int64_t> values;
          values.reserve(static_cast<std::size_t>(list.i_size()));
          for (int64_t item : list.i()) values.push_back(item);
          info.attrs[name] = std::move(values);
        } else if (list.f_size() > 0) {
          std::vector<double> values;
          values.reserve(static_cast<std::size_t>(list.f_size()));
          for (float item : list.f()) values.push_back(item);
          info.attrs[name] = std::move(values);
        } else if (list.b_size() > 0) {
          std::vector<bool> values;
          values.reserve(static_cast<std::size_t>(list.b_size()));
          for (bool item : list.b()) values.push_back(item);
          info.attrs[name] = std::move(values);
        } else if (list.s_size() > 0) {
          std::vector<std::string> values;
          values.reserve(static_cast<std::size_t>(list.s_size()));
          for (const std::string& item : list.s()) values.push_back(item);
          info.attrs[name] = std::move(values);
        }
        break;
      }
      default:
        break;
    }
  }
}

}  // namespace

std::string NodeInfoAdapter::builderTensorName(const TensorRef& ref) {
  return ref.output_index == 0 ? ref.node : ref.canonicalName();
}

bool NodeInfoAdapter::adapt(const ResolvedTfGraph& resolved,
                            std::vector<annc::NodeInfo>& result,
                            std::string& error) const {
  result.clear();
  error.clear();
  for (const TfNode& node : resolved.graph.nodes) {
    if (!node.source) {
      error = "node '" + node.name + "' has no source NodeDef";
      return false;
    }
    const bool is_input =
        node.op == "Placeholder" || node.op == "PlaceholderV2";
    // Ops without an OpSpec row are no longer fatal here: the builder lowers
    // them to atir.opaque with a warning (see MLIRBuilder::buildOpaqueOp),
    // keeping the graph structurally convertible for the fusion stage.

    annc::NodeInfo info;
    info.name = node.name;
    // Aliases (AddV2, GatherV2, ...) are declared inside the OpSpec table
    // itself, so the builder resolves them directly; no normalization layer.
    info.op_type = node.op;
    info.isInputNode = is_input;
    info.tf_attrs["tf.name"] = node.name;
    info.tf_attrs["tf.op"] = node.op;
    copyComputeAttrs(*node.source, info);
    for (std::size_t i = 0; i < node.inputs.size(); ++i) {
      const TensorRef& input = node.inputs[i];
      if (!resolved.find(input)) {
        error = "node '" + node.name + "' references unresolved input '" +
                input.canonicalName() + "'";
        return false;
      }
      const std::string key = builderTensorName(input);
      info.inputs.push_back(key);
      info.tf_attrs["tf.input." + std::to_string(i)] = input.canonicalName();
    }
    for (const TensorRef& ref : node.outputs) {
      const TensorDescriptor* descriptor = resolved.find(ref);
      if (!descriptor) {
        error = "node '" + node.name + "' output '" + ref.canonicalName() +
                "' has no resolved descriptor";
        return false;
      }
      info.outputs.push_back(makeOutput(ref, *descriptor));
    }
    if (info.outputs.empty()) {
      error = "node '" + node.name + "' has no resolved outputs";
      return false;
    }
    if (node.source->op() == "Const" || node.source->op() == "HostConst") {
      const auto value = node.source->attr().find("value");
      if (value == node.source->attr().end()) {
        error = "Const node '" + node.name + "' has no TensorProto value";
        return false;
      }
      if (info.outputs.front().dtype == "string") {
        if (!TensorProtoDecoder::decodeStrings(value->second.tensor(),
                                               info.string_values, error)) {
          return false;
        }
      } else {
        std::vector<uint8_t> bytes;
        if (!TensorProtoDecoder::decode(value->second.tensor(),
                                        info.outputs.front().dtype, bytes,
                                        error)) {
          if (error.empty())
            error = "Const node '" + node.name + "' has no TensorProto value";
          return false;
        }
        info.raw_data = std::move(bytes);
      }
    }
    result.push_back(std::move(info));
  }

  if (!buildSwitchDiamonds(result, error)) return false;

  // Every graph result receives a private Identity boundary. This lets the
  // existing builder return multiple output slots without changing its
  // tensorValues_ keying, while tf.output_tensor preserves the real TF name.
  for (std::size_t i = 0; i < resolved.graph.outputs.size(); ++i) {
    const TensorRef& output = resolved.graph.outputs[i];
    const TensorDescriptor* descriptor = resolved.find(output);
    if (!descriptor) {
      error = "graph output '" + output.canonicalName() +
              "' has no resolved descriptor";
      return false;
    }
    annc::NodeInfo boundary;
    boundary.name = "__annc_output_" + std::to_string(i);
    boundary.op_type = "Identity";
    boundary.isOutputNode = true;
    boundary.inputs.push_back(builderTensorName(output));
    boundary.outputs.push_back(
        {boundary.name, descriptor->dtype, descriptor->shape,
         descriptor->rank_known});
    boundary.tf_attrs["tf.output_tensor"] = output.canonicalName();
    result.push_back(std::move(boundary));
  }
  return true;
}

}  // namespace annc::tf2atir
