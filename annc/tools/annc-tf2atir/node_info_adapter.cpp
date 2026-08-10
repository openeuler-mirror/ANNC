#include "node_info_adapter.h"

#include <utility>

#include "tensor_proto_decoder.h"

namespace annc::tf2atir {
namespace {

annc::OutputInfo makeOutput(const TensorRef& ref,
                            const TensorDescriptor& descriptor) {
  return annc::OutputInfo{
      ref.output_index == 0 ? ref.node : ref.canonicalName(), descriptor.dtype,
      descriptor.shape};
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
        {boundary.name, descriptor->dtype, descriptor->shape});
    boundary.tf_attrs["tf.output_tensor"] = output.canonicalName();
    result.push_back(std::move(boundary));
  }
  return true;
}

}  // namespace annc::tf2atir
