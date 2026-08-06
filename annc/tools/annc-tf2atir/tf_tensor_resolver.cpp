#include "tf_tensor_resolver.h"

#include <optional>
#include <unordered_set>

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace annc::tf2atir {
namespace {

using tensorflow::AttrValue;
using tensorflow::DataType;
using tensorflow::NodeDef;
using tensorflow::TensorShapeProto;

std::optional<std::string> dtypeName(DataType dtype) {
  switch (dtype) {
    case tensorflow::DT_FLOAT:
      return "float32";
    case tensorflow::DT_DOUBLE:
      return "float64";
    case tensorflow::DT_HALF:
      return "float16";
    case tensorflow::DT_BFLOAT16:
      return "bfloat16";
    case tensorflow::DT_INT8:
      return "int8";
    case tensorflow::DT_INT16:
      return "int16";
    case tensorflow::DT_INT32:
      return "int32";
    case tensorflow::DT_INT64:
      return "int64";
    case tensorflow::DT_UINT8:
      return "uint8";
    case tensorflow::DT_UINT16:
      return "uint16";
    case tensorflow::DT_UINT32:
      return "uint32";
    case tensorflow::DT_UINT64:
      return "uint64";
    case tensorflow::DT_BOOL:
      return "bool";
    case tensorflow::DT_STRING:
      return "string";
    case tensorflow::DT_COMPLEX64:
      return "complex64";
    case tensorflow::DT_COMPLEX128:
      return "complex128";
    default:
      return std::nullopt;
  }
}

bool shapeFromProto(const TensorShapeProto& source, std::vector<int64_t>& shape,
                    std::string& error, const std::string& context) {
  if (source.unknown_rank()) {
    error = context + " has unknown rank, which ATIR cannot represent";
    return false;
  }
  shape.clear();
  shape.reserve(static_cast<std::size_t>(source.dim_size()));
  for (const auto& dim : source.dim()) shape.push_back(dim.size());
  return true;
}

std::optional<DataType> typeAttr(const NodeDef& node, const char* name) {
  const auto it = node.attr().find(name);
  if (it == node.attr().end() || it->second.value_case() != AttrValue::kType)
    return std::nullopt;
  return it->second.type();
}

std::optional<DataType> typeListAttr(const NodeDef& node, const char* name,
                                     int output_index) {
  const auto it = node.attr().find(name);
  if (it == node.attr().end() || it->second.value_case() != AttrValue::kList ||
      output_index < 0 || output_index >= it->second.list().type_size())
    return std::nullopt;
  return static_cast<DataType>(it->second.list().type(output_index));
}

bool isComparisonOp(const std::string& op) {
  static const std::unordered_set<std::string> comparisons = {
      "Equal",   "NotEqual",     "Less",          "LessEqual",
      "Greater", "GreaterEqual", "GreaterEqualV2"};
  return comparisons.count(op) != 0;
}

bool outputShapes(const NodeDef& node,
                  std::vector<const TensorShapeProto*>& out,
                  std::string& error) {
  out.clear();
  const auto it = node.attr().find("_output_shapes");
  if (it != node.attr().end() && it->second.value_case() == AttrValue::kList &&
      it->second.list().shape_size() > 0) {
    for (const auto& shape : it->second.list().shape()) out.push_back(&shape);
    return true;
  }
  if (node.op() == "Placeholder" || node.op() == "PlaceholderV2") {
    const auto shape_it = node.attr().find("shape");
    if (shape_it != node.attr().end() &&
        shape_it->second.value_case() == AttrValue::kShape) {
      out.push_back(&shape_it->second.shape());
      return true;
    }
  }
  error = "node '" + node.name() +
          "' has no _output_shapes annotation or declared placeholder shape";
  return false;
}

bool resolveDtype(const TfNode& node, int output_index,
                  const std::unordered_map<TensorRef, TensorDescriptor,
                                           TensorRefHash>& tensors,
                  std::string& dtype, std::string& error) {
  const NodeDef& source = *node.source;
  auto setType = [&](DataType type) {
    const std::optional<std::string> name = dtypeName(type);
    if (!name) {
      error = "unsupported TensorFlow dtype for node '" + node.name +
              "' output " + std::to_string(output_index) +
              ": ATIR cannot represent it";
      return false;
    }
    dtype = *name;
    return true;
  };
  auto inputDtype = [&](std::size_t input_index) {
    if (input_index >= node.inputs.size()) {
      error = "node '" + node.name + "' op '" + node.op +
              "' lacks required input " + std::to_string(input_index);
      return false;
    }
    const auto it = tensors.find(node.inputs[input_index]);
    if (it == tensors.end()) {
      error = "node '" + node.name + "' depends on unresolved input '" +
              node.inputs[input_index].canonicalName() + "'";
      return false;
    }
    dtype = it->second.dtype;
    return true;
  };

  if (source.op() == "Const") {
    const auto value = source.attr().find("value");
    if (value == source.attr().end() ||
        value->second.value_case() != AttrValue::kTensor) {
      error = "Const node '" + node.name + "' has no TensorProto value";
      return false;
    }
    return setType(value->second.tensor().dtype());
  }
  if (source.op() == "Placeholder" || source.op() == "PlaceholderV2") {
    const auto type = typeAttr(source, "dtype");
    if (type) return setType(*type);
    error = "Placeholder node '" + node.name + "' has no dtype attribute";
    return false;
  }
  if (source.op() == "VarHandleOp") {
    // ATIR represents an inference-only resource handle as the value of its
    // persistent VariableOp. ReadVariableOp then preserves that tensor value.
    const auto type = typeAttr(source, "dtype");
    if (type) return setType(*type);
    error = "VarHandleOp node '" + node.name + "' has no dtype attribute";
    return false;
  }
  if (isComparisonOp(source.op())) return setType(tensorflow::DT_BOOL);
  if (source.op() == "StringToHashBucketFast")
    return setType(tensorflow::DT_INT64);
  if (source.op() == "Where") {
    if (node.inputs.size() == 1) return setType(tensorflow::DT_INT64);
    if (node.inputs.size() == 3) return inputDtype(1);
    error = "Where node '" + node.name + "' must have one or three inputs";
    return false;
  }
  if (source.op() == "SparseReshape") return inputDtype(0);
  if (source.op() == "SparseFillEmptyRows") {
    if (output_index == 0 || output_index == 3)
      return setType(tensorflow::DT_INT64);
    if (output_index == 1) {
      const auto type = typeAttr(source, "T");
      if (type) return setType(*type);
    }
    if (output_index == 2) return setType(tensorflow::DT_BOOL);
    error = "SparseFillEmptyRows node '" + node.name +
            "' has invalid output index " + std::to_string(output_index);
    return false;
  }
  if (source.op() == "TopK" || source.op() == "TopKV2") {
    if (output_index == 0) {
      const auto type = typeAttr(source, "T");
      if (type) return setType(*type);
    } else if (output_index == 1) {
      return setType(
          typeAttr(source, "index_type").value_or(tensorflow::DT_INT32));
    }
    error = "TopK node '" + node.name + "' has no type rule for output " +
            std::to_string(output_index);
    return false;
  }
  for (const char* key : {"DstT", "out_type", "output_type"}) {
    if (const auto type = typeAttr(source, key)) return setType(*type);
  }
  if (const auto type = typeListAttr(source, "Tout", output_index))
    return setType(*type);
  for (const char* key : {"Tout", "T", "Tparams", "dtype"}) {
    if (const auto type = typeAttr(source, key)) return setType(*type);
  }
  error = "cannot determine TensorFlow dtype for node '" + node.name +
          "' output " + std::to_string(output_index) +
          ": no GraphDef fact or local op signature rule";
  return false;
}

}  // namespace

const TensorDescriptor* ResolvedTfGraph::find(const TensorRef& ref) const {
  const auto it = tensors.find(ref);
  return it == tensors.end() ? nullptr : &it->second;
}

bool TfTensorResolver::resolve(const TfGraph& graph, ResolvedTfGraph& result,
                               std::string& error,
                               ShapeOverridePolicy shape_policy) const {
  result = ResolvedTfGraph{};
  result.graph = graph;
  error.clear();
  for (TfNode& node : result.graph.nodes) {
    if (!node.source) {
      error = "parsed node '" + node.name + "' has no source NodeDef";
      return false;
    }
    if (node.source->op() == "Const") {
      const auto value = node.source->attr().find("value");
      if (value == node.source->attr().end() ||
          value->second.value_case() != AttrValue::kTensor) {
        error = "Const node '" + node.name + "' has no TensorProto value";
        return false;
      }
      TensorDescriptor descriptor;
      if (!shapeFromProto(value->second.tensor().tensor_shape(),
                          descriptor.shape, error,
                          "Const node '" + node.name + "'"))
        return false;
      if (!resolveDtype(node, 0, result.tensors, descriptor.dtype, error))
        return false;
      const TensorRef output{node.name, 0};
      result.tensors.emplace(output, std::move(descriptor));
      node.outputs = {output};
      continue;
    }

    std::vector<const TensorShapeProto*> shapes;
    if (!outputShapes(*node.source, shapes, error)) return false;
    node.outputs.clear();
    for (std::size_t index = 0; index < shapes.size(); ++index) {
      TensorDescriptor descriptor;
      const std::string context =
          "node '" + node.name + "' output " + std::to_string(index);
      if (!shapeFromProto(*shapes[index], descriptor.shape, error, context))
        return false;
      if (!resolveDtype(node, static_cast<int>(index), result.tensors,
                        descriptor.dtype, error))
        return false;
      const TensorRef output{node.name, static_cast<int>(index)};
      result.tensors.emplace(output, std::move(descriptor));
      node.outputs.push_back(output);
    }
  }

  if (shape_policy.batch_size > 0) {
    for (auto& entry : result.tensors) {
      std::vector<int64_t>& shape = entry.second.shape;
      if (!shape.empty() && shape.front() == -1)
        shape.front() = shape_policy.batch_size;
    }
  }
  return true;
}

}  // namespace annc::tf2atir
