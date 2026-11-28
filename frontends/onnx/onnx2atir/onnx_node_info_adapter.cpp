#include "onnx_node_info_adapter.h"

#include <algorithm>
#include <set>
#include <utility>

#include "onnx_mlir_builder.h"
#include "onnx_tensor_decoder.h"

namespace annc::onnx2atir {
namespace {

// ONNX attributes that are neither ATIR attributes nor needed for metadata
// interpretation still have to survive: they are copied verbatim into
// NodeInfo::attrs so the metadata keeps the source semantics.
void copyNodeAttrs(const onnx::NodeProto& proto, annc::NodeInfo& info,
                   bool& hasSparseValue, bool& hasGraphValue) {
  for (const onnx::AttributeProto& attr : proto.attribute()) {
    const std::string& name = attr.name();
    switch (attr.type()) {
      case onnx::AttributeProto::FLOAT:
        info.attrs[name] = static_cast<double>(attr.f());
        break;
      case onnx::AttributeProto::INT:
        info.attrs[name] = static_cast<int64_t>(attr.i());
        break;
      case onnx::AttributeProto::STRING:
        info.attrs[name] = attr.s();
        break;
      case onnx::AttributeProto::FLOATS: {
        std::vector<double> values(attr.floats().begin(), attr.floats().end());
        info.attrs[name] = std::move(values);
        break;
      }
      case onnx::AttributeProto::INTS: {
        std::vector<int64_t> values(attr.ints().begin(), attr.ints().end());
        info.attrs[name] = std::move(values);
        break;
      }
      case onnx::AttributeProto::STRINGS: {
        std::vector<std::string> values(attr.strings().begin(),
                                        attr.strings().end());
        info.attrs[name] = std::move(values);
        break;
      }
      case onnx::AttributeProto::TENSOR:
        // The tensor payload travels in NodeInfo::raw_data (Constant nodes)
        // or is intentionally not represented; NodeInfo::TfAttrValue has no
        // tensor shape and inventing one would lose it silently.
        break;
      case onnx::AttributeProto::SPARSE_TENSOR:
      case onnx::AttributeProto::SPARSE_TENSORS:
        hasSparseValue = true;
        break;
      case onnx::AttributeProto::GRAPH:
      case onnx::AttributeProto::GRAPHS:
        hasGraphValue = true;
        break;
      default:
        break;
    }
  }
}

// True when the tensor type of `name` is statically sized, i.e. every
// dimension has a concrete value. ATIR can only carry materialized constants
// of known shape.
bool hasStaticShape(const TensorDescriptor* descriptor) {
  if (descriptor == nullptr || !descriptor->rank_known) return false;
  for (int64_t dim : descriptor->shape)
    if (dim == kDynamicDim) return false;
  return true;
}

// Decodes an ONNX Constant node's payload into NodeInfo's constant fields, the
// same shape an initializer takes. Returns false when the node carries a form
// this frontend cannot hand to atir.constant (sparse payload, or one of the
// scalar/list `value_*` spellings instead of a TENSOR `value`).
bool decodeConstantNode(const onnx::NodeProto& proto, const std::string& dtype,
                        annc::NodeInfo& info, std::string& error) {
  const onnx::AttributeProto* value = nullptr;
  for (const onnx::AttributeProto& attr : proto.attribute()) {
    if (attr.name() == "value") {
      value = &attr;
      continue;
    }
    // Sparse payloads and the value_int/value_float/... spellings are a
    // different encoding of the same constant; refuse rather than mis-decode.
    if (attr.name() == "sparse_value") {
      error = "sparse_value";
      return false;
    }
    if (attr.name().rfind("value_", 0) == 0) {
      error = "constant node uses attribute '" + attr.name() + "'";
      return false;
    }
  }
  if (value == nullptr || !value->has_t()) {
    error = "constant node has no tensor value";
    return false;
  }
  const onnx::TensorProto& tensor = value->t();
  if (dtype == "string")
    return decodeTensorStrings(tensor, info.string_values, error);
  return decodeTensorBytes(tensor, dtype, info.raw_data, error);
}

// Converts the ONNX Slice operand form (starts/ends plus optional axes/steps,
// clamping and negative indices) into the full-rank stride-1 (begin, size)
// window atir.Slice wants, and stores it on the node as two private
// attributes. Returns false when the node has no faithful atir.Slice form:
// non-constant operands, steps other than 1, an unknown input rank, or a
// dynamic dimension that makes the window underivable.
bool normalizeSliceWindow(const OnnxNode& node, const OnnxConstTable& constants,
                          const ResolvedOnnxGraph& resolved,
                          annc::NodeInfo& info) {
  if (node.inputs.size() < 3) return false;
  std::vector<int64_t> starts, ends, axes, steps;
  if (!constants.getInts(node.inputs[1], starts)) return false;
  if (!constants.getInts(node.inputs[2], ends)) return false;
  if (node.inputs.size() >= 4 && !constants.getInts(node.inputs[3], axes))
    return false;
  if (node.inputs.size() >= 5 && !constants.getInts(node.inputs[4], steps))
    return false;
  if (starts.empty() || starts.size() != ends.size()) return false;

  const TensorDescriptor* data = resolved.find(node.inputs[0]);
  if (!hasStaticShape(data)) return false;
  const int64_t rank = static_cast<int64_t>(data->shape.size());

  if (axes.empty())
    for (size_t i = 0; i < starts.size(); ++i)
      axes.push_back(static_cast<int64_t>(i));
  if (steps.empty()) steps.assign(starts.size(), 1);
  if (axes.size() != starts.size() || steps.size() != starts.size()) return false;

  std::vector<int64_t> begin(static_cast<size_t>(rank), 0);
  std::vector<int64_t> size(data->shape.begin(), data->shape.end());
  for (size_t i = 0; i < axes.size(); ++i) {
    // atir.Slice has no stride.
    if (steps[i] != 1) return false;
    const int64_t axis = axes[i] < 0 ? axes[i] + rank : axes[i];
    if (axis < 0 || axis >= rank) return false;
    int64_t length = 0;
    if (!normalizeSliceAxis(starts[i], ends[i], steps[i],
                            data->shape[static_cast<size_t>(axis)], length))
      return false;
    if (length == kDynamicDim) return false;
    begin[static_cast<size_t>(axis)] = starts[i];
    size[static_cast<size_t>(axis)] = length;
  }

  info.attrs[kSliceBeginAttr.str()] = std::move(begin);
  info.attrs[kSliceSizeAttr.str()] = std::move(size);
  return true;
}

// Reads an integer attribute of the ONNX node; false when absent or not an int.
bool readIntAttr(const onnx::NodeProto& proto, const std::string& name,
                 int64_t& out) {
  for (const onnx::AttributeProto& attr : proto.attribute()) {
    if (attr.name() == name) {
      out = attr.i();
      return true;
    }
  }
  return false;
}

// Ops whose schema moved between opset 13 and later versions (attribute vs
// operand placement, axis defaults). Under --relaxed-opset a model declaring
// another version still maps these through the opset-13 table only if the op
// is absent from this list.
bool isOpsetSensitiveOp(const std::string& op_type) {
  static const std::set<std::string> kOps = {
      "Softmax", "Unsqueeze", "Squeeze", "ReduceSum",
      "Slice",   "Tile",      "Cast",    "Concat"};
  return kOps.count(op_type) != 0;
}

// True when any output tensor of the node is known to carry an element type
// ATIR cannot represent (the resolver marks those descriptors; see
// TensorDescriptor::dtype_unsupported). Such a node must not be imported as
// mapped: the builder would emit an unknown-typed op, or worse, reinterpret
// constant bytes across dtypes when a file declaration masks the type.
bool hasUnsupportedDtypeOutput(const OnnxNode& node,
                               const ResolvedOnnxGraph& resolved) {
  for (const std::string& name : node.outputs) {
    const TensorDescriptor* descriptor = resolved.find(name);
    if (descriptor != nullptr && descriptor->dtype_unsupported) return true;
  }
  return false;
}

// ONNX ops whose ATIR target has no integer path: the interpret implementations
// of atir.RealDiv/Sqrt/Reciprocal/Abs/Minimum/Maximum, of the comparison ops
// and of atir.FloorMod all read their operands through getFloatValues, and
// atir.Pow has no integer story either. Integer operands would only be correct
// while float32 happens to represent them exactly (|v| <= 2^24), so such nodes
// fall back until ATIR grows integer kernels.
//
// The list is deliberately curated rather than "every op the interpret path
// computes in float": integer Add/Sub/Mul/Tile/... are the shape-and-index
// arithmetic real models are built on, and their float32 bind is only lossy
// above 2^24. Both halves of that split are recorded in ADR-011, so the next
// reviewer does not have to re-derive which ops are covered by what.
bool isFloatOnlyOp(const std::string& op_type) {
  static const std::set<std::string> kOps = {
      "Div", "Pow", "Sqrt",  "Reciprocal", "Abs", "Min",
      "Max", "Mod", "Equal", "Greater",    "Less"};
  return kOps.count(op_type) != 0;
}

bool isIntegerDtypeName(const std::string& dtype) {
  static const std::set<std::string> kIntDtypes = {
      "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"};
  return kIntDtypes.count(dtype) != 0;
}

bool hasIntegerInputToFloatOnlyOp(const OnnxNode& node,
                                  const ResolvedOnnxGraph& resolved) {
  if (!isFloatOnlyOp(node.op_type)) return false;
  for (const std::string& name : node.inputs) {
    const TensorDescriptor* descriptor = resolved.find(name);
    if (descriptor != nullptr && isIntegerDtypeName(descriptor->dtype))
      return true;
  }
  return false;
}

}  // namespace

bool OnnxNodeInfoAdapter::adapt(const OnnxGraph& graph,
                                const ResolvedOnnxGraph& resolved,
                                const AdaptOptions& options,
                                AdaptedGraph& result, std::string& error) const {
  error.clear();
  result = AdaptedGraph();
  result.model_nodes = static_cast<int64_t>(graph.nodes.size());
  result.initializers = static_cast<int64_t>(graph.initializers.size());
  result.runtime_inputs = static_cast<int64_t>(graph.inputs.size());
  result.output_boundaries = static_cast<int64_t>(graph.outputs.size());

  const OnnxConstTable& constants = resolved.constants;
  auto fillOutput = [&](const std::string& tensor_name,
                        annc::OutputInfo& out) -> bool {
    const TensorDescriptor* descriptor = resolved.find(tensor_name);
    if (descriptor == nullptr) return false;
    out.name = tensor_name;
    // An unresolved element type becomes DType::Unknown ("unknown") rather
    // than an empty string, which the builder rejects outright. That keeps the
    // node importable-and-reported instead of aborting the whole model.
    out.dtype = descriptor->dtype.empty() ? "unknown" : descriptor->dtype;
    out.shape = descriptor->shape;
    out.rankKnown = descriptor->rank_known;
    return true;
  };

  // ---- runtime inputs ----------------------------------------------------
  for (const std::string& name : graph.inputs) {
    const TensorDescriptor* descriptor = resolved.find(name);
    if (descriptor == nullptr) {
      error = "graph input '" + name + "' has no resolved type";
      return false;
    }
    annc::NodeInfo info;
    info.name = name;
    info.op_type = "Input";
    info.isInputNode = true;
    annc::OutputInfo out;
    if (!fillOutput(name, out)) {
      error = "graph input '" + name + "' has no resolved descriptor";
      return false;
    }
    info.outputs.push_back(std::move(out));
    info.tf_attrs["tf.name"] = name;
    info.tf_attrs["tf.op"] = "Input";
    result.nodes.push_back(std::move(info));
  }

  // ---- initializers ------------------------------------------------------
  // ONNX lists initializers as constants, not as graph nodes; they become
  // Constant nodes here so the builder emits atir.constant for each. Sparse
  // initializers have no dense payload to hand over and are reported as
  // explicit fallbacks instead of being dropped. Iteration follows the model's
  // declaration order (graph.initializer), not the lookup map's hash order, so
  // the emitted module -- and everything downstream of its op order -- is a
  // function of the model alone.
  for (const std::string& name : graph.initializer_names) {
    const onnx::TensorProto& tensor = *graph.findInitializer(name);
    annc::NodeInfo info;
    info.name = name;
    info.op_type = "Constant";
    info.tf_attrs["tf.name"] = name;
    info.tf_attrs["tf.op"] = "Constant";
    annc::OutputInfo out;
    if (!fillOutput(name, out)) {
      error = "initializer '" + name + "' has no resolved descriptor";
      return false;
    }
    if (tensor.data_location() == onnx::TensorProto::EXTERNAL) {
      error = "initializer '" + name +
              "' stores its data in an external file, which is not supported";
      return false;
    }
    if (out.dtype == "unknown") {
      error = "initializer '" + name + "' has an unsupported element type";
      return false;
    }
    if (out.dtype == "string") {
      if (!decodeTensorStrings(tensor, info.string_values, error))
        return false;
    } else {
      if (!decodeTensorBytes(tensor, out.dtype, info.raw_data, error))
        return false;
    }
    info.outputs.push_back(std::move(out));
    result.nodes.push_back(std::move(info));
  }

  // A sparse initializer has no dense payload to hand to atir.constant, so it
  // becomes an atir.opaque rather than disappearing: the module keeps a node
  // that produces the tensor, the provenance keeps a reason, and the
  // conservation arithmetic stays honest (an entry that is counted as a
  // fallback must also exist in the module).
  for (const std::string& sparse : graph.sparse_initializers) {
    annc::NodeInfo info;
    info.name = sparse;
    info.op_type = "SparseInitializer";
    info.tf_attrs["tf.name"] = sparse;
    info.tf_attrs["tf.op"] = "SparseInitializer";
    info.attrs[kFallbackAttr.str()] = static_cast<int64_t>(1);
    info.attrs[kFallbackReasonAttr.str()] = std::string(kReasonUnsupportedDtype);
    const TensorDescriptor* descriptor = resolved.find(sparse);
    annc::OutputInfo out;
    out.name = sparse;
    out.dtype = "unknown";
    out.rankKnown = false;
    if (descriptor != nullptr) {
      out.dtype = descriptor->dtype.empty() ? "unknown" : descriptor->dtype;
      out.shape = descriptor->shape;
      out.rankKnown = descriptor->rank_known;
    }
    info.outputs.push_back(std::move(out));
    result.nodes.push_back(std::move(info));

    NodeProvenance entry;
    entry.node_name = sparse;
    entry.op_type = "SparseInitializer";
    entry.mapped = false;
    entry.reason = kReasonUnsupportedDtype;
    entry.sparse_initializer = true;
    result.provenance.push_back(std::move(entry));
  }

  // ---- graph nodes -------------------------------------------------------
  for (const OnnxNode& node : graph.nodes) {
    NodeProvenance provenance;
    provenance.node_name = node.name;
    provenance.op_type = node.op_type;
    provenance.domain = node.domain;

    annc::NodeInfo info;
    // Constants are addressed by their *tensor* name, not their node name:
    // buildConstantNode registers its result under NodeInfo::name, and the
    // shared engine's convention (inherited from TF, where a Const node and
    // its output carry the same name) is that those coincide. ONNX keeps them
    // apart -- Equal_9_const produces Equal_9:0 -- so the tensor name wins and
    // the ONNX node name stays in the provenance trail.
    info.name = node.op_type == "Constant" ? node.outputs.front() : node.name;
    info.op_type = node.op_type;
    info.tf_attrs["tf.name"] = info.name;
    info.tf_attrs["tf.op"] = node.op_type;
    if (node.op_type == "Constant" && info.name != node.name)
      info.tf_attrs["tf.onnx_node"] = node.name;
    if (!node.domain.empty() && node.domain != "ai.onnx")
      info.tf_attrs["tf.domain"] = node.domain;

    bool hasSparseValue = false;
    bool hasGraphValue = false;
    copyNodeAttrs(*node.source, info, hasSparseValue, hasGraphValue);
    for (std::size_t i = 0; i < node.inputs.size(); ++i)
      info.tf_attrs["tf.input." + std::to_string(i)] = node.inputs[i];

    // Inputs are copied verbatim; the builder resolves them by tensor name.
    // Squeeze is the one exception: atir has no Squeeze, so the op maps onto
    // atir.Reshape with the target shape rebuilt from the resolved output shape
    // -- the ONNX axes operand carries no information the output shape does not
    // already have, and the Reshape row accepts exactly one input. The copy
    // happens before the fallback decisions so that the arity check below
    // validates the normalized operand count.
    info.inputs = node.inputs;
    if (node.op_type == "Squeeze" && !info.inputs.empty())
      info.inputs.resize(1);

    // ---- per-node fallback decisions ------------------------------------
    // Everything that cannot be represented faithfully is pushed to
    // atir.opaque with a machine-readable reason rather than being dropped or
    // silently mistranslated (design decisions D4/D9).
    const OpSpec* spec = lookupOnnxOpSpec(node.op_type);
    std::string reason;
    if (options.relaxed_opset && options.opset_version != 13 &&
        isOpsetSensitiveOp(node.op_type)) {
      reason = kReasonOpsetMismatch;
    } else if (hasGraphValue) {
      // The parser rejects GRAPH attributes model-wide; reaching this point
      // would mean a subgraph survived, which must never be imported partially.
      reason = kReasonControlFlow;
    } else if (hasSparseValue) {
      reason = kReasonUnsupportedAttr;
    } else if (!node.domain.empty() && node.domain != "ai.onnx") {
      reason = kReasonNoSpec;
    } else if (spec == nullptr) {
      reason = kReasonNoSpec;
    } else if (hasUnsupportedDtypeOutput(node, resolved)) {
      // An output whose element type ATIR cannot carry (float8/int4 Cast
      // targets and constants, and everything downstream of them) is a
      // per-node fallback, never a mapped op with an unknown-typed result.
      reason = kReasonUnsupportedDtype;
    } else if (info.inputs.size() < spec->minInputs ||
               info.inputs.size() > spec->maxInputs ||
               (spec->numOutputs != kVariableOutputs &&
                static_cast<unsigned>(node.outputs.size()) !=
                    spec->numOutputs)) {
      // Operand/output counts outside the table row's contract are per-node
      // fallbacks -- the ONNX spec makes Min/Max/Sum variadic, and a lawful
      // 3-input Max must not reject the whole model when the row maps exactly
      // two operands.
      reason = kReasonUnsupportedAttr;
    } else if (hasIntegerInputToFloatOnlyOp(node, resolved)) {
      reason = kReasonUnsupportedDtype;
    } else if (node.op_type == "Slice") {
      // atir.Slice is stride-1 and needs full-rank static begin/size.
      if (!normalizeSliceWindow(node, constants, resolved, info)) {
        reason = kReasonUnsupportedAttr;
      }
    } else if (node.op_type == "Unsqueeze") {
      std::vector<int64_t> axes;
      if (node.inputs.size() >= 2) constants.getInts(node.inputs[1], axes);
      if (axes.size() != 1) reason = kReasonUnsupportedAttr;
    } else if (node.op_type == "Mod") {
      // ONNX `Mod` defaults to fmod=0, the integer modulus whose sign follows
      // the divisor -- that is exactly atir.FloorMod. fmod=1 is C's truncated
      // remainder (sign follows the dividend) and has no ATIR counterpart, so
      // it must fall back rather than be silently mis-signed.
      int64_t fmod = 0;
      if (readIntAttr(*node.source, "fmod", fmod) && fmod != 0)
        reason = kReasonUnsupportedAttr;
    } else if (node.op_type == "Softmax") {
      int64_t axis = -1;
      const bool axisDeclared = readIntAttr(*node.source, "axis", axis);
      const TensorDescriptor* input =
          node.inputs.empty() ? nullptr : resolved.find(node.inputs[0]);
      if (axisDeclared &&
          (input == nullptr || !input->rank_known || input->shape.empty())) {
        // An explicit axis can only be checked against the last dimension when
        // the rank is known. Without it, atir.Softmax's last-dim semantics
        // would be assumed rather than verified.
        reason = kReasonShapeUnknown;
      } else if (input != nullptr && input->rank_known && !input->shape.empty()) {
        int64_t normalized =
            axis < 0 ? axis + static_cast<int64_t>(input->shape.size()) : axis;
        if (normalized != static_cast<int64_t>(input->shape.size()) - 1)
          reason = kReasonUnsupportedAttr;
      }
    } else if (node.op_type == "ReduceSum") {
      // keepdims=1 preserves the rank whatever the axes are. keepdims=0 does
      // not, and a non-constant axes operand would make the baked output shape
      // a guess that the runtime could contradict.
      int64_t keepdims = 1;
      readIntAttr(*node.source, "keepdims", keepdims);
      std::vector<int64_t> axes;
      if (keepdims == 0 && node.inputs.size() >= 2 &&
          !constants.getInts(node.inputs[1], axes))
        reason = kReasonShapeUnknown;
    } else if (node.op_type == "Squeeze") {
      // transformSqueeze rebuilds the squeezed shape from the resolved output
      // shape, so an unknown output rank leaves nothing to reshape to.
      const TensorDescriptor* out =
          node.outputs.empty() ? nullptr : resolved.find(node.outputs[0]);
      if (out == nullptr || !out->rank_known) reason = kReasonShapeUnknown;
    } else if (node.op_type == "Transpose") {
      bool hasPerm = false;
      for (const onnx::AttributeProto& attr : node.source->attribute())
        if (attr.name() == "perm") hasPerm = true;
      const TensorDescriptor* out =
          node.outputs.empty() ? nullptr : resolved.find(node.outputs[0]);
      if (!hasPerm && (out == nullptr || !out->rank_known))
        reason = kReasonShapeUnknown;
    } else if (node.op_type == "GlobalAveragePool") {
      const TensorDescriptor* input =
          node.inputs.empty() ? nullptr : resolved.find(node.inputs[0]);
      if (input == nullptr || !input->rank_known) {
        reason = kReasonShapeUnknown;
      } else if (input->shape.size() < 3) {
        // ONNX averages over the spatial dims [2, rank); a rank-2 input has
        // none and keeps its [N, C] shape, while the ReduceMean mapping gets
        // an empty axes list, which ATIR reads as "reduce every dimension".
        // Fall back instead of emitting an op whose semantics contradict the
        // shape the resolver already published for it.
        reason = kReasonUnsupportedAttr;
      }
    } else if (node.op_type == "Constant") {
      const TensorDescriptor* out =
          node.outputs.empty() ? nullptr : resolved.find(node.outputs[0]);
      std::string decode_error;
      if (out == nullptr || out->dtype.empty() || out->dtype == "unknown") {
        reason = kReasonUnsupportedDtype;
      } else if (!decodeConstantNode(*node.source, out->dtype, info,
                                     decode_error)) {
        reason = kReasonUnsupportedAttr;
      } else if (out->rank_known) {
        // Placement of every element is fixed once the shape is known; a
        // Constant with a dynamic shape could not be materialized at all.
        for (int64_t dim : out->shape)
          if (dim == kDynamicDim) reason = kReasonShapeUnknown;
      }
    }

    for (const std::string& tensor_name : node.outputs) {
      annc::OutputInfo out;
      if (!fillOutput(tensor_name, out)) {
        // A produced tensor with no descriptor at all aborts the import: the
        // conservation trail would have a hole in it.
        result.provenance.push_back(provenance);
        error = "node '" + node.name + "' output '" + tensor_name +
                "' has no resolved type";
        return false;
      }
      info.outputs.push_back(std::move(out));
    }

    if (!reason.empty()) {
      info.attrs[kFallbackAttr.str()] = static_cast<int64_t>(1);
      info.attrs[kFallbackReasonAttr.str()] = reason;
      provenance.mapped = false;
      provenance.reason = reason;
    } else {
      // The spec is guaranteed non-null here: a missing row already produced
      // kReasonNoSpec above.
      provenance.mapped = true;
      provenance.atir_op = spec->atirOp.str();
    }
    result.provenance.push_back(std::move(provenance));
    result.nodes.push_back(std::move(info));
  }

  // ---- graph outputs -----------------------------------------------------
  // Reuses the TF frontend's convention: each graph result gets a private
  // Identity boundary named `__annc_output_<i>`, with the real name kept in
  // tf.output_tensor. This is what the fusion endpoint contract reads.
  for (std::size_t i = 0; i < graph.outputs.size(); ++i) {
    const std::string& name = graph.outputs[i];
    const TensorDescriptor* descriptor = resolved.find(name);
    if (descriptor == nullptr) {
      error = "graph output '" + name + "' has no resolved descriptor";
      return false;
    }
    annc::NodeInfo boundary;
    boundary.name = "__annc_output_" + std::to_string(i);
    boundary.op_type = "Identity";
    boundary.isOutputNode = true;
    boundary.inputs.push_back(name);
    annc::OutputInfo out;
    out.name = boundary.name;
    out.dtype = descriptor->dtype.empty() ? "unknown" : descriptor->dtype;
    out.shape = descriptor->shape;
    out.rankKnown = descriptor->rank_known;
    boundary.outputs.push_back(std::move(out));
    boundary.tf_attrs["tf.output_tensor"] = name;
    result.nodes.push_back(std::move(boundary));
  }
  return true;
}

}  // namespace annc::onnx2atir
