#include "onnx_tensor_resolver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "onnx_tensor_decoder.h"

namespace annc::onnx2atir {
namespace {

using Dims = std::vector<int64_t>;

bool isDynamic(int64_t dim) { return dim == kDynamicDim; }

const onnx::AttributeProto* findAttr(const onnx::NodeProto& node,
                                     const std::string& name) {
  for (const onnx::AttributeProto& attr : node.attribute())
    if (attr.name() == name) return &attr;
  return nullptr;
}

bool getAttrInts(const onnx::NodeProto& node, const std::string& name,
                 std::vector<int64_t>& out) {
  const onnx::AttributeProto* attr = findAttr(node, name);
  if (!attr) return false;
  out.assign(attr->ints().begin(), attr->ints().end());
  return true;
}

bool getAttrInt(const onnx::NodeProto& node, const std::string& name,
                int64_t& out) {
  const onnx::AttributeProto* attr = findAttr(node, name);
  if (!attr) return false;
  out = attr->i();
  return true;
}

// numpy-style broadcast of two shapes. Returns false when the ranks are
// incompatible (a real error) or when either rank is unknown (not an error:
// the result simply stays rank-unknown).
bool broadcastShape(const Dims& a, const Dims& b, Dims& out, bool& rankKnown) {
  const size_t rank = std::max(a.size(), b.size());
  out.assign(rank, 1);
  rankKnown = true;
  for (size_t i = 0; i < rank; ++i) {
    const int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
    const int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
    int64_t result;
    if (da == 1) {
      result = db;
    } else if (db == 1) {
      result = da;
    } else if (isDynamic(da) || isDynamic(db)) {
      result = kDynamicDim;
    } else if (da == db) {
      result = da;
    } else {
      return false;
    }
    out[rank - 1 - i] = result;
  }
  return true;
}

bool isUnarySameShape(const std::string& op) {
  return op == "Relu" || op == "Sigmoid" || op == "LeakyRelu" ||
         op == "Reciprocal" || op == "Sqrt" || op == "Tanh" ||
         op == "Identity" || op == "Abs" || op == "Not" || op == "Softmax";
}

bool isBinaryElementwise(const std::string& op) {
  return op == "Add" || op == "Sub" || op == "Mul" || op == "Div" ||
         op == "Mod" || op == "Pow" || op == "Sum" || op == "Min" ||
         op == "Max" || op == "And" || op == "Or";
}

bool isComparison(const std::string& op) {
  return op == "Equal" || op == "Greater" || op == "Less";
}

bool isReduce(const std::string& op) {
  return op == "ReduceSum" || op == "ReduceMean" || op == "ReduceProd" ||
         op == "ReduceMax";
}

// Descriptor built from a declared ONNX TypeProto. Returns false when the
// declaration is not a tensor type (sequence/map/optional are out of scope).
bool descriptorFromType(const onnx::TypeProto& type, TensorDescriptor& out) {
  if (!type.has_tensor_type()) return false;
  const onnx::TypeProto::Tensor& tensor = type.tensor_type();
  if (!onnxDtypeName(tensor.elem_type(), out.dtype)) {
    out.dtype.clear();
    out.dtype_unsupported = true;
  }
  if (!tensor.has_shape()) {
    out.rank_known = false;
    return true;
  }
  out.rank_known = true;
  for (const onnx::TensorShapeProto::Dimension& dim : tensor.shape().dim()) {
    if (dim.has_dim_value()) {
      out.shape.push_back(dim.dim_value());
    } else if (dim.has_dim_param()) {
      out.shape.push_back(kDynamicDim);
      ++out.sym_dims;
    } else {
      out.shape.push_back(kDynamicDim);
      ++out.unknown_dims;
    }
  }
  return true;
}

// Reads the raw int payload of a constant tensor. int32/int64 are the only
// dtypes axes/shape/perm operands use in practice.
bool readConstInts(const onnx::TensorProto& tensor, std::vector<int64_t>& out) {
  const int64_t count = tensorElementCount(tensor);
  if (count < 0) return false;
  out.clear();
  out.reserve(static_cast<size_t>(std::max<int64_t>(count, 0)));
  switch (tensor.data_type()) {
    case onnx::TensorProto::INT64:
      if (!tensor.raw_data().empty()) {
        if (tensor.raw_data().size() != static_cast<size_t>(count) * 8)
          return false;
        // raw_data() only guarantees byte alignment, so the payload is copied
        // out instead of being reinterpreted in place. ONNX raw data is
        // little-endian; the host assumption is checked in the decoder header.
        out.resize(static_cast<size_t>(count));
        std::memcpy(out.data(), tensor.raw_data().data(),
                    static_cast<size_t>(count) * sizeof(int64_t));
      } else {
        out.assign(tensor.int64_data().begin(), tensor.int64_data().end());
      }
      break;
    case onnx::TensorProto::INT32:
      if (!tensor.raw_data().empty()) {
        if (tensor.raw_data().size() != static_cast<size_t>(count) * 4)
          return false;
        const char* data = tensor.raw_data().data();
        for (int64_t i = 0; i < count; ++i) {
          int32_t value = 0;
          std::memcpy(&value, data + static_cast<size_t>(i) * sizeof(value),
                      sizeof(value));
          out.push_back(value);
        }
      } else {
        for (int32_t v : tensor.int32_data()) out.push_back(v);
      }
      break;
    default:
      return false;
  }
  return static_cast<int64_t>(out.size()) == count;
}

bool readConstFloat(const onnx::TensorProto& tensor, double& out) {
  if (tensorElementCount(tensor) != 1) return false;
  if (tensor.data_type() == onnx::TensorProto::FLOAT) {
    if (!tensor.raw_data().empty()) {
      if (tensor.raw_data().size() != 4) return false;
      float value = 0.0f;
      std::memcpy(&value, tensor.raw_data().data(), sizeof(value));
      out = value;
    } else if (tensor.float_data_size() == 1) {
      out = tensor.float_data(0);
    } else {
      return false;
    }
    return true;
  }
  if (tensor.data_type() == onnx::TensorProto::DOUBLE) {
    if (!tensor.raw_data().empty()) {
      if (tensor.raw_data().size() != 8) return false;
      double value = 0.0;
      std::memcpy(&value, tensor.raw_data().data(), sizeof(value));
      out = value;
    } else if (tensor.double_data_size() == 1) {
      out = tensor.double_data(0);
    } else {
      return false;
    }
    return true;
  }
  return false;
}

}  // namespace

bool normalizeSliceAxis(int64_t& start, int64_t& end, int64_t step,
                        int64_t dim, int64_t& length) {
  if (step == 0) return false;
  if (isDynamic(dim)) {
    // Cannot resolve clamping or lengths without a static dimension; the
    // caller degrades the corresponding output dimension instead.
    length = kDynamicDim;
    return true;
  }
  const int64_t kMax = std::numeric_limits<int64_t>::max();
  const int64_t kMin = std::numeric_limits<int64_t>::min();
  if (step > 0) {
    int64_t s = start == kMin ? 0 : (start < 0 ? start + dim : start);
    int64_t e = end == kMax ? dim : (end == kMin ? 0 : (end < 0 ? end + dim : end));
    s = std::min(std::max<int64_t>(s, 0), dim);
    e = std::min(std::max<int64_t>(e, 0), dim);
    start = s;
    length = e > s ? (e - s + step - 1) / step : 0;
  } else {
    int64_t s = start == kMax ? dim - 1 : (start < 0 ? start + dim : start);
    int64_t e = end == kMin ? -1 : (end == kMax ? dim - 1 : (end < 0 ? end + dim : end));
    s = std::min(std::max<int64_t>(s, -1), dim - 1);
    e = std::min(std::max<int64_t>(e, -1), dim - 1);
    start = s;
    length = s > e ? (s - e - step - 1) / (-step) : 0;
  }
  return true;
}

bool OnnxTensorResolver::resolve(const OnnxGraph& graph,
                                 ResolvedOnnxGraph& result,
                                 std::string& error) const {
  error.clear();
  result = ResolvedOnnxGraph();
  result.graph = &graph;
  OnnxConstTable& constants = result.constants;

  // ---- seed: initializers and declared types -----------------------------
  for (const auto& entry : graph.initializers) {
    const onnx::TensorProto& tensor = *entry.second;
    TensorDescriptor descriptor;
    if (!onnxDtypeName(tensor.data_type(), descriptor.dtype)) {
      // Unsupported element type: recorded with an empty dtype and the
      // unsupported marker, so a later file declaration cannot mask it and
      // the adapter reports an explicit UNSUPPORTED_DTYPE fallback for its
      // consumers.
      descriptor.dtype.clear();
      descriptor.dtype_unsupported = true;
    }
    descriptor.rank_known = true;
    for (int64_t dim : tensor.dims()) descriptor.shape.push_back(dim);
    result.tensors[entry.first] = descriptor;
    if (descriptor.dtype != "string") {
      std::vector<int64_t> ints;
      if (tensor.data_type() == onnx::TensorProto::INT64 ||
          tensor.data_type() == onnx::TensorProto::INT32) {
        if (readConstInts(tensor, ints)) constants.add(entry.first, ints);
      }
      double scalar = 0.0;
      if (readConstFloat(tensor, scalar)) constants.addFloat(entry.first, scalar);
    } else {
      std::vector<std::string> strings;
      if (decodeTensorStrings(tensor, strings, error))
        (void)strings;  // value payload is consumed by the adapter
    }
  }
  for (const std::string& sparse : graph.sparse_initializers) {
    TensorDescriptor descriptor;
    descriptor.rank_known = false;
    result.tensors[sparse] = descriptor;
  }

  auto applyDeclared = [&](const std::string& name,
                           const onnx::ValueInfoProto* value) {
    if (!value) return;
    TensorDescriptor declared;
    if (!descriptorFromType(value->type(), declared)) return;
    declared.declared_in_file = true;
    auto it = result.tensors.find(name);
    if (it == result.tensors.end()) {
      result.tensors[name] = declared;
      return;
    }
    // Inference wins on shape (it can be more specific than a symbolic
    // declaration); the declaration only fills gaps. An element type the
    // resolver knows to be unrepresentable is never filled in: the
    // declaration would reinterpret the payload across dtypes.
    TensorDescriptor& existing = it->second;
    if (existing.dtype.empty() && !existing.dtype_unsupported)
      existing.dtype = declared.dtype;
    if (!existing.rank_known) {
      existing.shape = declared.shape;
      existing.rank_known = declared.rank_known;
      existing.sym_dims = declared.sym_dims;
      existing.unknown_dims = declared.unknown_dims;
    }
    existing.declared_in_file = true;
  };

  for (const std::string& name : graph.inputs) {
    auto it = graph.graph_input_info.find(name);
    applyDeclared(name, it == graph.graph_input_info.end() ? nullptr : it->second);
  }

  // ---- per-node inference -------------------------------------------------
  auto descOf = [&](const std::string& name) -> const TensorDescriptor* {
    auto it = result.tensors.find(name);
    return it == result.tensors.end() ? nullptr : &it->second;
  };

  for (const OnnxNode& node : graph.nodes) {
    const onnx::NodeProto& proto = *node.source;
    std::vector<const TensorDescriptor*> ins;
    ins.reserve(node.inputs.size());
    for (const std::string& name : node.inputs) ins.push_back(descOf(name));

    // Constant nodes contribute their payload to the constant table so that
    // axes/shape/perm consumers can read through them.
    if (node.op_type == "Constant" && proto.attribute_size() > 0) {
      const onnx::AttributeProto* value = findAttr(proto, "value");
      if (value && value->has_t()) {
        std::vector<int64_t> ints;
        if (readConstInts(value->t(), ints)) constants.add(node.outputs[0], ints);
        double scalar = 0.0;
        if (readConstFloat(value->t(), scalar))
          constants.addFloat(node.outputs[0], scalar);
      }
    }

    std::vector<TensorDescriptor> outs(node.outputs.size());
    const std::string& op = node.op_type;
    const TensorDescriptor* first = ins.empty() ? nullptr : ins[0];
    auto inheritDtype = [&](const TensorDescriptor* from) {
      for (TensorDescriptor& out : outs) {
        if (!from) continue;
        out.dtype = from->dtype;
        // An unrepresentable element type propagates to consumers: their
        // outputs cannot be typed either, and a file declaration must not
        // paper over that with a different dtype.
        out.dtype_unsupported = from->dtype_unsupported;
      }
    };
    auto requireRank = [&](const TensorDescriptor* from) -> bool {
      if (from && from->rank_known) return true;
      for (TensorDescriptor& out : outs) out.rank_known = false;
      return false;
    };

    if (op == "Constant") {
      const onnx::AttributeProto* value = findAttr(proto, "value");
      if (value && value->has_t()) {
        outs[0].rank_known = true;
        if (!onnxDtypeName(value->t().data_type(), outs[0].dtype)) {
          outs[0].dtype.clear();
          outs[0].dtype_unsupported = true;
        }
        for (int64_t dim : value->t().dims()) outs[0].shape.push_back(dim);
      } else {
        outs[0].rank_known = false;
      }
    } else if (isUnarySameShape(op) || op == "Cast") {
      const bool rankKnown = requireRank(first);
      inheritDtype(first);
      for (TensorDescriptor& out : outs) {
        out.rank_known = rankKnown;
        if (rankKnown) {
          out.shape = first->shape;
          // Shape-preserving ops carry the dimension provenance with them:
          // a symbolic or unknown dimension stays one.
          out.sym_dims = first->sym_dims;
          out.unknown_dims = first->unknown_dims;
        }
      }
      if (op == "Cast") {
        int64_t to = 0;
        if (getAttrInt(proto, "to", to)) {
          if (!onnxDtypeName(static_cast<int>(to), outs[0].dtype)) {
            outs[0].dtype.clear();
            outs[0].dtype_unsupported = true;
          }
        } else {
          outs[0].dtype.clear();
        }
      }
      if (op == "Not") outs[0].dtype = "bool";
    } else if (isBinaryElementwise(op) || isComparison(op) || op == "Where") {
      Dims shape;
      bool rankKnown = true;
      for (size_t i = 0; i < ins.size(); ++i) {
        if (!ins[i] || !ins[i]->rank_known) {
          rankKnown = false;
          break;
        }
        if (i == 0) {
          shape = ins[0]->shape;
          continue;
        }
        // `broadcastShape` assigns its result before reading the left operand,
        // so the accumulator must not be passed as both operands.
        Dims merged;
        if (!broadcastShape(shape, ins[i]->shape, merged, rankKnown)) {
          rankKnown = false;
          break;
        }
        shape = std::move(merged);
      }
      for (TensorDescriptor& out : outs) {
        out.rank_known = rankKnown;
        out.shape = rankKnown ? shape : Dims{};
      }
      if (isComparison(op) || op == "Not") {
        outs[0].dtype = "bool";
      } else if (op == "Where") {
        // Where(condition, x, y) yields x's element type; the condition is a
        // separate operand and must not be mistaken for the value type.
        const TensorDescriptor* value =
            ins.size() >= 2 ? ins[1] : (ins.size() >= 3 ? ins[2] : nullptr);
        if (value == nullptr && ins.size() >= 3) value = ins[2];
        inheritDtype(value != nullptr ? value : first);
      } else if (op == "And" || op == "Or") {
        // Operands are bool; ATIR carries the result as bool as well.
        outs[0].dtype = first ? first->dtype : "bool";
      } else {
        inheritDtype(first);
      }
    } else if (op == "MatMul") {
      if (ins.size() >= 2 && ins[0] && ins[1] && ins[0]->rank_known &&
          ins[1]->rank_known) {
        const Dims& a = ins[0]->shape;
        const Dims& b = ins[1]->shape;
        Dims out_shape;
        bool ok = true;
        if (a.size() < 2 || b.size() < 2) {
          ok = false;  // rank-1 forms are not exercised by the target models
        } else {
          const int64_t m = a[a.size() - 2], k1 = a[a.size() - 1];
          const int64_t k2 = b[b.size() - 2], n = b[b.size() - 1];
          if (!isDynamic(k1) && !isDynamic(k2) && k1 != k2) ok = false;
          Dims a_batch(a.begin(), a.end() - 2);
          Dims b_batch(b.begin(), b.end() - 2);
          Dims batch;
          bool batchKnown = true;
          if (!broadcastShape(a_batch, b_batch, batch, batchKnown)) ok = false;
          if (ok) {
            out_shape = batch;
            out_shape.push_back(m);
            out_shape.push_back(n);
          }
        }
        outs[0].rank_known = ok;
        if (ok) outs[0].shape = out_shape;
      } else {
        outs[0].rank_known = false;
      }
      inheritDtype(first);
    } else if (op == "Gemm") {
      // A plugin EP is handed ORT's *optimized* graph, where the optimizer has
      // already rewritten MatMul + Add(bias) into Gemm; the frontend imports
      // that node as atir.opaque (no ATIR form yet), so without a shape rule
      // here its output stays rank-unknown and every consumer fails its rank
      // checks -- measured on model4: 19 of 27 LayerNorm chains sit behind a
      // Gemm projection and were lost this way.  Only the shape is resolved;
      // the node's execution stays with the host.
      int64_t trans_a = 0;
      int64_t trans_b = 0;
      getAttrInt(proto, "transA", trans_a);
      getAttrInt(proto, "transB", trans_b);
      if (ins.size() >= 2 && ins[0] && ins[1] && ins[0]->rank_known &&
          ins[1]->rank_known) {
        const Dims& a = ins[0]->shape;
        const Dims& b = ins[1]->shape;
        Dims out_shape;
        bool ok = true;
        if (a.size() != 2 || b.size() != 2) {
          ok = false;  // ONNX defines Gemm on rank-2 operands only
        } else {
          const int64_t m = trans_a ? a[1] : a[0];
          const int64_t k1 = trans_a ? a[0] : a[1];
          const int64_t k2 = trans_b ? b[1] : b[0];
          const int64_t n = trans_b ? b[0] : b[1];
          if (!isDynamic(k1) && !isDynamic(k2) && k1 != k2) ok = false;
          if (ok) out_shape = {m, n};
        }
        outs[0].rank_known = ok;
        if (ok) outs[0].shape = out_shape;
      } else {
        outs[0].rank_known = false;
      }
      // C broadcasts into the [M, N] result, so it never changes the shape.
      inheritDtype(first);
    } else if (op == "Reshape") {
      const TensorDescriptor* shape_desc = ins.size() >= 2 ? ins[1] : nullptr;
      int64_t allowzero = 0;
      if (!getAttrInt(proto, "allowzero", allowzero)) allowzero = 0;
      std::vector<int64_t> target;
      if (node.inputs.size() >= 2 &&
          constants.getInts(node.inputs[1], target) && first &&
          first->rank_known) {
        Dims out_shape(target.begin(), target.end());
        int64_t inferred = -1;
        int64_t known_product = 1;
        for (size_t i = 0; i < out_shape.size(); ++i) {
          if (out_shape[i] == 0 && !allowzero) {
            out_shape[i] = i < first->shape.size() ? first->shape[i] : kDynamicDim;
          } else if (out_shape[i] == -1) {
            inferred = static_cast<int64_t>(i);
          } else if (isDynamic(out_shape[i])) {
            known_product = -1;
          }
        }
        if (known_product != -1) {
          for (size_t i = 0; i < out_shape.size(); ++i) {
            if (static_cast<int64_t>(i) == inferred) continue;
            if (isDynamic(out_shape[i])) {
              known_product = -1;
              break;
            }
            known_product *= out_shape[i];
          }
        }
        int64_t input_product = 1;
        for (int64_t d : first->shape) {
          if (isDynamic(d)) {
            input_product = -1;
            break;
          }
          input_product *= d;
        }
        if (inferred >= 0 && known_product > 0 && input_product >= 0)
          out_shape[inferred] = input_product / known_product;
        outs[0].rank_known = true;
        outs[0].shape = out_shape;
      } else {
        // Shape operand is not a known constant: rank comes from the target
        // tensor's length when available, dimensions stay dynamic.
        if (shape_desc && shape_desc->rank_known &&
            shape_desc->shape.size() == 1 && !isDynamic(shape_desc->shape[0])) {
          outs[0].rank_known = true;
          outs[0].shape.assign(static_cast<size_t>(shape_desc->shape[0]),
                               kDynamicDim);
        } else {
          outs[0].rank_known = false;
        }
      }
      inheritDtype(first);
    } else if (op == "Transpose") {
      std::vector<int64_t> perm;
      if (!getAttrInts(proto, "perm", perm) && first && first->rank_known) {
        for (int64_t i = static_cast<int64_t>(first->shape.size()) - 1; i >= 0; --i)
          perm.push_back(i);
      }
      if (first && first->rank_known && perm.size() == first->shape.size()) {
        outs[0].rank_known = true;
        for (int64_t p : perm) {
          if (p < 0 || p >= static_cast<int64_t>(first->shape.size())) {
            outs[0].rank_known = false;
            break;
          }
          outs[0].shape.push_back(first->shape[static_cast<size_t>(p)]);
        }
      } else {
        outs[0].rank_known = false;
      }
      inheritDtype(first);
    } else if (op == "Concat") {
      int64_t axis = 0;
      getAttrInt(proto, "axis", axis);
      bool ok = first && first->rank_known;
      if (ok) {
        const size_t rank = first->shape.size();
        int64_t normalized = axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
        if (normalized < 0 || normalized >= static_cast<int64_t>(rank)) {
          ok = false;
        } else {
          Dims out_shape = first->shape;
          int64_t total = 0;
          for (const TensorDescriptor* in : ins) {
            if (!in || !in->rank_known || in->shape.size() != rank) {
              ok = false;
              break;
            }
            const int64_t dim = in->shape[static_cast<size_t>(normalized)];
            if (isDynamic(dim)) {
              total = kDynamicDim;
            } else if (total != kDynamicDim) {
              total += dim;
            }
          }
          if (ok) {
            out_shape[static_cast<size_t>(normalized)] = total;
            outs[0].rank_known = true;
            outs[0].shape = out_shape;
          }
        }
      }
      if (!ok) outs[0].rank_known = false;
      inheritDtype(first);
    } else if (op == "Gather") {
      int64_t axis = 0;
      getAttrInt(proto, "axis", axis);
      const TensorDescriptor* indices = ins.size() >= 2 ? ins[1] : nullptr;
      if (first && first->rank_known && indices && indices->rank_known) {
        int64_t normalized =
            axis < 0 ? axis + static_cast<int64_t>(first->shape.size()) : axis;
        if (normalized < 0 ||
            normalized >= static_cast<int64_t>(first->shape.size())) {
          outs[0].rank_known = false;
        } else {
          outs[0].rank_known = true;
          for (size_t i = 0; i < static_cast<size_t>(normalized); ++i)
            outs[0].shape.push_back(first->shape[i]);
          for (int64_t dim : indices->shape) outs[0].shape.push_back(dim);
          for (size_t i = static_cast<size_t>(normalized) + 1;
               i < first->shape.size(); ++i)
            outs[0].shape.push_back(first->shape[i]);
        }
      } else {
        outs[0].rank_known = false;
      }
      inheritDtype(first);
    } else if (op == "Shape") {
      outs[0].dtype = "int64";
      if (first && first->rank_known) {
        outs[0].rank_known = true;
        outs[0].shape.push_back(static_cast<int64_t>(first->shape.size()));
      } else {
        outs[0].rank_known = false;
      }
    } else if (op == "Unsqueeze") {
      std::vector<int64_t> axes;
      const bool haveAxes =
          node.inputs.size() >= 2 && constants.getInts(node.inputs[1], axes);
      if (haveAxes && first && first->rank_known) {
        const int64_t out_rank = static_cast<int64_t>(first->shape.size()) +
                                 static_cast<int64_t>(axes.size());
        std::vector<char> inserted(static_cast<size_t>(out_rank), 0);
        bool valid = true;
        for (int64_t& axis : axes) {
          if (axis < 0) axis += out_rank;
          if (axis < 0 || axis >= out_rank) {
            valid = false;
            break;
          }
          // ONNX requires the axes to be unique; a repeat would leave fewer
          // inserted positions than axes and read past the input shape below.
          if (inserted[static_cast<size_t>(axis)]) {
            valid = false;
            break;
          }
          inserted[static_cast<size_t>(axis)] = 1;
        }
        if (valid) {
          Dims out_shape;
          size_t src = 0;
          for (int64_t i = 0; i < out_rank; ++i) {
            if (inserted[static_cast<size_t>(i)])
              out_shape.push_back(1);
            else
              out_shape.push_back(first->shape[src++]);
          }
          outs[0].rank_known = true;
          outs[0].shape = out_shape;
        } else {
          outs[0].rank_known = false;
        }
      } else {
        outs[0].rank_known = false;
      }
      inheritDtype(first);
    } else if (op == "Squeeze") {
      std::vector<int64_t> axes;
      const bool haveAxes =
          node.inputs.size() >= 2 && constants.getInts(node.inputs[1], axes);
      if (first && first->rank_known) {
        if (haveAxes) {
          const int64_t rank = static_cast<int64_t>(first->shape.size());
          std::vector<char> dropped(static_cast<size_t>(rank), 0);
          bool valid = true;
          for (int64_t axis : axes) {
            if (axis < 0) axis += rank;
            // ONNX requires unique in-range axes: a repeat would drop an
            // extra dimension (and, at the last axis, erase past the end).
            if (axis < 0 || axis >= rank ||
                dropped[static_cast<size_t>(axis)]) {
              valid = false;
              break;
            }
            dropped[static_cast<size_t>(axis)] = 1;
          }
          if (valid) {
            Dims out_shape;
            for (size_t i = 0; i < first->shape.size(); ++i)
              if (!dropped[i]) out_shape.push_back(first->shape[i]);
            outs[0].rank_known = true;
            outs[0].shape = out_shape;
          } else {
            outs[0].rank_known = false;
          }
        } else {
          Dims out_shape;
          for (int64_t dim : first->shape)
            if (dim != 1) out_shape.push_back(dim);
          outs[0].rank_known = true;
          outs[0].shape = out_shape;
        }
      } else {
        outs[0].rank_known = false;
      }
      inheritDtype(first);
    } else if (op == "Slice") {
      Dims starts, ends, axes, steps;
      const bool haveStarts =
          node.inputs.size() >= 2 && constants.getInts(node.inputs[1], starts);
      const bool haveEnds =
          node.inputs.size() >= 3 && constants.getInts(node.inputs[2], ends);
      if (node.inputs.size() >= 4) constants.getInts(node.inputs[3], axes);
      if (node.inputs.size() >= 5) constants.getInts(node.inputs[4], steps);
      if (first && first->rank_known && haveStarts && haveEnds &&
          !starts.empty() && starts.size() == ends.size()) {
        Dims out_shape = first->shape;
        const int64_t rank = static_cast<int64_t>(out_shape.size());
        if (axes.empty())
          for (int64_t i = 0; i < static_cast<int64_t>(starts.size()); ++i)
            axes.push_back(i);
        if (steps.empty()) steps.assign(starts.size(), 1);
        bool ok = axes.size() == starts.size() && steps.size() == starts.size();
        for (size_t i = 0; ok && i < axes.size(); ++i) {
          int64_t axis = axes[i] < 0 ? axes[i] + rank : axes[i];
          if (axis < 0 || axis >= rank) {
            ok = false;
            break;
          }
          int64_t length = 0;
          if (!normalizeSliceAxis(starts[i], ends[i], steps[i],
                                  out_shape[static_cast<size_t>(axis)], length)) {
            ok = false;
            break;
          }
          out_shape[static_cast<size_t>(axis)] = length;
        }
        if (ok) {
          outs[0].rank_known = true;
          outs[0].shape = out_shape;
        }
      }
      inheritDtype(first);
    } else if (op == "Tile") {
      std::vector<int64_t> repeats;
      if (first && first->rank_known && node.inputs.size() >= 2 &&
          constants.getInts(node.inputs[1], repeats) &&
          repeats.size() == first->shape.size()) {
        outs[0].rank_known = true;
        for (size_t i = 0; i < repeats.size(); ++i) {
          const int64_t dim = first->shape[i];
          outs[0].shape.push_back(isDynamic(dim) ? kDynamicDim : dim * repeats[i]);
        }
      } else {
        outs[0].rank_known = false;
      }
      inheritDtype(first);
    } else if (op == "Expand") {
      const TensorDescriptor* target_desc = ins.size() >= 2 ? ins[1] : nullptr;
      std::vector<int64_t> target;
      if (node.inputs.size() >= 2 &&
          constants.getInts(node.inputs[1], target) && first &&
          first->rank_known) {
        Dims target_shape(target.begin(), target.end());
        Dims out_shape;
        bool rankKnown = true;
        if (broadcastShape(first->shape, target_shape, out_shape, rankKnown)) {
          outs[0].rank_known = true;
          outs[0].shape = out_shape;
        }
      } else if (first && first->rank_known && target_desc &&
                 target_desc->rank_known && target_desc->shape.size() == 1 &&
                 !isDynamic(target_desc->shape[0])) {
        // The shape operand is a computed 1-D tensor of statically known
        // length. Its values are unknown, but broadcasting a rank-A tensor
        // against a rank-B shape yields rank max(A, B) -- enough to keep the
        // rank (and therefore every downstream rank) instead of collapsing to
        // "unknown" and cascading through the rest of the graph.
        const int64_t target_rank = target_desc->shape[0];
        const int64_t out_rank =
            std::max(static_cast<int64_t>(first->shape.size()), target_rank);
        outs[0].rank_known = true;
        outs[0].shape.assign(static_cast<size_t>(out_rank), kDynamicDim);
      }
      inheritDtype(first);
    } else if (op == "GlobalAveragePool") {
      Dims out_shape;
      bool rankKnown = first && first->rank_known && first->shape.size() >= 2;
      if (rankKnown) {
        out_shape.push_back(first->shape[0]);
        out_shape.push_back(first->shape[1]);
        for (size_t i = 2; i < first->shape.size(); ++i) out_shape.push_back(1);
      }
      outs[0].rank_known = rankKnown;
      outs[0].shape = out_shape;
      inheritDtype(first);
    } else if (isReduce(op)) {
      std::vector<int64_t> axes;
      bool haveAxes = getAttrInts(proto, "axes", axes);
      if (op == "ReduceSum" && node.inputs.size() >= 2)
        haveAxes = constants.getInts(node.inputs[1], axes);
      int64_t keepdims = 1;
      getAttrInt(proto, "keepdims", keepdims);
      if (first && first->rank_known) {
        const int64_t rank = static_cast<int64_t>(first->shape.size());
        if (!haveAxes || axes.empty()) {
          for (int64_t i = 0; i < rank; ++i) axes.push_back(i);
        }
        std::vector<char> reduced(static_cast<size_t>(rank), 0);
        bool ok = true;
        for (int64_t& axis : axes) {
          if (axis < 0) axis += rank;
          if (axis < 0 || axis >= rank) {
            ok = false;
            break;
          }
          reduced[static_cast<size_t>(axis)] = 1;
        }
        if (ok) {
          outs[0].rank_known = true;
          for (int64_t i = 0; i < rank; ++i) {
            if (reduced[static_cast<size_t>(i)]) {
              if (keepdims) outs[0].shape.push_back(1);
            } else {
              outs[0].shape.push_back(first->shape[static_cast<size_t>(i)]);
            }
          }
        } else {
          outs[0].rank_known = false;
        }
      } else {
        outs[0].rank_known = false;
      }
      inheritDtype(first);
    } else {
      // Unknown op: no shape/type reasoning is available. The adapter reports
      // it as an explicit fallback; consumers see a rank-unknown tensor.
      for (TensorDescriptor& out : outs) out.rank_known = false;
    }

    for (size_t i = 0; i < node.outputs.size(); ++i) {
      if (i < outs.size()) result.tensors[node.outputs[i]] = outs[i];
      result.producer_of[node.outputs[i]] = &node;
    }
  }

  // ---- declared types for outputs and remaining gaps ----------------------
  for (const std::string& name : graph.outputs) {
    auto it = graph.declared_types.find(name);
    applyDeclared(name, it == graph.declared_types.end() ? nullptr : it->second);
  }
  for (const auto& entry : graph.declared_types) applyDeclared(entry.first, entry.second);

  // ---- four-state tally over node-produced tensors ------------------------
  for (const OnnxNode& node : graph.nodes) {
    for (const std::string& name : node.outputs) {
      const TensorDescriptor* descriptor = result.find(name);
      if (descriptor == nullptr) {
        result.unresolved.push_back(name);
        continue;
      }
      ++result.produced.tensors;
      if (!descriptor->dtype.empty()) ++result.produced.with_type;
      if (!descriptor->rank_known) {
        ++result.produced.no_shape;
      }
      result.produced.sym_dims += descriptor->sym_dims;
      result.produced.unknown_dims += descriptor->unknown_dims;
      for (int64_t dim : descriptor->shape)
        if (isDynamic(dim)) ++result.produced.dynamic_dims;
      if (descriptor->declared_in_file) {
        ++result.produced_declared_in_file;
      } else {
        ++result.produced_inferred_only;
      }
    }
  }
  return true;
}

}  // namespace annc::onnx2atir
