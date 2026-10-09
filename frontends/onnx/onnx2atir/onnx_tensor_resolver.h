#ifndef ANNC_ONNX2ATIR_ONNX_TENSOR_RESOLVER_H
#define ANNC_ONNX2ATIR_ONNX_TENSOR_RESOLVER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_frontend.h"

namespace annc::onnx2atir {

// ONNX Slice parameter normalization (spec "Slice-13" clamping rules).
// `start`/`end` are rewritten to the resolved [start, start+length) window and
// `length` receives the number of elements the window covers. Returns false
// for step == 0. When `dim` is dynamic the window cannot be resolved and
// `length` is set to kDynamicDim.
bool normalizeSliceAxis(int64_t& start, int64_t& end, int64_t step, int64_t dim,
                        int64_t& length);

// kDynamic, matching ShapedType::kDynamic: a dimension whose concrete value is
// unknown. ONNX symbolic dimensions (`dim_param`) and dimensions that could not
// be inferred both land here -- ATIR has no symbolic-dimension representation,
// so "same dimension" equivalence constraints are lost at this boundary.
inline constexpr int64_t kDynamicDim = -1;

struct TensorDescriptor {
  std::string dtype;            // annc canonical name; empty when unknown
  // The element type is known to be unrepresentable in ATIR (float8/int4
  // variants, UNDEFINED). `dtype` stays empty, and a file declaration must
  // not fill it: that would reinterpret the raw bytes across dtypes. This is
  // distinct from a merely undeclared/underived dtype, which a declaration
  // may legitimately supply.
  bool dtype_unsupported = false;
  std::vector<int64_t> shape;   // concrete dims, kDynamicDim for the rest
  bool rank_known = false;
  // Four-state provenance for G2:
  int sym_dims = 0;          // dims that came from a `dim_param`
  int unknown_dims = 0;      // dims with neither dim_value nor dim_param
  bool declared_in_file = false;  // named in graph.input/output/value_info
};

// Integer payload of a constant tensor (initializer or Constant node), used to
// resolve axes/shape/perm operands without a full tensor runtime.
class OnnxConstTable {
 public:
  void add(const std::string& name, std::vector<int64_t> values) {
    ints_[name] = std::move(values);
  }
  // Scalar float payload, for the few transformers that need one (Reciprocal).
  void addFloat(const std::string& name, double value) { floats_[name] = value; }
  bool getInts(const std::string& name, std::vector<int64_t>& out) const {
    auto it = ints_.find(name);
    if (it == ints_.end()) return false;
    out = it->second;
    return true;
  }
  bool getFloat(const std::string& name, double& out) const {
    auto it = floats_.find(name);
    if (it == floats_.end()) return false;
    out = it->second;
    return true;
  }

 private:
  std::unordered_map<std::string, std::vector<int64_t>> ints_;
  std::unordered_map<std::string, double> floats_;
};

// Per-model four-state tally. Mirrors the `shape-inference.json` schema of the
// recorded shape-inference dumps so the two can be diffed directly.
struct FourStateStats {
  int64_t tensors = 0;
  int64_t with_type = 0;
  int64_t no_shape = 0;  // tensor with no shape at all => rank unknown
  int64_t sym_dims = 0;
  int64_t unknown_dims = 0;
  // Dimensions that end up dynamic in the ATIR, whatever their source (a
  // symbolic declaration, an unknown dimension, or an inference that could
  // only narrow the rank). This is the honest measure of the information loss
  // from ATIR having no symbolic-dimension representation; `sym_dims` above
  // only counts what the model file spelled out.
  int64_t dynamic_dims = 0;
};

struct ResolvedOnnxGraph {
  const OnnxGraph* graph = nullptr;
  // Constant payloads seen during resolution; the adapter consults this for
  // the few ops whose ONNX form splits a value across attribute and operand.
  // Owned by value: a resolver may be used for more than one graph, and a
  // second resolve() must not invalidate the first graph's constant table.
  OnnxConstTable constants;
  std::unordered_map<std::string, TensorDescriptor> tensors;
  std::unordered_map<std::string, const OnnxNode*> producer_of;
  // Stats over node-produced tensors only (graph inputs/initializers excluded),
  // matching the `produced_*` fields of the recorded census.
  FourStateStats produced;
  int64_t produced_declared_in_file = 0;
  int64_t produced_inferred_only = 0;
  // Node outputs that could not be typed at all; the adapter turns them into
  // explicit fallbacks rather than dropping them.
  std::vector<std::string> unresolved;

  const TensorDescriptor* find(const std::string& name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
  }
};

class OnnxTensorResolver {
 public:
  // Constant payloads discovered while resolving are owned by `result`; the
  // adapter reuses them for ops whose ATIR form takes an operand where ONNX
  // uses an attribute.
  bool resolve(const OnnxGraph& graph, ResolvedOnnxGraph& result,
               std::string& error) const;
};

}  // namespace annc::onnx2atir

#endif  // ANNC_ONNX2ATIR_ONNX_TENSOR_RESOLVER_H
