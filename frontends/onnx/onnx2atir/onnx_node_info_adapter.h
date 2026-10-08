#ifndef ANNC_ONNX2ATIR_ONNX_NODE_INFO_ADAPTER_H
#define ANNC_ONNX2ATIR_ONNX_NODE_INFO_ADAPTER_H

#include <cstdint>
#include <string>
#include <vector>

#include "Builder/MLIROpBuilder.h"
#include "onnx_frontend.h"
#include "onnx_tensor_resolver.h"

namespace annc::onnx2atir {

// Machine-readable fallback reasons (design decision D9 of
// projects/annc/architecture/onnx-frontend-design.md).
inline constexpr const char* kReasonNoSpec = "NO_SPEC";
inline constexpr const char* kReasonUnsupportedDtype = "UNSUPPORTED_DTYPE";
inline constexpr const char* kReasonUnsupportedAttr = "UNSUPPORTED_ATTR";
inline constexpr const char* kReasonControlFlow = "CONTROL_FLOW";
inline constexpr const char* kReasonShapeUnknown = "SHAPE_UNKNOWN";
inline constexpr const char* kReasonOpsetMismatch = "OP_SET_MISMATCH";
inline constexpr const char* kReasonInitializerDynamic = "INITIALIZER_DYNAMIC";

// One row of the "no node is silently dropped" audit trail (goal G3).
struct NodeProvenance {
  std::string node_name;   // ONNX node name (or synthesized)
  std::string op_type;     // ONNX op type
  std::string domain;      // "" for the default domain
  bool mapped = false;     // false => landed in atir.opaque
  std::string reason;      // fallback reason code, empty when mapped
  std::string atir_op;     // ATIR target when mapped
  // True for the rows that stand for a sparse initializer rather than for a
  // graph node. They are part of the audit trail (they become atir.opaque too)
  // but they are not model nodes, so the conservation check counts them
  // separately instead of folding them into the node count.
  bool sparse_initializer = false;
};

struct AdaptedGraph {
  std::vector<annc::NodeInfo> nodes;
  std::vector<NodeProvenance> provenance;
  int64_t model_nodes = 0;
  int64_t runtime_inputs = 0;
  int64_t initializers = 0;
  int64_t output_boundaries = 0;
};

struct AdaptOptions {
  // Under --relaxed-opset the model's opset is not 13; ops whose schema moved
  // between opset 13 and the model's version are forced to the opaque fallback
  // instead of being mapped through a table that may not describe them.
  bool relaxed_opset = false;
  int64_t opset_version = 13;
};

class OnnxNodeInfoAdapter {
 public:
  // `result.nodes` mirrors the source graph one-for-one: every ONNX node,
  // every initializer and every graph output produces exactly one NodeInfo (or
  // an explicit fallback entry), so the conservation check has something to
  // balance against.
  bool adapt(const OnnxGraph& graph, const ResolvedOnnxGraph& resolved,
             const AdaptOptions& options, AdaptedGraph& result,
             std::string& error) const;
};

}  // namespace annc::onnx2atir

#endif  // ANNC_ONNX2ATIR_ONNX_NODE_INFO_ADAPTER_H
