#ifndef ANNC_ONNX2ATIR_ONNX_FRONTEND_H
#define ANNC_ONNX2ATIR_ONNX_FRONTEND_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx/onnx.pb.h"

namespace annc::onnx2atir {

// One graph node. ONNX tensor names are unique graph-wide, so a tensor is
// identified by its name alone (no TF-style "node:port" indexing).
struct OnnxNode {
  std::string name;    // node name; synthesized when the model omits it
  std::string op_type;
  std::string domain;  // "" and "ai.onnx" both denote the default domain
  const onnx::NodeProto* source = nullptr;
  // Omitted optional inputs (empty names in the model) are dropped; the
  // remaining entries keep their original order.
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  int index = 0;  // declaration order in the model file
};

struct LoadedOnnxModel {
  onnx::ModelProto model;
  // Version declared for the default ONNX domain (0 when absent).
  int64_t opset_version = 0;
  std::vector<std::pair<std::string, int64_t>> opset_imports;
};

class OnnxModelLoader {
 public:
  bool load(const std::string& model_path, LoadedOnnxModel& result,
            std::string& error) const;
};

struct OnnxGraph {
  std::vector<OnnxNode> nodes;  // topologically sorted
  std::vector<std::string> inputs;   // runtime inputs, initializers excluded
  std::vector<std::string> outputs;  // graph outputs, in declaration order
  // Initializers are constants, not runtime inputs: ONNX lets a name appear in
  // both `graph.input` and `graph.initializer`, and this frontend keeps the
  // initializer reading. Sparse initializers are tracked by name only so that
  // a model carrying them is reported rather than silently mis-imported.
  std::unordered_map<std::string, const onnx::TensorProto*> initializers;
  // The same initializers in `graph.initializer` declaration order. The map is
  // only a lookup: iterating it would emit the constants in hash order, which
  // makes the module (and everything downstream of its op order) depend on the
  // standard library rather than on the model.
  std::vector<std::string> initializer_names;
  std::vector<std::string> sparse_initializers;
  std::unordered_map<std::string, const onnx::ValueInfoProto*> graph_input_info;
  std::unordered_map<std::string, const onnx::ValueInfoProto*> declared_types;

  const onnx::TensorProto* findInitializer(const std::string& name) const;
};

class OnnxGraphParser {
 public:
  // Returns false only for conditions that make the whole model unusable
  // (control flow, local functions, cyclic graph). Per-node problems are left
  // to the resolver/adapter so they can be reported as explicit fallbacks.
  bool parse(const onnx::GraphProto& graph, OnnxGraph& result,
             std::string& error) const;
};

}  // namespace annc::onnx2atir

#endif  // ANNC_ONNX2ATIR_ONNX_FRONTEND_H
