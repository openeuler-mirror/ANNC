#ifndef ANNC_TF2ATIR_TF_FRONTEND_H
#define ANNC_TF2ATIR_TF_FRONTEND_H

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace annc::tf2atir {

struct TensorRef {
  std::string node;
  int output_index = 0;

  std::string canonicalName() const;

  friend bool operator==(const TensorRef& lhs, const TensorRef& rhs) {
    return lhs.node == rhs.node && lhs.output_index == rhs.output_index;
  }
};

struct TensorRefHash {
  std::size_t operator()(const TensorRef& ref) const noexcept;
};

struct TfNode {
  std::string name;
  std::string op;
  const tensorflow::NodeDef* source = nullptr;
  std::vector<TensorRef> inputs;
  std::vector<std::string> control_inputs;
  // Parser records output slots referenced by graph edges. The resolver
  // replaces this with the complete output-slot list from the TF signature.
  std::vector<TensorRef> outputs;
};

struct TfGraph {
  std::vector<TfNode> nodes;
  std::vector<TensorRef> inputs;
  std::vector<TensorRef> outputs;

  const TfNode* findNode(const std::string& name) const;
};

struct LoadedTfModel {
  std::unique_ptr<tensorflow::GraphDef> owned_graph_def;
  std::unique_ptr<tensorflow::SavedModel> saved_model;
  const tensorflow::GraphDef* graph_def = nullptr;
  std::vector<std::string> output_names;
};

class TfModelLoader {
 public:
  bool load(const std::string& model_path,
            const std::vector<std::string>& explicit_output_names,
            LoadedTfModel& result, std::string& error) const;
};

class TfGraphParser {
 public:
  bool parse(const tensorflow::GraphDef& graph,
             const std::vector<std::string>& output_names, TfGraph& result,
             std::string& error) const;

  static bool parseTensorName(const std::string& value, TensorRef& result,
                              std::string& error);

 private:
  static bool parseDataInput(const std::string& value, TensorRef& result,
                             std::string& error);
};

}  // namespace annc::tf2atir

#endif  // ANNC_TF2ATIR_TF_FRONTEND_H
