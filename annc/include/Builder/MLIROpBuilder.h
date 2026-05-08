#ifndef ANNC_OP_BUILDER_H
#define ANNC_OP_BUILDER_H
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>

#include "Dialect/Atir/AtirOps.h"
#include "Helper.h"

using namespace mlir;
namespace annc {

// 输出张量信息
struct OutputInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
};

// 节点信息
struct NodeInfo {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<OutputInfo> outputs;
    std::string raw_data;
    using TfAttrValue =
        std::variant<int64_t, double, bool, std::string, std::vector<int64_t>,
                     std::vector<double>, std::vector<bool>,
                     std::vector<std::string>>;
    std::unordered_map<std::string, TfAttrValue> attrs;
    bool has_numBuckets = false;
    int64_t numBuckets = 0;
    bool isInputNode = false;
    bool isOutputNode = false;
};

class MLIRBuilder {
 public:
  MLIRBuilder(ModuleOp& module)
      : module_(module), builder_(module.getContext()) {}
  virtual ~MLIRBuilder() = default;

  void buildFromNodes(const std::vector<NodeInfo>& nodes);

  // Op handlers will be added in subsequent PRs
  void createUnsupportedNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins);
  void createCustomizeNode(const NodeInfo& node, ArrayRef<Type> outs, ArrayRef<Value> ins);

 private:
  ModuleOp module_;
  OpBuilder builder_;

  func::FuncOp mainFunc_;
  Value noneValue_;

  std::vector<Value> addGraphInputs(const std::vector<NodeInfo>& inputNodes);
  std::vector<Value> addGraphOutputs(const std::vector<NodeInfo>& outputNodes);

  mlir::Value addConstantNode(const NodeInfo& node);

  void addNode(const NodeInfo& node);

  std::unordered_map<std::string, Value> tensorValues_;
  atir::TensorType getTensorType(const std::string& name,
                                 const std::string& dtype,
                                 const std::vector<int64_t>& shape);

  template <typename T>
  std::vector<T> loadData(const std::string& dataStr) {
    std::vector<T> result;
    return result;
  }
};

}  // namespace annc
#endif  // ANNC_OP_BUILDER_H