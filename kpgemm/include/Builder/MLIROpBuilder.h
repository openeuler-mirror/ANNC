#ifndef KPGEMM_OP_BUILDER_H
#define KPGEMM_OP_BUILDER_H
#include <nlohmann/json.hpp>

#include "Dialect/Pimp/PimpOps.h"
#include "Helper.h"

using json = nlohmann::json;
using namespace mlir;
namespace kpgemm {

// struct NodeInfo{
//   std::string name;
//   std::string op_type;
//   std::vector<std::string> inputs;

//   std::string raw_data;

//   bool isInputNode  = false;
//   bool isOutputNode = false;

//   struct OutputInfo {
//     std::string name;
//     std::string dtype;
//     std::vector<int64_t> shape;
//     int id = 0;

//     std::string getName() const {
//       if (id == 0) return name;
//       return name + ":" + std::to_string(id);
//     }
//   };
//   std::vector<OutputInfo> outputs;
//   // std::vector<mlir::Attribute> attributes;
  
//   NodeInfo() = default;
//   void addOutput(int id, const std::string& name, const std::vector<int64_t>& s, const std::string& d) {
//     OutputInfo out;
//     out.shape = s;
//     out.dtype = d;
//     out.id = id;
//     out.name = name;
//     outputs.push_back(out);
//   }
//   // bool isInputNode() const {
//   //   return op_type == "Placeholder";
//   // }
// };

class MLIRBuilder {
 public:
  MLIRBuilder(ModuleOp& module)
      : module_(module), builder_(module.getContext()) {}
  virtual ~MLIRBuilder() = default;

  void jsonConvertor(const json& graph);

 private:
  ModuleOp module_;
  OpBuilder builder_;

  func::FuncOp mainFunc_;
  Value noneValue_;

  std::vector<Value> addGraphInput(const json& graph);
  std::vector<Value> addGraphOutput(const json& graph);

  mlir::Value addConstantNode(const json& node);

  void addNode(const json& node);

  void createMatMulOp(const json& node, ArrayRef<Type> outs, ArrayRef<Value> ins,
                      ArrayRef<NamedAttribute> attrs);
  void createAddOp(const json& node, ArrayRef<Type> outs, ArrayRef<Value> ins,
                   ArrayRef<NamedAttribute> attrs);

  std::unordered_map<std::string, Value> tensorValues_;
  pimp::TensorType getTensorType(const std::string& name,
                                 const std::string& dtype,
                                 const std::vector<int64_t>& shape);

  template <typename T>
  std::vector<T> loadData(const std::string& dataStr) {
    std::vector<T> result;
    return result;
  }
};

}  // namespace kpgemm
#endif  // KPGEMM_OP_BUILDER_H