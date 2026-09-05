#ifndef ANNC_OP_BUILDER_H
#define ANNC_OP_BUILDER_H
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Builder/DType.h"
#include "Builder/OpSpec.h"
#include "Dialect/Atir/AtirOps.h"
#include "Helper.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
namespace annc {

struct OutputInfo {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  bool rankKnown = true;
};

struct NodeInfo {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<OutputInfo> outputs;
  // Raw Const bytes in TensorProto tensor_content layout (little-endian).
  std::vector<uint8_t> raw_data;
  // Const(DT_STRING) payload: real string values (ComplexType<f32> carrier).
  std::vector<std::string> string_values;
  using TfAttrValue = std::variant<int64_t, double, bool, std::string,
                                   std::vector<int64_t>, std::vector<double>,
                                   std::vector<bool>, std::vector<std::string>>;
  // Typed TF attributes (transpose_a, keep_dims, axis, ...) driving ATIR op
  // semantics.
  std::unordered_map<std::string, TfAttrValue> attrs;
  // String metadata for GraphDef rewrite / converter recovery; kept separate
  // from attrs so TF source names/inputs never pollute typed attr application.
  std::unordered_map<std::string, std::string> tf_attrs;
  bool isInputNode = false;
  bool isOutputNode = false;

  // Typed attribute lookup; true only for an exact type T match (no coercion).
  template <typename T>
  bool getAttr(const std::string& name, T& out) const {
    auto it = attrs.find(name);
    if (it == attrs.end()) return false;
    if (auto* value = std::get_if<T>(&it->second)) {
      out = *value;
      return true;
    }
    return false;
  }
};

template <typename T>
bool OpContext::getAttr(const NodeInfo& node, llvm::StringRef name,
                        T& out) const {
  return node.getAttr(name.str(), out);
}

// Table-driven TF graph -> ATIR builder: addNode() validates a node against
// its declarative OpSpec, builds the op (generic for 1:1 ops, transformer
// otherwise), applies attr mappings, attaches metadata and registers results.
// All failure paths report a diagnostic and return failure; nothing aborts.
class MLIRBuilder {
 public:
  MLIRBuilder(ModuleOp& module)
      : module_(module), builder_(module.getContext()) {}
  virtual ~MLIRBuilder() = default;

  bool buildFromNodes(const std::vector<NodeInfo>& nodes);
  static bool isSupportedOp(llvm::StringRef opType);
  // All declared op specs (drives isSupportedOp); used by tests and tooling.
  static llvm::ArrayRef<OpSpec> getAllOpSpecs();

 private:
  const OpSpec* lookupSpec(llvm::StringRef opType) const;
  mlir::LogicalResult addNode(const NodeInfo& node);
  mlir::LogicalResult buildControlFlowMirrorNode(const NodeInfo& node);
  mlir::LogicalResult buildConstantNode(const NodeInfo& node);
  // Fallback for TF ops with no OpSpec: emit a warning and build an
  // atir.opaque placeholder that passes inputs/outputs through structurally.
  mlir::LogicalResult buildOpaqueOp(const NodeInfo& node);
  mlir::FailureOr<Operation*> buildGenericOp(const OpSpec& spec,
                                             const NodeInfo& node,
                                             llvm::ArrayRef<Type> outs,
                                             llvm::ArrayRef<Value> ins);
  mlir::LogicalResult applyAttrMappings(Operation* op, const NodeInfo& node,
                                        const OpSpec& spec);
  void attachMetadata(Operation* op, const NodeInfo& node, const OpSpec& spec);
  mlir::FailureOr<mlir::Type> getTensorType(const NodeInfo& node,
                                            unsigned outIdx);
  // Resolves a name to a produced value; strips a ":0" output-index suffix.
  mlir::Value resolveValue(llvm::StringRef name) const;
  mlir::LogicalResult emitNodeError(const NodeInfo& node,
                                    const std::string& message);

  std::vector<Value> addGraphInputs(const std::vector<NodeInfo>& inputNodes);
  std::vector<Value> addGraphOutputs(const std::vector<NodeInfo>& outputNodes);

  ModuleOp module_;
  OpBuilder builder_;
  func::FuncOp mainFunc_;
  // Output name -> produced SSA value (later nodes resolve inputs by name).
  std::unordered_map<std::string, Value> tensorValues_;
  // Name -> NodeInfo, so handlers can resolve constant inputs by name.
  std::unordered_map<std::string, const NodeInfo*> nodesByName_;
  // Set when a node cannot be built; buildFromNodes reports and stops.
  bool failed_ = false;
};
}  // namespace annc
#endif  // ANNC_OP_BUILDER_H
