#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "Dialect/Atir/AtirOps.h"
#include "FusionMetadata/FusionMetadata.h"
#include "HloGraph.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "nlohmann/json.hpp"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using Json = nlohmann::json;

namespace {
using annc::stablehlo::HloGraph;
// The producer finalizer writes ONNX/HLO relations once. Binding only parses
// that data into lookups for this process; it never reconstructs provenance.
struct OnnxSourceGraph {
  Json graph;
  std::map<std::string, const Json *> nodes;
  // Tensor aliases carried by HLO. Binding resolves the names needed by each
  // candidate against the original graph; eliminated intermediates may be
  // absent.
  std::map<std::string, mlir::Value> values;
  llvm::DenseMap<Value, SmallVector<StringRef>> aliases;
  std::map<std::string, const Json *> tensors;

  explicit OnnxSourceGraph(mlir::ModuleOp module);
};

OnnxSourceGraph::OnnxSourceGraph(ModuleOp module) {
  auto attr = module->getAttrOfType<StringAttr>("annc.onnx_model");
  if (!attr) throw std::runtime_error("missing imported ONNX graph index");
  auto index = Json::parse(attr.getValue().str());
  if (index.at("schema") != "annc-model-index/2")
    throw std::runtime_error("expected the imported ONNX graph index");
  graph = std::move(index.at("graph"));
  for (const auto &node : graph.at("nodes"))
    nodes.emplace(node.at("name"), &node);
  if (module->getAttrOfType<StringAttr>("annc.provenance") !=
      StringAttr::get(module.getContext(), "onnx-node-name/2"))
    throw std::runtime_error("IR provenance method mismatch");
  SmallVector<func::FuncOp> functions;
  for (auto func : module.getOps<func::FuncOp>())
    if (!func->hasAttr("annc.kernel")) functions.push_back(func);
  if (functions.size() != 1 || !llvm::hasSingleElement(functions[0].getBody()))
    throw std::runtime_error("requires one single-block function");
  auto function = functions[0];
  for (const auto &tensor : graph.at("tensors"))
    tensors.emplace(tensor.at("name").get<std::string>(), &tensor);
  auto add = [&](Value value, Attribute attribute, StringRef name) {
    auto aliases = dyn_cast_or_null<ArrayAttr>(attribute);
    if (!aliases)
      throw std::runtime_error("invalid " + name.str() + ": expected an array");
    for (Attribute attr : aliases) {
      auto alias = dyn_cast<StringAttr>(attr);
      if (!alias)
        throw std::runtime_error("invalid " + name.str() +
                                 ": expected a tensor name");
      values.emplace(alias.str(), value);
    }
  };
  for (BlockArgument arg : function.getArguments())
    add(arg, function.getArgAttr(arg.getArgNumber(), "annc.tensor_names"),
        "annc.tensor_names");
  function.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() != "stablehlo" &&
        !isa<atir::OpaqueOp>(op))
      return;
    auto results = op->getAttrOfType<ArrayAttr>("annc.result_tensors");
    if (!results || results.size() != op->getNumResults())
      throw std::runtime_error(
          "invalid annc.result_tensors: expected one alias array per result");
    for (auto [value, aliases] : llvm::zip(op->getResults(), results))
      add(value, aliases, "annc.result_tensors");
  });
  // Preserve the existing name order when resolving shared constant aliases.
  for (const auto &[name, value] : values) aliases[value].push_back(name);
}

bool isImmutableTensor(const Json &tensor) {
  return tensor.value("initializer", false) && !tensor.contains("graph_input");
}

// Binding reads ONNX identity/topology after HLO matching. A failed binding
// leaves the independently compiled kernel in mixed IR, without deployment
// permission. Neither the HLO graph nor its numerical proof is rewritten.
bool bindKernel(func::FuncOp kernel, const OnnxSourceGraph &source,
                const HloGraph &graph, std::set<std::string> &claimed,
                std::string &reason) {
  auto fail = [&](StringRef text) {
    reason = text.str();
    return false;
  };
  auto parsed = annc::fusion::readFusionInfo(kernel);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return fail("invalid FusionInfo");
  }
  const auto &info = *parsed;
  auto region = kernel->getAttrOfType<DictionaryAttr>("annc.hlo_region");
  auto version = region ? region.getAs<IntegerAttr>("version") : IntegerAttr();
  if (!version || version.getInt() != 1) return fail("missing HLO region");
  auto refs = [&](StringRef key, size_t count, SmallVectorImpl<Value> &out) {
    auto array = region.getAs<ArrayAttr>(key);
    if (!array || array.size() != count) return false;
    for (auto attr : array) {
      auto id = dyn_cast<StringAttr>(attr);
      auto it = id ? graph.values.find(id.str()) : graph.values.end();
      if (it == graph.values.end()) return false;
      out.push_back(it->second);
    }
    return true;
  };
  SmallVector<Value> inputs, results;
  if (!refs("arguments", info.args.size(), inputs) ||
      !refs("outputs", info.outputs.size(), results))
    return fail("unresolved HLO boundary reference");
  auto sameType = [](Value value, const annc::fusion::FusionArg &arg) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != arg.rank ||
        !llvm::equal(type.getShape(), arg.shape))
      return false;
    std::string dtype;
    llvm::raw_string_ostream stream(dtype);
    type.getElementType().print(stream);
    return dtype == arg.dtype;
  };
  for (auto [value, arg] : llvm::zip(inputs, info.args))
    if (!sameType(value, arg))
      return fail("HLO argument type differs from FusionInfo");
  for (auto [value, arg] : llvm::zip(results, info.outputs))
    if (!sameType(value, arg))
      return fail("HLO output type differs from FusionInfo");
  auto function = region.getAs<StringAttr>("function");
  if (!function || results.empty())
    return fail("missing HLO function or output");
  for (Value value : llvm::concat<Value>(inputs, results)) {
    auto owner = dyn_cast<func::FuncOp>(value.getParentBlock()->getParentOp());
    if (!owner || owner.getSymName() != function.getValue())
      return fail("HLO boundary belongs to a different function");
  }
  std::set<Operation *> operations;
  std::set<std::string> origins;
  auto membersAttr = region.getAs<ArrayAttr>("operations");
  if (!membersAttr) return fail("missing HLO members");
  for (auto attr : membersAttr) {
    auto id = dyn_cast<StringAttr>(attr);
    auto it = id ? graph.operations.find(id.str()) : graph.operations.end();
    if (it == graph.operations.end()) return fail("unknown HLO member");
    if (!operations.insert(it->second).second)
      return fail("duplicate HLO member");
    it->second->walk([&](Operation *op) {
      if (auto ids = op->getAttrOfType<ArrayAttr>("source_ids"))
        for (auto source : ids)
          if (auto name = dyn_cast<StringAttr>(source))
            origins.insert(name.str());
    });
  }
  // Serialized references must describe exactly the backward HLO slice. A
  // missing member cannot grant permission to remove its original source.
  llvm::DenseSet<Value> boundary(inputs.begin(), inputs.end());
  std::set<Operation *> slice;
  bool complete = true;
  std::function<void(Value)> collectHlo = [&](Value value) {
    if (boundary.count(value)) return;
    auto *op = value.getDefiningOp();
    if (!op || !graph.nodes.count(op)) {
      complete = false;
      return;
    }
    if (!slice.insert(op).second) return;
    if (HloGraph::shapeBoundaryUse(op, inputs)) return;
    for (Value operand : op->getOperands()) collectHlo(operand);
  };
  for (Value result : results) collectHlo(result);
  if (!complete || slice.empty() || slice != operations)
    return fail("HLO members differ from the proven boundary slice");
  const auto &nodes = source.nodes;
  for (const auto &name : origins)
    if (!nodes.count(name))
      return fail("unknown original source node in kernel region");
  auto escapes = [&](const Json &tensor, const std::set<std::string> &members) {
    if (tensor.contains("graph_outputs")) return true;
    for (const auto &use : tensor.at("consumers"))
      if (!members.count(use.at("node"))) return true;
    return false;
  };
  std::vector<std::string> outputNames;
  for (Value result : results) {
    std::vector<std::string> choices, escaping;
    for (StringRef alias : source.aliases.lookup(result)) {
      auto name = alias.str();
      auto found = source.tensors.find(name);
      if (found == source.tensors.end()) continue;
      const auto &tensor = *found->second;
      if (!tensor.contains("producer") ||
          !origins.count(tensor.at("producer").at("node")))
        continue;
      choices.push_back(name);
      if (escapes(tensor, origins)) escaping.push_back(name);
    }
    if (escaping.size() == 1) choices = escaping;
    if (choices.size() != 1)
      return fail("ambiguous or missing original output");
    outputNames.push_back(choices[0]);
  }
  // CSE may give several original names to one constant. Resolve aliases
  // against the actual incoming edges of this original region, not globally.
  std::set<std::string> boundaries;
  for (Value input : inputs)
    for (StringRef name : source.aliases.lookup(input))
      if (source.tensors.count(name.str())) boundaries.insert(name.str());
  std::set<std::string> members, visited;
  bool valid = true;
  std::function<void(const std::string &)> collect =
      [&](const std::string &name) {
        if (boundaries.count(name) || !visited.insert(name).second) return;
        const auto &tensor = *source.tensors.at(name);
        if (isImmutableTensor(tensor)) return;
        if (!tensor.contains("producer")) {
          valid = false;
          return;
        }
        std::string id = tensor.at("producer").at("node");
        if (!nodes.count(id) || claimed.count(id)) {
          valid = false;
          return;
        }
        if (!members.insert(id).second) return;
        for (const std::string input : nodes.at(id)->at("inputs"))
          if (!input.empty()) collect(input);
      };
  for (const auto &name : outputNames) collect(name);
  if (!valid || members.empty())
    return fail("original region is overlapping or unbound");
  // Conversion preserves this framework-owned operator as atir.opaque.
  // Keep it outside compiled regions so ORT executes the original node.
  for (const auto &id : members) {
    const auto &node = *nodes.at(id);
    if (node.value("domain", "") == "com.microsoft" &&
        node.value("op_type", "") == "FusedTensordotMatMul")
      return fail(
          "framework-owned operator: com.microsoft::FusedTensordotMatMul");
  }
  std::set<std::string> incoming;
  for (const auto &id : members) {
    for (const std::string name : nodes.at(id)->at("inputs")) {
      if (name.empty()) continue;
      const auto &tensor = *source.tensors.at(name);
      if (!tensor.contains("producer") ||
          !members.count(tensor.at("producer").at("node")))
        incoming.insert(name);
    }
    for (const std::string name : nodes.at(id)->at("outputs")) {
      if (name.empty()) continue;
      if (escapes(*source.tensors.at(name), members) &&
          !llvm::is_contained(outputNames, name))
        return fail("original intermediate escapes the kernel region");
    }
  }
  std::vector<std::string> inputNames;
  std::set<std::string> used;
  for (auto [slot, value] : llvm::enumerate(inputs)) {
    std::vector<std::string> choices;
    for (const auto &name : incoming) {
      auto it = source.values.find(name);
      if (it != source.values.end() && it->second == value &&
          ((info.args[slot].role != "fixed" &&
            info.args[slot].role != "constant") ||
           isImmutableTensor(*source.tensors.at(name))))
        choices.push_back(name);
    }
    // Equal immutable constants are interchangeable. A mutable alias is not.
    if (choices.empty() ||
        (choices.size() > 1 && !llvm::all_of(choices, [&](const auto &name) {
           return isImmutableTensor(*source.tensors.at(name));
         })))
      return fail("unproven original argument or immutable parameter");
    auto unused = llvm::find_if(
        choices, [&](const auto &name) { return !used.count(name); });
    const auto &name = unused == choices.end() ? choices.front() : *unused;
    inputNames.push_back(name);
    used.insert(name);
  }
  // Keep deployment boundaries at the same level of evidence as the
  // framework baseline.  StableHLO may carry a ranked type recovered from
  // computation even when the original ONNX value had no rank metadata;
  // that is sufficient for matching, but not for replacing the ONNX node.
  if (info.pattern == "batched_matmul_add") {
    if (inputNames.empty()) return fail("BMMAdd has no data input");
    const auto &data = *source.tensors.at(inputNames.front());
    auto type = data.find("type");
    auto hloType = dyn_cast<RankedTensorType>(inputs.front().getType());
    if (type == data.end() || !type->is_object() || !type->contains("shape") ||
        !(*type)["shape"].is_array() || !hloType ||
        (*type)["shape"].size() != static_cast<size_t>(hloType.getRank()))
      return fail("BMMAdd data input lacks a proven ONNX rank");
  }
  if (info.pattern == "layer_norm") {
    if (inputNames.empty()) return fail("LayerNorm has no data input");
    const auto &data = *source.tensors.at(inputNames.front());
    auto producer = data.find("producer");
    if (producer == data.end() || !producer->is_object())
      return fail("LayerNorm data input has no ONNX producer");
    auto node = nodes.find(producer->value("node", std::string()));
    if (node == nodes.end())
      return fail("LayerNorm data input producer is unknown");
  }
  for (const auto &name : incoming)
    if (!used.count(name) && !isImmutableTensor(*source.tensors.at(name)))
      return fail("original input is absent from the kernel boundary");
  // A source may lower to several HLO operations. Never remove the whole
  // source node after proving only one of its live data computations.
  for (const auto &[id, op] : graph.operations) {
    if (operations.count(op) || isa<mlir::stablehlo::ConstantOp>(op)) continue;
    auto sources = op->getAttrOfType<ArrayAttr>("source_ids");
    if (!sources || !llvm::any_of(sources, [&](Attribute attr) {
          auto source = dyn_cast<StringAttr>(attr);
          return source && members.count(source.str());
        }))
      continue;
    for (Value value : op->getResults()) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type)
        return fail("unranked operation outside the proven source region");
      if (!isa<FloatType>(type.getElementType()) &&
          !type.getElementType().isInteger(1))
        continue;
      if (llvm::any_of(value.getUsers(), [&](Operation *user) {
            return !HloGraph::shapeUse(user);
          }))
        return fail("source node is only partially covered by the HLO kernel");
    }
  }
  Builder b(kernel.getContext());
  auto metadata = kernel->getAttrOfType<DictionaryAttr>("fusion.metadata");
  NamedAttrList updated(metadata);
  auto rename = [&](StringRef key, const std::vector<std::string> &names) {
    SmallVector<Attribute> attrs;
    for (auto [index, attr] : llvm::enumerate(metadata.getAs<ArrayAttr>(key))) {
      NamedAttrList fields(cast<DictionaryAttr>(attr));
      fields.set("tf_name", b.getStringAttr(names[index]));
      // HLO constants become immutable model bindings, not embedded context
      // arguments. The proof above has already rejected overridable aliases.
      if (key == "args" && info.args[index].role == "constant")
        fields.set("role", b.getStringAttr("fixed"));
      attrs.push_back(fields.getDictionary(b.getContext()));
    }
    updated.set(key, b.getArrayAttr(attrs));
  };
  rename("args", inputNames);
  rename("outputs", outputNames);
  std::set<std::string> names;
  for (const auto &id : members) names.insert(nodes.at(id)->at("name"));
  SmallVector<StringRef> memberNames;
  for (const auto &name : names) memberNames.push_back(name);
  updated.set("member_node_names", b.getStrArrayAttr(memberNames));
  kernel->setAttr("fusion.metadata", updated.getDictionary(b.getContext()));
  auto checked = annc::fusion::readFusionInfo(kernel);
  if (!checked) {
    llvm::consumeError(checked.takeError());
    kernel->setAttr("fusion.metadata", metadata);
    return fail("original endpoint violates FusionInfo contract");
  }
  kernel->setAttr("annc.onnx_bound", b.getUnitAttr());
  claimed.insert(members.begin(), members.end());
  return true;
}

struct BindPass : PassWrapper<BindPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BindPass)
  BindPass() = default;
  BindPass(const BindPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "annc-stablehlo-bind-onnx"; }
  StringRef getDescription() const final {
    return "Bind independently proven HLO kernels to ONNX regions";
  }
  void runOnOperation() override {
    auto module = getOperation();
    try {
      OnnxSourceGraph source(module);
      HloGraph graph(module);
      std::set<std::string> claimed;
      Builder b(&getContext());
      SmallVector<Attribute> rejected;
      for (auto kernel : module.getOps<func::FuncOp>()) {
        if (!kernel->hasAttr("annc.kernel")) continue;
        std::string reason;
        if (!bindKernel(kernel, source, graph, claimed, reason)) {
          kernel->setAttr("annc.onnx_rejection", b.getStringAttr(reason));
          rejected.push_back(b.getStringAttr(reason));
        }
      }
      module->setAttr("annc.onnx_binding_rejections", b.getArrayAttr(rejected));
    } catch (const std::exception &error) {
      module.emitError() << "StableHLO ONNX binding: " << error.what();
      signalPassFailure();
    }
  }
};
}  // namespace

void registerStablehloOnnxBindingPass() { PassRegistration<BindPass>(); }
