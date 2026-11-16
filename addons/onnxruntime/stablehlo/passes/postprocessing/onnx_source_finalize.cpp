// Bind original ONNX identities after producer canonicalization. Never expose
// intermediate tensors as public results or infer a result slot from op order.
#include <map>
#include <set>
#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {
using Sources = std::set<std::string>;

void collectSourceIds(Location location,
                      const std::map<std::string, std::string> &names,
                      Sources &ids) {
  if (auto name = dyn_cast<NameLoc>(location)) {
    auto found = names.find(name.getName().str());
    if (found != names.end()) ids.insert(found->second);
    collectSourceIds(name.getChildLoc(), names, ids);
  } else if (auto fused = dyn_cast<FusedLoc>(location)) {
    for (Location child : fused.getLocations())
      collectSourceIds(child, names, ids);
  } else if (auto call = dyn_cast<CallSiteLoc>(location)) {
    collectSourceIds(call.getCaller(), names, ids);
    collectSourceIds(call.getCallee(), names, ids);
  }
}

// ONNX type information comes from source nodes/tensors, not extra results.
// Shape compatibility filters identity evidence; it never establishes identity.
bool hasCompatibleTensorType(Value value, const llvm::json::Object &tensor) {
  auto *info = tensor.getObject("type");
  auto type = dyn_cast<RankedTensorType>(value.getType());
  auto *shape = info ? info->getArray("shape") : nullptr;
  auto elem = info ? info->getInteger("elem_type") : std::nullopt;
  if (!type || !elem || (shape && shape->size() != size_t(type.getRank())))
    return false;
  Type element = type.getElementType();
  // Only compare element types whose ONNX/MLIR correspondence is unambiguous.
  bool matches =
      (*elem == 1 && element.isF32()) || (*elem == 10 && element.isF16()) ||
      (*elem == 11 && element.isF64()) || (*elem == 16 && element.isBF16()) ||
      (*elem == 9 && element.isInteger(1)) ||
      (*elem == 6 && element.isInteger(32)) ||
      (*elem == 7 && element.isInteger(64));
  if (!matches) return false;
  // A unique source-region exit can establish identity even if ONNX has no
  // rank. Keep the ONNX query unchanged; the HLO value already carries its
  // ranked type. Known dimensions must still agree.
  if (!shape) return true;
  for (auto [axis, dim] : llvm::enumerate(*shape))
    if (auto extent = dim.getAsInteger())
      if (!type.isDynamicDim(axis) && type.getDimSize(axis) != *extent)
        return false;
  return true;
}

class OnnxSourceFinalizePass
    : public PassWrapper<OnnxSourceFinalizePass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OnnxSourceFinalizePass)
  StringRef getArgument() const final { return "annc-finalize-source-map"; }
  StringRef getDescription() const final {
    return "Resolve ONNX node sources and unambiguous tensor boundaries";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](StringRef reason) {
      module.emitError(reason);
      signalPassFailure();
    };
    auto index = module->getAttrOfType<StringAttr>("annc.onnx_model");
    if (!index) return fail("missing imported ONNX graph index");
    auto parsed = llvm::json::parse(index.getValue());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return fail("invalid model-index JSON");
    }
    auto *report = parsed->getAsObject();
    auto *graph = report ? report->getObject("graph") : nullptr;
    auto *nodes = graph ? graph->getArray("nodes") : nullptr;
    auto *tensors = graph ? graph->getArray("tensors") : nullptr;
    if (!nodes || !tensors ||
        report->getString("schema") != "annc-model-index/2")
      return fail("expected the imported ONNX graph index");
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    if (functions.size() != 1 ||
        !llvm::hasSingleElement(functions[0].getBody()))
      return fail("source mapping requires one single-block function");
    auto function = functions[0];
    auto ret = dyn_cast<func::ReturnOp>(function.front().getTerminator());
    if (!ret) return fail("missing public return");

    OpBuilder builder(module.getContext());
    std::map<std::string, std::string> names;
    // The importer already checks original node and tensor uniqueness.
    for (auto &item : *nodes) {
      auto *node = item.getAsObject();
      if (!node || !node->getString("name") ||
          !node->getArray("inputs") || !node->getArray("outputs"))
        return fail("invalid node record");
      auto name = node->getString("name")->str();
      names.emplace(name, name);
    }
    llvm::DenseMap<Operation *, Sources> sources;
    function.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getName().getDialectNamespace() != "stablehlo" &&
          !isa<atir::OpaqueOp>(op))
        return;
      Sources &ids = sources[op];
      collectSourceIds(op->getLoc(), names, ids);
      if (auto existing = op->getAttrOfType<ArrayAttr>("source_ids"))
        for (Attribute attr : existing)
          if (auto id = dyn_cast<StringAttr>(attr))
            if (names.count(id.str())) ids.insert(id.str());
      if (ids.empty()) {
        auto parent = sources.find(op->getParentOp());
        if (parent != sources.end()) ids = parent->second;
      }
      SmallVector<Attribute> attrs;
      for (const auto &id : ids) attrs.push_back(builder.getStringAttr(id));
      op->setAttr("source_ids", builder.getArrayAttr(attrs));
    });

    std::map<std::string, const llvm::json::Object *> records;
    std::map<int64_t, std::string> publicNames;
    for (auto &item : *tensors) {
      auto *tensor = item.getAsObject();
      auto name = tensor ? tensor->getString("name") : std::nullopt;
      if (!name) return fail("invalid tensor record");
      records.emplace(name->str(), tensor);
      if (auto *outputs = tensor->getArray("graph_outputs"))
        for (auto &output : *outputs) {
          auto slot = output.getAsInteger();
          if (!slot || *slot < 0) return fail("invalid public output slot");
          publicNames.emplace(*slot, name->str());
        }
    }
    if (publicNames.size() != ret.getNumOperands())
      return fail("public output count differs from source ONNX");
    std::map<std::string, Value> values;
    auto bind = [&](StringRef name, Value value) {
      if (!records.count(name.str())) return false;
      auto [it, added] = values.emplace(name.str(), value);
      return added || it->second == value;
    };
    for (BlockArgument arg : function.getArguments()) {
      auto name = function.getArgAttrOfType<StringAttr>(arg.getArgNumber(),
                                                        "onnx.name");
      if (!name || !bind(name.getValue(), arg))
        return fail("missing source graph input identity");
    }
    for (auto &[slot, name] : publicNames) {
      if (slot >= ret.getNumOperands())
        return fail("invalid public output slot");
      auto imported =
          function.getResultAttrOfType<StringAttr>(slot, "onnx.name");
      if (!imported || imported.getValue() != name ||
          !bind(name, ret.getOperand(slot)))
        return fail("missing or reordered public output identity");
    }
    // Initializers carry the importer's exact Initializer_<tensor> location.
    // A colliding ONNX node name makes that evidence ambiguous.
    for (auto &[name, record] : records) {
      if (!record->getBoolean("initializer").value_or(false) ||
          names.count("Initializer_" + name))
        continue;
      std::map<std::string, std::string> initializer{
          {"Initializer_" + name, name}};
      SmallVector<Value> candidates;
      function.walk([&](Operation *op) {
        if (!isa<stablehlo::ConstantOp>(op)) return;
        Sources found;
        collectSourceIds(op->getLoc(), initializer, found);
        if (!found.empty() &&
            hasCompatibleTensorType(op->getResult(0), *record))
          candidates.push_back(op->getResult(0));
      });
      if (candidates.size() == 1 && !bind(name, candidates[0]))
        return fail("conflicting initializer identity");
    }
    // A single ONNX output can be bound only to a unique escaping value of
    // that node's source region. Helpers internal to the region are excluded.
    // Multiple outputs or fused source sets require explicit public anchors;
    // output slots are never guessed from traversal order or matching shapes.
    for (auto &item : *nodes) {
      auto &node = *item.getAsObject();
      auto *outputs = node.getArray("outputs");
      if (outputs->size() != 1) continue;
      auto name = (*outputs)[0].getAsString();
      if (!name || !records.count(name->str()) || values.count(name->str()))
        continue;
      Sources expected{node.getString("name")->str()};
      SmallVector<Value> candidates;
      for (auto &[op, ids] : sources) {
        if (ids != expected || op->getBlock() != &function.getBody().front())
          continue;
        for (Value value : op->getResults()) {
          if (!hasCompatibleTensorType(value, *records.at(name->str())))
            continue;
          bool escapes = llvm::any_of(value.getUsers(), [&](Operation *user) {
            auto found = sources.find(user);
            return found == sources.end() || found->second != expected;
          });
          if (escapes) candidates.push_back(value);
        }
      }
      if (candidates.size() == 1) bind(*name, candidates[0]);
    }
    // A merged source set identifies a region, not an arbitrary member's
    // output. Bind only its sole original exit to its sole compatible HLO
    // exit. In particular, do not invent aliases for eliminated intermediates.
    std::set<Sources> groups;
    for (auto &[op, ids] : sources)
      if (ids.size() > 1) groups.insert(ids);
    for (const Sources &group : groups) {
      SmallVector<StringRef> exits;
      for (auto &[name, record] : records) {
        auto *producer = record->getObject("producer");
        auto id = producer ? producer->getString("node") : std::nullopt;
        if (!id || !group.count(id->str())) continue;
        bool escapes = record->getArray("graph_outputs") != nullptr;
        for (auto &use : *record->getArray("consumers"))
          escapes |= !group.count(use.getAsObject()->getString("node")->str());
        if (escapes) exits.push_back(name);
      }
      if (exits.size() != 1 || values.count(exits.front().str())) continue;
      SmallVector<Value> candidates;
      for (auto &[op, ids] : sources) {
        if (ids != group || op->getBlock() != &function.front()) continue;
        for (Value value : op->getResults()) {
          if (!hasCompatibleTensorType(value, *records.at(exits.front().str())))
            continue;
          bool escapes = llvm::any_of(value.getUsers(), [&](Operation *user) {
            auto found = sources.find(user);
            return found == sources.end() || found->second != group;
          });
          if (escapes) candidates.push_back(value);
        }
      }
      if (candidates.size() == 1) bind(exits.front(), candidates.front());
    }
    // CSE can erase Initializer_<name> locations. Recover an initializer only
    // through a proven binary boundary. Elementwise ops need the other input;
    // a unique standard MatMul lowering also fixes its positional operands.
    // Never search constants by payload or shape: both could match unrelated
    // initializers. Multiple ONNX initializers may legitimately share one SSA
    // constant after CSE.
    auto leaf = [](Value value) {
      while (Operation *op = value.getDefiningOp()) {
        StringRef kind = op->getName().getStringRef();
        if (kind != "stablehlo.broadcast_in_dim" &&
            kind != "stablehlo.dynamic_broadcast_in_dim")
          break;
        value = op->getOperand(0);
      }
      return value;
    };
    for (auto &item : *nodes) {
      auto &node = *item.getAsObject();
      auto *inputs = node.getArray("inputs");
      auto *outputs = node.getArray("outputs");
      auto domain = node.getString("domain");
      if (!domain || (*domain != "" && *domain != "ai.onnx") ||
          inputs->size() != 2 || outputs->size() != 1)
        continue;
      auto kind = node.getString("op_type");
      StringRef hlo;
      bool commute = false;
      if (kind == "Add") {
        hlo = "stablehlo.add";
        commute = true;
      } else if (kind == "Mul") {
        hlo = "stablehlo.multiply";
        commute = true;
      } else if (kind == "Sub")
        hlo = "stablehlo.subtract";
      else if (kind == "Div")
        hlo = "stablehlo.divide";
      else if (kind == "MatMul")
        hlo = "stablehlo.dot_general";
      else
        continue;
      auto out = values.find((*outputs)[0].getAsString()->str());
      Operation *op =
          out == values.end() ? nullptr : out->second.getDefiningOp();
      Sources expected{node.getString("name")->str()};
      if (kind == "MatMul" && op) {
        // Batched MatMul may have identity broadcast wrappers. Preserve
        // operand order and require the exact standard contraction axes.
        if (op->getName().getStringRef() == "stablehlo.broadcast_in_dim" ||
            op->getName().getStringRef() ==
                "stablehlo.dynamic_broadcast_in_dim")
          op = leaf(out->second).getDefiningOp();
        if (isa_and_nonnull<stablehlo::DotOp>(op)) {
          // Rank-two ONNX MatMul lowers to dot rather than dot_general.
          // Its operand order is fixed by StableHLO dot semantics.
          hlo = "stablehlo.dot";
          if (cast<RankedTensorType>(op->getOperand(0).getType()).getRank() !=
                  2 ||
              cast<RankedTensorType>(op->getOperand(1).getType()).getRank() !=
                  2)
            continue;
        } else {
          auto dot = dyn_cast_or_null<stablehlo::DotGeneralOp>(op);
          if (!dot) continue;
          auto lhs = dyn_cast<RankedTensorType>(dot.getLhs().getType());
          auto rhs = dyn_cast<RankedTensorType>(dot.getRhs().getType());
          if (!lhs || !rhs || lhs.getRank() < 2 ||
              lhs.getRank() != rhs.getRank())
            continue;
          SmallVector<int64_t> batch;
          for (int64_t i = 0; i < lhs.getRank() - 2; ++i) batch.push_back(i);
          auto dims = dot.getDotDimensionNumbers();
          if (dims.getLhsBatchingDimensions() != ArrayRef<int64_t>(batch) ||
              dims.getRhsBatchingDimensions() != ArrayRef<int64_t>(batch) ||
              dims.getLhsContractingDimensions() !=
                  ArrayRef<int64_t>{lhs.getRank() - 1} ||
              dims.getRhsContractingDimensions() !=
                  ArrayRef<int64_t>{rhs.getRank() - 2})
            continue;
        }
      }
      if (!op || op->getName().getStringRef() != hlo || sources[op] != expected)
        continue;
      bool positionalMatmul =
          kind == "MatMul" &&
          llvm::count_if(sources, [&](const auto &entry) {
            return entry.second == expected &&
                   entry.first->getName().getStringRef() == hlo;
          }) == 1;
      for (unsigned slot = 0; slot != 2; ++slot) {
        auto name = (*inputs)[slot].getAsString();
        auto other = values.find((*inputs)[1 - slot].getAsString()->str());
        auto record = name ? records.find(name->str()) : records.end();
        if (record == records.end() || values.count(name->str()) ||
            (other == values.end() && !positionalMatmul) ||
            !record->second->getBoolean("initializer").value_or(false) ||
            record->second->getInteger("graph_input"))
          continue;
        SmallVector<Value> candidates;
        for (unsigned operand = 0; operand != 2; ++operand) {
          if (!commute && operand != slot) continue;
          Value constant = leaf(op->getOperand(operand));
          Operation *def = constant.getDefiningOp();
          // A unique standard MatMul rooted at its mapped output has fixed
          // input slots even if canonicalization eliminated the other input's
          // ONNX boundary. Elementwise recovery still needs its other anchor.
          if ((positionalMatmul ||
               leaf(op->getOperand(1 - operand)) == other->second) &&
              def && def->getName().getStringRef() == "stablehlo.constant" &&
              hasCompatibleTensorType(constant, *record->second) &&
              !llvm::is_contained(candidates, constant))
            candidates.push_back(constant);
        }
        if (candidates.size() == 1) bind(*name, candidates.front());
      }
    }
    // Elimination of ONNX Identity preserves tensor equality, including aliases
    // of multi-result nodes. This uses ONNX semantics, never SSA numbering.
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &item : *nodes) {
        auto &node = *item.getAsObject();
        auto domain = node.getString("domain");
        if (node.getString("op_type") != "Identity" || !domain ||
            (*domain != "" && *domain != "ai.onnx") ||
            node.getArray("inputs")->size() != 1 ||
            node.getArray("outputs")->size() != 1)
          continue;
        auto input = (*node.getArray("inputs"))[0].getAsString();
        auto output = (*node.getArray("outputs"))[0].getAsString();
        if (!input || !output) continue;
        auto a = values.find(input->str()), b = values.find(output->str());
        if (a != values.end() && b != values.end()) {
          if (a->second != b->second) return fail("conflicting Identity alias");
        } else if (a != values.end() || b != values.end()) {
          if (!bind(a != values.end() ? *output : *input,
                    a != values.end() ? a->second : b->second))
            return fail("invalid Identity tensor");
          changed = true;
        }
      }
    }
    llvm::DenseMap<Value, SmallVector<StringRef>> aliases;
    for (auto &[name, value] : values) aliases[value].push_back(name);
    function.walk([&](Operation *op) {
      if (!sources.count(op)) return;
      SmallVector<Attribute> results;
      for (Value result : op->getResults())
        results.push_back(builder.getStrArrayAttr(aliases[result]));
      op->setAttr("annc.result_tensors", builder.getArrayAttr(results));
    });
    for (BlockArgument arg : function.getArguments())
      function.setArgAttr(arg.getArgNumber(), "annc.tensor_names",
                          builder.getStrArrayAttr(aliases[arg]));

    module->setAttr("annc.provenance",
                    builder.getStringAttr("onnx-node-name/2"));
  }
};
}  // namespace

void registerOnnxSourceFinalizePass() {
  PassRegistration<OnnxSourceFinalizePass>();
}
