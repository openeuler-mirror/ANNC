#include "HloGraph.h"

#include <functional>

#include "Dialect/Atir/BuiltinKernelSchema.h"
#include "FusionMetadata/FusionMetadata.h"
#include "mlir/IR/Verifier.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace annc::stablehlo {

HloGraph::HloGraph(ModuleOp module) : module(module) {
  unsigned f = 0;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func->hasAttr("annc.kernel")) continue;
    const std::string prefix = "h" + std::to_string(f++);
    if (!llvm::hasSingleElement(func.getBody())) continue;
    for (auto arg : func.getArguments()) {
      auto id = prefix + "_arg" + std::to_string(arg.getArgNumber()) + ":0";
      names[arg] = id;
      values.emplace(id, arg);
    }
    unsigned n = 0;
    for (auto &op : func.getBody().front()) {
      auto id = prefix + "_op" + std::to_string(n++);
      nodes[&op] = id;
      operations.emplace(id, &op);
      for (auto result : op.getResults()) {
        auto value = id + ":" + std::to_string(result.getResultNumber());
        names[result] = value;
        values.emplace(value, result);
      }
    }
  }
}

std::string HloGraph::name(Value value) const {
  auto it = names.find(value);
  return it == names.end() ? std::string() : it->second;
}
std::string HloGraph::node(Operation *op) const {
  auto it = nodes.find(op);
  return it == nodes.end() ? std::string() : it->second;
}
bool HloGraph::shapeUse(Operation *op) {
  return isa<mlir::stablehlo::GetDimensionSizeOp>(op);
}

Value HloGraph::identitySource(Value value) {
  while (auto transpose = value.getDefiningOp<mlir::stablehlo::TransposeOp>()) {
    for (auto [axis, source] : llvm::enumerate(transpose.getPermutation()))
      if (static_cast<int64_t>(axis) != source) return value;
    value = transpose.getOperand();
  }
  return value;
}

bool HloGraph::shapeBoundaryUse(Operation *op, ValueRange inputs) {
  if (!shapeUse(op)) return false;
  Value source = identitySource(op->getOperand(0));
  return llvm::any_of(
      inputs, [&](Value input) { return identitySource(input) == source; });
}

bool HloGraph::fuse(StringRef pattern, StringRef builtin, ValueRange inputs,
                    ValueRange outputs, ArrayRef<int64_t> attrs,
                    SmallVectorImpl<Attribute> &rejections) {
  OpBuilder b(module.getContext());
  auto schema = atir::getContextKernelSchema(builtin, attrs);
  if (!schema || outputs.empty()) return false;
  auto record = [&](Value v, bool output) -> Attribute {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || name(v).empty()) return {};
    std::string dtype;
    llvm::raw_string_ostream os(dtype);
    t.getElementType().print(os);
    return b.getDictionaryAttr(
        {b.getNamedAttr("tf_name", b.getStringAttr(name(v))),
         b.getNamedAttr(
             "role",
             b.getStringAttr(output ? "output"
                             : v.getDefiningOp<mlir::stablehlo::ConstantOp>()
                                 ? "constant"
                                 : "dynamic")),
         b.getNamedAttr("dtype", b.getStringAttr(dtype)),
         b.getNamedAttr("rank", b.getI64IntegerAttr(t.getRank())),
         b.getNamedAttr("shape", b.getI64ArrayAttr(t.getShape()))});
  };
  SmallVector<Attribute> args, results;
  for (Value v : inputs) {
    auto attr = record(v, false);
    if (!attr) return false;
    args.push_back(attr);
  }
  for (Value v : outputs) {
    auto attr = record(v, true);
    if (!attr) return false;
    results.push_back(attr);
  }
  auto anchor = node(outputs.front().getDefiningOp());
  auto symbol = "annc_" + pattern.str() + "_" + anchor;
  auto kernel = func::FuncOp::create(
      outputs.front().getLoc(), symbol,
      b.getFunctionType(inputs.getTypes(), outputs.getTypes()));
  kernel.setPrivate();
  kernel->setAttr("annc.kernel", b.getUnitAttr());
  kernel->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  kernel->setAttr("fusion.pattern", b.getStringAttr(pattern));
  kernel->setAttr(
      "fusion.metadata",
      b.getDictionaryAttr(
          {b.getNamedAttr("tf.name", b.getStringAttr(anchor)),
           b.getNamedAttr("fusion.pattern", b.getStringAttr(pattern)),
           b.getNamedAttr("execution_mode", b.getStringAttr("aot")),
           b.getNamedAttr("kernel_name", b.getStringAttr(symbol)),
           b.getNamedAttr("abi", b.getStringAttr("annc_context_v3")),
           b.getNamedAttr("context_i64_attrs", b.getI64ArrayAttr(attrs)),
           b.getNamedAttr("args", b.getArrayAttr(args)),
           b.getNamedAttr("outputs", b.getArrayAttr(results))}));
  b.setInsertionPointToStart(kernel.addEntryBlock());
  OperationState state(kernel.getLoc(), "stablehlo.custom_call");
  state.addOperands(kernel.getArguments());
  state.addTypes(outputs.getTypes());
  state.addAttribute("call_target_name", b.getStringAttr(builtin));
  state.addAttribute("backend_config", b.getStringAttr(""));
  state.addAttribute("has_side_effect", b.getBoolAttr(false));
  state.addAttribute(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          b.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL));
  state.addAttribute("called_computations", b.getArrayAttr({}));
  state.addAttribute("annc.kernel_schema", schema->toMetadata(b.getContext()));
  auto *call = b.create(state);
  b.create<func::ReturnOp>(kernel.getLoc(), call->getResults());
  bool accepted = succeeded(verify(kernel)) && publish(kernel, rejections);
  kernel.erase();
  return accepted;
}

bool HloGraph::publish(func::FuncOp kernel,
                       SmallVectorImpl<Attribute> &rejections) {
  Builder b(module.getContext());
  auto reject = [&](StringRef reason) {
    rejections.push_back(b.getStringAttr(reason));
    return false;
  };
  auto info = fusion::readFusionInfo(kernel);
  if (!info) {
    llvm::consumeError(info.takeError());
    return reject("invalid local FusionInfo");
  }
  if (info->abi != "annc_context_v3" || !info->kernelArgOrder.empty() ||
      info->outputs.empty() || module.lookupSymbol(kernel.getSymName()))
    return reject("kernel ABI or symbol mismatch");
  llvm::DenseSet<Value> boundary, outputs;
  SmallVector<Value> inputs;
  SmallVector<StringRef> args, results;
  for (const auto &arg : info->args) {
    auto it = values.find(arg.tfName);
    if (it == values.end()) return reject("unresolved local kernel argument");
    boundary.insert(it->second);
    inputs.push_back(it->second);
    args.push_back(names.find(it->second)->second);
  }
  for (const auto &out : info->outputs) {
    auto it = values.find(out.tfName);
    if (it == values.end()) return reject("unresolved local kernel output");
    outputs.insert(it->second);
    results.push_back(names.find(it->second)->second);
  }
  std::set<Operation *> region;
  bool valid = true;
  std::function<void(Value)> collect = [&](Value value) {
    if (boundary.count(value)) return;
    auto *op = value.getDefiningOp();
    if (!op || !nodes.count(op)) {
      valid = false;
      return;
    }
    if (!region.insert(op).second) return;
    // A dimension query may precede an identity transpose at the boundary.
    // Its extent is still supplied by that argument, not an extra data input.
    if (shapeBoundaryUse(op, inputs)) return;
    for (Value input : op->getOperands()) collect(input);
  };
  for (Value output : outputs) collect(output);
  if (!valid || region.empty())
    return reject("HLO region has an unbound input");
  // Shape programs and constants may be shared; computation may not overlap.
  auto computation = [](Operation *op) {
    return !isa<mlir::stablehlo::ConstantOp>(op) &&
           llvm::any_of(op->getResultTypes(), [](Type type) {
             auto t = dyn_cast<RankedTensorType>(type);
             return t && (isa<FloatType>(t.getElementType()) ||
                          t.getElementType().isInteger(1));
           });
  };
  for (Operation *op : region) {
    if (op->getName().getStringRef() == "atir.opaque")
      return reject("framework fallback cannot be absorbed into a kernel");
    if (!computation(op)) continue;
    if (claimed.count(op)) return reject("overlapping HLO computation");
    for (Value result : op->getResults()) {
      if (outputs.count(result)) continue;
      for (Operation *user : result.getUsers())
        if (!region.count(user) && !shapeUse(user))
          return reject("HLO intermediate escapes the kernel region");
    }
  }
  SmallVector<Attribute> members;
  for (const auto &[id, op] : operations)
    if (region.count(op)) members.push_back(b.getStringAttr(id));
  auto function = (*outputs.begin()).getParentBlock()->getParentOp();
  kernel->setAttr(
      "annc.hlo_region",
      b.getDictionaryAttr(
          {b.getNamedAttr("version", b.getI64IntegerAttr(1)),
           b.getNamedAttr("function",
                          cast<func::FuncOp>(function).getSymNameAttr()),
           b.getNamedAttr("operations", b.getArrayAttr(members)),
           b.getNamedAttr("arguments", b.getStrArrayAttr(args)),
           b.getNamedAttr("outputs", b.getStrArrayAttr(results))}));
  module.getBody()->push_back(kernel->clone());
  for (Operation *op : region)
    if (computation(op)) claimed.insert(op);
  return true;
}
}  // namespace annc::stablehlo
