#include "tf_mlir_builder.h"

#include "Builder/Transformers.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tf_transformers.h"

using namespace mlir;
namespace annc::tf2atir {
namespace {

Attribute defBoolFalse(OpBuilder& b) { return b.getBoolAttr(false); }
Attribute defF32NegOne(OpBuilder& b) { return b.getF32FloatAttr(-1.0f); }
Attribute defI32Zero(OpBuilder& b) { return b.getI32IntegerAttr(0); }
Attribute defPredEQ(OpBuilder& b) { return b.getStringAttr("EQ"); }
Attribute defPredLT(OpBuilder& b) { return b.getStringAttr("LT"); }
Attribute defPredNE(OpBuilder& b) { return b.getStringAttr("NE"); }
Attribute defPredGT(OpBuilder& b) { return b.getStringAttr("GT"); }
Attribute defPredGE(OpBuilder& b) { return b.getStringAttr("GE"); }
Attribute defPredLE(OpBuilder& b) { return b.getStringAttr("LE"); }

const AttrDefault kDefaultsAdd[] = {{"do_relu", defBoolFalse},
                                    {"relu_limit", defF32NegOne}};
const AttrDefault kDefaultsConcat[] = {{"axis", defI32Zero},
                                       {"do_relu", defBoolFalse},
                                       {"relu_limit", defF32NegOne},
                                       {"round_mode", defI32Zero}};
const AttrDefault kDefaultsRelu[] = {{"relu_limit", defF32NegOne}};
const AttrDefault kDefaultsReduceMean[] = {{"keep_dims", defBoolFalse}};
const AttrDefault kDefaultsPredEQ[] = {{"comparisonDirection", defPredEQ}};
const AttrDefault kDefaultsPredLT[] = {{"comparisonDirection", defPredLT}};
const AttrDefault kDefaultsPredNE[] = {{"comparisonDirection", defPredNE}};
const AttrDefault kDefaultsPredGT[] = {{"comparisonDirection", defPredGT}};
const AttrDefault kDefaultsPredGE[] = {{"comparisonDirection", defPredGE}};
const AttrDefault kDefaultsPredLE[] = {{"comparisonDirection", defPredLE}};

// Equal/NotEqual incompatible_shape_error: read as bool (TF default true) or
// int64, mapped onto atir.Compare's incompatibleShapeError attr.
Attribute transformIncompatibleShapeError(OpBuilder& b, const NodeInfo& node,
                                          StringRef atirAttr) {
  bool flag = true;
  if (node.getAttr("incompatible_shape_error", flag))
    return b.getBoolAttr(flag);
  if (int64_t v; node.getAttr("incompatible_shape_error", v))
    return b.getBoolAttr(v != 0);
  return b.getBoolAttr(true);
}

const AttrMapping kMappingKeepDims[] = {
    {"keep_dims", "keep_dims", AttrKind::Direct, nullptr}};
const AttrMapping kMappingIncompatibleShapeError[] = {
    {"incompatible_shape_error", "incompatibleShapeError", AttrKind::Transform,
     transformIncompatibleShapeError}};

}  // namespace

#include "tf_op_spec_table.inc"

namespace {

// Shared storage for the table (referenced by the name map and getAllOpSpecs).
const llvm::SmallVector<OpSpec, 0>& getOpSpecStorage() {
  static const llvm::SmallVector<OpSpec, 0> kSpecs = [] {
    llvm::SmallVector<OpSpec, 0> specs;
    populateOpSpecTable(specs);
    return specs;
  }();
  return kSpecs;
}

const llvm::StringMap<const OpSpec*>& getOpSpecTable() {
  static llvm::StringMap<const OpSpec*> table = [] {
    llvm::StringMap<const OpSpec*> map;
    for (const auto& spec : getOpSpecStorage())
      for (llvm::StringRef tfOp : spec.tfOps) map[tfOp] = &spec;
    return map;
  }();
  return table;
}

}  // namespace

llvm::ArrayRef<OpSpec> TfMLIRBuilder::getAllOpSpecs() {
  return getOpSpecStorage();
}

bool TfMLIRBuilder::isSupportedOp(llvm::StringRef opType) {
  return getOpSpecTable().contains(opType);
}

const OpSpec* TfMLIRBuilder::lookupSpec(llvm::StringRef opType) const {
  const auto& table = getOpSpecTable();
  auto it = table.find(opType);
  return it == table.end() ? nullptr : it->second;
}

std::optional<LogicalResult> TfMLIRBuilder::buildFrontendNode(
    const NodeInfo& node) {
  const std::string& type = node.op_type;
  if (type == "Switch" || type == "RefSwitch" || type == "Merge" ||
      type == "RefMerge" || type == "Enter" || type == "RefEnter" ||
      type == "Exit" || type == "RefExit" || type == "NextIteration" ||
      type == "RefNextIteration" || type == "LoopCond")
    return buildControlFlowMirrorNode(node);

  return std::nullopt;
}

LogicalResult TfMLIRBuilder::buildControlFlowMirrorNode(
    const NodeInfo& node) {
  SmallVector<Value> inputs;
  for (const std::string& inputName : node.inputs) {
    Value input = resolveValue(inputName);
    if (!input) {
      // A NextIteration -> Merge edge is a TensorFlow loop backedge. It is
      // preserved in tf.input.N metadata and intentionally omitted from the
      // acyclic MLIR operand list when its producer is not available yet.
      if (node.op_type == "Merge" || node.op_type == "RefMerge") continue;
      return emitNodeError(node, "unknown control-flow input tensor '" +
                                     inputName + "'");
    }
    inputs.push_back(input);
  }

  SmallVector<Type> outputs;
  for (unsigned i = 0; i < node.outputs.size(); ++i) {
    auto type = getTensorType(node, i);
    if (failed(type)) return failure();
    outputs.push_back(*type);
  }

  Location loc = annc::getLoc(builder_.getContext(), node.name);
  OperationState state(loc, "atir.opaque");
  if (node.op_type == "Switch" || node.op_type == "RefSwitch") {
    if (inputs.size() != 2 || outputs.size() != 2)
      return emitNodeError(node, "Switch requires two inputs and two outputs");
    state.name = OperationName("atir.switch", builder_.getContext());
  } else if (node.op_type == "Merge" || node.op_type == "RefMerge") {
    if (inputs.empty() || outputs.size() != 2)
      return emitNodeError(node,
                           "Merge requires at least one available input and two outputs");
    state.name = OperationName("atir.merge", builder_.getContext());
  } else if (node.op_type == "Enter" || node.op_type == "RefEnter") {
    state.name = OperationName("atir.enter", builder_.getContext());
  } else if (node.op_type == "Exit" || node.op_type == "RefExit") {
    state.name = OperationName("atir.exit", builder_.getContext());
  } else if (node.op_type == "NextIteration" ||
             node.op_type == "RefNextIteration") {
    state.name = OperationName("atir.next_iteration", builder_.getContext());
  } else {
    state.name = OperationName("atir.loop_cond", builder_.getContext());
  }
  state.operands.append(inputs.begin(), inputs.end());
  state.types.append(outputs.begin(), outputs.end());

  SmallVector<NamedAttribute> metadata;
  metadata.push_back(
      builder_.getNamedAttr("tf.name", builder_.getStringAttr(node.name)));
  metadata.push_back(builder_.getNamedAttr(
      "tf.op", builder_.getStringAttr(node.op_type)));
  for (const auto& [key, value] : node.tf_attrs) {
    if (key == "tf.name" || key == "tf.op") continue;
    metadata.push_back(
        builder_.getNamedAttr(key, builder_.getStringAttr(value)));
  }
  for (const auto& [key, value] : node.attrs) {
    if (key.empty() || key[0] == '_') continue;
    metadata.push_back(builder_.getNamedAttr(
        key, builder_.getStringAttr(tfAttrValueToString(value))));
  }
  state.addAttribute(builder_.getStringAttr("metadata"),
                     DictionaryAttr::get(builder_.getContext(), metadata));

  Operation* op = builder_.create(state);
  for (unsigned i = 0; i < node.outputs.size(); ++i)
    tensorValues_[node.outputs[i].name] = op->getResult(i);
  return success();
}

}  // namespace annc::tf2atir
