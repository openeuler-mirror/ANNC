#include "Builder/MLIROpBuilder.h"
#include "Builder/Transformers.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <sstream>

#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
namespace annc {

namespace {

std::string tfAttrValueToString(const NodeInfo::TfAttrValue& value) {
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
          return v;
        } else if constexpr (std::is_same_v<T, bool>) {
          return v ? "true" : "false";
        } else if constexpr (std::is_arithmetic_v<T>) {
          return std::to_string(v);
        } else {
          std::ostringstream os;
          for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) os << ",";
            if constexpr (std::is_same_v<typename T::value_type, bool>) {
              os << (v[i] ? "true" : "false");
            } else {
              os << v[i];
            }
          }
          return os.str();
        }
      },
      value);
}

// Converts a TF attr value to the existing MLIR attr type `target` (typed path
// for Declared attr mappings; null when unrepresentable).
Attribute tfAttrToMlirAttr(OpBuilder& builder, const NodeInfo& node,
                           StringRef tfName, Attribute target) {
  if (auto iAttr = dyn_cast<IntegerAttr>(target)) {
    if (int64_t v; node.getAttr(tfName.str(), v))
      return IntegerAttr::get(iAttr.getType(), v);
    if (bool b; node.getAttr(tfName.str(), b))
      return IntegerAttr::get(iAttr.getType(), b ? 1 : 0);
    if (double d; node.getAttr(tfName.str(), d))
      return IntegerAttr::get(iAttr.getType(), static_cast<int64_t>(d));
    return Attribute();
  }
  if (auto fAttr = dyn_cast<FloatAttr>(target)) {
    double d;
    if (node.getAttr(tfName.str(), d))
      return FloatAttr::get(fAttr.getType(), d);
    if (int64_t v; node.getAttr(tfName.str(), v))
      return FloatAttr::get(fAttr.getType(), static_cast<double>(v));
    return Attribute();
  }
  if (auto bAttr = dyn_cast<BoolAttr>(target)) {
    if (bool b; node.getAttr(tfName.str(), b))
      return BoolAttr::get(builder.getContext(), b);
    if (int64_t v; node.getAttr(tfName.str(), v))
      return BoolAttr::get(builder.getContext(), v != 0);
    return Attribute();
  }
  if (auto sAttr = dyn_cast<StringAttr>(target)) {
    if (std::string s; node.getAttr(tfName.str(), s))
      return StringAttr::get(builder.getContext(), s);
    if (int64_t v; node.getAttr(tfName.str(), v))
      return StringAttr::get(builder.getContext(), std::to_string(v));
    return Attribute();
  }
  if (auto denseI64 = dyn_cast<DenseI64ArrayAttr>(target)) {
    (void)denseI64;
    if (std::vector<int64_t> v; node.getAttr(tfName.str(), v))
      return DenseI64ArrayAttr::get(builder.getContext(), v);
    return Attribute();
  }
  if (auto arrAttr = dyn_cast<ArrayAttr>(target)) {
    if (arrAttr.size() == 0) return Attribute();
    Attribute first = arrAttr[0];
    if (dyn_cast<IntegerAttr>(first)) {
      if (std::vector<int64_t> v; node.getAttr(tfName.str(), v)) {
        auto ty = cast<IntegerAttr>(first).getType();
        SmallVector<Attribute> elems;
        elems.reserve(v.size());
        for (auto x : v) elems.push_back(IntegerAttr::get(ty, x));
        return ArrayAttr::get(builder.getContext(), elems);
      }
    } else if (dyn_cast<FloatAttr>(first)) {
      if (std::vector<double> v; node.getAttr(tfName.str(), v)) {
        auto ty = cast<FloatAttr>(first).getType();
        SmallVector<Attribute> elems;
        elems.reserve(v.size());
        for (auto x : v) elems.push_back(FloatAttr::get(ty, x));
        return ArrayAttr::get(builder.getContext(), elems);
      }
    } else if (dyn_cast<BoolAttr>(first)) {
      if (std::vector<bool> v; node.getAttr(tfName.str(), v)) {
        SmallVector<Attribute> elems;
        elems.reserve(v.size());
        for (auto x : v)
          elems.push_back(BoolAttr::get(builder.getContext(), x));
        return ArrayAttr::get(builder.getContext(), elems);
      }
    } else if (dyn_cast<StringAttr>(first)) {
      if (std::vector<std::string> v; node.getAttr(tfName.str(), v)) {
        SmallVector<Attribute> elems;
        elems.reserve(v.size());
        for (const auto& x : v)
          elems.push_back(StringAttr::get(builder.getContext(), x));
        return ArrayAttr::get(builder.getContext(), elems);
      }
    }
    return Attribute();
  }
  return Attribute();
}

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

#include "OpSpecTable.inc"

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

llvm::ArrayRef<OpSpec> MLIRBuilder::getAllOpSpecs() {
  return getOpSpecStorage();
}

Location OpContext::loc(llvm::StringRef name) const {
  return annc::getLoc(builder_.getContext(), name.str());
}

LogicalResult OpContext::emitError(const NodeInfo& node,
                                   llvm::StringRef message) {
  mlir::emitError(annc::getLoc(builder_.getContext(), node.name))
      << "building node '" << node.name << "' (" << node.op_type
      << "): " << message;
  return failure();
}

bool decodeIntConstValues(const NodeInfo* cnode, std::vector<int64_t>& out) {
  if (!cnode || cnode->outputs.empty() || cnode->raw_data.empty()) return false;
  const std::string& dtype = cnode->outputs[0].dtype;
  const std::vector<uint8_t>& decoded = cnode->raw_data;
  if (dtype == "int32") {
    size_t n = decoded.size() / sizeof(int32_t);
    const int32_t* d = reinterpret_cast<const int32_t*>(decoded.data());
    for (size_t i = 0; i < n; ++i) out.push_back(static_cast<int64_t>(d[i]));
    return true;
  }
  if (dtype == "int64") {
    size_t n = decoded.size() / sizeof(int64_t);
    const int64_t* d = reinterpret_cast<const int64_t*>(decoded.data());
    for (size_t i = 0; i < n; ++i) out.push_back(d[i]);
    return true;
  }
  return false;
}

bool MLIRBuilder::isSupportedOp(llvm::StringRef opType) {
  return getOpSpecTable().contains(opType);
}

const OpSpec* MLIRBuilder::lookupSpec(llvm::StringRef opType) const {
  const auto& table = getOpSpecTable();
  auto it = table.find(opType);
  return it == table.end() ? nullptr : it->second;
}

bool MLIRBuilder::buildFromNodes(const std::vector<NodeInfo>& nodes) {
  failed_ = false;
  auto unknownLoc = UnknownLoc::get(module_.getContext());
  mainFunc_ = builder_.create<func::FuncOp>(unknownLoc, "main",
                                            builder_.getFunctionType({}, {}));
  module_.push_back(mainFunc_);

  auto entryBlock = mainFunc_.addEntryBlock();
  builder_.setInsertionPointToStart(entryBlock);

  // Name -> NodeInfo lookup so transformers can resolve constant inputs.
  nodesByName_.clear();
  for (const auto& node : nodes) nodesByName_[node.name] = &node;
  std::vector<NodeInfo> inputNodes;
  std::vector<NodeInfo> outputNodes;
  std::vector<NodeInfo> computeNodes;
  for (const auto& node : nodes) {
    if (node.isInputNode)
      inputNodes.push_back(node);
    else if (node.isOutputNode)
      outputNodes.push_back(node);
    else
      computeNodes.push_back(node);
  }

  std::vector<Value> inputs = addGraphInputs(inputNodes);

  for (const auto& node : computeNodes) {
    // addNode 内部已通过 emitNodeError 输出带节点名的 MLIR 诊断，此处只传播失败。
    if (failed(addNode(node))) return false;
  }

  for (const auto& node : outputNodes) {
    if (failed(addNode(node))) return false;
  }

  std::vector<Value> outputs = addGraphOutputs(outputNodes);
  if (failed_) return false;

  llvm::SmallVector<Type> inputTypes{};
  std::for_each(inputs.begin(), inputs.end(),
                [&](Value val) { inputTypes.push_back(val.getType()); });
  llvm::SmallVector<Type> outputTypes{};
  std::for_each(outputs.begin(), outputs.end(),
                [&](Value val) { outputTypes.push_back(val.getType()); });
  mainFunc_.setType(builder_.getFunctionType(inputTypes, outputTypes));
  builder_.create<func::ReturnOp>(unknownLoc, outputs);
  if (failed(module_.verify())) {
    mlir::emitError(module_.getLoc())
        << "builder produced an invalid module (see verifier diagnostics "
           "above)";
    return false;
  }
  return true;
}

std::vector<Value> MLIRBuilder::addGraphInputs(
    const std::vector<NodeInfo>& inputNodes) {
  std::vector<Value> inputValues;
  for (const auto& node : inputNodes) {
    if (node.outputs.empty()) continue;
    const std::string& name = node.name;
    auto type = getTensorType(node, 0);
    if (failed(type)) {
      // getTensorType 内部已打印具体原因；这里补一层节点定位。
      mlir::emitError(annc::getLoc(builder_.getContext(), name))
          << "failed to type input node '" << name << "'";
      failed_ = true;
      continue;
    }
    Block* entryBlock = &mainFunc_.getBody().back();
    auto argVal = entryBlock->addArgument(
        *type, annc::getLoc(builder_.getContext(), name));
    inputValues.push_back(argVal);
    tensorValues_[name] = argVal;
  }
  return inputValues;
}

std::vector<Value> MLIRBuilder::addGraphOutputs(
    const std::vector<NodeInfo>& outputNodes) {
  std::vector<Value> outputValues;
  for (const auto& node : outputNodes) {
    const std::string& out = node.name;
    Value val = resolveValue(out);
    if (val == nullptr) {
      mlir::emitError(annc::getLoc(builder_.getContext(), out))
          << "unknown output value: " << out;
      failed_ = true;
      return {};
    }
    outputValues.emplace_back(val);
  }
  return outputValues;
}

Value MLIRBuilder::resolveValue(llvm::StringRef name) const {
  auto it = tensorValues_.find(name.str());
  if (it != tensorValues_.end()) return it->second;
  if (name.ends_with(":0")) {
    auto it0 = tensorValues_.find(name.drop_back(2).str());
    if (it0 != tensorValues_.end()) return it0->second;
  }
  return Value();
}

LogicalResult MLIRBuilder::emitNodeError(const NodeInfo& node,
                                         const std::string& message) {
  mlir::emitError(annc::getLoc(builder_.getContext(), node.name))
      << "building node '" << node.name << "' (" << node.op_type
      << "): " << message;
  return failure();
}

LogicalResult MLIRBuilder::addNode(const NodeInfo& node) {
  const std::string& type = node.op_type;

  const OpSpec* spec = lookupSpec(type);
  if (spec == nullptr) return buildOpaqueOp(node);

  if (spec->kind == OpKind::Constant) return buildConstantNode(node);

  if (node.inputs.size() < spec->minInputs ||
      node.inputs.size() > spec->maxInputs)
    return emitNodeError(
        node,
        "input count " + std::to_string(node.inputs.size()) +
            " out of range [" + std::to_string(spec->minInputs) + ", " +
            (spec->maxInputs == kUnbounded ? "unbounded"
                                           : std::to_string(spec->maxInputs)) +
            "]");
  if (spec->numOutputs != kVariableOutputs &&
      node.outputs.size() != spec->numOutputs)
    return emitNodeError(
        node, "output count " + std::to_string(node.outputs.size()) +
                  " != declared " + std::to_string(spec->numOutputs));

  SmallVector<Type> outs;
  outs.reserve(node.outputs.size());
  for (unsigned i = 0; i < node.outputs.size(); ++i) {
    auto type = getTensorType(node, i);
    if (failed(type)) return failure();
    outs.push_back(*type);
  }

  SmallVector<Value> ins;
  ins.reserve(node.inputs.size());
  for (const auto& input : node.inputs) {
    Value val = resolveValue(input);
    if (val == nullptr)
      return emitNodeError(node, "unknown input tensor '" + input + "'");
    ins.push_back(val);
  }

  Operation* createdOp = nullptr;
  if (spec->transformer != nullptr) {
    OpContext ctx(builder_, tensorValues_, nodesByName_);
    if (failed(spec->transformer(node, outs, ins, ctx))) return failure();
    auto it = tensorValues_.find(node.outputs[0].name);
    if (it != tensorValues_.end() && it->second) {
      createdOp = it->second.getDefiningOp();
    }
  } else {
    auto op = buildGenericOp(*spec, node, outs, ins);
    if (failed(op)) return failure();
    createdOp = *op;
  }

  if (createdOp != nullptr) {
    if (failed(applyAttrMappings(createdOp, node, *spec))) return failure();
    attachMetadata(createdOp, node, *spec);
  }
  return success();
}

FailureOr<Operation*> MLIRBuilder::buildGenericOp(const OpSpec& spec,
                                                  const NodeInfo& node,
                                                  ArrayRef<Type> outs,
                                                  ArrayRef<Value> ins) {
  OperationState state(annc::getLoc(builder_.getContext(), node.name),
                       spec.atirOp);
  if (spec.needsBuffer) {
    auto buffer = builder_.create<atir::BufferOp>(state.location, outs[0]);
    state.operands.push_back(buffer.getResult());
  }
  state.operands.append(ins.begin(), ins.end());
  state.types.append(outs.begin(), outs.end());
  for (const auto& def : spec.attrDefaults)
    state.addAttribute(def.name, def.build(builder_));
  Operation* op = builder_.create(state);
  for (unsigned i = 0; i < node.outputs.size(); ++i)
    tensorValues_[node.outputs[i].name] = op->getResult(i);
  return op;
}

// Fallback for TF ops with no OpSpec row: the graph still converts. The node
// becomes an atir.opaque placeholder that passes inputs/outputs through
// structurally (types/shapes from graph facts). A warning flags the op for
// later per-op addition; if no fusion pattern absorbs it, the lowering pass
// reports it instead of silently dropping it.
LogicalResult MLIRBuilder::buildOpaqueOp(const NodeInfo& node) {
  mlir::emitWarning(annc::getLoc(builder_.getContext(), node.name))
      << "unsupported op '" << node.op_type << "' at node '" << node.name
      << "' lowered to atir.opaque: no ATIR semantics yet, expected to be "
         "absorbed by fusion";

  SmallVector<Type> outs;
  outs.reserve(node.outputs.size());
  for (unsigned i = 0; i < node.outputs.size(); ++i) {
    auto type = getTensorType(node, i);
    if (failed(type)) return failure();
    outs.push_back(*type);
  }

  SmallVector<Value> ins;
  ins.reserve(node.inputs.size());
  for (const auto& input : node.inputs) {
    Value val = resolveValue(input);
    if (val == nullptr)
      return emitNodeError(node, "unknown input tensor '" + input + "'");
    ins.push_back(val);
  }

  OperationState state(annc::getLoc(builder_.getContext(), node.name),
                       "atir.opaque");
  state.operands.append(ins.begin(), ins.end());
  state.types.append(outs.begin(), outs.end());
  state.addAttribute(builder_.getStringAttr("opType"),
                     builder_.getStringAttr(node.op_type));
  // Keep the raw TF attrs in metadata for converter recovery.
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("tf.name", builder_.getStringAttr(node.name)));
  attrs.push_back(
      builder_.getNamedAttr("tf.op", builder_.getStringAttr(node.op_type)));
  for (const auto& [key, value] : node.tf_attrs) {
    if (key == "tf.name" || key == "tf.op") continue;
    attrs.push_back(builder_.getNamedAttr(key, builder_.getStringAttr(value)));
  }
  for (const auto& [key, value] : node.attrs) {
    if (key == "tf.name" || key == "tf.op" || key.empty() || key[0] == '_')
      continue;
    attrs.push_back(builder_.getNamedAttr(
        key, builder_.getStringAttr(tfAttrValueToString(value))));
  }
  state.addAttribute(builder_.getStringAttr("metadata"),
                     DictionaryAttr::get(builder_.getContext(), attrs));

  Operation* op = builder_.create(state);
  for (unsigned i = 0; i < node.outputs.size(); ++i)
    tensorValues_[node.outputs[i].name] = op->getResult(i);
  return success();
}

LogicalResult MLIRBuilder::applyAttrMappings(Operation* op,
                                             const NodeInfo& node,
                                             const OpSpec& spec) {
  for (const auto& mapping : spec.attrMappings) {
    if (mapping.kind == AttrKind::Transform) {
      Attribute attr = mapping.transform(builder_, node, mapping.atirAttr);
      if (attr) op->setAttr(mapping.atirAttr, attr);
      continue;
    }
    if (mapping.kind == AttrKind::DocumentedDrop)
      continue;  // kept in metadata only
    Attribute existing = op->getAttr(mapping.atirAttr);
    if (!existing) continue;
    Attribute attr = tfAttrToMlirAttr(builder_, node, mapping.tfAttr, existing);
    if (attr) op->setAttr(mapping.atirAttr, attr);
  }
  return success();
}

void MLIRBuilder::attachMetadata(Operation* op, const NodeInfo& node,
                                 const OpSpec& spec) {
  // Framework-level attrs (type params, tensor payload) and transformer-consumed
  // semantics (read via node.getAttr, not attrMappings) never map to ATIR attrs.
  static const llvm::StringLiteral kHarmlessTfAttrs[] = {
      "T", "Tidx", "Tparams", "Tout", "DstT", "out_type", "output_type",
      "Tpaddings", "Tshape", "Tperm", "Tindices", "Tsegment_ids",
      "Timage", "Tfilter", "Tkey", "Tvalue", "Tsplits", "Taxis", "Tdim",
      "N", "dtype", "value", "shape", "device",
      // TF variable resource naming attrs: only affect resource-container
      // addressing at TF runtime, never tensor computation. Variables are
      // modeled as atir.variable in the inference path, so these are inert.
      "container", "shared_name",
      "rhs_format", "num_split", "batch_dims", "num_segments", "num_buckets",
      "keepdims", "keep_dims", "sorted", "perm", "axis", "squeeze_dims",
      "begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask",
      "shrink_axis_mask",
      "transpose_a", "transpose_b",
      // BatchMatMulV2 adjoint flags: atir.BatchMatMul has no transpose
      // support; only the default (false,false) is harmless. A non-default
      // model silently loses the transpose — flag for future op extension.
      "adj_x", "adj_y",
      // SparseTensorDenseMatMul adjoint flags: consumed by the transformer and
      // encoded into the ATIR op attrs (adjointA/adjointB).
      "adjoint_a", "adjoint_b",
      // DynamicPartition num_partitions: consumed indirectly by the
      // transformer via the resolved output count.
      "num_partitions",
      // String-op semantics consumed by their transformers (kept in metadata
      // and encoded into the ATIR op attrs, never applied generically).
      "pattern", "rewrite", "replace_global", "skip_empty",
      // SparseToDense validate_indices: ATIR assumes valid indices (semantic
      // superset of the default validate=true; identical when indices valid).
      "validate_indices",
      // TF validation flag on Equal/NotEqual: false broadcasts instead of
      // erroring; atir.Compare always broadcasts, a superset, so no data loss.
      "incompatible_shape_error",
      // Cast float->int rounding flag: verified on TF 2.15 CPU that
      // Truncate=true/false both truncate (2.5->2, 3.5->3, -2.5->-2), matching
      // ATIR's static_cast conversion. No behavior difference to preserve.
      "Truncate",
  };
  auto isHarmless = [&](llvm::StringRef key) {
    for (const auto& s : kHarmlessTfAttrs)
      if (s == key) return true;
    return false;
  };
  auto isDeclared = [&](llvm::StringRef key) {
    for (const auto& m : spec.attrMappings)
      if (m.tfAttr == key) return true;
    return false;
  };

  SmallVector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("tf.name", builder_.getStringAttr(node.name)));
  attrs.push_back(
      builder_.getNamedAttr("tf.op", builder_.getStringAttr(node.op_type)));
  for (const auto& [key, value] : node.tf_attrs) {
    if (key == "tf.name" || key == "tf.op") continue;
    attrs.push_back(builder_.getNamedAttr(key, builder_.getStringAttr(value)));
  }
  for (const auto& [key, value] : node.attrs) {
    if (key == "tf.name" || key == "tf.op") continue;
    if (!isDeclared(key) && !isHarmless(key) && !key.empty() &&
        key[0] != '_') {
      mlir::emitWarning(annc::getLoc(builder_.getContext(), node.name))
          << "TF attribute '" << key << "' of node '" << node.name
          << "' is not mapped to an ATIR attribute (kept in metadata only)";
    }
    attrs.push_back(builder_.getNamedAttr(
        key, builder_.getStringAttr(tfAttrValueToString(value))));
  }
  op->setAttr("metadata", DictionaryAttr::get(builder_.getContext(), attrs));
}

FailureOr<atir::TensorType> MLIRBuilder::getTensorType(const NodeInfo& node,
                                                       unsigned outIdx) {
  const OutputInfo& out = node.outputs[outIdx];
  std::vector<int64_t> tensorShape = out.shape;
  for (size_t i = 0; i < tensorShape.size(); ++i)
    if (tensorShape[i] == -1) tensorShape[i] = ShapedType::kDynamic;

  auto dtype = parseDType(out.dtype);
  if (!dtype) {
    llvm::errs() << "Error: unknown dtype '" << out.dtype << "' for node '"
                 << node.name << "'\n";
    return failure();
  }
  auto name = builder_.getStringAttr(out.name);
  mlir::Attribute encoding;
  switch (*dtype) {
    case DType::F32:
      return atir::TensorType::get(tensorShape, builder_.getF32Type(), name,
                                   encoding);
    case DType::F64:
      return atir::TensorType::get(tensorShape, builder_.getF64Type(), name,
                                   encoding);
    case DType::F16:
      return atir::TensorType::get(tensorShape, builder_.getF16Type(), name,
                                   encoding);
    case DType::BF16:
      return atir::TensorType::get(tensorShape,
                                   BFloat16Type::get(builder_.getContext()),
                                   name, encoding);
    case DType::U8:
      return atir::TensorType::get(tensorShape, builder_.getIntegerType(8),
                                   name, encoding);
    case DType::U16:
      return atir::TensorType::get(tensorShape, builder_.getIntegerType(16),
                                   name, encoding);
    case DType::U32:
      return atir::TensorType::get(tensorShape, builder_.getIntegerType(32),
                                   name, encoding);
    case DType::U64:
      return atir::TensorType::get(tensorShape, builder_.getIntegerType(64),
                                   name, encoding);
    case DType::I8:
      return atir::TensorType::get(tensorShape, builder_.getI8Type(), name,
                                   encoding);
    case DType::I16:
      return atir::TensorType::get(tensorShape, builder_.getI16Type(), name,
                                   encoding);
    case DType::I32:
      return atir::TensorType::get(tensorShape, builder_.getI32Type(), name,
                                   encoding);
    case DType::I64:
      return atir::TensorType::get(tensorShape, builder_.getI64Type(), name,
                                   encoding);
    case DType::Bool:
      encoding = builder_.getStringAttr("bool");
      return atir::TensorType::get(tensorShape, builder_.getI32Type(), name,
                                   encoding);
    case DType::String:
      // No native string type: carry real values in ComplexType<f32> with an
      // encoding="string" marker (consistent with constants and the printer).
      return atir::TensorType::get(tensorShape,
                                   ComplexType::get(builder_.getF32Type()),
                                   name, builder_.getStringAttr("string"));
    case DType::Complex64:
      return atir::TensorType::get(
          tensorShape, ComplexType::get(builder_.getF32Type()), name, encoding);
    case DType::Complex128:
      return atir::TensorType::get(
          tensorShape, ComplexType::get(builder_.getF64Type()), name, encoding);
  }
  llvm_unreachable("unhandled DType");
}

LogicalResult MLIRBuilder::buildConstantNode(const NodeInfo& node) {
  const std::string& name = node.name;
  if (node.outputs.empty())
    return emitNodeError(node, "constant node has no output info");
  const OutputInfo& out = node.outputs[0];
  const std::string& dtype = out.dtype;
  const std::vector<int64_t>& shape = out.shape;

  std::vector<uint8_t> decoded = node.raw_data;

  auto requireBytes = [&](size_t elemSize) -> LogicalResult {
    size_t elemCount = 1;
    for (int64_t d : shape) {
      if (d < 0) continue;
      elemCount *= static_cast<size_t>(d);
    }
    if (elemCount == 0 && decoded.empty()) return success();
    if (decoded.empty() || decoded.size() % elemSize != 0)
      return emitNodeError(
          node, "constant raw_data size mismatch for dtype '" + dtype + "'");
    return success();
  };

  auto emitConstant = [&](DenseElementsAttr elems, Type eltType,
                          Attribute encoding = {}) -> LogicalResult {
    atir::TensorType tensorTy =
        atir::TensorType::get(shape, eltType, builder_.getStringAttr(name),
                              encoding, {}, {}, {}, {}, {}, {}, elems);
    auto gOp = builder_.create<atir::ConstantOp>(
        annc::getLoc(builder_.getContext(), name), tensorTy,
        builder_.getStringAttr(name), builder_.getStringAttr("public"));
    tensorValues_[name] = gOp.getResult();
    return success();
  };

  DenseElementsAttr elems;
  auto d = parseDType(dtype);
  if (!d) return emitNodeError(node, "unsupported dtype '" + dtype + "'");
  switch (*d) {
    case DType::F32: {
      if (failed(requireBytes(sizeof(float)))) return failure();
      const float* data = reinterpret_cast<const float*>(decoded.data());
      size_t n = decoded.size() / sizeof(float);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getF32Type()),
          ArrayRef<float>(data, n));
      return emitConstant(elems, builder_.getF32Type());
    }
    case DType::F64: {
      if (failed(requireBytes(sizeof(double)))) return failure();
      const double* data = reinterpret_cast<const double*>(decoded.data());
      size_t n = decoded.size() / sizeof(double);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getF64Type()),
          ArrayRef<double>(data, n));
      return emitConstant(elems, builder_.getF64Type());
    }
    case DType::F16: {
      if (failed(requireBytes(sizeof(uint16_t)))) return failure();
      auto ranked = RankedTensorType::get(shape, builder_.getF16Type());
      elems = DenseElementsAttr::getFromRawBuffer(
          ranked, ArrayRef<char>(reinterpret_cast<const char*>(decoded.data()),
                                 decoded.size()));
      return emitConstant(elems, builder_.getF16Type());
    }
    case DType::BF16: {
      if (failed(requireBytes(sizeof(uint16_t)))) return failure();
      auto bf16Type = BFloat16Type::get(builder_.getContext());
      auto ranked = RankedTensorType::get(shape, bf16Type);
      elems = DenseElementsAttr::getFromRawBuffer(
          ranked, ArrayRef<char>(reinterpret_cast<const char*>(decoded.data()),
                                 decoded.size()));
      return emitConstant(elems, bf16Type);
    }
    case DType::I64: {
      if (failed(requireBytes(sizeof(int64_t)))) return failure();
      const int64_t* data = reinterpret_cast<const int64_t*>(decoded.data());
      size_t n = decoded.size() / sizeof(int64_t);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getI64Type()),
          ArrayRef<int64_t>(data, n));
      return emitConstant(elems, builder_.getI64Type());
    }
    case DType::I32: {
      if (failed(requireBytes(sizeof(int32_t)))) return failure();
      const int32_t* data = reinterpret_cast<const int32_t*>(decoded.data());
      size_t n = decoded.size() / sizeof(int32_t);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getI32Type()),
          ArrayRef<int32_t>(data, n));
      return emitConstant(elems, builder_.getI32Type());
    }
    case DType::I16: {
      if (failed(requireBytes(sizeof(int16_t)))) return failure();
      const int16_t* data = reinterpret_cast<const int16_t*>(decoded.data());
      size_t n = decoded.size() / sizeof(int16_t);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getI16Type()),
          ArrayRef<int16_t>(data, n));
      return emitConstant(elems, builder_.getI16Type());
    }
    case DType::I8: {
      if (failed(requireBytes(sizeof(int8_t)))) return failure();
      const int8_t* data = reinterpret_cast<const int8_t*>(decoded.data());
      size_t n = decoded.size() / sizeof(int8_t);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getI8Type()),
          ArrayRef<int8_t>(data, n));
      return emitConstant(elems, builder_.getI8Type());
    }
    case DType::U8: {
      if (failed(requireBytes(sizeof(uint8_t)))) return failure();
      const uint8_t* data = reinterpret_cast<const uint8_t*>(decoded.data());
      size_t n = decoded.size() / sizeof(uint8_t);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getIntegerType(8)),
          ArrayRef<uint8_t>(data, n));
      return emitConstant(elems, builder_.getIntegerType(8));
    }
    case DType::U16: {
      if (failed(requireBytes(sizeof(uint16_t)))) return failure();
      const uint16_t* data = reinterpret_cast<const uint16_t*>(decoded.data());
      size_t n = decoded.size() / sizeof(uint16_t);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getIntegerType(16)),
          ArrayRef<uint16_t>(data, n));
      return emitConstant(elems, builder_.getIntegerType(16));
    }
    case DType::U32: {
      if (failed(requireBytes(sizeof(uint32_t)))) return failure();
      const uint32_t* data = reinterpret_cast<const uint32_t*>(decoded.data());
      size_t n = decoded.size() / sizeof(uint32_t);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getIntegerType(32)),
          ArrayRef<uint32_t>(data, n));
      return emitConstant(elems, builder_.getIntegerType(32));
    }
    case DType::U64: {
      if (failed(requireBytes(sizeof(uint64_t)))) return failure();
      const uint64_t* data = reinterpret_cast<const uint64_t*>(decoded.data());
      size_t n = decoded.size() / sizeof(uint64_t);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getIntegerType(64)),
          ArrayRef<uint64_t>(data, n));
      return emitConstant(elems, builder_.getIntegerType(64));
    }
    case DType::Bool: {
      if (failed(requireBytes(sizeof(uint8_t)))) return failure();
      std::vector<int32_t> asI32;
      asI32.reserve(decoded.size());
      for (uint8_t b : decoded) asI32.push_back(b ? 1 : 0);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, builder_.getI32Type()),
          ArrayRef<int32_t>(asI32));
      return emitConstant(elems, builder_.getI32Type(),
                          builder_.getStringAttr("bool"));
    }
    case DType::Complex64: {
      if (failed(requireBytes(2 * sizeof(float)))) return failure();
      size_t n = decoded.size() / (2 * sizeof(float));
      const float* p = reinterpret_cast<const float*>(decoded.data());
      std::vector<std::complex<float>> vals(n);
      for (size_t i = 0; i < n; ++i)
        vals[i] = std::complex<float>(p[2 * i], p[2 * i + 1]);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, ComplexType::get(builder_.getF32Type())),
          ArrayRef<std::complex<float>>(vals));
      return emitConstant(elems, ComplexType::get(builder_.getF32Type()));
    }
    case DType::Complex128: {
      if (failed(requireBytes(2 * sizeof(double)))) return failure();
      size_t n = decoded.size() / (2 * sizeof(double));
      const double* p = reinterpret_cast<const double*>(decoded.data());
      std::vector<std::complex<double>> vals(n);
      for (size_t i = 0; i < n; ++i)
        vals[i] = std::complex<double>(p[2 * i], p[2 * i + 1]);
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, ComplexType::get(builder_.getF64Type())),
          ArrayRef<std::complex<double>>(vals));
      return emitConstant(elems, ComplexType::get(builder_.getF64Type()));
    }
    case DType::String: {
      size_t elemCount = 1;
      for (int64_t d : shape) {
        if (d < 0) continue;
        elemCount *= static_cast<size_t>(d);
      }
      if (node.string_values.size() != elemCount)
        return emitNodeError(
            node, "string constant value count mismatch: expected " +
                      std::to_string(elemCount) + ", got " +
                      std::to_string(node.string_values.size()));
      SmallVector<StringRef> values;
      values.reserve(node.string_values.size());
      for (const std::string& v : node.string_values)
        values.emplace_back(v.data(), v.size());
      auto stringType = ComplexType::get(builder_.getF32Type());
      elems = DenseElementsAttr::get(
          RankedTensorType::get(shape, stringType), ArrayRef<StringRef>(values));
      return emitConstant(elems, stringType, builder_.getStringAttr("string"));
    }
  }
  llvm_unreachable("unhandled DType");
}

}  // namespace annc
