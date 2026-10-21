#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/Passes/Patterns/FusionBoundaryUtils.h"
#include "Dialect/Atir/Passes/Patterns/PatternRegistry.h"
#include "Dialect/Atir/TemplateFingerprint.h"
#include "Helper.h"
#include "Kernel/KernelPriorityResolver.h"
#include "Support/FusionSwitch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SHA256.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cstring>

using llvm::ArrayRef;
using llvm::SHA256;
using llvm::SmallPtrSet;
using llvm::SmallString;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;
using namespace mlir;
using namespace atir;

namespace {

constexpr StringLiteral kAotExecutionMode = "aot";
constexpr StringLiteral kJitExecutionMode = "jit";
static void setExecutionMode(func::FuncOp function,
                             PatternRewriter &rewriter, StringRef mode) {
  function->setAttr("annc.execution_mode", rewriter.getStringAttr(mode));
}

static void addExecutionModeMetadata(
    SmallVectorImpl<NamedAttribute> &metadata, PatternRewriter &rewriter,
    StringRef mode) {
  metadata.push_back(rewriter.getNamedAttr("execution_mode",
                                           rewriter.getStringAttr(mode)));
}

static std::string sanitizeName(std::string name) {
  for (char &c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
  }
  return name;
}

static std::string getTfName(Operation *op) {
  if (!op) return "";
  if (auto dict = op->getAttrOfType<DictionaryAttr>("metadata")) {
    if (auto attr = dyn_cast_or_null<StringAttr>(dict.get("tf.name"))) {
      return attr.str();
    }
  }
  std::string locName = annc::getLocName(op);
  if (!locName.empty()) return locName;
  if (op->getNumResults() > 0) {
    if (auto tensorType =
            dyn_cast<atir::TensorType>(op->getResult(0).getType())) {
      return tensorType.getValueOfName();
    }
  }
  return "";
}

static std::string getTfOp(Operation *op) {
  if (!op) return "";
  if (auto dict = op->getAttrOfType<DictionaryAttr>("metadata")) {
    if (auto attr = dyn_cast_or_null<StringAttr>(dict.get("tf.op"))) {
      return attr.str();
    }
  }
  return "";
}

// 站点成员名单（EP 运行时编译的站点报告用）：被这次融合吞并的图节点名。
// ONNX 前端把 ONNX 节点名写在 metadata.tf.name；Constant / Input 是例外
// （它们写的是张量名，对应 ORT 的 initializer / 图输入，不在节点表里），
// 故按 tf.op 排除。名单必须与消费方看到的图命名一致，否则消费方无法把
// 它映射回自己的节点、只能整站点回落。
static SmallVector<Attribute> collectMemberNodeNames(
    PatternRewriter &rewriter, ArrayRef<Operation *> ops) {
  SmallVector<Attribute> names;
  for (Operation *op : ops) {
    if (!op || isa<ConstantOp, BufferOp>(op)) continue;
    std::string kind = getTfOp(op);
    if (kind == "Constant" || kind == "Input") continue;
    std::string name = getTfName(op);
    if (name.empty()) continue;
    names.push_back(rewriter.getStringAttr(name));
  }
  return names;
}

static std::string getValueName(Value value) {
  if (auto tensorType = dyn_cast<atir::TensorType>(value.getType())) {
    std::string name = tensorType.getValueOfName();
    if (!name.empty()) return name;
  }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto loc = dyn_cast<NameLoc>(blockArg.getLoc()))
      return loc.getName().str();
    return "arg" + std::to_string(blockArg.getArgNumber());
  }
  if (Operation *op = value.getDefiningOp()) {
    return getTfName(op);
  }
  return "";
}

static std::string getFusionOutputName(Value value) {
  std::string name;
  if (auto result = dyn_cast<OpResult>(value)) {
    if (auto metadata =
            result.getOwner()->getAttrOfType<DictionaryAttr>("metadata")) {
      if (auto endpoints =
              dyn_cast_or_null<ArrayAttr>(metadata.get("tf.output_tensors"));
          endpoints && result.getResultNumber() < endpoints.size()) {
        if (auto endpoint =
                dyn_cast<StringAttr>(endpoints[result.getResultNumber()]);
            endpoint && !endpoint.empty()) {
          name = endpoint.str();
        }
      }
    }
    if (name.empty()) name = getValueName(value);
    if (!name.empty() && name.find(':') == std::string::npos)
      name += ":" + std::to_string(result.getResultNumber());
    return name;
  }
  return getValueName(value);
}

static std::string getStableMatMulKernelName(Operation *matmul) {
  std::string sig = getTfName(matmul);
  SHA256 sha;
  sha.update(sig);
  auto hash = sha.final();
  SmallString<16> hex;
  for (size_t i = 0; i < 4; ++i) hex += llvm::toHex(hash[i]);
  return ("fused_matmul_" + hex).str();
}

static std::string getStableMatMulFusionKernelName(Operation *matmul,
                                                   StringRef pattern) {
  std::string sig = getTfName(matmul);
  SHA256 sha;
  sha.update(sig);
  sha.update(pattern);
  auto hash = sha.final();
  SmallString<16> hex;
  for (size_t i = 0; i < 4; ++i) hex += llvm::toHex(hash[i]);
  return (Twine("fused_") + pattern + "_" + hex).str();
}

static std::string getStableFusionKernelName(StringRef prefix,
                                             StringRef pattern) {
  SHA256 sha;
  sha.update(prefix);
  sha.update(pattern);
  auto hash = sha.final();
  SmallString<16> hex;
  for (size_t i = 0; i < 4; ++i) hex += llvm::toHex(hash[i]);
  return (Twine("fused_") + pattern + "_" + hex).str();
}

static std::string uniquifySymbolName(ModuleOp module, StringRef baseName) {
  std::string unique = baseName.str();
  unsigned suffix = 1;
  while (module.lookupSymbol(unique)) {
    unique = (Twine(baseName) + "_" + Twine(suffix++)).str();
  }
  return unique;
}

static std::string uniquifyFusionName(ModuleOp module, StringRef baseName) {
  std::string unique = baseName.str();
  unsigned suffix = 1;
  bool changed = true;
  while (changed) {
    changed = false;
    module.walk([&](func::FuncOp funcOp) {
      if (changed) return;
      auto metadata = funcOp->getAttrOfType<DictionaryAttr>("fusion.metadata");
      if (!metadata) return;
      auto name = dyn_cast_or_null<StringAttr>(metadata.get("tf.name"));
      if (name && name.getValue() == unique) changed = true;
    });
    if (changed) unique = (Twine(baseName) + "_" + Twine(suffix++)).str();
  }
  return unique;
}

static ArrayAttr makeI64Array(MLIRContext *ctx, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  Builder b(ctx);
  for (int64_t value : values) attrs.push_back(b.getI64IntegerAttr(value));
  return ArrayAttr::get(ctx, attrs);
}

static int64_t getRank(Type type);
static SmallVector<int64_t> getShape(Type type);

static std::string getElementDTypeString(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type)) {
    if (floatType.isF16()) return "f16";
    if (floatType.isF32()) return "f32";
    if (floatType.isF64()) return "f64";
    if (floatType.isF80()) return "f80";
    if (floatType.isF128()) return "f128";
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    StringRef prefix = "i";
    if (intType.isSigned()) {
      prefix = "si";
    } else if (intType.isUnsigned()) {
      prefix = "ui";
    }
    return (Twine(prefix) + Twine(intType.getWidth())).str();
  }
  if (type.isIndex()) return "index";
  return "";
}

static std::string getDTypeString(Type type) {
  if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
    if (auto encoding =
            dyn_cast_or_null<StringAttr>(tensorType.getEncoding())) {
      if (!encoding.getValue().empty()) return encoding.str();
    }
    return getElementDTypeString(tensorType.getElementType());
  }
  return getElementDTypeString(type);
}

static DictionaryAttr makeFusionArgAttr(MLIRContext *ctx, StringRef role,
                                        StringRef tfName, Type type) {
  Builder builder(ctx);
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("role", builder.getStringAttr(role)));
  attrs.push_back(
      builder.getNamedAttr("tf_name", builder.getStringAttr(tfName)));
  attrs.push_back(
      builder.getNamedAttr("shape", makeI64Array(ctx, getShape(type))));
  attrs.push_back(
      builder.getNamedAttr("rank", builder.getI64IntegerAttr(getRank(type))));
  attrs.push_back(builder.getNamedAttr(
      "dtype", builder.getStringAttr(getDTypeString(type))));
  return DictionaryAttr::get(ctx, attrs);
}

struct FusionArgSpec {
  std::string role;
  std::string tfName;
  Type type;
};

static ArrayAttr makeFusionArgArray(MLIRContext *ctx,
                                    ArrayRef<FusionArgSpec> specs) {
  SmallVector<Attribute> attrs;
  for (const auto &spec : specs) {
    attrs.push_back(makeFusionArgAttr(ctx, spec.role, spec.tfName, spec.type));
  }
  return ArrayAttr::get(ctx, attrs);
}

static int64_t getRank(Type type) {
  if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
    return static_cast<int64_t>(tensorType.getShape().size());
  }
  return 0;
}

static SmallVector<int64_t> getShape(Type type) {
  if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
    return SmallVector<int64_t>(tensorType.getShape());
  }
  return {};
}

static bool hasMatchingBatchDim(Type inputType, Type outputType) {
  SmallVector<int64_t> inputShape = getShape(inputType);
  SmallVector<int64_t> outputShape = getShape(outputType);
  return !inputShape.empty() && !outputShape.empty() &&
         inputShape.front() == outputShape.front();
}

static bool hasCompatibleBiasDim(Type outputType, Type biasType) {
  SmallVector<int64_t> outputShape = getShape(outputType);
  SmallVector<int64_t> biasShape = getShape(biasType);
  if (outputShape.size() != 2 || biasShape.size() != 1) return false;

  int64_t outputCols = outputShape[1];
  int64_t biasLen = biasShape[0];
  if (outputCols < 0 || biasLen < 0) return true;
  return outputCols == biasLen;
}

static bool startsWith(StringRef value, StringRef prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

static bool hasTfNamePrefix(Operation *op, StringRef prefix) {
  return startsWith(getTfName(op), prefix);
}

static bool definesAny(Operation *op, ArrayRef<Value> values) {
  for (Value result : op->getResults()) {
    for (Value value : values) {
      if (result == value) return true;
    }
  }
  return false;
}

static void collectDefiningOpsPostOrder(Operation *op,
                                        ArrayRef<Value> boundaryValues,
                                        SmallPtrSetImpl<Operation *> &visited,
                                        SmallVectorImpl<Operation *> &ops) {
  if (!op || visited.contains(op) || isa<VariableOp>(op)) return;
  if (definesAny(op, boundaryValues)) return;
  visited.insert(op);

  for (Value operand : op->getOperands()) {
    if (llvm::is_contained(boundaryValues, operand)) continue;
    collectDefiningOpsPostOrder(operand.getDefiningOp(), boundaryValues,
                                visited, ops);
  }
  ops.push_back(op);
}

static bool isClosedKernelOpSet(ArrayRef<Operation *> ops,
                                ArrayRef<Value> boundaryValues) {
  SmallPtrSet<Operation *, 32> opSet(ops.begin(), ops.end());
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      if (llvm::is_contained(boundaryValues, operand)) continue;
      Operation *def = operand.getDefiningOp();
      if (!def || !opSet.contains(def)) return false;
    }
  }
  return true;
}

template <typename OpT>
static OpT findUserOf(Value value) {
  for (Operation *user : value.getUsers()) {
    if (auto op = dyn_cast<OpT>(user)) return op;
  }
  return nullptr;
}

static AddOp findUniqueMatMulAddUser(MatMulOp matmulOp, Value &bias) {
  AddOp found = nullptr;
  for (Operation *user : matmulOp.getResult().getUsers()) {
    auto addOp = dyn_cast<AddOp>(user);
    if (!addOp || addOp->getNumOperands() != 3) continue;

    Value candidateBias;
    if (addOp.getOperand(1) == matmulOp.getResult()) {
      candidateBias = addOp.getOperand(2);
    } else if (addOp.getOperand(2) == matmulOp.getResult()) {
      candidateBias = addOp.getOperand(1);
    } else {
      continue;
    }
    if (getRank(candidateBias.getType()) != 1 ||
        !hasCompatibleBiasDim(matmulOp.getResult().getType(),
                              candidateBias.getType())) {
      continue;
    }
    if (found) return nullptr;
    found = addOp;
    bias = candidateBias;
  }
  return found;
}

static ReluOp findUniqueAddReluUser(AddOp addOp) {
  ReluOp found = nullptr;
  for (Operation *user : addOp.getResult().getUsers()) {
    auto reluOp = dyn_cast<ReluOp>(user);
    if (!reluOp) continue;
    if (found) return nullptr;
    found = reluOp;
  }
  return found;
}

template <typename OpT>
static OpT findNamedOp(func::FuncOp func, StringRef name) {
  OpT found = nullptr;
  func.walk([&](OpT op) {
    if (!found && getTfName(op.getOperation()) == name) found = op;
  });
  return found;
}

static MatMulOp createMatMulBodyOp(PatternRewriter &rewriter, Location loc,
                                   Type resultType, Value output, Value lhs,
                                   Value rhs, MatMulOp source) {
  return rewriter.create<MatMulOp>(
      loc, resultType, output, lhs, rhs, ValueRange{},
      rewriter.getBoolAttr(false),
      rewriter.getBoolAttr(source.getRightTranspose()),
      rewriter.getBoolAttr(source.getLeftTranspose()),
      rewriter.getBoolAttr(source.getOutputTranspose()),
      rewriter.getBoolAttr(false), source.getReluLimitAttr(),
      source.getMStartAttr(), source.getNStartAttr(), source.getKStartAttr(),
      source.getMSizeAttr(), source.getNSizeAttr(), source.getKSizeAttr(),
      source->getAttrOfType<StringAttr>("rhs_format"));
}

static SmallVector<Value> collectExecutionV2Inputs(
    ArrayRef<Operation *> fusedOps) {
  SmallPtrSet<Operation *, 8> fusedSet(fusedOps.begin(), fusedOps.end());
  llvm::SetVector<Value> inputs;
  for (Operation *op : fusedOps) {
    for (Value operand : op->getOperands().drop_front()) {
      if (!fusedSet.contains(operand.getDefiningOp())) inputs.insert(operand);
    }
  }
  return SmallVector<Value>(inputs.begin(), inputs.end());
}

static func::FuncOp createExecutionV2KernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    ArrayRef<Operation *> fusedOps, ArrayRef<Value> boundaryInputs,
    ArrayRef<Value> boundaryOutputs) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {
      LLVM::LLVMPointerType::get(rewriter.getContext())};
  for (Value input : boundaryInputs) inputTypes.push_back(input.getType());
  SmallVector<Type> resultTypes;
  for (Value output : boundaryOutputs) resultTypes.push_back(output.getType());

  auto funcType = rewriter.getFunctionType(inputTypes, resultTypes);
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
  func->setAttr("fusion.pattern", rewriter.getStringAttr("matmul_add_relu"));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  IRMapping mapper;
  for (auto [index, input] : llvm::enumerate(boundaryInputs)) {
    mapper.map(input, entry->getArgument(index + 1));
  }
  for (Operation *op : fusedOps) {
    if (!isa<UniqueOp, ConstantOp, BufferOp>(op)) {
      Value outputBuffer = op->getOperand(0);
      if (!mapper.contains(outputBuffer)) {
        Value localBuffer =
            rewriter.create<BufferOp>(func.getLoc(), outputBuffer.getType());
        mapper.map(outputBuffer, localBuffer);
      }
    }
    rewriter.clone(*op, mapper);
  }

  SmallVector<Value> returns;
  for (Value output : boundaryOutputs) returns.push_back(mapper.lookup(output));
  rewriter.create<func::ReturnOp>(func.getLoc(), returns);
  return func;
}

static func::FuncOp createDnnEmbeddingHashBucketKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    Type dynamicInputType, Type weightType, Type outputType, int64_t numBuckets,
    ArrayRef<Operation *> kernelOps, Value dynamicInput, Value embeddingWeight,
    Value outputBuffer) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {dynamicInputType, weightType, outputType};
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern",
                rewriter.getStringAttr("dnn_embedding_hash_bucket"));
  func->setAttr("fusion.num_buckets", rewriter.getI64IntegerAttr(numBuckets));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);

  IRMapping mapper;
  mapper.map(dynamicInput, entry->getArgument(0));
  mapper.map(embeddingWeight, entry->getArgument(1));
  mapper.map(outputBuffer, entry->getArgument(2));

  for (Operation *op : kernelOps) {
    rewriter.clone(*op, mapper);
  }
  rewriter.create<func::ReturnOp>(func.getLoc());

  return func;
}

static func::FuncOp createKernelFunc(ModuleOp module, PatternRewriter &rewriter,
                                     StringRef kernelName, MatMulOp matmulOp) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {matmulOp.getLhs().getType(),
                                  matmulOp.getRhs().getType(),
                                  matmulOp.getC().getType()};
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern", rewriter.getStringAttr("matmul"));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  Value lhs = entry->getArgument(0);
  Value rhs = entry->getArgument(1);
  Value c = entry->getArgument(2);

  auto matmul = createMatMulBodyOp(rewriter, func.getLoc(), c.getType(), c, lhs,
                                   rhs, matmulOp);
  (void)matmul;
  rewriter.create<func::ReturnOp>(func.getLoc());

  return func;
}

static StringRef epilogueOpcodeName(EpilogueOpcode opcode) {
  switch (opcode) {
    case EpilogueOpcode::kAdd:
      return "add";
    case EpilogueOpcode::kMul:
      return "mul";
    case EpilogueOpcode::kRelu:
      return "relu";
    case EpilogueOpcode::kSigmoid:
      return "sigmoid";
  }
  llvm_unreachable("unknown epilogue opcode");
}

static std::string epiloguePatternName(const EpilogueProgram &program) {
  std::string pattern = "matmul";
  for (const EpilogueStep &step : program.steps)
    pattern += (Twine("_") + epilogueOpcodeName(step.opcode)).str();
  return pattern;
}

static Value epilogueExternalInput(const EpilogueStep &step) {
  assert(isBinaryEpilogueOpcode(step.opcode) && step.source);
  return step.inputIndex == 1 ? step.source->getOperand(2)
                              : step.source->getOperand(1);
}

static Operation *epilogueTerminalOp(const EpilogueProgram &program) {
  return program.terminalViewSource ? program.terminalViewSource
                                    : program.steps.back().source;
}

static func::FuncOp createMatMulEpilogueKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    MatMulOp matmulOp, const EpilogueProgram &program, Value output) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {matmulOp.getLhs().getType(),
                                  matmulOp.getRhs().getType(),
                                  output.getType()};
  for (const EpilogueStep &step : program.steps)
    if (isBinaryEpilogueOpcode(step.opcode))
      inputTypes.push_back(epilogueExternalInput(step).getType());
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern",
                rewriter.getStringAttr(epiloguePatternName(program)));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  IRMapping mapper;
  mapper.map(matmulOp.getLhs(), entry->getArgument(0));
  mapper.map(matmulOp.getRhs(), entry->getArgument(1));
  mapper.map(output, entry->getArgument(2));

  auto mapOutput = [&](Value destination) {
    if (mapper.contains(destination)) return;
    mapper.map(destination,
               rewriter.create<BufferOp>(func.getLoc(), destination.getType())
                   .getResult());
  };
  mapOutput(matmulOp.getC());
  rewriter.clone(*matmulOp, mapper);

  unsigned inputIndex = 3;
  for (const EpilogueStep &step : program.steps) {
    mapOutput(step.source->getOperand(0));
    if (isBinaryEpilogueOpcode(step.opcode)) {
      Value input = epilogueExternalInput(step);
      if (!mapper.contains(input))
        mapper.map(input, entry->getArgument(inputIndex));
      ++inputIndex;
    }
    rewriter.clone(*step.source, mapper);
  }
  if (program.terminalViewSource) {
    Value shape = program.terminalViewSource->getOperand(2);
    if (!mapper.contains(shape))
      rewriter.clone(*shape.getDefiningOp(), mapper);
    rewriter.clone(*program.terminalViewSource, mapper);
  }
  rewriter.create<func::ReturnOp>(func.getLoc());

  return func;
}

struct DnnEmbeddingHashBucketMatch {
  Value dynamicInput;
  Value embeddingWeight;
  Value outputBuffer;
  ReshapeOp finalReshape;
  SparseSegmentMeanOp sparseSegmentMean;
  StringToHashBucketFastOp hashBucket;
  SmallVector<Operation *> candidateOps;
  SmallVector<Operation *> kernelOps;
  std::string prefix;
};

static LogicalResult matchDnnEmbeddingHashBucket(
    ExpandDimsOp expandDims, DnnEmbeddingHashBucketMatch &match) {
  std::string expandName = getTfName(expandDims);
  StringRef suffix = "/ExpandDims";
  if (!StringRef(expandName).ends_with(suffix)) return failure();
  match.prefix = expandName.substr(0, expandName.size() - suffix.size());
  if (match.prefix.find("_embedding") == std::string::npos) return failure();

  auto func = expandDims->getParentOfType<func::FuncOp>();
  if (!func) return failure();

  std::string finalReshapeName = match.prefix + "/Reshape";
  auto finalReshape = findNamedOp<ReshapeOp>(func, finalReshapeName);
  if (!finalReshape) return failure();

  SparseSegmentMeanOp sparseSegmentMean = nullptr;
  func.walk([&](SparseSegmentMeanOp op) {
    if (sparseSegmentMean) return;
    std::string name = getTfName(op);
    if (startsWith(name, match.prefix) &&
        StringRef(name).contains("embedding_lookup_sparse")) {
      sparseSegmentMean = op;
    }
  });
  if (!sparseSegmentMean) return failure();

  auto embeddingGather =
      sparseSegmentMean.getOperation()->getOperand(1).getDefiningOp<GatherOp>();
  if (!embeddingGather) return failure();
  Value embeddingWeight = embeddingGather.getOperation()->getOperand(1);

  Value dynamicInput = expandDims.getOperation()->getOperand(1);
  Value outputBuffer = finalReshape.getOperation()->getOperand(0);

  if (!findUserOf<CompareOp>(expandDims.getResult()) ||
      !findUserOf<ShapeOp>(expandDims.getResult())) {
    return failure();
  }
  auto gatherNd = findUserOf<GatherNdOp>(expandDims.getResult());
  if (!gatherNd) return failure();
  auto hashBucket = findUserOf<StringToHashBucketFastOp>(gatherNd.getResult());
  if (!hashBucket) {
    return failure();
  }

  match.dynamicInput = dynamicInput;
  match.embeddingWeight = embeddingWeight;
  match.outputBuffer = outputBuffer;
  match.finalReshape = finalReshape;
  match.sparseSegmentMean = sparseSegmentMean;
  match.hashBucket = hashBucket;

  SmallVector<Value, 3> boundaryValues = {
      match.dynamicInput, match.embeddingWeight, match.outputBuffer};
  SmallPtrSet<Operation *, 32> visitedKernelOps;
  collectDefiningOpsPostOrder(match.finalReshape.getOperation(), boundaryValues,
                              visitedKernelOps, match.kernelOps);
  if (!isClosedKernelOpSet(match.kernelOps, boundaryValues)) return failure();

  func.walk([&](Operation *op) {
    if (!hasTfNamePrefix(op, match.prefix)) return;
    if (!isa<ConstantOp, VariableOp>(op)) {
      match.candidateOps.push_back(op);
    }
  });

  return success();
}

static void eraseDeadFusionOps(PatternRewriter &rewriter,
                               ArrayRef<Operation *> ops,
                               Operation *keepOutputBufferDef) {
  SmallPtrSet<Operation *, 32> opSet(ops.begin(), ops.end());
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::reverse(ops)) {
      if (!op || !opSet.contains(op) || op == keepOutputBufferDef) continue;
      bool hasUse = false;
      for (Value result : op->getResults()) {
        if (!result.use_empty()) {
          hasUse = true;
          break;
        }
      }
      if (hasUse) continue;
      rewriter.eraseOp(op);
      opSet.erase(op);
      changed = true;
    }
  }
}


// ==== KP fusion (kp-01/02/03) migrated from ANNC-812new ====
// match helpers + one-level patterns; ABI stays mlir_ciface.

static FailureOr<int64_t> getConstantIntValue(Value v) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) return failure();
  auto tensorType = dyn_cast<atir::TensorType>(v.getType());
  if (!tensorType) return failure();
  DenseElementsAttr dataAttr = tensorType.getCacheData();
  if (!dataAttr) return failure();
  if (!dataAttr.getElementType().isIntOrIndex()) return failure();
  if (dataAttr.isSplat()) {
    return dataAttr.getSplatValue<APInt>().getSExtValue();
  }
  if (dataAttr.getNumElements() != 1) return failure();
  return (*dataAttr.getValues<APInt>().begin()).getSExtValue();
}

static bool isConstantInt(Value v, int64_t expected) {
  FailureOr<int64_t> valOr = getConstantIntValue(v);
  return succeeded(valOr) && *valOr == expected;
}

static bool isConstantZero(Value v) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) return false;
  auto tensorType = dyn_cast<atir::TensorType>(v.getType());
  if (!tensorType) return false;
  DenseElementsAttr dataAttr = tensorType.getCacheData();
  if (!dataAttr) return false;
  auto isZero = [&](DenseElementsAttr attr) -> bool {
    if (attr.getElementType().isF32()) {
      return attr.isSplat() ? attr.getSplatValue<float>() == 0.0f
                            : (*attr.getValues<float>().begin()) == 0.0f;
    }
    if (attr.getElementType().isIntOrIndex()) {
      return attr.isSplat() ? attr.getSplatValue<APInt>().isZero()
                            : (*attr.getValues<APInt>().begin()).isZero();
    }
    return false;
  };
  return isZero(dataAttr);
}

static bool areConstantInts(Value v, ArrayRef<int64_t> expected) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) return false;
  auto tensorType = dyn_cast<atir::TensorType>(v.getType());
  if (!tensorType) return false;
  DenseElementsAttr dataAttr = tensorType.getCacheData();
  if (!dataAttr) return false;
  if (!dataAttr.getElementType().isIntOrIndex()) return false;
  ArrayRef<int64_t> shape = tensorType.getShape();
  if (shape.size() != 1) return false;
  SmallVector<int64_t> values;
  for (const APInt &val : dataAttr.getValues<APInt>()) {
    values.push_back(val.getSExtValue());
  }
  if (values.size() != expected.size()) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (values[i] != expected[i]) return false;
  }
  return true;
}

struct KpEmbeddingActionIdGatherMatch {
  Value indices1;
  Value params;
  Value indices2;
  Value packDim;
  Value pack;
  Value outputBuffer;
  ConcatV2Op anchor;
  SmallVector<Operation *> kernelOps;
};

static LogicalResult matchKpEmbeddingActionIdGather(
    ConcatV2Op anchor, KpEmbeddingActionIdGatherMatch &match) {
  // anchor: ConcatV2 with 2 values and axis constant -1.

  if (anchor.getValues().size() != 2) { return failure(); }
  if (!isConstantInt(anchor.getAxis(), -1)) { return failure(); }

  auto reshape =
      anchor.getValues()[0].getDefiningOp<ReshapeOp>();
  if (!reshape) { return failure(); }
  auto packShape = reshape.getTargetShape().getDefiningOp<PackOp>();
  if (!packShape || packShape.getInputs().size() != 2) { return failure(); }
  if (packShape.getAxis() != 0) { return failure(); }
  if (!isConstantInt(packShape.getInputs()[1], -1)) { return failure(); }
  auto outer = reshape.getInput().getDefiningOp<GatherOp>();
  if (!outer || !isConstantInt(outer.getAxis(), 0)) { return failure(); }
  auto inner = outer.getParams().getDefiningOp<GatherOp>();
  if (!inner || !isConstantInt(inner.getAxis(), 0)) return failure();

  auto fill = anchor.getValues()[1].getDefiningOp<FillOp>();
  if (!fill) { return failure(); }
  if (!isConstantZero(fill.getValueInput())) { return failure(); }
  auto packDims = fill.getShapeInput().getDefiningOp<PackOp>();
  if (!packDims || packDims.getInputs().size() != 2) { return failure(); }
  if (packDims.getAxis() != 0) { return failure(); }
  // The Reshape shape Pack's first input must be the same pack_size value
  // the kernel receives (packDims input 0).  812's rewriter does not check
  // this; a mismatch would make the reshape shape and the kernel's pack_size
  // disagree, silently corrupting semantics.
  if (packShape.getInputs()[0] != packDims.getInputs()[0]) { return failure(); }

  // dtype contract: params float32; indices1/indices2 为 int32/int64 的
  // 4 种组合之一 (JD 注册同款 4 组合, registry 按 T1/T2 约束选 kernel)。
  auto i1Type = dyn_cast<atir::TensorType>(inner.getIndices().getType());
  auto i2Type = dyn_cast<atir::TensorType>(outer.getIndices().getType());
  auto pType = dyn_cast<atir::TensorType>(inner.getParams().getType());
  if (!i1Type || !(i1Type.getElementType().isInteger(32) ||
                   i1Type.getElementType().isInteger(64)))
    return failure();
  if (!i2Type || !(i2Type.getElementType().isInteger(32) ||
                   i2Type.getElementType().isInteger(64)))
    return failure();
  if (!pType || !pType.getElementType().isF32()) return failure();
  if (i1Type.getShape().size() < 1 || i1Type.getShape().size() > 2)
    return failure();
  if (i2Type.getShape().size() < 1 || i2Type.getShape().size() > 2)
    return failure();

  // Single-output contract: intermediate results may not escape.
  for (Operation *user : inner.getResult().getUsers()) {
    if (user != outer.getOperation()) return failure();
  }
  for (Operation *user : outer.getResult().getUsers()) {
    if (user != reshape.getOperation()) return failure();
  }
  for (Operation *user : reshape.getResult().getUsers()) {
    if (user != anchor.getOperation()) return failure();
  }
  // Single-output contract relaxed for packShape/fill: sister anchors may
  // share them (hmv 930 — three ConcatV2 anchors share one packShape, two of
  // them also share one Fill).  Safe because each kernel func gets its own
  // clone, and eraseDeadFusionOps only erases ops whose results went unused,
  // so a shared packShape/fill survives in the main graph for the sisters.
  // Kept strict on packDims below: it feeds only the Fill, and letting it
  // escape would mean the pack_size the kernel receives is also consumed
  // elsewhere under an unverified contract.
  for (Operation *user : packDims.getResult().getUsers()) {
    if (user != fill.getOperation()) return failure();
  }

  match.indices1 = inner.getIndices();
  match.params = inner.getParams();
  match.indices2 = outer.getIndices();
  match.packDim = packDims.getInputs()[0];
  match.pack = packDims.getInputs()[1];
  match.outputBuffer = anchor.getOutput();
  match.anchor = anchor;

  SmallVector<Value, 6> boundaryValues = {
      match.indices1, match.params,  match.indices2,
      match.packDim,  match.pack,    match.outputBuffer};
  SmallPtrSet<Operation *, 32> visitedKernelOps;
  collectDefiningOpsPostOrder(anchor.getOperation(), boundaryValues,
                              visitedKernelOps, match.kernelOps);
  if (!isClosedKernelOpSet(match.kernelOps, boundaryValues)) return failure();

  return success();
}

static func::FuncOp createKpEmbeddingActionIdGatherKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    Type indices1Type, Type paramsType, Type indices2Type, Type packDimType,
    Type packType, Type outputType, ArrayRef<Operation *> kernelOps,
    Value indices1, Value params, Value indices2, Value packDim, Value pack,
    Value outputBuffer) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {indices1Type, paramsType, indices2Type,
                                  packDimType, packType, outputType};
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func = rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern",
                rewriter.getStringAttr("kp_embedding_action_id_gather"));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);

  IRMapping mapper;
  mapper.map(indices1, entry->getArgument(0));
  mapper.map(params, entry->getArgument(1));
  mapper.map(indices2, entry->getArgument(2));
  mapper.map(packDim, entry->getArgument(3));
  mapper.map(pack, entry->getArgument(4));
  mapper.map(outputBuffer, entry->getArgument(5));

  for (Operation *op : kernelOps) {
    rewriter.clone(*op, mapper);
  }
  rewriter.create<func::ReturnOp>(func.getLoc());

  return func;
}

static bool isConstantFloat(Value v, float expected) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) return false;
  auto tensorType = dyn_cast<atir::TensorType>(v.getType());
  if (!tensorType) return false;
  DenseElementsAttr dataAttr = tensorType.getCacheData();
  if (!dataAttr) return false;
  if (!dataAttr.getElementType().isF32()) return false;
  if (dataAttr.isSplat()) return dataAttr.getSplatValue<float>() == expected;
  if (dataAttr.getNumElements() != 1) return false;
  return (*dataAttr.getValues<float>().begin()) == expected;
}

// ── KP target-behavior interaction (一级) ───────────────────────────────────
//
// anchor = ConcatV2(axis=-1, 4 values):
//   values[0] = Mul(realdiv, 0.5) ← AddV2(add_v2_2) ← {Abs ← AddV2(add_v2_1
//               ← {BatchMatMul, bias}), add_v2_1}
//   values[1] = Tile（三处引用同一 Tile: values[1]、Sub.y、Mul.y）
//   values[2] = Sub(realdiv, tile)
//   values[3] = Mul(realdiv, tile)
// 语义: relu = ReLU(BatchMatMul(batch_input, weight) + bias)
//       output = Concat([relu, tile, relu-tile, relu*tile], -1)
// 移植补查 (812 未查): Tile multiples=[1,1,1]、BatchMatMul transpose 全 false、
// Add do_relu=false。

struct KpTargetBehaviorInteractionMatch {
  Value batchInput;
  Value weight;
  Value bias;
  Value tileInput;
  Value outputBuffer;
  ConcatV2Op anchor;
  SmallVector<Operation *> kernelOps;
};

static LogicalResult matchKpTargetBehaviorInteraction(
    ConcatV2Op anchor, KpTargetBehaviorInteractionMatch &match) {
  if (anchor.getValues().size() != 4) return failure();
  if (!isConstantInt(anchor.getAxis(), -1)) {

    return failure();
  }

  auto realdiv = anchor.getValues()[0].getDefiningOp<MulOp>();
  if (!realdiv) {

    return failure();
  }
  if (!isConstantFloat(realdiv.getY(), 0.5f)) {

    return failure();
  }

  auto add22 = realdiv.getX().getDefiningOp<AddOp>();
  if (!add22 || add22.getInputs().size() != 2) return failure();
  if (add22.getDoRelu()) return failure();

  AbsOp abs = nullptr;
  AddOp add21 = nullptr;
  if (auto a = add22.getInputs()[0].getDefiningOp<AbsOp>()) {
    abs = a;
    add21 = add22.getInputs()[1].getDefiningOp<AddOp>();
  } else if (auto a = add22.getInputs()[1].getDefiningOp<AbsOp>()) {
    abs = a;
    add21 = add22.getInputs()[0].getDefiningOp<AddOp>();
  }
  if (!abs || !add21) return failure();
  if (abs.getInput() != add21.getResult()) return failure();
  if (add21.getInputs().size() != 2) return failure();
  if (add21.getDoRelu()) return failure();

  BatchMatMulOp bmm = nullptr;
  Value biasVal;
  if (auto b0 = add21.getInputs()[0].getDefiningOp<BatchMatMulOp>()) {
    bmm = b0;
    biasVal = add21.getInputs()[1];
  } else if (auto b1 = add21.getInputs()[1].getDefiningOp<BatchMatMulOp>()) {
    bmm = b1;
    biasVal = add21.getInputs()[0];
  }
  if (!bmm) return failure();
  if (bmm.getTransposeA() || bmm.getTransposeB()) return failure();

  auto tileOp = anchor.getValues()[1].getDefiningOp<TileOp>();
  if (!tileOp) return failure();
  // Tile 必须恒等 (812 未查; kernel 假设 tile_input == [B,M,N] 直用)。
  if (!areConstantInts(tileOp.getMultiples(), {1, 1, 1})) return failure();

  auto sub = anchor.getValues()[2].getDefiningOp<SubOp>();
  if (!sub || sub.getX() != realdiv.getResult()) return failure();
  auto mul = anchor.getValues()[3].getDefiningOp<MulOp>();
  if (!mul || mul.getX() != realdiv.getResult()) return failure();
  if (sub.getY() != tileOp.getResult() || mul.getY() != tileOp.getResult())
    return failure();

  // dtype contract: 全 float32。
  auto biType = dyn_cast<atir::TensorType>(bmm.getA().getType());
  auto wType = dyn_cast<atir::TensorType>(bmm.getB().getType());
  auto bType = dyn_cast<atir::TensorType>(biasVal.getType());
  auto tType = dyn_cast<atir::TensorType>(tileOp.getInput().getType());
  if (!biType || !biType.getElementType().isF32() ||
      biType.getShape().size() != 3)
    return failure();
  if (!wType || !wType.getElementType().isF32() ||
      wType.getShape().size() != 2)
    return failure();
  if (!bType || !bType.getElementType().isF32() ||
      bType.getShape().size() != 1)
    return failure();
  if (!tType || !tType.getElementType().isF32() ||
      tType.getShape().size() != 3)
    return failure();

  // Single-output contract: intermediate results may not escape.
  for (Operation *user : bmm.getResult().getUsers()) {
    if (user != add21.getOperation()) return failure();
  }
  for (Operation *user : add21.getResult().getUsers()) {
    if (user != abs.getOperation() && user != add22.getOperation())
      return failure();
  }
  for (Operation *user : abs.getResult().getUsers()) {
    if (user != add22.getOperation()) return failure();
  }
  for (Operation *user : add22.getResult().getUsers()) {
    if (user != realdiv.getOperation()) return failure();
  }
  for (Operation *user : realdiv.getResult().getUsers()) {
    if (user != anchor.getOperation() && user != sub.getOperation() &&
        user != mul.getOperation())
      return failure();
  }
  for (Operation *user : tileOp.getResult().getUsers()) {
    if (user != anchor.getOperation() && user != sub.getOperation() &&
        user != mul.getOperation())
      return failure();
  }
  for (Operation *user : sub.getResult().getUsers()) {
    if (user != anchor.getOperation()) return failure();
  }
  for (Operation *user : mul.getResult().getUsers()) {
    if (user != anchor.getOperation()) return failure();
  }

  match.batchInput = bmm.getA();
  match.weight = bmm.getB();
  match.bias = biasVal;
  match.tileInput = tileOp.getInput();
  match.outputBuffer = anchor.getOutput();
  match.anchor = anchor;

  SmallVector<Value, 5> boundaryValues = {
      match.batchInput, match.weight, match.bias, match.tileInput,
      match.outputBuffer};
  SmallPtrSet<Operation *, 32> visitedKernelOps;
  collectDefiningOpsPostOrder(anchor.getOperation(), boundaryValues,
                              visitedKernelOps, match.kernelOps);

  if (!isClosedKernelOpSet(match.kernelOps, boundaryValues)) {

    return failure();
  }

  return success();
}

static func::FuncOp createKpTargetBehaviorInteractionKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    Type batchInputType, Type weightType, Type biasType, Type tileInputType,
    Type outputType, ArrayRef<Operation *> kernelOps, Value batchInput,
    Value weight, Value bias, Value tileInput, Value outputBuffer) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {batchInputType, weightType, biasType,
                                  tileInputType, outputType};
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func = rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern",
                rewriter.getStringAttr("kp_target_behavior_interaction"));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);

  IRMapping mapper;
  mapper.map(batchInput, entry->getArgument(0));
  mapper.map(weight, entry->getArgument(1));
  mapper.map(bias, entry->getArgument(2));
  mapper.map(tileInput, entry->getArgument(3));
  mapper.map(outputBuffer, entry->getArgument(4));

  for (Operation *op : kernelOps) {
    rewriter.clone(*op, mapper);
  }
  rewriter.create<func::ReturnOp>(func.getLoc());

  return func;
}

// ── KP sparse dynamic stitch (一级) ─────────────────────────────────────────
//
// anchor = ParallelDynamicStitch (N 个 indices + N 个 data):
//   indices[0] = DynamicPartition(Range(0, Size(x), 1), Cast(FloorMod(x, N)))
//   indices[j] = 同一 partition 的 outputs[j] (补查, 812 未查)
//   data[i] = GatherV2(axis=0, params=variables_i,
//                      indices=DynamicPartition(FloorDiv(x, N)).outputs[i])
// 语义: output[i] = variables[x[i] % N][x[i] / N]。
// 移植补查 (812 未查): Range start=0、GatherV2 axis=0、FloorMod/FloorDiv 同
// 一 x、其余 indices 与左分支 partition 对应输出一致。

struct KpSparseDynamicStitchMatch {
  Value x;
  SmallVector<Value> variables;
  Value outputBuffer;
  ParallelDynamicStitchOp anchor;
  SmallVector<Operation *> kernelOps;
};

static LogicalResult matchKpSparseDynamicStitch(
    ParallelDynamicStitchOp anchor, KpSparseDynamicStitchMatch &match) {
  const int64_t N = anchor.getIndices().size();
  if (N < 2 || N > 16) return failure();
  if (anchor.getData().size() != N) return failure();

  // Left branch: indices[0] and all other indices come from the same
  // DynamicPartition (the fused kernel only needs x and the variables, so
  // all index streams must agree with the partition semantics).
  auto partition =
      anchor.getIndices()[0].getDefiningOp<DynamicPartitionOp>();
  if (!partition || partition.getNumPartitions() != N) return failure();
  for (int64_t j = 1; j < N; ++j) {
    if (anchor.getIndices()[j] != partition.getOutputs()[j]) return failure();
  }

  auto range = partition.getData().getDefiningOp<RangeOp>();
  if (!range || !isConstantInt(range.getDelta(), 1)) return failure();
  if (!isConstantInt(range.getStart(), 0)) return failure();
  auto sizeOp = range.getLimit().getDefiningOp<SizeOp>();
  if (!sizeOp) return failure();
  Value x = sizeOp.getInput();

  auto cast = partition.getPartitions().getDefiningOp<CastOp>();
  if (!cast) return failure();
  auto floorMod = cast.getInput().getDefiningOp<FloorModOp>();
  if (!floorMod || floorMod.getX() != x) return failure();
  if (!isConstantInt(floorMod.getY(), N)) return failure();

  // dtype contract: x int64; variables 2D float32.
  auto xType = dyn_cast<atir::TensorType>(x.getType());
  if (!xType || !xType.getElementType().isInteger(64)) return failure();

  // Right branch: one Gather per partition.
  for (int64_t i = 0; i < N; ++i) {

    auto gather = anchor.getData()[i].getDefiningOp<GatherOp>();
    if (!gather) {

      return failure();
    }
    if (!isConstantInt(gather.getAxis(), 0)) {

      return failure();
    }
    auto vType = dyn_cast<atir::TensorType>(gather.getParams().getType());
    if (!vType || !vType.getElementType().isF32() ||
        vType.getShape().size() != 2) {

      return failure();
    }
    match.variables.push_back(gather.getParams());
    auto partition1 = gather.getIndices().getDefiningOp<DynamicPartitionOp>();
    if (!partition1) {

      return failure();
    }
    if (partition1.getNumPartitions() != N) {
      return failure();
    }
    if (gather.getIndices() != partition1.getOutputs()[i]) {

      return failure();
    }
    auto floorDiv = partition1.getData().getDefiningOp<FloorDivOp>();
    if (!floorDiv) {

      return failure();
    }
    if (floorDiv.getX() != x) {

      return failure();
    }
    if (!isConstantInt(floorDiv.getY(), N)) {

      return failure();
    }
  }

  // Single-output contract: intermediate results may not escape.
  for (Operation *user : range.getResult().getUsers()) {
    if (user != partition.getOperation()) return failure();
  }
  for (Operation *user : cast.getResult().getUsers()) {
    // The cast feeds the partitions of BOTH DynamicPartitions (left and
    // right branches share the same partitions stream).
    if (!isa<DynamicPartitionOp>(user)) return failure();
  }
  for (Operation *user : floorMod.getResult().getUsers()) {
    if (user != cast.getOperation()) return failure();
  }
  for (int64_t j = 0; j < N; ++j) {
    for (Operation *user : partition.getOutputs()[j].getUsers()) {
      if (user != anchor.getOperation()) return failure();
    }
    GatherOp gatherJ = anchor.getData()[j].getDefiningOp<GatherOp>();
    for (Operation *user : gatherJ.getResult().getUsers()) {
      if (user != anchor.getOperation()) return failure();
    }
    DynamicPartitionOp partitionJ =
        gatherJ.getIndices().getDefiningOp<DynamicPartitionOp>();
    for (Operation *user : partitionJ.getOutputs()[j].getUsers()) {
      if (user != gatherJ.getOperation()) return failure();
    }
    FloorDivOp floorDivJ =
        partitionJ.getData().getDefiningOp<FloorDivOp>();
    for (Operation *user : floorDivJ.getResult().getUsers()) {
      if (user != partitionJ.getOperation()) return failure();
    }
  }

  match.x = x;
  match.outputBuffer = anchor.getOutput();
  match.anchor = anchor;

  SmallVector<Value, 10> boundaryValues;
  boundaryValues.push_back(match.x);
  for (Value v : match.variables) boundaryValues.push_back(v);
  boundaryValues.push_back(match.outputBuffer);
  SmallPtrSet<Operation *, 32> visitedKernelOps;
  collectDefiningOpsPostOrder(anchor.getOperation(), boundaryValues,
                              visitedKernelOps, match.kernelOps);
  if (!isClosedKernelOpSet(match.kernelOps, boundaryValues)) return failure();

  return success();
}

static func::FuncOp createKpSparseDynamicStitchKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    ArrayRef<Value> inputs, Value outputBuffer, ArrayRef<Operation *> kernelOps) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes;
  for (Value v : inputs) inputTypes.push_back(v.getType());
  inputTypes.push_back(outputBuffer.getType());
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func = rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern",
                rewriter.getStringAttr("kp_sparse_dynamic_stitch"));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);

  IRMapping mapper;
  for (size_t i = 0; i < inputs.size(); ++i) {
    mapper.map(inputs[i], entry->getArgument(i));
  }
  mapper.map(outputBuffer, entry->getArgument(inputs.size()));

  for (Operation *op : kernelOps) {
    rewriter.clone(*op, mapper);
  }
  rewriter.create<func::ReturnOp>(func.getLoc());

  return func;
}

// ── KP sparse segment reduce (一级, 2 输出) ──────────────────────────────────
//
// anchor = StridedSlice(shrink=1, begin 常量 1 元素) ← Shape ←
//          SparseSegmentMean|Sum(data, indices, segment_ids=StridedSlice(keys))
// keys 列 StridedSlice: shrink=2, begin_mask=1, end_mask=1, begin 常量 shape{2}
// 2 输出: segment 结果 + 标量 slice_output (= begin_1[0]==0 ? batch : embed)。
// 移植补查 (812 未查): numSegments=0、strides=[1,1]、ellipsis/newaxis=0。


struct FuseDnnEmbeddingHashBucketAsFuncCallPattern
    : public OpRewritePattern<ExpandDimsOp> {
  using OpRewritePattern<ExpandDimsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpandDimsOp expandDims,
                                PatternRewriter &rewriter) const override {
    DnnEmbeddingHashBucketMatch match;
    if (failed(matchDnnEmbeddingHashBucket(expandDims, match))) {
      return failure();
    }

    ModuleOp module = expandDims->getParentOfType<ModuleOp>();
    if (!module) return failure();

    std::string kernelBase =
        getStableFusionKernelName(match.prefix, "dnn_embedding_hash_bucket");
    std::string kernelName = uniquifySymbolName(module, kernelBase);
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(match.prefix + "/rec_embed_kp_dnn_bucket"));

    auto kernelFunc = createDnnEmbeddingHashBucketKernelFunc(
        module, rewriter, kernelName, match.dynamicInput.getType(),
        match.embeddingWeight.getType(), match.outputBuffer.getType(),
        match.hashBucket.getNumBuckets(), match.kernelOps, match.dynamicInput,
        match.embeddingWeight, match.outputBuffer);
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("dnn_embedding_hash_bucket")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> argSpecs;
    argSpecs.push_back({"fixed", getValueName(match.embeddingWeight),
                        match.embeddingWeight.getType()});
    argSpecs.push_back({"dynamic", getValueName(match.dynamicInput),
                        match.dynamicInput.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), argSpecs)));
    SmallVector<FusionArgSpec> outputSpecs;
    outputSpecs.push_back({"output", getTfName(match.finalReshape),
                           match.outputBuffer.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputSpecs)));
    metadata.push_back(
        rewriter.getNamedAttr("abi", rewriter.getStringAttr("mlir_ciface")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {1, 0, 2})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr(
        "symbolic_signature", rewriter.getStringAttr("F:0|S:1;?,?")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));

    rewriter.setInsertionPoint(match.finalReshape);
    rewriter.create<func::CallOp>(
        match.finalReshape.getLoc(), kernelFunc,
        ValueRange{match.dynamicInput, match.embeddingWeight,
                   match.outputBuffer});

    Operation *outputBufferDef = match.outputBuffer.getDefiningOp();
    rewriter.replaceAllUsesWith(match.finalReshape.getResult(),
                                match.outputBuffer);
    eraseDeadFusionOps(rewriter, match.candidateOps, outputBufferDef);
    return success();
  }
};

#ifdef ANNC_ENABLE_KDNN_ADAPTOR
// Outlines a GEMM chain as an AOT kdnn site on the annc_execution_v2 ABI
// (one calling convention for the TF and ONNX hosts; the kernel takes its
// thread pool from the execution context). Only active at ANNC_GEMM=1;
// level 2 belongs to the epilogue path. Greedy prefix truncation from
// matmul_add_relu down to matmul: intermediate results must be single-use
// (a single-output kernel cannot reproduce them), while consumers of the
// terminal result stay in the main graph.
struct FuseKdnnMatMulAsFuncCallPattern : public OpRewritePattern<MatMulOp> {
  FuseKdnnMatMulAsFuncCallPattern(MLIRContext *context,
                                  const CustomOpTypeFilter &customOpFilter)
      : OpRewritePattern<MatMulOp>(context, /*benefit=*/2),
        customOpFilter(customOpFilter) {}

  LogicalResult matchAndRewrite(MatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (annc::anncGemmFusionLevel() != 1) return failure();
    if (matmulOp->hasAttr("annc.fusion_materialized")) return failure();
    if (matmulOp->hasAttr("annc.postop_fused")) return failure();
    if (matmulOp.getWithBias() || matmulOp.getDoRelu() ||
        matmulOp.getRightTranspose() || matmulOp.getLeftTranspose() ||
        matmulOp.getOutputTranspose() || matmulOp.getMStart() ||
        matmulOp.getNStart() || matmulOp.getKStart() || matmulOp.getMSize() ||
        matmulOp.getNSize() || matmulOp.getKSize() ||
        matmulOp->getAttr("rhs_format"))
      return failure();
    if (!matmulOp.getEpilogueInputs().empty()) return failure();

    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    if (!module) return failure();

    auto hasUsersBeyond = [](Value value, Operation *allowed) {
      for (Operation *user : value.getUsers())
        if (user != allowed) return true;
      return false;
    };
    Value bias;
    AddOp addOp = findUniqueMatMulAddUser(matmulOp, bias);
    if (addOp && (addOp.getDoRelu() || addOp.getScalar() ||
                  hasUsersBeyond(matmulOp.getResult(), addOp.getOperation())))
      addOp = nullptr;
    ReluOp reluOp = addOp ? findUniqueAddReluUser(addOp) : nullptr;
    if (reluOp && (reluOp.getReluLimit().convertToFloat() != -1.0f ||
                   hasUsersBeyond(addOp.getResult(), reluOp.getOperation())))
      reluOp = nullptr;

    SmallVector<Operation *> fusedOps = {matmulOp.getOperation()};
    if (addOp) fusedOps.push_back(addOp.getOperation());
    if (reluOp) fusedOps.push_back(reluOp.getOperation());
    Operation *terminal = reluOp ? reluOp.getOperation() : addOp.getOperation();
    if (!terminal) terminal = matmulOp.getOperation();
    SmallVector<Value> escaping = collectEscapingResults(fusedOps);
    if (escaping.size() != 1 || escaping[0] != terminal->getResult(0))
      return failure();

    StringRef formOpType = reluOp  ? StringRef("MatMulAddRelu")
                           : addOp ? StringRef("MatMulAdd")
                                   : StringRef("MatMul");
    if (!customOpFilter.isEnabled(formOpType)) return failure();

    // Site-level availability probe (kdnn registers f32 kernels only); opt
    // runs before asm so the annc.enable_kdnn module attribute is not set.
    CustomOpSchema schema = CustomOpSchema::get(formOpType)
                                .TypeVar("T")
                                .MemRefArg("lhs", 2, "T")
                                .MemRefArg("rhs", 2, "T");
    if (addOp) schema.MemRefArg("bias", 1, "T");
    schema.Result(reluOp  ? StringRef("relu")
                  : addOp ? StringRef("add")
                          : StringRef("output"),
                  2, "T");
    SmallVector<Type> schemaInputTypes = {matmulOp.getLhs().getType(),
                                          matmulOp.getRhs().getType()};
    if (addOp) schemaInputTypes.push_back(bias.getType());
    auto typeConstraints = inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(schemaInputTypes),
        TypeRange{terminal->getResult(0).getType()});
    if (!typeConstraints) {
      llvm::consumeError(typeConstraints.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = formOpType;
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*typeConstraints);
    if (!annc::kernels::hasAnyAvailableKernel(request, /*enableKdnn=*/true))
      return failure();

    StringRef pattern = reluOp  ? StringRef("matmul_add_relu")
                        : addOp ? StringRef("matmul_add")
                                : StringRef("matmul");
    std::string matmulName = getTfName(matmulOp);
    std::string outputName =
        addOp ? getFusionOutputName(terminal->getResult(0)) : matmulName;
    if (outputName.empty()) outputName = getTfName(terminal);
    std::string kernelName = uniquifySymbolName(
        module, getStableMatMulFusionKernelName(matmulOp, pattern));
    std::string clusterName =
        uniquifyFusionName(module, sanitizeName("annc_fused_" + outputName));

    // Fixed (weight-like) vs dynamic (batch-carrying) args follow the JIT
    // matmul sites. The v2 host passes inputs positionally after the
    // execution pointer and resolves dynamic output dims from the input at
    // index num_fixed, so the kernel signature and the metadata args share
    // the fixed-first order.
    Type outputType = terminal->getResult(0).getType();
    bool lhsIsDynamic =
        hasMatchingBatchDim(matmulOp.getLhs().getType(), outputType);
    bool rhsIsDynamic =
        hasMatchingBatchDim(matmulOp.getRhs().getType(), outputType);
    if (lhsIsDynamic == rhsIsDynamic) {
      lhsIsDynamic = true;
      rhsIsDynamic = false;
    }
    Value fixedInput = lhsIsDynamic ? matmulOp.getRhs() : matmulOp.getLhs();
    Value dynamicInput = lhsIsDynamic ? matmulOp.getLhs() : matmulOp.getRhs();
    SmallVector<Value> boundaryInputs = {fixedInput};
    if (addOp) boundaryInputs.push_back(bias);
    boundaryInputs.push_back(dynamicInput);
    SmallVector<Value> boundaryOutputs = {terminal->getResult(0)};

    auto kernelFunc = createExecutionV2KernelFunc(
        module, rewriter, kernelName, fusedOps, boundaryInputs, boundaryOutputs);
    kernelFunc->setAttr("fusion.pattern", rewriter.getStringAttr(pattern));
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<FusionArgSpec> argSpecs;
    argSpecs.push_back(
        {"fixed", getValueName(fixedInput), fixedInput.getType()});
    if (addOp)
      argSpecs.push_back({"fixed", getValueName(bias), bias.getType()});
    argSpecs.push_back(
        {"dynamic", getValueName(dynamicInput), dynamicInput.getType()});

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr("fusion.pattern",
                                             rewriter.getStringAttr(pattern)));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(
        rewriter.getNamedAttr("tf.name", rewriter.getStringAttr(clusterName)));
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), argSpecs)));
    SmallVector<FusionArgSpec> outputSpecs;
    outputSpecs.push_back({"output", outputName, outputType});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputSpecs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(
        rewriter.getNamedAttr("symbolic_signature", rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    metadata.push_back(rewriter.getNamedAttr(
        "member_node_names",
        rewriter.getArrayAttr(collectMemberNodeNames(rewriter, fusedOps))));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));

    // The v2 ABI needs an execution context that only exists at the host
    // boundary, so the main-graph region stays as the fallback and is only
    // marked to keep other patterns from building a site on it.
    for (Operation *op : fusedOps) {
      op->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
    }
    return success();
  }

 private:
  CustomOpTypeFilter customOpFilter;
};
#endif  // ANNC_ENABLE_KDNN_ADAPTOR

struct FuseMatMulAsFuncCallPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    const int gemmLevel = annc::anncGemmFusionLevel();
    if (gemmLevel == 0) return failure();
    if (matmulOp->hasAttr("annc.fusion_materialized")) return failure();
    if (matmulOp->hasAttr("annc.postop_fused")) return failure();
    if (matmulOp.getWithBias() || matmulOp.getDoRelu()) return failure();

    std::string matmulName = getTfName(matmulOp);
    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    if (!module) return failure();

    Value executionV2Bias;
    AddOp executionV2Add = findUniqueMatMulAddUser(matmulOp, executionV2Bias);
    ReluOp executionV2Relu =
        executionV2Add ? findUniqueAddReluUser(executionV2Add) : nullptr;
    if (isExecutionV2BuiltinCompatible(matmulOp, executionV2Add,
                                       executionV2Relu)) {
      SmallVector<Operation *> fusedOps = {matmulOp.getOperation(),
                                           executionV2Add.getOperation(),
                                           executionV2Relu.getOperation()};
      SmallVector<Value> boundaryOutputs = collectEscapingResults(fusedOps);
      bool hasExactTwoOutputContract =
          boundaryOutputs.size() == 2 &&
          boundaryOutputs[0] == executionV2Add.getResult() &&
          boundaryOutputs[1] == executionV2Relu.getResult();
      if (hasExactTwoOutputContract) {
        SmallVector<Value> boundaryInputs = collectExecutionV2Inputs(fusedOps);
        if (boundaryInputs.size() != 3 ||
            boundaryInputs[0] != matmulOp.getLhs() ||
            boundaryInputs[1] != matmulOp.getRhs() ||
            boundaryInputs[2] != executionV2Bias) {
          return failure();
        }
        auto schema = CustomOpSchema::get("MatMulAddReluWithAddOutput")
                          .TypeVar("T")
                          .MemRefArg("lhs", 2, "T")
                          .MemRefArg("rhs", 2, "T")
                          .MemRefArg("bias", 1, "T")
                          .Result("add", 2, "T")
                          .Result("relu", 2, "T");
        SmallVector<Type> inputTypes;
        for (Value input : boundaryInputs)
          inputTypes.push_back(input.getType());
        SmallVector<Type> outputTypes;
        for (Value output : boundaryOutputs)
          outputTypes.push_back(output.getType());
        auto typeConstraints = inferTypeConstraintsFromSchema(
            schema.toMetadata(rewriter.getContext()), TypeRange(inputTypes),
            TypeRange(outputTypes));
        if (!typeConstraints) {
          llvm::consumeError(typeConstraints.takeError());
          return failure();
        }
        annc::kernels::KernelResolveRequest request;
        request.op_type = "MatMulAddReluWithAddOutput";
        request.abi = "annc_execution_v2";
        request.type_constraints = std::move(*typeConstraints);
        auto enableKdnnAttr =
            module->getAttrOfType<BoolAttr>("annc.enable_kdnn");
        if (!annc::kernels::hasAnyAvailableKernel(
                request, enableKdnnAttr && enableKdnnAttr.getValue())) {
          return failure();
        }

        std::string kernelName = uniquifySymbolName(
            module,
            getStableMatMulFusionKernelName(matmulOp, "matmul_add_relu"));
        std::string clusterName = uniquifyFusionName(
            module,
            sanitizeName("annc_fused_" + getValueName(boundaryOutputs.back())));
        auto kernelFunc =
            createExecutionV2KernelFunc(module, rewriter, kernelName, fusedOps,
                                        boundaryInputs, boundaryOutputs);
        setExecutionMode(kernelFunc, rewriter, kJitExecutionMode);
        std::string templateFingerprint =
            atir::computeAtirTemplateFingerprint(kernelFunc);

        SmallVector<NamedAttribute> metadata;
        metadata.push_back(rewriter.getNamedAttr(
            "fusion.pattern", rewriter.getStringAttr("matmul_add_relu")));
        addExecutionModeMetadata(metadata, rewriter, kJitExecutionMode);
        metadata.push_back(rewriter.getNamedAttr(
            "kernel_name", rewriter.getStringAttr(kernelName)));
        metadata.push_back(rewriter.getNamedAttr(
            "template_fingerprint",
            rewriter.getStringAttr(templateFingerprint)));
        metadata.push_back(rewriter.getNamedAttr(
            "tf.name", rewriter.getStringAttr(clusterName)));

        SmallVector<FusionArgSpec> argSpecs;
        for (Value input : boundaryInputs) {
          StringRef role = input == matmulOp.getLhs() ? "dynamic" : "fixed";
          argSpecs.push_back(
              {role.str(), getValueName(input), input.getType()});
        }
        metadata.push_back(rewriter.getNamedAttr(
            "args", makeFusionArgArray(rewriter.getContext(), argSpecs)));

        SmallVector<FusionArgSpec> outputSpecs;
        for (Value output : boundaryOutputs) {
          outputSpecs.push_back(
              {"output", getFusionOutputName(output), output.getType()});
        }
        metadata.push_back(rewriter.getNamedAttr(
            "outputs", makeFusionArgArray(rewriter.getContext(), outputSpecs)));
        metadata.push_back(rewriter.getNamedAttr(
            "abi", rewriter.getStringAttr("annc_execution_v2")));
        metadata.push_back(rewriter.getNamedAttr(
            "kernel_arg_order",
            makeI64Array(rewriter.getContext(), ArrayRef<int64_t>{})));
        metadata.push_back(rewriter.getNamedAttr(
            "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
        metadata.push_back(rewriter.getNamedAttr(
            "symbolic_signature", rewriter.getStringAttr("F:0|S:1;?,?")));
        metadata.push_back(rewriter.getNamedAttr(
            "fallback_function", rewriter.getStringAttr("original_subgraph")));
        kernelFunc->setAttr(
            "fusion.metadata",
            DictionaryAttr::get(rewriter.getContext(), metadata));

        matmulOp->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
        return success();
      }
    }

    std::optional<EpilogueProgram> epilogue;
    // Epilogue candidates are only consumed at ANNC_GEMM=2 (the pipeline only
    // emits the attr at that level; this also guards direct annc-opt runs).
    auto candidate = gemmLevel == 2
                         ? matmulOp->getAttrOfType<DictionaryAttr>(
                               kGemmEpilogueCandidateAttr)
                         : DictionaryAttr();
    if (candidate) {
      auto parsed = parseEpilogueCandidatePlan(candidate);
      if (failed(parsed) || failed(bindEpilogueProgramToSource(matmulOp,
                                                               *parsed))) {
        matmulOp.emitOpError(
            "epilogue candidate does not match the current source chain");
        return failure();
      }
      epilogue = std::move(*parsed);
    }

    Operation *outputOp =
        epilogue ? epilogueTerminalOp(*epilogue) : matmulOp.getOperation();
    Value output = epilogue ? outputOp->getOperand(0) : matmulOp.getC();
    Type outputType = output.getType();
    std::string pattern =
        epilogue ? epiloguePatternName(*epilogue) : std::string("matmul");
    std::string outputName =
        epilogue ? getFusionOutputName(epilogue->result) : matmulName;
    if (outputName.empty()) outputName = getTfName(outputOp);
    std::string kernelName = epilogue
        ? uniquifySymbolName(
              module, getStableMatMulFusionKernelName(matmulOp, pattern))
        : uniquifySymbolName(module, getStableMatMulKernelName(matmulOp));
    std::string clusterName =
        uniquifyFusionName(module, sanitizeName("annc_fused_" + outputName));

    auto kernelFunc = epilogue
        ? createMatMulEpilogueKernelFunc(module, rewriter, kernelName, matmulOp,
                                         *epilogue, output)
        : createKernelFunc(module, rewriter, kernelName, matmulOp);
    setExecutionMode(kernelFunc, rewriter, kJitExecutionMode);
    std::string templateFingerprint =
        atir::computeAtirTemplateFingerprint(kernelFunc);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr("fusion.pattern",
                                             rewriter.getStringAttr(pattern)));
    addExecutionModeMetadata(metadata, rewriter, kJitExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "template_fingerprint", rewriter.getStringAttr(templateFingerprint)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));

    SmallVector<FusionArgSpec> argSpecs;
    SmallVector<int64_t> kernelArgOrder;
    if (epilogue) {
      struct PendingArg {
        Value value;
        unsigned kernelIndex;
      };
      SmallVector<PendingArg> fixed = {{matmulOp.getRhs(), 1}};
      SmallVector<PendingArg> dynamic = {{matmulOp.getLhs(), 0}};
      unsigned kernelIndex = 3;
      for (const EpilogueStep &step : epilogue->steps) {
        if (!isBinaryEpilogueOpcode(step.opcode)) continue;
        PendingArg input{epilogueExternalInput(step), kernelIndex++};
        if (step.broadcast == BroadcastKind::kM ||
            step.broadcast == BroadcastKind::kMatrix)
          dynamic.push_back(input);
        else
          fixed.push_back(input);
      }
      kernelArgOrder.resize(kernelIndex);
      auto appendArgs = [&](ArrayRef<PendingArg> args, StringRef role) {
        for (const PendingArg &arg : args) {
          kernelArgOrder[arg.kernelIndex] = argSpecs.size();
          argSpecs.push_back(
              {role.str(), getValueName(arg.value), arg.value.getType()});
        }
      };
      appendArgs(fixed, "fixed");
      appendArgs(dynamic, "dynamic");
      kernelArgOrder[2] = argSpecs.size();
    } else {
      bool lhsIsDynamic =
          hasMatchingBatchDim(matmulOp.getLhs().getType(), outputType);
      bool rhsIsDynamic =
          hasMatchingBatchDim(matmulOp.getRhs().getType(), outputType);
      if (lhsIsDynamic == rhsIsDynamic) {
        lhsIsDynamic = true;
        rhsIsDynamic = false;
      }
      Value fixedInput = lhsIsDynamic ? matmulOp.getRhs() : matmulOp.getLhs();
      Value dynamicInput =
          lhsIsDynamic ? matmulOp.getLhs() : matmulOp.getRhs();
      kernelArgOrder = lhsIsDynamic ? SmallVector<int64_t>{1, 0, 2}
                                    : SmallVector<int64_t>{0, 1, 2};
      argSpecs.push_back(
          {"fixed", getValueName(fixedInput), fixedInput.getType()});
      argSpecs.push_back(
          {"dynamic", getValueName(dynamicInput), dynamicInput.getType()});
    }
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), argSpecs)));
    SmallVector<FusionArgSpec> outputSpecs;
    outputSpecs.push_back({"output", outputName, outputType});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputSpecs)));
    metadata.push_back(
        rewriter.getNamedAttr("abi", rewriter.getStringAttr("mlir_ciface")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order",
        makeI64Array(rewriter.getContext(), kernelArgOrder)));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr(
        "symbolic_signature", rewriter.getStringAttr("F:0|S:1;?,?")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));

    rewriter.setInsertionPoint(outputOp);
    if (epilogue) {
      SmallVector<Value> callArgs = {matmulOp.getLhs(), matmulOp.getRhs(),
                                     output};
      for (const EpilogueStep &step : epilogue->steps)
        if (isBinaryEpilogueOpcode(step.opcode))
          callArgs.push_back(epilogueExternalInput(step));
      rewriter.create<func::CallOp>(outputOp->getLoc(), kernelFunc, callArgs);
      rewriter.replaceAllUsesWith(epilogue->result, output);
      if (epilogue->terminalViewSource)
        rewriter.eraseOp(epilogue->terminalViewSource);
      for (auto it = epilogue->steps.rbegin(); it != epilogue->steps.rend();
           ++it)
        rewriter.eraseOp(it->source);
      rewriter.eraseOp(matmulOp);
    } else {
      rewriter.create<func::CallOp>(
          matmulOp.getLoc(), kernelFunc,
          ValueRange{matmulOp.getLhs(), matmulOp.getRhs(), matmulOp.getC()});
      rewriter.replaceOp(matmulOp, matmulOp.getC());
    }
    return success();
  }
};

// ==== KP fusion one-level rewrite patterns (kp-01/02/03) ====

struct FuseKpSparseDynamicStitchAsFuncCallPattern
    : public OpRewritePattern<ParallelDynamicStitchOp> {
  using OpRewritePattern<ParallelDynamicStitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelDynamicStitchOp anchor,
                                PatternRewriter &rewriter) const override {
    KpSparseDynamicStitchMatch match;
    if (failed(matchKpSparseDynamicStitch(anchor, match))) {
      return failure();
    }

    ModuleOp module = anchor->getParentOfType<ModuleOp>();
    if (!module) return failure();

    std::string anchorName = getTfName(match.anchor);
    std::string kernelName = uniquifySymbolName(
        module, getStableFusionKernelName(anchorName, "kp_sparse_dynamic_stitch"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/kp_fused"));

    SmallVector<Value, 10> callInputs;
    callInputs.push_back(match.x);
    for (Value v : match.variables) callInputs.push_back(v);

    auto kernelFunc = createKpSparseDynamicStitchKernelFunc(
        module, rewriter, kernelName, callInputs, match.outputBuffer,
        match.kernelOps);
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("kp_sparse_dynamic_stitch")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> argSpecs;
    argSpecs.push_back({"dynamic", getValueName(match.x), match.x.getType()});
    for (Value v : match.variables) {
      argSpecs.push_back({"fixed", getValueName(v), v.getType()});
    }
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), argSpecs)));
    SmallVector<FusionArgSpec> outputSpecs;
    outputSpecs.push_back(
        {"output", anchorName, match.outputBuffer.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputSpecs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("mlir_ciface")));
    // The GraphDef rewrite emits the inputs fixed-first, dynamic-last:
    //   [v0..vN-1, x, output]
    // while the kernel function signature is [x, v0..vN-1, output].
    // Map the runtime list onto that signature explicitly.
    SmallVector<int64_t> kernelArgOrder;
    const int64_t numVariables =
        static_cast<int64_t>(match.variables.size());
    kernelArgOrder.push_back(numVariables);  // x
    for (int64_t i = 0; i < numVariables; ++i) {
      kernelArgOrder.push_back(i);  // v0..vN-1
    }
    kernelArgOrder.push_back(numVariables + 1);  // output
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), kernelArgOrder)));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr(
        "symbolic_signature", rewriter.getStringAttr("F:1|S:?;?")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
        DictionaryAttr::get(rewriter.getContext(), metadata));

    rewriter.setInsertionPoint(anchor);
    SmallVector<Value, 10> callArgs(callInputs);
    callArgs.push_back(match.outputBuffer);
    rewriter.create<func::CallOp>(anchor.getLoc(), kernelFunc, callArgs);

    Operation *outputBufferDef = match.outputBuffer.getDefiningOp();
    rewriter.replaceAllUsesWith(anchor.getResult(), match.outputBuffer);
    eraseDeadFusionOps(rewriter, match.kernelOps, outputBufferDef);
    return success();
  }
};

struct FuseKpTargetBehaviorInteractionAsFuncCallPattern
    : public OpRewritePattern<ConcatV2Op> {
  using OpRewritePattern<ConcatV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatV2Op anchor,
                                PatternRewriter &rewriter) const override {
    KpTargetBehaviorInteractionMatch match;
    if (failed(matchKpTargetBehaviorInteraction(anchor, match))) {
      return failure();
    }

    ModuleOp module = anchor->getParentOfType<ModuleOp>();
    if (!module) return failure();

    std::string anchorName = getTfName(match.anchor);
    std::string kernelName = uniquifySymbolName(
        module,
        getStableFusionKernelName(anchorName, "kp_target_behavior_interaction"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/kp_fused"));

    auto kernelFunc = createKpTargetBehaviorInteractionKernelFunc(
        module, rewriter, kernelName, match.batchInput.getType(),
        match.weight.getType(), match.bias.getType(),
        match.tileInput.getType(), match.outputBuffer.getType(),
        match.kernelOps, match.batchInput, match.weight, match.bias,
        match.tileInput, match.outputBuffer);
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern",
        rewriter.getStringAttr("kp_target_behavior_interaction")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> argSpecs;
    argSpecs.push_back({"dynamic", getValueName(match.batchInput),
                        match.batchInput.getType()});
    argSpecs.push_back(
        {"fixed", getValueName(match.weight), match.weight.getType()});
    argSpecs.push_back(
        {"fixed", getValueName(match.bias), match.bias.getType()});
    argSpecs.push_back({"dynamic", getValueName(match.tileInput),
                        match.tileInput.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), argSpecs)));
    SmallVector<FusionArgSpec> outputSpecs;
    outputSpecs.push_back(
        {"output", anchorName, match.outputBuffer.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputSpecs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("mlir_ciface")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order",
        makeI64Array(rewriter.getContext(), {0, 1, 2, 3, 4})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr(
        "symbolic_signature", rewriter.getStringAttr("F:2|S:2;?,?,?")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
        DictionaryAttr::get(rewriter.getContext(), metadata));

    rewriter.setInsertionPoint(anchor);
    rewriter.create<func::CallOp>(
        anchor.getLoc(), kernelFunc,
        ValueRange{match.batchInput, match.weight, match.bias,
                   match.tileInput, match.outputBuffer});

    Operation *outputBufferDef = match.outputBuffer.getDefiningOp();
    rewriter.replaceAllUsesWith(anchor.getResult(), match.outputBuffer);
    eraseDeadFusionOps(rewriter, match.kernelOps, outputBufferDef);
    return success();
  }
};

struct FuseKpEmbeddingActionIdGatherAsFuncCallPattern
    : public OpRewritePattern<ConcatV2Op> {
  using OpRewritePattern<ConcatV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatV2Op anchor,
                                PatternRewriter &rewriter) const override {
    KpEmbeddingActionIdGatherMatch match;
    if (failed(matchKpEmbeddingActionIdGather(anchor, match))) {
      return failure();
    }

    ModuleOp module = anchor->getParentOfType<ModuleOp>();
    if (!module) return failure();

    std::string anchorName = getTfName(match.anchor);
    std::string kernelName = uniquifySymbolName(
        module,
        getStableFusionKernelName(anchorName, "kp_embedding_action_id_gather"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/kp_fused"));

    auto kernelFunc = createKpEmbeddingActionIdGatherKernelFunc(
        module, rewriter, kernelName, match.indices1.getType(),
        match.params.getType(), match.indices2.getType(),
        match.packDim.getType(), match.pack.getType(),
        match.outputBuffer.getType(), match.kernelOps, match.indices1,
        match.params, match.indices2, match.packDim, match.pack,
        match.outputBuffer);
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern",
        rewriter.getStringAttr("kp_embedding_action_id_gather")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> argSpecs;
    argSpecs.push_back(
        {"dynamic", getValueName(match.indices1), match.indices1.getType()});
    argSpecs.push_back(
        {"fixed", getValueName(match.params), match.params.getType()});
    argSpecs.push_back(
        {"dynamic", getValueName(match.indices2), match.indices2.getType()});
    argSpecs.push_back(
        {"dynamic", getValueName(match.packDim), match.packDim.getType()});
    argSpecs.push_back(
        {"dynamic", getValueName(match.pack), match.pack.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), argSpecs)));
    SmallVector<FusionArgSpec> outputSpecs;
    outputSpecs.push_back(
        {"output", anchorName, match.outputBuffer.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputSpecs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("mlir_ciface")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order",
        makeI64Array(rewriter.getContext(), {0, 1, 2, 3, 4, 5})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr(
        "symbolic_signature", rewriter.getStringAttr("F:1|S:4;?,?,?,?")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
        DictionaryAttr::get(rewriter.getContext(), metadata));

    rewriter.setInsertionPoint(anchor);
    rewriter.create<func::CallOp>(
        anchor.getLoc(), kernelFunc,
        ValueRange{match.indices1, match.params, match.indices2,
                   match.packDim, match.pack, match.outputBuffer});

    Operation *outputBufferDef = match.outputBuffer.getDefiningOp();
    rewriter.replaceAllUsesWith(anchor.getResult(), match.outputBuffer);
    eraseDeadFusionOps(rewriter, match.kernelOps, outputBufferDef);
    return success();
  }
};


// ── KP sparse segment reduce (一级, Execution V2) ────────────────────────────
//
// anchor = StridedSlice(shrink=1) ← Shape ← SparseSegmentMean|Sum(
//     segment_ids = StridedSlice(shrink=2, keys))
// 2 输出: output f32[batch,E] (batch = max(seg_id)+1 值依赖) + slice_output
// i32 标量。V2 ABI 原生解决值依赖形状; combiner 与 Tidx 走多名字特化。

struct KpSparseSegmentReduceMatch {
  Value data;
  Value indices;
  Value keys;
  Value begin;
  Value begin1;
  int64_t combiner;  // 0=SUM, 1=MEAN
  Value ssResult;
  Value sliceResult;
  StridedSliceOp anchor;
  Operation *ssOp = nullptr;
};

static LogicalResult matchKpSparseSegmentReduce(
    StridedSliceOp anchor, KpSparseSegmentReduceMatch &match) {
  // anchor: shrink=1, begin 常量 1-D 1 元素 (值运行时传给 kernel)。
  if (anchor.getShrinkAxisMask() != 1) {
    return failure();
  }
  if (anchor.getEllipsisMask() != 0 || anchor.getNewAxisMask() != 0)
    return failure();
  {
    auto t = dyn_cast<atir::TensorType>(anchor.getBegin().getType());
    if (!t || t.getShape().size() != 1 || t.getShape()[0] != 1)
      return failure();
  }

  auto shape = anchor.getInput().getDefiningOp<ShapeOp>();
  if (!shape) {
    return failure();
  }

  // ss_reduce: SparseSegmentMean (combiner=1) 或 SparseSegmentSum (0)。
  Value dataVal, indicesVal, segmentIdsVal, numSegmentsVal, ssResult;
  int64_t combiner = -1;
  if (auto m = shape.getInput().getDefiningOp<SparseSegmentMeanOp>()) {
    match.ssOp = m.getOperation();
    combiner = 1;
    dataVal = m.getData();
    indicesVal = m.getIndices();
    segmentIdsVal = m.getSegmentIds();
    numSegmentsVal = m.getNumSegments();
    ssResult = m.getResult();
  } else if (auto s2 = shape.getInput().getDefiningOp<SparseSegmentSumOp>()) {
    match.ssOp = s2.getOperation();
    combiner = 0;
    dataVal = s2.getInput();
    indicesVal = s2.getIndices();
    segmentIdsVal = s2.getSegmentIds();
    numSegmentsVal = s2.getNumSegments();
    ssResult = s2.getResult();
  }
  if (combiner < 0) {
    return failure();
  }
  // numSegments 必须 0 (batch 从 segment_ids 推断; kernel 语义)。
  if (!isConstantInt(numSegmentsVal, 0)) {
    return failure();
  }

  // 图侧常见 Cast(i32) ← StridedSlice(i64) 包装 (serving 导出把 seg_ids 转
  // i32): 接受 Cast 在外一层, Cast 进融合区, 边界 keys 仍取 ss 的输入
  // (i64 2D 契约不变)。Cast 的 DstT 不查 (SparseSegmentSum/Mean 的
  // Tsegmentids 语义允许 i32/i64)。
  auto keysSs = segmentIdsVal.getDefiningOp<StridedSliceOp>();
  if (!keysSs) {
    auto segCast = segmentIdsVal.getDefiningOp<CastOp>();
    if (!segCast) return failure();
    keysSs = segCast.getInput().getDefiningOp<StridedSliceOp>();
    if (!keysSs) return failure();
  }
  if (keysSs.getShrinkAxisMask() != 2) return failure();
  if (keysSs.getBeginMask() != 1 || keysSs.getEndMask() != 1) return failure();
  if (keysSs.getEllipsisMask() != 0 || keysSs.getNewAxisMask() != 0)
    return failure();

  if (!areConstantInts(keysSs.getStrides(), {1, 1})) {
    return failure();
  }
  {
    auto t = dyn_cast<atir::TensorType>(keysSs.getBegin().getType());
    if (!t || t.getShape().size() != 1 || t.getShape()[0] != 2)
      return failure();
  }

  // dtype contract: data f32 2D; keys i64 2D; indices i32/i64 1D.
  auto dType = dyn_cast<atir::TensorType>(dataVal.getType());
  auto kType = dyn_cast<atir::TensorType>(keysSs.getInput().getType());
  auto iType = dyn_cast<atir::TensorType>(indicesVal.getType());
  if (!dType || !dType.getElementType().isF32() ||
      dType.getShape().size() != 2)
    return failure();
  if (!kType || !kType.getElementType().isInteger(64) ||
      kType.getShape().size() != 2)
    return failure();
  if (!iType || !(iType.getElementType().isInteger(32) ||
                  iType.getElementType().isInteger(64)) ||
      iType.getShape().size() != 1)
    return failure();

  // Single-output contract: intermediates may not escape (except the two
  // fused outputs consumed by the enclosing func return).  keysSs result may
  // feed ssOp directly or through a Cast wrapper (Cast is inside the fused
  // region).
  for (Operation *user : keysSs.getResult().getUsers()) {
    if (user == match.ssOp) continue;
    if (isa<CastOp>(user)) {
      for (Operation *castUser : user->getResult(0).getUsers()) {
        if (castUser != match.ssOp) return failure();
      }
      continue;
    }
    return failure();
  }
  for (Operation *user : shape.getResult().getUsers()) {
    if (user != anchor.getOperation()) return failure();
  }

  match.data = dataVal;
  match.indices = indicesVal;
  match.keys = keysSs.getInput();
  match.begin = keysSs.getBegin();
  match.begin1 = anchor.getBegin();
  match.combiner = combiner;
  match.ssResult = ssResult;
  match.sliceResult = anchor.getResult();
  match.anchor = anchor;

  return success();
}

// Multi-name scheme: combiner and Tidx are kernel specializations without an
// attrs channel in the B KernelRegistry, so each specialization gets its own
// op name.
static std::string getKpSparseSegmentReduceOpName(int64_t combiner,
                                                  Type indicesType) {
  auto tensorType = dyn_cast<atir::TensorType>(indicesType);
  std::string name = "KPFusedSparseSegmentReduce";
  name += (tensorType && tensorType.getElementType().isInteger(64)) ? "I64"
                                                                    : "I32";
  name += (combiner == 1) ? "Mean" : "Sum";
  return name;
}

static CustomOpSchema getKpSparseSegmentReduceSchema(StringRef opName) {
  return CustomOpSchema::get(opName)
      .TypeVar("T")
      .TypeVar("Tidx")
      .MemRefArg("keys", 2, "")
      .MemRefArg("begin", 1, "")
      .MemRefArg("data", 2, "T")
      .MemRefArg("indices", 1, "Tidx")
      .MemRefArg("begin_1", 1, "")
      .Result("output", 2, "T")
      .Result("slice_output", 0, "");
}

struct FuseKpSparseSegmentReduceAsFuncCallPattern
    : public OpRewritePattern<StridedSliceOp> {
  using OpRewritePattern<StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StridedSliceOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor->hasAttr("annc.fusion_materialized")) return failure();
    KpSparseSegmentReduceMatch match;
    if (failed(matchKpSparseSegmentReduce(anchor, match))) return failure();
    ModuleOp module = anchor->getParentOfType<ModuleOp>();
    if (!module) return failure();

    SmallVector<Value, 5> boundaryInputs = {match.keys, match.begin,
                                            match.data, match.indices,
                                            match.begin1};
    SmallVector<Value, 2> boundaryOutputs = {match.ssResult,
                                             match.sliceResult};
    SmallVector<Operation *> fusedOps;
    SmallPtrSet<Operation *, 32> visited;
    for (Value output : boundaryOutputs) {
      collectDefiningOpsPostOrder(output.getDefiningOp(), boundaryInputs,
                                  visited, fusedOps);
    }
    if (fusedOps.empty() || !isClosedKernelOpSet(fusedOps, boundaryInputs))
      return failure();

    std::string customOpName =
        getKpSparseSegmentReduceOpName(match.combiner, match.indices.getType());
    auto schema = getKpSparseSegmentReduceSchema(customOpName);
    SmallVector<Type> inputTypes;
    for (Value input : boundaryInputs) inputTypes.push_back(input.getType());
    SmallVector<Type> outputTypes;
    for (Value output : boundaryOutputs)
      outputTypes.push_back(output.getType());
    auto inferred = inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(inputTypes),
        TypeRange(outputTypes));
    if (!inferred) {
      llvm::consumeError(inferred.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = customOpName;
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*inferred);
    if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

    std::string anchorName = getTfName(match.anchor);
    std::string kernelName = uniquifySymbolName(
        module,
        getStableFusionKernelName(anchorName, "kp_sparse_segment_reduce"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/kp_fused"));
    auto kernelFunc = createExecutionV2KernelFunc(
        module, rewriter, kernelName, fusedOps, boundaryInputs,
        boundaryOutputs);
    kernelFunc->setAttr("fusion.pattern",
                        rewriter.getStringAttr("kp_sparse_segment_reduce"));
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("kp_sparse_segment_reduce")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> args;
    for (Value input : boundaryInputs)
      args.push_back({"dynamic", getValueName(input), input.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), args)));
    SmallVector<FusionArgSpec> outputs;
    auto outputName = [](Value value, StringRef fallback) {
      std::string name = getFusionOutputName(value);
      return name.empty() ? fallback.str() : name;
    };
    outputs.push_back({"output", outputName(match.ssResult, "ss_mean:0"),
                       match.ssResult.getType()});
    outputs.push_back({"output", outputName(match.sliceResult, "slice_out:0"),
                       match.sliceResult.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                             rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));
    anchor->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
    return success();
  }
};


// ── KP embedding padding (一级, Execution V2) ────────────────────────────────
//
// 公共链: Cast(origin_shape) ─► StridedSlice ─► Sub ─► Pack ─► Fill ─►
//         ConcatV2(data, fill) ─► Reshape
// Padding:    anchor = Reshape (用户是 ConcatV2), 2 输出: sub 标量
//             (padding_rows) + anchor 结果 (padded data 2D f32)
// PaddingFast: anchor = StridedSlice(shrink=1) ← Shape ← Reshape, 2 输出:
//             sub 标量 (padding_rows) + anchor 标量 (reshape_rows)
// reshape_rows 值依赖 (origin_shape[0] - input_rows) —— V2 原生解决。
// 移植补查 (812 未查): Fill value=0、ConcatV2 axis=0、ss begin/end/strides。

struct KpEmbeddingPaddingMatch {
  Value originShape;
  Value data;
  Value inputRows;
  Value reshapeSizes;
  Value pack;
  Value subResult;
  Value anchorResult;
  Operation *anchorOp = nullptr;
};

// 从 concat 反链匹配 Cast→StridedSlice→Sub→Pack→Fill 并填充 match。
// 从 Sub.x 出发剥掉 Cast / StridedSlice 外壳, 还原 originShape。
// 两种图侧顺序都还原为同一语义 int32(originShape[0]):
//   规范序 (cvr 12 处): Sub.x <- ss{0}  <- Cast          <- originShape
//   镜像序 (presort)  : Sub.x <- Cast   <- ss{0} <- ss{0,0} <- originShape
// 剥掉的层进融合区 (originShape 是边界输入, 逐级检查由 collectDefiningOps-
// PostOrder + isClosedKernelOpSet 兜底)。失败返回空 Value。
static Value peelKpEmbeddingPaddingOriginShape(Value subX) {
  Value cur = subX;
  int numCast = 0;
  int numSlice = 0;
  while (true) {
    Operation *def = cur.getDefiningOp();
    if (auto castOp = dyn_cast_or_null<CastOp>(def)) {
      if (++numCast > 1) return Value();
      cur = castOp.getInput();
      continue;
    }
    auto ssOp = dyn_cast_or_null<StridedSliceOp>(def);
    if (!ssOp) break;
    if (++numSlice > 2) return Value();
    if (ssOp.getShrinkAxisMask() != 1) return Value();
    // 1 元素形态: 原有检查原样保留 (不加严, 避免回归 cvr 现有命中)。
    bool oneElem = areConstantInts(ssOp.getBegin(), {0}) &&
                   areConstantInts(ssOp.getEnd(), {1}) &&
                   areConstantInts(ssOp.getStrides(), {1});
    // 2 元素形态: 只在 new_axis_mask=2 下等价 —— spec 项 0 对应输入轴 0
    // (shrink 取 [0]), 项 1 是新插入轴, 结果 [1] 承载 originShape[0]。
    // 若 new_axis_mask=0, 项 1 会落到输入轴 1 且 begin=end=0 → 空切片,
    // 外层再取 [0] 即越界, 故必须查死。
    bool twoElem = ssOp.getNewAxisMask() == 2 && ssOp.getEllipsisMask() == 0 &&
                   areConstantInts(ssOp.getBegin(), {0, 0}) &&
                   areConstantInts(ssOp.getEnd(), {1, 0}) &&
                   areConstantInts(ssOp.getStrides(), {1, 1});
    if (!oneElem && !twoElem) return Value();
    cur = ssOp.getInput();
  }
  if (numSlice < 1) return Value();
  return cur;
}

static LogicalResult matchKpEmbeddingPaddingChain(
    ConcatV2Op concat, KpEmbeddingPaddingMatch &match) {
  if (concat.getValues().size() != 2) return failure();
  if (!isConstantInt(concat.getAxis(), 0)) return failure();

  auto fill = concat.getValues()[1].getDefiningOp<FillOp>();
  if (!fill || !isConstantZero(fill.getValueInput())) return failure();
  auto pack = fill.getShapeInput().getDefiningOp<PackOp>();
  if (!pack || pack.getInputs().size() != 2) return failure();
  if (pack.getAxis() != 0) return failure();

  // pack 的两个输入: 恰一个 Const (pack 常量), 另一个 Sub。
  Value subVal, packVal;
  if (pack.getInputs()[0].getDefiningOp<SubOp>()) {
    subVal = pack.getInputs()[0];
    packVal = pack.getInputs()[1];
  } else if (pack.getInputs()[1].getDefiningOp<SubOp>()) {
    subVal = pack.getInputs()[1];
    packVal = pack.getInputs()[0];
  } else {
    return failure();
  }
  auto sub = subVal.getDefiningOp<SubOp>();
  if (!sub) return failure();
  Value originShape = peelKpEmbeddingPaddingOriginShape(sub.getX());
  if (!originShape) return failure();

  match.originShape = originShape;
  match.data = concat.getValues()[0];
  match.inputRows = sub.getY();
  match.pack = packVal;
  match.subResult = sub.getResult();
  return success();
}

static LogicalResult matchKpEmbeddingPadding(
    ReshapeOp anchor, KpEmbeddingPaddingMatch &match) {
  // anchor 结果必须有 ConcatV2 用户 (812 检查)。
  bool hasConcatUser = false;
  for (Operation *user : anchor.getResult().getUsers()) {
    if (isa<ConcatV2Op>(user)) hasConcatUser = true;
  }
  if (!hasConcatUser) {
    return failure();
  }

  auto concat = anchor.getInput().getDefiningOp<ConcatV2Op>();
  if (!concat) {
    return failure();
  }
  if (failed(matchKpEmbeddingPaddingChain(concat, match))) {
    return failure();
  }
  match.reshapeSizes = anchor.getTargetShape();
  match.anchorResult = anchor.getResult();
  match.anchorOp = anchor.getOperation();

  // dtype contract。
  auto oType = dyn_cast<atir::TensorType>(match.originShape.getType());
  auto dType = dyn_cast<atir::TensorType>(match.data.getType());
  if (!oType || !oType.getElementType().isInteger(64) ||
      oType.getShape().size() != 1)
    return failure();
  if (!dType || !dType.getElementType().isF32() ||
      dType.getShape().size() != 2)
    return failure();

  // 逃逸检查: 中间结果不逃逸 (concat 的结果只被 anchor 用; anchor 结果逃逸
  // 是输出)。
  for (Operation *user : concat.getResult().getUsers()) {
    if (user != anchor.getOperation()) return failure();
  }

  return success();
}

static LogicalResult matchKpEmbeddingPaddingFast(
    StridedSliceOp anchor, KpEmbeddingPaddingMatch &match) {
  if (anchor.getShrinkAxisMask() != 1) {
    return failure();
  }
  // anchor 必须取 shape 的第 0 元素 (kernel 输出 reshape_rows = shape[0])。
  // 812 只查 begin/end 形状不查值 — 移植补查。
  if (!areConstantInts(anchor.getBegin(), {0}) ||
      !areConstantInts(anchor.getEnd(), {1}) ||
      !areConstantInts(anchor.getStrides(), {1}))
    return failure();
  auto beginTy = dyn_cast<atir::TensorType>(anchor.getBegin().getType());
  auto endTy = dyn_cast<atir::TensorType>(anchor.getEnd().getType());
  if (!beginTy || !endTy || beginTy.getShape().size() != 1 ||
      beginTy.getShape()[0] != 1 || endTy.getShape().size() != 1 ||
      endTy.getShape()[0] != 1)
    return failure();

  auto shape = anchor.getInput().getDefiningOp<ShapeOp>();
  if (!shape) return failure();
  auto reshape = shape.getInput().getDefiningOp<ReshapeOp>();
  if (!reshape) return failure();
  auto concat = reshape.getInput().getDefiningOp<ConcatV2Op>();
  if (!concat) return failure();
  if (failed(matchKpEmbeddingPaddingChain(concat, match))) return failure();
  match.reshapeSizes = reshape.getTargetShape();
  match.anchorResult = anchor.getResult();
  match.anchorOp = anchor.getOperation();

  auto oType = dyn_cast<atir::TensorType>(match.originShape.getType());
  auto dType = dyn_cast<atir::TensorType>(match.data.getType());
  if (!oType || !oType.getElementType().isInteger(64) ||
      oType.getShape().size() != 1)
    return failure();
  if (!dType || !dType.getElementType().isF32() ||
      dType.getShape().size() != 2)
    return failure();

  for (Operation *user : concat.getResult().getUsers()) {
    if (user != reshape.getOperation()) return failure();
  }
  for (Operation *user : reshape.getResult().getUsers()) {
    if (user != shape.getOperation()) return failure();
  }
  for (Operation *user : shape.getResult().getUsers()) {
    if (user != anchor.getOperation()) return failure();
  }

  return success();
}

static CustomOpSchema getKpEmbeddingPaddingSchema(StringRef opName,
                                                  bool fast) {
  auto schema = CustomOpSchema::get(opName)
                    .TypeVar("T")
                    .MemRefArg("origin_shape", 1, "")
                    .MemRefArg("input_rows", 0, "")
                    .MemRefArg("pack", 0, "")
                    .MemRefArg("data", 2, "T")
                    .MemRefArg("reshape_sizes", 1, "");
  schema.Result("out_padding_rows", 0, "");
  if (fast) {
    schema.Result("out_reshape_rows", 0, "");
  } else {
    schema.Result("out_data", 2, "T");
  }
  return schema;
}

static LogicalResult fuseKpEmbeddingPadding(
    Operation *anchorOp, KpEmbeddingPaddingMatch &match, bool fast,
    StringRef patternName, PatternRewriter &rewriter) {
  ModuleOp module = anchorOp->getParentOfType<ModuleOp>();
  if (!module) return failure();

  SmallVector<Value, 5> boundaryInputs = {match.originShape, match.inputRows,
                                          match.pack, match.data,
                                          match.reshapeSizes};
  SmallVector<Value, 2> boundaryOutputs = {match.subResult,
                                           match.anchorResult};
  SmallVector<Operation *> fusedOps;
  SmallPtrSet<Operation *, 32> visited;
  for (Value output : boundaryOutputs) {
    collectDefiningOpsPostOrder(output.getDefiningOp(), boundaryInputs,
                                visited, fusedOps);
  }
  if (fusedOps.empty() || !isClosedKernelOpSet(fusedOps, boundaryInputs))
    return failure();

  const char *customOpName =
      fast ? "KPFusedEmbeddingPaddingFast" : "KPFusedEmbeddingPadding";
  auto schema = getKpEmbeddingPaddingSchema(customOpName, fast);
  SmallVector<Type> inputTypes;
  for (Value input : boundaryInputs) inputTypes.push_back(input.getType());
  SmallVector<Type> outputTypes;
  for (Value output : boundaryOutputs) outputTypes.push_back(output.getType());
  auto inferred = inferTypeConstraintsFromSchema(
      schema.toMetadata(rewriter.getContext()), TypeRange(inputTypes),
      TypeRange(outputTypes));
  if (!inferred) {
    llvm::consumeError(inferred.takeError());
    return failure();
  }
  annc::kernels::KernelResolveRequest request;
  request.op_type = customOpName;
  request.abi = "annc_execution_v2";
  request.type_constraints = std::move(*inferred);
  if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

  std::string anchorName = getTfName(anchorOp);
  std::string kernelName = uniquifySymbolName(
      module, getStableFusionKernelName(anchorName, patternName));
  std::string clusterName = uniquifyFusionName(
      module, sanitizeName(anchorName + "/kp_fused"));
  auto kernelFunc = createExecutionV2KernelFunc(
      module, rewriter, kernelName, fusedOps, boundaryInputs, boundaryOutputs);
  kernelFunc->setAttr("fusion.pattern", rewriter.getStringAttr(patternName));
  setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

  SmallVector<NamedAttribute> metadata;
  metadata.push_back(rewriter.getNamedAttr(
      "fusion.pattern", rewriter.getStringAttr(patternName)));
  addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
  metadata.push_back(rewriter.getNamedAttr(
      "kernel_name", rewriter.getStringAttr(kernelName)));
  metadata.push_back(rewriter.getNamedAttr(
      "tf.name", rewriter.getStringAttr(clusterName)));
  SmallVector<FusionArgSpec> args;
  for (Value input : boundaryInputs)
    args.push_back({"dynamic", getValueName(input), input.getType()});
  metadata.push_back(rewriter.getNamedAttr(
      "args", makeFusionArgArray(rewriter.getContext(), args)));
  SmallVector<FusionArgSpec> outputs;
  auto outputName = [](Value value, StringRef fallback) {
    std::string name = getFusionOutputName(value);
    return name.empty() ? fallback.str() : name;
  };
  outputs.push_back({"output",
                     outputName(match.subResult, "padding_rows:0"),
                     match.subResult.getType()});
  outputs.push_back(
      {"output", outputName(match.anchorResult, anchorName + ":0"),
       match.anchorResult.getType()});
  metadata.push_back(rewriter.getNamedAttr(
      "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
  metadata.push_back(rewriter.getNamedAttr(
      "abi", rewriter.getStringAttr("annc_execution_v2")));
  metadata.push_back(rewriter.getNamedAttr(
      "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
  metadata.push_back(rewriter.getNamedAttr(
      "dynamic_dims", makeI64Array(rewriter.getContext(), {1})));
  metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                           rewriter.getStringAttr("")));
  metadata.push_back(rewriter.getNamedAttr(
      "fallback_function", rewriter.getStringAttr("original_subgraph")));
  kernelFunc->setAttr("fusion.metadata",
                      DictionaryAttr::get(rewriter.getContext(), metadata));
  anchorOp->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
  return success();
}

struct FuseKpEmbeddingPaddingAsFuncCallPattern
    : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor->hasAttr("annc.fusion_materialized")) return failure();
    KpEmbeddingPaddingMatch match;
    if (failed(matchKpEmbeddingPadding(anchor, match))) return failure();
    return fuseKpEmbeddingPadding(anchor.getOperation(), match,
                                  /*fast=*/false, "kp_embedding_padding",
                                  rewriter);
  }
};

struct FuseKpEmbeddingPaddingFastAsFuncCallPattern
    : public OpRewritePattern<StridedSliceOp> {
  using OpRewritePattern<StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StridedSliceOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor->hasAttr("annc.fusion_materialized")) return failure();
    KpEmbeddingPaddingMatch match;
    if (failed(matchKpEmbeddingPaddingFast(anchor, match))) return failure();
    return fuseKpEmbeddingPadding(anchor.getOperation(), match,
                                  /*fast=*/true, "kp_embedding_padding_fast",
                                  rewriter);
  }
};


// ── KP sparse reshape (一级, Execution V2) ──────────────────────────────────
//
// 匹配链 (与二级 KpEmbeddingSparseReshapeRewrite 一致):
//   keys ──┬─► Shape ─► StridedSlice(shrink=1) ─► Pack ◄─ pack_const
//          │                                        │
//          │  input_shape ◄─ Cast ◄─────────────────┘
//          │
//          ├─► StridedSlice(shrink=2, begin_mask=1, end_mask=1) ─► Reshape{-1,1} ─┐
//          │                                                                     ├─► ConcatV2(axis={-1}) ─► indices
//          └─► Range(0, limit, 1) ─► Reshape{-1,1} ─► Cast ──────────────────────┘
//   new_shape ───────────────────────────────────► SparseReshape (anchor)
//
// anchor 是 functional op (operand 0 = indices, 无输出 buffer), 故 kernel
// func 用下方变体 helper (clone 时跳过 SparseReshapeOp 的 dest-style buffer
// 映射)。输出: out_indices i64[R,rank] (R 值依赖) + out_shape i64[rank]。
// T = pack_const 元素类型 (i32/i64, 多名字特化); new_shape 固定 i64。

struct KpSparseReshapeMatch {
  Value keys;
  Value begin;
  Value newShape;
  Value packConst;
  Value rangeLimit;  // 动态 Range.limit 边界 (常量为 null)
  Value outIndicesResult;
  Value outShapeResult;
  SparseReshapeOp anchor;
};

static LogicalResult matchKpSparseReshape(SparseReshapeOp anchor,
                                          KpSparseReshapeMatch &match) {
  // ── indices 分支: ConcatV2([cast_1, reshape], axis={-1}) ──
  auto concat = anchor.getIndices().getDefiningOp<ConcatV2Op>();
  if (!concat) return failure();
  if (concat.getValues().size() != 2) return failure();
  if (!isConstantInt(concat.getAxis(), -1)) return failure();

  // 分支 A: Cast(Reshape(Range(0, limit, 1), {-1,1})) — 常量行号列。
  auto cast1 = concat.getValues()[0].getDefiningOp<CastOp>();
  if (!cast1) return failure();
  auto reshape1 = cast1.getInput().getDefiningOp<ReshapeOp>();
  if (!reshape1) return failure();
  if (!areConstantInts(reshape1.getTargetShape(), {-1, 1})) return failure();
  auto range = reshape1.getInput().getDefiningOp<RangeOp>();
  if (!range) return failure();
  // 812 只查 delta=1; 移植补查 start=0。limit 允许常量或 0-D i32 动态值
  // (varlen batch 图: StridedSlice(Shape(x)) 提取); 动态时作为额外边界输入
  // 传入 kernel func——kernel 不用 limit 值, 内部自行重生成行号列。
  if (!isConstantInt(range.getStart(), 0)) return failure();
  if (!isConstantInt(range.getDelta(), 1)) return failure();
  bool limitIsConst = range.getLimit().getDefiningOp<ConstantOp>() != nullptr;
  if (!limitIsConst) {
    auto lType = dyn_cast<atir::TensorType>(range.getLimit().getType());
    if (!lType || !lType.getElementType().isInteger(32) ||
        lType.getShape().size() != 0)
      return failure();
  }
  // 分支 B: Reshape(StridedSlice(keys, begin), {-1,1}) — keys 列。
  auto reshape = concat.getValues()[1].getDefiningOp<ReshapeOp>();
  if (!reshape) return failure();
  if (!areConstantInts(reshape.getTargetShape(), {-1, 1})) return failure();
  auto ss = reshape.getInput().getDefiningOp<StridedSliceOp>();
  if (!ss) return failure();
  if (ss.getShrinkAxisMask() != 2) return failure();
  if (ss.getBeginMask() != 1 || ss.getEndMask() != 1) return failure();
  if (ss.getEllipsisMask() != 0 || ss.getNewAxisMask() != 0) return failure();
  if (!ss.getEnd().getDefiningOp<ConstantOp>() ||
      !ss.getStrides().getDefiningOp<ConstantOp>())
    return failure();
  {
    auto t = dyn_cast<atir::TensorType>(ss.getBegin().getType());
    if (!t || t.getShape().size() != 1 || t.getShape()[0] != 2)
      return failure();
  }

  // ── input_shape 分支: Cast(Pack([StridedSlice(shrink=1)(Shape(keys)),
  // pack_const])) ──
  auto cast = anchor.getInputShape().getDefiningOp<CastOp>();
  if (!cast) return failure();
  auto pack = cast.getInput().getDefiningOp<PackOp>();
  if (!pack || pack.getInputs().size() != 2) return failure();
  if (pack.getAxis() != 0) return failure();
  auto ss1 = pack.getInputs()[0].getDefiningOp<StridedSliceOp>();
  if (!ss1 || ss1.getShrinkAxisMask() != 1) return failure();
  if (!areConstantInts(ss1.getBegin(), {0}) ||
      !areConstantInts(ss1.getEnd(), {1}) ||
      !areConstantInts(ss1.getStrides(), {1}))
    return failure();
  auto shape = ss1.getInput().getDefiningOp<ShapeOp>();
  if (!shape) return failure();

  // D-2 放宽 2: 两个分支交叉引用同一个 num_rows。ss1 = Shape(keys)[0] 既喂
  // input_shape 分支的 Pack, 又当 indices 分支 Range 的 limit (cvr history /
  // hmv session 族实测)。这是链内交叉引用而非逃逸——两端都在融合区内。
  // 此时 limit 不作边界输入: 它就是 num_rows, kernel 从 slice_input->sizes[0]
  // 已能得到, 故走非 DynLimit 特化。
  bool limitFromSs1 = (range.getLimit() == ss1.getResult());

  match.keys = shape.getInput();
  match.begin = ss.getBegin();
  match.newShape = anchor.getNewShape();
  match.packConst = pack.getInputs()[1];
  match.rangeLimit =
      (limitIsConst || limitFromSs1) ? Value() : range.getLimit();
  match.anchor = anchor;
  match.outIndicesResult = anchor.getOutputIndices();
  match.outShapeResult = anchor.getOutputShape();

  // dtype 契约: keys i64 2D; begin i32 1D 2 元素; new_shape 固定 i64; T =
  // pack_const (i32/i64); 两处 Cast 结果为 i64。
  auto kType = dyn_cast<atir::TensorType>(match.keys.getType());
  auto bType = dyn_cast<atir::TensorType>(match.begin.getType());
  auto nType = dyn_cast<atir::TensorType>(match.newShape.getType());
  auto pType = dyn_cast<atir::TensorType>(match.packConst.getType());
  if (!kType || !kType.getElementType().isInteger(64) ||
      kType.getShape().size() != 2)
    return failure();
  if (!bType || !bType.getElementType().isInteger(32) ||
      bType.getShape().size() != 1 || bType.getShape()[0] != 2)
    return failure();
  if (!nType || !nType.getElementType().isInteger(64) ||
      nType.getShape().size() != 1 || nType.getShape()[0] != 2)
    return failure();
  if (!pType || !(pType.getElementType().isInteger(32) ||
                  pType.getElementType().isInteger(64)) ||
      pType.getShape().size() != 0)
    return failure();
  {
    auto t1 = dyn_cast<atir::TensorType>(cast1.getResult().getType());
    auto t2 = dyn_cast<atir::TensorType>(cast.getResult().getType());
    if (!t1 || !t1.getElementType().isInteger(64) || !t2 ||
        !t2.getElementType().isInteger(64))
      return failure();
  }

  // 逃逸检查: 链中间结果只被链内下一 op 使用 (anchor 结果逃逸是输出)。
  for (Operation *user : range.getResult().getUsers()) {
    if (user != reshape1.getOperation()) return failure();
  }
  for (Operation *user : reshape1.getResult().getUsers()) {
    if (user != cast1.getOperation()) return failure();
  }
  for (Operation *user : cast1.getResult().getUsers()) {
    if (user != concat.getOperation()) return failure();
  }
  for (Operation *user : ss.getResult().getUsers()) {
    if (user != reshape.getOperation()) return failure();
  }
  for (Operation *user : reshape.getResult().getUsers()) {
    if (user != concat.getOperation()) return failure();
  }
  for (Operation *user : concat.getResult().getUsers()) {
    if (user != anchor.getOperation()) return failure();
  }
  for (Operation *user : shape.getResult().getUsers()) {
    if (user != ss1.getOperation()) return failure();
  }
  for (Operation *user : ss1.getResult().getUsers()) {
    if (user == pack.getOperation()) continue;
    // 链内交叉引用: indices 分支的 Range 用 ss1 当 limit (见上 limitFromSs1)。
    if (limitFromSs1 && user == range.getOperation()) continue;
    return failure();
  }
  for (Operation *user : pack.getResult().getUsers()) {
    if (user != cast.getOperation()) return failure();
  }
  for (Operation *user : cast.getResult().getUsers()) {
    if (user != anchor.getOperation()) return failure();
  }
  // anchor 结果必须有主图用户 (TF 图中 SparseReshape 输出必有消费者)。
  if (anchor.getOutputIndices().use_empty() &&
      anchor.getOutputShape().use_empty())
    return failure();

  return success();
}

// Multi-name scheme: T (pack_const element type) is a kernel specialization;
// the DynLimit variants take an extra unused 0-D i32 range_limit boundary
// (dynamic Range.limit in varlen-batch graphs).
static std::string getKpSparseReshapeOpName(Type packConstType,
                                            bool hasRangeLimit) {
  auto tensorType = dyn_cast<atir::TensorType>(packConstType);
  std::string name = "KPFusedSparseReshape";
  name +=
      (tensorType && tensorType.getElementType().isInteger(64)) ? "I64" : "I32";
  if (hasRangeLimit) name += "DynLimit";
  return name;
}

static CustomOpSchema getKpSparseReshapeSchema(StringRef opName,
                                               bool hasRangeLimit) {
  auto schema = CustomOpSchema::get(opName)
                    .TypeVar("T")
                    .MemRefArg("slice_input", 2, "")
                    .MemRefArg("begin", 1, "")
                    .MemRefArg("pack_const", 0, "T")
                    .MemRefArg("new_shape", 1, "");
  if (hasRangeLimit) schema.MemRefArg("range_limit", 0, "");
  return schema.Result("out_indices", 2, "").Result("out_shape", 1, "");
}

// Functional-anchor variant of createExecutionV2KernelFunc: SparseReshapeOp
// takes its semantic input as operand 0 (no dest-style output buffer), so the
// buffer mapping is skipped for it during clone.
static func::FuncOp createKpSparseReshapeV2KernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    ArrayRef<Operation *> fusedOps, ArrayRef<Value> boundaryInputs,
    ArrayRef<Value> boundaryOutputs) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {
      LLVM::LLVMPointerType::get(rewriter.getContext())};
  for (Value input : boundaryInputs) inputTypes.push_back(input.getType());
  SmallVector<Type> resultTypes;
  for (Value output : boundaryOutputs) resultTypes.push_back(output.getType());

  auto funcType = rewriter.getFunctionType(inputTypes, resultTypes);
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  IRMapping mapper;
  for (auto [index, input] : llvm::enumerate(boundaryInputs)) {
    mapper.map(input, entry->getArgument(index + 1));
  }
  for (Operation *op : fusedOps) {
    if (!isa<UniqueOp, ConstantOp, BufferOp, SparseReshapeOp>(op)) {
      Value outputBuffer = op->getOperand(0);
      if (!mapper.contains(outputBuffer)) {
        Value localBuffer =
            rewriter.create<BufferOp>(func.getLoc(), outputBuffer.getType());
        mapper.map(outputBuffer, localBuffer);
      }
    }
    rewriter.clone(*op, mapper);
  }

  SmallVector<Value> returns;
  for (Value output : boundaryOutputs) returns.push_back(mapper.lookup(output));
  rewriter.create<func::ReturnOp>(func.getLoc(), returns);
  return func;
}

struct FuseKpSparseReshapeAsFuncCallPattern
    : public OpRewritePattern<SparseReshapeOp> {
  using OpRewritePattern<SparseReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SparseReshapeOp anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor->hasAttr("annc.fusion_materialized")) return failure();
    KpSparseReshapeMatch match;
    if (failed(matchKpSparseReshape(anchor, match))) return failure();
    ModuleOp module = anchor->getParentOfType<ModuleOp>();
    if (!module) return failure();

    // 边界顺序 = kernel wrapper 序 (slice_input, begin, pack_const,
    // new_shape[, range_limit]), 二级 getOrderedBoundaryInputs 与此一致。
    SmallVector<Value, 4> boundaryInputs = {match.keys, match.begin,
                                            match.packConst, match.newShape};
    bool hasRangeLimit = static_cast<bool>(match.rangeLimit);
    if (hasRangeLimit) boundaryInputs.push_back(match.rangeLimit);
    SmallVector<Value, 2> boundaryOutputs = {match.outIndicesResult,
                                             match.outShapeResult};
    SmallVector<Operation *> fusedOps;
    SmallPtrSet<Operation *, 32> visited;
    for (Value output : boundaryOutputs) {
      collectDefiningOpsPostOrder(output.getDefiningOp(), boundaryInputs,
                                  visited, fusedOps);
    }
    if (fusedOps.empty() || !isClosedKernelOpSet(fusedOps, boundaryInputs))
      return failure();

    std::string customOpName =
        getKpSparseReshapeOpName(match.packConst.getType(), hasRangeLimit);
    auto schema = getKpSparseReshapeSchema(customOpName, hasRangeLimit);
    SmallVector<Type> inputTypes;
    for (Value input : boundaryInputs) inputTypes.push_back(input.getType());
    SmallVector<Type> outputTypes;
    for (Value output : boundaryOutputs)
      outputTypes.push_back(output.getType());
    auto inferred = inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(inputTypes),
        TypeRange(outputTypes));
    if (!inferred) {
      llvm::consumeError(inferred.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = customOpName;
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*inferred);
    if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

    std::string anchorName = getTfName(match.anchor);
    std::string kernelName = uniquifySymbolName(
        module, getStableFusionKernelName(anchorName, "kp_sparse_reshape"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/kp_fused"));
    auto kernelFunc = createKpSparseReshapeV2KernelFunc(
        module, rewriter, kernelName, fusedOps, boundaryInputs,
        boundaryOutputs);
    kernelFunc->setAttr("fusion.pattern",
                        rewriter.getStringAttr("kp_sparse_reshape"));
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("kp_sparse_reshape")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> args;
    for (Value input : boundaryInputs)
      args.push_back({"dynamic", getValueName(input), input.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), args)));
    SmallVector<FusionArgSpec> outputs;
    auto outputName = [](Value value, StringRef fallback) {
      std::string name = getFusionOutputName(value);
      return name.empty() ? fallback.str() : name;
    };
    outputs.push_back({"output",
                       outputName(match.outIndicesResult,
                                  anchorName + "/output_indices"),
                       match.outIndicesResult.getType()});
    outputs.push_back({"output",
                       outputName(match.outShapeResult,
                                  anchorName + "/output_shape"),
                       match.outShapeResult.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                             rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));
    anchor->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
    return success();
  }
};


// ── KP sparse select (一级, Execution V2) ────────────────────────────────────
//
//   a ─► Reshape{-1,1} ─► Compare(GT, gt) ─► Cast ─┐
//                                                  ├► Where(equal, (fill, ·)) ─┐
//   b ─► Reshape{-1,1} ─┬► Compare(EQ, eq1) ─► Where(equal1, (fill0, ·)) ──────┘
//                       └► Compare(EQ, eq2) ────────────────────────────────────► ConcatV2(axis=1)
//   c ─► Reshape{-1,1} ─► Compare(EQ, eq3) ─► Where(equal2, (fill2, fill1)) ───►
//   empty [?,0] (Fill) ─────────────────────────────────────────────────────────►
//
// 输出: x = reshape_4 结果 (i32 [N,1]), y = select_0 结果 (f32 [N,1]),
//       w = concat 结果 (f32 [N,2] = [y, 1.0])。N 值依赖 → V2。
// 7 个边界输入 (a/b/c + 4 比较常量) = V2 输入上限 7, 恰好卡线。

struct KpSparseSelectMatch {
  Value a;
  Value b;
  Value c;
  Value gt;   // greater 常量 (0-D i32)
  Value eq1;  // equal 常量
  Value eq2;  // equal_1 常量
  Value eq3;  // equal_2 常量
  Value xResult;  // reshape_4 结果
  Value yResult;  // select_0 结果
  Value wResult;  // concat 结果
  ConcatV2Op anchor;
};

// 匹配 Compare(reshape(x), const)（direction EQ/GT, reshape 形状 {-1,1},
// 常量 0-D i32, 常量在任一侧）。返回 ReshapeOp 或 nullptr。
static ReshapeOp matchSelectCompareAgainstConst(CompareOp cmp, StringRef dir,
                                                Value *constOut) {
  if (!cmp || cmp.getComparisonDirection() != dir) return nullptr;
  auto reshape = cmp.getLhs().getDefiningOp<ReshapeOp>();
  Value constV = cmp.getRhs();
  if (!reshape) {
    reshape = cmp.getRhs().getDefiningOp<ReshapeOp>();
    constV = cmp.getLhs();
  }
  if (!reshape) return nullptr;
  if (!areConstantInts(reshape.getTargetShape(), {-1, 1})) return nullptr;
  if (!constV.getDefiningOp<ConstantOp>()) return nullptr;
  auto t = dyn_cast<atir::TensorType>(constV.getType());
  if (!t || !t.getElementType().isInteger(32) || t.getShape().size() != 0)
    return nullptr;
  *constOut = constV;
  return reshape;
}

// 检查 Where(select) 的 then 分支 (inputs[1], TF 序 (cond, then, else)) 是
// Fill(1.0f)。
static bool isWhereThenFillOne(WhereOp where, FillOp *fillOut) {
  if (!where || where.getInputs().size() != 3) return false;
  auto fill = where.getInputs()[1].getDefiningOp<FillOp>();
  if (!fill || !isConstantFloat(fill.getValueInput(), 1.0f)) return false;
  if (fillOut) *fillOut = fill;
  return true;
}

static LogicalResult matchKpSparseSelect(ConcatV2Op anchor,
                                         KpSparseSelectMatch &match) {
  // anchor: ConcatV2 3 inputs, axis=1 常量, values[2] 静态形状 [?,0]
  // (812 图第三输入恒空, kernel 输出 w = [y, 1.0] 依赖此形态)。
  if (anchor.getValues().size() != 3) return failure();
  if (!isConstantInt(anchor.getAxis(), 1)) return failure();
  auto emptyType = dyn_cast<atir::TensorType>(anchor.getValues()[2].getType());
  if (!emptyType || !emptyType.getElementType().isF32() ||
      emptyType.getShape().size() != 2 || emptyType.getShape()[1] != 0)
    return failure();

    // left branch: select_0 = Where(equal_1, (fill_0, select_1))
  auto select0 = anchor.getValues()[0].getDefiningOp<WhereOp>();
  if (!select0 || select0.getInputs().size() != 3) return failure();
  auto equal1 = select0.getInputs()[0].getDefiningOp<CompareOp>();
  if (!equal1) return failure();
  FillOp fill0;
  if (!isWhereThenFillOne(select0, &fill0)) return failure();
  auto select1 = select0.getInputs()[2].getDefiningOp<WhereOp>();
  if (!select1 || select1.getInputs().size() != 3) return failure();
  auto equal = select1.getInputs()[0].getDefiningOp<CompareOp>();
  if (!equal) return failure();
  FillOp fill;
  if (!isWhereThenFillOne(select1, &fill)) return failure();
  auto cast = select1.getInputs()[2].getDefiningOp<CastOp>();
  if (!cast) return failure();
  {
    auto t = dyn_cast<atir::TensorType>(cast.getResult().getType());
    if (!t || !t.getElementType().isF32()) return failure();
  }
  auto greater = cast.getInput().getDefiningOp<CompareOp>();
  if (!greater) return failure();

    auto reshape4 = matchSelectCompareAgainstConst(greater, "GT", &match.gt);
  auto reshape1a = matchSelectCompareAgainstConst(equal, "EQ", &match.eq1);
  auto reshape1b = matchSelectCompareAgainstConst(equal1, "EQ", &match.eq2);
  if (!reshape4 || !reshape1a || !reshape1b || reshape1a != reshape1b)
    return failure();

    // right branch: select_2 = Where(equal_2, (fill_2, fill_1)) — 恒 1.0
  auto select2 = anchor.getValues()[1].getDefiningOp<WhereOp>();
  if (!select2 || select2.getInputs().size() != 3) return failure();
  auto equal2 = select2.getInputs()[0].getDefiningOp<CompareOp>();
  if (!equal2) return failure();
  FillOp fill2;
  if (!isWhereThenFillOne(select2, &fill2)) return failure();
  auto fill1Op = select2.getInputs()[2].getDefiningOp<FillOp>();
  if (!fill1Op || !isConstantFloat(fill1Op.getValueInput(), 1.0f))
    return failure();

  auto reshape2 = matchSelectCompareAgainstConst(equal2, "EQ", &match.eq3);
  if (!reshape2) return failure();

    match.a = reshape4.getInput();
  match.b = reshape1a.getInput();
  match.c = reshape2.getInput();
  match.anchor = anchor;

  // dtype 契约: a/b/c i32 任意形状; 常量 0-D i32 (matchCompare 已查);
  // select_0 结果与 concat 结果 f32 2D。
  for (Value v : {match.a, match.b, match.c}) {
    auto t = dyn_cast<atir::TensorType>(v.getType());
    if (!t || !t.getElementType().isInteger(32)) return failure();
  }
  {
    auto t = dyn_cast<atir::TensorType>(select0.getResult().getType());
    if (!t || !t.getElementType().isF32() || t.getShape().size() != 2)
      return failure();
    auto tw = dyn_cast<atir::TensorType>(anchor.getResult().getType());
    if (!tw || !tw.getElementType().isF32() || tw.getShape().size() != 2)
      return failure();
  }

    // 逃逸检查: 链中间结果只被链内 op 使用; x/w 输出须有主图消费者 (y 在
  // 812 图中只喂 concat, 不要求主图用户)。
  auto checkOnly = [](Value v, ArrayRef<Operation *> allowed) {
    for (Operation *user : v.getUsers()) {
      if (!llvm::is_contained(allowed, user)) return false;
    }
    return true;
  };
  // 注: reshape_4 结果 (x 输出) 在 V2 ABI 下由 kernel func return 消费,
  // 无需主图用户 (二级 checkSparseSelectEscape 的 checkOnlyOrReturn 已
  // 覆盖该契约), 故一级不再要求 x 逃逸到主图。
  if (!checkOnly(greater.getResult(), {cast.getOperation()})) return failure();
  if (!checkOnly(cast.getResult(), {select1.getOperation()})) return failure();
  // reshape_1 结果除 equal/equal_1 外还被 fill/fill_0 的 Shape 输入使用。
  {
    SmallVector<Operation *> r1Users = {equal.getOperation(),
                                        equal1.getOperation()};
    if (auto s = fill.getShapeInput().getDefiningOp<ShapeOp>())
      r1Users.push_back(s.getOperation());
    if (auto s = fill0.getShapeInput().getDefiningOp<ShapeOp>())
      r1Users.push_back(s.getOperation());
    if (!checkOnly(reshape1a.getResult(), r1Users)) return failure();
  }
  if (!checkOnly(equal.getResult(), {select1.getOperation()})) return failure();
  if (!checkOnly(equal1.getResult(), {select0.getOperation()}))
    return failure();
  if (!checkOnly(fill.getResult(), {select1.getOperation()})) return failure();
  if (!checkOnly(fill0.getResult(), {select0.getOperation()}))
    return failure();
  if (!checkOnly(select1.getResult(), {select0.getOperation()}))
    return failure();
  if (!checkOnly(select0.getResult(), {anchor.getOperation()}))
    return failure();
  // reshape_2 结果除 equal_2 外还被 fill_1/fill_2 的 Shape 输入使用。
  {
    SmallVector<Operation *> r2Users = {equal2.getOperation()};
    if (auto s = fill1Op.getShapeInput().getDefiningOp<ShapeOp>())
      r2Users.push_back(s.getOperation());
    if (auto s = fill2.getShapeInput().getDefiningOp<ShapeOp>())
      r2Users.push_back(s.getOperation());
    if (!checkOnly(reshape2.getResult(), r2Users)) return failure();
  }
  if (!checkOnly(equal2.getResult(), {select2.getOperation()}))
    return failure();
  if (!checkOnly(fill1Op.getResult(), {select2.getOperation()}))
    return failure();
  if (!checkOnly(fill2.getResult(), {select2.getOperation()}))
    return failure();
  if (!checkOnly(select2.getResult(), {anchor.getOperation()}))
    return failure();
  if (anchor.getResult().use_empty()) return failure();

  match.xResult = reshape4.getResult();
  match.yResult = select0.getResult();
  match.wResult = anchor.getResult();
  return success();
}

struct FuseKpSparseSelectAsFuncCallPattern
    : public OpRewritePattern<ConcatV2Op> {
  using OpRewritePattern<ConcatV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatV2Op anchor,
                                PatternRewriter &rewriter) const override {
    if (anchor->hasAttr("annc.fusion_materialized")) return failure();
    KpSparseSelectMatch match;
    if (failed(matchKpSparseSelect(anchor, match))) return failure();
    ModuleOp module = anchor->getParentOfType<ModuleOp>();
    if (!module) return failure();

    // 7 边界输入 = V2 上限 (恰好可用)。
    SmallVector<Value, 7> boundaryInputs = {match.a,  match.b,  match.c,
                                            match.gt, match.eq1, match.eq2,
                                            match.eq3};
    SmallVector<Value, 3> boundaryOutputs = {match.xResult, match.yResult,
                                             match.wResult};
    SmallVector<Operation *> fusedOps;
    SmallPtrSet<Operation *, 32> visited;
    for (Value output : boundaryOutputs) {
      collectDefiningOpsPostOrder(output.getDefiningOp(), boundaryInputs,
                                  visited, fusedOps);
    }
    if (fusedOps.empty() || !isClosedKernelOpSet(fusedOps, boundaryInputs))
      return failure();

    static const char kOpName[] = "KPFusedSparseSelect";
    auto schema = CustomOpSchema::get(kOpName)
                      .TypeVar("T")
                      .MemRefArg("a", 1, "T")
                      .MemRefArg("b", 1, "T")
                      .MemRefArg("c", 1, "T")
                      .MemRefArg("gt", 0, "T")
                      .MemRefArg("eq1", 0, "T")
                      .MemRefArg("eq2", 0, "T")
                      .MemRefArg("eq3", 0, "T")
                      .Result("out_x", 2, "")
                      .Result("out_y", 2, "")
                      .Result("out_w", 2, "");
    SmallVector<Type> inputTypes;
    for (Value input : boundaryInputs) inputTypes.push_back(input.getType());
    SmallVector<Type> outputTypes;
    for (Value output : boundaryOutputs)
      outputTypes.push_back(output.getType());
    auto inferred = inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(inputTypes),
        TypeRange(outputTypes));
    if (!inferred) {
      llvm::consumeError(inferred.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = kOpName;
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*inferred);
    if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

    std::string anchorName = getTfName(match.anchor);
    std::string kernelName = uniquifySymbolName(
        module, getStableFusionKernelName(anchorName, "kp_sparse_select"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/kp_fused"));
    auto kernelFunc = createExecutionV2KernelFunc(
        module, rewriter, kernelName, fusedOps, boundaryInputs,
        boundaryOutputs);
    kernelFunc->setAttr("fusion.pattern",
                        rewriter.getStringAttr("kp_sparse_select"));
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("kp_sparse_select")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> args;
    for (Value input : boundaryInputs)
      args.push_back({"dynamic", getValueName(input), input.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), args)));
    SmallVector<FusionArgSpec> outputs;
    auto outputName = [](Value value, StringRef fallback) {
      std::string name = getFusionOutputName(value);
      return name.empty() ? fallback.str() : name;
    };
    outputs.push_back({"output",
                       outputName(match.xResult, anchorName + "/output_x"),
                       match.xResult.getType()});
    outputs.push_back({"output",
                       outputName(match.yResult, anchorName + "/output_y"),
                       match.yResult.getType()});
    outputs.push_back({"output",
                       outputName(match.wResult, anchorName + "/output_w"),
                       match.wResult.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                             rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));
    anchor->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
    return success();
  }
};


// ══ FusedTensordotMatMul（execution v2，ONNX 站点）═════════════════════════
//
// ONNX 的 Tensordot 导出为固定形态的块，ATIR 上共 16 个算子：恒等 perm 的
// Transpose、10 个算子的 shape 链（Shape/Cast/2×Gather/2×Prod/2×ExpandDims/
// Concat/Cast）算出 [prod(free), contract]、展平 Reshape、MatMul（rank 2 常量
// 权重）、反展平目标的 Concat+Cast、反展平 Reshape。（ORT 侧口径记作
// "14 节点"，其编号不含反展平的 Cast 与 Reshape。）
//
// row-major 下展平前导维不搬数据，除 MatMul 外的 15 个算子又都是纯元数据操作
// ——其中 13 个（Shape/Gather/Prod/ExpandDims/Cast/Transpose/Reshape）在
// annc/lib/Conversion 下根本没有 lowering，另 2 个是 atir.Concat——因此整块
// 等价于「按前导维展平后与末维做矩阵乘」，落成单个自定义算子最省。
//
// 匹配条件对齐 ORT 的 FusedTensordotMatMulFusion，但收紧一条：free 轴必须恰为
// 前导维。ORT 只校验 contract 轴是末维（fused_tensordot_matmul_fusion.cc:
// 328-330），free 轴取 shape 链里 Gather 常量原值；两侧 kernel 都按连续
// row-major 线性读取输入，free 不是前导维时会算错而 ORT 不检查。
//
// execution v2 形态（本线版本）：kernel 函数首参为执行上下文，输出由 kernel
// 经上下文回调分配（不再有输出 memref 形参），线程池由上下文携带；主图区域
// 只标记 annc.fusion_materialized、不重写，AOT 裁剪会丢弃主图。

struct TensordotMatMulMatch {
  Value input;         // 被展平张量 x（rank 3）
  Value weight;        // rank 2 常量权重 W
  Value output;        // 反展平 Reshape 的结果
  Value outputBuffer;  // 反展平 Reshape 的输出缓冲
};

// 形状链里的 Cast 可有可无：ORT 侧 Concat 出口既可能是 Cast 也可能是 Concat
// 本身（fused_tensordot_matmul_fusion.cc:303-311），ATIR 侧同形。
static Value skipFtdmmCast(Value value) {
  if (auto cast = value.getDefiningOp<CastOp>()) return cast.getInput();
  return value;
}

static bool isFtdmmConstInts(Value value, ArrayRef<int64_t> expected) {
  auto tensorType = dyn_cast<atir::TensorType>(value.getType());
  if (!tensorType || !value.getDefiningOp<ConstantOp>()) return false;
  auto data = dyn_cast_or_null<DenseElementsAttr>(tensorType.getCacheData());
  if (!data || data.getNumElements() != static_cast<int64_t>(expected.size()) ||
      !data.getElementType().isIntOrIndex()) {
    return false;
  }
  size_t index = 0;
  for (const APInt &element : data.getValues<APInt>()) {
    if (element.getSExtValue() != expected[index++]) return false;
  }
  return true;
}

// value 是否由 Shape(data) 经（可选）Cast 得到。
static bool isFtdmmShapeOf(Value value, Value data) {
  auto shape = skipFtdmmCast(value).getDefiningOp<ShapeOp>();
  return shape && shape.getInput() == data;
}

// value == Gather(Shape(data), axis=0, indices == axes)。
static bool isFtdmmAxesGatherOf(Value value, Value data,
                                ArrayRef<int64_t> axes) {
  auto gather = skipFtdmmCast(value).getDefiningOp<GatherOp>();
  if (!gather) return false;
  if (!isFtdmmConstInts(gather.getAxis(), {0})) return false;
  if (!isFtdmmShapeOf(gather.getParams(), data)) return false;
  return isFtdmmConstInts(gather.getIndices(), axes);
}

// value == ExpandDims(Prod(Gather(Shape(data), indices == axes), axis=0), 0)
// 或 Prod{keep_dims=true}(Gather(Shape(data), indices == axes), axis=0)。
//
// Unsqueeze 的轴恒为 0：被扩的是 ReduceProd 产生的 0 维标量，ONNX 对 0 维
// 输入的合法轴是 [-1, 0]——轴 1 连 onnxruntime 的 shape inference 都拒绝。
// keep_dims=true 的 Prod 与 ExpandDims(Prod) 语义等价（结果同为 [1] 秩 1
// 张量），对应 TF reduce_prod(keepdims=true) 的直译 lowering。
static bool isFtdmmAxesProdExpandDimsOf(Value value, Value data,
                                        ArrayRef<int64_t> axes) {
  auto prodOf = [](Value v) -> ProdOp {
    if (auto expand = v.getDefiningOp<ExpandDimsOp>()) {
      if (!isFtdmmConstInts(expand.getAxis(), {0})) return {};
      return expand.getInput().getDefiningOp<ProdOp>();
    }
    if (auto prod = v.getDefiningOp<ProdOp>()) {
      if (!prod.getKeepDims()) return {};
      return prod;
    }
    return {};
  };
  auto prod = prodOf(value);
  if (!prod) return false;
  if (!isFtdmmConstInts(prod.getIndices(), {0})) return false;
  return isFtdmmAxesGatherOf(prod.getInput(), data, axes);
}

static SmallVector<int64_t> ftdmmLeadingAxes(int64_t rank) {
  SmallVector<int64_t> axes;
  for (int64_t axis = 0; axis + 1 < rank; ++axis) axes.push_back(axis);
  return axes;
}

static SmallVector<int64_t> ftdmmIdentityPerm(int64_t rank) {
  SmallVector<int64_t> perm;
  for (int64_t axis = 0; axis < rank; ++axis) perm.push_back(axis);
  return perm;
}

// 展平目标形状 [prod(free), contract]。
static bool isFtdmmFlattenShapeTarget(Value target, Value data,
                                      int64_t rank) {
  auto concat = skipFtdmmCast(target).getDefiningOp<ConcatOp>();
  if (!concat || concat.getAxis() != 0) return false;
  auto inputs = concat.getInputs();
  if (inputs.size() != 2) return false;
  return isFtdmmAxesProdExpandDimsOf(inputs[0], data, ftdmmLeadingAxes(rank)) &&
         isFtdmmAxesProdExpandDimsOf(inputs[1], data, {rank - 1});
}

// 反展平目标形状 [free 维..., N]。
static bool isFtdmmUnflattenShapeTarget(Value target, Value data, int64_t rank,
                                        int64_t outputDim) {
  auto concat = skipFtdmmCast(target).getDefiningOp<ConcatOp>();
  if (!concat || concat.getAxis() != 0) return false;
  auto inputs = concat.getInputs();
  if (inputs.size() != 2) return false;
  return isFtdmmAxesGatherOf(inputs[0], data, ftdmmLeadingAxes(rank)) &&
         isFtdmmConstInts(inputs[1], {outputDim});
}

// 恒等 perm 的 Transpose；perm 操作数若为常量则一并核对。
static bool isFtdmmIdentityTranspose(TransposeOp transpose, int64_t rank) {
  auto permutation = transpose.getPermutation();
  if (!permutation || static_cast<int64_t>(permutation.size()) != rank) {
    return false;
  }
  for (int64_t axis = 0; axis < rank; ++axis) {
    auto element = dyn_cast<IntegerAttr>(permutation[axis]);
    if (!element || element.getInt() != axis) return false;
  }
  if (transpose.getPerm().getDefiningOp<ConstantOp>()) {
    return isFtdmmConstInts(transpose.getPerm(), ftdmmIdentityPerm(rank));
  }
  return true;
}

// No shape gate: every site the matcher accepts is fused, so the site count
// matches the patch's on the target models (model4: 15/15). Kernel backend
// selection is independent: ANNC_KERNEL_MLAS uses MLAS when the execution
// context supplies a non-null native pool; otherwise the scalar row path runs.
// The scalar fallback does not restore a matching-size limit.

// value 的消费者必须恰为唯一一个且类型为 OpT。
template <typename OpT>
static OpT ftdmmSingleConsumerOf(Value value) {
  OpT result;
  for (Operation *user : value.getUsers()) {
    if (auto typed = dyn_cast<OpT>(user)) {
      if (result) return OpT();  // 多个同类消费者，放弃
      result = typed;
      continue;
    }
    return OpT();  // 存在其它类型的消费者
  }
  return result;
}

static FailureOr<TensordotMatMulMatch> matchTensordotMatMul(MatMulOp matmul) {
  if (matmul->hasAttr("annc.fusion_materialized")) return failure();
  if (matmul.getWithBias() || matmul.getDoRelu()) return failure();
  // 比 dev 版多一条：显式拒绝带转置的 MatMul。dev 不查，而 kernel 按连续
  // row-major 读两侧操作数——转置旗标一旦出现就会静默算错。ONNX 前端把
  // Transpose 显式降成 atir.Transpose（不产生这些属性），TF 前端则会把
  // transpose_a/transpose_b 映射到 left/right_transpose（Builder/Transformers/
  // Matrix.cpp）；本 pattern 目前被 annc.contract_version 门限制在 ONNX 模块，
  // 这条拒绝作为契约守卫保留，不改变现有站点的匹配结果。
  if (matmul.getLeftTranspose() || matmul.getRightTranspose() ||
      matmul.getOutputTranspose()) {
    return failure();
  }

  TensordotMatMulMatch match;

  // (1) 权重：rank 2 的 f32 常量，两维静态且为正。
  auto weightType = dyn_cast<atir::TensorType>(matmul.getRhs().getType());
  if (!weightType || weightType.getShape().size() != 2) return failure();
  if (!weightType.getElementType().isF32()) return failure();
  if (!matmul.getRhs().getDefiningOp<ConstantOp>()) return failure();
  if (!weightType.getCacheData()) return failure();
  const int64_t contractDim = weightType.getShape()[0];
  const int64_t outputDim = weightType.getShape()[1];
  if (contractDim <= 0 || outputDim <= 0) return failure();

  // (2) lhs 由 Reshape 产出，且该 Reshape 的结果只喂给本 MatMul。
  auto flattenReshape = matmul.getLhs().getDefiningOp<ReshapeOp>();
  if (!flattenReshape || flattenReshape.getResult() != matmul.getLhs()) {
    return failure();
  }
  if (!ftdmmSingleConsumerOf<MatMulOp>(flattenReshape.getResult())) {
    return failure();
  }

  // (3) 展平的数据输入 rank 固定为 3；恒等 perm 的 Transpose 可选。TF 动态
  // batch 图里 tensordot 展平输入直接来自上游算子（Concat/BiasAdd 等），
  // 转换器只在部分形态下保留恒等 Transpose。无 Transpose 时输入布局与
  // kernel 假设的 row-major 3D 一致（末维等于 contractDim 校验兜底）；
  // 有 Transpose 时仍须恒等 perm，非恒等转置由上面 (L3406) 的旗标拒绝
  // 与本处 isFtdmmIdentityTranspose 双重挡住。
  Value input = flattenReshape.getInput();
  constexpr int64_t kRank = 3;
  if (auto transpose = input.getDefiningOp<TransposeOp>()) {
    if (transpose.getResult() != input) return failure();
    if (!isFtdmmIdentityTranspose(transpose, kRank)) return failure();
    input = transpose.getInput();
  }
  auto inputType = dyn_cast<atir::TensorType>(input.getType());
  if (!inputType || inputType.getShape().size() != 3) return failure();
  if (!inputType.getElementType().isF32()) return failure();
  const int64_t inputContractDim = inputType.getShape()[kRank - 1];
  if (inputContractDim >= 0 && inputContractDim != contractDim) return failure();

  // (4) 展平目标形状必须是 [prod(前导维), 末维]。
  if (!isFtdmmFlattenShapeTarget(flattenReshape.getTargetShape(), input,
                                 kRank)) {
    return failure();
  }

  // (5) MatMul 的唯一消费者是反展平 Reshape，其目标形状必须是 [前导维..., N]。
  auto unfoldReshape = ftdmmSingleConsumerOf<ReshapeOp>(matmul.getResult());
  if (!unfoldReshape) return failure();
  if (!isFtdmmUnflattenShapeTarget(unfoldReshape.getTargetShape(), input, kRank,
                                   outputDim)) {
    return failure();
  }
  if (unfoldReshape.getNumOperands() == 0) return failure();
  Operation *outputBuffer = unfoldReshape.getOperand(0).getDefiningOp();
  if (!outputBuffer || !isa<BufferOp>(outputBuffer)) return failure();

  match.input = input;
  match.weight = matmul.getRhs();
  match.output = unfoldReshape.getResult();
  match.outputBuffer = unfoldReshape.getOperand(0);
  return match;
}

// Execution v2 的 kernel 函数：(execution, input, weight) -> output，体内只有
// 一次 atir.CustomizeOp；CustomizeCallLowering 把首参（!llvm.ptr）透传给手写
// kernel `TensordotMatMul`（annc/lib/Kernel/builtin_kernels/tensordot_matmul_*），
// 输出由 kernel 经上下文分配后作为 CustomizeOp 的结果返回。
static atir::CustomOpSchema getTensordotMatMulSchema() {
  return atir::CustomOpSchema::get("TensordotMatMul")
      .TypeVar("T")
      .MemRefArg("input", 3, "T")
      .MemRefArg("weight", 2, "T")
      .Result("output", 3, "T");
}

static func::FuncOp createTensordotMatMulKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    const TensordotMatMulMatch &match) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {
      LLVM::LLVMPointerType::get(rewriter.getContext()), match.input.getType(),
      match.weight.getType()};
  auto funcType = rewriter.getFunctionType(inputTypes, match.output.getType());
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
  func->setAttr("annc.kernel", rewriter.getUnitAttr());
  func->setAttr("fusion.pattern",
                rewriter.getStringAttr("tensordot_matmul"));

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  auto customCall = rewriter.create<atir::CustomizeOp>(
      func.getLoc(), TypeRange{match.output.getType()},
      ValueRange{entry->getArgument(1), entry->getArgument(2)},
      rewriter.getStringAttr("TensordotMatMul"),
      getTensordotMatMulSchema().toMetadata(rewriter.getContext()));
  rewriter.create<func::ReturnOp>(func.getLoc(), customCall.getResult(0));
  return func;
}

struct FuseTensordotMatMulAsFuncCallPattern
    : public OpRewritePattern<MatMulOp> {
  // benefit 2 > FuseMatMulAsFuncCallPattern 的默认 1：同一 MatMul 上两者都
  // 可能匹配（后者是 jit 的通用 MatMul fusion），本 pattern 的形态更具体、
  // 且是 AOT 站点，必须先试。
  explicit FuseTensordotMatMulAsFuncCallPattern(MLIRContext *context)
      : OpRewritePattern<MatMulOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto match = matchTensordotMatMul(matmulOp);
    if (failed(match)) return failure();
    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    if (!module) return failure();

    // 站点本轮只服务 ONNX 导入：ONNX 前端在模块上留 annc.contract_version
    // （全仓唯一写入点），据此把本 pattern 限定在 ONNX 模块，TF 侧的融合行为
    // 保持不变。TF 宿主其实也能执行本 kernel——它构造的上下文池为空，
    // resolveThreadPool 回落 TLS——限定的目的是不改动 TF 侧的融合配置，
    // 而不是兼容性问题；将来要让 TF 也用上这个站点，去掉这道门即可。
    if (!module->hasAttr("annc.contract_version")) return failure();

    // kernel 必须先确认可用，否则会留下无法 lower 的自定义算子。
    auto schema = getTensordotMatMulSchema();
    SmallVector<Type> schemaInputTypes = {match->input.getType(),
                                          match->weight.getType()};
    auto inferred = atir::inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(schemaInputTypes),
        TypeRange{match->output.getType()});
    if (!inferred) {
      llvm::consumeError(inferred.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = "TensordotMatMul";
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*inferred);
    if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

    // 被融合的算子：反展平 Reshape 往上游回溯，边界是 x 与 W。
    SmallVector<Value, 2> boundaryInputs = {match->input, match->weight};
    SmallVector<Operation *> fusedOps;
    SmallPtrSet<Operation *, 32> visited;
    collectDefiningOpsPostOrder(match->output.getDefiningOp(), boundaryInputs,
                                visited, fusedOps);
    if (fusedOps.empty()) return failure();

    // 站点只有一路输出：除反展平结果外不得有结果逃逸，否则成员名单与融合
    // 节点的输出对不上（EP 在 Compile 阶段还会按输出数核对一次）。站点间
    // 共享的常量/缓冲不算逃逸——它们既不是成员（Constant/Input 不入名单），
    // 也不参与内核调用。
    llvm::SmallPtrSet<Operation *, 32> fusedSet(fusedOps.begin(),
                                                fusedOps.end());
    SmallVector<Value> escaping;
    for (Operation *op : fusedOps) {
      if (isa<ConstantOp, BufferOp>(op)) continue;
      for (Value result : op->getResults()) {
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !fusedSet.contains(user);
            })) {
          escaping.push_back(result);
        }
      }
    }
    if (escaping.size() != 1 || escaping[0] != match->output) return failure();

    std::string anchorName = getTfName(match->output.getDefiningOp());
    std::string kernelName = uniquifySymbolName(
        module, getStableFusionKernelName(anchorName, "tensordot_matmul"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/tensordot_matmul"));

    auto kernelFunc =
        createTensordotMatMulKernelFunc(module, rewriter, kernelName, *match);
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("tensordot_matmul")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    // args 按 kernel 调用顺序：v2 的形参是 (input, weight)；输出由 kernel 经
    // 执行上下文分配，不在调用参数里，故 kernel_arg_order 为空。
    SmallVector<FusionArgSpec> args;
    args.push_back(
        {"dynamic", getValueName(match->input), match->input.getType()});
    args.push_back(
        {"fixed", getValueName(match->weight), match->weight.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), args)));
    SmallVector<FusionArgSpec> outputs;
    outputs.push_back({"output", getFusionOutputName(match->output),
                       match->output.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                             rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    metadata.push_back(rewriter.getNamedAttr(
        "member_node_names",
        rewriter.getArrayAttr(collectMemberNodeNames(rewriter, fusedOps))));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));

    // execution v2 的主图区域保留原样（AOT 裁剪丢弃主图函数），只做标记，
    // 避免通用 MatMul fusion 在同一区域上再建站点。
    for (Operation *op : fusedOps) {
      op->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
    }
    return success();
  }
};

// ══ FusedMaskedMul（execution v2，ONNX 站点）═════════════════════════════════
//
// ONNX 的掩码乘链，每站点 6~7 个算子：
//
//   ids ─ Equal(pad) → Not → Cast(f32) → Unsqueeze(末轴) → [Tile] ─┐
//   data_i ─────────────────────────────────────────────────────── Mul_i → y_i
//
// ids 为 i64 rank2、data 为 f32 rank3，Tile 只沿末维重复（纯广播、不搬信息）。
// 链的语义就是逐行取因子再缩放 data 的末维：
//
//   y[b, t, n] = data[b, t, n] * (ids[b, t] != pad ? 1.0f : 0.0f)
//
// 站点是「这条链 + 链喂的全部 Mul」这一整块子图，融合节点的输出就是子图对外
// 提供的全部值：每个 Mul 的结果，加上链上仍被图外读的值（model3 的
// `Shape/Expand` 读的 Cast 输出）。Mul 之间若夹着逐元素路径（model1 的
// `Sum`（前端降成多输入 atir.Add）→ `RealDiv` 除常量），这些节点也一并进站点、
// 由 kernel 算掉：成员之间不允许夹组外节点，否则 ORT 会作废整次提交
// （onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc:396）。
//
// kernel 逐元素计算（p = 直接吃外部 data 的 Mul 数，q = 吃站点内 d 的 Mul 数）：
//
//   f      = (ids != pad) ? 1.0f : 0.0f
//   y_i    = x_i * f              i < p；x_i 来自 data 实参（见 argMap 属性）
//   S      = Σ y_i + Σ e_j        Sum 的输入：上面的结果 + 站点的其它实参
//   d      = S / c                c = Div 的常量初值（divisorBits 属性）
//   y_fed  = d * f
//   [可选] flat = f               链外消费者读的 Cast 输出
//
// 匹配条件对齐 ORT 的 FusedMaskedMul（annc_opt 0005，
// masked-mul-patch/0005-add-fused-masked-mul.patch::MatchMaskedMul）。
// 与 dev 线（任务 20260924-016）的差别只在交付形态：kernel 函数首参为执行
// 上下文、输出由 kernel 经上下文回调分配（不再有输出 memref 形参），主图区域
// 只标记 annc.fusion_materialized、不重写，AOT 裁剪会丢弃主图。

// 站点里的一个 Mul。
struct MaskedMulSite {
  MulOp mul;
  Value output;        // Mul 的结果
  Value outputBuffer;  // Mul 的输出缓冲（BufferOp 结果）
  Value data;          // 非 fed 的 Mul 的 data 源（可能是另一个 Mul 的源）
  bool fed = false;    // data 是站点内算出的 d（Sum/Div 的产出）
};

struct MaskedMulMatch {
  Value ids;                     // Equal 的 rank-2 i64 输入
  int64_t padding = 0;           // Equal 折叠出的常量
  int64_t tileLastMultiple = 0;  // 无 Tile 时为 0
  // 成员：链（由外向内：Tile?, ExpandDims, Cast, Compare(NE), Compare(EQ)）
  // + 吸收的逐元素节点（Sum/Div）。
  SmallVector<Operation *, 6> chain;
  SmallVector<Operation *, 4> absorbed;
  SmallVector<MaskedMulSite, 4> sites;  // raw 在前、fed 在后
  // data 实参（每个不同的源一个）+ Sum 的其它输入；前 rawCount 个是 ×f 的 data。
  SmallVector<Value, 4> args;
  int64_t rawCount = 0;
  int64_t fedCount = 0;
  // 每个 Mul（raw 部分）用第几个 data 实参，4 bit 一个（同源的两个 Mul 共用，
  // 融合节点的实参不能重复指向同一个输入，EP 会拒绝）。
  int64_t argMap = 0;
  // 哪些 raw Mul 的结果进了中间那个逐元素节点（按 raw 站点顺序一个 bit）:
  // model1 有 3 个 raw Mul 而 Sum 只吃其中 2 个的链。
  int64_t sumMask = 0;
  // 中间节点的算子：0 = Sum（逐元素相加），1 = 乘积。
  int64_t interiorKind = 0;
  float divisor = 1.0f;  // Div 的常量（无 Div 时为 1）
  Value flatMask;        // 逃出的 Cast 输出，无则空
};

// 读取整数常量张量的元素（splat 展开）。非 atir.constant 或非整数 cacheData
// 一律失败。
static bool readConstInts(Value value, SmallVectorImpl<int64_t> *out) {
  auto tensorType = dyn_cast<atir::TensorType>(value.getType());
  if (!tensorType || !value.getDefiningOp<ConstantOp>()) return false;
  auto data = dyn_cast_or_null<DenseElementsAttr>(tensorType.getCacheData());
  if (!data || !data.getElementType().isIntOrIndex()) return false;
  out->clear();
  if (data.isSplat()) {
    out->assign(static_cast<size_t>(data.getNumElements()),
                data.getSplatValue<APInt>().getSExtValue());
    return true;
  }
  for (const APInt &element : data.getValues<APInt>()) {
    out->push_back(element.getSExtValue());
  }
  return true;
}

// 单值常量（0 维、单元素或 splat），所有元素必须一致。
static bool readConstScalar(Value value, int64_t *out) {
  SmallVector<int64_t, 4> values;
  if (!readConstInts(value, &values) || values.empty()) return false;
  for (int64_t candidate : values) {
    if (candidate != values[0]) return false;
  }
  *out = values[0];
  return true;
}

// f32 的位模式（Div 的除数用 i64 属性传给 kernel；ABI 只有 i64 属性）。
static int64_t f32Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<int64_t>(bits);
}

// 单元素 f32 常量（0 维或 rank-1 长 1），用于 Div 的除数。
static bool readConstF32Scalar(Value value, float *out) {
  auto tensorType = dyn_cast<atir::TensorType>(value.getType());
  if (!tensorType || !tensorType.getElementType().isF32()) return false;
  if (!value.getDefiningOp<ConstantOp>()) return false;
  auto data = dyn_cast_or_null<DenseElementsAttr>(tensorType.getCacheData());
  if (!data || data.getNumElements() != 1) return false;
  *out = data.getSplatValue<APFloat>().convertToFloat();
  return true;
}

// ONNX 前端把 i1（bool）张量标成 encoding == "bool"。
static bool isBoolEncoded(Value value) {
  auto tensorType = dyn_cast<atir::TensorType>(value.getType());
  if (!tensorType) return false;
  auto encoding = dyn_cast_or_null<StringAttr>(tensorType.getEncoding());
  return encoding && encoding.getValue() == "bool";
}

// onnxTransformNot 为 Not(Equal(...)) 造的 i1 "true" 常量。
static bool isMaskTrueConstant(Value value) {
  if (!isBoolEncoded(value)) return false;
  int64_t flag = 0;
  return readConstScalar(value, &flag) && flag == 1;
}

// 把 mask 操作数回溯到 ids/padding。失败时 match 可能已被部分写入，调用方
// 负责重置（与 dev 相同）。
static LogicalResult matchMaskChain(Value mask, MaskedMulMatch *match) {
  SmallVector<Operation *, 6> chain;
  Value current = mask;

  if (auto tile = current.getDefiningOp<TileOp>()) {
    if (tile.getResult() != current) return failure();
    SmallVector<int64_t, 4> multiples;
    if (!readConstInts(tile.getMultiples(), &multiples) ||
        multiples.size() != 3) {
      return failure();
    }
    // 纯广播：只有末轴可重复，且必须重复。
    if (multiples[0] != 1 || multiples[1] != 1 || multiples[2] < 1) {
      return failure();
    }
    match->tileLastMultiple = multiples[2];
    chain.push_back(tile.getOperation());
    current = tile.getInput();
  }

  auto expand = current.getDefiningOp<ExpandDimsOp>();
  if (!expand || expand.getResult() != current) return failure();
  auto expandType = dyn_cast<atir::TensorType>(expand.getType());
  if (!expandType || expandType.getShape().size() != 3) return failure();
  const int64_t rank = static_cast<int64_t>(expandType.getShape().size());
  int64_t axis = 0;
  if (!readConstScalar(expand.getAxis(), &axis)) return failure();
  if (axis < 0) axis += rank;
  if (axis != rank - 1) return failure();
  chain.push_back(expand.getOperation());

  auto cast = expand.getInput().getDefiningOp<CastOp>();
  if (!cast || cast.getResult() != expand.getInput()) return failure();
  auto castType = dyn_cast<atir::TensorType>(cast.getType());
  if (!castType || !castType.getElementType().isF32()) return failure();
  if (!isBoolEncoded(cast.getInput())) return failure();
  chain.push_back(cast.getOperation());

  auto notCompare = cast.getInput().getDefiningOp<CompareOp>();
  if (!notCompare || notCompare.getResult() != cast.getInput()) return failure();
  if (notCompare.getComparisonDirection() != "NE") return failure();
  Value notInput;
  if (isMaskTrueConstant(notCompare.getRhs())) {
    notInput = notCompare.getLhs();
  } else if (isMaskTrueConstant(notCompare.getLhs())) {
    notInput = notCompare.getRhs();
  } else {
    return failure();
  }
  chain.push_back(notCompare.getOperation());

  // Equal(ids, pad)：真实掩码链的 ids 是 i64，而 ONNX 前端按 ADR-011 的
  // dtype 门禁拒绝整型比较，把该节点降级成 atir.opaque。两种形态都接受：
  // atir.opaque 的文档写明它"expected to be absorbed by a fusion pattern"，
  // 门禁要防的两件事（比较的类型语义、常量约束）由本匹配器重新校验——ids
  // 必须是 rank-2 i64、padding 必须是单个整型常量；命中后整条链由
  // masked_mul kernel 以 i64 精确执行，opaque 节点不再存活。
  int64_t padding = 0;
  Value ids;
  Operation *equalOp = nullptr;
  if (auto equalCompare = notInput.getDefiningOp<CompareOp>()) {
    if (equalCompare.getResult() != notInput) return failure();
    if (equalCompare.getComparisonDirection() != "EQ") return failure();
    if (readConstScalar(equalCompare.getRhs(), &padding)) {
      ids = equalCompare.getLhs();
    } else if (readConstScalar(equalCompare.getLhs(), &padding)) {
      ids = equalCompare.getRhs();
    } else {
      return failure();
    }
    equalOp = equalCompare.getOperation();
  } else if (auto equalOpaque = notInput.getDefiningOp<OpaqueOp>()) {
    if (equalOpaque.getOpType() != "Equal") return failure();
    if (equalOpaque->getNumOperands() != 2 || equalOpaque.getNumResults() != 1) {
      return failure();
    }
    if (equalOpaque.getOutput().front() != notInput) return failure();
    Value lhs = equalOpaque.getInputs()[0];
    Value rhs = equalOpaque.getInputs()[1];
    if (readConstScalar(rhs, &padding)) {
      ids = lhs;
    } else if (readConstScalar(lhs, &padding)) {
      ids = rhs;
    } else {
      return failure();
    }
    equalOp = equalOpaque.getOperation();
  } else {
    return failure();
  }
  auto idsType = dyn_cast<atir::TensorType>(ids.getType());
  if (!idsType || idsType.getShape().size() != 2) return failure();
  if (!idsType.getElementType().isInteger(64)) return failure();

  chain.push_back(equalOp);
  match->ids = ids;
  match->padding = padding;
  match->chain = std::move(chain);
  return success();
}

// 两个成员之间是否存在"绕出去"的路径。这种组合 ORT 会拒绝，而且会连带丢弃
// 同一 EP 提交的**全部**分组，所以站点根本不能申报。
//
// 2026-10-13 复验（本线，新版插件 EP API）：这道守卫仍然必需。
// `onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc:396`
// 在 `CreateSupportedPartitions` 给出的分组数 != 1 时打印
// "set nodes that cannot be fused together" 并 `return {}`——整次 GetCapability
// 提交的分组全部作废。判断 EP 是否真的执行了 kernel 不能只看 claimed，要看
// Compute 侧证据（`ANNC_EP_DEBUG=1` 的逐实参绑定日志）或 ORT 的 cannot-be-fused
// 报错。
static bool hasHoleBetweenMembers(ArrayRef<Operation *> members) {
  SmallPtrSet<Operation *, 16> memberSet(members.begin(), members.end());
  for (Operation *member : members) {
    for (Value result : member->getResults()) {
      SmallVector<Operation *, 16> worklist;
      SmallPtrSet<Operation *, 16> visited;
      for (Operation *user : result.getUsers()) {
        if (!memberSet.contains(user)) worklist.push_back(user);
      }
      while (!worklist.empty()) {
        Operation *current = worklist.pop_back_val();
        if (!visited.insert(current).second) continue;
        for (Value res : current->getResults()) {
          for (Operation *user : res.getUsers()) {
            if (memberSet.contains(user)) return true;
            worklist.push_back(user);
          }
        }
      }
    }
  }
  return false;
}

// 站点里的一个 Mul：mask 必须是链的最外层值，data 为 rank-3 f32、前缀与 ids
// 对齐、写进缓冲。
static bool maskedMulOperands(MulOp mul, Value mask, Value ids, Value *data,
                              Value *output, Value *outputBuffer) {
  if (mul->hasAttr("annc.fusion_materialized")) return false;
  Value other;
  if (mul.getY() == mask) {
    other = mul.getX();
  } else if (mul.getX() == mask) {
    other = mul.getY();
  } else {
    return false;
  }
  auto dataType = dyn_cast<atir::TensorType>(other.getType());
  if (!dataType || dataType.getShape().size() != 3) return false;
  if (!dataType.getElementType().isF32()) return false;
  auto idsType = dyn_cast<atir::TensorType>(ids.getType());
  if (!idsType) return false;
  // 前缀形状必须对齐（kernel 按行索引 ids）；动态维留给运行时校验。
  for (int64_t axis = 0; axis < 2; ++axis) {
    const int64_t dataDim = dataType.getShape()[axis];
    const int64_t idsDim = idsType.getShape()[axis];
    if (dataDim >= 0 && idsDim >= 0 && dataDim != idsDim) return false;
  }
  if (dataType.getShape()[2] == 0) return false;

  Value out = mul.getOutput();
  Operation *bufferOp = out ? out.getDefiningOp() : nullptr;
  if (!bufferOp || !isa<BufferOp>(bufferOp)) return false;
  if (mul.getResult().getType() != out.getType()) return false;

  *data = other;
  *output = mul.getResult();
  *outputBuffer = out;
  return true;
}

// 以 anchor 为起点匹配整块子图：链 + 链喂的全部 Mul（+ 它们之间的逐元素路径）。
// 链上每个值的链外消费者只能是「最外层值」（就是那些 Mul）或「Cast 输出」
// （f32 mask 被 Shape/Expand 之类读）；其它逃逸都表达不了。
static FailureOr<MaskedMulMatch> matchMaskedMul(MulOp anchor) {
  MaskedMulMatch match;
  Value mask;
  if (succeeded(matchMaskChain(anchor.getY(), &match))) {
    mask = anchor.getY();
  } else {
    match = MaskedMulMatch();
    if (succeeded(matchMaskChain(anchor.getX(), &match))) {
      mask = anchor.getX();
    } else {
      return failure();
    }
  }

  // 链已经被别的 anchor 认下（一条链一个站点）：本 anchor 不再成站。
  for (Operation *op : match.chain) {
    if (op->hasAttr("annc.fusion_materialized")) return failure();
  }

  SmallVector<MulOp, 4> muls;
  SmallVector<Value, 4> mulOutputs;
  SmallVector<Value, 4> mulData;
  bool anchorListed = false;
  bool flatEscapes = false;
  // 已钉住的 data 末维（-1 = 未钉住）：kernel 按同一个 n 读写全部实参与
  // 输出，同一条链喂的 Mul 末维必须一致。
  int64_t dataLastDim = -1;
  for (Operation *op : match.chain) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (llvm::is_contained(match.chain, user)) continue;
        auto mul = dyn_cast<MulOp>(user);
        if (mul && result == mask) {
          if (llvm::is_contained(muls, mul)) continue;
          Value data, output, buffer;
          if (!maskedMulOperands(mul, mask, match.ids, &data, &output,
                                 &buffer)) {
            return failure();
          }
          // Tile 必须恰好广播到 data 的末维（已知时逐个消费者核对）。
          auto dataType = cast<atir::TensorType>(data.getType());
          const int64_t lastDim = dataType.getShape()[2];
          if (match.tileLastMultiple != 0 && lastDim >= 0 &&
              lastDim != match.tileLastMultiple) {
            return failure();
          }
          // 末维一致性：无 Tile 时 mask [d0,d1,1] 能广播到任意 n，末维不同
          // 的 Mul 会匹配成功、却要到 kernel 运行时才被实参同形检查拒掉
          //（session 失败而不是回退），故在匹配期拒绝；动态维留给运行时核对。
          if (lastDim >= 0) {
            if (dataLastDim >= 0 && dataLastDim != lastDim) return failure();
            dataLastDim = lastDim;
          }
          muls.push_back(mul);
          mulOutputs.push_back(output);
          mulData.push_back(data);
          anchorListed |= (mul == anchor);
          continue;
        }
        // Cast 输出（未 Unsqueeze 的 f32 mask）可以逃出去给形状类消费者，
        // 它就是站点的一个输出；别的任何逃逸都表达不了。
        if (isa<CastOp>(op) && result == op->getResult(0) && !mul) {
          flatEscapes = true;
          continue;
        }
        return failure();
      }
    }
  }
  if (!anchorListed || muls.empty()) return failure();
  if (flatEscapes) {
    for (Operation *op : match.chain) {
      if (isa<CastOp>(op)) match.flatMask = op->getResult(0);
    }
    if (!match.flatMask) return failure();
  }

  // data 来源分类：外部实参 / 站点内 Mul 的源（幂等，见下）/ 站点内算出的 d。
  //   mask ⊙ (mask ⊙ x) == mask ⊙ x：mask 的取值恰是 0/1，所以下游那个 Mul 与
  //   上游同源，两者共用同一个实参（EP 不接受两个实参指向同一个输入）。
  SmallVector<Value, 4> extras;
  SmallVector<Value, 4> sumMembers;
  SmallVector<MaskedMulSite, 4> rawSites;
  SmallVector<MaskedMulSite, 4> fedSites;
  SmallPtrSet<Operation *, 2> absorbed;
  for (size_t i = 0; i < muls.size(); ++i) {
    MaskedMulSite site;
    site.mul = muls[i];
    site.output = mulOutputs[i];
    site.outputBuffer = muls[i].getOutput();

    Value source = mulData[i];
    for (int guard = 0; guard < 8; ++guard) {
      auto producer = source.getDefiningOp<MulOp>();
      if (!producer || !llvm::is_contained(muls, producer)) break;
      source = producer.getX() == mask ? producer.getY() : producer.getX();
    }

    // Sum（多输入 Add）/ Div(Sum, 常量)：吸收进站点，本 Mul 吃算出来的 d。
    Operation *producer = source.getDefiningOp();
    RealDivOp div;
    AddOp sum;
    MulOp interiorMul;
    if (producer) {
      if (auto maybeDiv = dyn_cast<RealDivOp>(producer)) {
        div = maybeDiv;
        sum = div.getX().getDefiningOp<AddOp>();
        float constant = 0.0f;
        if (!sum || !readConstF32Scalar(div.getY(), &constant)) {
          return failure();
        }
        match.divisor = constant;
      } else if (auto maybeAdd = dyn_cast<AddOp>(producer)) {
        sum = maybeAdd;
      } else if (auto maybeMul = dyn_cast<MulOp>(producer);
                 maybeMul && producer != anchor.getOperation() &&
                 !llvm::is_contained(muls, maybeMul) &&
                 maybeMul.getY() != mask && maybeMul.getX() != mask) {
        // 中间是一个普通逐元素乘（model1 的 NotEqual 链：两个 masked 结果相乘，
        // 积再乘 mask）。两侧可以是本站点的 Mul 结果或外部张量。
        interiorMul = maybeMul;
        match.interiorKind = 1;
      }
    }
    if (!sum && !interiorMul) {
      site.data = source;
      rawSites.push_back(site);
      continue;
    }

    // 中间节点的产出只能喂这个 Div（若有）或本站点的 Mul；Div 的产出只能喂本站
    // 点的 Mul——否则那些值逃出分组，而 kernel 不产出它们。
    Operation *interior = sum ? sum.getOperation()
                              : interiorMul.getOperation();
    for (Operation *user : interior->getResult(0).getUsers()) {
      if (div && user == div.getOperation()) continue;
      if (auto userMul = dyn_cast<MulOp>(user);
          userMul && llvm::is_contained(muls, userMul)) {
        continue;
      }
      return failure();
    }
    if (div) {
      for (Operation *user : div.getResult().getUsers()) {
        auto userMul = dyn_cast<MulOp>(user);
        if (!userMul || !llvm::is_contained(muls, userMul)) return failure();
      }
    }
    // 一个站点只承载一个中间节点：kernel 的 ABI 只有一份 sumMask /
    // interiorKind / divisorBits，且把同一个 d 写进全部 fed 输出。第二个
    // 不同的中间节点（各自的输入集合或除数不同）会被算成同一个值而无任何
    // 报错——宁可不融。
    if (!absorbed.contains(interior) && !absorbed.empty()) return failure();
    // 中间节点的输入：站点内 Mul 的结果（kernel 里已经算了，直接进和式/乘积）
    // 或外部张量（额外实参）。
    if (absorbed.insert(interior).second) {
      match.absorbed.push_back(interior);
      if (div) match.absorbed.push_back(div.getOperation());
      SmallVector<Value, 4> interiorInputs;
      if (sum) {
        interiorInputs.append(sum.getInputs().begin(), sum.getInputs().end());
      } else {
        interiorInputs.push_back(interiorMul.getX());
        interiorInputs.push_back(interiorMul.getY());
      }
      for (Value value : interiorInputs) {
        if (llvm::is_contained(mulOutputs, value)) {
          if (!llvm::is_contained(sumMembers, value)) sumMembers.push_back(value);
          continue;
        }
        auto type = dyn_cast<atir::TensorType>(value.getType());
        if (!type || type.getShape().size() != 3 ||
            !type.getElementType().isF32()) {
          return failure();
        }
        // 与 data 实参同要求：kernel 把所有实参按同形 [d0,d1,n] 读，前缀
        // 对齐 ids、末维对齐已钉住的 n，形状不符在匹配期回退（动态维留给
        // kernel 的运行时同形检查）。
        auto idsType = cast<atir::TensorType>(match.ids.getType());
        for (int64_t axis = 0; axis < 2; ++axis) {
          const int64_t dim = type.getShape()[axis];
          const int64_t idsDim = idsType.getShape()[axis];
          if (dim >= 0 && idsDim >= 0 && dim != idsDim) return failure();
        }
        if (type.getShape()[2] >= 0 && dataLastDim >= 0 &&
            type.getShape()[2] != dataLastDim) {
          return failure();
        }
        if (!llvm::is_contained(extras, value)) extras.push_back(value);
      }
    }
    site.fed = true;
    fedSites.push_back(site);
  }
  if (rawSites.empty()) return failure();

  // 实参表：每个不同的 data 源一个（前 rawCount 个），随后是 Sum 的其它输入。
  for (const MaskedMulSite &site : rawSites) {
    if (!llvm::is_contained(match.args, site.data)) {
      match.args.push_back(site.data);
    }
  }
  match.rawCount = static_cast<int64_t>(match.args.size());
  for (Value extra : extras) match.args.push_back(extra);
  // 同源的两个 Mul 共用同一个实参：4 bit 一个下标。
  for (const MaskedMulSite &site : rawSites) {
    const int64_t index =
        static_cast<int64_t>(llvm::find(match.args, site.data) -
                             match.args.begin());
    match.argMap |= index << (4 * match.sites.size());
    match.sites.push_back(site);
  }
  for (Value value : sumMembers) {
    const auto it = llvm::find_if(rawSites, [&](const MaskedMulSite &site) {
      return site.output == value;
    });
    if (it == rawSites.end()) return failure();
    match.sumMask |= int64_t{1} << (it - rawSites.begin());
  }
  for (const MaskedMulSite &site : fedSites) match.sites.push_back(site);
  match.fedCount = static_cast<int64_t>(fedSites.size());
  return match;
}

// kernel 算子名：实参个数（data + Sum 的其它输入）+ 是否多写一个 flat 输出。
static std::string getMaskedMulOpName(const MaskedMulMatch &match) {
  std::string name = "MaskedMul" + std::to_string(match.args.size());
  if (match.flatMask) name += "Flat";
  return name;
}

// 实参顺序 (data..., extras..., ids)；padding / argMap / divisorBits 是属性，
// 输出在 kernel 侧经上下文按槽位分配，故不在调用参数里。
static atir::CustomOpSchema getMaskedMulSchema(const MaskedMulMatch &match) {
  auto schema =
      atir::CustomOpSchema::get(getMaskedMulOpName(match))
          .TypeVar("T")
          .TypeVar("TMask");
  for (size_t i = 0; i < match.args.size(); ++i) {
    schema.MemRefArg("arg" + std::to_string(i), 3, "T");
  }
  schema.MemRefArg("ids", 2, "TMask");
  for (size_t i = 0; i < match.sites.size(); ++i) {
    schema.Result("output" + std::to_string(i), 3, "T");
  }
  if (match.flatMask) schema.Result("flat", 2, "T");
  schema.I64AttrArg("dataArgs", match.rawCount);
  schema.I64AttrArg("rawMuls", static_cast<int64_t>(match.sites.size()) - match.fedCount);
  schema.I64AttrArg("fedMuls", match.fedCount);
  schema.I64AttrArg("argMap", match.argMap);
  schema.I64AttrArg("sumMask", match.sumMask);
  schema.I64AttrArg("interiorKind", match.interiorKind);
  schema.I64AttrArg("divisorBits", f32Bits(match.divisor));
  schema.I64AttrArg("padding", match.padding);
  return schema;
}

// Execution v2 的 kernel 函数：(execution, data..., extras..., ids) ->
// (output...[, flat])，体内只有一次 atir.CustomizeOp；CustomizeCallLowering 把
// 首参（!llvm.ptr）透传给手写 kernel（annc/lib/Kernel/builtin_kernels/
// masked_mul_*），输出由 kernel 经上下文按槽位分配后作为 CustomizeOp 的结果
// 返回（槽位顺序 = sites 顺序，最后是 flat）。
static func::FuncOp createMaskedMulKernelFunc(ModuleOp module,
                                              PatternRewriter &rewriter,
                                              StringRef kernelName,
                                              const MaskedMulMatch &match) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {
      LLVM::LLVMPointerType::get(rewriter.getContext())};
  for (Value arg : match.args) inputTypes.push_back(arg.getType());
  inputTypes.push_back(match.ids.getType());
  SmallVector<Type> resultTypes;
  for (const MaskedMulSite &site : match.sites) {
    resultTypes.push_back(site.output.getType());
  }
  if (match.flatMask) resultTypes.push_back(match.flatMask.getType());

  auto funcType = rewriter.getFunctionType(inputTypes, resultTypes);
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
  func->setAttr("annc.kernel", rewriter.getUnitAttr());
  func->setAttr("fusion.pattern", rewriter.getStringAttr("masked_mul"));

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  SmallVector<Value> operands;
  for (size_t i = 0; i < match.args.size(); ++i) {
    operands.push_back(entry->getArgument(i + 1));
  }
  operands.push_back(entry->getArgument(match.args.size() + 1));
  auto customCall = rewriter.create<atir::CustomizeOp>(
      func.getLoc(), TypeRange(resultTypes), ValueRange(operands),
      rewriter.getStringAttr(getMaskedMulOpName(match)),
      getMaskedMulSchema(match).toMetadata(rewriter.getContext()));
  rewriter.create<func::ReturnOp>(func.getLoc(), customCall.getResults());
  return func;
}

struct FuseMaskedMulAsFuncCallPattern : public OpRewritePattern<MulOp> {
  // 与 FTDMM 相同的 benefit 次序：同一个 Mul 上本 pattern 与通用逐元素融合
  // 都可能匹配，本形态更具体，必须先试。
  explicit FuseMaskedMulAsFuncCallPattern(MLIRContext *context)
      : OpRewritePattern<MulOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulOp mulOp,
                                PatternRewriter &rewriter) const override {
    auto match = matchMaskedMul(mulOp);
    if (failed(match)) return failure();
    ModuleOp module = mulOp->getParentOfType<ModuleOp>();
    if (!module) return failure();
    // 站点本轮只服务 ONNX 导入（同 FTDMM）：ONNX 前端在模块上留
    // annc.contract_version，据此把 pattern 限定在 ONNX 模块，TF 侧融合行为
    // 不变（kernel 本身在 TF 宿主下也能跑，池走 TLS 回退）。
    if (!module->hasAttr("annc.contract_version")) return failure();

    SmallVector<Operation *, 8> fusedOps;
    for (const MaskedMulSite &site : match->sites) {
      fusedOps.push_back(static_cast<Operation *>(site.mul));
    }
    fusedOps.append(match->absorbed.begin(), match->absorbed.end());
    fusedOps.append(match->chain.begin(), match->chain.end());

    if (hasHoleBetweenMembers(fusedOps)) return failure();
    // 站点实参必须来自区域外：kernel 以实参读取，区域内产出就没有值可传。
    for (Value arg : match->args) {
      Operation *producer = arg.getDefiningOp();
      if (producer && llvm::is_contained(fusedOps, producer)) return failure();
    }
    // 区域里除站点输出（每个 Mul 的结果 + flat）外不得有结果逃逸：融合节点的
    // 输出必须恰是这些值，否则成员名单与 EP 的申报对不上。
    SmallVector<Value> declared;
    for (const MaskedMulSite &site : match->sites) {
      declared.push_back(site.output);
    }
    if (match->flatMask) declared.push_back(match->flatMask);
    SmallVector<Value> escaping = atir::collectEscapingResults(fusedOps);
    if (escaping.size() != declared.size()) return failure();
    for (Value value : declared) {
      if (!llvm::is_contained(escaping, value)) return failure();
    }

    // kernel 必须先确认可用，否则会留下无法 lower 的自定义算子。
    auto schema = getMaskedMulSchema(*match);
    SmallVector<Type> schemaInputTypes;
    for (Value arg : match->args) schemaInputTypes.push_back(arg.getType());
    schemaInputTypes.push_back(match->ids.getType());
    SmallVector<Type> schemaResultTypes;
    for (const MaskedMulSite &site : match->sites) {
      schemaResultTypes.push_back(site.output.getType());
    }
    if (match->flatMask) {
      schemaResultTypes.push_back(match->flatMask.getType());
    }
    auto inferred = atir::inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(schemaInputTypes),
        TypeRange(schemaResultTypes));
    if (!inferred) {
      llvm::consumeError(inferred.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = getMaskedMulOpName(*match);
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*inferred);
    if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

    std::string anchorName =
        getTfName(static_cast<Operation *>(match->sites.front().mul));
    std::string kernelName = uniquifySymbolName(
        module, getStableFusionKernelName(anchorName, "masked_mul"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/masked_mul"));

    auto kernelFunc =
        createMaskedMulKernelFunc(module, rewriter, kernelName, *match);
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("masked_mul")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> args;
    for (Value arg : match->args) {
      args.push_back({"dynamic", getValueName(arg), arg.getType()});
    }
    args.push_back({"fixed", getValueName(match->ids), match->ids.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), args)));
    SmallVector<FusionArgSpec> outputs;
    for (const MaskedMulSite &site : match->sites) {
      outputs.push_back({"output", getFusionOutputName(site.output),
                         site.output.getType()});
    }
    if (match->flatMask) {
      outputs.push_back({"output", getFusionOutputName(match->flatMask),
                         match->flatMask.getType()});
    }
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    // 输出由 kernel 经执行上下文分配，不在调用参数里，故 kernel_arg_order 为空。
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                             rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    metadata.push_back(rewriter.getNamedAttr(
        "member_node_names",
        rewriter.getArrayAttr(collectMemberNodeNames(rewriter, fusedOps))));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));

    // execution v2 的主图区域保留原样（AOT 裁剪丢弃主图），只做标记。
    for (Operation *op : fusedOps) {
      op->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
    }
    return success();
  }
};

// ══ FusedLayerNorm（execution v2，ONNX 站点）═════════════════════════════════
//
// ONNX（TF 导出）的 reciprocal-multiply LayerNorm 链，11 个算子：
//
//   mean  = ReduceMean(x, 末轴, keepdims)
//   d     = Sub(x, mean)
//   var   = ReduceMean(Mul(d, d), 末轴, keepdims)
//   rsqrt = Reciprocal(Sqrt(Add(var, eps)))
//   g     = Mul(rsqrt, gamma)
//   y     = Add(Mul(x, g), Sub(beta, Mul(mean, g)))
//
// 融合后由一个 kernel 重算逐行矩并套同样的因式分解
// （annc/lib/Kernel/builtin_kernels/layer_norm_*），整链变一个节点。
//
// 只融"闭式"形态：每个中间值必须恰好只有链上列出的消费者。共享的
// mean/variance 会活下来，而 kernel 单输出提供不了，这类站点一律不动。
//
// 匹配条件对齐 ORT 的 LayerNorm/RM-LN 扩展（annc_opt 0001，
// layer_norm_fusion.cc::TryFuseReciprocalMulLayerNorm）。

// 恰好一个消费者，且是 expected。
static bool hasSingleUser(Value value, Operation *expected) {
  for (Operation *user : value.getUsers()) {
    if (user != expected) return false;
  }
  return value.getUsers().begin() != value.getUsers().end();
}

// 消费者全部在 allowed 里，且至少有一个。
static bool allUsersIn(Value value, ArrayRef<Operation *> allowed) {
  bool any = false;
  for (Operation *user : value.getUsers()) {
    any = true;
    if (!llvm::is_contained(allowed, user)) return false;
  }
  return any;
}

static bool isConstF32(Value value, int64_t rank) {
  auto tensorType = dyn_cast<atir::TensorType>(value.getType());
  if (!tensorType || !tensorType.getElementType().isF32()) return false;
  if (static_cast<int64_t>(tensorType.getShape().size()) != rank) return false;
  if (!value.getDefiningOp<ConstantOp>()) return false;
  return static_cast<bool>(tensorType.getCacheData());
}

// 末轴 keepdims 的 ReduceMean，链上两次归约都要求这一形态。
static bool isLastAxisMean(ReduceMeanOp mean, int64_t rank) {
  if (!mean.getKeepDims()) return false;
  SmallVector<int64_t, 4> axes;
  if (!readConstInts(mean.getIndices(), &axes) || axes.size() != 1) return false;
  int64_t axis = axes[0];
  if (axis < 0) axis += rank;
  return axis == rank - 1;
}

struct LayerNormMatch {
  Value input;   // rank-2 f32 x
  Value gamma;   // rank-1 f32 常量
  Value beta;    // rank-1 f32 常量
  Value eps;     // rank-0 f32 常量
  Value output;  // 末端 Add 的结果
  Value outputBuffer;
  SmallVector<Operation *, 10> chainOps;  // x 与输出之间的全部算子
};

static FailureOr<LayerNormMatch> matchLayerNorm(AddOp out) {
  if (out->hasAttr("annc.fusion_materialized")) return failure();
  if (out.getInputs().size() != 2) return failure();

  Value outputBuffer = out.getOutput();
  Operation *bufferOp = outputBuffer ? outputBuffer.getDefiningOp() : nullptr;
  if (!bufferOp || !isa<BufferOp>(bufferOp)) return failure();
  if (out.getResult().getType() != outputBuffer.getType()) return failure();

  auto outputType = dyn_cast<atir::TensorType>(out.getResult().getType());
  if (!outputType || outputType.getShape().size() != 2) return failure();
  if (!outputType.getElementType().isF32()) return failure();

  // y = Add(Mul(x, g), Sub(beta, Mul(mean, g)))
  Value xTerm, biasTerm;
  if (out.getInputs()[0].getDefiningOp<MulOp>()) {
    xTerm = out.getInputs()[0];
    biasTerm = out.getInputs()[1];
  } else {
    xTerm = out.getInputs()[1];
    biasTerm = out.getInputs()[0];
  }
  if (out.getDoRelu() || out.getScalar()) return failure();
  auto xMul = xTerm.getDefiningOp<MulOp>();
  auto biasSub = biasTerm.getDefiningOp<SubOp>();
  if (!xMul || !biasSub) return failure();
  if (!hasSingleUser(out.getInputs()[0], out.getOperation()) ||
      !hasSingleUser(out.getInputs()[1], out.getOperation())) {
    return failure();
  }
  if (biasSub.getX().getDefiningOp<ConstantOp>() == nullptr) return failure();
  if (biasSub.getY().getDefiningOp<MulOp>() == nullptr) return failure();
  if (!hasSingleUser(biasSub.getResult(), out.getOperation())) return failure();

  Value beta = biasSub.getX();
  Value meanGTerm = biasSub.getY();
  auto meanMul = meanGTerm.getDefiningOp<MulOp>();
  if (!isConstF32(beta, 1)) return failure();
  if (!hasSingleUser(meanGTerm, biasSub.getOperation())) return failure();

  // x* 与 mean* 乘的是同一个 g = Mul(rsqrt, gamma)。
  Value x = xMul.getX();
  Value g = xMul.getY();
  if (x.getDefiningOp<MulOp>() && !g.getDefiningOp<MulOp>()) {
    std::swap(x, g);
  }
  auto gMul = g.getDefiningOp<MulOp>();
  if (!gMul) return failure();
  if (meanMul.getX() != g && meanMul.getY() != g) return failure();
  Value mean = (meanMul.getX() == g) ? meanMul.getY() : meanMul.getX();
  if (!allUsersIn(g, {xMul.getOperation(), meanMul.getOperation()})) {
    return failure();
  }

  // g = Mul(rsqrt, gamma)，gamma 是常量因子。
  Value rsqrt = gMul.getX();
  Value gamma = gMul.getY();
  if (rsqrt.getDefiningOp<ConstantOp>() && !gamma.getDefiningOp<ConstantOp>()) {
    std::swap(rsqrt, gamma);
  }
  if (!isConstF32(gamma, 1)) return failure();
  if (!hasSingleUser(gamma, gMul.getOperation())) return failure();

  // rsqrt = Reciprocal(Sqrt(Add(var, eps)))
  auto reciprocal = rsqrt.getDefiningOp<ReciprocalOp>();
  if (!reciprocal || !hasSingleUser(rsqrt, gMul.getOperation())) {
    return failure();
  }
  Value sqrt = reciprocal.getX();
  auto sqrtOp = sqrt.getDefiningOp<SqrtOp>();
  if (!sqrtOp || !hasSingleUser(sqrt, reciprocal.getOperation())) {
    return failure();
  }
  Value add = sqrtOp.getX();
  auto addOp = add.getDefiningOp<AddOp>();
  if (!addOp || addOp.getInputs().size() != 2 ||
      !hasSingleUser(add, sqrtOp.getOperation())) {
    return failure();
  }
  if (addOp.getDoRelu() || addOp.getScalar()) return failure();
  Value variance, eps;
  if (addOp.getInputs()[0].getDefiningOp<ConstantOp>()) {
    eps = addOp.getInputs()[0];
    variance = addOp.getInputs()[1];
  } else {
    variance = addOp.getInputs()[0];
    eps = addOp.getInputs()[1];
  }
  if (!isConstF32(eps, 0)) return failure();

  // variance = ReduceMean(Mul(d, d));  d = Sub(x, mean);  mean = ReduceMean(x)
  auto varianceMean = variance.getDefiningOp<ReduceMeanOp>();
  if (!varianceMean || !hasSingleUser(variance, addOp.getOperation())) {
    return failure();
  }
  Value squared = varianceMean.getInput();
  auto squaredMul = squared.getDefiningOp<MulOp>();
  if (!squaredMul || !hasSingleUser(squared, varianceMean.getOperation())) {
    return failure();
  }
  Value d = squaredMul.getX();
  if (squaredMul.getY() != d) return failure();  // Mul(d, d)
  auto sub = d.getDefiningOp<SubOp>();
  if (!sub || !hasSingleUser(d, squaredMul.getOperation())) return failure();
  if (sub.getX() != x) return failure();  // Sub(x, mean)
  Value meanValue = sub.getY();
  if (meanValue != mean) return failure();
  auto meanOp = meanValue.getDefiningOp<ReduceMeanOp>();
  if (!meanOp) return failure();
  // mean 只喂减法和 mean*g 两项。
  if (!allUsersIn(meanValue, {sub.getOperation(), meanMul.getOperation()})) {
    return failure();
  }

  auto xType = dyn_cast<atir::TensorType>(x.getType());
  if (!xType || xType.getShape().size() != 2) return failure();
  if (!xType.getElementType().isF32()) return failure();
  const int64_t rank = 2;
  if (!isLastAxisMean(meanOp, rank) || !isLastAxisMean(varianceMean, rank)) {
    return failure();
  }
  // 归约轴 keepdims，且类型上必须是 1，逐元素广播才对得上。
  auto meanType = dyn_cast<atir::TensorType>(meanValue.getType());
  if (!meanType || meanType.getShape().size() != 2) return failure();
  if (meanType.getShape()[1] != 1) return failure();
  if (meanOp.getInput() != x) return failure();
  // x 是边界输入：站点外的消费者继续读它没问题（融合节点以实参取 x）；
  // 不能逃逸的是链上的中间值，逐个在上面查过。

  LayerNormMatch match;
  match.input = x;
  match.gamma = gamma;
  match.beta = beta;
  match.eps = eps;
  match.output = out.getResult();
  match.outputBuffer = outputBuffer;
  match.chainOps = {out.getOperation(),          xMul.getOperation(),
                    biasSub.getOperation(),      meanMul.getOperation(),
                    gMul.getOperation(),         reciprocal.getOperation(),
                    sqrtOp.getOperation(),       addOp.getOperation(),
                    varianceMean.getOperation(), squaredMul.getOperation(),
                    sub.getOperation(),          meanOp.getOperation()};
  return match;
}

// Execution v2 的 kernel 函数：(execution, input, gamma, beta, eps) -> output，
// 体内一次 atir.CustomizeOp；输出由 kernel 经执行上下文按槽位 0 分配。
static atir::CustomOpSchema getLayerNormSchema() {
  return atir::CustomOpSchema::get("LayerNorm")
      .TypeVar("T")
      .MemRefArg("input", 2, "T")
      .MemRefArg("gamma", 1, "T")
      .MemRefArg("beta", 1, "T")
      .MemRefArg("eps", 0, "T")
      .Result("output", 2, "T");
}

static func::FuncOp createLayerNormKernelFunc(ModuleOp module,
                                              PatternRewriter &rewriter,
                                              StringRef kernelName,
                                              const LayerNormMatch &match) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {
      LLVM::LLVMPointerType::get(rewriter.getContext()), match.input.getType(),
      match.gamma.getType(), match.beta.getType(), match.eps.getType()};
  auto funcType = rewriter.getFunctionType(inputTypes, match.output.getType());
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
  func->setAttr("annc.kernel", rewriter.getUnitAttr());
  func->setAttr("fusion.pattern", rewriter.getStringAttr("layer_norm"));

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  auto customCall = rewriter.create<atir::CustomizeOp>(
      func.getLoc(), TypeRange{match.output.getType()},
      ValueRange{entry->getArgument(1), entry->getArgument(2),
                 entry->getArgument(3), entry->getArgument(4)},
      rewriter.getStringAttr("LayerNorm"),
      getLayerNormSchema().toMetadata(rewriter.getContext()));
  rewriter.create<func::ReturnOp>(func.getLoc(), customCall.getResult(0));
  return func;
}

struct FuseLayerNormAsFuncCallPattern : public OpRewritePattern<AddOp> {
  // benefit 3：同一 Add 上还有通用逐元素融合（以及 masked_mul 的落点在
  // Mul 上），本形态最具体，先试。
  explicit FuseLayerNormAsFuncCallPattern(MLIRContext *context)
      : OpRewritePattern<AddOp>(context, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    auto match = matchLayerNorm(addOp);
    if (failed(match)) return failure();
    ModuleOp module = addOp->getParentOfType<ModuleOp>();
    if (!module) return failure();
    // 站点本轮只服务 ONNX 导入（同 FTDMM / masked_mul）。
    if (!module->hasAttr("annc.contract_version")) return failure();

    // kernel 必须先确认可用，否则会留下无法 lower 的自定义算子。
    auto schema = getLayerNormSchema();
    SmallVector<Type> schemaInputTypes = {
        match->input.getType(), match->gamma.getType(), match->beta.getType(),
        match->eps.getType()};
    auto inferred = atir::inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(schemaInputTypes),
        TypeRange{match->output.getType()});
    if (!inferred) {
      llvm::consumeError(inferred.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = "LayerNorm";
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*inferred);
    if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

    SmallVector<Operation *, 16> fusedOps(match->chainOps.begin(),
                                          match->chainOps.end());
    SmallVector<Value> escaping = atir::collectEscapingResults(fusedOps);
    if (escaping.size() != 1 || escaping[0] != match->output) return failure();

    std::string anchorName = getTfName(match->output.getDefiningOp());
    std::string kernelName = uniquifySymbolName(
        module, getStableFusionKernelName(anchorName, "layer_norm"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/layer_norm"));

    auto kernelFunc =
        createLayerNormKernelFunc(module, rewriter, kernelName, *match);
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("layer_norm")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> args;
    args.push_back(
        {"dynamic", getValueName(match->input), match->input.getType()});
    args.push_back(
        {"fixed", getValueName(match->gamma), match->gamma.getType()});
    args.push_back({"fixed", getValueName(match->beta), match->beta.getType()});
    args.push_back({"fixed", getValueName(match->eps), match->eps.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), args)));
    SmallVector<FusionArgSpec> outputs;
    outputs.push_back({"output", getFusionOutputName(match->output),
                       match->output.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    // 输出由 kernel 经执行上下文分配，不在调用参数里。
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                             rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    metadata.push_back(rewriter.getNamedAttr(
        "member_node_names",
        rewriter.getArrayAttr(collectMemberNodeNames(rewriter, fusedOps))));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));

    for (Operation *op : fusedOps) {
      op->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
    }
    return success();
  }
};

// ══ FusedBatchedMatMulAdd（execution v2，ONNX 站点）═════════════════════════
//
// ONNX 的 rank-3 MatMul + 广播 bias Add：
//
//   MatMul(data [B, d1, K], W [B, K, N]) -> Add(., bias [B, 1, N])
//
// bias 的前导维严格等于输出 batch；权重可以广播或按 batch 存储。
// 融合后是逐 batch 的行级小 GEMM 加对应 bias 初值
// （annc/lib/Kernel/builtin_kernels/batched_matmul_add_*）：
//
//   out[i, j, n] = bias[i, 0, n] + sum_k data[i, j, k] * W[i, k, n]
//
// bias 在 k 求和之后加，与源链的顺序一致。
//
// 匹配条件对齐 ORT 的 FusedBatchedMatMulAdd（annc_opt 0003/0006，
// batched-matmul-add-patch::FuseMatMulAdd）；支持 rank-3 常量权重与
// [B,1,N] bias，以及安全的前置常量仿射折叠；rank-4 仍不支持。

struct BatchedMatMulAddMatch {
  Value data;          // rank-3 f32 lhs
  Value weight;        // rank-3 f32 常量 [B, K, N]
  Value bias;          // rank-3 f32 常量 [B, 1, N]
  Value scale;         // pre-affine scale；无该链时为空
  Value offset;        // pre-affine offset；无该链时为空
  Value affineResult;  // pre-affine Add 的结果；无该链时为空
  Value scaledResult;  // pre-affine Mul 的结果；无该链时为空
  Value output;        // Add 的结果（站点输出）
  Value outputBuffer;  // Add 的输出缓冲
  Value matmulResult;  // 与 Add 一起进站点
  DenseElementsAttr foldedWeight;
  DenseElementsAttr foldedBias;
};

// 前导维等于 leadingDims 的 rank-3 f32 常量；返回其类型或空。
static atir::TensorType matchBatchedConst(Value value,
                                        ArrayRef<int64_t> leadingDims) {
  auto tensorType = dyn_cast<atir::TensorType>(value.getType());
  if (!tensorType || tensorType.getShape().size() != 3) return {};
  if (!tensorType.getElementType().isF32()) return {};
  if (!value.getDefiningOp<ConstantOp>()) return {};
  if (!tensorType.getCacheData()) return {};
  for (size_t axis = 0; axis < leadingDims.size(); ++axis) {
    if (leadingDims[axis] >= 0 &&
        tensorType.getShape()[axis] != leadingDims[axis]) {
      return {};
    }
  }
  return tensorType;
}

static bool isF32Constant(Value value) {
  auto tensorType = dyn_cast<atir::TensorType>(value.getType());
  return tensorType && tensorType.getElementType().isF32() &&
         value.getDefiningOp<ConstantOp>() && tensorType.getCacheData();
}

// Pre-affine constants may broadcast over the batch axes and K, but never
// over M: otherwise one folded W/bias cannot represent every output row.
static bool isBatchedAffineConstant(Value value, ArrayRef<int64_t> batchShape,
                                   int64_t contractDim) {
  auto type = dyn_cast<atir::TensorType>(value.getType());
  if (!type || !isF32Constant(value)) return false;
  ArrayRef<int64_t> shape = type.getShape();
  const size_t rank = batchShape.size() + 2;
  if (shape.size() > rank) return false;
  const size_t leading = rank - shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    const size_t axis = leading + i;
    const int64_t expected = axis < batchShape.size() ? batchShape[axis]
                            : axis == rank - 1 ? contractDim : 1;
    if (shape[i] != 1 && shape[i] != expected) return false;
  }
  return true;
}

static bool matchBatchedPreAffine(Value value, ArrayRef<int64_t> batchShape,
                                 int64_t contractDim, Value &data,
                                 Value &scale, Value &offset) {
  auto affine = value.getDefiningOp<AddOp>();
  if (!affine || affine.getInputs().size() != 2 || affine.getDoRelu() ||
      affine.getScalar()) {
    return false;
  }
  Value mulResult;
  Value candidateOffset;
  if (affine.getInputs()[0].getDefiningOp<MulOp>()) {
    mulResult = affine.getInputs()[0];
    candidateOffset = affine.getInputs()[1];
  } else if (affine.getInputs()[1].getDefiningOp<MulOp>()) {
    mulResult = affine.getInputs()[1];
    candidateOffset = affine.getInputs()[0];
  } else {
    return false;
  }
  auto mul = mulResult.getDefiningOp<MulOp>();
  if (!mul) return false;
  Value candidateScale = isF32Constant(mul.getX()) ? mul.getX() : mul.getY();
  Value candidateData = candidateScale == mul.getX() ? mul.getY() : mul.getX();
  if (!isBatchedAffineConstant(candidateScale, batchShape, contractDim) ||
      !isBatchedAffineConstant(candidateOffset, batchShape, contractDim) ||
      isF32Constant(candidateData)) {
    return false;
  }
  auto dataType = dyn_cast<atir::TensorType>(candidateData.getType());
  auto affineType = dyn_cast<atir::TensorType>(value.getType());
  auto mulType = dyn_cast<atir::TensorType>(mulResult.getType());
  if (!dataType || !affineType || !mulType ||
      dataType.getShape() != affineType.getShape() ||
      dataType.getShape() != mulType.getShape() ||
      !dataType.getElementType().isF32() ||
      !mulType.getElementType().isF32()) {
    return false;
  }
  if (!value.hasOneUse()) return false;
  for (Operation *user : mulResult.getUsers()) {
    if (user != affine.getOperation()) return false;
  }
  data = candidateData;
  scale = candidateScale;
  offset = candidateOffset;
  return data && scale && offset;
}

static FailureOr<BatchedMatMulAddMatch> matchBatchedMatMulAdd(AddOp add) {
  if (add->hasAttr("annc.fusion_materialized")) return failure();

  Value outputBuffer = add.getOutput();
  Operation *bufferOp = outputBuffer ? outputBuffer.getDefiningOp() : nullptr;
  if (!bufferOp || !isa<BufferOp>(bufferOp)) return failure();
  if (add.getResult().getType() != outputBuffer.getType()) return failure();

  auto outputType = dyn_cast<atir::TensorType>(add.getResult().getType());
  if (!outputType || (outputType.getShape().size() != 3 &&
                      outputType.getShape().size() != 4)) return failure();
  if (!outputType.getElementType().isF32()) return failure();

  // rank-4 形态（model3 的 domain_2 站点）：[B0,B1,M,K] × W[W0,W1,K,N] +
  // bias[B0,B1,1,N] -> [B0,B1,M,N]。前导两维按 numpy 广播（data/W 逐维为 1
  // 或输出维），bias 的批量前缀必须与输出逐维一致——与 patch 的
  // HasSupportedShapes 同口径（batched-matmul-add-patch/0006，FuseMatMulAdd）。
  // 常量 W/bias 取 cacheData，末两维（K/N 与 M）在已知时必须自洽；M 通常
  // 是动态维，留给 kernel 从 memref 描述符读实际值。
  if (outputType.getShape().size() == 4) {
    if (add.getInputs().size() != 2 || add.getDoRelu() || add.getScalar()) {
      return failure();
    }
    Value matmulResult = add.getInputs()[0];
    Value bias = add.getInputs()[1];
    if (!matmulResult.getDefiningOp<MatMulOp>()) std::swap(matmulResult, bias);
    auto matmul = matmulResult.getDefiningOp<MatMulOp>();
    if (!matmul || matmul.getResult() != matmulResult) return failure();
    if (matmul.getWithBias() || matmul.getDoRelu() ||
        matmul.getRightTranspose() || matmul.getLeftTranspose() ||
        matmul.getOutputTranspose() || matmul.getMStart() ||
        matmul.getNStart() || matmul.getKStart() || matmul.getMSize() ||
        matmul.getNSize() || matmul.getKSize() ||
        matmul->getAttr("rhs_format")) {
      return failure();
    }
    // MatMul 只喂这个 Add；折叠进第二个消费者会悄悄重算它的输入。
    for (Operation *user : matmulResult.getUsers()) {
      if (user != add.getOperation()) return failure();
    }
    auto dataType = dyn_cast<atir::TensorType>(matmul.getLhs().getType());
    auto weightType = dyn_cast<atir::TensorType>(matmul.getRhs().getType());
    auto biasType = dyn_cast<atir::TensorType>(bias.getType());
    if (!dataType || !weightType || !biasType ||
        dataType.getShape().size() != 4 || weightType.getShape().size() != 4 ||
        biasType.getShape().size() != 4 || !dataType.getElementType().isF32() ||
        !isF32Constant(matmul.getRhs()) || !isF32Constant(bias)) {
      return failure();
    }
    ArrayRef<int64_t> dataShape = dataType.getShape();
    ArrayRef<int64_t> weightShape = weightType.getShape();
    ArrayRef<int64_t> biasShape = biasType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    const int64_t contractDim = weightShape[2];
    const int64_t outputDim = weightShape[3];
    if (contractDim <= 0 || outputDim <= 0 || biasShape[2] != 1 ||
        biasShape[3] != outputDim || outputShape[3] != outputDim) {
      return failure();
    }
    if (dataShape[3] >= 0 && dataShape[3] != contractDim) return failure();
    if (dataShape[3] == 0) return failure();
    for (size_t axis = 0; axis < 2; ++axis) {
      const int64_t batch = outputShape[axis];
      if (batch <= 0 || biasShape[axis] != batch) return failure();
      if (dataShape[axis] != 1 && dataShape[axis] != batch) return failure();
      if (weightShape[axis] != 1 && weightShape[axis] != batch)
        return failure();
    }
    if (dataShape[2] >= 0 && outputShape[2] >= 0 &&
        dataShape[2] != outputShape[2]) {
      return failure();
    }
    if (dataShape[2] == 0) return failure();

    BatchedMatMulAddMatch match;
    match.data = matmul.getLhs();
    if (matchBatchedPreAffine(matmul.getLhs(), outputShape.take_front(2),
                              contractDim, match.data, match.scale, match.offset)) {
      match.affineResult = matmul.getLhs();
      auto affine = cast<AddOp>(matmul.getLhs().getDefiningOp());
      match.scaledResult = affine.getInputs()[0].getDefiningOp<MulOp>()
                               ? affine.getInputs()[0]
                               : affine.getInputs()[1];
    }
    match.weight = matmul.getRhs();
    match.bias = bias;
    match.output = add.getResult();
    match.outputBuffer = outputBuffer;
    match.matmulResult = matmulResult;
    return match;
  }

  // 两操作数一个是 MatMul 结果，另一个是 bias 常量。
  if (add.getInputs().size() != 2) return failure();
  Value bias;
  Value matmulResult;
  atir::TensorType biasType = matchBatchedConst(add.getInputs()[1], {-1, 1});
  if (biasType) {
    bias = add.getInputs()[1];
    matmulResult = add.getInputs()[0];
  } else {
    biasType = matchBatchedConst(add.getInputs()[0], {-1, 1});
    if (!biasType) return failure();
    bias = add.getInputs()[0];
    matmulResult = add.getInputs()[1];
  }
  const int64_t outputDim = biasType.getShape()[2];
  if (outputDim <= 0 || outputType.getShape()[2] != outputDim) return failure();
  const int64_t batchDim = biasType.getShape()[0];
  if (batchDim <= 0 || outputType.getShape()[0] != batchDim) {
    return failure();
  }

  auto matmul = matmulResult.getDefiningOp<MatMulOp>();
  if (!matmul || matmul.getResult() != matmulResult) return failure();
  // 与既有 matmul 融合相同的白名单：任何 transpose / start-size 标注 /
  // rhs_format 都会改变 kernel 的计算。
  if (matmul.getWithBias() || matmul.getDoRelu() ||
      matmul.getRightTranspose() || matmul.getLeftTranspose() ||
      matmul.getOutputTranspose() || matmul.getMStart() || matmul.getNStart() ||
      matmul.getKStart() || matmul.getMSize() || matmul.getNSize() ||
      matmul.getKSize() || matmul->getAttr("rhs_format") || add.getDoRelu() ||
      add.getScalar()) {
    return failure();
  }

  // MatMul 只喂这个 Add；折叠进第二个消费者会悄悄重算它的输入。
  for (Operation *user : matmulResult.getUsers()) {
    if (user != add.getOperation()) return failure();
  }

  atir::TensorType weightType = matchBatchedConst(matmul.getRhs(), {-1});
  if (!weightType) return failure();
  const int64_t weightBatch = weightType.getShape()[0];
  if (weightBatch != 1 && weightBatch != batchDim) return failure();
  const int64_t contractDim = weightType.getShape()[1];
  if (contractDim <= 0 || weightType.getShape()[2] != outputDim) {
    return failure();
  }

  auto dataType = dyn_cast<atir::TensorType>(matmul.getLhs().getType());
  if (!dataType || dataType.getShape().size() != 3) return failure();
  if (!dataType.getElementType().isF32()) return failure();
  if (dataType.getShape()[2] >= 0 && dataType.getShape()[2] != contractDim) {
    return failure();
  }
  if (dataType.getShape()[2] == 0) return failure();
  if ((dataType.getShape()[0] != 1 && dataType.getShape()[0] != batchDim) ||
      dataType.getShape()[1] != outputType.getShape()[1]) {
    return failure();
  }
  if (dataType.getShape()[0] == 1 && weightBatch != batchDim) return failure();

  BatchedMatMulAddMatch match;
  match.data = matmul.getLhs();
  if (matchBatchedPreAffine(matmul.getLhs(), ArrayRef<int64_t>{batchDim}, contractDim,
                           match.data, match.scale, match.offset)) {
    match.affineResult = matmul.getLhs();
    auto affine = cast<AddOp>(matmul.getLhs().getDefiningOp());
    match.scaledResult = affine.getInputs()[0].getDefiningOp<MulOp>()
                             ? affine.getInputs()[0]
                             : affine.getInputs()[1];
  }
  match.weight = matmul.getRhs();
  match.bias = bias;
  match.output = add.getResult();
  match.outputBuffer = outputBuffer;
  match.matmulResult = matmulResult;
  return match;
}

// 站点的两个形态共用一套代码：rank-3（[B,K,N] 权重）沿用原 kernel 名，rank-4
// 用独立符号（执行 v2 的 C 签名按符号固定，rank 无法参数化）。
static int64_t batchedMatMulAddRank(Type type) {
  auto tensorType = dyn_cast<atir::TensorType>(type);
  return tensorType ? static_cast<int64_t>(tensorType.getShape().size()) : 0;
}

static const char *getBatchedMatMulAddOpName(Type type) {
  return batchedMatMulAddRank(type) == 4 ? "BatchedMatMulAdd4"
                                         : "BatchedMatMulAdd";
}

// Execution v2 的 kernel 函数：(execution, data, weight, bias) -> output，体内
// 一次 atir.CustomizeOp；输出由 kernel 经执行上下文按槽位 0 分配。
static atir::CustomOpSchema getBatchedMatMulAddSchema(int64_t rank) {
  return atir::CustomOpSchema::get(rank == 4 ? "BatchedMatMulAdd4"
                                             : "BatchedMatMulAdd")
      .TypeVar("T")
      .MemRefArg("data", rank, "T")
      .MemRefArg("weight", rank, "T")
      .MemRefArg("bias", rank, "T")
      .Result("output", rank, "T");
}

static func::FuncOp createBatchedMatMulAddKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    const BatchedMatMulAddMatch &match) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {
      LLVM::LLVMPointerType::get(rewriter.getContext()), match.data.getType(),
      match.weight.getType(), match.bias.getType()};
  auto funcType = rewriter.getFunctionType(inputTypes, match.output.getType());
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
  func->setAttr("annc.kernel", rewriter.getUnitAttr());
  func->setAttr("fusion.pattern",
                rewriter.getStringAttr("batched_matmul_add"));

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  Value weight = entry->getArgument(2);
  Value bias = entry->getArgument(3);
  if (match.foldedWeight) {
    // Keep original initializer arguments in the site input plan. Their
    // names bind to real ORT inputs; the transformed payloads live inside
    // this function and never masquerade as those initializer values.
    auto weightType = cast<atir::TensorType>(match.weight.getType());
    auto biasType = cast<atir::TensorType>(match.bias.getType());
    weight = rewriter.create<ConstantOp>(
        func.getLoc(), weightType.cloneWithCacheData(match.foldedWeight),
        rewriter.getStringAttr("pre_affine_weight"),
        rewriter.getStringAttr("private"));
    weight.getDefiningOp()->setAttr("annc.effective_weight",
                                   rewriter.getUnitAttr());
    bias = rewriter.create<ConstantOp>(
        func.getLoc(), biasType.cloneWithCacheData(match.foldedBias),
        rewriter.getStringAttr("pre_affine_bias"),
        rewriter.getStringAttr("private"));
  }
  auto customCall = rewriter.create<atir::CustomizeOp>(
      func.getLoc(), TypeRange{match.output.getType()},
      ValueRange{entry->getArgument(1), weight, bias},
      rewriter.getStringAttr(getBatchedMatMulAddOpName(match.data.getType())),
      getBatchedMatMulAddSchema(batchedMatMulAddRank(match.data.getType()))
          .toMetadata(rewriter.getContext()));
  rewriter.create<func::ReturnOp>(func.getLoc(), customCall.getResult(0));
  return func;
}

static LogicalResult foldBatchedPreAffine(BatchedMatMulAddMatch &match) {
  if (!match.scale) return success();

  auto weightType = cast<atir::TensorType>(match.weight.getType());
  auto biasType = cast<atir::TensorType>(match.bias.getType());
  auto weightAttr = cast<DenseElementsAttr>(weightType.getCacheData());
  auto biasAttr = cast<DenseElementsAttr>(biasType.getCacheData());
  auto scaleAttr = cast<DenseElementsAttr>(
      cast<atir::TensorType>(match.scale.getType()).getCacheData());
  auto offsetAttr = cast<DenseElementsAttr>(
      cast<atir::TensorType>(match.offset.getType()).getCacheData());

  const auto weightShape = weightType.getShape();
  const auto biasShape = biasType.getShape();
  const size_t rank = weightShape.size();
  if ((rank != 3 && rank != 4) || biasShape.size() != rank) return failure();
  int64_t batchDim = 1;
  for (size_t axis = 0; axis + 2 < rank; ++axis) {
    // Keep broadcast weights on the original path: per-output-batch affine
    // constants would require expanding the weight initializer's shape.
    if (weightShape[axis] <= 0 || weightShape[axis] != biasShape[axis])
      return failure();
    batchDim *= weightShape[axis];
  }
  const int64_t contractDim = weightShape[rank - 2];
  const int64_t outputDim = weightShape[rank - 1];
  if (contractDim <= 0 || outputDim <= 0 || biasShape[rank - 2] != 1 ||
      biasShape[rank - 1] != outputDim) return failure();

  std::vector<float> weight(weightAttr.value_begin<float>(),
                            weightAttr.value_end<float>());
  std::vector<float> bias(biasAttr.value_begin<float>(),
                         biasAttr.value_end<float>());
  std::vector<float> scale(scaleAttr.value_begin<float>(),
                          scaleAttr.value_end<float>());
  std::vector<float> offset(offsetAttr.value_begin<float>(),
                           offsetAttr.value_end<float>());
  if (weight.size() != static_cast<size_t>(batchDim * contractDim * outputDim) ||
      bias.size() != static_cast<size_t>(batchDim * outputDim)) {
    return failure();
  }

  std::vector<float> foldedWeight(weight.size());
  std::vector<float> foldedBias(bias.size());
  for (int64_t batch = 0; batch < batchDim; ++batch) {
    for (int64_t n = 0; n < outputDim; ++n) {
      float value = bias[batch * outputDim + n];
      for (int64_t k = 0; k < contractDim; ++k) {
        const size_t index =
            static_cast<size_t>((batch * contractDim + k) * outputDim + n);
        auto affineIndex = [&](Value value) {
          auto shape = cast<atir::TensorType>(value.getType()).getShape();
          if (shape.empty()) return int64_t{0};
          const size_t leading = rank - shape.size();
          int64_t coordinates[4] = {0, 0, 0, 0};
          coordinates[rank - 1] = k;
          int64_t remaining = batch;
          for (int64_t axis = static_cast<int64_t>(rank) - 3; axis >= 0; --axis) {
            coordinates[axis] = remaining % weightShape[axis];
            remaining /= weightShape[axis];
          }
          int64_t index = 0;
          for (size_t axis = 0; axis < shape.size(); ++axis)
            index = index * shape[axis] +
                    (shape[axis] == 1 ? 0 : coordinates[leading + axis]);
          return index;
        };
        const float scaleValue = scale[affineIndex(match.scale)];
        const float offsetValue = offset[affineIndex(match.offset)];
        foldedWeight[index] = scaleValue * weight[index];
        value += offsetValue * weight[index];
      }
      foldedBias[batch * outputDim + n] = value;
    }
  }

  auto f32Type = weightAttr.getElementType();
  match.foldedWeight = DenseElementsAttr::get(
      RankedTensorType::get(weightShape, f32Type), ArrayRef<float>(foldedWeight));
  match.foldedBias = DenseElementsAttr::get(
      RankedTensorType::get(biasShape, f32Type), ArrayRef<float>(foldedBias));
  return success();
}

struct FuseBatchedMatMulAddAsFuncCallPattern : public OpRewritePattern<AddOp> {
  // benefit 2：同一 Add 上通用逐元素融合也可能匹配，本形态更具体。
  // LayerNorm 是 benefit 3，先于本 pattern 试。
  explicit FuseBatchedMatMulAddAsFuncCallPattern(MLIRContext *context)
      : OpRewritePattern<AddOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    auto match = matchBatchedMatMulAdd(addOp);
    if (failed(match)) return failure();
    ModuleOp module = addOp->getParentOfType<ModuleOp>();
    if (!module) return failure();
    // 站点本轮只服务 ONNX 导入（同 FTDMM / masked_mul / layer_norm）。
    if (!module->hasAttr("annc.contract_version")) return failure();

    // kernel 必须先确认可用，否则会留下无法 lower 的自定义算子。
    auto schema =
        getBatchedMatMulAddSchema(batchedMatMulAddRank(match->data.getType()));
    SmallVector<Type> schemaInputTypes = {
        match->data.getType(), match->weight.getType(), match->bias.getType()};
    auto inferred = atir::inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(schemaInputTypes),
        TypeRange{match->output.getType()});
    if (!inferred) {
      llvm::consumeError(inferred.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = getBatchedMatMulAddOpName(match->data.getType());
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*inferred);
    if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

    SmallVector<Operation *, 6> fusedOps = {
        addOp.getOperation(), match->matmulResult.getDefiningOp()};
    if (match->affineResult) {
      fusedOps.push_back(match->affineResult.getDefiningOp());
      fusedOps.push_back(match->scaledResult.getDefiningOp());
    }
    SmallVector<Value> escaping = atir::collectEscapingResults(fusedOps);
    if (escaping.size() != 1 || escaping[0] != match->output) return failure();

    // Folding is pure preparation. Unsupported affine/weight broadcasting
    // keeps the original affine result as the BMM input.
    if (failed(foldBatchedPreAffine(*match))) {
      match->data = cast<MatMulOp>(match->matmulResult.getDefiningOp()).getLhs();
      match->scale = {};
      match->offset = {};
      match->affineResult = {};
      match->scaledResult = {};
      fusedOps.resize(2);
    }

    std::string anchorName = getTfName(match->output.getDefiningOp());
    std::string kernelName = uniquifySymbolName(
        module, getStableFusionKernelName(anchorName, "batched_matmul_add"));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(anchorName + "/batched_matmul_add"));

    auto kernelFunc = createBatchedMatMulAddKernelFunc(module, rewriter,
                                                       kernelName, *match);
    setExecutionMode(kernelFunc, rewriter, kAotExecutionMode);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("batched_matmul_add")));
    addExecutionModeMetadata(metadata, rewriter, kAotExecutionMode);
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    SmallVector<FusionArgSpec> args;
    args.push_back(
        {"dynamic", getValueName(match->data), match->data.getType()});
    args.push_back(
        {"fixed", getValueName(match->weight), match->weight.getType()});
    args.push_back({"fixed", getValueName(match->bias), match->bias.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "args", makeFusionArgArray(rewriter.getContext(), args)));
    SmallVector<FusionArgSpec> outputs;
    outputs.push_back({"output", getFusionOutputName(match->output),
                       match->output.getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    // 输出由 kernel 经执行上下文分配，不在调用参数里。
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                             rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    metadata.push_back(rewriter.getNamedAttr(
        "member_node_names",
        rewriter.getArrayAttr(collectMemberNodeNames(rewriter, fusedOps))));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));

    for (Operation *op : fusedOps) {
      op->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
    }
    return success();
  }
};

}  // namespace

namespace atir {
void populateAtirOpFusionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<FuseDnnEmbeddingHashBucketAsFuncCallPattern>(ctx);
  patterns.add<FuseMatMulAsFuncCallPattern>(ctx);
}

class AtirOpFusionPass : public AtirOpFusionBase<AtirOpFusionPass> {
 public:
  AtirOpFusionPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    if (!mainFunc) return;

    WalkResult validation = mainFunc.walk([&](MatMulOp matmul) {
      auto candidate = matmul->getAttrOfType<DictionaryAttr>(
          kGemmEpilogueCandidateAttr);
      if (!candidate) return WalkResult::advance();
      auto parsed = parseEpilogueCandidatePlan(candidate);
      if (failed(parsed) ||
          failed(bindEpilogueProgramToSource(matmul, *parsed))) {
        matmul.emitOpError(
            "epilogue candidate does not match the current source chain");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (validation.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<FuseDnnEmbeddingHashBucketAsFuncCallPattern>(&getContext());
    patterns.add<FuseMatMulAsFuncCallPattern>(&getContext());
    if (annc::fusionPatternEnabled("tensordot_matmul"))
      patterns.add<FuseTensordotMatMulAsFuncCallPattern>(&getContext());
    if (annc::fusionPatternEnabled("masked_mul"))
      patterns.add<FuseMaskedMulAsFuncCallPattern>(&getContext());
    if (annc::fusionPatternEnabled("layer_norm"))
      patterns.add<FuseLayerNormAsFuncCallPattern>(&getContext());
    if (annc::fusionPatternEnabled("batched_matmul_add"))
      patterns.add<FuseBatchedMatMulAddAsFuncCallPattern>(&getContext());
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
    SmallVector<std::string, 4> envEnabledCustomOps;
    appendCustomOpTypesFromEnv(kEnableCustomOpsEnv, envEnabledCustomOps);
    SmallVector<std::string, 4> envDisabledCustomOps;
    appendCustomOpTypesFromEnv(kDisableCustomOpsEnv, envDisabledCustomOps);
    patterns.add<FuseKdnnMatMulAsFuncCallPattern>(
        &getContext(),
        CustomOpTypeFilter(envEnabledCustomOps, envDisabledCustomOps));
#endif
    patterns.add<FuseKpEmbeddingActionIdGatherAsFuncCallPattern>(
        &getContext());
    patterns.add<FuseKpTargetBehaviorInteractionAsFuncCallPattern>(
        &getContext());
    patterns.add<FuseKpSparseDynamicStitchAsFuncCallPattern>(&getContext());
    patterns.add<FuseKpSparseSegmentReduceAsFuncCallPattern>(&getContext());
    patterns.add<FuseKpEmbeddingPaddingAsFuncCallPattern>(&getContext());
    patterns.add<FuseKpEmbeddingPaddingFastAsFuncCallPattern>(&getContext());
    patterns.add<FuseKpSparseReshapeAsFuncCallPattern>(&getContext());
    patterns.add<FuseKpSparseSelectAsFuncCallPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(mainFunc, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionPass() {
  return std::make_unique<AtirOpFusionPass>();
}
}  // namespace atir
