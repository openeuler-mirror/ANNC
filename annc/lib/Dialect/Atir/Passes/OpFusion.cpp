#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/Passes/Patterns/FusionBoundaryUtils.h"
#include "Dialect/Atir/TemplateFingerprint.h"
#include "Helper.h"
#include "Kernel/KernelPriorityResolver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SHA256.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

static bool isReturnUser(Operation *op) { return isa<func::ReturnOp>(op); }

static bool hasOnlyNonReturnUser(Value value, Operation *allowedUser) {
  for (Operation *user : value.getUsers()) {
    if (isReturnUser(user)) continue;
    if (user != allowedUser) return false;
  }
  return true;
}

static void replaceNonReturnUsesWith(Value value, Value replacement) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : value.getUses()) {
    if (!isReturnUser(use.getOwner())) uses.push_back(&use);
  }
  for (OpOperand *use : uses) use->set(replacement);
}

static AddOp findCompatibleMatMulAddUser(MatMulOp matmulOp, Value &bias) {
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
    if (!hasOnlyNonReturnUser(matmulOp.getResult(), addOp)) continue;

    bias = candidateBias;
    return addOp;
  }
  return nullptr;
}

static ReluOp findCompatibleAddReluUser(AddOp addOp) {
  for (Operation *user : addOp->getResult(0).getUsers()) {
    auto reluOp = dyn_cast<ReluOp>(user);
    if (!reluOp) continue;
    if (hasOnlyNonReturnUser(addOp->getResult(0), reluOp)) return reluOp;
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

static bool isExecutionV2BuiltinCompatible(MatMulOp matmul, AddOp add,
                                           ReluOp relu) {
  if (!matmul || !add || !relu) return false;
  if (matmul.getWithBias() || matmul.getDoRelu() ||
      matmul.getRightTranspose() || matmul.getLeftTranspose() ||
      matmul.getOutputTranspose() || matmul.getMStart() || matmul.getNStart() ||
      matmul.getKStart() || matmul.getMSize() || matmul.getNSize() ||
      matmul.getKSize() || matmul->getAttr("rhs_format"))
    return false;
  if (add.getDoRelu() || add.getScalar()) return false;
  return relu.getReluLimit().convertToFloat() == -1.0f;
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
      loc, resultType, output, lhs, rhs, Value{}, rewriter.getBoolAttr(false),
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

static func::FuncOp createMatMulPostOpKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    MatMulOp matmulOp, Value output, Value bias, StringRef pattern,
    StringRef customOpName) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {matmulOp.getLhs().getType(),
                                  matmulOp.getRhs().getType(), output.getType(),
                                  bias.getType()};
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func =
      rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern", rewriter.getStringAttr(pattern));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  Value lhs = entry->getArgument(0);
  Value rhs = entry->getArgument(1);
  Value out = entry->getArgument(2);
  Value biasArg = entry->getArgument(3);

  auto matmulBuffer =
      rewriter.create<BufferOp>(func.getLoc(), out.getType()).getResult();
  auto matmul = createMatMulBodyOp(rewriter, func.getLoc(), out.getType(),
                                   matmulBuffer, lhs, rhs, matmulOp);

  Value postOpInput = matmul.getResult();
  if (pattern == "matmul_add_relu") {
    auto addBuffer =
        rewriter.create<BufferOp>(func.getLoc(), out.getType()).getResult();
    auto add = rewriter.create<AddOp>(
        func.getLoc(), out.getType(), addBuffer,
        ValueRange{postOpInput, biasArg}, rewriter.getBoolAttr(false),
        rewriter.getF32FloatAttr(-1.0f), FloatAttr());
    postOpInput = add.getResult();
    rewriter.create<ReluOp>(func.getLoc(), out.getType(), out, postOpInput,
                            rewriter.getF32FloatAttr(-1.0f));
  } else {
    rewriter.create<AddOp>(func.getLoc(), out.getType(), out,
                           ValueRange{postOpInput, biasArg},
                           rewriter.getBoolAttr(false),
                           rewriter.getF32FloatAttr(-1.0f), FloatAttr());
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

struct FuseMatMulAsFuncCallPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
#ifdef ANNC_ENABLE_CONSTANT_FOLDING
    if (!matmulOp.getRhsFormat()) return failure();
#endif
    if (matmulOp->hasAttr("annc.fusion_materialized")) return failure();
    if (matmulOp->hasAttr("annc.postop_fused")) return failure();
    if (matmulOp.getWithBias() || matmulOp.getDoRelu()) return failure();

    AddOp addOp = nullptr;
    ReluOp reluOp = nullptr;
    Value bias;
    bool hasBiasPostOp = false;
    bool hasReluPostOp = false;

    addOp = findCompatibleMatMulAddUser(matmulOp, bias);
    if (addOp) {
      hasBiasPostOp = true;
      reluOp = findCompatibleAddReluUser(addOp);
      hasReluPostOp = reluOp != nullptr;
    }

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
            atir::computeAtirTemplateFingerprint(module, kernelFunc);

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

    Operation *outputOp = hasReluPostOp   ? reluOp.getOperation()
                          : hasBiasPostOp ? addOp.getOperation()
                                          : matmulOp.getOperation();
    Value output = hasBiasPostOp ? outputOp->getOperand(0) : matmulOp.getC();
    Type outputType = output.getType();
    StringRef pattern = hasReluPostOp   ? StringRef("matmul_add_relu")
                        : hasBiasPostOp ? StringRef("matmul_add")
                                        : StringRef("matmul");
    StringRef customOpName = hasReluPostOp   ? StringRef("MatMulAddRelu")
                             : hasBiasPostOp ? StringRef("MatMulAdd")
                                             : StringRef("MatMul");
    std::string outputName = hasBiasPostOp ? getTfName(outputOp) : matmulName;
    std::string kernelName =
        hasBiasPostOp
            ? uniquifySymbolName(
                  module, getStableMatMulFusionKernelName(matmulOp, pattern))
            : uniquifySymbolName(module, getStableMatMulKernelName(matmulOp));
    std::string clusterName =
        uniquifyFusionName(module, sanitizeName("annc_fused_" + outputName));

    auto kernelFunc = hasBiasPostOp
        ? createMatMulPostOpKernelFunc(module, rewriter, kernelName, matmulOp,
                                       output, bias, pattern, customOpName)
        : createKernelFunc(module, rewriter, kernelName, matmulOp);
    setExecutionMode(kernelFunc, rewriter, kJitExecutionMode);
    std::string templateFingerprint =
        atir::computeAtirTemplateFingerprint(module, kernelFunc);

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
    SmallVector<int64_t> kernelArgOrder = lhsIsDynamic
                                              ? SmallVector<int64_t>{1, 0, 2}
                                              : SmallVector<int64_t>{0, 1, 2};
    SmallVector<FusionArgSpec> argSpecs;
    if (hasBiasPostOp) {
      argSpecs.push_back({"fixed", getValueName(matmulOp.getRhs()),
                          matmulOp.getRhs().getType()});
      argSpecs.push_back({"fixed", getValueName(bias), bias.getType()});
      argSpecs.push_back({"dynamic", getValueName(matmulOp.getLhs()),
                          matmulOp.getLhs().getType()});
    } else {
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
    if (hasBiasPostOp) {
      kernelArgOrder = {2, 0, 3, 1};
    }
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
    if (hasBiasPostOp) {
      rewriter.create<func::CallOp>(
          outputOp->getLoc(), kernelFunc,
          ValueRange{matmulOp.getLhs(), matmulOp.getRhs(), output, bias});

      rewriter.replaceAllUsesWith(outputOp->getResult(0), output);
      if (hasReluPostOp) {
        rewriter.eraseOp(reluOp);
        replaceNonReturnUsesWith(addOp->getResult(0), output);
      }
      bool erasedAdd = false;
      if (addOp->getResult(0).use_empty()) {
        rewriter.eraseOp(addOp);
        erasedAdd = true;
      }

      if (erasedAdd) replaceNonReturnUsesWith(matmulOp.getResult(), output);
      if (erasedAdd && matmulOp.getResult().use_empty()) {
        rewriter.eraseOp(matmulOp);
      } else {
        matmulOp->setAttr("annc.postop_fused", rewriter.getUnitAttr());
      }
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
    SmallVector<int64_t> kernelArgOrder;
    for (int64_t i = 0; i < static_cast<int64_t>(callInputs.size()) + 1; ++i) {
      kernelArgOrder.push_back(i);
    }
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

    RewritePatternSet patterns(&getContext());
    patterns.add<FuseDnnEmbeddingHashBucketAsFuncCallPattern>(&getContext());
    patterns.add<FuseMatMulAsFuncCallPattern>(&getContext());
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
