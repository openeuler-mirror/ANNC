#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "Dialect/Atir/Passes/Patterns/CustomPatterns/KPFusedGatherMatch.h"
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
    std::string templateFingerprint =
        atir::computeAtirTemplateFingerprint(module, kernelFunc);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("dnn_embedding_hash_bucket")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "template_fingerprint", rewriter.getStringAttr(templateFingerprint)));
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

struct FuseKPFusedGatherAsFuncCallPattern : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp outerGather,
                                PatternRewriter &rewriter) const override {
    if (outerGather->hasAttr("annc.fusion_materialized")) return failure();
    auto match = matchKPFusedGather(outerGather);
    if (failed(match)) return failure();
    ModuleOp module = outerGather->getParentOfType<ModuleOp>();
    if (!module) return failure();

    SmallVector<Value, 3> boundaryInputs = {match->data, match->keys,
                                            match->begin};
    SmallVector<Operation *> fusedOps;
    SmallPtrSet<Operation *, 32> visited;
    for (Value output : match->boundaryOutputs) {
      collectDefiningOpsPostOrder(output.getDefiningOp(), boundaryInputs,
                                  visited, fusedOps);
    }
    if (fusedOps.empty() || !isClosedKernelOpSet(fusedOps, boundaryInputs))
      return failure();

    auto schema = CustomOpSchema::get("KPFusedGather")
                      .TypeVar("T")
                      .TypeVar("Tkeys")
                      .TypeVar("Tbegin")
                      .TypeVar("Tindices")
                      .MemRefArg("data", 2, "T")
                      .MemRefArg("keys", 2, "Tkeys")
                      .MemRefArg("begin", 1, "Tbegin")
                      .Result("unique_values", 1, "Tkeys")
                      .Result("unique_indices", 1, "Tindices")
                      .Result("gathered", 2, "T");
    SmallVector<Type> inputTypes;
    for (Value input : boundaryInputs) inputTypes.push_back(input.getType());
    SmallVector<Type> outputTypes;
    for (Value output : match->boundaryOutputs)
      outputTypes.push_back(output.getType());
    auto inferred = inferTypeConstraintsFromSchema(
        schema.toMetadata(rewriter.getContext()), TypeRange(inputTypes),
        TypeRange(outputTypes));
    if (!inferred) {
      llvm::consumeError(inferred.takeError());
      return failure();
    }
    annc::kernels::KernelResolveRequest request;
    request.op_type = "KPFusedGather";
    request.abi = "annc_execution_v2";
    request.type_constraints = std::move(*inferred);
    if (!annc::kernels::hasAnyAvailableKernel(request, false)) return failure();

    std::string kernelName = uniquifySymbolName(
        module, getStableFusionKernelName("kp_fused_gather", "execution_v2"));
    auto kernelFunc =
        createExecutionV2KernelFunc(module, rewriter, kernelName, fusedOps,
                                    boundaryInputs, match->boundaryOutputs);
    kernelFunc->setAttr("fusion.pattern",
                        rewriter.getStringAttr("kp_fused_gather"));
    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("kp_fused_gather")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr("kp_fused_gather")));
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
                       outputName(match->firstUnique.getY(), "first_unique:0"),
                       match->firstUnique.getY().getType()});
    outputs.push_back(
        {"output", outputName(match->firstUnique.getIdx(), "first_unique:1"),
         match->firstUnique.getIdx().getType()});
    outputs.push_back(
        {"output", outputName(match->outerGather.getResult(), "outer_gather:0"),
         match->outerGather.getResult().getType()});
    metadata.push_back(rewriter.getNamedAttr(
        "outputs", makeFusionArgArray(rewriter.getContext(), outputs)));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("annc_execution_v2")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {})));
    metadata.push_back(rewriter.getNamedAttr("symbolic_signature",
                                             rewriter.getStringAttr("")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
                        DictionaryAttr::get(rewriter.getContext(), metadata));
    outerGather->setAttr("annc.fusion_materialized", rewriter.getUnitAttr());
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

        SmallVector<NamedAttribute> metadata;
        metadata.push_back(rewriter.getNamedAttr(
            "fusion.pattern", rewriter.getStringAttr("matmul_add_relu")));
        metadata.push_back(rewriter.getNamedAttr(
            "kernel_name", rewriter.getStringAttr(kernelName)));
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
    std::string templateFingerprint =
        atir::computeAtirTemplateFingerprint(module, kernelFunc);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr("fusion.pattern",
                                             rewriter.getStringAttr(pattern)));
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

}  // namespace

namespace atir {
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
    patterns.add<FuseKPFusedGatherAsFuncCallPattern>(&getContext());
    patterns.add<FuseMatMulAsFuncCallPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(mainFunc, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionPass() {
  return std::make_unique<AtirOpFusionPass>();
}
}  // namespace atir
