#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Passes.h"

#include "Helper.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SHA256.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
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
    if (auto tensorType = dyn_cast<atir::TensorType>(op->getResult(0).getType())) {
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
    if (auto loc = dyn_cast<NameLoc>(blockArg.getLoc())) return loc.getName().str();
    return "arg" + std::to_string(blockArg.getArgNumber());
  }
  if (Operation *op = value.getDefiningOp()) {
    return getTfName(op);
  }
  return "";
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

static ArrayAttr makeStringArray(MLIRContext *ctx, ArrayRef<std::string> values) {
  SmallVector<Attribute> attrs;
  for (const auto &value : values) attrs.push_back(StringAttr::get(ctx, value));
  return ArrayAttr::get(ctx, attrs);
}

static ArrayAttr makeI64Array(MLIRContext *ctx, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  Builder b(ctx);
  for (int64_t value : values) attrs.push_back(b.getI64IntegerAttr(value));
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

static std::string shapeToString(Type type) {
  auto tensorType = dyn_cast<atir::TensorType>(type);
  if (!tensorType) return "";

  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::interleaveComma(tensorType.getShape(), os,
                        [&](int64_t dim) { os << dim; });
  return os.str();
}

static bool startsWith(StringRef value, StringRef prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

static bool hasTfNamePrefix(Operation *op, StringRef prefix) {
  return startsWith(getTfName(op), prefix);
}

template <typename OpT>
static OpT findUserOf(Value value) {
  for (Operation *user : value.getUsers()) {
    if (auto op = dyn_cast<OpT>(user)) return op;
  }
  return nullptr;
}

template <typename OpT>
static OpT findNamedOp(func::FuncOp func, StringRef name) {
  OpT found = nullptr;
  func.walk([&](OpT op) {
    if (!found && getTfName(op.getOperation()) == name) found = op;
  });
  return found;
}

static func::FuncOp createDnnEmbeddingHashBucketKernelFunc(
    ModuleOp module, PatternRewriter &rewriter, StringRef kernelName,
    Type dynamicInputType, Type weightType, Type outputType,
    int64_t numBuckets) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kernelName)) {
    return existing;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  SmallVector<Type> inputTypes = {dynamicInputType, weightType, outputType};
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func = rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern",
                rewriter.getStringAttr("dnn_embedding_hash_bucket"));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  Value dynamicInput = entry->getArgument(0);
  Value weight = entry->getArgument(1);
  Value output = entry->getArgument(2);

  auto schema = CustomOpSchema::get("KPFusedDnnEmbeddingWithHashBucket")
                    .TypeVar("T")
                    .MemRefArg("output", getRank(outputType), "T")
                    .MemRefArg("input", getRank(dynamicInputType), "")
                    .MemRefArg("embedding_weight", getRank(weightType), "T")
                    .I64AttrArg("num_buckets", numBuckets);
  auto metadata = schema.toMetadata(rewriter.getContext());
  auto customCall = rewriter.create<CustomizeOp>(
      func.getLoc(), TypeRange{outputType},
      ValueRange{output, dynamicInput, weight},
      rewriter.getStringAttr("KPFusedDnnEmbeddingWithHashBucket"), metadata);
  (void)customCall;
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

  SmallVector<Type> inputTypes = {
      matmulOp.getLhs().getType(), matmulOp.getRhs().getType(),
      matmulOp.getC().getType()};
  auto funcType = rewriter.getFunctionType(inputTypes, TypeRange{});
  auto func = rewriter.create<func::FuncOp>(module.getLoc(), kernelName, funcType);
  func.setPrivate();
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(rewriter.getContext()));
  func->setAttr("fusion.pattern", rewriter.getStringAttr("matmul"));
  func->setAttr("annc.kernel", rewriter.getUnitAttr());

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  Value lhs = entry->getArgument(0);
  Value rhs = entry->getArgument(1);
  Value c = entry->getArgument(2);

  // Keep the public fused kernel symbol stable while delegating the actual
  // MatMul body to the registered builtin kernel during lowering.
  auto schema = CustomOpSchema::get("MatMul")
      .TypeVar("T")
      .MemRefArg("lhs", 2, "T")
      .MemRefArg("rhs", 2, "T")
      .MemRefArg("output", 2, "T");
  SmallVector<NamedAttribute> matmulAttrs;
  matmulAttrs.push_back(
      rewriter.getNamedAttr("opType", rewriter.getStringAttr("MatMul")));
  matmulAttrs.push_back(
      rewriter.getNamedAttr("withBias", rewriter.getBoolAttr(false)));
  matmulAttrs.push_back(rewriter.getNamedAttr(
        "metadata", schema.toMetadata(rewriter.getContext())));
  rewriter.create<MatMulOp>(
      func.getLoc(), TypeRange{c.getType()}, ValueRange{c, lhs, rhs}, matmulAttrs);
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

  func.walk([&](Operation *op) {
    if (hasTfNamePrefix(op, match.prefix) && !isa<ConstantOp, VariableOp>(op)) {
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

    std::string kernelBase = getStableFusionKernelName(
        match.prefix, "dnn_embedding_hash_bucket");
    std::string kernelName = uniquifySymbolName(module, kernelBase);
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName(match.prefix + "/rec_embed_kp_dnn_bucket"));

    auto kernelFunc = createDnnEmbeddingHashBucketKernelFunc(
        module, rewriter, kernelName, match.dynamicInput.getType(),
        match.embeddingWeight.getType(), match.outputBuffer.getType(),
        match.hashBucket.getNumBuckets());

    SmallVector<std::string> fusedNodeNames;
    for (Operation *op : match.candidateOps) {
      std::string name = getTfName(op);
      if (!name.empty()) fusedNodeNames.push_back(name);
    }

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("dnn_embedding_hash_bucket")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.nodes", makeStringArray(rewriter.getContext(), fusedNodeNames)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.inputs", makeStringArray(rewriter.getContext(),
                                     {getValueName(match.embeddingWeight),
                                      getValueName(match.dynamicInput)})));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.input_shapes",
        makeStringArray(rewriter.getContext(),
                        {shapeToString(match.embeddingWeight.getType()),
                         shapeToString(match.dynamicInput.getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.output_shape",
        rewriter.getStringAttr(shapeToString(match.outputBuffer.getType()))));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("mlir_ciface")));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.output", rewriter.getStringAttr(getTfName(match.finalReshape))));
    metadata.push_back(rewriter.getNamedAttr(
        "Nconstants", rewriter.getI64IntegerAttr(0)));
    metadata.push_back(rewriter.getNamedAttr(
        "Nfixed", rewriter.getI64IntegerAttr(1)));
    metadata.push_back(rewriter.getNamedAttr(
        "Ndynamic", rewriter.getI64IntegerAttr(1)));
    metadata.push_back(rewriter.getNamedAttr(
        "num_outputs", rewriter.getI64IntegerAttr(1)));
    metadata.push_back(rewriter.getNamedAttr(
        "output_ranks",
        makeI64Array(rewriter.getContext(),
                     {getRank(match.outputBuffer.getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "input_ranks",
        makeI64Array(rewriter.getContext(),
                     {getRank(match.embeddingWeight.getType()),
                      getRank(match.dynamicInput.getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "output_shapes",
        makeStringArray(rewriter.getContext(),
                        {shapeToString(match.outputBuffer.getType())})));
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
        ValueRange{match.dynamicInput, match.embeddingWeight, match.outputBuffer});

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
    if (matmulOp.getWithBias() || matmulOp.getDoRelu()) return failure();

    std::string matmulName = getTfName(matmulOp);
    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    if (!module) return failure();
    std::string kernelName =
        uniquifySymbolName(module, getStableMatMulKernelName(matmulOp));
    std::string clusterName = uniquifyFusionName(
        module, sanitizeName("annc_fused_" + matmulName));

    auto kernelFunc = createKernelFunc(module, rewriter, kernelName, matmulOp);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("matmul")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.nodes", makeStringArray(rewriter.getContext(), {matmulName})));

    bool lhsIsDynamic =
        hasMatchingBatchDim(matmulOp.getLhs().getType(), matmulOp.getResult().getType());
    bool rhsIsDynamic =
        hasMatchingBatchDim(matmulOp.getRhs().getType(), matmulOp.getResult().getType());
    if (lhsIsDynamic == rhsIsDynamic) {
      lhsIsDynamic = true;
      rhsIsDynamic = false;
    }
    Value fixedInput = lhsIsDynamic ? matmulOp.getRhs() : matmulOp.getLhs();
    Value dynamicInput = lhsIsDynamic ? matmulOp.getLhs() : matmulOp.getRhs();
    SmallVector<int64_t> kernelArgOrder =
        lhsIsDynamic ? SmallVector<int64_t>{1, 0, 2}
                     : SmallVector<int64_t>{0, 1, 2};

    metadata.push_back(rewriter.getNamedAttr(
        "tf.inputs", makeStringArray(rewriter.getContext(),
                                     {getValueName(fixedInput),
                                      getValueName(dynamicInput)})));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.input_shapes",
        makeStringArray(rewriter.getContext(),
                        {shapeToString(fixedInput.getType()),
                         shapeToString(dynamicInput.getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.output_shape",
        rewriter.getStringAttr(shapeToString(matmulOp.getResult().getType()))));
    metadata.push_back(rewriter.getNamedAttr(
        "abi", rewriter.getStringAttr("mlir_ciface")));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.output", rewriter.getStringAttr(matmulName)));
    metadata.push_back(rewriter.getNamedAttr(
        "Nconstants", rewriter.getI64IntegerAttr(0)));
    metadata.push_back(rewriter.getNamedAttr(
        "Nfixed", rewriter.getI64IntegerAttr(1)));
    metadata.push_back(rewriter.getNamedAttr(
        "Ndynamic", rewriter.getI64IntegerAttr(1)));
    metadata.push_back(rewriter.getNamedAttr(
        "num_outputs", rewriter.getI64IntegerAttr(1)));
    metadata.push_back(rewriter.getNamedAttr(
        "output_ranks",
        makeI64Array(rewriter.getContext(), {getRank(matmulOp.getResult().getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "input_ranks",
        makeI64Array(rewriter.getContext(),
                     {getRank(fixedInput.getType()),
                      getRank(dynamicInput.getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "output_shapes",
        makeStringArray(rewriter.getContext(),
                        {shapeToString(matmulOp.getResult().getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), kernelArgOrder)));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr(
        "symbolic_signature", rewriter.getStringAttr("F:0|S:1;?,?")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));
    kernelFunc->setAttr("fusion.metadata",
        DictionaryAttr::get(rewriter.getContext(), metadata));

    rewriter.create<func::CallOp>(matmulOp.getLoc(), kernelFunc,
        ValueRange{matmulOp.getLhs(), matmulOp.getRhs(), matmulOp.getC()});

    rewriter.replaceOp(matmulOp, matmulOp.getC());
    return success();
  }
};

}  // namespace

namespace atir {
class AtirOpFusionPass : public AtirOpFusionBase<AtirOpFusionPass> {
public:
  AtirOpFusionPass() = default;

  void runOnOperation() override {
    auto module = getOperation();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    if (!mainFunc) return;

    RewritePatternSet patterns(&getContext());
    patterns.add<FuseDnnEmbeddingHashBucketAsFuncCallPattern>(&getContext());
    // patterns.add<FuseMatMulAsFuncCallPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(mainFunc, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionPass() {
  return std::make_unique<AtirOpFusionPass>();
}
}  // namespace atir
