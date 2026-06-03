#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/CustomOpSchema.h"
#include "Dialect/Atir/Passes/Passes.h"

#include "Helper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
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
  return annc::getLocName(op);
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

static std::string shapeToString(Type type) {
  auto tensorType = dyn_cast<atir::TensorType>(type);
  if (!tensorType) return "";

  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::interleaveComma(tensorType.getShape(), os,
                        [&](int64_t dim) { os << dim; });
  return os.str();
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
  matmulAttrs.push_back(rewriter.getNamedAttr(
      "metadata", schema.toMetadata(rewriter.getContext())));
  rewriter.create<CustomizeOp>(
      func.getLoc(), TypeRange{}, ValueRange{lhs, rhs, c}, matmulAttrs);
  rewriter.create<func::ReturnOp>(func.getLoc());

  return func;
}

struct FuseMatMulAsCustomizePattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (matmulOp.getWithBias() || matmulOp.getDoRelu()) return failure();

    // Keep TensorFlow input order fixed-first, dynamic-second so ANNCFused can
    // infer the output shape from input_x while using weight's last dimension.
    SmallVector<Value> fusedInputs = {
        matmulOp.getRhs(), matmulOp.getLhs()};
    SmallVector<Type> fusedResultTypes = {matmulOp.getResult().getType()};

    std::string matmulName = getTfName(matmulOp);
    std::string kernelName = getStableMatMulKernelName(matmulOp);
    std::string clusterName = sanitizeName("annc_fused_" + matmulName);
    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    if (!module) return failure();
    createKernelFunc(module, rewriter, kernelName, matmulOp);

    SmallVector<NamedAttribute> metadata;
    metadata.push_back(rewriter.getNamedAttr(
        "fusion.pattern", rewriter.getStringAttr("matmul")));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_name", rewriter.getStringAttr(kernelName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.name", rewriter.getStringAttr(clusterName)));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.nodes", makeStringArray(rewriter.getContext(), {matmulName})));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.inputs", makeStringArray(rewriter.getContext(),
                                     {getValueName(matmulOp.getRhs()),
                                      getValueName(matmulOp.getLhs())})));
    metadata.push_back(rewriter.getNamedAttr(
        "tf.input_shapes",
        makeStringArray(rewriter.getContext(),
                        {shapeToString(matmulOp.getRhs().getType()),
                         shapeToString(matmulOp.getLhs().getType())})));
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
                     {getRank(matmulOp.getRhs().getType()),
                      getRank(matmulOp.getLhs().getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "output_shapes",
        makeStringArray(rewriter.getContext(),
                        {shapeToString(matmulOp.getResult().getType())})));
    metadata.push_back(rewriter.getNamedAttr(
        "kernel_arg_order", makeI64Array(rewriter.getContext(), {1, 0, 2})));
    metadata.push_back(rewriter.getNamedAttr(
        "dynamic_dims", makeI64Array(rewriter.getContext(), {0})));
    metadata.push_back(rewriter.getNamedAttr(
        "symbolic_signature", rewriter.getStringAttr("F:0|S:1;?,?")));
    metadata.push_back(rewriter.getNamedAttr(
        "fallback_function", rewriter.getStringAttr("original_subgraph")));

    SmallVector<NamedAttribute> customAttrs;
    customAttrs.push_back(
        rewriter.getNamedAttr("opType", rewriter.getStringAttr("ANNCFused")));
    customAttrs.push_back(rewriter.getNamedAttr(
        "metadata", DictionaryAttr::get(rewriter.getContext(), metadata)));
    auto fused = rewriter.create<CustomizeOp>(
        matmulOp.getLoc(), fusedResultTypes, fusedInputs, customAttrs);

    rewriter.replaceOp(matmulOp, fused.getResults());
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
    patterns.add<FuseMatMulAsCustomizePattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(mainFunc, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionPass() {
  return std::make_unique<AtirOpFusionPass>();
}
}  // namespace atir
