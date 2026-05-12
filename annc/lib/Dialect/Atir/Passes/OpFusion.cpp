#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/SmallString.h"

using namespace llvm;
using namespace mlir;
using namespace atir;

namespace {

std::string getFusedFuncName(Type aType, Type bType, Type biasType, Type resultType) {
  std::string sig;
  llvm::raw_string_ostream os(sig);
  os << "fused_matmul_add_relu_";
  aType.print(os); os << "_";
  bType.print(os); os << "_";
  biasType.print(os); os << "_";
  resultType.print(os);

  SHA256 sha;
  sha.update(sig);
  auto hash = sha.final();
  SmallString<64> hex;
  for (size_t i = 0; i < 8; ++i) {
    hex += llvm::toHex(hash[i]);
  }
  return ("fused_matmul_add_relu_" + hex).str();
}

func::FuncOp getOrCreateFusedFunc(ModuleOp module, Type matmulOutputType,
                            Type aType, Type bType, Type cType,Type biasType, Type resultType,
                            PatternRewriter &rewriter) {
  std::string funcName = getFusedFuncName(aType, bType, biasType, resultType);

  if (auto existing = module.lookupSymbol<func::FuncOp>(funcName))
    return existing;

  auto funcType = rewriter.getFunctionType({aType, bType, cType, biasType},  resultType);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(module.getBody());

  auto func = rewriter.create<func::FuncOp>(module.getLoc(), funcName, funcType);
  func.setPrivate();

MLIRContext *ctx = func.getContext();
  auto emitCAttr = UnitAttr::get(ctx);
  func->setAttr("llvm.emit_c_interface", emitCAttr);

  func->setAttr("fusion.pattern", StringAttr::get(rewriter.getContext(), "matmul_add_relu"));

  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);

  Value A = entry->getArgument(0);
  Value B = entry->getArgument(1);
  Value C = entry->getArgument(2);
  Value bias = entry->getArgument(3);
  bool hasBias = false;
  SmallVector<Value> matmulInputs = {A, B, C, };
  auto matmul = rewriter.create<MatMulOp>(
      func.getLoc(), matmulOutputType, A, B, C, Value{},rewriter.getBoolAttr(false),
      rewriter.getBoolAttr(false), //  right_transpose
      rewriter.getBoolAttr(false), // left_transpose
      rewriter.getBoolAttr(false),  // output_transpose
      rewriter.getBoolAttr(false),   // do_relu
      rewriter.getF32FloatAttr(-1.0f),  // relu_limit
      IntegerAttr(),
      IntegerAttr(),
      IntegerAttr(),
      IntegerAttr(),
      IntegerAttr(),
      IntegerAttr());

  SmallVector<Value> addInputs = {matmul.getResult(), bias};
  auto doReluAttr = rewriter.getBoolAttr(false);
  auto reluLimitAttr = rewriter.getF32FloatAttr(-1.0f);
  auto addOutput = rewriter.create<atir::BufferOp>(func.getLoc(), matmulOutputType);
  auto add = rewriter.create<AddOp>(
      func.getLoc(), matmulOutputType, addOutput.getResult(), addInputs, doReluAttr, reluLimitAttr, FloatAttr());

  auto reluOutput = rewriter.create<atir::BufferOp>(func.getLoc(), resultType);
  auto relu = rewriter.create<ReluOp>(
      func.getLoc(), resultType, reluOutput.getResult(), add.getResult(), rewriter.getF32FloatAttr(-1.0f));

  rewriter.create<func::ReturnOp>(func.getLoc(), relu.getResult());

  return func;
}

struct FuseMatMulAddReluAsCallPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp relu,
                                PatternRewriter &rewriter) const override {
    auto addOp = dyn_cast_or_null<AddOp>(relu.getInput().getDefiningOp());
    if (!addOp) return failure();
    auto inputs = addOp.getInputs();
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    MatMulOp matmulOp = nullptr;
    Value bias;


    if (auto mm = dyn_cast_or_null<MatMulOp>(lhs.getDefiningOp())) {
      matmulOp = mm;
      bias = rhs;
    } else if (auto mm = dyn_cast_or_null<MatMulOp>(rhs.getDefiningOp())) {
      matmulOp = mm;
      bias = lhs;
    } else {
      return failure();
    }

    ModuleOp module = relu->getParentOfType<ModuleOp>();
    if (!module) return failure();
    Type aType = matmulOp.getLhs().getType();
    Type bType = matmulOp.getRhs().getType();
    Type cType  = matmulOp.getC().getType();

    Type matmulOutputType = matmulOp.getResult().getType();
    Type biasType = bias.getType();
    Type resultType = relu.getResult().getType();

    func::FuncOp fusedFunc = getOrCreateFusedFunc(module,matmulOutputType, aType, bType,cType, biasType, resultType, rewriter);
    auto call = rewriter.create<func::CallOp>(relu.getLoc(), fusedFunc,
                                        ValueRange{ matmulOp.getLhs(), matmulOp.getRhs(), matmulOp.getC(), bias});

    rewriter.replaceOp(relu, call.getResult(0));
    rewriter.eraseOp(addOp);
    rewriter.eraseOp(matmulOp);
    return success();
  }
};

} // anonymous namespace

namespace atir {
class AtirOpFusionPass : public AtirOpFusionBase<AtirOpFusionPass> {
public:
  AtirOpFusionPass() = default;

  void runOnOperation() override {
    auto module = getOperation();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main");
    if (!mainFunc) return;

    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatMulAddReluAsCallPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(mainFunc, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpFusionPass() {
  return std::make_unique<AtirOpFusionPass>();
}
} // namespace atir