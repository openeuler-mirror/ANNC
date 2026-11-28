// BatchNormFold.cpp — fold inference BatchNorm primitive chains into
// MatMul weights and bias.
//
// TF SavedModel exports often expand FusedBatchNorm into a primitive chain:
//
//   y = (Wx + b) * scale / sqrt(var + eps)
//       + (offset - mean * scale / sqrt(var + eps))
//
// which lowers to atir as:
//
//   %biasadd = atir.add %buf, %matmul_res, %bias_val        // TF BiasAdd
//   %mul1    = atir.mul %buf, %rsqrt_res, %scale_val        // scale * rsqrt
//   %rsqrt   = atir.rsqrt %buf, %add3_res                   // 1/sqrt(var+eps)
//   %add3    = atir.add %buf, %variance_val, %epsilon_val   // var + eps
//   %mul2    = atir.mul %buf, %mean_val, %mul1              // mean * (scale*rsqrt)
//   %sub     = atir.sub %buf, %offset_val, %mul2_res
//   %mulA    = atir.mul %buf, %biasadd_res, %mul1           // xhat * scale_factor
//   %y       = atir.add %buf, %mulA_res, %sub_res
//
// At inference time the chain only rescales/shifts the weights, so it folds
// into fused_weight / fused_bias constants (ported from the 812 TF-graph
// rewrite KPFusedMatMulBiasAddBNRewriter):
//
//   scale_factor[i] = scale[i] / sqrt(var[i] + eps)
//   W'[j,i]         = W[j,i] * scale_factor[i]
//   b'[i]           = (bias[i] - mean[i]) * scale_factor[i] + offset[i]
//
// The whole BN chain then becomes dead code and the graph degrades to a
// plain MatMul + BiasAdd.

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>
#include <vector>

using namespace mlir;

namespace atir {
namespace {

// Extract the DenseElementsAttr carried by an atir.constant result. Fails when
// the value is not a constant or carries no compile-time data (e.g. an
// inference-only VariableOp), which safely disables the fold.
FailureOr<DenseElementsAttr> getConstData(Value value) {
  auto constant = value.getDefiningOp<atir::ConstantOp>();
  if (!constant) return failure();
  auto tensorTy = dyn_cast<atir::TensorType>(value.getType());
  if (!tensorTy || !tensorTy.getCacheData()) return failure();
  return tensorTy.getCacheData();
}

// Helper to pick the constant data of one operand while remembering which
// operand it came from (mirrors check_unordered_operands in the 812 version:
// port order is not required).
FailureOr<std::pair<DenseElementsAttr, unsigned>> pickConstOperand(
    Value a, Value b, unsigned indexA, unsigned indexB) {
  auto dataA = getConstData(a);
  if (succeeded(dataA)) return std::make_pair(*dataA, indexA);
  auto dataB = getConstData(b);
  if (succeeded(dataB)) return std::make_pair(*dataB, indexB);
  return failure();
}

class FoldMatMulBiasAddBatchNorm final
    : public OpRewritePattern<atir::AddOp> {
 public:
  using OpRewritePattern<atir::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(atir::AddOp anchor,
                                PatternRewriter &rewriter) const override {
    // (1) Anchor: y = Add(Mul, Sub) — the BN head.
    if (anchor.getInputs().size() != 2) return failure();
    atir::MulOp mulA = anchor.getInputs()[0].getDefiningOp<atir::MulOp>();
    atir::SubOp sub = anchor.getInputs()[1].getDefiningOp<atir::SubOp>();
    if (!mulA || !sub) {
      mulA = anchor.getInputs()[1].getDefiningOp<atir::MulOp>();
      sub = anchor.getInputs()[0].getDefiningOp<atir::SubOp>();
      if (!mulA || !sub) return failure();
    }

    // (2) mulA = Mul(biasadd_res, mul1_res).
    auto biasadd = mulA.getX().getDefiningOp<atir::AddOp>();
    Value mul1Res = mulA.getY();
    if (!biasadd) {
      biasadd = mulA.getY().getDefiningOp<atir::AddOp>();
      mul1Res = mulA.getX();
      if (!biasadd) return failure();
    }

    // (3) biasadd = Add(matmul_res, bias_val) — the TF BiasAdd. ATIR MatMul
    // may carry its own fused bias (withBias=true); that shape is not the 812
    // pattern, skip it.
    if (biasadd.getInputs().size() != 2) return failure();
    auto matmul = biasadd.getInputs()[0].getDefiningOp<atir::MatMulOp>();
    Value biasVal = biasadd.getInputs()[1];
    if (!matmul) {
      matmul = biasadd.getInputs()[1].getDefiningOp<atir::MatMulOp>();
      biasVal = biasadd.getInputs()[0];
      if (!matmul) return failure();
    }
    if (matmul.getWithBias() || matmul.getLeftTranspose() ||
        matmul.getRightTranspose())
      return failure();

    // (4) sub = Sub(offset_val, mul2_res).
    Value offsetVal = sub.getX();
    auto mul2 = sub.getY().getDefiningOp<atir::MulOp>();
    if (!mul2) {
      offsetVal = sub.getY();
      mul2 = sub.getX().getDefiningOp<atir::MulOp>();
      if (!mul2) return failure();
    }

    // (5) mul2 = Mul(mean_val, mul1_res). Stricter than the 812 version: the
    // second operand must be the very same mul1 value, otherwise the chain
    // does not share one scale_factor and the fold would be wrong.
    Value meanVal;
    if (mul2.getX() == mul1Res)
      meanVal = mul2.getY();
    else if (mul2.getY() == mul1Res)
      meanVal = mul2.getX();
    else
      return failure();

    // (6) mul1 = Mul(rsqrt_res, scale_val).
    auto mul1 = mul1Res.getDefiningOp<atir::MulOp>();
    if (!mul1) return failure();
    Value rsqrtRes = mul1.getX();
    Value scaleVal = mul1.getY();
    if (!rsqrtRes.getDefiningOp<atir::RsqrtOp>()) {
      rsqrtRes = mul1.getY();
      scaleVal = mul1.getX();
      if (!rsqrtRes.getDefiningOp<atir::RsqrtOp>()) return failure();
    }
    auto rsqrt = cast<atir::RsqrtOp>(rsqrtRes.getDefiningOp());

    // (7) rsqrt = Rsqrt(add3_res); add3 = Add(variance_val, epsilon_val).
    auto add3 = rsqrt.getX().getDefiningOp<atir::AddOp>();
    if (!add3 || add3.getInputs().size() != 2) return failure();
    Value varianceVal = add3.getInputs()[0];
    Value epsilonVal = add3.getInputs()[1];

    // (8) Every weight-side input must be a constant with f32 data. Any
    // mismatch disables the fold (safest degradation).
    auto weightData = getConstData(matmul.getRhs());
    if (failed(weightData) || !weightData->getElementType().isF32())
      return failure();
    auto biasPair = pickConstOperand(
        biasadd.getInputs()[0], biasadd.getInputs()[1], 1, 2);
    if (failed(biasPair)) return failure();
    auto offsetData = getConstData(offsetVal);
    if (failed(offsetData) || !offsetData->getElementType().isF32())
      return failure();
    auto meanData = getConstData(meanVal);
    if (failed(meanData) || !meanData->getElementType().isF32())
      return failure();
    auto scaleData = getConstData(scaleVal);
    if (failed(scaleData) || !scaleData->getElementType().isF32())
      return failure();
    auto varianceData = getConstData(varianceVal);
    if (failed(varianceData) || !varianceData->getElementType().isF32())
      return failure();
    auto epsilonData = getConstData(epsilonVal);
    if (failed(epsilonData) || !epsilonData->getElementType().isF32())
      return failure();

    // (9) Shape guards, ported from the 812 rewrite: weight rows must match
    // the lhs inner dim; unknown lhs shape disables the fold.
    auto lhsTy = dyn_cast<atir::TensorType>(matmul.getLhs().getType());
    if (!lhsTy || lhsTy.getShape().size() != 2) return failure();
    auto weightShape = weightData->getType().getShape();
    if (weightShape.size() != 2 || lhsTy.getShape()[1] != weightShape[0])
      return failure();
    const int64_t rows = weightShape[0];
    const int64_t cols = weightShape[1];
    if (rows <= 0 || cols <= 0) return failure();
    if (epsilonData->getNumElements() == 0) return failure();
    const float epsilon = *epsilonData->value_begin<float>();

    std::vector<float> weightVals(weightData->value_begin<float>(),
                                  weightData->value_end<float>());
    std::vector<float> biasVals(biasPair->first.value_begin<float>(),
                                biasPair->first.value_end<float>());
    std::vector<float> offsetVals(offsetData->value_begin<float>(),
                                  offsetData->value_end<float>());
    std::vector<float> meanVals(meanData->value_begin<float>(),
                                meanData->value_end<float>());
    std::vector<float> scaleVals(scaleData->value_begin<float>(),
                                 scaleData->value_end<float>());
    std::vector<float> varianceVals(varianceData->value_begin<float>(),
                                    varianceData->value_end<float>());
    if (weightVals.size() != static_cast<size_t>(rows * cols) ||
        biasVals.size() != static_cast<size_t>(cols) ||
        offsetVals.size() != static_cast<size_t>(cols) ||
        meanVals.size() != static_cast<size_t>(cols) ||
        scaleVals.size() != static_cast<size_t>(cols) ||
        varianceVals.size() != static_cast<size_t>(cols))
      return failure();

    // (10) The fold itself: per-column scale_factor absorbed into W and b.
    std::vector<float> fusedWeight(rows * cols);
    std::vector<float> fusedBias(cols);
    for (int64_t i = 0; i < cols; ++i) {
      const float scaleFactor =
          scaleVals[i] / std::sqrt(varianceVals[i] + epsilon);
      fusedBias[i] = (biasVals[i] - meanVals[i]) * scaleFactor + offsetVals[i];
      for (int64_t j = 0; j < rows; ++j)
        fusedWeight[j * cols + i] = weightVals[j * cols + i] * scaleFactor;
    }

    // (11) Materialize the two fused constants. The TensorType keeps the
    // original weight descriptor (stride/layout/memType) and swaps in the
    // folded cacheData; the symbol gets a distinct name. Both constants are
    // inserted before the MatMul so they dominate their uses.
    rewriter.setInsertionPoint(matmul);

    auto f32Type = weightData->getElementType();
    auto weightTy = cast<atir::TensorType>(matmul.getRhs().getType());
    auto fusedWeightAttr = DenseElementsAttr::get(
        RankedTensorType::get({rows, cols}, f32Type), ArrayRef<float>(fusedWeight));
    auto fusedWeightTy = weightTy.cloneWithCacheData(fusedWeightAttr);
    std::string fusedWeightName =
        weightTy.getName() ? weightTy.getName().str() + "/bn_fused_weight"
                           : "bn_fused_weight";
    auto fusedWeightOp = rewriter.create<atir::ConstantOp>(
        matmul.getLoc(), fusedWeightTy, rewriter.getStringAttr(fusedWeightName),
        rewriter.getStringAttr("public"));

    auto biasTy = cast<atir::TensorType>(biasVal.getType());
    auto fusedBiasAttr = DenseElementsAttr::get(
        RankedTensorType::get({cols}, f32Type), ArrayRef<float>(fusedBias));
    auto fusedBiasTy = biasTy.cloneWithCacheData(fusedBiasAttr);
    std::string fusedBiasName =
        biasTy.getName() ? biasTy.getName().str() + "/bn_fused_bias"
                         : "bn_fused_bias";
    auto fusedBiasOp = rewriter.create<atir::ConstantOp>(
        biasadd.getLoc(), fusedBiasTy, rewriter.getStringAttr(fusedBiasName),
        rewriter.getStringAttr("public"));

    // (12) Rewire: MatMul.rhs and the BiasAdd bias input are replaced; an
    // Identity boundary bridges the BiasAdd output to the BN head's users.
    // The BiasAdd result keeps its own descriptor (needed by downstream
    // fusion passes); the Identity normalizes the tensor name, and the
    // identity-canonicalize pass later retypes the producer directly.
    matmul.setOperand(2, fusedWeightOp.getResult());
    biasadd.setOperand(biasPair->second, fusedBiasOp.getResult());
    Value anchorBuffer = anchor.getOutput();
    // The bridge consumes the BiasAdd result, so it must live after the
    // BiasAdd (insertion point currently sits before the MatMul).
    rewriter.setInsertionPointAfter(anchor);
    auto bridge = rewriter.create<atir::IdentityOp>(
        anchor.getLoc(), anchor.getResult().getType(), anchorBuffer,
        biasadd.getResult());
    rewriter.replaceOp(anchor, bridge.getResult());

    // (13) Clean the dead BN chain. Collect the buffer operands first —
    // getters on erased operations would be use-after-free. The anchor
    // buffer stays alive: the bridge Identity uses it.
    Value bufMulA = mulA.getOutput();
    Value bufSub = sub.getOutput();
    Value bufMul2 = mul2.getOutput();
    Value bufMul1 = mul1.getOutput();
    Value bufRsqrt = rsqrt.getOutput();
    Value bufAdd3 = add3.getOutput();
    SmallVector<Operation*> deadChain = {mulA, sub, mul2, mul1, rsqrt, add3};
    for (Operation* op : deadChain) {
      if (op->use_empty()) rewriter.eraseOp(op);
    }
    auto eraseDeadBuffer = [&](Value bufferLike) {
      auto buffer = bufferLike.getDefiningOp<atir::BufferOp>();
      if (buffer && buffer->use_empty()) rewriter.eraseOp(buffer);
    };
    eraseDeadBuffer(bufMulA);
    eraseDeadBuffer(bufSub);
    eraseDeadBuffer(bufMul2);
    eraseDeadBuffer(bufMul1);
    eraseDeadBuffer(bufRsqrt);
    eraseDeadBuffer(bufAdd3);
    return success();
  }
};

class AtirFoldBatchNormPass final
    : public AtirFoldBatchNormBase<AtirFoldBatchNormPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldMatMulBiasAddBatchNorm>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAtirFoldBatchNormPass() {
  return std::make_unique<AtirFoldBatchNormPass>();
}

}  // namespace atir
