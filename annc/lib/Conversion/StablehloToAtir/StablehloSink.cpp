#include "Conversion/StablehloToAtir/StablehloSink.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Conversion/StablehloToAtir/ResidueProvider.h"
#include "Dialect/Atir/AtirOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// The generated pass bases use unqualified ModuleOp and need it complete:
// include the headers here, after `using namespace mlir`.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#define GEN_PASS_CLASSES
#include "Conversion/StablehloToAtir/Passes.h.inc"

namespace annc {

void printSinkStat(const SinkStat& stat) {
  auto render = [](llvm::StringRef label,
                   const std::map<std::string, int>& counts) {
    llvm::errs() << "[residue-sink] stat " << label << ":";
    if (counts.empty()) {
      llvm::errs() << " {}\n";
      return;
    }
    for (const auto& [key, value] : counts)
      llvm::errs() << " " << key << "=" << value;
    llvm::errs() << "\n";
  };
  render("residue-mapped", stat.residueMapped);
  render("residue-fallback", stat.residueFallback);
  render("residue-fallback-reason", stat.residueFallbackReason);
  render("keep", stat.keep);
}

}  // namespace annc

namespace {

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

// V2 type conversion: builtin tensors stay builtin.
// Only the TF string element type is normalized to
// `!atir.string`, so cast bridges exist solely at string-domain boundaries.
class ResidueTypeConverter : public TypeConverter {
 public:
  explicit ResidueTypeConverter(MLIRContext* ctx) {
    addConversion([ctx](Type type) -> std::optional<Type> {
      auto tensorType = dyn_cast<TensorType>(type);
      if (!tensorType) return type;
      if (!annc::tfcompat::isTfStringType(tensorType.getElementType()))
        return type;
      Type elem = atir::StringType::get(ctx);
      if (auto ranked = dyn_cast<RankedTensorType>(tensorType))
        return RankedTensorType::get(ranked.getShape(), elem);
      return UnrankedTensorType::get(elem);
    });
    auto bridge = [](OpBuilder& builder, Type resultType, ValueRange inputs,
                     Location loc) -> Value {
      if (inputs.size() != 1) return Value();
      return builder
          .create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
          .getOutputs()[0];
    };
    addTargetMaterialization(bridge);
    addSourceMaterialization(bridge);
  }
};

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

// Benefit 3: constants keep their payload (provider-owned).
class ResidueConstantPattern : public ConversionPattern {
 public:
  ResidueConstantPattern(const TypeConverter* converter,
                         const annc::ResidueProvider& provider,
                         annc::SinkStat& stat, MLIRContext* ctx)
      : ConversionPattern(*converter, MatchAnyOpTypeTag(), /*benefit=*/3, ctx),
        provider_(provider),
        stat_(stat) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    std::optional<annc::ResidueView> view = provider_.match(op);
    if (!view) return failure();

    SmallVector<Type> resultTypes;
    resultTypes.reserve(op->getNumResults());
    for (Type ty : op->getResultTypes()) {
      Type converted = getTypeConverter()->convertType(ty);
      if (!converted) return failure();
      resultTypes.push_back(converted);
    }

    SmallVector<Value> results;
    if (failed(provider_.emitConstant(rewriter, op, *view, resultTypes,
                                      results)))
      return failure();
    if (results.size() != resultTypes.size()) return failure();

    rewriter.replaceOp(op, results);
    stat_.residueMapped[view->opType]++;
    return success();
  }

 private:
  const annc::ResidueProvider& provider_;
  annc::SinkStat& stat_;
};

// Benefit 2: residues a frontend provider can map to real atir ops.
class ResidueMappedPattern : public ConversionPattern {
 public:
  ResidueMappedPattern(const TypeConverter* converter,
                       const annc::ResidueProvider& provider,
                       annc::SinkStat& stat, MLIRContext* ctx)
      : ConversionPattern(*converter, MatchAnyOpTypeTag(), /*benefit=*/2, ctx),
        provider_(provider),
        stat_(stat) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    std::optional<annc::ResidueView> view = provider_.match(op);
    if (!view) return failure();

    SmallVector<Type> resultTypes;
    resultTypes.reserve(op->getNumResults());
    for (Type ty : op->getResultTypes()) {
      Type converted = getTypeConverter()->convertType(ty);
      if (!converted) return failure();
      resultTypes.push_back(converted);
    }

    SmallVector<Value> results;
    if (failed(provider_.emitMapped(rewriter, op, *view, operands, resultTypes,
                                    results)))
      return failure();  // driver rolls back; fallback takes over
    if (results.size() != resultTypes.size()) return failure();

    rewriter.replaceOp(op, results);
    stat_.residueMapped[view->opType]++;
    return success();
  }

 private:
  const annc::ResidueProvider& provider_;
  annc::SinkStat& stat_;
};

// Benefit 1: registered fallback carrier for an unmappable residue.
class ResidueFallbackPattern : public ConversionPattern {
 public:
  ResidueFallbackPattern(const TypeConverter* converter,
                         const annc::ResidueProvider& provider,
                         annc::SinkStat& stat, MLIRContext* ctx)
      : ConversionPattern(*converter, MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        provider_(provider),
        stat_(stat),
        ctx_(ctx) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    std::optional<annc::ResidueView> view = provider_.match(op);
    if (!view) return failure();

    SmallVector<Type> resultTypes;
    resultTypes.reserve(op->getNumResults());
    for (Type ty : op->getResultTypes()) {
      Type converted = getTypeConverter()->convertType(ty);
      resultTypes.push_back(converted ? converted : ty);
    }

    rewriter.setInsertionPoint(op);
    OperationState state(op->getLoc(), "atir.unsupported");
    state.operands.append(operands.begin(), operands.end());
    state.types.append(resultTypes.begin(), resultTypes.end());
    state.addAttribute("opType", rewriter.getStringAttr(view->opType));
    if (view->sourceAttrs && !view->sourceAttrs.empty())
      state.addAttribute("metadata", view->sourceAttrs);

    // Disposition reason: keep "unknown op" apart from "mappable but the
    // mapped build rejected it" so runtime dispatch and audits can route
    // each class.
    std::string reason = "unknown_op";
    std::string detail;
    if (provider_.hasMapping(*view, ctx_)) reason = "spec_mismatch";
    state.addAttribute("reason", rewriter.getStringAttr(reason));
    state.addAttribute("detail", rewriter.getStringAttr(detail));
    Operation* fallback = rewriter.create(state);
    rewriter.replaceOp(op, fallback->getResults());
    stat_.residueFallback[view->opType]++;
    stat_.residueFallbackReason[reason]++;
    return success();
  }

 private:
  const annc::ResidueProvider& provider_;
  annc::SinkStat& stat_;
  MLIRContext* ctx_;
};

// Rebuilds the function terminator so its operands match the converted
// signature. Region terminators (stablehlo.return inside while/sort/reduce)
// are left untouched: they must keep the region's own signature.
class ResidueReturnOpConversion : public ConversionPattern {
 public:
  ResidueReturnOpConversion(const TypeConverter* converter, MLIRContext* ctx)
      : ConversionPattern(*converter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          ctx) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    if (!isa<func::ReturnOp>(op)) return failure();
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addAttributes(op->getAttrs());
    Operation* newReturn = rewriter.create(state);
    rewriter.replaceOp(op, newReturn->getResults());
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Shared runner + passes
//===----------------------------------------------------------------------===//

namespace annc {

static LogicalResult runResidueSink(ModuleOp module,
                                    const ResidueProvider& provider) {
  MLIRContext* ctx = module.getContext();
  SinkStat stat;

  ResidueTypeConverter converter(ctx);
  ConversionTarget target(*ctx);
  // Registered dialects stay legal; only the provider's residues convert.
  // - func.FuncOp: legal once its signature is converted (args become
  //   !atir.string), so no function-boundary materialization is needed.
  // - func.ReturnOp: legal only once its operands are converted;
  //   ResidueReturnOpConversion rebuilds it in place. Region terminators are
  //   deliberately not listed: rewriting stablehlo.return would break
  //   while/sort/reduce signatures.
  // - everything else: legal unless the provider claims it as a residue.
  auto returnLegal = [&](Operation* op) {
    return converter.isLegal(op->getOperandTypes());
  };
  target.addDynamicallyLegalOp<func::ReturnOp>(returnLegal);
  target.markUnknownOpDynamicallyLegal([&](Operation* op) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op))
      return converter.isSignatureLegal(funcOp.getFunctionType());
    return !provider.isResidue(op);
  });

  RewritePatternSet patterns(ctx);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  patterns.add<ResidueReturnOpConversion>(&converter, ctx);
  // Benefit 3 (constant) > 2 (mapped) > 1 (fallback): constants keep their
  // payload; a rolled-back mapping attempt lands in the fallback carrier.
  patterns.add<ResidueConstantPattern>(&converter, provider, stat, ctx);
  patterns.add<ResidueMappedPattern>(&converter, provider, stat, ctx);
  patterns.add<ResidueFallbackPattern>(&converter, provider, stat, ctx);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    return failure();
  }

  // Post-walk census: what stayed as-is (the principle's default outcome).
  // Scope: only the kept scaffolding dialects (stablehlo/shape/tensor/
  // tf_executor); func/arith ops are conversion plumbing and uncounted.
  // Unregistered ops claimed by the provider are residues, not "kept" ops.
  module->walk([&](Operation* op) {
    llvm::StringRef name = op->getName().getStringRef();
    llvm::StringRef ns =
        op->getDialect() ? op->getDialect()->getNamespace() : llvm::StringRef();
    bool kept = ns == "stablehlo" || ns == "shape" || ns == "tensor" ||
                ns == "tf_executor";
    if (!kept && ns.empty() && !provider.isResidue(op)) kept = true;
    if (kept) stat.keep[name.str()]++;
  });
  printSinkStat(stat);
  return success();
}

class StablehloSinkPass : public StablehloSinkBase<StablehloSinkPass> {
 public:
  StablehloSinkPass() = default;

  void runOnOperation() override {
    // TF is the integration entry point today; other frontends construct the
    // same driver with their own provider.
    std::unique_ptr<ResidueProvider> provider = createTfResidueProvider();
    if (failed(runResidueSink(getOperation(), *provider)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createStablehloSink() {
  return std::make_unique<StablehloSinkPass>();
}

}  // namespace annc
