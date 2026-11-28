#include <exception>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "Dialect/Atir/Passes/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nlohmann/json.hpp"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

using namespace llvm;
using namespace mlir;

namespace atir {
namespace {

struct CanonicalizeStaticDynamicBroadcast
    : public OpRewritePattern<stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getOperand().getType().hasStaticShape() ||
        !op.getType().hasStaticShape() ||
        failed(hlo::matchInts(op.getOutputDimensions())))
      return failure();
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), op.getOperand(), op.getBroadcastDimensionsAttr());
    return success();
  }
};

LogicalResult canonicalizeStaticDynamicBroadcasts(func::FuncOp function) {
  RewritePatternSet patterns(function.getContext());
  patterns.add<CanonicalizeStaticDynamicBroadcast>(function.getContext());
  return applyPatternsGreedily(
      function, FrozenRewritePatternSet(std::move(patterns)));
}

struct RuntimeArgumentShape {
  int64_t index;
  std::vector<int64_t> shape;
};

FailureOr<std::vector<RuntimeArgumentShape>> readShapeSpec(
    ModuleOp module, StringRef path, std::string &kernelName) {
  std::ifstream input(path.str());
  if (!input.is_open()) {
    module.emitError() << "cannot open runtime shape spec '" << path << "'";
    return failure();
  }

  nlohmann::json json;
  try {
    input >> json;
    kernelName = json.at("kernel_name").get<std::string>();
    std::vector<RuntimeArgumentShape> arguments;
    for (const auto &argument : json.at("arguments")) {
      int64_t index = argument.at("index").get<int64_t>();
      if (index < 0) {
        module.emitError("runtime shape argument index must be non-negative");
        return failure();
      }
      arguments.push_back(RuntimeArgumentShape{
          index, argument.at("shape").get<std::vector<int64_t>>()});
    }
    return arguments;
  } catch (const std::exception &error) {
    module.emitError() << "invalid runtime shape spec '" << path
                       << "': " << error.what();
    return failure();
  }
}

class AnncSpecializeShapesPass
    : public AnncSpecializeShapesBase<AnncSpecializeShapesPass> {
 public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    registry.insert<stablehlo::StablehloDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (shapeSpec.empty()) {
      module.emitError("annc-specialize-shapes requires shape-spec");
      signalPassFailure();
      return;
    }

    std::string kernelName;
    FailureOr<std::vector<RuntimeArgumentShape>> runtimeShapes =
        readShapeSpec(module, shapeSpec, kernelName);
    if (failed(runtimeShapes)) {
      signalPassFailure();
      return;
    }

    func::FuncOp function = module.lookupSymbol<func::FuncOp>(kernelName);
    if (!function || !function->hasAttr("annc.kernel")) {
      module.emitError() << "shape spec references unknown StableHLO kernel '"
                         << kernelName << "'";
      signalPassFailure();
      return;
    }
    if (runtimeShapes->size() != function.getNumArguments()) {
      function.emitError() << "runtime shape spec has " << runtimeShapes->size()
                           << " arguments, expected "
                           << function.getNumArguments();
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    if (functions.size() != 1 || functions.front() != function) {
      function.emitError(
          "StableHLO shape refinement requires the selected kernel to be the "
          "only function in the module");
      signalPassFailure();
      return;
    }

    SmallVector<Type> refinedTypes(function.getArgumentTypes().begin(),
                                   function.getArgumentTypes().end());
    std::vector<bool> seen(function.getNumArguments(), false);
    for (const RuntimeArgumentShape &runtimeShape : *runtimeShapes) {
      if (runtimeShape.index >=
              static_cast<int64_t>(function.getNumArguments()) ||
          seen[runtimeShape.index]) {
        function.emitError() << "invalid or duplicate runtime argument index "
                             << runtimeShape.index;
        signalPassFailure();
        return;
      }
      seen[runtimeShape.index] = true;
      unsigned argumentIndex = static_cast<unsigned>(runtimeShape.index);

      auto tensorType = dyn_cast<RankedTensorType>(
          function.getArgument(argumentIndex).getType());
      if (!tensorType) {
        function.emitError() << "argument " << runtimeShape.index
                             << " is not a ranked builtin tensor";
        signalPassFailure();
        return;
      }
      if (tensorType.getRank() !=
          static_cast<int64_t>(runtimeShape.shape.size())) {
        function.emitError()
            << "argument " << runtimeShape.index << " rank mismatch";
        signalPassFailure();
        return;
      }
      for (auto [declared, actual] :
           llvm::zip_equal(tensorType.getShape(), runtimeShape.shape)) {
        if (actual < 0 ||
            (!ShapedType::isDynamic(declared) && declared != actual)) {
          function.emitError() << "argument " << runtimeShape.index
                               << " has incompatible runtime shape";
          signalPassFailure();
          return;
        }
      }

      refinedTypes[argumentIndex] =
          RankedTensorType::get(runtimeShape.shape, tensorType.getElementType(),
                                tensorType.getEncoding());
    }

    PassManager pm(module.getContext());
    pm.addPass(stablehlo::createStablehloRefineArgumentsPass(refinedTypes));
    pm.addPass(stablehlo::createStablehloRefineShapesPass());
    if (failed(pm.run(module))) {
      signalPassFailure();
      return;
    }
    if (failed(canonicalizeStaticDynamicBroadcasts(function))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAnncSpecializeShapesPass() {
  return std::make_unique<AnncSpecializeShapesPass>();
}

}  // namespace atir
