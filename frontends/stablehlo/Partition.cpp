#include "HloGraph.h"
#include "Patterns/BMMAdd.h"
#include "Patterns/LayerNorm.h"
#include "Patterns/MaskedMul.h"
#include "Patterns/Tensordot.h"
#include "Support/FusionSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
namespace {
struct PartitionPass : PassWrapper<PartitionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionPass)
  StringRef getArgument() const final { return "annc-stablehlo-partition"; }
  StringRef getDescription() const final {
    return "Match StableHLO SSA and publish directly fused custom_call kernels";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect, func::FuncDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();
    // A repeated invocation must not duplicate kernels or change references.
    for (auto func : module.getOps<func::FuncOp>())
      if (func->hasAttr("annc.hlo_region")) {
        module.emitError("StableHLO partition has already run");
        signalPassFailure();
        return;
      }
    annc::stablehlo::HloGraph graph(module);
    SmallVector<Attribute> rejections;
    if (annc::fusionPatternEnabled("batched_matmul_add"))
      annc::stablehlo::partitionBMMAdd(graph, rejections);
    if (annc::fusionPatternEnabled("layer_norm"))
      annc::stablehlo::partitionLayerNorm(graph, rejections);
    if (annc::fusionPatternEnabled("masked_mul"))
      annc::stablehlo::partitionMaskedMul(graph, rejections);
    if (annc::fusionPatternEnabled("tensordot_matmul"))
      annc::stablehlo::partitionTensordot(graph, rejections);
    module->setAttr("annc.partition_rejections",
                    Builder(&getContext()).getArrayAttr(rejections));
  }
};

struct ExtractPass : PassWrapper<ExtractPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractPass)
  StringRef getArgument() const final { return "annc-stablehlo-kernels"; }
  StringRef getDescription() const final {
    return "Extract ONNX-bound kernels as schema-checked standard calls for "
           "the context backend";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();
    module->removeAttr("annc.onnx_model");
    module->removeAttr("annc.provenance");
    module->removeAttr("annc.partition_rejections");
    module->removeAttr("annc.onnx_binding_rejections");
    for (auto &op :
         llvm::make_early_inc_range(module.getBody()->getOperations()))
      if (!isa<func::FuncOp>(op) || !op.hasAttr("annc.kernel") ||
          !op.hasAttr("annc.onnx_bound"))
        op.erase();
    // StableHLO and the public backend may use different MLIR builds. Cross
    // that process boundary using registered standard IR, never an unknown op.
    SmallVector<func::FuncOp> kernels(module.getOps<func::FuncOp>());
    OpBuilder b(module.getContext());
    for (auto kernel : kernels) {
      kernel->removeAttr("annc.hlo_region");
      kernel->removeAttr("annc.onnx_bound");
      bool hasFallback = false;
      kernel.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "atir.opaque") hasFallback = true;
      });
      if (hasFallback) {
        kernel.emitError(
            "framework fallback must remain outside compiled kernels");
        signalPassFailure();
        return;
      }
      auto calls = kernel.getOps<stablehlo::CustomCallOp>();
      if (!llvm::hasSingleElement(kernel.getBody()) ||
          !llvm::hasSingleElement(calls)) {
        kernel.emitError("fused kernel requires exactly one custom_call");
        signalPassFailure();
        return;
      }
      auto call = *calls.begin();
      auto schema = call->getAttrOfType<DictionaryAttr>("annc.kernel_schema");
      auto config = call.getBackendConfig();
      auto configString = config ? dyn_cast<StringAttr>(*config) : StringAttr{};
      if (!schema || call.getHasSideEffect() ||
          (config && (!configString || !configString.empty())) ||
          call.getApiVersion() !=
              stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL ||
          !call.getCalledComputations().empty() ||
          !call.getOutputOperandAliases().empty() || call.getOperandLayouts() ||
          call.getResultLayouts()) {
        call.emitError("unsupported fused custom_call contract");
        signalPassFailure();
        return;
      }
      SmallVector<stablehlo::ConstantOp> constants;
      for (auto value : kernel.getOps<stablehlo::ConstantOp>())
        constants.push_back(value);
      for (auto value : constants) {
        b.setInsertionPoint(value);
        auto replacement =
            b.create<arith::ConstantOp>(value.getLoc(), value.getValue());
        if (value->hasAttr("annc.effective_weight"))
          replacement->setAttr("annc.effective_weight", b.getUnitAttr());
        value->replaceAllUsesWith(ValueRange{replacement.getResult()});
        value.erase();
      }
      auto symbol = "__annc_builtin_" + kernel.getSymName().str();
      if (module.lookupSymbol(symbol)) {
        kernel.emitError("builtin declaration symbol already exists");
        signalPassFailure();
        return;
      }
      b.setInsertionPointToEnd(module.getBody());
      auto declaration = b.create<func::FuncOp>(
          call.getLoc(), symbol,
          b.getFunctionType(call->getOperandTypes(), call->getResultTypes()));
      declaration.setPrivate();
      declaration->setAttr("annc.builtin", call.getCallTargetNameAttr());
      declaration->setAttr("metadata", schema);
      b.setInsertionPoint(call);
      auto bridge = b.create<func::CallOp>(call.getLoc(), declaration,
                                           call->getOperands());
      bridge->setAttr("annc.builtin", call.getCallTargetNameAttr());
      bridge->setAttr("metadata", schema);
      call->replaceAllUsesWith(bridge.getResults());
      call.erase();
    }
  }
};
}  // namespace

void registerStablehloPartitionPasses() {
  PassRegistration<PartitionPass>();
  PassRegistration<ExtractPass>();
}
