// residue-sink pass tests: mapped residues (F1 custom_call and F2
// annc.residue carriers) sink to atir_v2 ops, unknown residues fall back to
// atir.unsupported, and stablehlo scaffolding stays in place.

#include "Conversion/StablehloToAtir/StablehloSink.h"

#include <gtest/gtest.h>

#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace {

class ResidueSinkPassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry_.insert<atir::AtirDialect, mlir::func::FuncDialect,
                     mlir::arith::ArithDialect, stablehlo::StablehloDialect>();
    annc::tfcompat::registerTfTypeCompatDialect(registry_);
    ctx_ = std::make_unique<mlir::MLIRContext>(registry_);
    ctx_->loadAllAvailableDialects();
    ctx_->allowUnregisteredDialects(true);
  }

  // Runs the aggregate sink pass over `mlirText`; on success walks the first
  // func op and returns {true, module}.
  std::pair<bool, mlir::OwningOpRef<mlir::ModuleOp>> runSink(
      const std::string &mlirText) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, ctx_.get());
    if (!module) return {false, nullptr};
    mlir::PassManager pm(ctx_.get());
    pm.addPass(annc::createStablehloSink());
    if (failed(pm.run(*module))) return {false, nullptr};
    return {true, std::move(module)};
  }

  static mlir::Operation *findOp(mlir::ModuleOp module,
                                 llvm::StringRef opName) {
    mlir::Operation *found = nullptr;
    module->walk([&](mlir::Operation *op) {
      if (!found && op->getName().getStringRef() == opName) found = op;
    });
    return found;
  }

  mlir::DialectRegistry registry_;
  std::unique_ptr<mlir::MLIRContext> ctx_;
};

constexpr const char *kModuleBody = R"mlir(
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %kept = stablehlo.add %arg0, %arg0 : tensor<2xf32>
    %res = "annc.residue"() {call_target_name = "Variable"} : () -> tensor<2xf32>
    return %res : tensor<2xf32>
  }
)mlir";

TEST_F(ResidueSinkPassTest, SinksMappedResidueCarrier) {
  auto [ok, module] = runSink(kModuleBody);
  ASSERT_TRUE(ok);
  mlir::Operation *fused = findOp(*module, "atir.variable_v2");
  ASSERT_NE(fused, nullptr);
  EXPECT_EQ(fused->getOperands().size(), 0u);
  // The carrier itself is gone; its output feeds the return.
  EXPECT_EQ(findOp(*module, "annc.residue"), nullptr);
  EXPECT_NE(findOp(*module, "stablehlo.add"), nullptr);
}

TEST_F(ResidueSinkPassTest, SinksCustomCallForm) {
  const char *text = R"mlir(
    func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
      %res = stablehlo.custom_call @Variable() : () -> tensor<2xf32>
      return %res : tensor<2xf32>
    }
  )mlir";
  auto [ok, module] = runSink(text);
  ASSERT_TRUE(ok);
  EXPECT_NE(findOp(*module, "atir.variable_v2"), nullptr);
  EXPECT_EQ(findOp(*module, "stablehlo.custom_call"), nullptr);
}

TEST_F(ResidueSinkPassTest, FallsBackUnknownResidue) {
  const char *text = R"mlir(
    func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
      %res = "annc.residue"(%arg0) {call_target_name = "tf.NotInMap"} : (tensor<2xf32>) -> tensor<2xf32>
      return %res : tensor<2xf32>
    }
  )mlir";
  auto [ok, module] = runSink(text);
  ASSERT_TRUE(ok);
  mlir::Operation *fallback = findOp(*module, "atir.unsupported");
  ASSERT_NE(fallback, nullptr);
  auto reason = fallback->getAttrOfType<mlir::StringAttr>("reason");
  ASSERT_NE(reason, nullptr);
  EXPECT_EQ(reason.getValue(), "unknown_op");
}

TEST_F(ResidueSinkPassTest, KeepsStablehloScaffolding) {
  auto [ok, module] = runSink(kModuleBody);
  ASSERT_TRUE(ok);
  mlir::Operation *add = findOp(*module, "stablehlo.add");
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(add->getOperands().size(), 2u);
}

TEST_F(ResidueSinkPassTest, BridgesTfStringTypes) {
  const char *text = R"mlir(
    func.func @main(%arg0: tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string> {
      %res = "annc.residue"(%arg0) {call_target_name = "tf.NotInMap"} : (tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string>
      return %res : tensor<2x!tf_type.string>
    }
  )mlir";
  auto [ok, module] = runSink(text);
  ASSERT_TRUE(ok);
  // The func signature was converted: !tf_type.string -> !atir.string.
  auto funcOp = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(funcOp, nullptr);
  auto argTy = mlir::cast<mlir::TensorType>(funcOp.getArgument(0).getType());
  EXPECT_TRUE(mlir::isa<atir::StringType>(argTy.getElementType()));
  // The fallback carries the materialized operand through the bridge.
  mlir::Operation *fb = findOp(*module, "atir.unsupported");
  ASSERT_NE(fb, nullptr);
  ASSERT_EQ(fb->getOperands().size(), 1u);
  auto opTy = mlir::cast<mlir::TensorType>(fb->getOperand(0).getType());
  EXPECT_TRUE(mlir::isa<atir::StringType>(opTy.getElementType()));
}

TEST_F(ResidueSinkPassTest, MapsUnrankedResidueThrough) {
  // Deliberate deviation from upstream 1030: unranked payloads map straight
  // to atir_v2 instead of falling back as unrank_blocked.
  const char *text = R"mlir(
    func.func @main() -> tensor<*xf32> {
      %res = "annc.residue"() {call_target_name = "Variable"} : () -> tensor<*xf32>
      return %res : tensor<*xf32>
    }
  )mlir";
  auto [ok, module] = runSink(text);
  ASSERT_TRUE(ok);
  EXPECT_EQ(findOp(*module, "atir.unsupported"), nullptr);
  mlir::Operation *mapped = findOp(*module, "atir.variable_v2");
  ASSERT_NE(mapped, nullptr);
  auto ty = mlir::dyn_cast<mlir::ShapedType>(mapped->getResult(0).getType());
  ASSERT_NE(ty, nullptr);
  EXPECT_FALSE(ty.hasRank());
}

TEST_F(ResidueSinkPassTest, FallsBackSpecMismatch) {
  const char *text = R"mlir(
    func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
      %res = "annc.residue"(%arg0) {call_target_name = "Variable"} : (tensor<2xf32>) -> tensor<2xf32>
      return %res : tensor<2xf32>
    }
  )mlir";
  auto [ok, module] = runSink(text);
  ASSERT_TRUE(ok);
  mlir::Operation *fb = findOp(*module, "atir.unsupported");
  ASSERT_NE(fb, nullptr);
  auto reason = fb->getAttrOfType<mlir::StringAttr>("reason");
  ASSERT_NE(reason, nullptr);
  EXPECT_EQ(reason.getValue(), "spec_mismatch");
}

}  // namespace
