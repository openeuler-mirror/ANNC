// Sink component contracts: target resolution, tf_type string detection and
// residue recognition (end-to-end FileCheck ships with the tool batch).

#include "Conversion/StablehloToAtir/ResidueProvider.h"
#include "Conversion/StablehloToAtir/StablehloSink.h"
#include "Conversion/StablehloToAtir/V2OpMatcher.h"
#include "Dialect/Atir/AtirOps.h"

#include <gtest/gtest.h>

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {

class StablehloSinkComponentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry_.insert<atir::AtirDialect, func::FuncDialect, arith::ArithDialect,
                     stablehlo::StablehloDialect>();
    annc::tfcompat::registerTfTypeCompatDialect(registry_);
    ctx_ = std::make_unique<MLIRContext>(registry_);
    ctx_->loadAllAvailableDialects();
    ctx_->allowUnregisteredDialects(true);
  }

  OwningOpRef<ModuleOp> parse(llvm::StringRef ir) {
    return parseSourceString<ModuleOp>(ir, ctx_.get());
  }

  DialectRegistry registry_;
  std::unique_ptr<MLIRContext> ctx_;
};

TEST_F(StablehloSinkComponentTest, ResolveV2TargetContract) {
  // Hit: registered target resolves.
  auto hit = annc::resolveV2Target(ctx_.get(), "SparseToDense",
                                   {{"SparseToDense", "sparse_to_dense_v2"}});
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(hit->getIdentifier().str(), "atir.sparse_to_dense_v2");
  // Miss: no map entry / unregistered mnemonic.
  EXPECT_FALSE(annc::resolveV2Target(ctx_.get(), "FooOp", {}).has_value());
  EXPECT_FALSE(annc::resolveV2Target(ctx_.get(), "X", {{"X", "no_such_op_v2"}})
                   .has_value());
}

TEST_F(StablehloSinkComponentTest, DetectsTfCompatStringType) {
  auto module = parse(R"mlir(
    func.func @f(%s: !tf_type.string, %t: tensor<4xf32>) {
      return
    }
  )mlir");
  ASSERT_TRUE(module);
  auto func = *module->getOps<func::FuncOp>().begin();
  EXPECT_TRUE(annc::tfcompat::isTfStringType(func.getArgument(0).getType()));
  EXPECT_FALSE(annc::tfcompat::isTfStringType(func.getArgument(1).getType()));
}

TEST_F(StablehloSinkComponentTest, ResidueRecognitionContract) {
  auto module = parse(R"mlir(
    func.func @f() -> tensor<4xf32> {
      %0 = "tf.FooOp"() : () -> tensor<4xf32>
      %1 = arith.constant dense<0.0> : tensor<4xf32>
      return %1 : tensor<4xf32>
    }
  )mlir");
  ASSERT_TRUE(module);
  auto provider = annc::createTfResidueProvider();
  Operation *residue = nullptr, *constant = nullptr;
  module->walk([&](Operation* op) {
    if (op->getName().getStringRef() == "tf.FooOp") residue = op;
    if (isa<arith::ConstantOp>(op)) constant = op;
  });
  ASSERT_NE(residue, nullptr);
  ASSERT_NE(constant, nullptr);
  EXPECT_TRUE(provider->isResidue(residue));
  auto view = provider->match(residue);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->opType, "FooOp");
  EXPECT_FALSE(provider->isResidue(constant));
  EXPECT_FALSE(provider->match(constant).has_value());
}

// Every mapped op must resolve to a registered atir.*_v2 target; keys mirror
// kOpMap verbatim (view carriers keep the `annc.` prefix).
TEST_F(StablehloSinkComponentTest, OpMapTargetsAllResolve) {
  auto provider = annc::createTfResidueProvider();
  const char* mapped[] = {
      "annc.getDim",          "annc.setDim",
      "annc.encodeString",    "annc.decodeString",
      "ResourceGather",       "SparseFillEmptyRows",
      "SparseReshape",        "SparseSegmentMean",
      "SparseSegmentSum",     "SparseTensorDenseMatMul",
      "SparseToDense",        "StaticRegexReplace",
      "StringSplit",          "StringToHashBucketFast",
      "StringToNumber",       "TopK",
      "TopKV2",               "Variable",
      "VariableV2",
  };
  for (const char* opType : mapped) {
    SCOPED_TRACE(opType);
    EXPECT_TRUE(provider->hasMapping({opType, {}}, ctx_.get()));
  }
  // VarHandleOp has no target op in the current V2 set (fallback carrier).
  EXPECT_FALSE(provider->hasMapping({"VarHandleOp", {}}, ctx_.get()));
}

TEST_F(StablehloSinkComponentTest, MatchesLegacyResidueForms) {
  {
    auto module = parse(R"mlir(
      func.func @f(%arg0: tensor<2xi64>) -> tensor<2xf32> {
        %0 = stablehlo.custom_call @tf.SparseToDense(%arg0) : (tensor<2xi64>) -> tensor<2xf32>
        return %0 : tensor<2xf32>
      }
    )mlir");
    ASSERT_TRUE(module);
    auto provider = annc::createTfResidueProvider();
    Operation* call = nullptr;
    module->walk([&](Operation* op) {
      if (isa<stablehlo::CustomCallOp>(op)) call = op;
    });
    ASSERT_NE(call, nullptr);
    EXPECT_TRUE(provider->isResidue(call));
    auto view = provider->match(call);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(view->opType, "SparseToDense");
  }
  {
    auto module = parse(R"mlir(
      func.func @f(%arg0: tensor<2xi64>) -> tensor<2xf32> {
        %0 = "custom.Legacy"(%arg0) {call_target_name = "SparseToDense"} : (tensor<2xi64>) -> tensor<2xf32>
        return %0 : tensor<2xf32>
      }
    )mlir");
    ASSERT_TRUE(module);
    auto provider = annc::createTfResidueProvider();
    Operation* declared = nullptr;
    module->walk([&](Operation* op) {
      if (op->getName().getStringRef() == "custom.Legacy") declared = op;
    });
    ASSERT_NE(declared, nullptr);
    EXPECT_TRUE(provider->isResidue(declared));
    auto view = provider->match(declared);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(view->opType, "SparseToDense");
  }
}
}  // namespace
