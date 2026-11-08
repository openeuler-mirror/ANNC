#include "Conversion/StablehloToAtir/ResidueProvider.h"
#include "Conversion/StablehloToAtir/StablehloSink.h"
#include "Conversion/StablehloToAtir/V2OpMatcher.h"
#include "Dialect/Atir/AtirOps.h"

#include <gtest/gtest.h>

#include <memory>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
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

  // Parses `ir` (kept alive in `module_`) and returns its first func.
  func::FuncOp func(llvm::StringRef ir) {
    module_ = parseSourceString<ModuleOp>(ir, ctx_.get());
    return module_ ? *module_->getOps<func::FuncOp>().begin() : func::FuncOp();
  }

  Operation* find(llvm::StringRef name) {
    Operation* found = nullptr;
    module_->walk([&](Operation* op) {
      if (op->getName().getStringRef() == name) found = op;
    });
    return found;
  }

  DialectRegistry registry_;
  std::unique_ptr<MLIRContext> ctx_;
  OwningOpRef<ModuleOp> module_;
};

TEST_F(StablehloSinkComponentTest, ResolveV2TargetContract) {
  auto hit = annc::resolveV2Target(
      ctx_.get(), "SparseToDense", {{"SparseToDense", "sparse_to_dense_v2"}});
  ASSERT_TRUE(hit);
  EXPECT_EQ(hit->getIdentifier().str(), "atir.sparse_to_dense_v2");
  EXPECT_FALSE(annc::resolveV2Target(ctx_.get(), "FooOp", {}).has_value());
  EXPECT_FALSE(annc::resolveV2Target(ctx_.get(), "X", {{"X", "x_v2"}}).has_value());
}

TEST_F(StablehloSinkComponentTest, DetectsTfCompatStringType) {
  auto f = func(R"mlir(func.func @f(%s: !tf_type.string, %t: tensor<4xf32>) { return })mlir");
  ASSERT_TRUE(f);
  EXPECT_TRUE(annc::tfcompat::isTfStringType(f.getArgument(0).getType()));
  EXPECT_FALSE(annc::tfcompat::isTfStringType(f.getArgument(1).getType()));
}

TEST_F(StablehloSinkComponentTest, ResidueFormsContract) {
  auto provider = annc::createTfResidueProvider();
  ASSERT_TRUE(func(
      R"mlir(func.func @f() -> tensor<4xf32> {
        %0 = "tf.FooOp"() : () -> tensor<4xf32>
        %1 = arith.constant dense<0.0> : tensor<4xf32>
        return %1 : tensor<4xf32> } )mlir"));
  EXPECT_FALSE(provider->isResidue(find("tf.FooOp")));
  EXPECT_FALSE(provider->isResidue(find("arith.constant")));
  ASSERT_TRUE(func(R"mlir(func.func @f(%arg0: tensor<2xi64>) -> tensor<2xf32> {
      %0 = stablehlo.custom_call @tf.SparseToDense(%arg0) : (tensor<2xi64>) -> tensor<2xf32>
      return %0 : tensor<2xf32> })mlir"));
  Operation* call = nullptr;
  module_->walk([&](stablehlo::CustomCallOp op) { call = op; });
  ASSERT_NE(call, nullptr);
  ASSERT_TRUE(provider->isResidue(call) && provider->match(call).has_value());
  EXPECT_EQ(provider->match(call)->opType, "SparseToDense");
  ASSERT_TRUE(func(R"mlir(func.func @f(%arg0: tensor<2xi64>) -> tensor<2xf32> {
      %0 = "annc.residue"(%arg0) {call_target_name = "SparseToDense"} : (tensor<2xi64>) -> tensor<2xf32>
      return %0 : tensor<2xf32> })mlir"));
  Operation* declared = find("annc.residue");
  ASSERT_NE(declared, nullptr);
  ASSERT_TRUE(provider->isResidue(declared) &&
              provider->match(declared).has_value());
  EXPECT_EQ(provider->match(declared)->opType, "SparseToDense");
}

// Every mapped op must resolve to a registered atir.*_v2 target; keys mirror
// kOpMap verbatim (view carriers keep the `annc.` prefix).
TEST_F(StablehloSinkComponentTest, OpMapTargetsAllResolve) {
  auto provider = annc::createTfResidueProvider();
  const char* mapped[] = {
      "annc.getDim", "annc.setDim", "annc.encodeString", "annc.decodeString",
      "ResourceGather", "SparseFillEmptyRows", "SparseReshape", "SparseSegmentMean",
      "SparseSegmentSum", "SparseTensorDenseMatMul", "SparseToDense", "TopK",
      "TopKV2", "StaticRegexReplace", "StringSplit", "StringToHashBucketFast",
      "StringToNumber", "Variable", "VariableV2"};
  for (const char* opType : mapped) {
    SCOPED_TRACE(opType);
    EXPECT_TRUE(provider->hasMapping({opType, {}}, ctx_.get()));
  }
  EXPECT_FALSE(provider->hasMapping({"Concat", {}}, ctx_.get()));
  // Resource domain not landed yet: VarHandleOp falls back (unknown_op).
  EXPECT_FALSE(provider->hasMapping({"VarHandleOp", {}}, ctx_.get()));
}

TEST_F(StablehloSinkComponentTest, ParsesParameterizedTfTypeTags) {
  ASSERT_TRUE(func(R"mlir(func.func @f(%r: !tf_type.resource<tensor<32xf32>>) { return })mlir"));
  std::string printed;
  llvm::raw_string_ostream os(printed);
  module_->print(os);
  EXPECT_NE(printed.find("!tf_type.resource<tensor<32xf32>>"), std::string::npos);
}

TEST_F(StablehloSinkComponentTest, EmitV2OpContract) {
  {  // Success: attribute assembly, preservation and the sentinel fill.
    auto f = func(R"mlir(func.func @f(%d: tensor<2xf32>, %i: tensor<2xi32>, %s: tensor<2xi32>) { return })mlir");
    ASSERT_TRUE(f);
    IRRewriter rewriter(ctx_.get());
    rewriter.setInsertionPoint(f.getBody().front().getTerminator());
    auto target = annc::resolveV2Target(
        ctx_.get(), "SparseSegmentMean", {{"SparseSegmentMean", "sparse_segment_mean_v2"}});
    ASSERT_TRUE(target);
    SmallVector<Value> operands(f.getArguments().begin(), f.getArguments().end());
    SmallVector<Type> results{RankedTensorType::get({2}, rewriter.getF32Type())};
    auto source = DictionaryAttr::get(
        ctx_.get(), {NamedAttribute(StringAttr::get(ctx_.get(), "custom_attr"),
                                    rewriter.getI64IntegerAttr(1)),
                     NamedAttribute(StringAttr::get(ctx_.get(), "residue_reason"),
                                    rewriter.getStringAttr("unknown_op"))});
    SmallVector<Value> emitted;
    ASSERT_TRUE(succeeded(annc::emitV2Op(rewriter, f.getLoc(), *target, operands,
                                         results, source, {}, {}, emitted)));
    ASSERT_EQ(emitted.size(), 1u);
    Operation* op = emitted[0].getDefiningOp();
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->getNumOperands(), 4u);  // 3 inputs + filled sentinel
    EXPECT_TRUE(op->hasAttr("custom_attr"));
    EXPECT_FALSE(op->hasAttr("residue_reason"));
    EXPECT_TRUE(isa<arith::ConstantOp>(op->getOperand(3).getDefiningOp()));
  }
  {  // Failure: TopK misses its `k`; the fill must NOT apply (0 is not a k).
    auto f = func(R"mlir(func.func @f(%x: tensor<4xf32>) { return })mlir");
    ASSERT_TRUE(f);
    IRRewriter rewriter(ctx_.get());
    rewriter.setInsertionPoint(f.getBody().front().getTerminator());
    auto target =
        annc::resolveV2Target(ctx_.get(), "TopKV2", {{"TopKV2", "top_k_v2"}});
    ASSERT_TRUE(target);
    SmallVector<Value> operands(f.getArguments().begin(), f.getArguments().end());
    SmallVector<Type> results{
        RankedTensorType::get({4, 2}, rewriter.getF32Type()),
        RankedTensorType::get({4, 2}, rewriter.getI32Type())};
    SmallVector<Value> emitted;
    EXPECT_TRUE(failed(annc::emitV2Op(rewriter, f.getLoc(), *target, operands,
                                      results, DictionaryAttr::get(ctx_.get()),
                                      {}, {}, emitted)));
    EXPECT_TRUE(emitted.empty() && find("atir.top_k_v2") == nullptr);
  }
}

}  // namespace
