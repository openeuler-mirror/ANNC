#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/CarrierMatchers.h"

#include <gtest/gtest.h>

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace atir::fusionv2;

namespace {

// Direct coverage of the matcher semantics: each view gets a happy path and
// at least one fail-closed rejection (AGENTS section 5: new matcher logic
// ships with positive and negative cases).
class FusionV2MatchersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry_.insert<atir::AtirDialect, func::FuncDialect, arith::ArithDialect,
                     shape::ShapeDialect, tensor::TensorDialect,
                     stablehlo::StablehloDialect>();
    ctx_ = std::make_unique<MLIRContext>(registry_);
    ctx_->loadAllAvailableDialects();
  }

  func::FuncOp parseMain(llvm::StringRef ir) {
    module_ = parseSourceString<ModuleOp>(ir, ctx_.get());
    EXPECT_TRUE(module_);
    if (!module_) return func::FuncOp();
    return (*module_).lookupSymbol<func::FuncOp>("main");
  }

  std::unique_ptr<MLIRContext> ctx_;
  OwningOpRef<ModuleOp> module_;
  DialectRegistry registry_;
};

TEST_F(FusionV2MatchersTest, GatherLikeMatchesTorchIndexSelectAndRejectsResidueAxis) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%t: tensor<8xf32>, %i: tensor<2xi64>) -> tensor<2xf32> {
  %c3 = stablehlo.constant dense<3> : tensor<i32>
  %0 = "stablehlo.torch_index_select"(%t, %i)
      <{batch_dims = 0 : i64, dim = 0 : i64}>
      : (tensor<8xf32>, tensor<2xi64>) -> tensor<2xf32>
  %g = "atir.unsupported"(%t, %i, %c3)
      {detail = "", metadata = {}, opType = "GatherV2", reason = "unknown_op"}
      : (tensor<8xf32>, tensor<2xi64>, tensor<i32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
)mlir");
  ASSERT_TRUE(main);
  Operation *bad = nullptr;
  for (Operation &op : main.getBody().front())
    if (auto t = op.getAttrOfType<StringAttr>("opType"))
      if (t.getValue() == "GatherV2") bad = &op;
  ASSERT_TRUE(bad);
  EXPECT_FALSE(GatherLike::match(bad)) << "non-zero axis must bail";

  Operation *tis = nullptr;
  for (Operation &op : main.getBody().front())
    if (op.getName().getStringRef() == "stablehlo.torch_index_select")
      tis = &op;
  ASSERT_TRUE(tis);
  GatherLike view = GatherLike::match(tis);
  ASSERT_TRUE(view);
  EXPECT_TRUE(view.axisIsZero());
  EXPECT_TRUE(view.indices() == tis->getOperand(1));
}

TEST_F(FusionV2MatchersTest, SliceLikeResidueRequiresFourOperands) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%x: tensor<8x4xf32>, %b: tensor<2xi32>, %e: tensor<2xi32>,
                %s: tensor<2xi32>) -> tensor<4xf32> {
  %0 = "atir.unsupported"(%x, %b, %e, %s)
      {detail = "", metadata = {}, opType = "StridedSlice", reason = "unknown_op"}
      : (tensor<8x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
      -> tensor<4xf32>
  %short = "atir.unsupported"(%x, %b)
      {detail = "", metadata = {}, opType = "StridedSlice", reason = "unknown_op"}
      : (tensor<8x4xf32>, tensor<2xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
)mlir");
  ASSERT_TRUE(main);
  SmallVector<Operation *> residues;
  for (Operation &op : main.getBody().front())
    if (auto t = op.getAttrOfType<StringAttr>("opType"))
      if (t.getValue() == "StridedSlice") residues.push_back(&op);
  ASSERT_EQ(residues.size(), 2u);
  SliceLike full = SliceLike::match(residues[0]);
  ASSERT_TRUE(full);
  EXPECT_TRUE(full.isResidue());
  EXPECT_TRUE(full.start() == residues[0]->getOperand(1));
  EXPECT_FALSE(SliceLike::match(residues[1])) << "3-operand form must bail";
}

TEST_F(FusionV2MatchersTest, FillLikeShapeEmptyForStaticBroadcast) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%s: tensor<f32>, %d: tensor<2xi64>) -> tensor<8x4xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%s)
      <{broadcast_dimensions = array<i64>}>
      : (tensor<f32>) -> tensor<8x4xf32>
  %1 = "stablehlo.dynamic_broadcast_in_dim"(%s, %d)
      <{broadcast_dimensions = array<i64>}>
      : (tensor<f32>, tensor<2xi64>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}
)mlir");
  ASSERT_TRUE(main);
  Operation *stat = nullptr, *dyn = nullptr;
  for (Operation &op : main.getBody().front()) {
    if (op.getName().getStringRef() == "stablehlo.broadcast_in_dim") stat = &op;
    if (op.getName().getStringRef() == "stablehlo.dynamic_broadcast_in_dim")
      dyn = &op;
  }
  ASSERT_TRUE(stat && dyn);
  FillLike staticView = FillLike::match(stat);
  ASSERT_TRUE(staticView);
  EXPECT_FALSE(staticView.shape()) << "static form carries sizes in the attr";
  EXPECT_TRUE(staticView.value() == stat->getOperand(0));
  FillLike dynamicView = FillLike::match(dyn);
  ASSERT_TRUE(dynamicView);
  EXPECT_TRUE(dynamicView.shape() == dyn->getOperand(1));
}

TEST_F(FusionV2MatchersTest, PartitionLikeResidueHappyPathAndArityReject) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%data: tensor<8xi32>, %parts: tensor<8xi32>) -> (tensor<3xi32>, tensor<3xi32>) {
  %0:2 = "atir.unsupported"(%data, %parts)
      {detail = "", metadata = {}, opType = "DynamicPartition", reason = "unknown_op"}
      : (tensor<8xi32>, tensor<8xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  %bad = "atir.unsupported"(%data)
      {detail = "", metadata = {}, opType = "DynamicPartition", reason = "unknown_op"}
      : (tensor<8xi32>) -> tensor<3xi32>
  return %0#0, %0#1 : tensor<3xi32>, tensor<3xi32>
}
)mlir");
  ASSERT_TRUE(main);
  SmallVector<Operation *> ops;
  for (Operation &op : main.getBody().front())
    if (auto t = op.getAttrOfType<StringAttr>("opType"))
      if (t.getValue() == "DynamicPartition") ops.push_back(&op);
  ASSERT_EQ(ops.size(), 2u);
  PartitionLike view = PartitionLike::match(ops[0]);
  ASSERT_TRUE(view);
  EXPECT_TRUE(view.residue);
  EXPECT_EQ(view.count(), 2);
  EXPECT_TRUE(view.data == ops[0]->getOperand(0));
  EXPECT_FALSE(PartitionLike::match(ops[1])) << "single-operand form must bail";
}

TEST_F(FusionV2MatchersTest, UniqueLikeRejectsMultipleExternalInputs) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%a: tensor<8xi64>, %b: tensor<8xi64>) -> (tensor<8xi64>, tensor<8xi64>) {
  %0 = "stablehlo.convert"(%a)
      {annc.source_node = "n", annc.source_op = "tf.Unique",
       annc.source_ports = dense<0> : tensor<1xi32>}
      : (tensor<8xi64>) -> tensor<8xi64>
  %1 = "stablehlo.convert"(%b)
      {annc.source_node = "n", annc.source_op = "tf.Unique",
       annc.source_ports = dense<1> : tensor<1xi32>}
      : (tensor<8xi64>) -> tensor<8xi64>
  return %0, %1 : tensor<8xi64>, tensor<8xi64>
}
)mlir");
  ASSERT_TRUE(main);
  Operation *first = &main.getBody().front().front();
  EXPECT_FALSE(UniqueLike::match(first)) << "two distinct inputs are ambiguous";
}

TEST_F(FusionV2MatchersTest, UniqueLikeResidueHappyPath) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%x: tensor<8xi64>) -> (tensor<4xi64>, tensor<4xi32>) {
  %u:2 = "atir.unsupported"(%x)
      {detail = "", metadata = {}, opType = "Unique", reason = "unknown_op"}
      : (tensor<8xi64>) -> (tensor<4xi64>, tensor<4xi32>)
  return %u#0, %u#1 : tensor<4xi64>, tensor<4xi32>
}
)mlir");
  ASSERT_TRUE(main);
  Operation *u = &main.getBody().front().front();
  UniqueLike view = UniqueLike::match(u);
  ASSERT_TRUE(view);
  EXPECT_TRUE(view.residue);
  EXPECT_TRUE(view.values == u->getResult(0));
  EXPECT_TRUE(view.indices == u->getResult(1));
  EXPECT_TRUE(view.input == u->getOperand(0));
}

}  // namespace
