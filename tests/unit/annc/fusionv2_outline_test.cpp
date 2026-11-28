#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/Passes/FusionV2/FusionV2Outline.h"

#include <gtest/gtest.h>

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {

// Exercises outlineSite directly: the pass-level entry (OpFusionV2) ships in
// the next batch, so the batch-local contract is verified here instead of
// through a FileCheck run.
class FusionV2OutlineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry_.insert<atir::AtirDialect, func::FuncDialect, arith::ArithDialect,
                     shape::ShapeDialect, stablehlo::StablehloDialect>();
    ctx_ = std::make_unique<MLIRContext>(registry_);
    ctx_->loadAllAvailableDialects();
  }

  func::FuncOp parseMain(llvm::StringRef ir) {
    module_ = parseSourceString<ModuleOp>(ir, ctx_.get());
    EXPECT_TRUE(module_);
    if (!module_) return func::FuncOp();
    return (*module_).lookupSymbol<func::FuncOp>("main");
  }

  atir::fusionv2::ProbeArg probeArg(Value v, llvm::StringRef tfName,
                                    llvm::StringRef role) {
    atir::fusionv2::ProbeArg arg;
    arg.type = cast<RankedTensorType>(v.getType());
    arg.value = v;
    arg.tfName = tfName.str();
    arg.role = role.str();
    return arg;
  }

  std::unique_ptr<MLIRContext> ctx_;
  OwningOpRef<ModuleOp> module_;
  DialectRegistry registry_;
};

// Happy path: a two-arg site (block arg + constant) outlined into a kernel
// func with the V1 metadata contract, and the tail replaced by a call.
TEST_F(FusionV2OutlineTest, OutlinesSiteWithMetadataContract) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%arg0: tensor<8xf32>) -> tensor<2xf32> {
  %c = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
  %0 = "stablehlo.torch_index_select"(%arg0, %c)
      <{batch_dims = 0 : i64, dim = 0 : i64}>
      : (tensor<8xf32>, tensor<2xi64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
)mlir");
  ASSERT_TRUE(main);

  stablehlo::TorchIndexSelectOp select =
      *main.getBody().front().getOps<stablehlo::TorchIndexSelectOp>().begin();
  Value arg0 = main.getArgument(0);
  Value cst = select->getOperand(1);

  PatternRewriter rewriter(ctx_.get());
  SmallVector<atir::fusionv2::ProbeArg> args{
      probeArg(arg0, "table", "dynamic"),
      probeArg(cst, "idx", "constant")};
  llvm::SmallVector<std::string> tailAnchors{"sel:0"};
  atir::fusionv2::CallPortTable portTable;
  LogicalResult status = atir::fusionv2::outlineSite(
      main, rewriter, args, {select.getOperation()}, {select.getOperation()},
      {select.getResult()}, "TestKernel", "unit_test", tailAnchors, {},
      portTable);
  ASSERT_TRUE(succeeded(status));

  ModuleOp module = main->getParentOfType<ModuleOp>();
  auto kernel = module.lookupSymbol<func::FuncOp>("TestKernel");
  ASSERT_TRUE(kernel);
  EXPECT_TRUE(kernel->hasAttr("annc.kernel"));
  EXPECT_TRUE(kernel.isPrivate());
  EXPECT_EQ(kernel->getAttrOfType<StringAttr>("fusion.pattern").getValue(),
            "unit_test");
  ASSERT_EQ(kernel.getNumArguments(), 2u);
  EXPECT_TRUE(kernel.getArgument(0).getType() ==
              RankedTensorType::get({8}, rewriter.getF32Type()));

  DictionaryAttr md = kernel->getAttrOfType<DictionaryAttr>("fusion.metadata");
  ASSERT_TRUE(md);
  EXPECT_EQ(md.getAs<StringAttr>("abi").getValue(), "annc_context_v3");
  auto mdArgs = md.getAs<ArrayAttr>("args");
  ASSERT_TRUE(mdArgs && mdArgs.size() == 2);
  EXPECT_EQ(mdArgs[0].cast<DictionaryAttr>().getAs<StringAttr>("tf_name")
                .getValue(),
            "table");
  EXPECT_EQ(mdArgs[1].cast<DictionaryAttr>().getAs<StringAttr>("role")
                .getValue(),
            "constant");
  auto mdOutputs = md.getAs<ArrayAttr>("outputs");
  ASSERT_TRUE(mdOutputs && mdOutputs.size() == 1);
  EXPECT_EQ(
      mdOutputs[0].cast<DictionaryAttr>().getAs<StringAttr>("tf_name").getValue(),
      "sel:0");

  auto call = *main.getBody().front().getOps<func::CallOp>().begin();
  EXPECT_EQ(call.getCallee(), "TestKernel");
  ASSERT_EQ(call.getNumOperands(), 2u);
  EXPECT_TRUE(call.getOperand(0) == arg0);
  EXPECT_TRUE(call.getOperand(1) == cst);
  auto ret = *main.getBody().front().getOps<func::ReturnOp>().begin();
  EXPECT_TRUE(ret.getOperand(0) == call.getResult(0));
  for (Operation &op : kernel.getBody().front())
    if (isa<stablehlo::TorchIndexSelectOp>(op)) return;
  FAIL() << "select op was not landed inside the kernel";
}

// Stay set: a member value with an out-of-site user is cloned into the
// kernel while the original stays in main for that user.
TEST_F(FusionV2OutlineTest, ClonesStayMembersInsteadOfMoving) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%arg0: tensor<8xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %c = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
  %0 = "stablehlo.torch_index_select"(%arg0, %c)
      <{batch_dims = 0 : i64, dim = 0 : i64}>
      : (tensor<8xf32>, tensor<2xi64>) -> tensor<2xf32>
  %1 = stablehlo.convert %0 : (tensor<2xf32>) -> tensor<2xf32>
  return %0, %1 : tensor<2xf32>, tensor<2xf32>
}
)mlir");
  ASSERT_TRUE(main);

  stablehlo::TorchIndexSelectOp select =
      *main.getBody().front().getOps<stablehlo::TorchIndexSelectOp>().begin();
  auto convert = *main.getBody().front().getOps<stablehlo::ConvertOp>().begin();
  Value arg0 = main.getArgument(0);
  Value cst = select->getOperand(1);

  PatternRewriter rewriter(ctx_.get());
  SmallVector<atir::fusionv2::ProbeArg> args{
      probeArg(arg0, "table", "dynamic"),
      probeArg(cst, "idx", "constant")};
  llvm::SmallVector<std::string> tailAnchors{"conv:0"};
  atir::fusionv2::CallPortTable stayTable;
  LogicalResult status = atir::fusionv2::outlineSite(
      main, rewriter, args, {select.getOperation()}, {convert.getOperation()},
      {convert.getResult()}, "StayKernel", "unit_test", tailAnchors, {},
      stayTable);
  ASSERT_TRUE(succeeded(status));

  ModuleOp module = main->getParentOfType<ModuleOp>();
  auto kernel = module.lookupSymbol<func::FuncOp>("StayKernel");
  ASSERT_TRUE(kernel);
  unsigned selectsInKernel = 0, convertsInKernel = 0;
  for (Operation &op : kernel.getBody().front()) {
    selectsInKernel += isa<stablehlo::TorchIndexSelectOp>(op);
    convertsInKernel += isa<stablehlo::ConvertOp>(op);
  }
  EXPECT_EQ(selectsInKernel, 1u);
  EXPECT_EQ(convertsInKernel, 1u);
  EXPECT_TRUE(isa<stablehlo::TorchIndexSelectOp>(select.getOperation()));
  EXPECT_TRUE(isa<stablehlo::ConvertOp>(convert.getOperation()));
}

// Contract defense: empty tails and unranked outputs must fail, leaving the
// module untouched.
TEST_F(FusionV2OutlineTest, RejectsEmptyTailAndUnrankedOutput) {
  func::FuncOp main = parseMain(R"mlir(
func.func @main(%arg0: tensor<8xf32>) -> tensor<2xf32> {
  %c = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
  %0 = "stablehlo.torch_index_select"(%arg0, %c)
      <{batch_dims = 0 : i64, dim = 0 : i64}>
      : (tensor<8xf32>, tensor<2xi64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
)mlir");
  ASSERT_TRUE(main);

  stablehlo::TorchIndexSelectOp select =
      *main.getBody().front().getOps<stablehlo::TorchIndexSelectOp>().begin();
  SmallVector<atir::fusionv2::ProbeArg> args{
      probeArg(main.getArgument(0), "table", "dynamic"),
      probeArg(select->getOperand(1), "idx", "constant")};
  PatternRewriter rewriter(ctx_.get());
  llvm::SmallVector<std::string> tailAnchors{"sel:0"};
  atir::fusionv2::CallPortTable badTable;
  EXPECT_TRUE(failed(atir::fusionv2::outlineSite(
      main, rewriter, args, {select.getOperation()}, {}, {select.getResult()},
      "Bad", "unit_test", tailAnchors, {}, badTable)));
  EXPECT_FALSE(main->getParentOfType<ModuleOp>().lookupSymbol("Bad"));
}

}  // namespace
