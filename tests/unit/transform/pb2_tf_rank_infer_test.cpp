// Unit tests for the tf-rank-inference pass (T2). Ops are written in the
// quoted generic form so the module parses without registering the tf
// dialect — exactly how unregistered residue carriers appear pre-legalize.

#include "pb2_test_harness.h"
#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "common/pb2_common.h"

namespace pb2 {
std::unique_ptr<mlir::Pass> createTfRankInferPass();
}

namespace {

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext& ctx,
                                        const char* ir) {
  // The quoted generic ops ride invented dialects ("some.", "tf.") that are
  // deliberately not registered in this context.
  ctx.allowUnregisteredDialects(true);
  return mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
}

// Rank of the value produced by the op tagged with test.tag = `tag`, or -1
// when that value stays unranked.
int rankOfTagged(mlir::ModuleOp module, llvm::StringRef tag) {
  int rank = -2;
  module->walk([&](mlir::Operation* op) {
    if (auto a = op->getAttrOfType<mlir::StringAttr>("test.tag"))
      if (a.getValue() == tag) {
        auto t = mlir::dyn_cast<mlir::RankedTensorType>(
            op->getResult(0).getType());
        rank = t ? (int)t.getRank() : -1;
      }
  });
  return rank;
}

static void TestSameShapePropagation() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %cst = "some.const"() {value = dense<0.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
      %0 = "tf.Cast"(%cst) {test.tag = "out"} : (tensor<2x3xf32>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(rankOfTagged(*module, "out") == 2);  // dims stay dynamic
}

static void TestBroadcastMaxRank() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %a = "some.const"() {value = dense<0.0> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
      %b = "some.const"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
      %0 = "tf.AddV2"(%a, %b) {test.tag = "out"} : (tensor<1x2x3xf32>, tensor<4xf32>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(rankOfTagged(*module, "out") == 3);
}

static void TestConflictingProposalsFreeze() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %r2 = "some.const"() {value = dense<0.0> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
      %r3 = "some.const"() {value = dense<0.0> : tensor<1x2x3xf32>} : () -> tensor<1x2x3xf32>
      %v = "some.source"() : () -> tensor<*xf32>
      %a = "tf.Identity"(%v, %r2) {test.tag = "ignored"} : (tensor<*xf32>, tensor<1x2xf32>) -> tensor<*xf32>
      %b = "tf.Identity"(%v, %r3) : (tensor<*xf32>, tensor<1x2x3xf32>) -> tensor<*xf32>
      %0 = "tf.Identity"(%v) {test.tag = "out"} : (tensor<*xf32>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  // %v receives rank-2 (via the a-group) and rank-3 (via the b-group)
  // proposals: the conflict freezes it and the tagged result stays unranked.
  PB2_CHECK(rankOfTagged(*module, "out") == -1);
}

// Combination semantics (layer matrix in frontend-transform.md): when the
// pb-domain official inference is disabled or failed, this engine is the
// only rank source for registered ops -- a rank must cross a multi-hop
// same-shape chain via the fixpoint, not just one hop.
static void TestTwoHopWithoutOfficialInference() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %cst = "some.const"() {value = dense<0.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
      %a = "tf.Cast"(%cst) : (tensor<2x3xf32>) -> tensor<*xf32>
      %0 = "tf.Cast"(%a) {test.tag = "out"} : (tensor<*xf32>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(rankOfTagged(*module, "out") == 2);  // crossed the middle hop
}

static void TestResidueCarrierDispatch() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %x = "some.const"() {value = dense<0.0> : tensor<4x5xf32>} : () -> tensor<4x5xf32>
      %y = "some.const"() {value = dense<0.0> : tensor<5x6xf32>} : () -> tensor<5x6xf32>
      %0 = "tf.residue"(%x, %y) {call_target_name = "tf.MatMul", test.tag = "out"} : (tensor<4x5xf32>, tensor<5x6xf32>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(rankOfTagged(*module, "out") == 2);  // MatMul: strictly rank-2
}

static void TestReshapeConstShapeVector() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %x = "some.const"() {value = dense<0.0> : tensor<24xf32>} : () -> tensor<24xf32>
      %s = "some.const"() {value = dense<[2, 3, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
      %0 = "tf.Reshape"(%x, %s) {test.tag = "out"} : (tensor<24xf32>, tensor<3xi64>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(rankOfTagged(*module, "out") == 3);  // rank = shape vector length
}

}  // namespace

// GatherV2 with rank-0 params/indices would compute rank -1; the guard
// must drop the proposal (no crash, no wrong upgrade).
static void TestNegativeRankProposalDropped() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %p = "some.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
      %i = "some.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %a = "some.const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
      %0 = "tf.GatherV2"(%p, %i, %a) {batch_dims = 0 : i64, test.tag = "out"} : (tensor<f32>, tensor<i32>, tensor<i64>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(rankOfTagged(*module, "out") == -1);  // stayed unranked
}

// A malformed "tf.Reshape" carrying fewer operands than the rule assumes
// must not read out of bounds: no proposal, module untouched.
static void TestShortArityReshapeIsIgnored() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %x = "some.const"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
      %0 = "tf.Reshape"(%x) {test.tag = "out"} : (tensor<4xf32>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(rankOfTagged(*module, "out") == -1);
}

// A negative StridedSlice mask is malformed IR: no equation may be
// fabricated from it (popcount(-1)=64 would propose a nonsense rank).
static void TestNegativeMaskIsRejected() {
  mlir::MLIRContext ctx;
  auto module = parse(ctx, R"mlir(
    module {
      %x = "some.const"() {value = dense<0.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
      %b = "some.const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
      %e = "some.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
      %s = "some.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
      %0 = "tf.StridedSlice"(%x, %b, %e, %s) {shrink_axis_mask = -1 : i64, test.tag = "out"} : (tensor<2x3xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createTfRankInferPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(rankOfTagged(*module, "out") == -1);
}

int main() {
  TestNegativeRankProposalDropped();
  TestShortArityReshapeIsIgnored();
  TestNegativeMaskIsRejected();
  TestSameShapePropagation();
  TestBroadcastMaxRank();
  TestConflictingProposalsFreeze();
  TestTwoHopWithoutOfficialInference();
  TestResidueCarrierDispatch();
  TestReshapeConstShapeVector();
  PB2_TEST_SUMMARY("rank-test");
}
