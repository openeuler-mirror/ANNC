// Unit tests for the T6 shape-wrap passes: the unrank getDim/setDim
// sandwich, the dynamic-reshape short-circuit, and the dynamic-input
// StridedSlice translation. Quoted generic IR; ops addressed via their
// residue-carrier form (call_target_name dispatch).

#include "pb2_test_harness.h"
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "common/pb2_common.h"

namespace pb2 {
std::unique_ptr<mlir::Pass> createUnrankWrapPass();
std::unique_ptr<mlir::Pass> createStridedSliceDynPass();
}  // namespace pb2

namespace {

void loadDialects(mlir::MLIRContext& ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  ctx.allowUnregisteredDialects(true);
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext& ctx,
                                        const char* ir) {
  return mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
}

mlir::Operation* findByTarget(mlir::ModuleOp module, llvm::StringRef target) {
  mlir::Operation* found = nullptr;
  module->walk([&](mlir::Operation* op) {
    auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
    if (ct && ct.getValue() == target && !found) found = op;
  });
  return found;
}

mlir::Operation* findByName(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::Operation* found = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == name && !found) found = op;
  });
  return found;
}

}  // namespace

// An allowlisted op (Relu) with an unranked operand becomes a
// getDim -> op(flat) -> setDim sandwich; the op itself stays as the
// flat-domain core.
static void TestUnrankWrap() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
        %0 = "tf.residue"(%arg0) {call_target_name = "tf.Relu", test.tag = "t"} : (tensor<*xf32>) -> tensor<*xf32>
        func.return %0 : tensor<*xf32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createUnrankWrapPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* enc = findByTarget(*module, "annc.getDim");
  auto* dec = findByTarget(*module, "annc.setDim");
  PB2_CHECK(enc != nullptr);
  PB2_CHECK(dec != nullptr);
  PB2_CHECK(enc->getNumResults() == 2);   // flat + shape vector
  PB2_CHECK(dec->getNumOperands() == 1);  // the flattened op result
  auto* core = findByTarget(*module, "tf.Relu");
  PB2_CHECK(core != nullptr);  // survived as the flat-domain core
  PB2_CHECK(core->getOperand(0).getDefiningOp() == enc);
}

// A constant-bounds StridedSlice on a dynamic dimension translates to
// real_dynamic_slice plus the shrink-axis folding reshape.
static void TestStridedSliceDyn() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
        %b = "some.const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
        %e = "some.const"() {value = dense<[2, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
        %s = "some.const"() {value = dense<[1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
        %0 = "tf.residue"(%arg0, %b, %e, %s) {call_target_name = "tf.StridedSlice", shrink_axis_mask = 1 : i64} : (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xf32>
        func.return %0 : tensor<2xf32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStridedSliceDynPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* slice = findByName(*module, "stablehlo.real_dynamic_slice");
  auto* reshape = findByName(*module, "stablehlo.dynamic_reshape");
  PB2_CHECK(slice != nullptr);
  PB2_CHECK(slice->getNumOperands() == 4);  // in + start/limit/stride
  PB2_CHECK(reshape != nullptr);            // shrink dim 0 folded away
  PB2_CHECK(findByTarget(*module, "tf.StridedSlice") == nullptr);
  mlir::Operation* ret = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "func.return") ret = op;
  });
  PB2_CHECK(ret && ret->getOperand(0).getDefiningOp() == reshape);
}

// Broadcasting binaries (Add) are NOT in the allowlist: an unranked
// operand plus a ranked one would flatten mismatched shapes into wrong
// math, so the pass must leave them as residue.
static void TestUnrankWrapSkipsBinaries() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<*xf32>, %arg1: tensor<3xf32>) -> tensor<*xf32> {
        %0 = "tf.residue"(%arg0, %arg1) {call_target_name = "tf.Add"} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
        func.return %0 : tensor<*xf32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createUnrankWrapPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* add = findByTarget(*module, "tf.Add");
  PB2_CHECK(add != nullptr);  // untouched
  PB2_CHECK(findByTarget(*module, "annc.getDim") == nullptr);
  PB2_CHECK(add->getNumOperands() == 2);  // operands not flattened
}

// A rank-0 input is skipped with the bucketed reason instead of crashing
// on empty start/limit/stride vectors.
static void TestStridedSliceRank0Skips() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %b = "some.const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
        %0 = "tf.residue"(%arg0, %b, %b, %b) {call_target_name = "tf.StridedSlice"} : (tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<f32>
        func.return %0 : tensor<f32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStridedSliceDynPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(findByTarget(*module, "tf.StridedSlice") != nullptr);  // kept
  PB2_CHECK(findByName(*module, "stablehlo.real_dynamic_slice") == nullptr);
}

// A negative begin beyond the dimension clamps to zero (TF semantics)
// instead of staying negative after normalization.
static void TestStridedSliceNegativeBeginClamps() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %b = "some.const"() {value = dense<-100> : tensor<1xi32>} : () -> tensor<1xi32>
        %e = "some.const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
        %s = "some.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
        %0 = "tf.residue"(%arg0, %b, %e, %s) {call_target_name = "tf.StridedSlice"} : (tensor<4xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xf32>
        func.return %0 : tensor<4xf32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStridedSliceDynPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  // start vector = constant 0: find the leading constant feeding the
  // concatenate chain of real_dynamic_slice's operand 1.
  auto* slice = findByName(*module, "stablehlo.real_dynamic_slice");
  PB2_CHECK(slice != nullptr);
  // The slice covers the whole dim: result type stays tensor<4xf32>.
  auto rt = llvm::cast<mlir::RankedTensorType>(slice->getResult(0).getType());
  PB2_CHECK(rt.getShape() == mlir::ArrayRef<int64_t>{4});
}

// Shrink on a dynamic dimension is skipped: a runtime-clamped start with
// a constant begin+1 limit could place the window past the dimension end.
static void TestStridedSliceShrinkDynamicSkips() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<?xf32>) -> tensor<f32> {
        %b = "some.const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
        %0 = "tf.residue"(%arg0, %b, %b, %b) {call_target_name = "tf.StridedSlice", shrink_axis_mask = 1 : i64} : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<f32>
        func.return %0 : tensor<f32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStridedSliceDynPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(findByTarget(*module, "tf.StridedSlice") != nullptr);  // kept
  PB2_CHECK(findByName(*module, "stablehlo.real_dynamic_slice") == nullptr);
}

// TF's negative-end semantics vs the old kDynamic sentinel: x[0:-1] is
// "drop the last element" (limit 3), NOT "to the dimension end" (4).
static void TestStridedSliceNegativeEndOne() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<3xf32> {
        %b = "some.const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
        %e = "some.const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
        %s = "some.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
        %0 = "tf.residue"(%arg0, %b, %e, %s) {call_target_name = "tf.StridedSlice"} : (tensor<4xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xf32>
        func.return %0 : tensor<3xf32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStridedSliceDynPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* slice = findByName(*module, "stablehlo.real_dynamic_slice");
  PB2_CHECK(slice != nullptr);
  auto rt = llvm::cast<mlir::RankedTensorType>(slice->getResult(0).getType());
  PB2_CHECK(rt.getShape() == mlir::ArrayRef<int64_t>{3});  // not 4
}

// x[10:] with end_mask on a dim of 4: begin clamps into the dimension,
// the result is empty -- not a negative-span illegal type.
static void TestStridedSliceOobBeginClamps() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<0xf32> {
        %b = "some.const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
        %e = "some.const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
        %s = "some.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
        %0 = "tf.residue"(%arg0, %b, %e, %s) {call_target_name = "tf.StridedSlice", end_mask = 1 : i64} : (tensor<4xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xf32>
        func.return %0 : tensor<0xf32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStridedSliceDynPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* slice = findByName(*module, "stablehlo.real_dynamic_slice");
  PB2_CHECK(slice != nullptr);
  auto rt = llvm::cast<mlir::RankedTensorType>(slice->getResult(0).getType());
  PB2_CHECK(rt.getShape() == mlir::ArrayRef<int64_t>{0});  // empty
}

// x[:-100] on a dim of 4: the far-out negative end clamps to an empty
// result -- never a negative limit.
static void TestStridedSliceFarNegativeEndClamps() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) -> tensor<0xf32> {
        %b = "some.const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
        %e = "some.const"() {value = dense<-100> : tensor<1xi32>} : () -> tensor<1xi32>
        %s = "some.const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
        %0 = "tf.residue"(%arg0, %b, %e, %s) {call_target_name = "tf.StridedSlice"} : (tensor<4xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xf32>
        func.return %0 : tensor<0xf32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStridedSliceDynPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* slice = findByName(*module, "stablehlo.real_dynamic_slice");
  PB2_CHECK(slice != nullptr);
  auto rt = llvm::cast<mlir::RankedTensorType>(slice->getResult(0).getType());
  PB2_CHECK(rt.getShape() == mlir::ArrayRef<int64_t>{0});  // empty
}

int main() {
  TestUnrankWrap();
  TestStridedSliceDyn();
  TestUnrankWrapSkipsBinaries();
  TestStridedSliceRank0Skips();
  TestStridedSliceNegativeBeginClamps();
  TestStridedSliceShrinkDynamicSkips();
  TestStridedSliceNegativeEndOne();
  TestStridedSliceOobBeginClamps();
  TestStridedSliceFarNegativeEndClamps();
  PB2_TEST_SUMMARY("wrap-test");
}
