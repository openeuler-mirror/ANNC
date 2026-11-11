// Unit tests for the T7 string wrap passes: the encode/decode transport
// sandwich and the post-legalize rollback (kept + rolled-back paths).
// The tf_type string element type requires the real dialect, everything
// else stays in quoted generic form.

#include "pb2_test_harness.h"
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "common/pb2_common.h"

namespace pb2 {
std::unique_ptr<mlir::Pass> createStringTransportPass();
std::unique_ptr<mlir::Pass> createStringUnwrapPass();
}  // namespace pb2

namespace {

void loadDialects(mlir::MLIRContext& ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tf_type::TFTypeDialect>();
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

mlir::Operation* findReturn(mlir::ModuleOp module) {
  mlir::Operation* ret = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "func.return") ret = op;
  });
  return ret;
}

}  // namespace

// A string Identity becomes an encodeString -> tf.Identity(i8) ->
// decodeString sandwich; the rebuilt op runs on the packed i8 domain.
static void TestTransportWrap() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string> {
        %0 = "tf.residue"(%arg0) {call_target_name = "tf.Identity"} : (tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string>
        func.return %0 : tensor<2x!tf_type.string>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStringTransportPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* enc = findByTarget(*module, "annc.encodeString");
  auto* dec = findByTarget(*module, "annc.decodeString");
  PB2_CHECK(enc != nullptr);
  PB2_CHECK(dec != nullptr);
  auto* id = findByName(*module, "tf.Identity");
  PB2_CHECK(id != nullptr);  // rebuilt on the i8 domain
  PB2_CHECK(id->getOperand(0).getDefiningOp() == enc);
  PB2_CHECK(dec->getOperand(0).getDefiningOp() == id);
  auto* ret = findReturn(*module);
  PB2_CHECK(ret && ret->getOperand(0).getDefiningOp() == dec);
}

// A sandwich whose op2 stayed in tf form is rolled back: operands rewired
// to the original string source, result type restored, boundaries dropped.
static void TestUnwrapRollsBackTfForm() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string> {
        %e = "tf.residue"(%arg0) {call_target_name = "annc.encodeString"} : (tensor<2x!tf_type.string>) -> tensor<2x?xi8>
        %t = "tf.Identity"(%e) : (tensor<2x?xi8>) -> tensor<2x?xi8>
        %d = "tf.residue"(%t) {call_target_name = "annc.decodeString"} : (tensor<2x?xi8>) -> tensor<2x!tf_type.string>
        func.return %d : tensor<2x!tf_type.string>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStringUnwrapPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* id = findByName(*module, "tf.Identity");
  PB2_CHECK(id != nullptr);
  PB2_CHECK(findByTarget(*module, "annc.encodeString") == nullptr);
  PB2_CHECK(findByTarget(*module, "annc.decodeString") == nullptr);
  auto* ret = findReturn(*module);
  PB2_CHECK(ret && ret->getOperand(0).getDefiningOp() == id);
  // Operand rewired to the original string entry, type restored.
  PB2_CHECK(llvm::isa<mlir::BlockArgument>(id->getOperand(0)));
  PB2_CHECK(llvm::isa<mlir::TensorType>(id->getResult(0).getType()));
}

// A sandwich whose op2 legalized to stablehlo is kept untouched.
static void TestUnwrapKeepsLegalized() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string> {
        %e = "tf.residue"(%arg0) {call_target_name = "annc.encodeString"} : (tensor<2x!tf_type.string>) -> tensor<2x?xi8>
        %t = "stablehlo.add"(%e, %e) : (tensor<2x?xi8>, tensor<2x?xi8>) -> tensor<2x?xi8>
        %d = "tf.residue"(%t) {call_target_name = "annc.decodeString"} : (tensor<2x?xi8>) -> tensor<2x!tf_type.string>
        func.return %d : tensor<2x!tf_type.string>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStringUnwrapPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(findByTarget(*module, "annc.encodeString") != nullptr);
  PB2_CHECK(findByTarget(*module, "annc.decodeString") != nullptr);
  auto* ret = findReturn(*module);
  PB2_CHECK(ret && ret->getOperand(0).getDefiningOp() ==
                    findByTarget(*module, "annc.decodeString"));
}

// ConcatV2 with axis=-1 on a rank-2 string tensor: the normalized axis
// hits the (about-to-be-appended) payload dim, so the adapter must SKIP
// the op entirely -- encoding would silently concatenate along bytes.
static void TestTransportSkipsLastDimAxis() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<2x3x!tf_type.string>, %arg1: tensor<2x3x!tf_type.string>) -> tensor<2x!tf_type.string> {
        %ax = "some.const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
        %0 = "tf.residue"(%arg0, %arg1, %ax) {call_target_name = "tf.ConcatV2"} : (tensor<2x3x!tf_type.string>, tensor<2x3x!tf_type.string>, tensor<i32>) -> tensor<2x!tf_type.string>
        func.return %0 : tensor<2x!tf_type.string>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStringTransportPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  // Untouched: no sandwich was built (axis would point at the payload dim).
  PB2_CHECK(findByTarget(*module, "annc.encodeString") == nullptr);
  auto* kept = findByTarget(*module, "tf.ConcatV2");
  PB2_CHECK(kept != nullptr && kept->getNumOperands() == 3);
}

// DynamicStitch layout is [indices..., data...]: the data half (operands
// 2..3 here) gets the encode sandwich, the index half stays untouched.
static void TestTransportDynamicStitchLayout() {
  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%i0: tensor<2xi32>, %i1: tensor<3xi32>, %d0: tensor<2x!tf_type.string>, %d1: tensor<3x!tf_type.string>) -> tensor<5x!tf_type.string> {
        %0 = "tf.residue"(%i0, %i1, %d0, %d1) {call_target_name = "tf.DynamicStitch"} : (tensor<2xi32>, tensor<3xi32>, tensor<2x!tf_type.string>, tensor<3x!tf_type.string>) -> tensor<5x!tf_type.string>
        func.return %0 : tensor<5x!tf_type.string>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb2::createStringTransportPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* st = findByName(*module, "tf.DynamicStitch");
  PB2_CHECK(st != nullptr);  // rebuilt with encoded data operands
  // Two encodes: one per data operand (the index half carries no strings).
  int encodes = 0;
  module->walk([&](mlir::Operation* op) {
    auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
    if (ct && ct.getValue() == "annc.encodeString") ++encodes;
  });
  PB2_CHECK(encodes == 2);
  PB2_CHECK(st->getOperand(0).getDefiningOp() == nullptr);  // raw index arg
}

int main() {
  TestTransportWrap();
  TestUnwrapRollsBackTfForm();
  TestUnwrapKeepsLegalized();
  TestTransportSkipsLastDimAxis();
  TestTransportDynamicStitchLayout();
  PB2_TEST_SUMMARY("string-test");
}
