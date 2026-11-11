// Unit tests for the MLIR-domain passes (T4): suffix strip, executor
// flattening (positive + count-mismatch bail-out), residue wrap, carrier
// rename. Quoted generic IR; g_opspecs preloaded where the wrap gate needs it.

#include "pb2_test_harness.h"
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

#include "common/pb2_common.h"
#include "passes/pb2_mlir_passes.h"

namespace {

void loadExecutorDialects(mlir::MLIRContext& ctx) {
  // tf_executor ops must bind to their ODS C++ classes (the flattener
  // matches by C++ type, not by name), so parse without the
  // unregistered-dialect fallback here.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tf_executor::TensorFlowExecutorDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
  ctx.allowUnregisteredDialects(true);  // the "some.*" payload ops
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext& ctx,
                                        const char* ir) {
  return mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
}

mlir::Operation* findOp(mlir::ModuleOp module, llvm::StringRef name,
                        llvm::StringRef tag = "") {
  mlir::Operation* found = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() != name) return;
    if (!tag.empty()) {
      auto t = op->getAttrOfType<mlir::StringAttr>("test.tag");
      if (!t || t.getValue() != tag) return;
    }
    found = op;
  });
  return found;
}

}  // namespace

// "x@_functionalize___1" NameLocs keep the "__..." tail: "x___1".
static void TestStripFunctionalizeSuffix() {
  mlir::MLIRContext ctx;
  loadExecutorDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      "some.op"() {test.tag = "t"} : () -> () loc("conv@_functionalize___1")
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb_passes::CreateStripFunctionalizeSuffixPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* op = findOp(*module, "some.op", "t");
  PB2_CHECK(op != nullptr);
  auto loc = mlir::dyn_cast<mlir::NameLoc>(op->getLoc());
  PB2_CHECK(static_cast<bool>(loc));
  PB2_CHECK(loc.getName() == "conv___1");
}

// A pure-dataflow executor graph (islands + fetch) is flattened: payload
// ops lift out, graph/island/fetch disappear, and the graph result is
// rewired to the island's forwarded value.
static void TestFlattenExecutorPositive() {
  mlir::MLIRContext ctx;
  loadExecutorDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %g = "tf_executor.graph"() ({
          %i:2 = "tf_executor.island"() ({
            %v = "some.op"(%arg0) {test.tag = "payload"} : (tensor<f32>) -> tensor<f32>
            "tf_executor.yield"(%v) : (tensor<f32>) -> ()
          }) : () -> (tensor<f32>, !tf_executor.control)
          "tf_executor.fetch"(%i#0) : (tensor<f32>) -> ()
        }) : () -> tensor<f32>
        func.return %g : tensor<f32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb_passes::CreateFlattenExecutorPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* payload = findOp(*module, "some.op", "payload");
  PB2_CHECK(payload != nullptr);
  PB2_CHECK(findOp(*module, "tf_executor.graph") == nullptr);
  PB2_CHECK(findOp(*module, "tf_executor.island") == nullptr);
  PB2_CHECK(findOp(*module, "tf_executor.fetch") == nullptr);
  // The return must now consume the payload result directly.
  mlir::Operation* ret = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "func.return") ret = op;
  });
  PB2_CHECK(ret != nullptr);
  PB2_CHECK(ret->getNumOperands() == 1);
  PB2_CHECK(ret->getOperand(0).getDefiningOp() == payload);
}

// Count mismatches bail out whole. The malformed island (two data
// results, yield forwarding one) is built in C++ on purpose: the ODS
// verifier rejects the text form, but rewrite paths can produce it anyway.
static void TestFlattenExecutorCountMismatchBailsOut() {
  mlir::MLIRContext ctx;
  loadExecutorDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %g = "tf_executor.graph"() ({
          %i:2 = "tf_executor.island"() ({
            %v = "some.op"(%arg0) {test.tag = "p0"} : (tensor<f32>) -> tensor<f32>
            "tf_executor.yield"(%v) : (tensor<f32>) -> ()
          }) : () -> (tensor<f32>, !tf_executor.control)
          "tf_executor.fetch"(%i#0) : (tensor<f32>) -> ()
        }) : () -> tensor<f32>
        func.return %g : tensor<f32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));

  // Rebuild the island: same payload and single-value yield, but TWO data
  // results, then re-wire the old island's users onto result 0.
  mlir::OpBuilder b(&ctx);
  mlir::Operation* island = findOp(*module, "tf_executor.island");
  mlir::Operation* payload = findOp(*module, "some.op", "p0");
  PB2_CHECK(island != nullptr && payload != nullptr);
  mlir::Type f32t = payload->getResult(0).getType();

  b.setInsertionPoint(island);
  mlir::OperationState st(island->getLoc(), "tf_executor.island");
  st.addTypes({f32t, f32t});  // two data results, no control token
  st.addRegion();  // empty body; ownership moves into the op on create
  mlir::Operation* bad = b.create(st);

  mlir::Block* blk = b.createBlock(&bad->getRegion(0));
  payload->moveBefore(blk, blk->end());
  b.setInsertionPointToEnd(blk);
  mlir::OperationState yst(payload->getLoc(), "tf_executor.yield");
  yst.addOperands(payload->getResult(0));
  b.create(yst);

  island->getResult(0).replaceAllUsesWith(bad->getResult(0));
  island->erase();

  mlir::PassManager pm(&ctx, "builtin.module");
  pm.enableVerifier(false);  // the malformed intermediate must run anyway
  pm.addPass(pb_passes::CreateFlattenExecutorPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(findOp(*module, "tf_executor.graph") != nullptr);  // untouched
  PB2_CHECK(findOp(*module, "some.op", "p0") != nullptr);
}

// Registered tf.* leftovers become tf.residue carriers: attrs ride along,
// call_target_name keeps the op name, and the reason distinguishes
// static-gated classes from plain no-pattern ones.
static void TestWrapRegisteredResidue() {
  g_opspecs.clear();
  tensorflow::OpDef d;
  d.set_name("Add");
  g_opspecs["Add"] = d;

  mlir::MLIRContext ctx;
  loadExecutorDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      %a = "some.const"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
      %b = "some.const"() {value = dense<2.0> : tensor<2xf32>} : () -> tensor<2xf32>
      %0 = "tf.Add"(%a, %b) {T = "DT_FLOAT", test.tag = "t"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb_passes::CreateWrapRegisteredResiduePass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* carrier = findOp(*module, "tf.residue", "t");
  PB2_CHECK(carrier != nullptr);
  auto ct = carrier->getAttrOfType<mlir::StringAttr>("call_target_name");
  PB2_CHECK(static_cast<bool>(ct));
  PB2_CHECK(ct.getValue() == "tf.Add");
  auto reason = carrier->getAttrOfType<mlir::StringAttr>("residue_reason");
  PB2_CHECK(static_cast<bool>(reason));
  PB2_CHECK(reason.getValue() == "no_legalize_pattern");  // Add is not static-gated
  PB2_CHECK(carrier->getAttrOfType<mlir::StringAttr>("T") != nullptr);
  PB2_CHECK(carrier->getNumOperands() == 2);
  PB2_CHECK(findOp(*module, "tf.Add", "t") == nullptr);  // original gone
  g_opspecs.clear();
}

// tf.residue -> annc.residue after the wheel is done: every attr and the
// operand/result signature move over unchanged (the sink-side contract).
static void TestRenameResidueCarrier() {
  mlir::MLIRContext ctx;
  loadExecutorDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      %x = "some.const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
      %0 = "tf.residue"(%x) {call_target_name = "tf.SparseToDense", residue_reason = "no_legalize_pattern", test.tag = "t"} : (tensor<3xi32>) -> tensor<3xi32>
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb_passes::CreateRenameResidueCarrierPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  auto* carrier = findOp(*module, "annc.residue", "t");
  PB2_CHECK(carrier != nullptr);
  PB2_CHECK(carrier->getAttrOfType<mlir::StringAttr>("call_target_name")
                .getValue() == "tf.SparseToDense");
  PB2_CHECK(carrier->getAttrOfType<mlir::StringAttr>("residue_reason")
                .getValue() == "no_legalize_pattern");
  PB2_CHECK(carrier->getNumOperands() == 1);
  PB2_CHECK(findOp(*module, "tf.residue", "t") == nullptr);
}

int main() {
  TestStripFunctionalizeSuffix();
  TestFlattenExecutorPositive();
  TestFlattenExecutorCountMismatchBailsOut();
  TestWrapRegisteredResidue();
  TestRenameResidueCarrier();
  PB2_TEST_SUMMARY("mlir-passes-test");
}
