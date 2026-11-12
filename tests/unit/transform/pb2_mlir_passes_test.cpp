// Unit tests for the MLIR-domain passes (T4): suffix strip, executor
// flattening (positive, island control edges, count-mismatch bail-out),
// residue wrap, carrier rename. Quoted generic IR; g_opspecs preloaded
// where the wrap gate needs it.

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

// Both legal dependency directions in one graph: an If consuming an
// island's output AND an island consuming the If's result. Lifting must
// keep the original block order (def-before-use both ways), or the module
// stops verifying. "tf.If" is left unregistered here on purpose: the
// flattener identifies carriers by name, so the generic form with a real
// region exercises exactly the same path.
static void TestFlattenExecutorInterleavedDominance() {
  mlir::MLIRContext ctx;
  loadExecutorDialects(ctx);
  // The generic tf.If would be rejected inside an executor graph by the
  // ODS verifier; skip the parse-time check, flatten, then verify -- by
  // then the graph is gone and legality is restored.
  mlir::ParserConfig pcfg(&ctx, /*verifyAfterParse=*/false);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %g = "tf_executor.graph"() ({
          %src:2 = "tf_executor.island"() ({
            %v1 = "some.op"(%arg0) {test.tag = "src"} : (tensor<f32>) -> tensor<f32>
            "tf_executor.yield"(%v1) : (tensor<f32>) -> ()
          }) : () -> (tensor<f32>, !tf_executor.control)
          %ifr = "tf.If"(%src#0) ({
            %inner = "some.op"(%src#0) {test.tag = "inner"} : (tensor<f32>) -> tensor<f32>
            "tf.If.yield"(%inner) : (tensor<f32>) -> ()
          }) {test.tag = "if"} : (tensor<f32>) -> tensor<f32>
          %cons:2 = "tf_executor.island"() ({
            %v2 = "some.op"(%ifr) {test.tag = "cons"} : (tensor<f32>) -> tensor<f32>
            "tf_executor.yield"(%v2) : (tensor<f32>) -> ()
          }) : () -> (tensor<f32>, !tf_executor.control)
          "tf_executor.fetch"(%cons#0) : (tensor<f32>) -> ()
        }) : () -> tensor<f32>
        func.return %g : tensor<f32>
      }
    })mlir",
                                    pcfg);
  PB2_CHECK(static_cast<bool>(module));
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.enableVerifier(false);  // input holds the executor-illegal generic If
  pm.addPass(pb_passes::CreateFlattenExecutorPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(findOp(*module, "tf_executor.graph") == nullptr);
  auto* src = findOp(*module, "some.op", "src");
  auto* ifr = findOp(*module, "tf.If", "if");
  auto* cons = findOp(*module, "some.op", "cons");
  PB2_CHECK(src && ifr && cons);
  // Original block order: src before the If, the If before its consumer.
  unsigned iSrc = 0, iIf = 0, iCons = 0, idx = 0;
  module->walk([&](mlir::Operation* op) {
    if (op == src) iSrc = idx;
    if (op == ifr) iIf = idx;
    if (op == cons) iCons = idx;
    ++idx;
  });
  PB2_CHECK(iSrc < iIf && iIf < iCons);
  PB2_CHECK(succeeded(mlir::verify(*module)));  // dominance holds
}

// Island-to-island control edges are the importer's norm and must not
// bail the flatten out. Regression: a pre-strip use_empty() guard used to
// reject such graphs, leaving stand-ins unreachable by the restore scan.
static void TestFlattenExecutorIslandControlEdges() {
  mlir::MLIRContext ctx;
  loadExecutorDialects(ctx);
  // Generic text cannot spell island operands (IslandOp::parse derives
  // them from results): parsed operand-less, edges added in C++, matching
  // the importer shape.
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %g = "tf_executor.graph"() ({
          %src:2 = "tf_executor.island"() ({
            %v0 = "some.op"(%arg0) {test.tag = "src"} : (tensor<f32>) -> tensor<f32>
            "tf_executor.yield"(%v0) : (tensor<f32>) -> ()
          }) : () -> (tensor<f32>, !tf_executor.control)
          %cons:2 = "tf_executor.island"() ({
            %v1 = "some.op"(%src#0) {test.tag = "cons"} : (tensor<f32>) -> tensor<f32>
            "tf_executor.yield"(%v1) : (tensor<f32>) -> ()
          }) : () -> (tensor<f32>, !tf_executor.control)
          "tf_executor.fetch"(%cons#0) : (tensor<f32>) -> ()
        }) : () -> tensor<f32>
        func.return %g : tensor<f32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  // Block order is source order: islands[0] produces, islands[1] consumes.
  llvm::SmallVector<mlir::Operation*> islands;
  module->walk([&](mlir::tf_executor::IslandOp io) { islands.push_back(io); });
  PB2_CHECK(islands.size() == 2);
  mlir::Operation* src_island = islands[0];
  mlir::Operation* cons_island = islands[1];
  cons_island->setOperands(
      {src_island->getResult(0), src_island->getResult(1)});  // data + control
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.addPass(pb_passes::CreateFlattenExecutorPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(findOp(*module, "tf_executor.graph") == nullptr);
  PB2_CHECK(findOp(*module, "tf_executor.island") == nullptr);
  PB2_CHECK(findOp(*module, "tf_executor.fetch") == nullptr);
  auto* src = findOp(*module, "some.op", "src");
  auto* cons = findOp(*module, "some.op", "cons");
  PB2_CHECK(src && cons);
  // The consumer now takes the producer payload directly (control edge gone).
  PB2_CHECK(cons->getOperand(0).getDefiningOp() == src);
  PB2_CHECK(mlir::succeeded(mlir::verify(*module)));
}

// Arity mismatch (graph results != fetch data operands) must bail BEFORE
// strip_ctrl: a post-strip bail would leave control edges already stripped.
// The fetch's trailing control operand (still present after the bail) is
// the witness of a pre-mutation bail.
static void TestFlattenExecutorArityBailsBeforeMutation() {
  mlir::MLIRContext ctx;
  loadExecutorDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %g = "tf_executor.graph"() ({
          %i:2 = "tf_executor.island"() ({
            %v = "some.op"(%arg0) {test.tag = "p"} : (tensor<f32>) -> tensor<f32>
            "tf_executor.yield"(%v) : (tensor<f32>) -> ()
          }) : () -> (tensor<f32>, !tf_executor.control)
          "tf_executor.fetch"(%i#0) : (tensor<f32>) -> ()
        }) : () -> tensor<f32>
        func.return %g : tensor<f32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  mlir::Operation* island = findOp(*module, "tf_executor.island");
  mlir::Operation* fetch = findOp(*module, "tf_executor.fetch");
  PB2_CHECK(island && fetch);
  // Two data operands + one control on a one-result graph: the ODS-illegal
  // intermediate the rewrite paths can produce, verified off.
  fetch->setOperands({island->getResult(0), island->getResult(0),
                      island->getResult(1)});
  mlir::PassManager pm(&ctx, "builtin.module");
  pm.enableVerifier(false);  // results-vs-fetch mismatch is the point
  pm.addPass(pb_passes::CreateFlattenExecutorPass());
  PB2_CHECK(mlir::succeeded(pm.run(module.get())));
  PB2_CHECK(findOp(*module, "tf_executor.graph") != nullptr);  // bailed
  PB2_CHECK(fetch->getNumOperands() == 3);  // control edge not stripped
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
  TestFlattenExecutorInterleavedDominance();
  TestFlattenExecutorIslandControlEdges();
  TestFlattenExecutorArityBailsBeforeMutation();
  TestWrapRegisteredResidue();
  TestRenameResidueCarrier();
  PB2_TEST_SUMMARY("mlir-passes-test");
}
