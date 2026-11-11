// Unit tests for the T5 restore + emit layer: the custom-op restore pass
// (verbatim rebuild from custom_ops, entry-function rewrite) and the
// weight-planning / attr dual-channel helpers. Same wheel-isolated domain
// as the other suites; quoted generic IR.

#include "pb2_test_harness.h"
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "common/pb2_common.h"
#include "passes/pb2_mlir_passes.h"

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

// Rebuilds the process-global node index around one fresh graph.
struct Fixture {
  tensorflow::GraphDef graph;
  Fixture() { reset(); }
  void reset() {
    g_nodes.clear();
    g_order.clear();
    g_opspecs.clear();
    g_assumed_dtypes.clear();
    g_synth_attr_keys.clear();
  }
  tensorflow::NodeDef* add(const std::string& name, const std::string& op,
                           std::vector<std::string> inputs) {
    tensorflow::NodeDef* nd = graph.add_node();
    nd->set_name(name);
    nd->set_op(op);
    for (auto& in : inputs) *nd->add_input() = in;
    NodeInfo& ni = g_nodes[name];
    ni.def = nd;
    ni.out_dtypes = {tensorflow::DT_FLOAT};
    g_order.push_back(name);
    return nd;
  }
};

}  // namespace

// A two-output custom op with slot stand-ins rebuilds as one tf.residue:
// identity attrs ride along, operands resolve to the feed arg, stand-ins
// vanish, and the entry contract is rewritten.
static void TestCustomOpRestore() {
  Fixture fx;
  NodeInfo& ni = g_nodes["myop"];
  ni.out_dtypes = {tensorflow::DT_FLOAT, tensorflow::DT_FLOAT};

  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %p0 = "some.ph"() {test.tag = "p0"} : () -> tensor<f32> loc("myop")
        %p1 = "some.ph"() {test.tag = "p1"} : () -> tensor<f32> loc("myop__1")
        func.return %p0 : tensor<f32>
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));

  CustomOpTable custom_ops;
  tensorflow::NodeDef& nd = custom_ops["myop"];
  nd.set_name("myop");
  nd.set_op("MyOp");
  *nd.add_input() = "feed:0";

  Options opt;
  opt.inputs = {"feed:0"};
  opt.outputs = {"out"};
  std::vector<FeedSpec> feeds = {FeedSpec{"feed"}};

  mlir::OpBuilder b(&ctx);
  PB2_CHECK(pb_passes::RunRestoreOpsPass(module.get(), opt, feeds,
                                         custom_ops, b, ctx) == 0);
  auto* res = findOp(*module, "tf.residue");
  PB2_CHECK(res != nullptr);
  PB2_CHECK(res->getAttrOfType<mlir::StringAttr>("call_target_name")
                .getValue() == "tf.MyOp");
  PB2_CHECK(res->getAttrOfType<mlir::StringAttr>("custom_op").getValue() ==
            "MyOp");
  PB2_CHECK(res->getAttrOfType<mlir::StringAttr>("residue_reason")
                .getValue() == "unregistered_custom_op");
  PB2_CHECK(res->getNumOperands() == 1);   // the feed arg
  PB2_CHECK(res->getNumResults() == 2);    // slot types from the stand-ins
  PB2_CHECK(findOp(*module, "some.ph", "p0") == nullptr);
  PB2_CHECK(findOp(*module, "some.ph", "p1") == nullptr);
  // return now consumes the rebuilt carrier's result 0.
  mlir::Operation* ret = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "func.return") ret = op;
  });
  PB2_CHECK(ret != nullptr &&
            ret->getOperand(0).getDefiningOp() == res);

  mlir::Operation* fn = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "func.func") fn = op;
  });
  auto ef = fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  PB2_CHECK(static_cast<bool>(ef));
  PB2_CHECK(llvm::cast<mlir::StringAttr>(ef.get("inputs")).getValue() == "feed:0");
  PB2_CHECK(llvm::cast<mlir::StringAttr>(ef.get("outputs")).getValue() == "out:0");
  auto an = fn->getAttrOfType<mlir::ArrayAttr>("tf.arg_names");
  PB2_CHECK(static_cast<bool>(an) && an.size() == 1 &&
            llvm::cast<mlir::StringAttr>(an[0]).getValue() == "feed");
}

// Two chained custom ops inside ONE branch function: the second references
// the first by its plain pb name. After the first rebuild, every alias of
// its stand-in must be gone from the name table (functionalize-stripped
// keys included) and the pb spelling must resolve to the new carrier --
// a stale alias would be a use-after-free.
static void TestBranchChainedCustomOps() {
  Fixture fx;
  NodeInfo& nx = g_nodes["X"];
  nx.out_dtypes = {tensorflow::DT_FLOAT};
  NodeInfo& ny = g_nodes["Y"];
  ny.out_dtypes = {tensorflow::DT_FLOAT};

  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main() {
        func.return
      }
      func.func @then_branch() {
        %s0 = "some.ph"() : () -> tensor<f32> loc("X@_functionalize_if_then_branch_0")
        %s1 = "some.ph"() : () -> tensor<f32> loc("Y@_functionalize_if_then_branch_0")
        func.return
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));

  CustomOpTable custom_ops;
  tensorflow::NodeDef& nx_nd = custom_ops["X@_functionalize_if_then_branch_0"];
  nx_nd.set_name("X");
  nx_nd.set_op("OpX");
  tensorflow::NodeDef& ny_nd = custom_ops["Y@_functionalize_if_then_branch_0"];
  ny_nd.set_name("Y");
  ny_nd.set_op("OpY");
  *ny_nd.add_input() = "X:0";  // plain pb name of the first op

  Options opt;
  std::vector<FeedSpec> feeds;
  mlir::OpBuilder b(&ctx);
  PB2_CHECK(pb_passes::RunRestoreOpsPass(module.get(), opt, feeds,
                                         custom_ops, b, ctx) == 0);
  mlir::Operation* cx = nullptr;
  mlir::Operation* cy = nullptr;
  module->walk([&](mlir::Operation* op) {
    auto ct = op->getAttrOfType<mlir::StringAttr>("call_target_name");
    if (!ct) return;
    if (ct.getValue() == "tf.OpX") cx = op;
    if (ct.getValue() == "tf.OpY") cy = op;
  });
  PB2_CHECK(cx != nullptr && cy != nullptr);
  PB2_CHECK(cx->getAttrOfType<mlir::StringAttr>("call_target_name")
                .getValue() == "tf.OpX");
  // The second carrier consumes the FIRST carrier's result -- not a stale
  // stand-in pointer (the pre-fix behavior was a dangling read here).
  PB2_CHECK(cy->getNumOperands() == 1);
  PB2_CHECK(cy->getOperand(0).getDefiningOp() == cx);
}

// tf.Case carries its branch functions as an ArrayAttr of symbol refs (not
// a single SymbolRefAttr like If/While); captures must bind through it.
static void TestCaseBranchCaptures() {
  Fixture fx;
  NodeInfo& nz = g_nodes["Z"];
  nz.out_dtypes = {tensorflow::DT_FLOAT};

  mlir::MLIRContext ctx;
  loadDialects(ctx);
  auto module = parse(ctx, R"mlir(
    module {
      func.func @main() {
        %src = "some.op"() {test.tag = "src"} : () -> tensor<f32> loc("src")
        func.return
      }
      func.func @case_branch(%cap: tensor<f32>) {
        %s0 = "some.ph"() {test.tag = "sz"} : () -> tensor<f32> loc("Z@_functionalize_case_branch_0")
        func.return
      }
    })mlir");
  PB2_CHECK(static_cast<bool>(module));
  // Attach the Case carrier manually (generic form): branch_index operand
  // occupies slot 0, so captures align from operand 1 (off = 1).
  mlir::OpBuilder b(&ctx);
  mlir::Operation* src = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getAttrOfType<mlir::StringAttr>("test.tag") &&
        op->getAttrOfType<mlir::StringAttr>("test.tag").getValue() == "src")
      src = op;
  });
  PB2_CHECK(src != nullptr);
  b.setInsertionPointAfter(src);
  mlir::OperationState st(mlir::UnknownLoc::get(&ctx), "tf.Case");
  // Slot 0 is the branch index (placeholder); captures align from operand 1.
  st.addOperands({src->getResult(0), src->getResult(0)});
  st.addAttribute("branches", b.getArrayAttr(
      {mlir::SymbolRefAttr::get(&ctx, "case_branch")}));
  b.create(st);

  CustomOpTable custom_ops;
  tensorflow::NodeDef& nz_nd = custom_ops["Z@_functionalize_case_branch_0"];
  nz_nd.set_name("Z");
  nz_nd.set_op("OpZ");
  *nz_nd.add_input() = "src:0";  // resolves through the capture binding

  Options opt;
  std::vector<FeedSpec> feeds;
  PB2_CHECK(pb_passes::RunRestoreOpsPass(module.get(), opt, feeds,
                                         custom_ops, b, ctx) == 0);
  mlir::Operation* cz = nullptr;
  module->walk([&](mlir::Operation* op) {
    if (op->getAttrOfType<mlir::StringAttr>("call_target_name") &&
        op->getAttrOfType<mlir::StringAttr>("call_target_name").getValue() ==
            "tf.OpZ")
      cz = op;
  });
  PB2_CHECK(cz != nullptr);          // resolved through the ArrayAttr capture
  PB2_CHECK(cz->getNumOperands() == 1);
  auto ba = llvm::dyn_cast<mlir::BlockArgument>(cz->getOperand(0));
  PB2_CHECK(static_cast<bool>(ba));  // bound to the branch function arg
}

// PlanWeights: representable consts stay, value-less ones argify; variable
// roots parametrize and the skip set propagates through the read chain.
static void TestPlanWeights() {
  Fixture fx;
  tensorflow::NodeDef* c_ok = fx.add("c_ok", "Const", {});
  (*c_ok->mutable_attr())["value"].mutable_tensor()->set_dtype(
      tensorflow::DT_FLOAT);
  fx.add("c_bad", "Const", {});  // no value attr: not representable
  fx.add("v", "VariableV2", {});
  fx.add("id1", "Identity", {"v"});
  fx.add("id2", "Identity", {"id1"});

  Options opt;
  mlir::MLIRContext ctx;
  mlir::Builder b(&ctx);
  BuildCtx bc;
  PlanWeights(opt, b, bc);
  PB2_CHECK(bc.argified.count("c_ok") == 0);
  PB2_CHECK(bc.argified.count("c_bad") == 1);
  PB2_CHECK(bc.var_arg.count("v") == 1);
  PB2_CHECK(bc.var_skip.count("v") == 1);
  PB2_CHECK(bc.var_skip.count("id1") == 1);
  PB2_CHECK(bc.var_skip.count("id2") == 1);
}

// The attr dual channel: MLIR-convertible attrs direct, the rest via the
// sorted custom_attrs JSON, plus the dtype-assumed markers.
static void TestAppendCustomOpAttrs() {
  Fixture fx;
  tensorflow::NodeDef nd;
  nd.set_name("myop");
  nd.set_op("MyOp");
  (*nd.mutable_attr())["x"].set_i(5);
  (*nd.mutable_attr())["fn"].mutable_func()->set_name("f0");
  g_assumed_dtypes["myop"] = {tensorflow::DT_FLOAT, tensorflow::DT_INT32};

  mlir::MLIRContext ctx;
  mlir::OpBuilder b(&ctx);
  mlir::OperationState st(mlir::UnknownLoc::get(&ctx), "tf.residue");
  AppendCustomOpAttrs(st, nd, b, "myop");
  PB2_CHECK(st.attributes.getDictionary(&ctx).size() == 4);  // x, custom_op, custom_attrs, dtype_assumed
  bool hasX = false, hasOp = false, hasJson = false, hasAssumed = false;
  std::string json;
  for (auto& na : st.attributes) {
    if (na.getName().str() == "x") hasX = true;
    if (na.getName().str() == "custom_op") hasOp = true;
    if (na.getName().str() == "custom_attrs") {
      hasJson = true;
      json = llvm::cast<mlir::StringAttr>(na.getValue()).getValue().str();
    }
    if (na.getName().str() == "dtype_assumed") hasAssumed = true;
  }
  PB2_CHECK(hasX && hasOp && hasJson && hasAssumed);
  PB2_CHECK(json.find("\"fn\"") != std::string::npos);  // JSON fallback
  PB2_CHECK(json.find("_dtype_assumed") != std::string::npos);
  g_assumed_dtypes.clear();
}

int main() {
  TestCustomOpRestore();
  TestPlanWeights();
  TestAppendCustomOpAttrs();
  TestBranchChainedCustomOps();
  TestCaseBranchCaptures();
  PB2_TEST_SUMMARY("t5-test");
}
