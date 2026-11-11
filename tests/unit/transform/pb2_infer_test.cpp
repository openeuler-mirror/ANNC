// Unit tests for the inference layer: GraphPassManager ordering, the
// dtype chain (explicit --input / graph-derived / backfill / enforce or
// assume), the synthetic-OpDef registry for unregistered custom ops.

#include "pb2_test_harness.h"

#include "common/pb2_common.h"
#include "graph/pb2_graph_pass.h"

namespace {

// The inference layer works on the process-global node index; each case
// rebuilds it from scratch.
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
                           std::vector<std::string> inputs,
                           int n_outputs = 1) {
    tensorflow::NodeDef* nd = graph.add_node();
    nd->set_name(name);
    nd->set_op(op);
    for (auto& in : inputs) *nd->add_input() = in;
    auto& ni = g_nodes[name];
    ni.def = nd;
    ni.out_dtypes.assign(n_outputs, tensorflow::DT_INVALID);
    g_order.push_back(name);
    return nd;
  }
  void index() {
    for (auto& nd : graph.node()) g_nodes[nd.name()].def = &nd;
  }
};

static tensorflow::NodeDef* SetType(tensorflow::NodeDef* nd, const char* key,
                                    tensorflow::DataType dt) {
  (*nd->mutable_attr())[key].set_type(dt);
  return nd;
}

static void TestGraphPassManager() {
  int order = 0, first = 0, second = 0;
  GraphPassManager mgr;
  mgr.addPass(makeGraphPass("a", [&](const tensorflow::GraphDef&, BuildCtx&) {
    first = ++order;
    return 0;
  }));
  mgr.addPass(makeGraphPass("b", [&](const tensorflow::GraphDef&, BuildCtx&) {
    second = ++order;
    return 7;
  }));
  mgr.addPass(makeGraphPass("unreached", [&](const tensorflow::GraphDef&, BuildCtx&) {
    ++order;  // must not run: short-circuit on rc=7
    return 0;
  }));
  tensorflow::GraphDef g;
  BuildCtx ctx;
  PB2_CHECK(mgr.run(g, ctx) == 7);
  PB2_CHECK(first == 1);
  PB2_CHECK(second == 2);
  PB2_CHECK(order == 2);
}

static void TestDtypeChain() {
  Fixture f;
  // Minimal MatMul OpDef: the dtype chain needs output_arg.type_attr = "T"
  // to reach the explicit node attr (g_opspecs is per-process global).
  tensorflow::OpDef& spec = g_opspecs["MatMul"];
  spec.set_name("MatMul");
  for (const char* in : {"a", "b"}) {
    auto* ia = spec.add_input_arg();
    ia->set_name(in);
    ia->set_type_attr("T");
  }
  auto* oa = spec.add_output_arg();
  oa->set_name("product");
  oa->set_type_attr("T");
  // p(Placeholder) -> m(MatMul, T=DT_FLOAT): the explicit T attr decides.
  f.add("p", "Placeholder", {});
  SetType(f.add("m", "MatMul", {"p", "p"}), "T", tensorflow::DT_FLOAT);
  f.index();
  Options opt;
  opt.inputs = {"p:DT_INT32:2,3"};
  BuildCtx ctx;
  InferEntryAndDtypes(opt, ctx);
  PB2_CHECK(ctx.entries.count("p") == 1);
  PB2_CHECK(ctx.entries.at("p").dt == tensorflow::DT_INT32);
  PB2_CHECK(ctx.entries.at("p").has_shape);
  PB2_CHECK(g_nodes.at("m").out_dtypes[0] == tensorflow::DT_FLOAT);
  PB2_CHECK(g_nodes.at("p").out_dtypes[0] == tensorflow::DT_INT32);  // override wins
}

static void TestUndeterminableIsFatal() {
  // No T attr, no OpDef entry, no downstream constraint: DT_INVALID must
  // stay and EnforceOrAssumeDtypes exits.
  PB2_CHECK_FATAL_EXIT({
    Fixture f;
    f.add("x", "MysteryOp", {});
    f.index();
    Options opt;
    BuildCtx ctx;
    InferEntryAndDtypes(opt, ctx);
  }, kExitGraphStructure);
}

static void TestAssumeOpaqueI8() {
  Fixture f;
  f.add("x", "MysteryOp", {});
  f.index();
  Options opt;
  opt.assume_opaque_i8 = true;
  BuildCtx ctx;
  InferEntryAndDtypes(opt, ctx);
  PB2_CHECK(g_nodes.at("x").out_dtypes[0] == tensorflow::DT_INT8);
  PB2_CHECK(g_assumed_dtypes.count("x") == 1);
  PB2_CHECK(g_assumed_dtypes.at("x") == std::vector<int>({0}));
}

static void TestSynthOpRegistry() {
  std::map<std::string, tensorflow::OpDef> synth;
  tensorflow::OpDef d;
  d.set_name("MyCustomOp");
  synth["MyCustomOp"] = d;
  SynthOpRegistry reg(synth);
  const tensorflow::OpRegistrationData* rd = nullptr;
  PB2_CHECK(reg.LookUp("MatMul", &rd).ok());          // global registry passthrough
  PB2_CHECK(rd != nullptr);
  PB2_CHECK(reg.LookUp("MyCustomOp", &rd).ok());      // synthetic fallback
  PB2_CHECK(rd->op_def.name() == "MyCustomOp");
  PB2_CHECK(!reg.LookUp("NeverRegistered", &rd).ok());
}

static void TestBuildUnregisteredOpDefs() {
  Fixture f;
  f.add("data", "Placeholder", {});
  tensorflow::NodeDef* custom =
      f.add("k", "MyCustomOp", {"data", "data:1"}, 1);
  f.index();
  // The second input references slot 1 -> the producer needs 2 dtype slots
  // tracked (actual count comes from the node index; slot count from refs).
  g_nodes.at("data").out_dtypes.assign(2, tensorflow::DT_FLOAT);
  std::map<std::string, tensorflow::OpDef> synth;
  tensorflow::GraphDef injected;
  BuildUnregisteredOpDefs(f.graph, synth, &injected);
  PB2_CHECK(synth.count("MyCustomOp") == 1);
  const tensorflow::OpDef& od = synth.at("MyCustomOp");
  // Variadic wildcard input: one arg keyed by a list(type); outputs stay
  // per-slot. The node states its own input count via the injected list.
  PB2_CHECK(od.input_arg_size() == 1);
  PB2_CHECK(od.input_arg(0).type_list_attr() == "tl");
  PB2_CHECK(od.output_arg_size() == 1);
  PB2_CHECK(od.output_arg(0).type_attr() == "o0");
  PB2_CHECK(g_synth_attr_keys.at("MyCustomOp") ==
        std::set<std::string>({"tl", "o0"}));
  PB2_CHECK(injected.node(1).attr().at("tl").list().type_size() == 2);
  PB2_CHECK(injected.node(1).attr().at("tl").list().type(0) ==
            tensorflow::DT_FLOAT);
  (void)custom;
}

}  // namespace

int main() {
  TestGraphPassManager();
  TestDtypeChain();
  TestUndeterminableIsFatal();
  TestAssumeOpaqueI8();
  TestSynthOpRegistry();
  TestBuildUnregisteredOpDefs();
  PB2_TEST_SUMMARY("infer-test");
}
