#include "graphdef_rewriter.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "fusion_meta.h"
#include <google/protobuf/text_format.h>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace annc::converter_v2 {
namespace {

// p(placeholder) -> m(matmul[p, c]) with c a const weight.
void makeSampleGraph(tensorflow::GraphDef &graph) {
  auto *p = graph.add_node();
  p->set_name("p");
  p->set_op("Placeholder");
  (*p->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);

  auto *c = graph.add_node();
  c->set_name("c");
  c->set_op("Const");
  (*c->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);

  auto *m = graph.add_node();
  m->set_name("m");
  m->set_op("MatMul");
  m->add_input("p:0");
  m->add_input("c:0");
}

FusionArg makeArg(const char *role, const char *tfName,
                  std::vector<int64_t> shape, int64_t rank) {
  FusionArg arg;
  arg.role = role;
  arg.tfName = tfName;
  arg.shape = std::move(shape);
  arg.rank = rank;
  arg.dtype = "float32";
  return arg;
}

FusionInfo sampleFusion() {
  FusionInfo f;
  f.name = "f0";
  f.pattern = "matmul_bias";
  f.kernelName = "k0";
  f.executionMode = "aot";
  f.args = {makeArg("dynamic", "p:0", {2}, 1),
            makeArg("fixed", "c:0", {2, 2}, 2)};
  f.outputs = {makeArg("output", "m:0", {2, 2}, 2)};
  return f;
}

TEST(GraphDefRewriter, RewritesDataInputs) {
  auto f = sampleFusion();
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(tensorflow::GraphDef{}, "/tmp/libk.so", "", map);
  EXPECT_EQ(builder.rewriteDataInput("m:0"), "f0:0");
  EXPECT_EQ(builder.rewriteDataInput("m"), "f0:0");  // port omitted
  EXPECT_EQ(builder.rewriteDataInput("other:0"), "other:0");
  EXPECT_EQ(builder.rewriteDataInput("^m"), "^m");  // control edge kept
}

TEST(GraphDefRewriter, ReplacesOutputNodes) {
  auto f = sampleFusion();
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(tensorflow::GraphDef{}, "/tmp/libk.so", "", map);
  EXPECT_TRUE(builder.replacesOutputNode("m"));
  EXPECT_FALSE(builder.replacesOutputNode("other"));
}

TEST(GraphDefRewriter, AppendsANNCFusedNode) {
  tensorflow::GraphDef graph;
  makeSampleGraph(graph);
  auto f = sampleFusion();
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(graph, "/tmp/libk.so", "", map);
  tensorflow::NodeDef *fused = builder.appendNode(graph, f);
  ASSERT_NE(fused, nullptr);
  EXPECT_EQ(fused->name(), "f0");
  EXPECT_EQ(fused->op(), "ANNCFused");
  // Role-grouped transport order: fixed before dynamic, then outputs.
  ASSERT_EQ(fused->input_size(), 2);
  EXPECT_EQ(fused->input(0), "c:0");
  EXPECT_EQ(fused->input(1), "p:0");
  EXPECT_EQ(fused->attr().at("kernel_name").s(), "k0");
  EXPECT_EQ(fused->attr().at("num_outputs").i(), 1);
  EXPECT_EQ(fused->attr().at("abi").s(), "mlir_ciface");
  EXPECT_EQ(fused->attr().at("shared_lib_path").s(), "/tmp/libk.so");
}

}  // namespace
TEST(GraphDefRewriter, AppendsKernelArgOrderDerivation) {
  tensorflow::GraphDef graph;
  makeSampleGraph(graph);
  auto f = sampleFusion();
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(graph, "/tmp/libk.so", "", map);
  tensorflow::NodeDef *fused = builder.appendNode(graph, f);
  ASSERT_NE(fused, nullptr);
  // Pattern order [p, c] becomes transport order [c, p]; the C-interface
  // permutation maps back to pattern order, then appends the output buffer.
  const auto &order = fused->attr().at("kernel_arg_order").list().i();
  ASSERT_EQ(order.size(), 3);
  EXPECT_EQ(order[0], 1);  // p is transport index 1
  EXPECT_EQ(order[1], 0);  // c is transport index 0
  EXPECT_EQ(order[2], 2);  // output buffer appended after inputs
}

TEST(GraphDefRewriter, AppendsContextV3Node) {
  tensorflow::GraphDef graph;
  makeSampleGraph(graph);
  auto f = sampleFusion();
  f.abi = "annc_context_v3";
  f.contextI64Attrs = {5, 6};
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(graph, "/tmp/libk.so", "", map);
  tensorflow::NodeDef *fused = builder.appendNode(graph, f);
  ASSERT_NE(fused, nullptr);
  ASSERT_EQ(fused->input_size(), 2);
  EXPECT_EQ(fused->input(0), "c:0");
  EXPECT_EQ(fused->input(1), "p:0");
  // canonical [p, c] sits at transport [1, 0].
  const auto &inv = fused->attr().at("context_input_order").list().i();
  ASSERT_EQ(inv.size(), 2);
  EXPECT_EQ(inv[0], 1);
  EXPECT_EQ(inv[1], 0);
  // kernel_arg_order is emitted empty for context v3 (no override).
  EXPECT_EQ(fused->attr().at("kernel_arg_order").list().i_size(), 0);
  const auto &ctx = fused->attr().at("context_i64_attrs").list().i();
  ASSERT_EQ(ctx.size(), 2);
  EXPECT_EQ(ctx[0], 5);
  EXPECT_EQ(ctx[1], 6);
}

TEST(GraphDefRewriter, AppendsJitNode) {
  tensorflow::GraphDef graph;
  makeSampleGraph(graph);
  auto f = sampleFusion();
  f.executionMode = "jit";
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(graph, "/tmp/libk.so", "/tmp/f.mlir", map);
  tensorflow::NodeDef *fused = builder.appendNode(graph, f);
  ASSERT_NE(fused, nullptr);
  EXPECT_EQ(fused->attr().at("shared_lib_path").s(), "");
  EXPECT_EQ(fused->attr().at("atir_module_path").s(), "/tmp/f.mlir");
}

TEST(GraphDefRewriter, AppendNodeRejectsEmptyOutputs) {
  tensorflow::GraphDef graph;
  makeSampleGraph(graph);
  auto f = sampleFusion();
  f.outputs.clear();
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(graph, "/tmp/libk.so", "", map);
  EXPECT_EQ(builder.appendNode(graph, f), nullptr);
}

TEST(GraphDefRewriter, AppendsOutputAliasIdentity) {
  tensorflow::GraphDef original;
  makeSampleGraph(original);
  auto f = sampleFusion();
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(original, "/tmp/libk.so", "", map);
  tensorflow::GraphDef out;
  std::string error;
  ASSERT_TRUE(builder.appendOutputAliases(out, &error)) << error;
  ASSERT_EQ(out.node_size(), 1);
  const auto &alias = out.node(0);
  EXPECT_EQ(alias.name(), "m");
  EXPECT_EQ(alias.op(), "Identity");
  ASSERT_EQ(alias.input_size(), 1);
  EXPECT_EQ(alias.input(0), "f0:0");
  EXPECT_EQ(alias.attr().at("T").type(), tensorflow::DT_FLOAT);
}

TEST(GraphDefRewriter, AppendsOutputAliasIdentityN) {
  tensorflow::GraphDef original;
  makeSampleGraph(original);
  auto *shapes = (*original.mutable_node(2)->mutable_attr())
                     ["_output_shapes"].mutable_list();
  shapes->add_shape();
  shapes->add_shape();
  auto f = sampleFusion();
  FusionArg second = f.outputs[0];
  second.tfName = "m:1";
  f.outputs.push_back(second);
  ANNCFusedNodeBuilder::FusionOutputMap map;
  map["m:0"] = {&f, 0};
  map["m:1"] = {&f, 1};
  ANNCFusedNodeBuilder builder(original, "/tmp/libk.so", "", map);
  tensorflow::GraphDef out;
  std::string error;
  ASSERT_TRUE(builder.appendOutputAliases(out, &error)) << error;
  ASSERT_EQ(out.node_size(), 1);
  const auto &alias = out.node(0);
  EXPECT_EQ(alias.op(), "IdentityN");
  ASSERT_EQ(alias.input_size(), 2);
  EXPECT_EQ(alias.input(0), "f0:0");
  EXPECT_EQ(alias.input(1), "f0:1");
  EXPECT_EQ(alias.attr().at("T").list().type_size(), 2);
}

TEST(GraphDefRewriter, AliasChecksConsumersInThePassedGraph) {
  tensorflow::GraphDef original;
  makeSampleGraph(original);
  auto *rel = original.add_node();
  rel->set_name("rel");
  rel->set_op("Relu");
  rel->add_input("m:0");
  auto f = sampleFusion();
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(original, "/tmp/libk.so", "", map);
  // The caller passes the rewritten graph, where rel's input is already
  // rerouted to the fused node: m has no consumers left and keeps its alias.
  tensorflow::GraphDef rewritten;
  auto *kept = rewritten.add_node();
  kept->set_name("rel");
  kept->set_op("Relu");
  kept->add_input("f0:0");
  std::string error;
  ASSERT_TRUE(builder.appendOutputAliases(rewritten, &error)) << error;
  ASSERT_EQ(rewritten.node_size(), 2);
  EXPECT_EQ(rewritten.node(1).name(), "m");
  EXPECT_EQ(rewritten.node(1).op(), "Identity");
  EXPECT_EQ(rewritten.node(1).input(0), "f0:0");
}

TEST(GraphDefRewriter, RejectsInvalidAliases) {
  auto assert_reason = [](const std::string &reason, const char *needle) {
    EXPECT_NE(reason.find(needle), std::string::npos) << reason;
  };
  auto declared_shapes = [](tensorflow::GraphDef &original, int count) {
    auto *shapes = (*original.mutable_node(2)->mutable_attr())
                       ["_output_shapes"].mutable_list();
    for (int i = 0; i < count; ++i) shapes->add_shape();
  };
  // Runs one alias pass against a sample graph with `count` declared output
  // shapes; returns "" on success or the rejection reason.
  auto run_alias_case = [&](const FusionInfo &fusion,
                            const ANNCFusedNodeBuilder::FusionOutputMap &map,
                            int count, int *outNodes = nullptr) {
    tensorflow::GraphDef original;
    makeSampleGraph(original);
    declared_shapes(original, count);
    ANNCFusedNodeBuilder builder(original, "", "", map);
    tensorflow::GraphDef out;
    std::string error;
    if (!builder.appendOutputAliases(out, &error)) {
      if (outNodes) *outNodes = out.node_size();
      return error;
    }
    return std::string();
  };

  {  // partial slot coverage against the declared output count
    auto f = sampleFusion();
    ANNCFusedNodeBuilder::FusionOutputMap map;
    map["m:0"] = {&f, 0};
    assert_reason(run_alias_case(f, map, 2), "must replace all 2");
  }
  {  // non-contiguous ports
    auto f = sampleFusion();
    ANNCFusedNodeBuilder::FusionOutputMap map;
    map["m:0"] = {&f, 0};
    map["m:1"] = {&f, 0};
    map["m:3"] = {&f, 0};  // port 2 missing: full count, gapped ports
    assert_reason(run_alias_case(f, map, 3), "contiguous");
  }
  {  // fusion output slot out of range
    auto f = sampleFusion();
    ANNCFusedNodeBuilder::FusionOutputMap map;
    map["m:0"] = {&f, 5};
    int nodes = 0;
    assert_reason(run_alias_case(f, map, 1, &nodes),
                  "invalid fusion output slot");
    EXPECT_EQ(nodes, 0);  // out is empty again: the alias was rolled back
  }
  {  // fusion node name collides with the alias
    auto f = sampleFusion();
    f.name = "m";
    ANNCFusedNodeBuilder::FusionOutputMap map;
    map["m:0"] = {&f, 0};
    int nodes = 0;
    assert_reason(run_alias_case(f, map, 1, &nodes), "conflicts");
    EXPECT_EQ(nodes, 0);  // out is empty again: the alias was rolled back
  }
  {  // unsupported output dtype
    auto f = sampleFusion();
    f.outputs[0].dtype = "bogus";
    ANNCFusedNodeBuilder::FusionOutputMap map;
    map["m:0"] = {&f, 0};
    int nodes = 0;
    assert_reason(run_alias_case(f, map, 1, &nodes), "unsupported dtype");
    EXPECT_EQ(nodes, 0);  // out is empty again: the alias was rolled back
  }
}

TEST(GraphDefRewriter, AppendsKernelArgOrderWithDuplicateInputs) {
  // Two args share tfName (MatMul(x, x)): the permutation is index-based, so
  // both inputs stay distinct — [0, 1] plus the output buffer.
  tensorflow::GraphDef graph;
  makeSampleGraph(graph);
  FusionInfo f;
  f.name = "f0";
  f.pattern = "matmul";
  f.kernelName = "k0";
  f.abi = "mlir_ciface";
  f.args.push_back({"dynamic", "x", {2}, 1, "float32"});
  f.args.push_back({"dynamic", "x", {2}, 1, "float32"});
  f.outputs.push_back({"output", "m:0", {2}, 1, "float32"});
  ANNCFusedNodeBuilder::FusionOutputMap map;  // kept alive: builder holds a ref
  ANNCFusedNodeBuilder builder(graph, "", "", map);
  tensorflow::NodeDef *fused = builder.appendNode(graph, f);
  ASSERT_NE(fused, nullptr);
  const auto &order = fused->attr().at("kernel_arg_order").list().i();
  ASSERT_EQ(order.size(), 3);
  EXPECT_EQ(order[0], 0);
  EXPECT_EQ(order[1], 1);
  EXPECT_EQ(order[2], 2);
}

TEST(GraphDefRewriter, ResolvesInputsAgainstOriginalGraph) {
  // A synthetic tfName missing from the graph whose slash variant exists in
  // `original` must resolve there — even though the caller appends into a
  // different output graph (the CLI flow shape); an out-based lookup would
  // miss and keep the dangling name.
  tensorflow::GraphDef original;
  makeSampleGraph(original);
  auto *xb = original.add_node();
  xb->set_name("x/begin");
  xb->set_op("Const");
  (*xb->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);

  FusionInfo f = sampleFusion();
  f.args = {makeArg("fixed", "c:0", {2, 2}, 2),
            makeArg("dynamic", "x_begin", {2}, 1)};
  ANNCFusedNodeBuilder::FusionOutputMap map;
  ANNCFusedNodeBuilder builder(original, "", "", map);
  tensorflow::GraphDef out;
  tensorflow::NodeDef *fused = builder.appendNode(out, f);
  ASSERT_NE(fused, nullptr);
  ASSERT_EQ(fused->input_size(), 2);
  EXPECT_EQ(fused->input(0), "c:0");      // fixed first (transport order)
  EXPECT_EQ(fused->input(1), "x/begin");  // slash variant resolved in original
}

TEST(GraphDefRewriter, RejectsUnknownRole) {
  // A mistyped role would be silently dropped by role grouping; the builder
  // must refuse the emission instead.
  tensorflow::GraphDef graph;
  makeSampleGraph(graph);
  FusionInfo f;
  f.name = "f0";
  f.pattern = "matmul";
  f.kernelName = "k0";
  f.abi = "mlir_ciface";
  f.args.push_back({"dynamic", "p", {2}, 1, "float32"});
  f.args.push_back({"dynmaic", "q", {2}, 1, "float32"});
  f.outputs.push_back({"output", "m:0", {2}, 1, "float32"});
  ANNCFusedNodeBuilder::FusionOutputMap map;  // kept alive: builder holds a ref
  ANNCFusedNodeBuilder builder(graph, "", "", map);
  std::string error;
  EXPECT_EQ(builder.appendNode(graph, f, &error), nullptr);
  EXPECT_NE(error.find("unknown role"), std::string::npos) << error;
}

// ---------------------------------------------------------------------------
// File-level rewrite: temp input graph + FusionInfo -> rewrite -> read back.

// Per-process temp dir: concurrent ctest runs never share files.
const std::string &tempDir() {
  static const std::string dir =
      (std::filesystem::temp_directory_path() /
       ("cv2_fl_" + std::to_string(::getpid())))
          .string();
  std::filesystem::create_directories(dir);
  return dir;
}

std::string writeFile(const std::string &path, const std::string &content,
                      bool binary) {
  std::ofstream out(path, binary ? std::ios::binary : std::ios::trunc);
  out << content;
  return path;
}

const char *kSampleTextGraph =
    R"pb(node { name: "p" op: "Placeholder" attr { key: "dtype" value { type: DT_FLOAT } } }
node { name: "c" op: "Const" attr { key: "dtype" value { type: DT_FLOAT } } }
node { name: "m" op: "MatMul" input: "p:0" input: "c:0" attr { key: "T" value { type: DT_FLOAT } } })pb";

// Text sample plus a consumer node of the fused output (cycle/alias cases).
std::string consumerTextGraph() {
  return std::string(kSampleTextGraph) +
         "\nnode { name: \"r\" op: \"Relu\" input: \"m:0\" attr { key: \"T\" "
         "value { type: DT_FLOAT } } }";
}

FusionInfo siteFusion(const char *name = "f0", const char *output = "m:0") {
  FusionInfo f;
  f.name = name;
  f.pattern = "matmul_bias";
  f.kernelName = "k0";
  f.executionMode = "aot";
  f.args.push_back({"dynamic", "p", {2}, 1, "float32"});
  f.args.push_back({"constant", "c", {2}, 1, "float32"});
  f.outputs.push_back({"output", output, {2}, 1, "float32"});
  return f;
}

GraphDefRewriteOptions fileOptions(const std::string &in,
                                   const std::string &out) {
  GraphDefRewriteOptions o;
  o.inputGraphPath = in;
  o.outputGraphPath = out;
  o.sharedLibPath = "/tmp/libgenerated.so";
  return o;
}

tensorflow::GraphDef readGraph(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  tensorflow::GraphDef graph;
  EXPECT_TRUE(graph.ParseFromString(data)) << path;
  return graph;
}

const tensorflow::NodeDef *find(const tensorflow::GraphDef &graph,
                                const char *name) {
  for (const auto &node : graph.node())
    if (node.name() == name) return &node;
  return nullptr;
}

TEST(GraphDefRewriterFile, RewritesGraphFiles) {
  const std::string &dir = tempDir();
  auto in = writeFile(dir + "/e2e.pbtxt", kSampleTextGraph, false);
  auto out = dir + "/e2e.pb";
  ASSERT_TRUE(
      rewriteGraphDefWithANNCFused({siteFusion()}, fileOptions(in, out)));
  auto graph = readGraph(out);
  const auto *fused = find(graph, "f0");
  ASSERT_NE(fused, nullptr);
  EXPECT_EQ(fused->op(), "ANNCFused");
  // Constant-role inputs are grouped first: c before p (non-fused inputs
  // keep the sample graph's port-less spelling).
  EXPECT_EQ(fused->input(0), "c");
  EXPECT_EQ(fused->input(1), "p");
  const auto *alias = find(graph, "m");
  ASSERT_NE(alias, nullptr);
  EXPECT_EQ(alias->op(), "Identity");
  EXPECT_EQ(alias->input(0), "f0:0");
  EXPECT_NE(find(graph, "p"), nullptr);

  // The same rewrite round-trips a binary input.
  tensorflow::GraphDef sample;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(kSampleTextGraph, &sample));
  auto bin = writeFile(dir + "/e2e.pb", sample.SerializeAsString(), true);
  auto binOut = dir + "/e2e_bin_out.pb";
  ASSERT_TRUE(rewriteGraphDefWithANNCFused({siteFusion()},
                                           fileOptions(bin, binOut)));
  EXPECT_NE(find(readGraph(binOut), "f0"), nullptr);
}

TEST(GraphDefRewriterFile, DropsUnresolvableSites) {
  const std::string &dir = tempDir();
  // Runs one rewrite with a single fusion; returns the output graph.
  auto rewrite = [&](const FusionInfo &fusion, const char *stem,
                     const std::string &text) {
    auto in = writeFile(dir + "/" + stem + ".pbtxt", text, false);
    auto out = dir + "/" + stem + ".pb";
    EXPECT_TRUE(rewriteGraphDefWithANNCFused({fusion}, fileOptions(in, out)));
    return readGraph(out);
  };
  {
    auto graph = rewrite(siteFusion("f0", "ghost:0"), "drop_out",
                         kSampleTextGraph);
    EXPECT_EQ(find(graph, "f0"), nullptr);  // dropped whole: original kept
    EXPECT_EQ(graph.node_size(), 3);
  }
  {
    auto fusion = siteFusion();
    fusion.args.push_back({"dynamic", "ghost", {2}, 1, "float32"});
    auto graph = rewrite(fusion, "drop_in", kSampleTextGraph);
    EXPECT_EQ(find(graph, "f0"), nullptr);
    EXPECT_EQ(graph.node_size(), 3);
  }
  {
    auto fusion = siteFusion();
    fusion.args.push_back({"dynamic", "m:0", {2}, 1, "float32"});
    auto graph = rewrite(fusion, "drop_self", kSampleTextGraph);
    EXPECT_EQ(find(graph, "f0"), nullptr);  // self-cycle dropped
    EXPECT_EQ(graph.node_size(), 3);
  }
  {
    // A control edge pointing at the replaced node would dangle after the
    // node is removed: the site is dropped instead of emitting an invalid
    // graph.
    auto text = std::string(kSampleTextGraph) +
                "\nnode { name: \"w\" op: \"NoOp\" input: \"^m\" }";
    auto graph = rewrite(siteFusion(), "drop_ctrl", text);
    EXPECT_EQ(find(graph, "f0"), nullptr);
    EXPECT_EQ(graph.node_size(), 4);
  }
}

TEST(GraphDefRewriterFile, RejectsInvalidRequests) {
  const std::string &dir = tempDir();
  {  // duplicate fusion names would emit two same-named nodes
    auto in = writeFile(dir + "/dup.pbtxt", kSampleTextGraph, false);
    ASSERT_FALSE(rewriteGraphDefWithANNCFused(
        {siteFusion("f0"), siteFusion("f0")},
        fileOptions(in, dir + "/dup.pb")));
  }
  {  // unwritable output path
    auto in = writeFile(dir + "/ro.pbtxt", kSampleTextGraph, false);
    ASSERT_FALSE(rewriteGraphDefWithANNCFused(
        {siteFusion()}, fileOptions(in, dir + "/no_such_dir/ro.pb")));
  }
  {  // AOT needs --shared_lib_path; JIT needs --atir_module_path
    auto in = writeFile(dir + "/mode.pbtxt", kSampleTextGraph, false);
    auto options = fileOptions(in, dir + "/mode.pb");
    options.sharedLibPath.clear();
    ASSERT_FALSE(rewriteGraphDefWithANNCFused({siteFusion()}, options));
    auto jit = siteFusion();
    jit.executionMode = "jit";
    ASSERT_FALSE(rewriteGraphDefWithANNCFused({jit}, options));
  }
  {  // --kernel_name override is single-site only
    auto in = writeFile(dir + "/kn.pbtxt", consumerTextGraph(), false);
    auto options = fileOptions(in, dir + "/kn.pb");
    options.kernelNameOverride = "k9";
    ASSERT_FALSE(rewriteGraphDefWithANNCFused(
        {siteFusion(), siteFusion("f1", "r:0")}, options));
  }
}

TEST(GraphDefRewriterFile, DropsCrossSiteCycle) {
  const std::string &dir = tempDir();
  auto in = writeFile(dir + "/cycle.pbtxt", consumerTextGraph(), false);
  auto out = dir + "/cycle.pb";
  auto f0 = siteFusion("f0", "m:0");
  f0.args.push_back({"dynamic", "r:0", {2}, 1, "float32"});  // wired to f1
  auto f1 = siteFusion("f1", "r:0");
  f1.args.clear();
  f1.args.push_back({"dynamic", "m:0", {2}, 1, "float32"});  // wired to f0
  ASSERT_TRUE(rewriteGraphDefWithANNCFused({f0, f1}, fileOptions(in, out)));
  auto graph = readGraph(out);
  EXPECT_EQ(find(graph, "f0"), nullptr);  // both cyclic sites dropped
  EXPECT_EQ(find(graph, "f1"), nullptr);
  EXPECT_EQ(graph.node_size(), 4);
}

TEST(GraphDefRewriterFile, KeepsAliasWhenConsumerIsAlsoReplaced) {
  const std::string &dir = tempDir();
  auto in = writeFile(dir + "/alias.pbtxt", consumerTextGraph(), false);
  auto out = dir + "/alias.pb";
  ASSERT_TRUE(rewriteGraphDefWithANNCFused({siteFusion(), siteFusion("f1", "r:0")},
                                           fileOptions(in, out)));
  auto graph = readGraph(out);
  // r (the only consumer of m) is replaced by f1: m keeps its alias anyway.
  const auto *alias = find(graph, "m");
  ASSERT_NE(alias, nullptr);
  EXPECT_EQ(alias->op(), "Identity");
  EXPECT_EQ(alias->input(0), "f0:0");
  EXPECT_NE(find(graph, "f1"), nullptr);
}

}  // namespace annc::converter_v2
