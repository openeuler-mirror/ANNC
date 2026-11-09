#include "graphdef_rewriter.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "fusion_meta.h"
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

TEST(GraphDefRewriter, SkipsAliasForConsumedOutputs) {
  tensorflow::GraphDef original;
  makeSampleGraph(original);
  auto *rel = original.add_node();
  rel->set_name("rel");
  rel->set_op("Relu");
  rel->add_input("m:0");
  auto f = sampleFusion();
  ANNCFusedNodeBuilder::FusionOutputMap map{{"m:0", {&f, 0}}};
  ANNCFusedNodeBuilder builder(original, "/tmp/libk.so", "", map);
  tensorflow::GraphDef out;
  std::string error;
  ASSERT_TRUE(builder.appendOutputAliases(out, &error)) << error;
  EXPECT_EQ(out.node_size(), 0);  // m still has consumers: no alias
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

}  // namespace annc::converter_v2
