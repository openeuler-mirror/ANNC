// Unit tests for the ONNX frontend stages that do not need a model file on
// disk: the tensor decoder, the graph parser (including control-flow refusal),
// the type/shape resolver, and the NodeInfo adapter's fallback decisions.
//
// The tool's end-to-end behaviour is covered by tests/test_onnx_frontend.py,
// which drives the annc-onnx2atir binary on generated models.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "Builder/MLIROpBuilder.h"
#include "Dialect/Atir/AtirOps.h"
#include "onnx_frontend.h"
#include "onnx_mlir_builder.h"
#include "onnx_transformers.h"
#include "onnx_node_info_adapter.h"
#include "onnx_tensor_decoder.h"
#include "onnx_tensor_resolver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace annc::onnx2atir {
namespace {

using onnx::AttributeProto;
using onnx::GraphProto;
using onnx::ModelProto;
using onnx::NodeProto;
using onnx::TensorProto;
using onnx::TensorShapeProto;
using onnx::TypeProto;
using onnx::ValueInfoProto;

// -------------------------------------------------------------------------
// Model construction helpers
// -------------------------------------------------------------------------

void addDims(TensorProto& tensor, std::initializer_list<int64_t> dims) {
  for (int64_t dim : dims) tensor.add_dims(dim);
}

TensorProto* addFloatInitializer(GraphProto& graph, const std::string& name,
                                 std::initializer_list<int64_t> dims,
                                 std::initializer_list<float> values) {
  TensorProto* tensor = graph.add_initializer();
  tensor->set_name(name);
  tensor->set_data_type(TensorProto::FLOAT);
  addDims(*tensor, dims);
  for (float value : values) tensor->add_float_data(value);
  return tensor;
}

TensorProto* addInt64Initializer(GraphProto& graph, const std::string& name,
                                 std::initializer_list<int64_t> dims,
                                 std::initializer_list<int64_t> values) {
  TensorProto* tensor = graph.add_initializer();
  tensor->set_name(name);
  tensor->set_data_type(TensorProto::INT64);
  addDims(*tensor, dims);
  for (int64_t value : values) tensor->add_int64_data(value);
  return tensor;
}

void addTensorInput(GraphProto& graph, const std::string& name,
                    std::initializer_list<int64_t> dims) {
  ValueInfoProto* value = graph.add_input();
  value->set_name(name);
  TypeProto::Tensor* tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  for (int64_t dim : dims)
    tensor->mutable_shape()->add_dim()->set_dim_value(dim);
}

void addSymbolicInput(GraphProto& graph, const std::string& name,
                      std::initializer_list<int64_t> dims) {
  ValueInfoProto* value = graph.add_input();
  value->set_name(name);
  TypeProto::Tensor* tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  for (int64_t dim : dims) {
    TensorShapeProto::Dimension* d = tensor->mutable_shape()->add_dim();
    if (dim < 0)
      d->set_dim_param("N");
    else
      d->set_dim_value(dim);
  }
}

void addTensorOutput(GraphProto& graph, const std::string& name,
                     std::initializer_list<int64_t> dims,
                     int elem_type = TensorProto::FLOAT) {
  ValueInfoProto* value = graph.add_output();
  value->set_name(name);
  TypeProto::Tensor* tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(elem_type);
  for (int64_t dim : dims)
    tensor->mutable_shape()->add_dim()->set_dim_value(dim);
}

void addTypedInput(GraphProto& graph, const std::string& name, int elem_type,
                   std::initializer_list<int64_t> dims) {
  ValueInfoProto* value = graph.add_input();
  value->set_name(name);
  TypeProto::Tensor* tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(elem_type);
  for (int64_t dim : dims)
    tensor->mutable_shape()->add_dim()->set_dim_value(dim);
}

NodeProto* addNode(GraphProto& graph, const std::string& op_type,
                   const std::string& name,
                   std::initializer_list<std::string> inputs,
                   std::initializer_list<std::string> outputs) {
  NodeProto* node = graph.add_node();
  node->set_op_type(op_type);
  node->set_name(name);
  for (const std::string& input : inputs) node->add_input(input);
  for (const std::string& output : outputs) node->add_output(output);
  return node;
}

void setInt(NodeProto* node, const std::string& name, int64_t value) {
  AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::INT);
  attr->set_i(value);
}

void addInt64Attribute(NodeProto* node, const std::string& name,
                       std::initializer_list<int64_t> values) {
  AttributeProto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::INTS);
  for (int64_t value : values) attr->add_ints(value);
}

// Runs the four frontend stages over an in-memory graph.
struct PipelineResult {
  bool ok = false;
  std::string error;
  OnnxGraph graph;
  ResolvedOnnxGraph resolved;
  AdaptedGraph adapted;
};

PipelineResult runPipeline(GraphProto& graph,
                           const AdaptOptions& options = AdaptOptions()) {
  PipelineResult result;
  OnnxGraphParser parser;
  if (!parser.parse(graph, result.graph, result.error)) return result;
  OnnxTensorResolver resolver;
  if (!resolver.resolve(result.graph, result.resolved, result.error))
    return result;
  OnnxNodeInfoAdapter adapter;
  if (!adapter.adapt(result.graph, result.resolved, options, result.adapted,
                     result.error))
    return result;
  result.ok = true;
  return result;
}

const NodeProvenance* findProvenance(const AdaptedGraph& adapted,
                                     const std::string& node_name) {
  for (const NodeProvenance& entry : adapted.provenance)
    if (entry.node_name == node_name) return &entry;
  return nullptr;
}

// -------------------------------------------------------------------------
// Tensor decoder
// -------------------------------------------------------------------------

TEST(OnnxTensorDecoder, MapsOnnxElementTypesToAtirNames) {
  std::string name;
  EXPECT_TRUE(onnxDtypeName(TensorProto::FLOAT, name));
  EXPECT_EQ(name, "float32");
  EXPECT_TRUE(onnxDtypeName(TensorProto::INT64, name));
  EXPECT_EQ(name, "int64");
  EXPECT_TRUE(onnxDtypeName(TensorProto::BOOL, name));
  EXPECT_EQ(name, "bool");
  // float8/int4 have no ATIR tensor type; refusing them is the contract.
  EXPECT_FALSE(onnxDtypeName(TensorProto::FLOAT8E4M3FN, name));
  EXPECT_FALSE(onnxDtypeName(TensorProto::UNDEFINED, name));
}

TEST(OnnxTensorDecoder, DecodesTypedFieldsIntoHostLayout) {
  TensorProto tensor;
  tensor.set_data_type(TensorProto::INT64);
  addDims(tensor, {2});
  tensor.add_int64_data(7);
  tensor.add_int64_data(-9);

  std::vector<uint8_t> bytes;
  std::string error;
  ASSERT_TRUE(decodeTensorBytes(tensor, "int64", bytes, error)) << error;
  ASSERT_EQ(bytes.size(), 16u);
  const int64_t* values = reinterpret_cast<const int64_t*>(bytes.data());
  EXPECT_EQ(values[0], 7);
  EXPECT_EQ(values[1], -9);
}

TEST(OnnxTensorDecoder, NarrowsInt32DataForSubWordDtypes) {
  TensorProto tensor;
  tensor.set_data_type(TensorProto::BOOL);
  addDims(tensor, {3});
  tensor.add_int32_data(1);
  tensor.add_int32_data(0);
  tensor.add_int32_data(1);

  std::vector<uint8_t> bytes;
  std::string error;
  ASSERT_TRUE(decodeTensorBytes(tensor, "bool", bytes, error)) << error;
  ASSERT_EQ(bytes.size(), 3u);
  EXPECT_EQ(bytes[0], 1);
  EXPECT_EQ(bytes[1], 0);
  EXPECT_EQ(bytes[2], 1);
}

TEST(OnnxTensorDecoder, RejectsExternalData) {
  TensorProto tensor;
  tensor.set_data_type(TensorProto::FLOAT);
  addDims(tensor, {1});
  tensor.set_data_location(TensorProto::EXTERNAL);

  std::vector<uint8_t> bytes;
  std::string error;
  EXPECT_FALSE(decodeTensorBytes(tensor, "float32", bytes, error));
  EXPECT_NE(error.find("external"), std::string::npos);
}

TEST(OnnxTensorDecoder, RejectsTruncatedRawData) {
  // dims=[3] with two floats of raw_data: neither a full payload nor the
  // single element of a splat. The truncated buffer previously reached
  // DenseElementsAttr and aborted the process.
  TensorProto tensor;
  tensor.set_data_type(TensorProto::FLOAT);
  addDims(tensor, {3});
  tensor.set_raw_data(std::string(8, '\x00'));

  std::vector<uint8_t> bytes;
  std::string error;
  EXPECT_FALSE(decodeTensorBytes(tensor, "float32", bytes, error));
  EXPECT_NE(error.find("size mismatch"), std::string::npos);
}

TEST(OnnxTensorDecoder, RejectsSingleElementRawDataForLargerShape) {
  // dims=[4] with one float of raw_data: previously the payload was splatted
  // silently across the whole shape (four equal weights, exit code 0).
  TensorProto tensor;
  tensor.set_data_type(TensorProto::FLOAT);
  addDims(tensor, {4});
  tensor.set_raw_data(std::string(4, '\x00'));

  std::vector<uint8_t> bytes;
  std::string error;
  EXPECT_FALSE(decodeTensorBytes(tensor, "float32", bytes, error));
  EXPECT_NE(error.find("size mismatch"), std::string::npos);
}

// -------------------------------------------------------------------------
// Graph parser
// -------------------------------------------------------------------------

TEST(OnnxGraphParser, SortsNodesTopologically) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  // Declared in reverse dependency order on purpose.
  addNode(graph, "Relu", "relu", {"add_out"}, {"y"});
  addNode(graph, "Add", "add", {"x", "x"}, {"add_out"});
  addTensorOutput(graph, "y", {2, 3});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  ASSERT_TRUE(parser.parse(graph, parsed, error)) << error;
  ASSERT_EQ(parsed.nodes.size(), 2u);
  EXPECT_EQ(parsed.nodes[0].op_type, "Add");
  EXPECT_EQ(parsed.nodes[1].op_type, "Relu");
}

TEST(OnnxGraphParser, RejectsControlFlowModelWide) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  NodeProto* branch = addNode(graph, "If", "cond", {"x"}, {"y"});
  setInt(branch, "cond_attr", 0);
  AttributeProto* subgraph = branch->add_attribute();
  subgraph->set_name("then_branch");
  subgraph->set_type(AttributeProto::GRAPH);
  subgraph->mutable_g()->set_name("then");
  addTensorOutput(graph, "y", {2, 3});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  EXPECT_FALSE(parser.parse(graph, parsed, error));
  EXPECT_NE(error.find("control flow"), std::string::npos);
  EXPECT_NE(error.find("then_branch"), std::string::npos);
}

TEST(OnnxGraphParser, RejectsDependencyCycle) {
  GraphProto graph;
  addNode(graph, "Relu", "a", {"b_out"}, {"a_out"});
  addNode(graph, "Relu", "b", {"a_out"}, {"b_out"});
  addNode(graph, "Relu", "c", {"a_out"}, {"y"});
  addTensorOutput(graph, "y", {2});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  EXPECT_FALSE(parser.parse(graph, parsed, error));
  EXPECT_NE(error.find("cycle"), std::string::npos);
}

TEST(OnnxGraphParser, TreatsInitializerAlsoListedAsInputAsAConstant) {
  GraphProto graph;
  addTensorInput(graph, "w", {2});
  addFloatInitializer(graph, "w", {2}, {1.0f, 2.0f});
  addNode(graph, "Relu", "relu", {"w"}, {"y"});
  addTensorOutput(graph, "y", {2});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  ASSERT_TRUE(parser.parse(graph, parsed, error)) << error;
  EXPECT_TRUE(parsed.inputs.empty());
  EXPECT_EQ(parsed.initializers.size(), 1u);
}

TEST(OnnxGraphParser, RejectsZeroOutputNodeWithoutCrashing) {
  GraphProto graph;
  addTensorInput(graph, "x", {2});
  // No name, no op_type, no outputs: the old code read output(0) out of
  // bounds before ever reaching its own zero-output check.
  addNode(graph, "", "", {"x"}, {});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  EXPECT_FALSE(parser.parse(graph, parsed, error));
  EXPECT_NE(error.find("has no output"), std::string::npos);
}

TEST(OnnxGraphParser, RejectsEmptyOpTypeInsteadOfForgingIt) {
  GraphProto graph;
  addTensorInput(graph, "x", {2});
  // The old code substituted the output tensor name for the empty op_type,
  // and the provenance trail recorded an operator the model does not contain.
  addNode(graph, "", "bad", {"x"}, {"y"});
  addTensorOutput(graph, "y", {2});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  EXPECT_FALSE(parser.parse(graph, parsed, error));
  EXPECT_NE(error.find("op_type"), std::string::npos);
}

TEST(OnnxGraphParser, AcceptsSparseValuedAttributes) {
  // A sparse-tensor attribute is a payload form, not control flow: the model
  // must parse. The adapter reports the node as a per-node fallback.
  GraphProto graph;
  addTensorInput(graph, "x", {2});
  NodeProto* node = addNode(graph, "Constant", "sc", {}, {"c"});
  AttributeProto* sparse = node->add_attribute();
  sparse->set_name("sparse_value");
  sparse->set_type(AttributeProto::SPARSE_TENSOR);
  sparse->mutable_sparse_tensor()->mutable_values()->set_name("c_values");
  addTensorOutput(graph, "c", {2});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  EXPECT_TRUE(parser.parse(graph, parsed, error)) << error;
}

TEST(OnnxGraphParser, RejectsNodeConsumingItsOwnOutput) {
  // The edge is a self-loop, so the topological sort would never see it: the
  // node reaches indegree 0 and the failure only surfaced later, as a
  // misleading "unknown input tensor" from the builder.
  GraphProto graph;
  addNode(graph, "Relu", "loop", {"loop_out"}, {"loop_out"});
  addTensorOutput(graph, "loop_out", {2});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  EXPECT_FALSE(parser.parse(graph, parsed, error));
  EXPECT_NE(error.find("consumes its own output"), std::string::npos);
}

TEST(OnnxGraphParser, RejectsDuplicateInitializerName) {
  // Two values behind one tensor name would emit two constants under it while
  // the conservation accounting counts a single initializer.
  GraphProto graph;
  addFloatInitializer(graph, "w", {2}, {1.0f, 2.0f});
  addFloatInitializer(graph, "w", {2}, {3.0f, 4.0f});
  addTensorOutput(graph, "w", {2});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  EXPECT_FALSE(parser.parse(graph, parsed, error));
  EXPECT_NE(error.find("twice"), std::string::npos);
}

// -------------------------------------------------------------------------
// Resolver
// -------------------------------------------------------------------------

TEST(OnnxTensorResolver, InfersDtypeAndShapeThroughAReshapeChain) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addInt64Initializer(graph, "shape", {2}, {3, 2});
  addNode(graph, "Reshape", "reshape", {"x", "shape"}, {"y"});
  addTensorOutput(graph, "y", {3, 2});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const TensorDescriptor* descriptor = result.resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->dtype, "float32");
  EXPECT_TRUE(descriptor->rank_known);
  EXPECT_EQ(descriptor->shape, (std::vector<int64_t>{3, 2}));
}

TEST(OnnxTensorResolver, ResolvesGemmShapesForOrtOptimizedGraphs) {
  // ORT's optimizer rewrites MatMul + Add into Gemm before a plugin EP sees the
  // graph; the frontend keeps the node as atir.opaque but must still know its
  // output shape, or every consumer behind the projection loses its rank
  // (measured on model4: 19 of 27 LayerNorm chains sit behind one).
  GraphProto graph;
  addTensorInput(graph, "a", {4, 8});
  addTensorInput(graph, "b", {8, 16});
  addTensorInput(graph, "c", {16});
  addTensorOutput(graph, "y", {4, 16});
  addNode(graph, "Gemm", "gemm", {"a", "b", "c"}, {"y"});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const TensorDescriptor* descriptor = result.resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(descriptor->rank_known);
  EXPECT_EQ(descriptor->shape, (std::vector<int64_t>{4, 16}));

  // transB flips B's dims: with B declared [16, 8] the contraction lines up
  // with A's 8; transA flips A's, so K comes from A's first dim.
  GraphProto transposed;
  addTensorInput(transposed, "a", {8, 4});
  addTensorInput(transposed, "b", {16, 8});
  addTensorOutput(transposed, "y", {4, 16});
  NodeProto* node =
      addNode(transposed, "Gemm", "gemm", {"a", "b"}, {"y"});
  setInt(node, "transA", 1);
  setInt(node, "transB", 1);

  PipelineResult flipped = runPipeline(transposed);
  ASSERT_TRUE(flipped.ok) << flipped.error;
  const TensorDescriptor* flipped_y = flipped.resolved.find("y");
  ASSERT_NE(flipped_y, nullptr);
  EXPECT_TRUE(flipped_y->rank_known);
  EXPECT_EQ(flipped_y->shape, (std::vector<int64_t>{4, 16}));

  // A contraction mismatch must leave the *intermediate* rank-unknown rather
  // than publishing a shape the host would then trust (a declared graph output
  // keeps its own annotation either way, so probe the intermediate).
  GraphProto mismatched;
  addTensorInput(mismatched, "a", {4, 8});
  addTensorInput(mismatched, "b", {16, 16});
  addNode(mismatched, "Gemm", "gemm", {"a", "b"}, {"g"});
  addNode(mismatched, "Relu", "relu", {"g"}, {"y"});
  addTensorOutput(mismatched, "y", {4, 16});

  PipelineResult bad = runPipeline(mismatched);
  ASSERT_TRUE(bad.ok) << bad.error;
  const TensorDescriptor* gemm_out = bad.resolved.find("g");
  ASSERT_NE(gemm_out, nullptr);
  EXPECT_FALSE(gemm_out->rank_known);
}

TEST(OnnxTensorResolver, PropagatesSymbolicDimensionsAsDynamic) {
  GraphProto graph;
  addSymbolicInput(graph, "x", {-1, 4});
  addNode(graph, "Relu", "relu", {"x"}, {"y"});
  addTensorOutput(graph, "y", {-1, 4});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const TensorDescriptor* descriptor = result.resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->shape.size(), 2u);
  EXPECT_EQ(descriptor->shape[0], kDynamicDim);
  EXPECT_EQ(descriptor->shape[1], 4);
  EXPECT_EQ(descriptor->sym_dims, 1);
}

TEST(OnnxTensorResolver, TracksRankUnknownOutputs) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  // No table entry and no declared output shape: rank stays unknown.
  addNode(graph, "MysteryOp", "mystery", {"x"}, {"y"});
  addSymbolicInput(graph, "unused", {1});

  OnnxGraph parsed;
  std::string error;
  OnnxGraphParser parser;
  ASSERT_TRUE(parser.parse(graph, parsed, error)) << error;
  ResolvedOnnxGraph resolved;
  OnnxTensorResolver resolver;
  ASSERT_TRUE(resolver.resolve(parsed, resolved, error)) << error;
  const TensorDescriptor* descriptor = resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_FALSE(descriptor->rank_known);
  EXPECT_EQ(resolved.produced.no_shape, 1);
}

TEST(OnnxTensorResolver, ComputesReduceProdOverAllDimensions) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3, 4});
  NodeProto* reduce = addNode(graph, "ReduceProd", "prod", {"x"}, {"y"});
  setInt(reduce, "keepdims", 0);
  addTensorOutput(graph, "y", {});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const TensorDescriptor* descriptor = result.resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(descriptor->shape.empty());
}

TEST(OnnxTensorResolver, ComputesSliceOutputLengths) {
  GraphProto graph;
  addTensorInput(graph, "x", {10});
  addInt64Initializer(graph, "starts", {1}, {2});
  addInt64Initializer(graph, "ends", {1}, {8});
  addInt64Initializer(graph, "axes", {1}, {0});
  addNode(graph, "Slice", "slice", {"x", "starts", "ends", "axes"}, {"y"});
  addTensorOutput(graph, "y", {6});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const TensorDescriptor* descriptor = result.resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->shape.size(), 1u);
  EXPECT_EQ(descriptor->shape[0], 6);
}

TEST(OnnxTensorResolver, RejectsDuplicateUnsqueezeAxes) {
  // ONNX requires unique axes. With a repeated one the insertion bitmap marks
  // fewer positions than there are axes, so the shape build reads past the
  // input shape; the node must leave its rank unknown instead.
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addInt64Initializer(graph, "axes", {2}, {0, 0});
  addNode(graph, "Unsqueeze", "unsq", {"x", "axes"}, {"y"});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const TensorDescriptor* descriptor = result.resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_FALSE(descriptor->rank_known);
}

TEST(OnnxTensorResolver, RejectsDuplicateSqueezeAxes) {
  // A repeated axis would drop an extra dimension -- and for the last axis
  // erase past the end -- so it must leave the rank unknown rather than
  // fabricate a shape that the Reshape mapping would then bake in.
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3, 4});
  addInt64Initializer(graph, "axes", {2}, {2, 2});
  addNode(graph, "Squeeze", "sq", {"x", "axes"}, {"y"});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const TensorDescriptor* descriptor = result.resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_FALSE(descriptor->rank_known);
}

TEST(OnnxTensorResolver, KeepsTheFirstGraphsConstantsAfterASecondResolve) {
  // The constant table is owned by each ResolvedOnnxGraph, so reusing one
  // resolver for another graph must not invalidate the first result.
  GraphProto first;
  addTensorInput(first, "x", {2});
  addInt64Initializer(first, "axes", {1}, {0});
  addNode(first, "Unsqueeze", "unsq", {"x", "axes"}, {"y"});

  GraphProto second;
  addTensorInput(second, "a", {4});

  OnnxGraphParser parser;
  std::string error;
  OnnxGraph parsed_first;
  OnnxGraph parsed_second;
  ASSERT_TRUE(parser.parse(first, parsed_first, error)) << error;
  ASSERT_TRUE(parser.parse(second, parsed_second, error)) << error;

  OnnxTensorResolver resolver;
  ResolvedOnnxGraph resolved_first;
  ResolvedOnnxGraph resolved_second;
  ASSERT_TRUE(resolver.resolve(parsed_first, resolved_first, error)) << error;
  ASSERT_TRUE(resolver.resolve(parsed_second, resolved_second, error)) << error;

  std::vector<int64_t> axes;
  EXPECT_TRUE(resolved_first.constants.getInts("axes", axes));
  EXPECT_EQ(axes, (std::vector<int64_t>{0}));
  EXPECT_FALSE(resolved_second.constants.getInts("axes", axes));
}

// -------------------------------------------------------------------------
// Adapter: fallback decisions and the conservation trail
// -------------------------------------------------------------------------

TEST(OnnxNodeInfoAdapter, RecordsNoSpecFallbackWithReason) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addNode(graph, "Or", "or_node", {"x", "x"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "or_node");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonNoSpec);

  // The fallback marker is what makes the builder emit atir.opaque.
  bool found = false;
  for (const annc::NodeInfo& node : result.adapted.nodes) {
    if (node.name != "or_node") continue;
    found = true;
    int64_t marker = 0;
    EXPECT_TRUE(node.getAttr(kFallbackAttr.str(), marker));
    EXPECT_EQ(marker, 1);
  }
  EXPECT_TRUE(found);
}

TEST(OnnxNodeInfoAdapter, RejectsSliceWithNonUnitStep) {
  GraphProto graph;
  addTensorInput(graph, "x", {10});
  addInt64Initializer(graph, "starts", {1}, {0});
  addInt64Initializer(graph, "ends", {1}, {10});
  addInt64Initializer(graph, "axes", {1}, {0});
  addInt64Initializer(graph, "steps", {1}, {2});
  addNode(graph, "Slice", "slice",
          {"x", "starts", "ends", "axes", "steps"}, {"y"});
  addTensorOutput(graph, "y", {5});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "slice");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedAttr);
}

TEST(OnnxNodeInfoAdapter, RejectsMultiAxisUnsqueeze) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addInt64Initializer(graph, "axes", {2}, {0, 2});
  addNode(graph, "Unsqueeze", "unsqueeze", {"x", "axes"}, {"y"});
  addTensorOutput(graph, "y", {1, 2, 1, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "unsqueeze");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedAttr);
}

TEST(OnnxNodeInfoAdapter, RejectsSqueezeWithUnknownOutputRank) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  // The rank is lost before the Squeeze (an op the inference subset cannot
  // reason about), so transformSqueeze has no target shape to reshape to and
  // the node must fall back instead of emitting a Reshape with an empty shape.
  addNode(graph, "UnknownProducer", "opaque", {"x"}, {"mid"});
  addNode(graph, "Squeeze", "squeeze", {"mid"}, {"y"});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "squeeze");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonShapeUnknown);
}

TEST(OnnxNodeInfoAdapter, SqueezesARankKnownTensorWithoutFallback) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3, 1});
  addNode(graph, "Squeeze", "squeeze", {"x"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "squeeze");
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->mapped);
  const TensorDescriptor* descriptor = result.resolved.find("y");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->shape, (std::vector<int64_t>{2, 3}));
}

TEST(OnnxNodeInfoAdapter, RejectsOpsetSensitiveOpsUnderRelaxedOpset) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addNode(graph, "Softmax", "softmax", {"x"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  AdaptOptions relaxed;
  relaxed.relaxed_opset = true;
  relaxed.opset_version = 18;
  PipelineResult result = runPipeline(graph, relaxed);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "softmax");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonOpsetMismatch);
}

TEST(OnnxNodeInfoAdapter, MappedOpsCarryNoFallbackMarker) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addNode(graph, "Relu", "relu", {"x"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "relu");
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->mapped);
  EXPECT_EQ(entry->atir_op, "atir.Relu");
  EXPECT_TRUE(entry->reason.empty());
}

TEST(OnnxNodeInfoAdapter, TurnsInitializersIntoConstantsAndOutputsIntoBoundaries) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addFloatInitializer(graph, "w", {3}, {1.0f, 2.0f, 3.0f});
  addNode(graph, "Mul", "mul", {"x", "w"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;

  bool saw_constant = false;
  bool saw_boundary = false;
  for (const annc::NodeInfo& node : result.adapted.nodes) {
    if (node.name == "w") {
      saw_constant = true;
      EXPECT_EQ(node.op_type, "Constant");
      EXPECT_EQ(node.raw_data.size(), 3u * sizeof(float));
    }
    if (node.name == "__annc_output_0") {
      saw_boundary = true;
      EXPECT_TRUE(node.isOutputNode);
      ASSERT_EQ(node.inputs.size(), 1u);
      EXPECT_EQ(node.inputs[0], "y");
      EXPECT_EQ(node.tf_attrs.at("tf.output_tensor"), "y");
    }
  }
  EXPECT_TRUE(saw_constant);
  EXPECT_TRUE(saw_boundary);
  EXPECT_EQ(result.adapted.model_nodes, 1);
  EXPECT_EQ(result.adapted.initializers, 1);
  EXPECT_EQ(result.adapted.output_boundaries, 1);
  // One NodeInfo per model node, initializer and graph output -- the property
  // the conservation check in the coverage report relies on.
  EXPECT_EQ(result.adapted.nodes.size(),
            static_cast<size_t>(1 + 1 + 1 + 1 /* runtime input */));
}

// -------------------------------------------------------------------------
// Builder: the ONNX table must not shadow the TF table, and the fallback
// marker must reach atir.opaque.
// -------------------------------------------------------------------------

// Looks up an op in the TF spec table (the engine's own table).
const OpSpec* lookupTfOpSpec(llvm::StringRef opType) {
  for (const OpSpec& spec : annc::MLIRBuilder::getAllOpSpecs())
    for (llvm::StringRef op : spec.tfOps)
      if (op == opType) return &spec;
  return nullptr;
}

void registerTestDialects(DialectRegistry& registry) {
  registry.insert<::mlir::func::FuncDialect, ::atir::AtirDialect,
                  ::mlir::arith::ArithDialect>();
}

// Builds the ATIR module for an adapted graph; `context` must outlive it.
::mlir::ModuleOp buildTestModule(const AdaptedGraph& adapted,
                                 ::mlir::MLIRContext& context) {
  ::mlir::ModuleOp module =
      ::mlir::ModuleOp::create(::mlir::UnknownLoc::get(&context));
  OnnxMLIRBuilder builder(module);
  EXPECT_TRUE(builder.buildFromNodes(adapted.nodes));
  return module;
}

::mlir::Operation* findOp(::mlir::ModuleOp module, llvm::StringRef op_name) {
  ::mlir::Operation* found = nullptr;
  module.walk([&](::mlir::Operation* op) {
    if (found == nullptr && op->getName().getStringRef() == op_name) found = op;
  });
  return found;
}

TEST(OnnxBuilder, ResolvesOnnxOpsWithoutDisturbingTheTfTable) {
  // ONNX `Sum` is a variadic elementwise add; TF `Sum` is a reduce. The whole
  // reason the tables are split is that one name cannot carry both.
  const OpSpec* onnx_sum = lookupOnnxOpSpec("Sum");
  ASSERT_NE(onnx_sum, nullptr);
  EXPECT_EQ(onnx_sum->atirOp, "atir.Add");

  const OpSpec* tf_sum = lookupTfOpSpec("Sum");
  ASSERT_NE(tf_sum, nullptr);
  EXPECT_EQ(tf_sum->atirOp, "atir.Sum");

  // Same divergence for Gather, from the other direction: the TF transformer
  // reads the axis from a third *operand* (TF GatherV2), while ONNX Gather
  // carries it in the `axis` attribute in every opset. The two rows therefore
  // cannot share a transformer, and this row admits exactly the two operands
  // the ONNX schema declares.
  const OpSpec* onnx_gather = lookupOnnxOpSpec("Gather");
  const OpSpec* tf_gather = lookupTfOpSpec("Gather");
  ASSERT_NE(onnx_gather, nullptr);
  ASSERT_NE(tf_gather, nullptr);
  EXPECT_NE(onnx_gather->transformer, tf_gather->transformer);
  EXPECT_EQ(onnx_gather->transformer, onnxTransformGather);
  EXPECT_EQ(onnx_gather->minInputs, 2u);
  EXPECT_EQ(onnx_gather->maxInputs, 2u);
}

TEST(OnnxBuilder, BuildsGatherWithTheTransformerBakedBatchDims) {
  // ONNX Gather has no batch_dims attribute at all (that one belongs to
  // GatherND), so atir.Gather's batch_dims must come from the transformer,
  // not from a table default that would silently be 1 if it ever applied.
  GraphProto graph;
  addTensorInput(graph, "data", {3, 4});
  addTypedInput(graph, "indices", TensorProto::INT64, {2});
  NodeProto* gather =
      addNode(graph, "Gather", "gather", {"data", "indices"}, {"y"});
  setInt(gather, "axis", 1);
  addTensorOutput(graph, "y", {3, 2});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;

  DialectRegistry registry;
  registerTestDialects(registry);
  ::mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  ::mlir::ModuleOp module = buildTestModule(result.adapted, context);

  ::mlir::Operation* gather_op = findOp(module, "atir.Gather");
  ASSERT_NE(gather_op, nullptr);
  EXPECT_EQ(
      gather_op->getAttrOfType<::mlir::IntegerAttr>("batch_dims").getInt(), 0);
}

TEST(OnnxNodeInfoAdapter, TurnsASparseInitializerIntoAnOpaqueNode) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  onnx::SparseTensorProto* sparse = graph.add_sparse_initializer();
  sparse->mutable_values()->set_name("sparse_w");
  sparse->add_dims(2);
  sparse->add_dims(3);
  addNode(graph, "Add", "add", {"x", "sparse_w"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;

  // The sparse initializer has no dense payload, but it must still exist as a
  // node in the module and as a row in the provenance table -- otherwise the
  // conservation arithmetic counts it as a fallback while the module has
  // nothing standing for it, and both checks go false by construction.
  int64_t sparse_rows = 0;
  int64_t node_rows = 0;
  for (const NodeProvenance& entry : result.adapted.provenance) {
    if (entry.sparse_initializer) {
      ++sparse_rows;
      EXPECT_FALSE(entry.mapped);
      EXPECT_EQ(entry.reason, kReasonUnsupportedDtype);
    } else {
      ++node_rows;
    }
  }
  EXPECT_EQ(node_rows, result.adapted.model_nodes);
  EXPECT_EQ(sparse_rows, 1);

  bool sparse_node_info = false;
  for (const annc::NodeInfo& node : result.adapted.nodes) {
    if (node.name != "sparse_w") continue;
    sparse_node_info = true;
    int64_t marker = 0;
    EXPECT_TRUE(node.getAttr(kFallbackAttr.str(), marker));
  }
  EXPECT_TRUE(sparse_node_info);

  DialectRegistry registry;
  registerTestDialects(registry);
  ::mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  ::mlir::ModuleOp module = buildTestModule(result.adapted, context);

  int64_t opaque_ops = 0;
  module.walk([&](::mlir::Operation* op) {
    if (op->getName().getStringRef() == "atir.opaque") ++opaque_ops;
  });
  int64_t opaque_rows = 0;
  for (const NodeProvenance& entry : result.adapted.provenance)
    if (!entry.mapped) ++opaque_rows;
  // Every provenance row marked opaque became exactly one atir.opaque -- the
  // property the coverage report's `opaque_nodes_match` asserts.
  EXPECT_EQ(opaque_ops, opaque_rows);
  EXPECT_GT(opaque_rows, 0);
}

TEST(OnnxNodeInfoAdapter, RejectsModWithFmodOne) {
  // ONNX Mod defaults to fmod=0 (sign follows the divisor), which is exactly
  // atir.FloorMod; fmod=1 is C's truncated remainder and has no counterpart.
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addTensorInput(graph, "y", {2, 3});
  NodeProto* mod = addNode(graph, "Mod", "mod", {"x", "y"}, {"z"});
  setInt(mod, "fmod", 1);

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "mod");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedAttr);
}

TEST(OnnxNodeInfoAdapter, KeepsModWithFmodZero) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addTensorInput(graph, "y", {2, 3});
  NodeProto* mod = addNode(graph, "Mod", "mod", {"x", "y"}, {"z"});
  setInt(mod, "fmod", 0);

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "mod");
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->mapped);
  EXPECT_EQ(entry->atir_op, "atir.FloorMod");
}

TEST(OnnxNodeInfoAdapter, FallsBackOnVariadicMaxWithThreeInputs) {
  // ONNX Min/Max are variadic (1..N). A lawful 3-input Max is a per-node
  // fallback; previously it made the whole import fail with a hard error.
  GraphProto graph;
  addTensorInput(graph, "a", {2});
  addTensorInput(graph, "b", {2});
  addTensorInput(graph, "c", {2});
  addNode(graph, "Max", "max3", {"a", "b", "c"}, {"y"});
  addTensorOutput(graph, "y", {2});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "max3");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedAttr);
}

TEST(OnnxNodeInfoAdapter, FallsBackOnSingleInputSum) {
  GraphProto graph;
  addTensorInput(graph, "a", {2});
  addNode(graph, "Sum", "sum1", {"a"}, {"y"});
  addTensorOutput(graph, "y", {2});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "sum1");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedAttr);
}

TEST(OnnxNodeInfoAdapter, FallsBackOnRankTwoGlobalAveragePool) {
  // A rank-2 input has no spatial dimensions: ONNX keeps its [N, C] shape,
  // while the ReduceMean mapping would see an empty axes list, which ATIR
  // reads as "reduce every dimension". The node must fall back instead.
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addNode(graph, "GlobalAveragePool", "gap", {"x"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "gap");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedAttr);
}

TEST(OnnxNodeInfoAdapter, FallsBackOnSparseValuedConstant) {
  // The sparse_value attribute is a legal ONNX Constant encoding: the node
  // falls back per-node instead of the model being rejected as control flow.
  GraphProto graph;
  addTensorInput(graph, "x", {2});
  NodeProto* node = addNode(graph, "Constant", "sc", {}, {"c"});
  AttributeProto* sparse = node->add_attribute();
  sparse->set_name("sparse_value");
  sparse->set_type(AttributeProto::SPARSE_TENSOR);
  sparse->mutable_sparse_tensor()->mutable_values()->set_name("c_values");
  addNode(graph, "Relu", "relu", {"c"}, {"y"});
  addTensorOutput(graph, "y", {2});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "sc");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedAttr);
}

TEST(OnnxNodeInfoAdapter, FallsBackOnUnsupportedCastTarget) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  NodeProto* cast = addNode(graph, "Cast", "cast", {"x"}, {"y"});
  AttributeProto* to = cast->add_attribute();
  to->set_name("to");
  to->set_type(AttributeProto::INT);
  to->set_i(TensorProto::FLOAT8E4M3FN);
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "cast");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedDtype);
}

TEST(OnnxNodeInfoAdapter, DeclaredTypeDoesNotMaskUnsupportedInitializer) {
  // A float8 initializer also declared as a FLOAT graph input: the
  // initializer's own data_type is authoritative. Previously the declaration
  // filled the empty dtype and the raw bytes were reinterpreted as float32
  // (exit 0, garbage values). Both spellings must fail identically.
  GraphProto declared;
  addTensorInput(declared, "w", {4});
  {
    TensorProto* w = declared.add_initializer();
    w->set_name("w");
    w->set_data_type(TensorProto::FLOAT8E4M3FN);
    addDims(*w, {4});
    w->set_raw_data(std::string(4, '\x3c'));
    addNode(declared, "Relu", "relu", {"w"}, {"y"});
    addTensorOutput(declared, "y", {4});
  }

  PipelineResult with_declaration = runPipeline(declared);
  ASSERT_FALSE(with_declaration.ok);
  EXPECT_NE(with_declaration.error.find("unsupported element type"),
            std::string::npos);

  GraphProto undeclared;
  {
    TensorProto* w = undeclared.add_initializer();
    w->set_name("w");
    w->set_data_type(TensorProto::FLOAT8E4M3FN);
    addDims(*w, {4});
    w->set_raw_data(std::string(4, '\x3c'));
    addNode(undeclared, "Relu", "relu", {"w"}, {"y"});
    addTensorOutput(undeclared, "y", {4});
  }

  PipelineResult without_declaration = runPipeline(undeclared);
  ASSERT_FALSE(without_declaration.ok);
  EXPECT_EQ(with_declaration.error, without_declaration.error);
}

TEST(OnnxNodeInfoAdapter, FallsBackOnIntegerDivision) {
  // atir.RealDiv computes through floating point (getFloatValues); integer
  // operands would be silently wrong once float32 cannot represent them
  // exactly, so the node falls back instead.
  GraphProto graph;
  addTypedInput(graph, "a", TensorProto::INT32, {2});
  addTypedInput(graph, "b", TensorProto::INT32, {2});
  addNode(graph, "Div", "div", {"a", "b"}, {"y"});
  addTensorOutput(graph, "y", {2});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "div");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedDtype);
}

TEST(OnnxNodeInfoAdapter, FallsBackOnIntegerComparisonAndMod) {
  // atir.Compare and atir.FloorMod read their operands through getFloatValues
  // as well, so integer operands share the Div limitation above.
  for (const std::string op : {"Equal", "Greater", "Less"}) {
    GraphProto graph;
    addTypedInput(graph, "a", TensorProto::INT64, {2});
    addTypedInput(graph, "b", TensorProto::INT64, {2});
    addNode(graph, op, "cmp", {"a", "b"}, {"y"});
    addTensorOutput(graph, "y", {2}, TensorProto::BOOL);

    PipelineResult result = runPipeline(graph);
    ASSERT_TRUE(result.ok) << result.error;
    const NodeProvenance* entry = findProvenance(result.adapted, "cmp");
    ASSERT_NE(entry, nullptr) << op;
    EXPECT_FALSE(entry->mapped) << op;
    EXPECT_EQ(entry->reason, kReasonUnsupportedDtype) << op;
  }

  GraphProto graph;
  addTypedInput(graph, "a", TensorProto::INT64, {2});
  addTypedInput(graph, "b", TensorProto::INT64, {2});
  addNode(graph, "Mod", "mod", {"a", "b"}, {"y"});
  addTensorOutput(graph, "y", {2}, TensorProto::INT64);

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "mod");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonUnsupportedDtype);
}

TEST(OnnxNodeInfoAdapter, EmitsInitializersInDeclarationOrder) {
  // The lookup map iterates in hash order; the emitted module must follow the
  // model, or the op order downstream of it stops being a function of the
  // model.
  GraphProto graph;
  addTensorInput(graph, "x", {2});
  addFloatInitializer(graph, "z_w", {2}, {1.0f, 2.0f});
  addFloatInitializer(graph, "a_w", {2}, {3.0f, 4.0f});
  addFloatInitializer(graph, "m_w", {2}, {5.0f, 6.0f});
  addNode(graph, "Sum", "sum", {"z_w", "a_w", "m_w"}, {"y"});
  addTensorOutput(graph, "y", {2});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;

  std::vector<std::string> constants;
  for (const annc::NodeInfo& node : result.adapted.nodes)
    if (node.op_type == "Constant") constants.push_back(node.name);
  EXPECT_EQ(constants, (std::vector<std::string>{"z_w", "a_w", "m_w"}));
}

TEST(OnnxBuilder, AppliesOnnxAttributesToTheAtirOp) {
  // The coverage report only counts ops, so a mapping that silently kept the
  // default (Concat's axis 0, ReduceMean's keep_dims false) would leave every
  // other test green. Every explicit value below differs from the default it
  // would otherwise fall back to, so these assertions are the ones that fail.
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addTensorInput(graph, "y", {2, 3});
  NodeProto* concat = addNode(graph, "Concat", "cat", {"x", "y"}, {"c"});
  setInt(concat, "axis", 1);
  NodeProto* transpose = addNode(graph, "Transpose", "tr", {"c"}, {"t"});
  addInt64Attribute(transpose, "perm", {1, 0});
  NodeProto* reduce = addNode(graph, "ReduceMean", "mean", {"t"}, {"r"});
  addInt64Attribute(reduce, "axes", {1});
  setInt(reduce, "keepdims", 1);
  addTensorOutput(graph, "r", {6, 1});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  ASSERT_EQ(result.adapted.provenance.size(), 3u);
  for (const NodeProvenance& entry : result.adapted.provenance)
    EXPECT_TRUE(entry.mapped) << entry.node_name;

  DialectRegistry registry;
  registerTestDialects(registry);
  ::mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  ::mlir::ModuleOp module = buildTestModule(result.adapted, context);

  ::mlir::Operation* concat_op = findOp(module, "atir.Concat");
  ASSERT_NE(concat_op, nullptr);
  // The row's axis anchor is 0; only a mapped attribute can make this 1.
  EXPECT_EQ(concat_op->getAttrOfType<::mlir::IntegerAttr>("axis").getInt(), 1);

  ::mlir::Operation* transpose_op = findOp(module, "atir.Transpose");
  ASSERT_NE(transpose_op, nullptr);
  auto permutation =
      transpose_op->getAttrOfType<::mlir::ArrayAttr>("permutation");
  ASSERT_TRUE(permutation);
  ASSERT_EQ(permutation.size(), 2u);
  EXPECT_EQ(::mlir::cast<::mlir::IntegerAttr>(permutation[0]).getInt(), 1);
  EXPECT_EQ(::mlir::cast<::mlir::IntegerAttr>(permutation[1]).getInt(), 0);

  ::mlir::Operation* reduce_op = findOp(module, "atir.ReduceMean");
  ASSERT_NE(reduce_op, nullptr);
  // ONNX defaults keepdims to 1, ATIR to false: a silently unapplied
  // attribute would show up here as false.
  EXPECT_TRUE(
      reduce_op->getAttrOfType<::mlir::BoolAttr>("keep_dims").getValue());
}

TEST(OnnxBuilder, OmittedConcatAxisKeepsTheResolverFallback) {
  // Concat-4 and later require the axis attribute, so omitting it is already
  // outside the schema; the emitted op must still agree with the shape the
  // resolver published for the node, which falls back to axis 0.
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addTensorInput(graph, "y", {2, 3});
  addNode(graph, "Concat", "cat", {"x", "y"}, {"c"});
  addTensorOutput(graph, "c", {4, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;

  const TensorDescriptor* descriptor = result.resolved.find("c");
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->shape, (std::vector<int64_t>{4, 3}));

  DialectRegistry registry;
  registerTestDialects(registry);
  ::mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  ::mlir::ModuleOp module = buildTestModule(result.adapted, context);

  ::mlir::Operation* concat_op = findOp(module, "atir.Concat");
  ASSERT_NE(concat_op, nullptr);
  EXPECT_EQ(concat_op->getAttrOfType<::mlir::IntegerAttr>("axis").getInt(), 0);
}

TEST(OnnxBuilder, KeepsPrivateAttributesOutOfTheMetadata) {
  // The adapter's normalized Slice window rides in NodeInfo::attrs under a
  // leading underscore; the emitted metadata is the recovery surface for
  // downstream consumers and must not carry frontend-internal keys.
  GraphProto graph;
  addTensorInput(graph, "x", {4, 3});
  addInt64Initializer(graph, "starts", {1}, {1});
  addInt64Initializer(graph, "ends", {1}, {3});
  addInt64Initializer(graph, "axes", {1}, {0});
  addNode(graph, "Slice", "slice", {"x", "starts", "ends", "axes"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;

  DialectRegistry registry;
  registerTestDialects(registry);
  ::mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  ::mlir::ModuleOp module = buildTestModule(result.adapted, context);

  ::mlir::Operation* slice_op = findOp(module, "atir.Slice");
  ASSERT_NE(slice_op, nullptr);
  auto metadata = slice_op->getAttrOfType<::mlir::DictionaryAttr>("metadata");
  ASSERT_TRUE(metadata);
  for (const ::mlir::NamedAttribute& entry : metadata) {
    EXPECT_FALSE(entry.getName().getValue().starts_with("_"))
        << "private attribute leaked into metadata: " << entry.getName().str();
  }
  EXPECT_TRUE(metadata.get("tf.name"));
}

TEST(OnnxNodeInfoAdapter, RejectsExplicitNonLastSoftmaxAxisWithoutRank) {
  // The default axis (-1) is the last dimension whatever the rank is, so it
  // needs no rank. An explicit axis does, and without one the check cannot
  // run: assuming last-dim semantics would be a guess.
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addNode(graph, "UnknownProducer", "opaque", {"x"}, {"mid"});
  NodeProto* softmax = addNode(graph, "Softmax", "softmax", {"mid"}, {"y"});
  setInt(softmax, "axis", 0);

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;
  const NodeProvenance* entry = findProvenance(result.adapted, "softmax");
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(entry->mapped);
  EXPECT_EQ(entry->reason, kReasonShapeUnknown);
}

TEST(OnnxBuilder, LowersFallbackNodesToOpaque) {
  GraphProto graph;
  addTensorInput(graph, "x", {2, 3});
  addNode(graph, "Or", "or_node", {"x", "x"}, {"y"});
  addTensorOutput(graph, "y", {2, 3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;

  DialectRegistry registry;
  registerTestDialects(registry);
  ::mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  ::mlir::ModuleOp module = buildTestModule(result.adapted, context);

  int64_t opaque = 0;
  module.walk([&](::mlir::Operation* op) {
    if (op->getName().getStringRef() == "atir.opaque") ++opaque;
  });
  EXPECT_EQ(opaque, 1);
}

TEST(OnnxBuilder, BuildsAConstantFromAnInitializer) {
  GraphProto graph;
  addTensorInput(graph, "x", {3});
  addFloatInitializer(graph, "w", {3}, {1.0f, 2.0f, 3.0f});
  addNode(graph, "Mul", "mul", {"x", "w"}, {"y"});
  addTensorOutput(graph, "y", {3});

  PipelineResult result = runPipeline(graph);
  ASSERT_TRUE(result.ok) << result.error;

  DialectRegistry registry;
  registerTestDialects(registry);
  ::mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  ::mlir::ModuleOp module = buildTestModule(result.adapted, context);

  int64_t constants = 0;
  module.walk([&](::mlir::Operation* op) {
    if (op->getName().getStringRef() == "atir.constant") ++constants;
  });
  EXPECT_EQ(constants, 1);
}

}  // namespace
}  // namespace annc::onnx2atir
