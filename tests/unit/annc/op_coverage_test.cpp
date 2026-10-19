// Op-coverage smoke test: every op declared in OpSpecTable.inc must build a
// verifiable ATIR module. A synthetic graph is generated per spec with typed
// inputs/outputs chosen so that the op's verifier accepts it. This is the
// safety net for "adding a row must just work" — it does not prove numerical
// semantics (covered by the e2e models), but it proves the construction path
// (table-driven generic build OR transformer) works for every declared op.

#include <memory>

#include "Builder/MLIROpBuilder.h"
#include "Dialect/Atir/AtirOps.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace annc;

namespace {

struct SynthInput {
  std::string dtype;
  std::vector<int64_t> shape;
};
struct SynthOutput {
  std::string dtype;
  std::vector<int64_t> shape;
};

NodeInfo makeInput(const std::string& name, const SynthInput& si) {
  NodeInfo n;
  n.name = name;
  n.op_type = "Placeholder";
  n.isInputNode = true;
  n.outputs.push_back({name, si.dtype, si.shape});
  return n;
}

// Per-op synthetic operand/output templates so verifiers accept the result.
// Anything not listed falls back to f32{2} inputs and f32{2} outputs.
// `op` is the concrete TF op name being built (not just the row's first
// alias), so variants like Gather (2 inputs) vs GatherV2 (3 inputs) each get
// an accurate template.
void synthTemplates(const OpSpec& spec, const std::string& op,
                    std::vector<SynthInput>& ins,
                    std::vector<SynthOutput>& outs) {
  auto set = [&](std::vector<SynthInput> i, std::vector<SynthOutput> o) {
    ins = std::move(i);
    outs = std::move(o);
  };

  if (op == "MatMul")
    return set({{"float32", {2, 2}}, {"float32", {2, 2}}},
               {{"float32", {2, 2}}});
  if (op == "Less" || op == "NotEqual" || op == "Greater" ||
      op == "GreaterEqual" || op == "LessEqual" || op == "Equal")
    return set({{"float32", {2}}, {"float32", {2}}}, {{"bool", {2}}});
  if (op == "LogicalAnd")
    return set({{"bool", {2}}, {"bool", {2}}}, {{"bool", {2}}});
  if (op == "Reshape")
    return set({{"float32", {2}}, {"int32", {1}}}, {{"float32", {2}}});
  if (op == "Squeeze") return set({{"float32", {2}}}, {{"float32", {2}}});
  if (op == "Transpose")
    return set({{"float32", {2}}, {"int32", {2}}}, {{"float32", {2}}});
  if (op == "ConcatV2")
    return set({{"float32", {2}}, {"int32", {}}}, {{"float32", {2}}});
  // Legacy Gather has no axis input; GatherV2 does.
  if (op == "Gather")
    return set({{"float32", {4}}, {"int32", {2}}}, {{"float32", {2}}});
  if (op == "GatherV2")
    return set({{"float32", {4}}, {"int32", {2}}, {"int32", {}}},
               {{"float32", {2}}});
  if (op == "Split")
    return set({{"int32", {}}, {"float32", {2}}}, {{"float32", {1}}});
  if (op == "Sum" || op == "Prod" || op == "Mean")
    return set({{"float32", {2}}, {"int32", {1}}}, {{"float32", {2}}});
  if (op == "Variable" || op == "VariableV2" || op == "VarHandleOp")
    return set({}, {{"float32", {2}}});
  if (op == "DynamicPartition")
    return set({{"float32", {2}}, {"int32", {2}}},
               {{"float32", {1}}, {"float32", {1}}});
  if (op == "Merge")
    return set({{"float32", {2}}, {"float32", {2}}},
               {{"float32", {2}}, {"int32", {}}});
  if (op == "SparseToDense")
    return set(
        {{"int32", {1, 1}}, {"int32", {1}}, {"float32", {1}}, {"float32", {}}},
        {{"float32", {2}}});
  if (op == "SparseTensorDenseMatMul")
    return set({{"int64", {1, 2}}, {"float32", {1}}, {"int64", {2}},
                {"float32", {2, 2}}},
               {{"float32", {1, 2}}});
  if (op == "SparseReshape")
    return set({{"int32", {1, 1}}, {"int32", {1}}, {"int32", {1}}},
               {{"int32", {1}}, {"int32", {1}}});
  if (op == "SparseFillEmptyRows")
    return set(
        {{"int32", {1, 1}}, {"float32", {1}}, {"int32", {1}}, {"float32", {}}},
        {{"int32", {1, 1}}, {"float32", {1}}, {"bool", {1}}, {"int32", {1}}});
  if (op == "SparseSegmentSum" || op == "SparseSegmentMin" ||
      op == "SparseSegmentMean")
    return set({{"float32", {4}}, {"int32", {2}}, {"int32", {2}}},
               {{"float32", {2}}});
  if (op == "ResourceGather")
    return set({{"float32", {4}}, {"int32", {2}}}, {{"float32", {2}}});
  if (op == "StringToHashBucketFast")
    return set({{"float32", {2}}}, {{"int64", {2}}});
  if (op == "StringToNumber")
    return set({{"string", {2}}}, {{"int64", {2}}});
  if (op == "StaticRegexReplace")
    return set({{"string", {2}}}, {{"string", {2}}});
  if (op == "StringSplit")
    return set({{"string", {2}}, {"string", {}}},
               {{"int64", {1, 2}}, {"string", {1}}, {"int64", {1}}});
  if (op == "Unique")
    return set({{"float32", {2}}}, {{"float32", {2}}, {"int32", {2}}});
  if (op == "Rank")
    return set({{"float32", {2}}}, {{"int32", {}}});
  // Switch routes data to false/true branches; both outputs equal the input.
  if (op == "Switch")
    return set({{"float32", {2}}, {"bool", {}}},
               {{"float32", {2}}, {"float32", {2}}});
  if (op == "TopK" || op == "TopKV2")
    return set({{"float32", {4}}, {"int32", {}}},
               {{"float32", {2}}, {"int32", {2}}});
  if (op == "Pack" || op == "Stack")
    return set({{"float32", {2}}, {"float32", {2}}}, {{"float32", {2, 2}}});
  if (op == "StridedSlice")
    return set({{"float32", {2}},
                {"float32", {2}},
                {"float32", {2}},
                {"float32", {2}}},
               {{"float32", {2}}});
  if (op == "BroadcastTo" || op == "Broadcast")
    return set({{"float32", {1}}}, {{"float32", {2}}});
  if (op == "Range")
    return set({{"int32", {}}, {"int32", {}}, {"int32", {}}}, {{"int32", {2}}});
  if (op == "Fill")
    return set({{"int32", {1}}, {"float32", {}}}, {{"float32", {2}}});
  if (op == "Where" || op == "Select" || op == "SelectV2")
    return set({{"float32", {2}}}, {{"float32", {2}}});
  if (op == "ParallelDynamicStitch")
    return set(
        {{"int32", {2}}, {"int32", {2}}, {"float32", {2}}, {"float32", {2}}},
        {{"float32", {2}}});
  if (op == "UnsortedSegmentMin")
    return set({{"float32", {4}}, {"int32", {2}}, {"int32", {}}},
               {{"float32", {2}}});
  if (op == "TensorScatterUpdate")
    return set({{"float32", {2}}, {"int32", {1, 1}}, {"float32", {1}}},
               {{"float32", {2}}});
  if (op == "Slice")
    return set({{"float32", {2}}, {"int32", {1}}, {"int32", {1}}},
               {{"float32", {1}}});
  if (op == "Pad" || op == "PadV2")
    return set({{"float32", {2}}, {"int32", {1, 2}}, {"float32", {}}},
               {{"float32", {2}}});
  if (op == "Tile")
    return set({{"float32", {2}}, {"int32", {1}}}, {{"float32", {2}}});
  if (op == "GatherNd")
    return set({{"float32", {4}}, {"int32", {1, 1}}}, {{"float32", {2}}});
  if (op == "BatchMatMul" || op == "BatchMatMulV2" || op == "Dot")
    return set({{"float32", {2, 2}}, {"float32", {2, 2}}},
               {{"float32", {2, 2}}});
  if (op == "Concat")
    return set({{"float32", {2}}, {"float32", {2}}}, {{"float32", {2}}});
}

struct CoverageTest : public ::testing::Test {
  void SetUp() override {
    registry
        .insert<func::FuncDialect, atir::AtirDialect, arith::ArithDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  bool build(const std::vector<NodeInfo>& nodes) {
    module = ModuleOp::create(UnknownLoc::get(context.get()));
    MLIRBuilder builder(module);
    return builder.buildFromNodes(nodes);
  }

  DialectRegistry registry;
  std::unique_ptr<MLIRContext> context;
  ModuleOp module;
};

// Builds a synthetic graph for one concrete TF op name: typed inputs + the
// op node. Iterating every tfOps alias (not just tfOps.front()) is what makes
// variants such as Gather/GatherV2 or Pad/PadV2 actually exercised.
std::vector<NodeInfo> makeGraph(const OpSpec& spec, const std::string& tfOp) {
  std::vector<SynthInput> ins;
  std::vector<SynthOutput> outs;
  synthTemplates(spec, tfOp, ins, outs);

  unsigned nIn = ins.size();
  // Zero-input rows (e.g. Variable family) keep zero inputs regardless of the
  // op name; everything else falls back to at least one synthetic input.
  if (nIn == 0) nIn = (spec.maxInputs == 0) ? 0 : std::max(spec.minInputs, 1u);

  std::vector<NodeInfo> nodes;
  std::vector<std::string> inputNames;
  for (unsigned i = 0; i < nIn; ++i) {
    const SynthInput& si =
        ins.empty() ? SynthInput{"float32", {2}} : ins[i % ins.size()];
    std::string name = tfOp + "_in" + std::to_string(i);
    nodes.push_back(makeInput(name, si));
    inputNames.push_back(name);
  }

  NodeInfo op;
  op.name = tfOp + "_node";
  op.op_type = tfOp;
  op.inputs = inputNames;
  unsigned nOut = outs.empty() ? 1 : outs.size();
  for (unsigned i = 0; i < nOut; ++i) {
    const SynthOutput& so =
        outs.empty() ? SynthOutput{"float32", {2}} : outs[i % outs.size()];
    op.outputs.push_back({op.name, so.dtype, so.shape});
  }
  // Constant rows need real bytes now that the builder rejects empty
  // raw_data instead of silently zero-filling.
  if (tfOp == "Const" || tfOp == "Constant" || tfOp == "HostConst") {
    const auto& so = op.outputs[0];
    size_t elemCount = 1;
    for (int64_t d : so.shape) elemCount *= static_cast<size_t>(d);
    size_t elemSize = 4;
    if (so.dtype == "float64" || so.dtype == "int64" ||
        so.dtype == "uint64" || so.dtype == "complex64")
      elemSize = 8;
    else if (so.dtype == "complex128")
      elemSize = 16;
    else if (so.dtype == "float16" || so.dtype == "bfloat16" ||
             so.dtype == "int16" || so.dtype == "uint16")
      elemSize = 2;
    else if (so.dtype == "int8" || so.dtype == "uint8" ||
             so.dtype == "bool")
      elemSize = 1;
    op.raw_data.assign(elemCount * elemSize, 0);
  }
  nodes.push_back(op);
  return nodes;
}

TEST_F(CoverageTest, EveryDeclaredOpBuilds) {
  unsigned checked = 0;
  for (const auto& spec : MLIRBuilder::getAllOpSpecs()) {
    // Build one graph per concrete TF op name: aliases and version variants
    // (Gather/GatherV2, Pad/PadV2, Add/AddV2, ...) must all be exercised,
    // otherwise a variant can be registered but never constructed.
    for (const auto& tfOpRef : spec.tfOps) {
      const std::string tfOp = tfOpRef.str();
      SCOPED_TRACE("op: " + tfOp);
      EXPECT_TRUE(build(makeGraph(spec, tfOp)));
      ++checked;
    }
  }
  // Guard against the table silently shrinking (89 names in the current
  // table: 66 rows, the rest are aliases).
  EXPECT_GE(checked, 80u);
}

// Semantic counterpart of the smoke test: every table-driven (1:1) op must
// emit exactly its declared ATIR op with the declared number of results.
// Transformer-driven rows get dedicated semantic tests in builder_test.cpp.
TEST_F(CoverageTest, EveryTableDrivenOpEmitsItsAtirOp) {
  unsigned checked = 0;
  for (const auto& spec : MLIRBuilder::getAllOpSpecs()) {
    if (spec.transformer != nullptr) continue;  // covered by dedicated tests
    for (const auto& tfOpRef : spec.tfOps) {
      const std::string tfOp = tfOpRef.str();
      SCOPED_TRACE("op: " + tfOp);
      ASSERT_TRUE(build(makeGraph(spec, tfOp)));

      size_t count = 0;
      Operation* found = nullptr;
      module->walk([&](Operation* op) {
        if (op->getName().getStringRef() == spec.atirOp) {
          ++count;
          found = op;
        }
      });
      EXPECT_EQ(count, 1u) << "expected exactly one " << spec.atirOp.str();
      if (found != nullptr) {
        unsigned expected =
            (spec.numOutputs == kVariableOutputs) ? 1 : spec.numOutputs;
        EXPECT_EQ(found->getNumResults(), expected);
      }
      ++checked;
    }
  }
  EXPECT_GE(checked, 40u);
}

}  // namespace
