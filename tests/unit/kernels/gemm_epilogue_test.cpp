#include <algorithm>
#include <optional>

#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "Target/aarch64/Gemm/GemmCodegenAdapter.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace {
namespace codegen = annc::aarch64::gemm::codegen;

mlir::DictionaryAttr step(mlir::Builder &builder, llvm::StringRef opcode,
                          int64_t input, llvm::StringRef broadcast,
                          std::optional<float> limit = std::nullopt) {
  mlir::NamedAttrList attributes;
  attributes.append("opcode", builder.getStringAttr(opcode));
  attributes.append("input", builder.getI64IntegerAttr(input));
  attributes.append("broadcast", builder.getStringAttr(broadcast));
  if (limit) attributes.append("limit", builder.getF32FloatAttr(*limit));
  return attributes.getDictionary(builder.getContext());
}

mlir::DictionaryAttr candidateAttr(mlir::Builder &builder,
                                   llvm::ArrayRef<mlir::Attribute> steps,
                                   llvm::StringRef role = "main_d",
                                   int64_t version = 1) {
  mlir::NamedAttrList attributes;
  attributes.append("version", builder.getI64IntegerAttr(version));
  attributes.append("output_role", builder.getStringAttr(role));
  attributes.append("terminal_view", builder.getStringAttr("identity"));
  attributes.append("steps", builder.getArrayAttr(steps));
  return attributes.getDictionary(builder.getContext());
}

TEST(GemmEpilogueCandidatePlanTest, PreservesOrderDirectionAndReluLimit) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  auto parsed = atir::parseEpilogueCandidatePlan(candidateAttr(
      builder, {step(builder, "add", 1, "n"), step(builder, "mul", 0, "scalar"),
                step(builder, "relu", 0, "scalar", 6.0f)}));
  ASSERT_TRUE(mlir::succeeded(parsed));
  ASSERT_EQ(parsed->steps.size(), 3u);
  EXPECT_EQ(parsed->steps[0].opcode, atir::EpilogueOpcode::kAdd);
  EXPECT_EQ(parsed->steps[0].inputIndex, 1u);
  EXPECT_EQ(parsed->steps[1].inputIndex, 0u);
  ASSERT_TRUE(parsed->steps[2].reluLimit);
  EXPECT_FLOAT_EQ(*parsed->steps[2].reluLimit, 6.0f);
}

TEST(GemmEpilogueCandidatePlanTest, RejectsIncompleteOrInvalidSchema) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  EXPECT_TRUE(mlir::failed(
      atir::parseEpilogueCandidatePlan(candidateAttr(builder, {}))));
  EXPECT_TRUE(mlir::failed(atir::parseEpilogueCandidatePlan(
      candidateAttr(builder, {step(builder, "add", 0, "n")}, "aux"))));
  EXPECT_TRUE(mlir::failed(atir::parseEpilogueCandidatePlan(
      candidateAttr(builder, {step(builder, "add", 0, "n")}, "main_d", 2))));
  EXPECT_TRUE(mlir::failed(atir::parseEpilogueCandidatePlan(
      candidateAttr(builder, {step(builder, "relu", 1, "scalar", 0.0f)}))));
  EXPECT_TRUE(mlir::failed(atir::parseEpilogueCandidatePlan(
      candidateAttr(builder, {step(builder, "relu", 0, "n", 0.0f)}))));
  EXPECT_TRUE(mlir::failed(atir::parseEpilogueCandidatePlan(
      candidateAttr(builder, {step(builder, "unknown", 0, "n")}))));
  EXPECT_TRUE(mlir::failed(atir::parseEpilogueCandidatePlan(
      candidateAttr(builder, {step(builder, "add", 2, "n")}))));
  mlir::NamedAttrList legacyStep(step(builder, "add", 0, "n").getValue());
  legacyStep.append("input2", builder.getI64IntegerAttr(1));
  EXPECT_TRUE(mlir::failed(atir::parseEpilogueCandidatePlan(candidateAttr(
      builder, {legacyStep.getDictionary(builder.getContext())}))));
}

TEST(GemmEpilogueCandidatePlanTest, RoundTripPreservesTerminalView) {
  mlir::MLIRContext context;
  mlir::Builder builder(&context);
  atir::EpilogueProgram epilogueProgram;
  epilogueProgram.terminalView = atir::OutputView::kTerminalInsertUnitDimension;
  epilogueProgram.steps.push_back(
      {atir::EpilogueOpcode::kSigmoid, 0, atir::BroadcastKind::kScalar});
  auto parsed = atir::parseEpilogueCandidatePlan(
      atir::buildEpilogueCandidateAttr(builder, epilogueProgram));
  ASSERT_TRUE(mlir::succeeded(parsed));
  EXPECT_EQ(parsed->terminalView,
            atir::OutputView::kTerminalInsertUnitDimension);
  ASSERT_EQ(parsed->steps.size(), 1u);
  EXPECT_EQ(parsed->steps[0].opcode, atir::EpilogueOpcode::kSigmoid);
}

codegen::KernelDesc kernel(
    codegen::Family family = codegen::Family::NeonPacked) {
  codegen::KernelDesc value;
  value.family = family;
  value.m = 2;
  value.n = family == codegen::Family::NeonMatvec ? 1 : 5;
  value.k = 17;
  return value;
}

#if !ANNC_GEMM_PACKAGE_AVAILABLE
TEST(GemmCodegenAdapterTest, ReportsUnavailableComponentWithoutSourceFallback) {
  ASSERT_FALSE(codegen::isAvailable());
  std::string error;
  codegen::PreparedKernel prepared;
  EXPECT_FALSE(codegen::prepare(kernel(), {{codegen::Opcode::Relu}}, prepared,
                                error));
  EXPECT_EQ(prepared.supportedPrefixLength, 0u);
  EXPECT_NE(error.find("component is unavailable"), std::string::npos);
  codegen::CompiledObject object;
  EXPECT_FALSE(codegen::compileSerializedRequest("00", "build", object, error));
  EXPECT_TRUE(object.bytes.empty());
}

#else
TEST(GemmCodegenAdapterTest, PreparesOnlyTheSupportedPrefix) {
  ASSERT_TRUE(codegen::isAvailable());
  std::vector<codegen::Step> steps(
      9, {codegen::Opcode::Add, codegen::Broadcast::Scalar});
  std::string error;
  codegen::PreparedKernel prepared;
  ASSERT_TRUE(codegen::prepare(kernel(), steps, prepared, error)) << error;
  EXPECT_EQ(prepared.supportedPrefixLength, 8u);
  EXPECT_FALSE(prepared.normalizedRequest.empty());
  steps = {{codegen::Opcode::Add, codegen::Broadcast::N},
           {codegen::Opcode::Mul, codegen::Broadcast::M},
           {codegen::Opcode::Relu}};
  ASSERT_TRUE(codegen::prepare(kernel(), steps, prepared, error)) << error;
  EXPECT_EQ(prepared.supportedPrefixLength, 1u);
  EXPECT_FALSE(prepared.normalizedRequest.empty());
  ASSERT_TRUE(codegen::prepare(kernel(), {{codegen::Opcode::Mul,
                                           codegen::Broadcast::Matrix}},
                               prepared, error))
      << error;
  EXPECT_EQ(prepared.supportedPrefixLength, 0u);
  EXPECT_TRUE(prepared.normalizedRequest.empty());
}

TEST(GemmCodegenAdapterTest, CompilesAllExistingFamiliesFromSemanticRequests) {
  ASSERT_TRUE(codegen::isAvailable());
  for (auto family : {codegen::Family::NeonPacked, codegen::Family::NeonDirect,
                      codegen::Family::SvePacked, codegen::Family::SveDirect,
                      codegen::Family::NeonMatvec}) {
    auto selected = kernel(family);
    std::vector<codegen::Step> steps = {
        {codegen::Opcode::Add, codegen::Broadcast::N},
        {codegen::Opcode::Mul, codegen::Broadcast::Scalar},
        {codegen::Opcode::Sigmoid},
        {codegen::Opcode::Relu, codegen::Broadcast::None, 6.0f}};
    std::string error;
    codegen::PreparedKernel first, second;
    ASSERT_TRUE(codegen::prepare(selected, steps, first, error)) << error;
    selected.accumulate = true;
    ASSERT_TRUE(codegen::prepare(selected, steps, second, error)) << error;
    EXPECT_NE(first.entrySymbol, second.entrySymbol);
    EXPECT_EQ(first.artifactKey, second.artifactKey);
    EXPECT_EQ(first.normalizedRequest, second.normalizedRequest);
    codegen::CompiledObject one, two;
    ASSERT_TRUE(codegen::compileSerializedRequest(first.normalizedRequest,
                                                  first.buildId, one, error))
        << error;
    ASSERT_TRUE(codegen::compileSerializedRequest(second.normalizedRequest,
                                                  second.buildId, two, error))
        << error;
    EXPECT_EQ(one.bytes, two.bytes);
    EXPECT_EQ(one.artifactKey, first.artifactKey);
    ASSERT_GE(one.bytes.size(), 4u);
    EXPECT_EQ(std::vector<uint8_t>(one.bytes.begin(), one.bytes.begin() + 4),
              (std::vector<uint8_t>{0x7f, 'E', 'L', 'F'}));
    EXPECT_NE(std::find(one.definedSymbols.begin(), one.definedSymbols.end(),
                        first.entrySymbol),
              one.definedSymbols.end());
    EXPECT_NE(std::find(one.definedSymbols.begin(), one.definedSymbols.end(),
                        second.entrySymbol),
              one.definedSymbols.end());
    EXPECT_TRUE(one.dependencySymbols.empty());
  }
}

TEST(GemmCodegenAdapterTest, RejectsCorruptionAndVersions) {
  ASSERT_TRUE(codegen::isAvailable());
  codegen::PreparedKernel prepared;
  std::string error;
  auto selected = kernel();
  ASSERT_TRUE(
      codegen::prepare(selected, {{codegen::Opcode::Relu}}, prepared, error))
      << error;
  codegen::CompiledObject object;
  EXPECT_FALSE(codegen::compileSerializedRequest("not-hex", prepared.buildId,
                                                 object, error));
  EXPECT_FALSE(codegen::compileSerializedRequest(prepared.normalizedRequest,
                                                 "wrong-build", object, error));
}
#endif
}  // namespace
