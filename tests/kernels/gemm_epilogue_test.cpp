#include <optional>

#include "Dialect/Atir/Passes/GemmEpilogueCandidate.h"
#include "Target/aarch64/Gemm/Epilogue/EpilogueEmitter.h"
#include "annc/lib/Target/aarch64/Gemm/GemmPlan.h"
#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace {
namespace epilogue = annc::aarch64::gemm::epilogue;
namespace gemm = annc::aarch64::gemm;

const auto &sveRegistry =
    epilogue::getEpilogueEmitterRegistry(gemm::GemmIsa::kSve);
const auto &neonRegistry =
    epilogue::getEpilogueEmitterRegistry(gemm::GemmIsa::kNeon);

mlir::FailureOr<epilogue::EmissionResult> noOpEmitter(
    const epilogue::EmitRequest &) {
  return epilogue::EmissionResult{};
}

gemm::GemmPlan gemmPlan(int64_t mr, int64_t panelLanes) {
  gemm::GemmPlan plan{};
  plan.version = gemm::kPlanVersion;
  plan.kernelTile = {mr, panelLanes};
  plan.isa = gemm::GemmIsa::kSve;
  return plan;
}

atir::EpilogueProgram program(
    std::initializer_list<atir::EpilogueOpcode> opcodes) {
  atir::EpilogueProgram result;
  for (atir::EpilogueOpcode opcode : opcodes)
    result.steps.push_back({opcode, 0, atir::BroadcastKind::kScalar});
  return result;
}

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

epilogue::FragmentRegisterBinding binding(unsigned groups) {
  epilogue::FragmentRegisterBinding result;
  result.accumulatorRegisters = {8, 9};
  result.scratchRegisters = {0, 1, 2, 3, 4, 5, 6, 7};
  result.nGroups = groups;
  return result;
}

bool contains(const epilogue::EmissionResult &result, llvm::StringRef text) {
  return llvm::any_of(result, [&](const std::string &instruction) {
    return llvm::StringRef(instruction).contains(text);
  });
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

TEST(GemmEpilogueBackendTest, SelectsLongestPrefixOnFixedCandidate) {
  auto full =
      program({atir::EpilogueOpcode::kAdd, atir::EpilogueOpcode::kSigmoid});
  auto fullCount =
      epilogue::selectFusiblePrefix(full, gemmPlan(6, 4), sveRegistry);
  ASSERT_TRUE(mlir::succeeded(fullCount));
  EXPECT_EQ(*fullCount, 2u);

  EXPECT_TRUE(mlir::failed(
      epilogue::selectFusiblePrefix(program({atir::EpilogueOpcode::kAdd}),
                                    gemmPlan(7, 4), sveRegistry)));
  EXPECT_TRUE(mlir::failed(
      epilogue::selectFusiblePrefix(program({atir::EpilogueOpcode::kAdd}),
                                    gemmPlan(6, 5), sveRegistry)));

  epilogue::EpilogueEmitterRegistry limited;
  limited.add(atir::EpilogueOpcode::kAdd, {1, 0, 0}, noOpEmitter);
  limited.add(atir::EpilogueOpcode::kSigmoid, {9, 0, 0}, noOpEmitter);
  auto prefix =
      epilogue::selectFusiblePrefix(full, gemmPlan(6, 4), limited);
  ASSERT_TRUE(mlir::succeeded(prefix));
  EXPECT_EQ(*prefix, 1u);

  auto unsupportedSuffix =
      program({atir::EpilogueOpcode::kAdd, atir::EpilogueOpcode::kMul});
  unsupportedSuffix.steps[1].broadcast = atir::BroadcastKind::kM;
  auto supported = epilogue::selectFusiblePrefix(
      unsupportedSuffix, gemmPlan(6, 4), sveRegistry);
  ASSERT_TRUE(mlir::succeeded(supported));
  EXPECT_EQ(*supported, 1u);

  atir::EpilogueProgram manyInputs;
  for (unsigned index = 0; index < 9; ++index)
    manyInputs.steps.push_back(
        {atir::EpilogueOpcode::kAdd, 0, atir::BroadcastKind::kScalar});
  auto inputPrefix = epilogue::selectFusiblePrefix(
      manyInputs, gemmPlan(1, 1), sveRegistry);
  ASSERT_TRUE(mlir::succeeded(inputPrefix));
  EXPECT_EQ(*inputPrefix, gemm::kEpilogueArgsPointerSlots);
}

TEST(GemmEpilogueBackendTest, ReusesStepScratchAfterEachEmission) {
  auto selected = epilogue::selectFusiblePrefix(
      program({atir::EpilogueOpcode::kAdd, atir::EpilogueOpcode::kMul,
               atir::EpilogueOpcode::kSigmoid, atir::EpilogueOpcode::kRelu}),
      gemmPlan(6, 4), sveRegistry);
  ASSERT_TRUE(mlir::succeeded(selected));
  EXPECT_EQ(*selected, 4u);
}

TEST(GemmEpilogueBackendTest, GeneratesDeterministicSveAndNeonKernels) {
  epilogue::MicrokernelRequest sve;
  sve.isa = gemm::GemmIsa::kSve;
  sve.mr = 1;
  sve.ng = 1;
  sve.microN = 4;
  sve.kcMode = gemm::KcMode::kOverwrite;
  sve.epilogueSuffix = "add_n_i0";
  sve.epilogueInstructions = {"fadd z8.s, p0/m, z8.s, z0.s"};
  auto first = epilogue::generateMicrokernel(sve);
  auto second = epilogue::generateMicrokernel(sve);
  ASSERT_TRUE(mlir::succeeded(first));
  ASSERT_TRUE(mlir::succeeded(second));
  EXPECT_EQ(first->assembly, second->assembly);
  EXPECT_EQ(first->symbol, second->symbol);
  EXPECT_NE(first->assembly.find(".include \"sve_kernel_f32.S\""),
            std::string::npos);
  EXPECT_NE(first->assembly.find("GENERATE_SVE_KERNEL 1, 1"),
            std::string::npos);

  epilogue::MicrokernelRequest neon = sve;
  neon.isa = gemm::GemmIsa::kNeon;
  neon.microN = 3;
  neon.kResidue = 1;
  neon.epilogueInstructions = {"fadd v8.4s, v8.4s, v0.4s"};
  auto generated = epilogue::generateMicrokernel(neon);
  ASSERT_TRUE(mlir::succeeded(generated));
  EXPECT_NE(generated->assembly.find(".include \"neon_kernel_f32.S\""),
            std::string::npos);
  EXPECT_NE(generated->assembly.find("GENERATE_NEON_KERNEL 1, 3, 1, 3, 1"),
            std::string::npos);

  epilogue::MicrokernelRequest sveRowMajor = sve;
  sveRowMajor.rowMajor = true;
  auto generatedSveRowMajor = epilogue::generateMicrokernel(sveRowMajor);
  ASSERT_TRUE(mlir::succeeded(generatedSveRowMajor));
  EXPECT_NE(generatedSveRowMajor->assembly.find(
                ".include \"sve_smallshape_rm_f32.S\""),
            std::string::npos);
  EXPECT_NE(generatedSveRowMajor->assembly.find(
                "GENERATE_SVE_KERNEL_RM 1, 1, _generated_add_n_i0"),
            std::string::npos);

  epilogue::MicrokernelRequest neonRowMajor = neon;
  neonRowMajor.rowMajor = true;
  auto generatedNeonRowMajor = epilogue::generateMicrokernel(neonRowMajor);
  ASSERT_TRUE(mlir::succeeded(generatedNeonRowMajor));
  EXPECT_NE(generatedNeonRowMajor->assembly.find(
                ".include \"neon_smallshape_rm_f32.S\""),
            std::string::npos);
  EXPECT_NE(generatedNeonRowMajor->assembly.find(
                "GENERATE_NEON_KERNEL_RM 1, 3, 1, 3, 1, "
                "_generated_add_n_i0"),
            std::string::npos);
  EXPECT_NE(generatedNeonRowMajor->symbol.find("_kernel_rm_mr1_n3_k1_generated_"),
            std::string::npos);

  epilogue::MicrokernelRequest neonMatvec = neonRowMajor;
  neonMatvec.mr = 4;
  neonMatvec.ng = 1;
  neonMatvec.microN = 1;
  neonMatvec.kResidue = 0;
  neonMatvec.kHasGroups = true;
  neonMatvec.matrixVector = true;
  auto generatedNeonMatvec = epilogue::generateMicrokernel(neonMatvec);
  ASSERT_TRUE(mlir::succeeded(generatedNeonMatvec));
  EXPECT_NE(generatedNeonMatvec->assembly.find(
                ".include \"neon_matvec_f32.S\""),
            std::string::npos);
  EXPECT_NE(generatedNeonMatvec->assembly.find(
                "GENERATE_NEON_MATVEC 4, 0, 1, _generated_add_n_i0"),
            std::string::npos);
  EXPECT_NE(generatedNeonMatvec->symbol.find(
                "_neon_matvec_mr4_kg_r0_generated_add_n_i0_f32"),
            std::string::npos);
  EXPECT_TRUE(mlir::failed(epilogue::generateMicrokernel(
      epilogue::MicrokernelRequest{gemm::GemmIsa::kSve,
                                   1,
                                   1,
                                   gemm::KcMode::kOverwrite,
                                   false,
                                   4,
                                   "bad",
                                   {"st1w z8.s, p0, [x0]"}})));
}

TEST(GemmEpilogueEmitterTest, SveReluUsesGroupPredicatesAndLimit) {
  epilogue::EmitRequest request{atir::EpilogueOpcode::kRelu, binding(2), 6.0f};
  request.binding.predicateRegister = 0;
  auto result = sveRegistry.emit(request);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_TRUE(contains(*result, "fmaxnm z8.s, p0/m"));
  EXPECT_TRUE(contains(*result, "fmaxnm z9.s, p1/m"));
  EXPECT_TRUE(contains(*result, "fminnm z8.s, p0/m"));
  EXPECT_TRUE(contains(*result, "fminnm z9.s, p1/m"));
  request.binding.predicateRegister.reset();
  EXPECT_TRUE(mlir::failed(sveRegistry.emit(request)));
}

TEST(GemmEpilogueEmitterTest, SveSigmoidUsesEachTailPredicate) {
  epilogue::EmitRequest request{atir::EpilogueOpcode::kSigmoid, binding(2)};
  request.binding.predicateRegister = 0;
  auto result = sveRegistry.emit(request);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_TRUE(contains(*result, "fcmuo p2.s, p0/z, z8.s"));
  EXPECT_TRUE(contains(*result, "fcmuo p2.s, p1/z, z9.s"));
  EXPECT_TRUE(contains(*result, "fneg z1.s, p1/m, z9.s"));
}

TEST(GemmEpilogueEmitterTest, NeonSigmoidPreservesSpecialValueMasks) {
  epilogue::EmitRequest request{atir::EpilogueOpcode::kSigmoid, binding(1)};
  auto result = neonRegistry.emit(request);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_TRUE(contains(*result, "fcmeq v6.4s, v4.4s, v5.4s"));
  EXPECT_TRUE(contains(*result, "bif v8.16b"));
  EXPECT_TRUE(contains(*result, "bit v8.16b"));
}

TEST(GemmEpilogueEmitterTest, NeonReluEmitsUpperClampWithoutPredicates) {
  epilogue::EmitRequest request{atir::EpilogueOpcode::kRelu, binding(1), 6.0f};
  auto result = neonRegistry.emit(request);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_TRUE(contains(*result, "fmaxnm v8.4s"));
  EXPECT_TRUE(contains(*result, "fminnm v8.4s"));
  for (const std::string &instruction : *result) {
    EXPECT_EQ(instruction.find("st1"), std::string::npos);
    EXPECT_EQ(instruction.find("ret"), std::string::npos);
  }
}

}  // namespace
