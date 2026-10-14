#include "annc_ep_compile.h"

#include <gtest/gtest.h>

namespace annc_ep {
namespace {

Site ValidSite() {
  Site site;
  site.pattern = "matmul";
  site.kernelName = "fused_matmul";
  site.executionMode = "aot";
  site.abi = "mlir_ciface";
  site.memberNodes = {"n0", "n1"};
  site.outputTensor = "t1_0";
  site.flattenXIndex = 0;
  site.weightIndex = 1;
  site.args = {{"x", "dynamic", "f32", 3, {1, 2, 4}},
               {"w", "fixed", "f32", 2, {4, 8}}};
  site.outputs = {{"y", "output", "f32", 3, {1, 2, 8}}};
  site.kernelArgOrder = {0, 1, 2};
  return site;
}

// The annc_execution_v2 contract: inputs only (the kernel allocates its own
// outputs through the execution context), no kernel argument order, distinct
// argument ranks, f32 throughout.
Site ValidContextSite() {
  Site site;
  site.pattern = "tensordot_matmul";
  site.kernelName = "fused_tensordot_matmul";
  site.executionMode = "aot";
  site.abi = "annc_execution_v2";
  site.memberNodes = {"n0", "n1"};
  site.args = {{"x", "dynamic", "f32", 3, {1, 2, 4}},
               {"w", "fixed", "f32", 2, {4, 8}}};
  site.outputs = {{"y", "output", "f32", 3, {1, 2, 1}}};
  return site;
}

TEST(AnncEpSiteValidation, AcceptsValidSite) {
  std::string error;
  EXPECT_TRUE(ValidateSite(ValidSite(), &error)) << error;
}

TEST(AnncEpSiteValidation, AcceptsAotTensordotMatMul) {
  Site site = ValidSite();
  site.pattern = "tensordot_matmul";
  site.args = {{"w", "fixed", "f32", 2, {4, 8}},
               {"x", "dynamic", "f32", 3, {1, 2, 4}}};
  site.kernelArgOrder = {1, 0, 2};
  std::string error;
  EXPECT_TRUE(ValidateSite(site, &error)) << error;
}

TEST(AnncEpSiteValidation, RejectsWrongTensordotKernelOrder) {
  Site site = ValidSite();
  site.pattern = "tensordot_matmul";
  site.args = {{"w", "fixed", "f32", 2, {4, 8}},
               {"x", "dynamic", "f32", 3, {1, 2, 4}}};
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("order"), std::string::npos);
}

TEST(AnncEpSiteValidation, RejectsUnknownAbi) {
  Site site = ValidSite();
  site.abi = "unknown";
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("ABI"), std::string::npos);
}

TEST(AnncEpSiteValidation, RejectsUnsupportedPattern) {
  Site site = ValidSite();
  site.pattern = "layer_norm";
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("pattern"), std::string::npos);
}

TEST(AnncEpSiteValidation, RejectsDuplicateMembers) {
  Site site = ValidSite();
  site.memberNodes.push_back("n1");
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("member"), std::string::npos);
}

TEST(AnncEpSiteValidation, RejectsConflictingOperandIndices) {
  Site site = ValidSite();
  site.flattenXIndex = 1;
  site.weightIndex = 1;
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("operand"), std::string::npos);
}

TEST(AnncEpSiteValidation, AllowsUnresolvedOperandIndices) {
  Site site = ValidSite();
  site.flattenXIndex = -1;
  site.weightIndex = -1;
  std::string error;
  EXPECT_TRUE(ValidateSite(site, &error)) << error;
}

TEST(AnncEpSiteValidation, RejectsMismatchedArgumentShape) {
  Site site = ValidSite();
  site.args[0].shape.pop_back();
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("shape"), std::string::npos);
}

TEST(AnncEpSiteValidation, RejectsNonPermutationKernelOrder) {
  Site site = ValidSite();
  site.kernelArgOrder = {0, 0, 2};
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("permutation"), std::string::npos);
}

TEST(AnncEpContextSiteValidation, AcceptsValidContextSite) {
  std::string error;
  EXPECT_TRUE(ValidateSite(ValidContextSite(), &error)) << error;
}

TEST(AnncEpContextSiteValidation, AcceptsSingleArgumentSite) {
  Site site = ValidContextSite();
  site.pattern = "matmul";
  site.args = {{"x", "dynamic", "f32", 2, {4, 8}}};
  site.outputs = {{"y", "output", "f32", 2, {4, 8}}};
  std::string error;
  EXPECT_TRUE(ValidateSite(site, &error)) << error;
}

TEST(AnncEpContextSiteValidation, RejectsContextSiteWithArgumentOrder) {
  Site site = ValidContextSite();
  site.kernelArgOrder = {0, 1};
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("argument order"), std::string::npos);
}

TEST(AnncEpContextSiteValidation, RejectsDuplicateArgumentRanks) {
  Site site = ValidContextSite();
  site.args = {{"a", "dynamic", "f32", 2, {4, 8}},
               {"b", "fixed", "f32", 2, {8, 4}}};
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("distinct ranks"), std::string::npos);
}

TEST(AnncEpContextSiteValidation, RejectsUnsupportedArgumentElementType) {
  Site site = ValidContextSite();
  site.args[0].dtype = "f16";
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("element type"), std::string::npos);
}

TEST(AnncEpContextSiteValidation, RejectsOutputWithUnsupportedRank) {
  Site site = ValidContextSite();
  site.outputs = {{"y", "output", "f32", 5, {1, 1, 1, 1, 1}}};
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("output"), std::string::npos);
}

TEST(AnncEpContextSiteValidation, RejectsMissingOutputs) {
  Site site = ValidContextSite();
  site.outputs.clear();
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("output"), std::string::npos);
}

TEST(AnncEpContextSiteValidation, RejectsTooManyArguments) {
  Site site = ValidContextSite();
  site.args.clear();
  for (int64_t rank = 0; rank < 4; ++rank) {
    std::vector<int64_t> shape(static_cast<size_t>(rank), 1);
    site.args.push_back({"a" + std::to_string(rank), "dynamic", "f32", rank,
                         shape});
  }
  std::string error;
  // Eight arguments exceed the V2 input limit.
  site.args.push_back({"x", "dynamic", "f32", 2, {1, 1}});
  site.args.push_back({"y", "dynamic", "f32", 3, {1, 1, 1}});
  site.args.push_back({"z", "dynamic", "f32", 1, {1}});
  site.args.push_back({"q", "dynamic", "f32", 1, {1}});
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("seven"), std::string::npos);
}

}  // namespace
}  // namespace annc_ep
