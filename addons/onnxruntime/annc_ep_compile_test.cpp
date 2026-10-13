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
  site.args = {{"x", "dynamic", 3, {1, 2, 4}},
               {"w", "fixed", 2, {4, 8}}};
  site.outputs = {{"y", "output", 3, {1, 2, 8}}};
  site.kernelArgOrder = {0, 1, 2};
  return site;
}

TEST(AnncEpSiteValidation, AcceptsValidSite) {
  std::string error;
  EXPECT_TRUE(ValidateSite(ValidSite(), &error)) << error;
}

TEST(AnncEpSiteValidation, AcceptsAotTensordotMatMul) {
  Site site = ValidSite();
  site.pattern = "tensordot_matmul";
  site.args = {{"w", "fixed", 2, {4, 8}},
               {"x", "dynamic", 3, {1, 2, 4}}};
  site.kernelArgOrder = {1, 0, 2};
  std::string error;
  EXPECT_TRUE(ValidateSite(site, &error)) << error;
}

TEST(AnncEpSiteValidation, RejectsWrongTensordotKernelOrder) {
  Site site = ValidSite();
  site.pattern = "tensordot_matmul";
  site.args = {{"w", "fixed", 2, {4, 8}},
               {"x", "dynamic", 3, {1, 2, 4}}};
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

}  // namespace
}  // namespace annc_ep
