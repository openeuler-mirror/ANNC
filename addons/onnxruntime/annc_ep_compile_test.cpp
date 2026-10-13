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
  return site;
}

TEST(AnncEpSiteValidation, AcceptsValidSite) {
  std::string error;
  EXPECT_TRUE(ValidateSite(ValidSite(), &error)) << error;
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

}  // namespace
}  // namespace annc_ep
