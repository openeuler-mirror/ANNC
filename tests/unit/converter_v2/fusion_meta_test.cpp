// fusion_meta contract tests: tensor-endpoint parsing, metadata validation
// and the JSON read surface (top-level fusions array and single object).

#include "fusion_meta.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace annc::converter_v2 {
namespace {

TEST(FusionMeta, ParseTensorEndpointContract) {
  auto data = parseTensorEndpoint("dense");
  ASSERT_TRUE(data.ok());
  EXPECT_EQ(data.value.node, "dense");
  EXPECT_EQ(data.value.port, 0);
  EXPECT_FALSE(data.value.control);

  auto port = parseTensorEndpoint("dense:2");
  ASSERT_TRUE(port.ok());
  EXPECT_EQ(port.value.node, "dense");
  EXPECT_EQ(port.value.port, 2);

  auto control = parseTensorEndpoint("^dense");
  ASSERT_TRUE(control.ok());
  EXPECT_EQ(control.value.node, "dense");
  EXPECT_TRUE(control.value.control);

  EXPECT_FALSE(parseTensorEndpoint("").ok());
  EXPECT_FALSE(parseTensorEndpoint("dense:notaport").ok());
}

TEST(FusionMeta, ValidateFusionInfoContract) {
  FusionInfo ok;
  ok.name = "f0";
  ok.pattern = "matmul_bias";
  ok.kernelName = "k0";
  ok.args.push_back({"dynamic", "dense", {2}, 1, "float32"});
  ok.outputs.push_back({"output", "dense:0", {2}, 1, "float32"});
  EXPECT_TRUE(validateFusionInfo(ok).ok());

  FusionInfo noName = ok;
  noName.name.clear();
  EXPECT_FALSE(validateFusionInfo(noName).ok());

  FusionInfo noPattern = ok;
  noPattern.pattern.clear();
  EXPECT_FALSE(validateFusionInfo(noPattern).ok());

  FusionInfo noKernel = ok;
  noKernel.kernelName.clear();
  EXPECT_FALSE(validateFusionInfo(noKernel).ok());

  FusionInfo badMode = ok;
  badMode.executionMode = "interpreted";
  EXPECT_FALSE(validateFusionInfo(badMode).ok());

  FusionInfo badFingerprint = ok;
  badFingerprint.templateFingerprint = "ABC";
  EXPECT_FALSE(validateFusionInfo(badFingerprint).ok());

  FusionInfo goodFingerprint = ok;
  goodFingerprint.templateFingerprint =
      std::string(64, 'a');  // 64 lowercase hex chars
  EXPECT_TRUE(validateFusionInfo(goodFingerprint).ok());
}

class FusionMetaJsonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() /
            "annc_converter_v2_fusion_meta_test.json";
    std::filesystem::remove(path_);
  }
  void TearDown() override { std::filesystem::remove(path_); }

  void write(const std::string& content) {
    std::ofstream out(path_);
    out << content;
  }

  std::filesystem::path path_;
};

TEST_F(FusionMetaJsonTest, ReadsTopLevelFusionsArray) {
  write(R"json(
    {"fusions": [
      {"name": "f0", "pattern": "matmul_bias", "kernel_name": "k0",
       "args": [{"role": "dynamic", "tf_name": "dense", "shape": [2],
                 "rank": 1, "dtype": "float32"}],
       "outputs": [{"role": "output", "tf_name": "dense:0", "shape": [2],
                    "rank": 1, "dtype": "float32"}]},
      {"name": "f1", "pattern": "softmax", "kernel_name": "k1",
       "execution_mode": "jit", "template_fingerprint": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
       "args": [{"role": "dynamic", "tf_name": "x", "shape": [4],
                 "rank": 1, "dtype": "float32"}],
       "outputs": [{"role": "output", "tf_name": "x:0", "shape": [4],
                    "rank": 1, "dtype": "float32"}]}
    ]})json");
  auto result = readFusionMetadataJson(path_.string());
  ASSERT_TRUE(result.ok()) << result.error;
  ASSERT_EQ(result.value.size(), 2u);
  EXPECT_EQ(result.value[0].name, "f0");
  EXPECT_EQ(result.value[0].pattern, "matmul_bias");
  EXPECT_EQ(result.value[1].name, "f1");
  EXPECT_EQ(result.value[1].executionMode, "jit");
}

TEST_F(FusionMetaJsonTest, ReadsSingleTopLevelObject) {
  write(R"json({"name": "solo", "pattern": "gather", "kernel_name": "ks",
    "args": [{"role": "dynamic", "tf_name": "idx", "shape": [4],
              "rank": 1, "dtype": "int32"}],
    "outputs": [{"role": "output", "tf_name": "idx:0", "shape": [4],
                 "rank": 1, "dtype": "int32"}]})json");
  auto result = readFusionMetadataJson(path_.string());
  ASSERT_TRUE(result.ok()) << result.error;
  ASSERT_EQ(result.value.size(), 1u);
  EXPECT_EQ(result.value[0].name, "solo");
}

TEST_F(FusionMetaJsonTest, RejectsMalformedJson) {
  write("{not json");
  auto result = readFusionMetadataJson(path_.string());
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.error.find("failed to parse"), std::string::npos);
}

TEST_F(FusionMetaJsonTest, RejectsMissingFile) {
  auto result = readFusionMetadataJson(path_.string());
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.error.find("cannot open"), std::string::npos);
}

TEST_F(FusionMetaJsonTest, RejectsInvalidFusionEntry) {
  write(R"json({"name": "f0", "kernel_name": "k0",
    "args": [{"role": "dynamic", "tf_name": "dense", "shape": [2],
              "rank": 1, "dtype": "float32"}],
    "outputs": [{"role": "output", "tf_name": "dense:0", "shape": [2],
                 "rank": 1, "dtype": "float32"}]})json");  // pattern missing
  auto result = readFusionMetadataJson(path_.string());
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace annc::converter_v2
