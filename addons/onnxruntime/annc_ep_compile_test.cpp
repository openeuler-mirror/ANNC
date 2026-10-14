#include "annc_ep_compile.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "onnx/onnx.pb.h"
#include "onnxruntime_c_api.h"

// The graph-export translation unit pulls in ORT's C++ API wrappers, which
// resolve OrtGetApiBase from the host process at load time -- in the plugin
// that host is ORT itself (the link allows the symbol to stay undefined).
// This suite never calls that API, so a stub satisfies the linker without
// dragging the runtime into the test.
extern "C" const OrtApiBase* OrtGetApiBase(void) { return nullptr; }

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
  site.pattern = "kp_fused_gather";  // TF-domain pattern: not an EP site
  std::string error;
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("pattern"), std::string::npos);
}

TEST(AnncEpContextSiteValidation, AcceptsEveryMigratedPattern) {
  for (const char* pattern :
       {"tensordot_matmul", "masked_mul", "layer_norm",
        "batched_matmul_add"}) {
    Site site = ValidContextSite();
    site.pattern = pattern;
    std::string error;
    EXPECT_TRUE(ValidateSite(site, &error)) << pattern << ": " << error;
  }
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

// Repeated ranks are legal once every name resolved back to the graph's: the
// mapping is then pinned by name, not by rank (masked_mul's data operands,
// layer_norm's gamma/beta).  Without resolved names the rank rule is the only
// one there is, so such a site must not be claimed -- that decision needs the
// rename table, so it is stated by SiteNeedsResolvedNames and enforced by the
// claim loop, not by ValidateSite.
TEST(AnncEpContextSiteValidation, RepeatedRanksNeedResolvedNames) {
  Site site = ValidContextSite();
  site.pattern = "layer_norm";
  site.args = {{"x", "dynamic", "f32", 2, {4, 8}},
               {"gamma", "fixed", "f32", 1, {8}},
               {"beta", "fixed", "f32", 1, {8}},
               {"eps", "fixed", "f32", 0, {}}};
  std::string error;
  EXPECT_TRUE(ValidateSite(site, &error)) << error;
  EXPECT_TRUE(SiteNeedsResolvedNames(site));

  Site distinct = ValidContextSite();
  EXPECT_FALSE(SiteNeedsResolvedNames(distinct));

  // More than one output: the report's order need not match ORT's numbering,
  // so the mapping has to come from names as well.
  distinct.outputs.push_back({"y2", "output", "f32", 3, {1, 2, 1}});
  EXPECT_TRUE(SiteNeedsResolvedNames(distinct));
}

// The masked_mul id tensor is i64; the EP describes i64 descriptors.
TEST(AnncEpContextSiteValidation, AcceptsIntegerMaskArgument) {
  Site site = ValidContextSite();
  site.pattern = "masked_mul";
  site.args = {{"data0", "dynamic", "f32", 3, {2, 3, 4}},
               {"ids", "fixed", "i64", 2, {2, 3}}};
  site.namesResolved = true;
  std::string error;
  EXPECT_TRUE(ValidateSite(site, &error)) << error;
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
  // Four more carry the site to the limit, the eleventh one past it: the
  // kernel-call dispatch table in annc_ep_execution.cc has one cast per
  // argument count, and the validation carries the same bound.
  site.args.push_back({"x", "dynamic", "f32", 2, {1, 1}});
  site.args.push_back({"y", "dynamic", "f32", 3, {1, 1, 1}});
  site.args.push_back({"z", "dynamic", "f32", 1, {1}});
  site.args.push_back({"q", "dynamic", "f32", 1, {1}});
  EXPECT_TRUE(ValidateSite(site, &error)) << error;
  for (int i = 0; i < 11; ++i) {
    site.args.push_back({"e" + std::to_string(i), "dynamic", "f32", 1, {1}});
  }
  EXPECT_FALSE(ValidateSite(site, &error));
  EXPECT_NE(error.find("fourteen"), std::string::npos);
}

// ---------------------------------------------------------------------------
// ModelSignature: what the disk cache key must be sensitive to.
//
// A graph where Mul and MatMul can fold a constant pre-affine chain into the
// BMM kernel carries its initializer *values* in the key: two models that
// differ only in the folded constants must not share an artifact, or a
// session would execute kernels holding the other model's constants.
// Graphs without that pair keep the structural key (initializer data
// cleared), so an unrelated value change does not throw away an identical
// artifact.

void SetTensorType(onnx::ValueInfoProto* value) {
  auto* tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(onnx::TensorProto::FLOAT);
  for (int64_t dim : {1, 2, 3}) {
    tensor->mutable_shape()->add_dim()->set_dim_value(dim);
  }
}

onnx::ModelProto SignatureModel(bool foldable, float weightValue) {
  onnx::ModelProto model;
  model.set_ir_version(8);
  auto* graph = model.mutable_graph();
  graph->set_name("g");
  auto* x = graph->add_input();
  x->set_name("x");
  SetTensorType(x);
  auto* y = graph->add_output();
  y->set_name(foldable ? "mm" : "y");
  SetTensorType(y);
  auto* w = graph->add_initializer();
  w->set_name("w");
  w->set_data_type(onnx::TensorProto::FLOAT);
  for (int64_t dim : {1, 3, 3}) w->add_dims(dim);
  for (int i = 0; i < 9; ++i) {
    w->add_float_data(weightValue + static_cast<float>(i));
  }
  auto* node = graph->add_node();
  node->set_op_type(foldable ? "Mul" : "Add");
  node->add_input("x");
  node->add_input("w");
  node->add_output("y");
  if (foldable) {
    auto* mm = graph->add_node();
    mm->set_op_type("MatMul");
    mm->add_input("y");
    mm->add_input("w");
    mm->add_output("mm");
  }
  return model;
}

std::string WriteSignatureModel(const std::string& path, bool foldable,
                                float weightValue) {
  std::ofstream out(path, std::ios::binary);
  EXPECT_TRUE(out.is_open()) << path;
  EXPECT_TRUE(SignatureModel(foldable, weightValue).SerializeToOstream(&out));
  out.close();
  return path;
}

TEST(AnncEpModelSignature, ValuesEnterTheKeyWhenAFoldIsPossible) {
  const std::string dir = ::testing::TempDir();
  const std::string a = WriteSignatureModel(dir + "signature-a.onnx", true, 1.0f);
  const std::string b = WriteSignatureModel(dir + "signature-b.onnx", true, 2.0f);
  std::string error;
  const std::string keyA = ModelSignature(a, "toolchain", &error);
  EXPECT_FALSE(keyA.empty()) << error;
  EXPECT_EQ(keyA, ModelSignature(a, "toolchain", &error));  // deterministic
  // The whole point: same structure, different folded constants.
  EXPECT_NE(keyA, ModelSignature(b, "toolchain", &error));
  // The toolchain fingerprint is part of the key as well.
  EXPECT_NE(keyA, ModelSignature(a, "other-toolchain", &error));
  EXPECT_TRUE(ModelSignature(dir + "missing.onnx", "toolchain", &error).empty());
  std::remove(a.c_str());
  std::remove(b.c_str());
}

TEST(AnncEpModelSignature, StaysStructuralWithoutMulAndMatMul) {
  const std::string dir = ::testing::TempDir();
  const std::string a =
      WriteSignatureModel(dir + "signature-add-a.onnx", false, 1.0f);
  const std::string b =
      WriteSignatureModel(dir + "signature-add-b.onnx", false, 2.0f);
  std::string error;
  const std::string keyA = ModelSignature(a, "toolchain", &error);
  EXPECT_FALSE(keyA.empty()) << error;
  // No fold is possible in this graph, so the value change must not
  // invalidate the artifact.
  EXPECT_EQ(keyA, ModelSignature(b, "toolchain", &error));
  std::remove(a.c_str());
  std::remove(b.c_str());
}

}  // namespace
}  // namespace annc_ep
