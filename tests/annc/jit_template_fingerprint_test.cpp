#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "Dialect/Atir/AtirOps.h"
#include "Dialect/Atir/TemplateFingerprint.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace {

std::string BuildKernel(llvm::StringRef functionName,
                        llvm::StringRef tensorPrefix, bool rightTranspose,
                        float reluLimit,
                        llvm::StringRef codegenOption = "default") {
  return R"mlir(
module {
  func.func private @)mlir" +
         functionName.str() + R"mlir((
      %lhs: !atir.tensor<?x128xf32, name = ")mlir" +
         tensorPrefix.str() + R"mlir(/lhs">,
      %rhs: !atir.tensor<128x64xf32, name = ")mlir" +
         tensorPrefix.str() + R"mlir(/rhs">,
      %out: !atir.tensor<?x64xf32, name = ")mlir" +
         tensorPrefix.str() + R"mlir(/out">)
      attributes {
        annc.kernel,
        fusion.metadata = {
          kernel_name = ")mlir" +
         functionName.str() + R"mlir(",
          tf.name = ")mlir" +
         tensorPrefix.str() + R"mlir(/cluster"
        },
        fusion.pattern = "matmul",
        llvm.emit_c_interface
      } {
    %buffer = "atir.buffer"() :
      () -> !atir.tensor<?x64xf32, name = ")mlir" +
         tensorPrefix.str() + R"mlir(/out">
    %result = "atir.MatMul"(%buffer, %lhs, %rhs) <{
      do_relu = false,
      left_transpose = false,
      output_transpose = false,
      relu_limit = )mlir" +
         std::to_string(reluLimit) + R"mlir( : f32,
      right_transpose = )mlir" +
         (rightTranspose ? "true" : "false") + R"mlir(,
      withBias = false
    }> {metadata = {"codegen.option" = ")mlir" +
         codegenOption.str() + R"mlir(", "tf.name" = ")mlir" +
         tensorPrefix.str() + R"mlir(/matmul"}} : (
      !atir.tensor<?x64xf32, name = ")mlir" +
         tensorPrefix.str() + R"mlir(/out">,
      !atir.tensor<?x128xf32, name = ")mlir" +
         tensorPrefix.str() + R"mlir(/lhs">,
      !atir.tensor<128x64xf32, name = ")mlir" +
         tensorPrefix.str() + R"mlir(/rhs">
    ) -> !atir.tensor<?x64xf32, name = ")mlir" +
         tensorPrefix.str() + R"mlir(/out">
    return
  }
}
)mlir";
}

std::string ReplaceAll(std::string text, llvm::StringRef from,
                       llvm::StringRef to) {
  size_t position = 0;
  while ((position = text.find(from.str(), position)) != std::string::npos) {
    text.replace(position, from.size(), to.str());
    position += to.size();
  }
  return text;
}

class TemplateFingerprintTest : public testing::Test {
 protected:
  TemplateFingerprintTest() {
    registry_.insert<mlir::func::FuncDialect, atir::AtirDialect>();
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
  }

  std::string Fingerprint(const std::string &moduleText,
                          std::optional<int64_t> intraThreadCount = {}) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(moduleText, &context_);
    EXPECT_TRUE(module);
    if (!module) return "";
    if (intraThreadCount) {
      (*module)->setAttr(
          "annc.intra_thread_count",
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context_, 64),
                                 *intraThreadCount));
    }
    mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
    return atir::computeAtirTemplateFingerprint(*module, function);
  }

 private:
  mlir::DialectRegistry registry_;
  mlir::MLIRContext context_;
};

TEST_F(TemplateFingerprintTest, IgnoresFunctionAndTensorflowIdentity) {
  EXPECT_EQ(Fingerprint(BuildKernel("dense_a", "model/a", false, -1.0f)),
            Fingerprint(BuildKernel("dense_b", "model/b", false, -1.0f)));
}

TEST_F(TemplateFingerprintTest,
       IgnoresConcreteDimensionsAndStaticDynamicSpelling) {
  std::string dynamic = BuildKernel("dense_a", "model/a", false, -1.0f);
  std::string staticShape = ReplaceAll(dynamic, "?x128", "5x128");
  staticShape = ReplaceAll(staticShape, "?x64", "5x64");

  EXPECT_EQ(Fingerprint(dynamic), Fingerprint(staticShape));
}

TEST_F(TemplateFingerprintTest, IncludesTransposeSemantics) {
  EXPECT_NE(Fingerprint(BuildKernel("dense_a", "model/a", false, -1.0f)),
            Fingerprint(BuildKernel("dense_b", "model/b", true, -1.0f)));
}

TEST_F(TemplateFingerprintTest, IncludesCodegenAttributes) {
  EXPECT_NE(Fingerprint(BuildKernel("dense_a", "model/a", false, -1.0f)),
            Fingerprint(BuildKernel("dense_b", "model/b", false, 6.0f)));
}

TEST_F(TemplateFingerprintTest, PreservesNonIdentityMetadata) {
  EXPECT_NE(
      Fingerprint(BuildKernel("dense_a", "model/a", false, -1.0f, "option-a")),
      Fingerprint(BuildKernel("dense_b", "model/b", false, -1.0f, "option-b")));
}

TEST_F(TemplateFingerprintTest, IncludesModuleIntraThreadCount) {
  const std::string module = BuildKernel("dense_a", "model/a", false, -1.0f);
  EXPECT_EQ(Fingerprint(module, 4), Fingerprint(module, 4));
  EXPECT_NE(Fingerprint(module, 4), Fingerprint(module, 16));
  EXPECT_EQ(Fingerprint(module), Fingerprint(module, 1));
}

}  // namespace
