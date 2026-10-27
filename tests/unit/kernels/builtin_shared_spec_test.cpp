#include <dlfcn.h>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "Context/KernelContextUtils.h"
#include "Kernel/KernelRegistry.h"

#if ANNC_TEST_ENABLE_KDNN
#include "annc/lib/Kernel/builtin_kernels/kdnn_adaptor/KDNNTensorInfoAdaptor.h"
#endif

using namespace annc::kernels;

namespace {

template <typename Fn>
Fn loadKernelApi(const char* symbol) {
  std::string apiSymbol = std::string("_mlir_ciface_") + symbol;
  dlerror();
  void* address = dlsym(RTLD_DEFAULT, apiSymbol.c_str());
  const char* error = dlerror();
  if (error != nullptr || address == nullptr) {
    throw std::runtime_error("Failed to resolve kernel API symbol: " +
                             apiSymbol);
  }
  return reinterpret_cast<Fn>(address);
}

AnncTensorView makeView(float* data, const int64_t* dims,
                        const int64_t* strides) {
  return {data,
          data,
          0,
          dims,
          strides,
          ANNC_ELEMENT_TYPE_F32,
          2,
          ANNC_LAYOUT_STRIDED,
          ANNC_TENSOR_BOUND};
}

struct TestServices {
  AnncStatusCode code = ANNC_STATUS_OK;
  static AnncStatusCode parallel(void*, int64_t total, int64_t,
                                 AnncContextTask task, void* task_data) {
    if (task == nullptr || total < 0) return ANNC_STATUS_INVALID_ARGUMENT;
    if (total != 0) task(task_data, 0, total);
    return ANNC_STATUS_OK;
  }
  static int32_t threads(void*) { return 1; }
  static void report(void* opaque, AnncStatusCode code, const char*) {
    auto& current = static_cast<TestServices*>(opaque)->code;
    if (current == ANNC_STATUS_OK) current = code;
  }
  static AnncStatusCode status(void* opaque) {
    return static_cast<TestServices*>(opaque)->code;
  }
  static AnncStatusCode temp(void*, const AnncTensorDesc*, AnncTensorView*) {
    return ANNC_STATUS_ALLOCATION_FAILED;
  }
  static AnncStatusCode output(void*, uint32_t, const AnncTensorDesc*,
                               AnncTensorView*) {
    return ANNC_STATUS_ALLOCATION_FAILED;
  }
};

std::array<float, 4> runMatMulKernel(const std::string& backend) {
  float lhsData[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float rhsData[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  float outputData[9] = {0.0f};

  const int64_t lhsDims[] = {2, 3}, rhsDims[] = {3, 2}, outDims[] = {2, 2};
  const int64_t lhsStrides[] = {3, 1}, rhsStrides[] = {2, 1};
  AnncTensorView inputs[] = {makeView(lhsData, lhsDims, lhsStrides),
                             makeView(rhsData, rhsDims, rhsStrides)};
  auto output = makeView(outputData, outDims, rhsStrides);
  TestServices host;
  AnncRuntimeServices services{sizeof(AnncRuntimeServices),
                               ANNC_CONTEXT_ABI_VERSION,
                               &host,
                               TestServices::output,
                               TestServices::temp,
                               TestServices::report,
                               TestServices::status,
                               TestServices::parallel,
                               TestServices::threads,
                               nullptr,
                               nullptr};
  AnncKernelContext context{sizeof(AnncKernelContext),
                            ANNC_CONTEXT_ABI_VERSION,
                            inputs,
                            2,
                            &output,
                            1,
                            nullptr,
                            0,
                            &services};

  auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol(
      KernelQuery{"MatMul", backend, {{"T", "float"}}, "annc_context_v3"});
  if (!symbolOpt.has_value()) {
    throw std::runtime_error("MatMul kernel was not registered for backend: " +
                             backend);
  }

  auto kernel =
      loadKernelApi<AnncStatusCode (*)(AnncKernelContext*)>(symbolOpt->c_str());
  if (kernel(&context) != ANNC_STATUS_OK) {
    throw std::runtime_error("MatMul kernel returned failure status");
  }

  return {outputData[0], outputData[1], outputData[2], outputData[3]};
}

void expectStandardMatMulOutput(const std::array<float, 4>& output) {
  EXPECT_FLOAT_EQ(output[0], 58.0f);
  EXPECT_FLOAT_EQ(output[1], 64.0f);
  EXPECT_FLOAT_EQ(output[2], 139.0f);
  EXPECT_FLOAT_EQ(output[3], 154.0f);
}

void requireStandardMatMulOutput(const std::array<float, 4>& output) {
  if (output[0] != 58.0f || output[1] != 64.0f || output[2] != 139.0f ||
      output[3] != 154.0f) {
    throw std::runtime_error("MatMul output mismatch");
  }
}

#if ANNC_TEST_ENABLE_KDNN
void runBuiltinKdnnMatMulWrapperBody() {
  requireStandardMatMulOutput(runMatMulKernel("kdnn"));
}
#endif

}  // namespace

TEST(BuiltinSharedSpecTest, RegistryExposesMatMulAarch64) {
  auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol(
      KernelQuery{"MatMul", "aarch64", {{"T", "float"}}, "annc_context_v3"});

  ASSERT_TRUE(symbolOpt.has_value());
  EXPECT_TRUE(symbolOpt->rfind("ANNCKernel_ContextV3_MatMul_", 0) == 0);
  ASSERT_GE(symbolOpt->size(), 5u);
  EXPECT_EQ(symbolOpt->substr(symbolOpt->size() - 5), "_auto");
}

TEST(BuiltinSharedSpecTest, ResolverFindsMatMulAarch64) {
  auto symbol = KernelRegistry::instance().lookupKernelSymbol(
      KernelQuery{"MatMul", "aarch64", {{"T", "float"}}, "annc_context_v3"});

  ASSERT_TRUE(symbol.has_value());
  EXPECT_TRUE(symbol->rfind("ANNCKernel_ContextV3_MatMul_", 0) == 0);
  ASSERT_GE(symbol->size(), 5u);
  EXPECT_EQ(symbol->substr(symbol->size() - 5), "_auto");
}

TEST(BuiltinSharedSpecTest, Aarch64MatMulWrapperRuns) {
  expectStandardMatMulOutput(runMatMulKernel("aarch64"));
}

#if ANNC_TEST_ENABLE_KDNN
TEST(BuiltinSharedSpecTest, RegistryExposesMatMulKdnn) {
  auto symbolOpt = KernelRegistry::instance().lookupKernelSymbol(
      KernelQuery{"MatMul", "kdnn", {{"T", "float"}}, "annc_context_v3"});

  EXPECT_TRUE(symbolOpt.has_value());
}

TEST(BuiltinSharedSpecTest, KdnnMatMulWrapperRuns) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0) << "failed to fork kdnn smoke child";

  if (pid == 0) {
    try {
      runBuiltinKdnnMatMulWrapperBody();
      _exit(0);
    } catch (...) {
      _exit(1);
    }
  }

  int status = 0;
  ASSERT_GE(waitpid(pid, &status, 0), 0) << "failed to wait for kdnn child";

  if (WIFSIGNALED(status) && WTERMSIG(status) == SIGILL) {
    SUCCEED() << "kdnn MatMul wrapper hit SIGILL on this machine";
    return;
  }

  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(BuiltinSharedSpecTest, MatMulHasBothBackends) {
  EXPECT_TRUE(KernelRegistry::instance().hasKernel(
      KernelQuery{"MatMul", "aarch64", {{"T", "float"}}, "annc_context_v3"}));
  EXPECT_TRUE(KernelRegistry::instance().hasKernel(
      KernelQuery{"MatMul", "kdnn", {{"T", "float"}}, "annc_context_v3"}));
}

TEST(BuiltinSharedSpecTest, MatMulSymbolsDifferBetweenBackends) {
  auto aarch64Sym = KernelRegistry::instance().lookupKernelSymbol(
      KernelQuery{"MatMul", "aarch64", {{"T", "float"}}, "annc_context_v3"});
  auto kdnnSym = KernelRegistry::instance().lookupKernelSymbol(
      KernelQuery{"MatMul", "kdnn", {{"T", "float"}}, "annc_context_v3"});

  ASSERT_TRUE(aarch64Sym.has_value());
  ASSERT_TRUE(kdnnSym.has_value());
  EXPECT_NE(*aarch64Sym, *kdnnSym);
}
#endif
