// Unit tests for the execution-context ABI helpers.
//
// The v3 context carries the host thread pool itself; the v2 layout (what a
// host built against the previous header hands over) stays accepted and falls
// back to the pool injected into the kernel library's TLS -- the TensorFlow
// channel. Reading `thread_pool` off a v2 context would overrun the host's
// object, so the helpers must gate the field on struct_size and abi_version.

#include <cstdint>
#include <cstddef>
#include <functional>
#include <utility>

#include <gtest/gtest.h>

#include "Kernel/ExecutionContext.h"
#include "Kernel/ExecutionContextUtils.h"
#include "Support/ThreadPool/ThreadPool.h"

namespace {

using annc::kernels::execution::resolveThreadPool;
using annc::kernels::execution::threadPool;
using annc::kernels::execution::validateContext;

// Never runs anything: these tests only exercise how a kernel finds the pool,
// not how the pool executes work.
class StubPool final : public annc::threadpool::AnncThreadPool {
 public:
  int num_threads() const override { return 4; }
  bool in_parallel_region() const override { return false; }
  void parallel_for(
      std::int64_t total, const annc::threadpool::ParallelForOptions&,
      const std::function<void(std::int64_t, std::int64_t)>& fn) override {
    if (total > 0) fn(0, total);
  }
};

AnncStatusCode StubAllocateOutput(AnncExecutionHandle*, uint32_t,
                                  AnncTensorDesc*) {
  return ANNC_STATUS_OK;
}
AnncStatusCode StubAllocateTemp(AnncExecutionHandle*, AnncTensorDesc*) {
  return ANNC_STATUS_OK;
}
void StubReportError(AnncExecutionHandle*, AnncStatusCode, const char*) {}

AnncExecutionHandle* DummyHandle() {
  return reinterpret_cast<AnncExecutionHandle*>(0x1);
}

AnncExecutionContext MakeV3(AnncThreadPoolHandle* pool) {
  AnncExecutionContext context{};
  context.struct_size = sizeof(AnncExecutionContext);
  context.abi_version = ANNC_EXECUTION_ABI_VERSION;
  context.handle = DummyHandle();
  context.allocate_output = &StubAllocateOutput;
  context.allocate_temp = &StubAllocateTemp;
  context.report_error = &StubReportError;
  context.thread_pool = pool;
  return context;
}

// The v2 host layout: the fields before `thread_pool`, at the same offsets as
// the v3 prefix. An old host reports this size.
struct V2HostContext {
  uint32_t struct_size;
  uint32_t abi_version;
  AnncExecutionHandle* handle;
  AnncAllocateOutputCallback allocate_output;
  AnncAllocateTempCallback allocate_temp;
  AnncReportErrorCallback report_error;
};

V2HostContext MakeV2() {
  V2HostContext context{};
  context.struct_size = ANNC_EXECUTION_V2_CONTEXT_SIZE;
  context.abi_version = ANNC_EXECUTION_V2_ABI_VERSION;
  context.handle = DummyHandle();
  context.allocate_output = &StubAllocateOutput;
  context.allocate_temp = &StubAllocateTemp;
  context.report_error = &StubReportError;
  return context;
}

bool Rejected(const AnncExecutionContext* context) {
  llvm::Error error = validateContext(context);
  if (!error) return false;
  llvm::consumeError(std::move(error));
  return true;
}

class ExecutionContextTest : public ::testing::Test {
 protected:
  void TearDown() override { annc::threadpool::clearCurrentThreadPool(); }
};

TEST_F(ExecutionContextTest, V2LayoutMatchesV3Prefix) {
  EXPECT_EQ(offsetof(V2HostContext, struct_size),
            offsetof(AnncExecutionContext, struct_size));
  EXPECT_EQ(offsetof(V2HostContext, abi_version),
            offsetof(AnncExecutionContext, abi_version));
  EXPECT_EQ(offsetof(V2HostContext, handle),
            offsetof(AnncExecutionContext, handle));
  EXPECT_EQ(offsetof(V2HostContext, allocate_output),
            offsetof(AnncExecutionContext, allocate_output));
  EXPECT_EQ(offsetof(V2HostContext, allocate_temp),
            offsetof(AnncExecutionContext, allocate_temp));
  EXPECT_EQ(offsetof(V2HostContext, report_error),
            offsetof(AnncExecutionContext, report_error));
  EXPECT_EQ(sizeof(V2HostContext), ANNC_EXECUTION_V2_CONTEXT_SIZE);
  EXPECT_GT(sizeof(AnncExecutionContext), sizeof(V2HostContext));
}

TEST_F(ExecutionContextTest, V3CarriesThePool) {
  StubPool context_pool;
  StubPool tls_pool;
  AnncExecutionContext context = MakeV3(
      reinterpret_cast<AnncThreadPoolHandle*>(&context_pool));
  EXPECT_FALSE(Rejected(&context));
  EXPECT_EQ(threadPool(&context), &context_pool);
  // The context pool wins over any TLS pool the host also injected.
  annc::threadpool::setCurrentThreadPool(&tls_pool);
  EXPECT_EQ(resolveThreadPool(&context), &context_pool);
}

TEST_F(ExecutionContextTest, V3WithoutPoolFallsBackToTls) {
  AnncExecutionContext context = MakeV3(nullptr);
  EXPECT_FALSE(Rejected(&context));
  EXPECT_EQ(threadPool(&context), nullptr);
  EXPECT_EQ(resolveThreadPool(&context), nullptr);
  StubPool tls_pool;
  annc::threadpool::setCurrentThreadPool(&tls_pool);
  EXPECT_EQ(resolveThreadPool(&context), &tls_pool);
}

TEST_F(ExecutionContextTest, V2FallsBackToTls) {
  V2HostContext v2 = MakeV2();
  const auto* context = reinterpret_cast<const AnncExecutionContext*>(&v2);
  EXPECT_FALSE(Rejected(context));
  EXPECT_EQ(threadPool(context), nullptr);
  EXPECT_EQ(resolveThreadPool(context), nullptr);
  StubPool tls_pool;
  annc::threadpool::setCurrentThreadPool(&tls_pool);
  EXPECT_EQ(resolveThreadPool(context), &tls_pool);
}

TEST_F(ExecutionContextTest, V2WithoutTlsFallsBackToNull) {
  V2HostContext v2 = MakeV2();
  const auto* context = reinterpret_cast<const AnncExecutionContext*>(&v2);
  EXPECT_FALSE(Rejected(context));
  EXPECT_EQ(resolveThreadPool(context), nullptr);
}

TEST_F(ExecutionContextTest, RejectsMismatchedSizeOrVersion) {
  EXPECT_TRUE(Rejected(nullptr));

  AnncExecutionContext context = MakeV3(nullptr);
  context.abi_version = ANNC_EXECUTION_V2_ABI_VERSION;
  EXPECT_TRUE(Rejected(&context));  // v3 size with a v2 version

  V2HostContext v2 = MakeV2();
  v2.abi_version = ANNC_EXECUTION_ABI_VERSION;
  EXPECT_TRUE(Rejected(reinterpret_cast<const AnncExecutionContext*>(&v2)));

  context = MakeV3(nullptr);
  context.struct_size = ANNC_EXECUTION_V2_CONTEXT_SIZE;
  EXPECT_TRUE(Rejected(&context));  // v2 size with a v3 version

  context = MakeV3(nullptr);
  context.struct_size = sizeof(uint64_t);
  EXPECT_TRUE(Rejected(&context));
}

TEST_F(ExecutionContextTest, RejectsMissingCallbacks) {
  AnncExecutionContext context = MakeV3(nullptr);
  context.handle = nullptr;
  EXPECT_TRUE(Rejected(&context));

  context = MakeV3(nullptr);
  context.allocate_output = nullptr;
  EXPECT_TRUE(Rejected(&context));

  context = MakeV3(nullptr);
  context.allocate_temp = nullptr;
  EXPECT_TRUE(Rejected(&context));

  context = MakeV3(nullptr);
  context.report_error = nullptr;
  EXPECT_TRUE(Rejected(&context));
}

}  // namespace
