#include "annc_jit_cache.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<int> packCalls{0};
void PrepareTestRhs(int64_t, const float *source, float *destination) {
  ++packCalls;
  std::copy(source, source + 16, destination);
}
AnncPackedRhsSpec TestSpec() { return {1, 1, 4, 4, 4, 4, 0, 4, 16, 64}; }

TEST(PackedRhsCacheTest, ReusesLayoutAndSeparatesNodeWeights) {
  annc::jit::PackedRhsCache first, second;
  float a[16], b[16];
  std::iota(a, a + 16, 1.0f);
  std::iota(b, b + 16, 2.0f);
  std::string error;
  packCalls = 0;
  auto s = TestSpec();
  auto one = first.GetOrPack(s, a, PrepareTestRhs, 0, &error);
  ASSERT_NE(one, nullptr) << error;
  auto two = second.GetOrPack(s, b, PrepareTestRhs, 0, &error);
  ASSERT_NE(two, nullptr) << error;
  EXPECT_EQ(first.GetOrPack(s, a, PrepareTestRhs, 0, &error), one);
  EXPECT_TRUE(std::equal(a, a + 16, static_cast<float *>(one->data)));
  EXPECT_TRUE(std::equal(b, b + 16, static_cast<float *>(two->data)));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(one->data) % 64, 0);
  s.kc = 2;
  auto other = first.GetOrPack(s, a, PrepareTestRhs, 0, &error);
  EXPECT_NE(other, one);
  EXPECT_EQ(first.GetOrPack(TestSpec(), a, PrepareTestRhs, 0, &error), one);
  EXPECT_EQ(packCalls, 3);
  EXPECT_EQ(first.size(), 2);
}

TEST(PackedRhsCacheTest, CompleteLayoutDeterminesReuse) {
  const auto original = TestSpec();
  for (auto field :
       {&AnncPackedRhsSpec::version, &AnncPackedRhsSpec::argument,
        &AnncPackedRhsSpec::k, &AnncPackedRhsSpec::n, &AnncPackedRhsSpec::kc,
        &AnncPackedRhsSpec::nc, &AnncPackedRhsSpec::isa,
        &AnncPackedRhsSpec::panel, &AnncPackedRhsSpec::elements,
        &AnncPackedRhsSpec::alignment}) {
    auto changed = original;
    ++(changed.*field);
    EXPECT_FALSE(annc::jit::SamePackedRhsLayout(original, changed));
  }
  EXPECT_TRUE(annc::jit::SamePackedRhsLayout(original, original));
}

TEST(PackedRhsCacheTest, ConcurrentFirstUsePacksOnceAndBufferOutlivesCache) {
  auto cache = std::make_unique<annc::jit::PackedRhsCache>();
  float a[16] = {3};
  packCalls = 0;
  std::vector<std::future<std::shared_ptr<annc::jit::PackedRhsBuffer>>> calls;
  for (int i = 0; i < 8; ++i)
    calls.push_back(std::async(std::launch::async, [&] {
      std::string error;
      return cache->GetOrPack(TestSpec(), a, PrepareTestRhs, 0, &error);
    }));
  auto first = calls.front().get();
  ASSERT_NE(first, nullptr);
  for (size_t i = 1; i < calls.size(); ++i) EXPECT_EQ(calls[i].get(), first);
  EXPECT_EQ(packCalls, 1);
  cache.reset();
  EXPECT_EQ(static_cast<float *>(first->data)[0], 3);
}

TEST(PackedRhsCacheTest, RejectsInvalidLayoutWithoutPublishing) {
  annc::jit::PackedRhsCache cache;
  float a[16] = {};
  std::string error;
  auto spec = TestSpec();
  spec.elements = 15;
  EXPECT_EQ(cache.GetOrPack(spec, a, PrepareTestRhs, 0, &error), nullptr);
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(cache.size(), 0);
}

using annc::jit::BuildJitCacheKey;
using annc::jit::CacheEvent;
using annc::jit::JitArgumentSignature;
using annc::jit::JitCacheKey;
using annc::jit::JitCompilationCache;
using annc::jit::JitCompileResult;
using annc::jit::JitExecutable;

std::shared_ptr<JitExecutable> MakeExecutable(
    std::string workDir = std::string()) {
  return std::make_shared<JitExecutable>(std::move(workDir), "", nullptr,
                                         nullptr, nullptr, nullptr, false);
}

JitCacheKey TestKey(const std::string &value) { return JitCacheKey{value}; }

JitCacheKey BuildTestCacheKey(
    const std::string &templateFingerprint,
    const std::vector<JitArgumentSignature> &arguments) {
  JitCacheKey key;
  std::string error;
  EXPECT_TRUE(BuildJitCacheKey(templateFingerprint, arguments, &key, &error))
      << error;
  return key;
}

TEST(JitCacheKeyTest, SameCompilationInputsProduceSameKey) {
  std::vector<JitArgumentSignature> arguments = {
      {{5, 128}}, {{128, 64}}, {{5, 64}}};

  EXPECT_EQ(BuildTestCacheKey("template", arguments),
            BuildTestCacheKey("template", arguments));
}

TEST(JitCacheKeyTest, RuntimeShapeChangesKey) {
  std::vector<JitArgumentSignature> batch5 = {
      {{5, 128}}, {{128, 64}}, {{5, 64}}};
  std::vector<JitArgumentSignature> batch8 = {
      {{8, 128}}, {{128, 64}}, {{8, 64}}};

  EXPECT_NE(BuildTestCacheKey("template", batch5),
            BuildTestCacheKey("template", batch8));
}

TEST(JitCacheKeyTest, InvalidInputsDoNotProduceAKey) {
  JitCacheKey key;
  std::string error;
  EXPECT_FALSE(BuildJitCacheKey("", {{{5, 8}}}, &key, &error));
  EXPECT_TRUE(key.digest.empty());
  EXPECT_FALSE(error.empty());
}

TEST(JitCompilationCacheTest, SequentialRequestsCompileOnce) {
  JitCompilationCache cache(4);
  std::atomic<int> compileCount{0};
  auto compile = [&] {
    ++compileCount;
    return JitCompileResult{MakeExecutable(), ""};
  };

  auto first = cache.GetOrCompile(TestKey("same"), compile);
  auto second = cache.GetOrCompile(TestKey("same"), compile);

  ASSERT_TRUE(first.ok()) << first.error;
  ASSERT_TRUE(second.ok()) << second.error;
  EXPECT_EQ(compileCount.load(), 1);
  EXPECT_EQ(first.executable, second.executable);
  EXPECT_EQ(first.event, CacheEvent::kMiss);
  EXPECT_EQ(second.event, CacheEvent::kHit);
}

TEST(JitCompilationCacheTest, ConcurrentRequestsUseSingleFlight) {
  JitCompilationCache cache(4);
  std::atomic<int> compileCount{0};
  std::promise<void> ownerStarted;
  std::shared_future<void> releaseOwner =
      std::async(std::launch::deferred, [] {}).share();
  std::promise<void> releasePromise;
  releaseOwner = releasePromise.get_future().share();
  auto compile = [&] {
    ++compileCount;
    ownerStarted.set_value();
    releaseOwner.wait();
    return JitCompileResult{MakeExecutable(), ""};
  };

  auto first = std::async(std::launch::async, [&] {
    return cache.GetOrCompile(TestKey("same"), compile);
  });
  ownerStarted.get_future().wait();
  auto second = std::async(std::launch::async, [&] {
    return cache.GetOrCompile(TestKey("same"), compile);
  });
  while (cache.stats().waits == 0) std::this_thread::yield();
  releasePromise.set_value();

  auto firstResult = first.get();
  auto secondResult = second.get();
  ASSERT_TRUE(firstResult.ok()) << firstResult.error;
  ASSERT_TRUE(secondResult.ok()) << secondResult.error;
  EXPECT_EQ(compileCount.load(), 1);
  EXPECT_EQ(firstResult.executable, secondResult.executable);
  EXPECT_TRUE(firstResult.event == CacheEvent::kMiss ||
              secondResult.event == CacheEvent::kMiss);
  EXPECT_TRUE(firstResult.event == CacheEvent::kWait ||
              secondResult.event == CacheEvent::kWait);
}

TEST(JitCompilationCacheTest, DifferentKeysCompileConcurrently) {
  JitCompilationCache cache(4);
  std::atomic<int> active{0};
  std::atomic<int> maxActive{0};
  auto compile = [&] {
    int now = ++active;
    int observed = maxActive.load();
    while (now > observed && !maxActive.compare_exchange_weak(observed, now)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    --active;
    return JitCompileResult{MakeExecutable(), ""};
  };

  auto first = std::async(std::launch::async, [&] {
    return cache.GetOrCompile(TestKey("first"), compile);
  });
  auto second = std::async(std::launch::async, [&] {
    return cache.GetOrCompile(TestKey("second"), compile);
  });

  EXPECT_TRUE(first.get().ok());
  EXPECT_TRUE(second.get().ok());
  EXPECT_EQ(maxActive.load(), 2);
}

TEST(JitCompilationCacheTest, FailureIsSharedAndNextRequestRetries) {
  JitCompilationCache cache(4);
  std::atomic<int> compileCount{0};
  auto failingCompile = [&] {
    ++compileCount;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    return JitCompileResult{nullptr, "compile failed"};
  };

  auto first = std::async(std::launch::async, [&] {
    return cache.GetOrCompile(TestKey("retry"), failingCompile);
  });
  auto second = std::async(std::launch::async, [&] {
    return cache.GetOrCompile(TestKey("retry"), failingCompile);
  });
  while (cache.stats().waits == 0) std::this_thread::yield();
  auto firstResult = first.get();
  auto secondResult = second.get();

  EXPECT_FALSE(firstResult.ok());
  EXPECT_FALSE(secondResult.ok());
  EXPECT_EQ(firstResult.error, "compile failed");
  EXPECT_EQ(secondResult.error, "compile failed");
  EXPECT_EQ(compileCount.load(), 1);

  auto retry = cache.GetOrCompile(TestKey("retry"), [&] {
    ++compileCount;
    return JitCompileResult{MakeExecutable(), ""};
  });
  EXPECT_TRUE(retry.ok()) << retry.error;
  EXPECT_EQ(compileCount.load(), 2);
}

TEST(JitCompilationCacheTest, CallbackExceptionIsSharedAndNextRequestRetries) {
  JitCompilationCache cache(4);
  std::atomic<int> compileCount{0};
  std::promise<void> ownerStarted;
  std::promise<void> releasePromise;
  std::shared_future<void> release = releasePromise.get_future().share();
  auto throwingCompile = [&] {
    ++compileCount;
    ownerStarted.set_value();
    release.wait();
    throw std::runtime_error("callback failure");
    return JitCompileResult{nullptr, "unreachable"};
  };

  auto first = std::async(std::launch::async, [&] {
    return cache.GetOrCompile(TestKey("throw"), throwingCompile);
  });
  ownerStarted.get_future().wait();
  auto second = std::async(std::launch::async, [&] {
    return cache.GetOrCompile(TestKey("throw"), throwingCompile);
  });
  while (cache.stats().waits == 0) std::this_thread::yield();
  releasePromise.set_value();

  auto firstResult = first.get();
  auto secondResult = second.get();
  EXPECT_FALSE(firstResult.ok());
  EXPECT_FALSE(secondResult.ok());
  EXPECT_NE(firstResult.error.find("callback failure"), std::string::npos);
  EXPECT_EQ(firstResult.error, secondResult.error);
  EXPECT_EQ(compileCount.load(), 1);

  auto retry = cache.GetOrCompile(TestKey("throw"), [&] {
    ++compileCount;
    return JitCompileResult{MakeExecutable(), ""};
  });
  EXPECT_TRUE(retry.ok()) << retry.error;
  EXPECT_EQ(compileCount.load(), 2);
}

TEST(JitCompilationCacheTest, EvictionPreservesActiveExecutableLifetime) {
  JitCompilationCache cache(1);
  std::filesystem::path workDir =
      std::filesystem::temp_directory_path() /
      ("annc_jit_cache_lifetime_" + std::to_string(getpid()));
  std::filesystem::create_directories(workDir);
  auto first = cache.GetOrCompile(TestKey("first"), [&] {
    return JitCompileResult{MakeExecutable(workDir.string()), ""};
  });
  ASSERT_TRUE(first.ok()) << first.error;
  std::weak_ptr<JitExecutable> lifetime = first.executable;

  auto second = cache.GetOrCompile(TestKey("second"), [&] {
    return JitCompileResult{MakeExecutable(), ""};
  });
  ASSERT_TRUE(second.ok()) << second.error;
  ASSERT_EQ(second.evictedKeys.size(), 1U);
  EXPECT_EQ(second.evictedKeys.front(), "first");
  EXPECT_EQ(cache.size(), 1U);
  EXPECT_FALSE(lifetime.expired());
  EXPECT_TRUE(std::filesystem::exists(workDir));

  first.executable.reset();
  EXPECT_TRUE(lifetime.expired());
  EXPECT_FALSE(std::filesystem::exists(workDir));
}

}  // namespace
