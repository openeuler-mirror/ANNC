#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <future>
#include <memory>
#include <numeric>
#include <vector>

#include ANNC_PACKED_RHS_HEADER

#if defined(__aarch64__) && defined(__linux__)
#include <sys/prctl.h>
#endif

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

TEST(PackedRhsCacheTest, RejectsMismatchedSveVectorLengthBeforePacking) {
  auto spec = TestSpec();
  spec.isa = 1;
  spec.panel = 32;
#if defined(__aarch64__) && defined(__linux__)
  const int vl = prctl(PR_SVE_GET_VL, 0, 0, 0, 0);
  if (vl >= 0 && (vl & PR_SVE_VL_LEN_MASK) == spec.panel) spec.panel = 16;
#endif
  spec.elements = spec.k * spec.panel;
  ASSERT_TRUE(annc::jit::ValidatePackedRhsSpec(spec));
  annc::jit::PackedRhsCache cache;
  float source[16] = {};
  std::string error;
  packCalls = 0;
  EXPECT_EQ(cache.GetOrPack(spec, source, PrepareTestRhs, 0, &error), nullptr);
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(packCalls, 0);
  EXPECT_EQ(cache.size(), 0);
}

}  // namespace
