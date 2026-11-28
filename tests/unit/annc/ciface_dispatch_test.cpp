#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "annc_ciface_dispatch.h"

namespace {

std::size_t g_calls = 0;
std::vector<void*> g_args;

// Parameter-pack aliases keep the 64-argument case readable.
#define ANNC_CIFACE_A1 void*
#define ANNC_CIFACE_A2 ANNC_CIFACE_A1, ANNC_CIFACE_A1
#define ANNC_CIFACE_A4 ANNC_CIFACE_A2, ANNC_CIFACE_A2
#define ANNC_CIFACE_A8 ANNC_CIFACE_A4, ANNC_CIFACE_A4
#define ANNC_CIFACE_A16 ANNC_CIFACE_A8, ANNC_CIFACE_A8
#define ANNC_CIFACE_A32 ANNC_CIFACE_A16, ANNC_CIFACE_A16
#define ANNC_CIFACE_A64 ANNC_CIFACE_A32, ANNC_CIFACE_A32

void f0() { ++g_calls; }

void f8(ANNC_CIFACE_A8) { ++g_calls; }

// Named parameters prove that the dispatch forwards the actual pointers.
void f10(void* a0, void* a1, void* a2, void* a3, void* a4, void* a5,
         void* a6, void* a7, void* a8, void* a9) {
  ++g_calls;
  g_args = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9};
}

void f14(ANNC_CIFACE_A8, ANNC_CIFACE_A4, ANNC_CIFACE_A2) { ++g_calls; }

void f64(ANNC_CIFACE_A64) { ++g_calls; }

template <typename Fn>
void* as_void_ptr(Fn fn) {
  return reinterpret_cast<void*>(fn);
}

TEST(CifaceDispatch, ZeroArguments) {
  g_calls = 0;
  EXPECT_TRUE(annc::ciface::Call(as_void_ptr(&f0), {}));
  EXPECT_EQ(g_calls, 1u);
}

TEST(CifaceDispatch, EightArguments) {
  g_calls = 0;
  const std::vector<void*> args(8, nullptr);
  EXPECT_TRUE(annc::ciface::Call(as_void_ptr(&f8), args));
  EXPECT_EQ(g_calls, 1u);
}

TEST(CifaceDispatch, TenArgumentsForwarded) {
  g_calls = 0;
  g_args.clear();
  int values[10] = {};
  std::vector<void*> args;
  for (int i = 0; i < 10; ++i) args.push_back(&values[i]);
  EXPECT_TRUE(annc::ciface::Call(as_void_ptr(&f10), args));
  EXPECT_EQ(g_calls, 1u);
  ASSERT_EQ(g_args.size(), args.size());
  for (std::size_t i = 0; i < args.size(); ++i) {
    EXPECT_EQ(g_args[i], args[i]);
  }
}

TEST(CifaceDispatch, FourteenArguments) {
  g_calls = 0;
  const std::vector<void*> args(14, nullptr);
  EXPECT_TRUE(annc::ciface::Call(as_void_ptr(&f14), args));
  EXPECT_EQ(g_calls, 1u);
}

TEST(CifaceDispatch, SixtyFourArguments) {
  g_calls = 0;
  const std::vector<void*> args(64, nullptr);
  EXPECT_TRUE(annc::ciface::Call(as_void_ptr(&f64), args));
  EXPECT_EQ(g_calls, 1u);
}

TEST(CifaceDispatch, SixtyFiveArgumentsRejected) {
  g_calls = 0;
  const std::vector<void*> args(65, nullptr);
  EXPECT_FALSE(annc::ciface::Call(as_void_ptr(&f64), args));
  EXPECT_EQ(g_calls, 0u);
}

}  // namespace
