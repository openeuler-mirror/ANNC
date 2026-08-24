#include <gtest/gtest.h>

extern "C" void annc_aarch64_neon_kernel_rm_mr1_n1_k1_f32(
    const float *, const float *, float *, int, int, int, int, int);
extern "C" void annc_aarch64_neon_kernel_rm_mr1_n1_k1_acc_f32(
    const float *, const float *, float *, int, int, int, int, int);

TEST(GemmSmallShapeMicrokernel, NeonRowMajorAbi) {
  const float a[] = {2.0f};
  const float b[] = {3.0f};
  float c[] = {0.0f};
  annc_aarch64_neon_kernel_rm_mr1_n1_k1_f32(a, b, c, 1, 1, 1, 1, 1);
  EXPECT_FLOAT_EQ(c[0], 6.0f);

  c[0] = 4.0f;
  annc_aarch64_neon_kernel_rm_mr1_n1_k1_acc_f32(a, b, c, 1, 1, 1, 1, 1);
  EXPECT_FLOAT_EQ(c[0], 10.0f);
}
