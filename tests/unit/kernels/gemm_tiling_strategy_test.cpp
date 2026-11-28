#include "GemmStrategy.h"
#include "GemmPlan.h"
#include "GemmPacking.h"

#include <gtest/gtest.h>

namespace annc::aarch64::gemm {
namespace {

TEST(GemmTilingStrategyTest, FixedCachePlanRetainsReductionRoles) {
  const GemmCacheStrategy strategy{GemmCacheStrategyKind::kFixed,
                                  FixedCacheParameters{6, 16, 4}};
  const GemmCacheTilingPlan plan =
      getGemmCacheTilingStrategy(strategy.kind)
          .buildPlan({strategy, {12, 20, 10}});

  ASSERT_EQ(plan.rows.size(), 1u);
  EXPECT_EQ(plan.rows[0].extent, 6);
  EXPECT_EQ(plan.rows[0].count, 2);
  ASSERT_EQ(plan.columns.size(), 2u);
  EXPECT_EQ(plan.columns[0].extent, 16);
  EXPECT_EQ(plan.columns[1].extent, 4);
  ASSERT_EQ(plan.reduction.size(), 3u);
  EXPECT_EQ(plan.reduction[0].role, GemmKRole::kFirst);
  EXPECT_EQ(plan.reduction[1].role, GemmKRole::kMiddle);
  EXPECT_EQ(plan.reduction[2].role, GemmKRole::kFinal);
}

TEST(GemmTilingStrategyTest, FixedKernelPlanDescribesTileGroups) {
  const GemmKernelStrategy strategy{GemmKernelStrategyKind::kFixed,
                                    FixedKernelParameters{4, 2}};
  const GemmKernelTilingPlan plan =
      getGemmKernelTilingStrategy(strategy.kind)
          .buildPlan({strategy, {40, 35, 16}, {32, GemmDataType::kF32,
                                                GemmExecutionKind::kGemm}});

  ASSERT_EQ(plan.groups.size(), 1u);
  const auto &group = plan.groups[0];
  EXPECT_EQ(group.order, GemmKernelOrder::kRowsFirst);
  EXPECT_EQ(group.outer.offset, 0);
  EXPECT_EQ(group.outer.extent, 4);
  EXPECT_EQ(group.outer.count, 10);
  ASSERT_EQ(group.inner.size(), 2u);
  EXPECT_EQ(group.inner[0].offset, 0);
  EXPECT_EQ(group.inner[0].extent, 16);
  EXPECT_EQ(group.inner[0].count, 2);
  EXPECT_EQ(group.inner[1].offset, 32);
  EXPECT_EQ(group.inner[1].extent, 3);
  EXPECT_EQ(group.inner[1].count, 1);
  EXPECT_EQ(getGemmKernelTilingStrategy(strategy.kind)
                .getTaskRowAlignment(strategy),
            4);
}

TEST(GemmTilingStrategyTest, FixedKernelAdaptsSpecializedExecution) {
  const GemmKernelStrategy configured{GemmKernelStrategyKind::kFixed,
                                      FixedKernelParameters{3, 2}};
  const auto &policy = getGemmKernelTilingStrategy(configured.kind);
  const GemmKernelABI &gemmAbi = getGemmKernelABI(
      GemmTarget::kHip09, GemmIsa::kNeon, GemmDataType::kF32,
      GemmExecutionKind::kGemm);
  auto gemm = policy.configureForExecution(
      configured, GemmExecutionKind::kGemm, GemmIsa::kNeon, gemmAbi);
  ASSERT_TRUE(gemm);
  EXPECT_EQ(getFixedKernelParameters(*gemm).mr, 3);
  EXPECT_EQ(getFixedKernelParameters(*gemm).panelLanes, 2);
  EXPECT_TRUE(policy.isValidForExecution(*gemm, GemmExecutionKind::kGemm,
                                         GemmIsa::kNeon, GemmDataType::kF32,
                                         gemmAbi));

  for (GemmExecutionKind kind : {GemmExecutionKind::kMatrixVector,
                                 GemmExecutionKind::kVectorMatrix}) {
    const GemmKernelABI &abi = getGemmKernelABI(
        GemmTarget::kHip09, GemmIsa::kNeon, GemmDataType::kF32, kind);
    auto selected =
        policy.configureForExecution(configured, kind, GemmIsa::kNeon, abi);
    ASSERT_TRUE(selected);
    EXPECT_EQ(getFixedKernelParameters(*selected).mr, abi.maxMr);
    EXPECT_EQ(getFixedKernelParameters(*selected).panelLanes, abi.maxPanelLanes);
    EXPECT_TRUE(policy.isValidForExecution(*selected, kind, GemmIsa::kNeon,
                                           GemmDataType::kF32, abi));
    EXPECT_FALSE(policy.isValidForExecution(
        configured, kind, GemmIsa::kNeon, GemmDataType::kF32, abi));
  }
}

TEST(GemmTilingStrategyTest, CommonGemmFallbackRecomputesRhsPacking) {
  const auto specialized = selectGemmPath(GemmIsa::kNeon, 1, 400, 400,
                                          false, false, true);
  EXPECT_EQ(specialized.executionKind, GemmExecutionKind::kVectorMatrix);
  EXPECT_EQ(specialized.rhsPacking, RhsPacking::kDirect);

  const auto fallback = selectGemmPath(GemmIsa::kNeon, 1, 400, 400,
                                       false, false, false);
  EXPECT_EQ(fallback.executionKind, GemmExecutionKind::kGemm);
  EXPECT_EQ(fallback.rhsPacking, RhsPacking::kPacked);
}

TEST(GemmTilingStrategyTest, RhsPrepackRequiresFixedCacheAndKernelStrategies) {
  EXPECT_TRUE(isRhsPrepackSupported(GemmCacheStrategyKind::kFixed,
                                    GemmKernelStrategyKind::kFixed));
  EXPECT_FALSE(isRhsPrepackSupported(
      static_cast<GemmCacheStrategyKind>(1), GemmKernelStrategyKind::kFixed));
  EXPECT_FALSE(isRhsPrepackSupported(
      GemmCacheStrategyKind::kFixed,
      static_cast<GemmKernelStrategyKind>(1)));
}

TEST(GemmTilingStrategyTest, ThreadPlanningRequiresFixedStrategies) {
  EXPECT_TRUE(isThreadPlanningSupported(GemmCacheStrategyKind::kFixed,
                                        GemmKernelStrategyKind::kFixed));
  EXPECT_FALSE(isThreadPlanningSupported(
      static_cast<GemmCacheStrategyKind>(1), GemmKernelStrategyKind::kFixed));
  EXPECT_FALSE(isThreadPlanningSupported(
      GemmCacheStrategyKind::kFixed,
      static_cast<GemmKernelStrategyKind>(1)));
}

TEST(GemmTilingStrategyTest, PackedBWorkspaceUsesLargestCacheBlock) {
  const GemmKernelABI &neon = getGemmKernelABI(
      GemmTarget::kHip12, GemmIsa::kNeon, GemmDataType::kF32,
      GemmExecutionKind::kGemm);
  int64_t maximum = updateGemmPackedBWorkspaceElements(0, neon, 8, 15);
  EXPECT_EQ(maximum, 128);
  maximum = updateGemmPackedBWorkspaceElements(maximum, neon, 8, 16);
  maximum = updateGemmPackedBWorkspaceElements(maximum, neon, 1, 15);
  EXPECT_EQ(maximum, 128);

  for (int64_t vectorLengthBytes : {16, 32, 64}) {
    const GemmKernelABI sve{"test-sve", vectorLengthBytes, 6, 4, 0, 4};
    const int64_t alignment = getGemmPackedBAlignment(sve);
    maximum = updateGemmPackedBWorkspaceElements(0, sve, 8, 15);
    EXPECT_EQ(maximum, 8 * alignment);
    maximum = updateGemmPackedBWorkspaceElements(maximum, sve, 8, 16);
    maximum = updateGemmPackedBWorkspaceElements(maximum, sve, 1, 15);
    EXPECT_EQ(maximum, 8 * ((16 + alignment - 1) / alignment) * alignment);
  }
}

}  // namespace
}  // namespace annc::aarch64::gemm
