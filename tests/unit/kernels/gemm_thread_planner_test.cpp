#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "GemmThreadPlanner.h"

namespace {

using annc::aarch64::gemm::FixedCacheParameters;
using annc::aarch64::gemm::GemmThreadPlannerConfig;
using annc::aarch64::gemm::GemmThreadPlannerCostModel;
using annc::aarch64::gemm::GemmThreadPlanningResult;
using annc::aarch64::gemm::GemmTaskSplit;
using annc::aarch64::gemm::kGemmTaskSplitAlignment;
using annc::aarch64::gemm::splitGemmTaskRange;

GemmThreadPlannerConfig makeConfig(std::int64_t m, std::int64_t n, std::int64_t k,
                             std::int64_t maxThreads) {
  return GemmThreadPlannerConfig{
      m,
      n,
      k,
      maxThreads,
      3,
      16,
      8,
      1,
      true,
      false,
      4,
      4,
      4,
      GemmThreadPlannerCostModel{0.171875, 0.171875, 1.0, 1.0, 1.0, 0.34375, 1.0},
      FixedCacheParameters{12, 16, 8}};
}

GemmThreadPlannerConfig makeSveConfig(std::int64_t m, std::int64_t n, std::int64_t k,
                                std::int64_t maxThreads) {
  GemmThreadPlannerConfig config = makeConfig(m, n, k, maxThreads);
  config.microkernel_m = 6;
  config.microkernel_n = 32;
  return config;
}

GemmThreadPlannerConfig makeAnalyticalConfig(std::int64_t m, std::int64_t n,
                                       std::int64_t k,
                                       std::int64_t maxThreads) {
  GemmThreadPlannerConfig config = makeConfig(m, n, k, maxThreads);
  config.microkernel_m = 6;
  config.microkernel_n = 32;
  config.cache_tile = FixedCacheParameters{72, 128, 256};
  return config;
}

GemmThreadPlanningResult plan(const GemmThreadPlannerConfig &config) {
  GemmThreadPlanningResult result;
  std::string error;
  EXPECT_TRUE(annc::aarch64::gemm::planGemmThreads(config, result, error)) << error;
  return result;
}

void expectAlignedGrid(const GemmThreadPlannerConfig &config,
                       const GemmThreadPlanningResult &result) {
  EXPECT_GT(result.tasks_m, 0);
  EXPECT_GT(result.tasks_n, 0);
  EXPECT_EQ(result.tasks_m * result.tasks_n, result.thread_count);
  EXPECT_LE(result.thread_count, config.max_threads);
  GemmTaskSplit splitM;
  GemmTaskSplit splitN;
  ASSERT_TRUE(splitGemmTaskRange(config.m, result.tasks_m, config.microkernel_m,
                                 splitM));
  ASSERT_TRUE(splitGemmTaskRange(config.n, result.tasks_n,
                                 kGemmTaskSplitAlignment, splitN));
}

// AArch64GemmThreadTiling materializes the aligned static grid as runtime
// arithmetic in IR: it decomposes taskId into (mTask, nTask) by shard
// direction, then computes each range as offset = task * base, size = (task
// == tasks - 1) ? tail : base, where base = alignDown(dim / tasks, align)
// and tail = dim - (tasks - 1) * base (N aligns to
// kGemmTaskSplitAlignment columns, M to whole microkernel rows). This
// helper mirrors exactly that arithmetic so the planner's chosen grids can
// be validated without running the pass.
struct ThreadTile {
  std::int64_t m_offset;
  std::int64_t m_size;
  std::int64_t n_offset;
  std::int64_t n_size;
};

ThreadTile computeThreadTile(std::int64_t taskId, std::int64_t m,
                             std::int64_t n, std::int64_t tasksM,
                             std::int64_t tasksN, std::int64_t mAlignment,
                             bool shardByColumns) {
  const std::int64_t mTask = shardByColumns ? taskId % tasksM : taskId / tasksN;
  const std::int64_t nTask = shardByColumns ? taskId / tasksM : taskId % tasksN;
  const auto alignedRange = [](std::int64_t task, std::int64_t tasks,
                               std::int64_t dimension, std::int64_t alignment) {
    GemmTaskSplit split{0, 0};
    EXPECT_TRUE(splitGemmTaskRange(dimension, tasks, alignment, split))
        << "dim=" << dimension << " tasks=" << tasks << " align=" << alignment;
    const std::int64_t base = split.base;
    const std::int64_t tail = split.tail;
    const bool last = task == tasks - 1;
    return std::pair<std::int64_t, std::int64_t>{task * base,
                                                 last ? tail : base};
  };
  const auto [mOffset, mSize] = alignedRange(mTask, tasksM, m, mAlignment);
  const auto [nOffset, nSize] =
      alignedRange(nTask, tasksN, n, kGemmTaskSplitAlignment);
  return ThreadTile{mOffset, mSize, nOffset, nSize};
}

void expectAlignedPartition(std::int64_t m, std::int64_t n, std::int64_t tasksM,
                            std::int64_t tasksN, std::int64_t mAlignment,
                            bool shardByColumns) {
  std::vector<std::int64_t> coverage(static_cast<std::size_t>(m * n), 0);
  for (std::int64_t taskId = 0; taskId < tasksM * tasksN; ++taskId) {
    const ThreadTile tile = computeThreadTile(taskId, m, n, tasksM, tasksN,
                                              mAlignment, shardByColumns);
    ASSERT_GT(tile.m_size, 0);
    ASSERT_GT(tile.n_size, 0);
    ASSERT_GE(tile.m_offset, 0);
    ASSERT_GE(tile.n_offset, 0);
    ASSERT_LE(tile.m_offset + tile.m_size, m);
    ASSERT_LE(tile.n_offset + tile.n_size, n);
    // Every task except the last one of each dimension receives a truncated
    // extent: a multiple of kGemmTaskSplitAlignment columns for N, a whole
    // number of microkernel rows for M.
    const bool mTaskIsLast = shardByColumns ? (taskId % tasksM == tasksM - 1)
                                            : (taskId / tasksN == tasksM - 1);
    const bool nTaskIsLast = shardByColumns ? (taskId / tasksM == tasksN - 1)
                                            : (taskId % tasksN == tasksN - 1);
    if (!mTaskIsLast && mAlignment > 1) {
      EXPECT_EQ(tile.m_size % mAlignment, 0)
          << "m=" << m << " taskId=" << taskId;
    }
    if (!nTaskIsLast) {
      EXPECT_EQ(tile.n_size % kGemmTaskSplitAlignment, 0)
          << "n=" << n << " taskId=" << taskId;
    }
    for (std::int64_t i = tile.m_offset; i < tile.m_offset + tile.m_size; ++i)
      for (std::int64_t j = tile.n_offset; j < tile.n_offset + tile.n_size; ++j)
        ++coverage[static_cast<std::size_t>(i * n + j)];
  }
  for (std::int64_t count : coverage)
    EXPECT_EQ(count, 1) << "m=" << m << " n=" << n << " tasksM=" << tasksM
                        << " tasksN=" << tasksN
                        << " shardByColumns=" << shardByColumns;
}

TEST(GemmThreadPlannerTest, SplitGemmTaskRangeTruncatesBaseDownAndFattensTail) {
  GemmTaskSplit split;
  // 100 / 3 = 33 truncates down to 32; the final task absorbs the remaining
  // 36 columns and never shrinks below the aligned base.
  ASSERT_TRUE(splitGemmTaskRange(100, 3, 16, split));
  EXPECT_EQ(split.base, 32);
  EXPECT_EQ(split.tail, 36);

  // Evenly divisible dimensions produce uniform tasks.
  ASSERT_TRUE(splitGemmTaskRange(48, 3, 16, split));
  EXPECT_EQ(split.base, 16);
  EXPECT_EQ(split.tail, 16);

  // A single task always covers the whole dimension, aligned or not.
  ASSERT_TRUE(splitGemmTaskRange(5, 1, 6, split));
  EXPECT_EQ(split.base, 5);
  EXPECT_EQ(split.tail, 5);

  // A truncated base that cannot keep a full non-final task invalidates
  // the split.
  EXPECT_FALSE(splitGemmTaskRange(5, 2, 6, split));
  EXPECT_FALSE(splitGemmTaskRange(8, 3, 4, split));
  EXPECT_FALSE(splitGemmTaskRange(8, 0, 4, split));
  EXPECT_FALSE(splitGemmTaskRange(8, 2, 0, split));
}

TEST(GemmThreadPlannerTest, MatrixVectorUsesRowShardedGrid) {
  GemmThreadPlannerConfig config = makeSveConfig(4096, 1, 256, 8);
  config.needs_runtime_pack = false;
  config.vectorize_reduction = true;
  config.packet_size = 4;
  config.microkernel_m = 4;
  config.microkernel_n = 1;
  const GemmThreadPlanningResult result = plan(config);

  expectAlignedGrid(config, result);
  EXPECT_GT(result.thread_count, 1);
  EXPECT_EQ(result.tasks_m, result.thread_count);
  EXPECT_EQ(result.tasks_n, 1);
  EXPECT_FALSE(result.shard_by_columns);
}

TEST(GemmThreadPlannerTest, NarrowProblemsUseTheCoreShardHeuristic) {
  const GemmThreadPlanningResult narrow = plan(makeSveConfig(4096, 8, 256, 8));
  const GemmThreadPlanningResult regular = plan(makeSveConfig(4096, 9, 256, 8));

  EXPECT_FALSE(narrow.shard_by_columns);
  EXPECT_EQ(narrow.tasks_n, 1);
  EXPECT_FALSE(regular.shard_by_columns);
  EXPECT_EQ(regular.thread_count, regular.tasks_m * regular.tasks_n);
}

TEST(GemmThreadPlannerTest, CostSelectedThreadsUseAvailableColumnTasks) {
  // n = 128 with microkernel N = 32 only hosts four column tasks above one
  // microkernel width (Eigen's pb_max_threads bound), so the planner caps
  // the thread budget at four tasks instead of eight thin ones.
  GemmThreadPlannerConfig config = makeSveConfig(5, 128, 1024, 8);
  const GemmThreadPlanningResult result = plan(config);

  expectAlignedGrid(config, result);
  EXPECT_EQ(result.thread_count, 4);
  EXPECT_EQ(result.tasks_m, 1);
  EXPECT_EQ(result.tasks_n, 4);
  EXPECT_TRUE(result.shard_by_columns);
}

TEST(GemmThreadPlannerTest, PlannedGridsPartitionTheirProblemsExactlyOnce) {
  constexpr std::array<std::array<std::int64_t, 3>, 9> shapes = {{
      {256, 256, 256},
      {5, 128, 1024},
      {96, 2, 1024},
      {8, 8, 8},
      {39, 64, 1540},
      {34, 1, 800},
      {7, 17, 511},
      {1024, 3, 64},
      {4096, 16, 256},
  }};
  for (const auto &shape : shapes) {
    const GemmThreadPlannerConfig config =
        makeSveConfig(shape[0], shape[1], shape[2], 8);
    const GemmThreadPlanningResult result = plan(config);
    expectAlignedGrid(config, result);
    expectAlignedPartition(shape[0], shape[1], result.tasks_m, result.tasks_n,
                           config.microkernel_m, result.shard_by_columns);
  }
}

TEST(GemmThreadPlannerTest, TfDispatchFormulaKeepsSmallProblemSerial) {
  const GemmThreadPlanningResult parallel =
      plan(makeAnalyticalConfig(256, 256, 256, 8));
  const GemmThreadPlanningResult serial = plan(makeAnalyticalConfig(8, 8, 8, 8));
  EXPECT_EQ(parallel.thread_count, 8);
  EXPECT_EQ(serial.thread_count, 1);
  EXPECT_EQ(serial.tasks_m, 1);
  EXPECT_EQ(serial.tasks_n, 1);
}

TEST(GemmThreadPlannerTest, PackedRhsShrinksPerTaskPackViaColumnSharding) {
  // Runtime PackB charges each task for its own N extent. Splitting M keeps
  // every task packing the full output width, while column tasks each pack
  // only their own slice of B, so the packed planner keeps all parallelism in
  // the column direction: (1,8) halves the per-task pack of (2,4).
  GemmThreadPlannerConfig packed = makeAnalyticalConfig(48, 256, 256, 8);
  packed.cost_model.load_cycles_per_byte = 1.0;
  packed.cost_model.store_cycles_per_byte = 1.0;
  packed.cost_model.compute_cycle_scale = 0.01;
  packed.cost_model.pack_cycles_per_byte = 10.0;
  const GemmThreadPlanningResult packedPlan = plan(packed);

  GemmThreadPlannerConfig rowMajor = packed;
  rowMajor.needs_runtime_pack = false;
  const GemmThreadPlanningResult rowMajorPlan = plan(rowMajor);

  expectAlignedGrid(packed, packedPlan);
  EXPECT_EQ(packedPlan.tasks_m, 1);
  EXPECT_EQ(packedPlan.tasks_n, 8);
  EXPECT_TRUE(packedPlan.shard_by_columns);
  // Without runtime packing the per-task cost is M/N symmetric, so the
  // column-first tie-break selects the same grid but for a different reason.
  EXPECT_EQ(rowMajorPlan.tasks_m, 1);
  EXPECT_EQ(rowMajorPlan.tasks_n, 8);
}

TEST(GemmThreadPlannerTest, NoPackPathDoesNotChargePackCost) {
  GemmThreadPlannerConfig baseline = makeAnalyticalConfig(5, 8, 32, 8);
  baseline.needs_runtime_pack = false;
  const GemmThreadPlanningResult first = plan(baseline);

  GemmThreadPlannerConfig expensivePackCall = baseline;
  expensivePackCall.cost_model.pack_cycles_per_byte = 1.0e12;
  expensivePackCall.cost_model.pack_call_cycles = 1.0e12;
  const GemmThreadPlanningResult second = plan(expensivePackCall);

  EXPECT_EQ(first.thread_count, second.thread_count);
  EXPECT_EQ(first.tasks_m, second.tasks_m);
  EXPECT_EQ(first.tasks_n, second.tasks_n);
}

TEST(GemmThreadPlannerTest, GemvComputeCoefficientChangesGrid) {
  GemmThreadPlannerConfig fastGemv = makeAnalyticalConfig(4096, 1, 256, 8);
  fastGemv.needs_runtime_pack = false;
  fastGemv.vectorize_reduction = true;
  fastGemv.packet_size = 4;
  fastGemv.microkernel_m = 4;
  fastGemv.microkernel_n = 1;
  fastGemv.cost_model.load_cycles_per_byte = 0.0001;
  fastGemv.cost_model.store_cycles_per_byte = 0.0001;
  fastGemv.cost_model.gemv_compute_cycle_scale = 0.0001;
  const GemmThreadPlanningResult fastPlan = plan(fastGemv);

  GemmThreadPlannerConfig slowGemv = fastGemv;
  slowGemv.cost_model.gemv_compute_cycle_scale = 100.0;
  const GemmThreadPlanningResult slowPlan = plan(slowGemv);

  EXPECT_EQ(fastPlan.thread_count, 1);
  EXPECT_GT(slowPlan.thread_count, fastPlan.thread_count);
}

TEST(GemmThreadPlannerTest, AdditionalKcBlockChargesOutputAccumulation) {
  GemmThreadPlannerConfig oneKc = makeAnalyticalConfig(5, 64, 5, 2);
  oneKc.needs_runtime_pack = false;
  oneKc.cache_tile.kc = 5;
  oneKc.cost_model.load_cycles_per_byte = 10.0;
  oneKc.cost_model.store_cycles_per_byte = 10.0;
  oneKc.cost_model.compute_cycle_scale = 0.001;
  oneKc.cost_model.microkernel_call_cycles = 0.0001;
  const GemmThreadPlanningResult oneKcPlan = plan(oneKc);

  GemmThreadPlannerConfig twoKc = oneKc;
  twoKc.cache_tile.kc = 4;
  const GemmThreadPlanningResult twoKcPlan = plan(twoKc);

  EXPECT_EQ(oneKcPlan.thread_count, 1);
  EXPECT_GT(twoKcPlan.thread_count, oneKcPlan.thread_count);
}

TEST(GemmThreadPlannerTest, RejectsNonFiniteCostPolicy) {
  GemmThreadPlannerConfig config = makeConfig(64, 64, 64, 4);
  config.cost_model.compute_cycle_scale =
      std::numeric_limits<double>::infinity();
  GemmThreadPlanningResult result;
  std::string error;

  EXPECT_FALSE(annc::aarch64::gemm::planGemmThreads(config, result, error));
  EXPECT_NE(error.find("finite positive"), std::string::npos);
}

TEST(GemmThreadPlannerTest, ThreadTiledGridCoversEveryOutputExactlyOnce) {
  // {m, n, tasksM, tasksN, mAlignment}
  constexpr std::array<std::array<std::int64_t, 5>, 5> grids = {{
      {17, 100, 3, 4, 3},   // both M and N leave fat tails
      {24, 64, 4, 4, 6},    // evenly divisible grid
      {7, 1, 7, 1, 1},      // single-column shard
      {12, 20, 1, 1, 3},    // serial task
      {100, 100, 3, 3, 6},  // large grid with fat tails
  }};
  for (const bool columns : {false, true}) {
    for (const auto &grid : grids)
      expectAlignedPartition(grid[0], grid[1], grid[2], grid[3], grid[4],
                             columns);
  }
}

}  // namespace
