#include "../GemmPlanner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace annc::aarch64::gemm {
namespace {

struct StaticTaskGrid {
  std::int64_t tasks_m = 1;
  std::int64_t tasks_n = 1;
  std::int64_t task_count = 1;
  long double estimated_cycles = 0.0L;
  bool valid = false;
};

// The only current runtime dispatcher is TensorFlow's fixed-block ParallelFor
// with grain_size=1. Empty-task measurements on hip09 with TensorFlow 2.20
// across 2..256 worker pools show that the selected task count dominates the
// overhead; the pool capacity has no stable additional effect once it caps Q.
// Keep this runtime policy out of the target tuning file. Q remains bounded by
// config.max_threads before this function is called.
long double tfFixedBlockDispatchCycles(std::int64_t taskCount) {
  if (taskCount <= 1) return 0.0L;
  constexpr long double kFixedCycles = 12000.0L;
  constexpr long double kCyclesPerTask = 3300.0L;
  return kFixedCycles + static_cast<long double>(taskCount) * kCyclesPerTask;
}

std::int64_t ceilDiv(std::int64_t numerator, std::int64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

std::int64_t alignUp(std::int64_t value, std::int64_t alignment) {
  const std::int64_t remainder = value % alignment;
  if (remainder == 0) return value;
  const std::int64_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::int64_t>::max() - increment)
    return std::numeric_limits<std::int64_t>::max();
  return value + increment;
}

std::int64_t multiplyClamped(std::int64_t lhs, std::int64_t rhs) {
  if (lhs <= 0 || rhs <= 0 ||
      lhs > std::numeric_limits<std::int64_t>::max() / rhs)
    return std::numeric_limits<std::int64_t>::max();
  return lhs * rhs;
}

template <typename Callback>
void forEachBlockShape(std::int64_t dimension, std::int64_t block,
                       Callback callback) {
  const std::int64_t fullCount = dimension / block;
  if (fullCount > 0) callback(block, fullCount);
  const std::int64_t tail = dimension % block;
  if (tail > 0) callback(tail, std::int64_t{1});
}

bool validateConfig(const GemmPlannerConfig &config, std::string &error) {
  if (config.m <= 0 || config.n <= 0 || config.k <= 0 ||
      config.max_threads <= 0 || config.microkernel_m <= 0 ||
      config.microkernel_n <= 0 || config.packet_size <= 0 ||
      config.rhs_memory_alignment <= 0 || config.lhs_scalar_bytes <= 0 ||
      config.rhs_scalar_bytes <= 0 || config.output_scalar_bytes <= 0 ||
      config.cache_tile.mc <= 0 || config.cache_tile.nc <= 0 ||
      config.cache_tile.kc <= 0) {
    error =
        "problem, thread, tile, kernel, packet and scalar sizes must be "
        "positive";
    return false;
  }
  if (!config.vectorize_reduction &&
      config.microkernel_n % config.packet_size != 0) {
    error = "microkernel N must be a multiple of the packet size";
    return false;
  }
  const GemmPlannerCostModel &cost = config.cost_model;
  const auto isFinitePositive = [](double value) {
    return std::isfinite(value) && value > 0.0;
  };
  if (!isFinitePositive(cost.load_cycles_per_byte) ||
      !isFinitePositive(cost.store_cycles_per_byte) ||
      !isFinitePositive(cost.compute_cycle_scale) ||
      !isFinitePositive(cost.gemv_compute_cycle_scale) ||
      !isFinitePositive(cost.microkernel_call_cycles) ||
      !isFinitePositive(cost.pack_cycles_per_byte) ||
      !isFinitePositive(cost.pack_call_cycles)) {
    error = "all cost-model values must be finite positive numbers";
    return false;
  }
  return true;
}

// Cache and task boundaries restart kernel tiling. Account for each local N
// tail instead of padding the complete task only once.
long double nVectorGroups(const GemmPlannerConfig &config, std::int64_t n) {
  const std::int64_t fullTiles = n / config.microkernel_n;
  const std::int64_t tail = n % config.microkernel_n;
  const std::int64_t groupsPerTile = config.microkernel_n / config.packet_size;
  // Keep the count in floating-point after validating the integer inputs;
  // the planner must not wrap before it is converted into an estimated cost.
  return static_cast<long double>(fullTiles) * groupsPerTile +
         (tail == 0
              ? 0.0L
              : static_cast<long double>(ceilDiv(tail, config.packet_size)));
}

std::int64_t kernelRhsElements(const GemmPlannerConfig &config,
                               std::int64_t n) {
  if (!config.needs_runtime_pack) return n;
  return alignUp(n, config.rhs_memory_alignment);
}

long double gemmCacheBlockCycles(const GemmPlannerConfig &config,
                                 std::int64_t m, std::int64_t n, std::int64_t k,
                                 bool accumulate) {
  long double compute = 0.0L;
  forEachBlockShape(
      m, config.microkernel_m, [&](std::int64_t kernelM, std::int64_t mCount) {
        forEachBlockShape(
            n, config.microkernel_n,
            [&](std::int64_t kernelN, std::int64_t nCount) {
              // Work unit = one vector FMA; the coefficient is the measured
              // effective throughput for this target/ISA.
              const long double workUnits = static_cast<long double>(kernelM) *
                                            nVectorGroups(config, kernelN) * k;
              compute += static_cast<long double>(mCount) * nCount * workUnits *
                         config.cost_model.compute_cycle_scale;
            });
      });

  const long double microM = ceilDiv(m, config.microkernel_m);
  const long double microN = ceilDiv(n, config.microkernel_n);

  // A is read by each JR leaf and B by each IR leaf. These are leaf-local
  // accesses; target calibration should use the appropriate near-cache cost.
  const long double lhsBytes =
      static_cast<long double>(m) * k * microN * config.lhs_scalar_bytes;
  const long double rhsBytes = static_cast<long double>(k) *
                               kernelRhsElements(config, n) * microM *
                               config.rhs_scalar_bytes;
  const long double outputBytes =
      static_cast<long double>(m) * n * config.output_scalar_bytes;
  const long double loads =
      lhsBytes + rhsBytes + (accumulate ? outputBytes : 0.0L);
  const long double stores = outputBytes;

  const long double leafCalls = microM * microN;
  return compute + loads * config.cost_model.load_cycles_per_byte +
         stores * config.cost_model.store_cycles_per_byte +
         leafCalls * config.cost_model.microkernel_call_cycles;
}

long double gemvCacheBlockCycles(const GemmPlannerConfig &config,
                                 std::int64_t m, std::int64_t n, std::int64_t k,
                                 bool accumulate) {
  // GEMV-AB executes four NEON vectors (16 f32 values) per grouped K loop.
  const long double microM = ceilDiv(m, config.microkernel_m);
  const std::int64_t reductionGroup = 4 * config.packet_size;
  const std::int64_t vectorGroups = k / reductionGroup;
  const std::int64_t scalarResidue = k % reductionGroup;
  const long double vectorIterations = 4 * vectorGroups;
  long double compute = 0.0L;
  forEachBlockShape(
      m, config.microkernel_m, [&](std::int64_t kernelM, std::int64_t mCount) {
        // GEMV-AB vectorizes the reduction dimension. Keep its work unit
        // separate from GEMM's output-column vectorization.
        const long double workUnits = static_cast<long double>(kernelM) *
                                      (vectorIterations + scalarResidue);
        compute += static_cast<long double>(mCount) * workUnits *
                   config.cost_model.gemv_compute_cycle_scale;
      });
  const long double lhsBytes =
      static_cast<long double>(m) * k * config.lhs_scalar_bytes;
  const long double rhsBytes =
      static_cast<long double>(k) * microM * config.rhs_scalar_bytes;
  const long double outputBytes =
      static_cast<long double>(m) * n * config.output_scalar_bytes;
  const long double loads =
      lhsBytes + rhsBytes + (accumulate ? outputBytes : 0.0L);
  return compute + loads * config.cost_model.load_cycles_per_byte +
         outputBytes * config.cost_model.store_cycles_per_byte +
         microM * config.cost_model.microkernel_call_cycles;
}

long double packCycles(const GemmPlannerConfig &config, std::int64_t n,
                       std::int64_t k) {
  if (!config.needs_runtime_pack) return 0.0L;
  const long double sourceBytes =
      static_cast<long double>(k) * n * config.rhs_scalar_bytes;
  const long double packedBytes = static_cast<long double>(k) *
                                  alignUp(n, config.rhs_memory_alignment) *
                                  config.rhs_scalar_bytes;
  return (sourceBytes + packedBytes) * config.cost_model.pack_cycles_per_byte +
         config.cost_model.pack_call_cycles;
}

long double taskCycles(const GemmPlannerConfig &config, std::int64_t m,
                       std::int64_t n) {
  long double total = 0.0L;
  forEachBlockShape(
      m, config.cache_tile.mc, [&](std::int64_t cacheM, std::int64_t mCount) {
        const auto addKBlocks = [&](std::int64_t cacheK, std::int64_t kCount,
                                    bool accumulate) {
          if (cacheK == 0 || kCount == 0) return;
          forEachBlockShape(
              n, config.cache_tile.nc,
              [&](std::int64_t cacheN, std::int64_t nCount) {
                const long double oneBlock =
                    packCycles(config, cacheN, cacheK) +
                    (config.vectorize_reduction
                         ? gemvCacheBlockCycles(config, cacheM, cacheN, cacheK,
                                                accumulate)
                         : gemmCacheBlockCycles(config, cacheM, cacheN, cacheK,
                                                accumulate));
                total += static_cast<long double>(mCount) * kCount * nCount *
                         oneBlock;
              });
        };

        const std::int64_t firstK = std::min(config.k, config.cache_tile.kc);
        addKBlocks(firstK, 1, false);
        const std::int64_t remainingK = config.k - firstK;
        addKBlocks(config.cache_tile.kc, remainingK / config.cache_tile.kc,
                   true);
        addKBlocks(remainingK % config.cache_tile.kc, 1, true);
      });
  return total;
}

StaticTaskGrid evaluateStaticTaskGrid(const GemmPlannerConfig &config,
                                      std::int64_t tasksM,
                                      std::int64_t tasksN) {
  if (tasksM <= 0 || tasksN <= 0) return {};
  // Truncated task extents: N truncates to sixteen columns (one output-row
  // cache line) while M truncates to whole microkernel rows, so every task
  // offset stays cache-line- and microkernel-aligned in both dimensions.
  GemmTaskSplit splitN;
  GemmTaskSplit splitM;
  if (!splitGemmTaskRange(config.n, tasksN, kGemmTaskSplitAlignment, splitN))
    return {};
  if (!splitGemmTaskRange(config.m, tasksM, config.microkernel_m, splitM))
    return {};
  const std::int64_t workItems = multiplyClamped(tasksM, tasksN);
  if (workItems <= 0 || workItems > config.max_threads) return {};

  // Every non-final task extent is the truncated base of its dimension; the
  // final task absorbs the remainder and never shrinks below the base, so the
  // corner tail task is the exact slowest work item without enumerating Q
  // items for every candidate grid. Charge its static load imbalance
  // explicitly instead of assuming perfectly balanced tasks.
  const long double slowest =
      taskCycles(config, std::max(splitM.base, splitM.tail),
                 std::max(splitN.base, splitN.tail));
  const long double dispatch = tfFixedBlockDispatchCycles(workItems);
  return StaticTaskGrid{tasksM, tasksN, workItems, slowest + dispatch, true};
}

StaticTaskGrid selectTaskGrid(const GemmPlannerConfig &config) {
  StaticTaskGrid best = evaluateStaticTaskGrid(config, 1, 1);
  const std::int64_t taskLimit = config.max_threads;
  // A dimension hosts aligned tasks only while it provides at least one full
  // microkernel extent per task.
  const std::int64_t maxTasksM =
      std::clamp<std::int64_t>(config.m / config.microkernel_m, 1, taskLimit);
  const std::int64_t maxTasksN =
      std::clamp<std::int64_t>(config.n / config.microkernel_n, 1, taskLimit);
  for (std::int64_t tasksM = 1; tasksM <= maxTasksM; ++tasksM) {
    const std::int64_t maxTasksNForM = std::min(maxTasksN, taskLimit / tasksM);
    for (std::int64_t tasksN = 1; tasksN <= maxTasksNForM; ++tasksN) {
      StaticTaskGrid candidate = evaluateStaticTaskGrid(config, tasksM, tasksN);
      if (!candidate.valid) continue;
      if (candidate.estimated_cycles < best.estimated_cycles ||
          (candidate.estimated_cycles == best.estimated_cycles &&
           candidate.task_count < best.task_count) ||
          // Prefer column sharding when cost and work items tie: N-aligned
          // tasks keep full vector panels and avoid duplicating packed B.
          (candidate.estimated_cycles == best.estimated_cycles &&
           candidate.task_count == best.task_count &&
           candidate.tasks_n > best.tasks_n))
        best = candidate;
    }
  }
  return best;
}

bool linearizeByColumns(const GemmPlannerConfig &config,
                        const StaticTaskGrid &grid) {
  if (grid.tasks_n == 1) return false;
  if (grid.tasks_m == 1) return true;
  GemmTaskSplit splitM;
  GemmTaskSplit splitN;
  if (!splitGemmTaskRange(config.m, grid.tasks_m, config.microkernel_m,
                          splitM) ||
      !splitGemmTaskRange(config.n, grid.tasks_n, kGemmTaskSplitAlignment,
                          splitN))
    return false;
  return splitN.base >= splitM.base;
}

}  // namespace

bool planGemm(const GemmPlannerConfig &config, GemmPlanningResult &result,
              std::string &error) {
  if (!validateConfig(config, error)) return false;
  const StaticTaskGrid selected = selectTaskGrid(config);
  if (!selected.valid) {
    error = "cannot construct a static GEMM task grid";
    return false;
  }
  result = GemmPlanningResult{selected.task_count, selected.tasks_m,
                              selected.tasks_n,
                              linearizeByColumns(config, selected)};
  return true;
}

}  // namespace annc::aarch64::gemm
