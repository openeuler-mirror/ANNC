#ifndef ANNC_LIB_KERNEL_BUILTIN_KERNELS_PARALLEL_GRAIN_H
#define ANNC_LIB_KERNEL_BUILTIN_KERNELS_PARALLEL_GRAIN_H

#include <algorithm>
#include <cstdint>

namespace annc {
namespace kernels {

// Grain for row-parallel kernels: how many rows one shard must carry before
// handing the range to the host pool pays for itself.
//
// The units are *element operations* per shard, not rows: a masked-mul row is
// n multiplies, a batched-matmul-add row is k*n MACs, so a fixed row count
// would over-shard the cheap kernels and under-shard the expensive ones.
//
// Basis (task 20261013-019, phase 2): through the ORT EP a shard costs ~1.4us
// of dispatch (measured on the target models: 48 shards -> 66us per call, 1 ->
// 13us, on sites of a few thousand element ops), while scalar arithmetic runs
// at a few elements per nanosecond.  Break-even therefore lands in the
// tens of thousands of element ops per shard (ORT's own kernels get the same effect
// from Eigen's cost model, which targets ~100k cost units per block); 65536 keeps a
// comfortable margin for the per-shard bookkeeping on top of the dispatch.
constexpr std::int64_t kElementsPerShard = 65536;

// Rows to give one shard when every row costs `elements_per_row` element ops.
// A grain of `rows` (or more) means the caller should stay serial, which is
// what annc::threadpool::parallel_for does when total <= grain.
inline std::int64_t grainForRowWork(std::int64_t elements_per_row) {
  if (elements_per_row <= 0) return 1;
  return std::max<std::int64_t>(1, kElementsPerShard / elements_per_row);
}

}  // namespace kernels
}  // namespace annc

#endif  // ANNC_LIB_KERNEL_BUILTIN_KERNELS_PARALLEL_GRAIN_H
