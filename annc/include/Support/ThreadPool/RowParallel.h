#ifndef ANNC_SUPPORT_THREADPOOL_ROW_PARALLEL_H
#define ANNC_SUPPORT_THREADPOOL_ROW_PARALLEL_H

#include <algorithm>
#include <cstdint>

#include "Support/ThreadPool/ThreadPool.h"

namespace annc {
namespace threadpool {

// ---------------------------------------------------------------------------
// The row-parallel scheduling contract
// ---------------------------------------------------------------------------
//
// A kernel that splits its work by rows hands the host two advisory numbers
// with every parallel_for; the host reads the one its framework knows:
//
//   cost_per_unit -- what one row costs, in the host's own units.  This is the
//     quantity ORT's kernels pass as TensorOpCost and Eigen's cost model turns
//     into a thread count, and what the KDNN adaptor forwards to its
//     scheduling parameters: the ORT EP adapter and the KDNN adaptor both
//     consume it, which is how a kernel keeps the scheduling decision its
//     framework would have made.
//   grain_size -- minimum rows per block, for hosts whose pool takes a fixed
//     block size instead of a cost model: the in-tree pool
//     (Support/ThreadPool/Parallel.cpp reads only grain_size) and the TF
//     adapter's kFixedBlockSize path.
//
// cost::* computes the first of these from the shape of the kernel's inner
// loop, so a new fusion kernel states what it computes -- an elementwise row,
// a row of dot products -- and inherits the host's scheduling instead of
// hand-summing bytes and cycles; rowOptions() then packages both signals.
//
// Nothing here knows about a specific operator, and nothing here is consulted
// by the hosts: this is only how the numbers are produced.

namespace cost {

// Cycles per byte loaded and stored, the weight Eigen's cost model applies to
// memory traffic (kLoadCycles = kStoreCycles = 11/64: L2 latency over a cache
// line, unsupported/Eigen/CXX11/src/Tensor/TensorCostModel.h).  Applying it
// here keeps a kernel's number on the scale hosts expect.
constexpr double kCyclesPerByte = 11.0 / 64.0;

// A row's cost from its parts: the compute cycles plus the memory traffic at
// the weight above.  The escape hatch for a kernel whose inner loop is not one
// of the shapes below.
inline std::int64_t inHostUnits(std::int64_t compute_cycles,
                                std::int64_t bytes_loaded,
                                std::int64_t bytes_stored) {
  const double memory =
      static_cast<double>(std::max<std::int64_t>(bytes_loaded, 0) +
                          std::max<std::int64_t>(bytes_stored, 0)) *
      kCyclesPerByte;
  return std::max<std::int64_t>(compute_cycles, 0) +
         static_cast<std::int64_t>(memory);
}

// One element of an elementwise row: `cycles_per_element` of arithmetic,
// `bytes_read_per_element` in, `bytes_written_per_element` out (4 = one f32
// result, the common case).  This covers the elementwise family -- a multiply,
// a masked multiply (whose mask operand is the extra bytes), a normalize pass.
inline std::int64_t element(std::int64_t cycles_per_element,
                            std::int64_t bytes_read_per_element,
                            std::int64_t bytes_written_per_element = 4) {
  return inHostUnits(cycles_per_element, bytes_read_per_element,
                     bytes_written_per_element);
}

// One element of a dot product of length `k`: two f32 operands read and one
// multiply-accumulate per step, one f32 result stored once it is complete.
// This covers the matmul family (matmul, batched matmul add, tensordot).
inline std::int64_t dot(std::int64_t k) {
  const std::int64_t macs = std::max<std::int64_t>(k, 0);
  return inHostUnits(macs, 8 * macs, 4);
}

// A whole row made of `elements_per_row` elementwise elements.
inline std::int64_t elementRow(std::int64_t elements_per_row,
                               std::int64_t cycles_per_element,
                               std::int64_t bytes_read_per_element,
                               std::int64_t bytes_written_per_element = 4) {
  const std::int64_t n = std::max<std::int64_t>(elements_per_row, 0);
  return inHostUnits(n * std::max<std::int64_t>(cycles_per_element, 0),
                     n * std::max<std::int64_t>(bytes_read_per_element, 0),
                     n * std::max<std::int64_t>(bytes_written_per_element, 0));
}

// A whole row made of `n` dot products of length `k` (a [k] x [k, n] product,
// which is what the matmul-family kernels parallelize over).
inline std::int64_t dotRow(std::int64_t k, std::int64_t n) {
  const std::int64_t macs =
      std::max<std::int64_t>(k, 0) * std::max<std::int64_t>(n, 0);
  return inHostUnits(macs, 8 * macs, 4 * std::max<std::int64_t>(n, 0));
}

}  // namespace cost

// Grain for row-parallel kernels: how many rows one shard must carry before
// handing the range to the host pool pays for itself.
//
// The units are *element operations* per shard, not rows: an elementwise row
// is a few multiplies, a matmul row is k*n MACs, so a fixed row count would
// over-shard the cheap kernels and under-shard the expensive ones.
//
// Basis (task 20261013-019, phase 2): through the ORT EP a shard costs ~1.4us
// of dispatch (measured on the target models: 48 shards -> 66us per call, 1 ->
// 13us, on sites of a few thousand element ops), while scalar arithmetic runs
// at a few elements per nanosecond.  Break-even therefore lands in the
// tens of thousands of element ops per shard (ORT's own kernels get the same
// effect from Eigen's cost model, which targets ~100k cost units per block);
// 65536 keeps a comfortable margin for the per-shard bookkeeping on top of the
// dispatch.  Hosts that model cost (ORT, KDNN) ignore this and use
// cost_per_unit; it matters for the in-tree pool and the TF adapter.
constexpr std::int64_t kElementsPerShard = 65536;

// Rows to give one shard when every row costs `elements_per_row` element ops.
// A grain of `rows` (or more) means the caller should stay serial, which is
// what annc::threadpool::parallel_for does when total <= grain.
inline std::int64_t grainForRowWork(std::int64_t elements_per_row) {
  if (elements_per_row <= 0) return 1;
  return std::max<std::int64_t>(1, kElementsPerShard / elements_per_row);
}

// Both scheduling signals for a row-parallel kernel, ready to pass to
// parallel_for: the row's cost (a cost::* value) and the grain fallback sized
// from the same row work in element operations.
inline ParallelForOptions rowOptions(std::int64_t cost_per_unit,
                                     std::int64_t elements_per_row) {
  ParallelForOptions options;
  options.cost_per_unit = cost_per_unit;
  options.grain_size = grainForRowWork(elements_per_row);
  return options;
}

}  // namespace threadpool
}  // namespace annc

#endif  // ANNC_SUPPORT_THREADPOOL_ROW_PARALLEL_H
